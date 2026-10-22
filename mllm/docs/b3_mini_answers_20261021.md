# B3 mini 回答采集与训练后 FP16 等待任务

当前状态（2026-10-22 复核）：FP16 完整回答集已经生成并通过服务器与本地独立审计，1680/1680 条，零空回答、零截断。服务器于北京时间 2026-10-21 22:39:16 完成审计打包，2026-10-22 已完整迁回本地。所有回答仍待审核，尚不是可直接训练的实例绑定标签。

历史启动策略：用户最终要求等待现有训练结束，再用 FP16 生成。INT8 试跑已停止并保留；独立等待任务未终止或修改原训练，原 checkpoint 保留。

## 输入与模型身份

- 官方 nuScenes `v1.0-mini` 十场景，1680 条原始 QA、1638 组唯一模型输入；相同输入的贪心回答复用，但保留全部原始 QA。
- 官方 `mini_train` 1360 条、`mini_val` 320 条。另存原 B4DL 来源与划分，不将两个划分混为一谈。
- 请求 SHA256：`46d27790b07a1dedac066e2a2ba842c828a866fb8eb115a00e5488f01b650760`。
- 整场景 SST 特征；meta2 仅来自问题帧号与传感器运动记录。不使用参考答案或 `answer_frames` 选择输入。
- 继续沿用本轮固定的历史三轮 B3：`checkpoints/_protected_b3_3ep_final_20260910`。等待当前两轮训练结束是为了释放 GPU；本任务没有切换到新两轮权重。
- 原 SST 缓存编码器 checkpoint 未独立核验，如实记录 `feature_encoder_checkpoint_verified=false`。

## 最终生成策略

源码提交 `caba55e6e0799456469d0f54905423ea98ff6877` 支持 FP16 与可选 INT8。用户最终选择的是 FP16：`--quantization none --cpu-offload-layers 0 --batch-size 8 --memory-fraction 0.82`，整模型在 GPU 上以 FP16 运行。默认生成 256 token，截断后最多重试 512 token，OOM 自动减小批量。全部输入展开后为 164–235 token，含最长重试预算仍在上下文上限内。

采用全新 `generation-fp16-after-training` 目录，不混入 INT8 回答。所有回答保持 `review_status=pending`、`eligible_for_binding_training=false`，不伪造目标实例、分割帧或 SEG。生成完成与指代/分割质量验收是两件事。

## 等待门控与服务器位置

服务器：`ssh -p 21358 root@connect.bjb1.seetacloud.com`。

独立根目录：`/root/autodl-tmp/mmb4dl-b3-mini-answers-20261021-v1/`。

等待进程 PID 128585，脚本 `wait_then_collect_fp16.py`。启动时原训练 PID 72247、DeepSpeed launcher PID 72077 仍运行，最近完整保存为 checkpoint-1600（epoch 1.3807057300，最终预算 2316 步）；该值是最近落盘进度，不是当前实时步数。

收口复核：等待进程仍运行，最新落盘更新为 checkpoint-1800（epoch 1.5533613899）；没有本任务的采集进程，FP16 输出目录尚未创建。本地状态快照见 `training_logs/answerseg_20261021/deployment/wait-fp16-v1/state.json`，服务器后续状态以实时文件为准。

门控同时要求：

1. 原训练及其 launcher 已退出，通过 PID、进程启动时间与命令指纹区分进程复用。
2. 根输出目录存在等待任务创建后新写入的 `trainer_state.json`、adapter 配置/权重及 `non_lora_trainables.bin`，均非空；epoch 至少 1.99，global_step 达到 max_steps。
3. 连续两次、间隔 60 秒观测 GPU 空闲显存至少 20000 MiB、利用率不高于 20%。若原流水线继续评测或其他任务占用 GPU，继续等待。
4. 固定推理 runtime 与历史 B3 文件指纹未变。异常训练退出、权重变动或 72 小时等待超时不启动生成，记录错误。

状态与产物：

| 路径（相对独立根目录） | 内容 |
|---|---|
| `wait-fp16-v1/job.json` | 固定模型身份、原训练进程身份与等待策略 |
| `wait-fp16-v1/state.json` | 当前状态，等待期每 60 秒更新 |
| `wait-fp16-v1/watcher.log` | 等待日志 |
| `wait-fp16-v1/error.json` | 失败时记录错误 |
| `generation-fp16-after-training/` | 已完成的 FP16 全量采集输出 |
| `generation-fp16-after-training.exit` | 采集结束后的退出码 |
| `wait-fp16-v1/audit.log` | 完成后独立审计日志 |
| `mini-answers-fp16-delivery.tar.gz` | 已完成并迁回的审计交付包 |

审计复核完整 QA 覆盖、分组、回答 token 解码、逐条指纹、特征与真实帧对应、部署源码及原始权重未变。文件下载与本地复核已在交付包生成后完成，详见下方完成记录。

## 已完成的验证与试跑边界

- 46 项 AnswerSeg CPU 检查在本地和服务器通过。实际 INT8 加载确认 224 个 decoder 线性权重为 int8，LiDAR 投影与输出层等保留 FP16；原 checkpoint 未覆盖。
- INT8 批量与两条逐条路径的前 8 token 等价检查通过。初始批量 32 触发 OOM 后降到 16；用户要求等待 FP16 后，已精确停止本任务的 INT8 进程，退出码 143，原训练继续。
- 冻结试跑的 80 组同输入对照中，56 条文字完全相同，24 条变化；部分变化涉及物体方位。两组均未逐条核验场景，批量不同，不能将 30% 文本变化称为错误率或质量下降率。INT8 结果不纳入最终 FP16 回答集。
- 等待门控的 6 项文件完成性检查本地通过，Linux 服务器另通过进程身份检查，共 7 项。现场 `--check-only` 返回 `waiting_training`，无 FP16 生成目录；随后实际后台启动等待任务。等待进程本身不加载语言模型、不占 GPU 推理显存。

运行产物和等待脚本位于本地 `training_logs/answerseg_20261021/deployment/`，不提交日志、预测或权重。等待脚本 SHA256：`498fadb3a9d18bc75b1e16a1743c00686bfb2cf1e12e8cda4bb45b87fbbcb297`；审计脚本 SHA256：`f4790326200b1f6329e555687051b1d3fcae2409176f7a1f19180fb7a67a6fe0`。

## 2026-10-22 完成复核与本地交付

原两轮训练达到 2316/2316 步、epoch 1.9988129923，最终 adapter、非 LoRA 参数及训练状态保存完整，原进程与 launcher 退出。等待任务确认连续两次 GPU 空闲后启动 FP16。生成和审计均退出 0，最终状态 `complete_fp16`；这不代表新两轮权重已经经过完整性能评估，本次回答仍由上述固定历史三轮 B3 生成。

全量 1638 组唯一输入生成耗时约 426 秒，保留全部 1680 条原始 QA；batch8，峰值 allocated/reserved 14112.55/15756 MiB。十场景完整覆盖，mini_train 1360 条、mini_val 320 条，1680 条均为 `ok`，空回答和截断均为 0。推理身份为 `quantization=none, compute_dtype=float16`，无 INT8 回答混入。

本地完整目录：`datageneration/data/b3-mini-answers/20261021-001/`。

- [问答阅读版](../../datageneration/data/b3-mini-answers/20261021-001/answers-preview.md)
- [全部问答 JSONL](../../datageneration/data/b3-mini-answers/20261021-001/answers.jsonl)
- [本地独立审计](../../datageneration/data/b3-mini-answers/20261021-001/independent-audit.json)

交付包 SHA256：`3002122877d96ffec6144dff1292659adcf58fdab1e34769cb4bdae6c448b7ee`；回答文件 SHA256：`57c5fd45f7b73e2633fceec689d6d876c962f69ce410afaf7258019f40c4594c`。本地复核 46 个清单文件、10 个 SST 文件、全部 token 解码与逐条指纹，回执与服务器审计逐字段相同。审计所需 protobuf 仅解压到本次独立审计目录，未修改原训练环境。

完成状态回执位于 `training_logs/answerseg_20261021/deployment/wait-fp16-v1/completed-state.json`，服务器与本地审计分别为 `fp16-server-audit.json`、`fp16-local-audit.json`。数据与日志属于运行产物，不提交 Git。回答内容及具体实例仍需审核，`binding_training_ready=false`；本轮未启动对齐头训练。
