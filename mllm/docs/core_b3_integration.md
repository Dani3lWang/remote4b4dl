# 原始 B3 的时序几何接入

日期：2026-10-23。**原始 B3 接入、CPU 回归与真实 7B GPU 工程验收已完成；正式训练和效果评测尚未执行。**

基底为已固定的 `_protected_b3_3ep_final_20260910`，模型资产身份见 `mllm/configs/core_geometry_baseline_original_b3.json`。本次不修改原权重，也不接入分割模块。

## 接入位置

`VTimeLLMLlamaForCausalLM.forward` 和生成参数准备函数接收 `frame_times`、`frame_geometry`、`frame_valid_mask`。`prepare_inputs_labels_for_multimodal` 在收到这些参数或已挂载适配器时，先执行帧模块，再投影并展开视觉 token。原调用不传帧参数、也未挂载适配器时仍走原路径。

有效特征先按 mask 取出，再以与原路径相同的有效帧总行数执行 projector，避免补齐帧进入语言序列。时间和位姿保持 float64；可训练参数使用 FP32，projector 计算沿用语言模型精度，避免初始阶段因参数精度变化造成不必要的行为差异。适配器默认 2 层/768 维/8 头，训练时梯度可穿过冻结语言模型返回视觉模块。

原始 B3 先通过既有 builder 完整恢复并合并其 LoRA/non-LoRA，随后再挂载新适配器。`attach` 支持：

| 分组 | 训练内容 |
|---|---|
| A0 | 原始 B3，仅推理，全部参数冻结 |
| A1 | projector，可选新 q/v LoRA |
| A2 | A1 加时间适配器 |
| A3 | A1 加时间与自车几何适配器 |

`--lora-rank` 默认 0；大于 0 时在已合并 B3 上新建 q/v LoRA，alpha=2r、dropout=0。A1–A3 比较必须使用相同 rank 和训练预算。该默认值只是接入验证策略，不是已选定的正式实验最优配方。

## 训练与生成

新入口为 `python -m b4dl_core.b3`，与原 `scripts/run_b3.sh` 独立；原脚本不会自动训练新模块。

训练提示使用原 `v1` 会话模板与一个 `<video>` 槽位，assistant 标签为 `{answer,evidence}` JSON 加 EOS；prompt、补齐和视觉 token 的标签为 IGNORE。推理只消费 `GeometryDataset.model_input()` 返回值，不拼接 target。展开后的完整长度超限会拒绝，而不是截断监督。

生成先按各样本有效帧长度展开、左侧补齐，之后才交给 Transformers 创建 cache_position。时序模块每次生成只执行一次；后续 token 解码不重复处理 SST。支持变长文本/帧批次。返回的是 continuation token，不包含视觉占位符前缀。

训练采用单设备 AdamW、固定学习率、梯度累积、梯度裁剪与非重入 gradient checkpointing。GPU 使用 FP16 autocast/GradScaler，所有可训练 master 参数保持 FP32。当前没有 DDP/DeepSpeed 接入或学习率调度器；正式实验预算与配置仍需冻结。

## 保存与恢复

扩展 checkpoint 包含所有可训练参数（适配器、projector、可选新 LoRA）、optimizer、scaler、torch/CUDA RNG、训练步数、配置和身份。冻结的原始 B3 权重不重复保存。先保存 `.partial`，完成后才重命名；拒绝覆盖已有 checkpoint。

恢复时检查基底清单 SHA256、完整数据 ID、实现代码 SHA256、arm/时序/LoRA 配置及训练配方。数据按 seed 和已完成步数确定性选取，恢复可延长总步数，不能悄悄改学习率、batch 或累积数。当前测试确认保存恢复后的下一步参数更新逐值一致。

必须通过本入口先恢复原始 B3 再加载扩展；这些 `.pt` 不是可直接交给 Hugging Face `from_pretrained` 的完整模型目录。代码改动导致实现身份改变时，恢复会拒绝；迁移必须明确验证，不能绕过身份检查。

## 命令

用户已明确授权上传与两步验证，实际通过版本为独立目录内的 `source-r3`，输出为 `verification-r3`。以下使用已有 wqlc Python 3.10；verify 已执行，其余 train/resume/infer 为后续使用示例，未在真实 7B 上额外执行：

```bash
cd /root/autodl-tmp/mmb4dl-core-integration-20261023-r1/source-r3
DATA=../mini-v2
export PYTHONPATH="$PWD/mllm"
PY=/root/autodl-tmp/mmb4dl-maskpls-20261016/runtime/wqlc/bin/python
BASE=mllm/configs/core_geometry_baseline_original_b3.json
PREP=mllm/configs/core_geometry_preparation.json

# 三组件实时哈希核对、真实 B3 logits/token 等价、变长生成、2 步 QA 工程训练、保存恢复。
"$PY" -m b4dl_core.b3 verify --dataset "$DATA" --baseline "$BASE" --preparation "$PREP" --output ../verification-new

# 独立短程训练示例；engineering 模式最多允许 20 个总优化步。
"$PY" -m b4dl_core.b3 train --dataset "$DATA" --baseline "$BASE" --preparation "$PREP" --arm A3 --lora-rank 0 --engineering --steps 2 --output training-smoke

# 在新的目录续训，保持 seed、lr、batch、grad-accum、LoRA 等配置相同。
"$PY" -m b4dl_core.b3 train --dataset "$DATA" --baseline "$BASE" --preparation "$PREP" --arm A3 --lora-rank 0 --engineering --steps 4 --resume training-smoke/step-2.pt --output training-resumed

# 正式推理默认最多 256 个新 token，保留 truncated 状态。
"$PY" -m b4dl_core.b3 infer --dataset "$DATA" --baseline "$BASE" --preparation "$PREP" --arm A0 --split test --max-new-tokens 256 --output baseline-predictions
"$PY" -m b4dl_core.b3 infer --dataset "$DATA" --baseline "$BASE" --preparation "$PREP" --arm A3 --checkpoint training-smoke/step-2.pt --condition shuffle_features --split test --output shuffled-predictions
"$PY" -m b4dl_core evaluate --dataset "$DATA" --split test --predictions shuffled-predictions/predictions.jsonl --run shuffled-predictions/run.json --output shuffled-report.json
```

首次运行的输出目录必须新建。2026-10-27 起，几何推理可在同一命令后加 `--resume-inference` 恢复当前实现创建的输出；训练 `--resume CHECKPOINT` 的语义不变。共同提交日志、身份与尾行恢复规则见 [环境与恢复说明](upcore_runtime.md)。GPU 操作共用 `/root/autodl-tmp/.b4dl_gpu0_eval.lock` 非阻塞独占锁，并要求 GPU 0 至少 20,000 MiB 空闲、利用率不超过 20%；不会自动停止别人的进程。每次加载前重新核对原始模型文件，源码/数据身份写入结果。`infer` 除评测兼容输出外，写出逐请求控制条件和实际特征数组 SHA256；缺少跨场景供体记 unavailable，保留完整分母。

`verify` 默认只生成 8 token，并只对 train 中一个样例做 2 个优化步以验证梯度；保存的工程 checkpoint 不代表正式训练完成。`train` 不带 `--engineering` 时，若准备门控未完成、数据为 mini 或 SST 来源未核验，会在加载模型前拒绝。

## 验证证据与当前限制

- 34 项核心检查在本地和远端通过：原有 25 项，加 9 项 B3 接入检查。新增覆盖历史 PEFT 合并后残留配置与未合并 LoRA 的拒绝。新检查覆盖未挂载/零初始化 logits 等价、一次视觉展开、变长生成与逐条生成一致、cache 开关一致、assistant-only 标签、LoRA 梯度、下一步恢复等价、实际训练入口及推理评测契约、基底篡改拒绝。
- 46 项既有 AnswerSeg/mini 采集回归通过，确认共享 B3 架构接口没有破坏既有调用。这里只运行既有回归，未修改分割算法。
- CPU 使用真实 Transformers/PEFT 类及合成小模型；训练入口测试确实执行优化和恢复，但不能替代 7B CUDA/FP16 验收。
- AutoDL3 RTX 4090 已完成真实 7B FP16 验收，退出码 0：40/39 帧两条输入的关闭模块与零初始化 logits 逐值一致；原路径/新路径生成 token 一致；批量/逐条生成一致；每次生成仅执行一次适配器；两步 QA 训练及保存恢复 logits 逐值一致。
- 两步 loss 为 1.50855136、1.00260293，第二步内部 attention 梯度绝对值和为 14.31890297；峰值 PyTorch allocated 为 13,172.31 MiB。两步使用同一 train 样例，仅证明优化链路可运行，不能作为质量提升证据。
- 首轮在旧路径 FP32 特征与 FP16 projector 不匹配处退出；随后修正参考输入精度及 PEFT 合并残留配置判断。第二轮在验证路径补齐 token 不一致处退出；统一 EOS 后第三轮全部通过。两次失败均在优化前，未发生训练更新，失败日志保留。
- 本地回执：`training_logs/core_geometry_20261022/integration-tests-r3.txt`、`shared-b3-regression-tests.txt`。最终代码校验以补充的 `integration-tests-final.txt` 为准。
- 后续 SST 来源及完整 trainval 已核验，见[全量说明](core_sst_trainval.md)。后续正式训练预算/种子与六任务协议已冻结，见[训练说明](core_geometry_training.md)；mini 工程验收不能作为性能提升证据。

## 真实验收身份与复核

- 本地回执目录：`training_logs/core_geometry_20261022/integration-evidence`；包含三轮日志、退出码、调用参数、错误回执和最终 `verification-r3/verification.json`。工程 checkpoint 留在远端 `verification-r3/engineering-extension.pt`，SHA256 为 `ff44d2872ec8ff5941d6bb721318d74bb67ca32f316b03f43aca3c626d9aef95`。
- 验收实现 SHA256：`21b2904177bf04439797a2f9a640281e093b2390ba6d4e739434b97c97903e3e`；基底清单 SHA256：`2c4bf02e1ad3d8a6428b06f175342faa85ac250bf7d39bf78e48c67b4c530d54`。42 个部署文件逐一核验，源码压缩包 r3 SHA256 为 `50e97f14da035ae1f96b9f1217fd194b0d79ceb1caabb629371f40787ae4b0d7`。
- 紧凑回执归档 SHA256：`8a0668266ceed3f49f00ec04a067a1a9bdf94646965bd74518a64e252f128763`。本地解包后逐项复核；原始 B3 三组件加载前实时核验通过，未覆盖原权重。
- 基底清单保留锁定时的历史字段；当前接入状态由准备配置及本轮回执表示。验收部署使用验收前的准备配置快照；本地准备配置仅更新已完成门控，不改变模型、数据或实现身份。
- 此验收使用 A3、无新增 LoRA、8-token 生成预算。真实 7B 的可选新 LoRA、长输出、全量任务性能和跨进程续训仍需对应实验；CPU 已验证新 LoRA 与下一步恢复等价。


2026-10-23 数据准备跟进：正式候选数据改为实际右后裁剪范围的完整 trainval，新来源绑定清单为 `core_geometry_baseline_original_b3_trainval.json`。正式训练额外核对数据与清单的 SST provenance_id/回执 SHA256，并要求输入范围声明；不匹配或回执被改动时在加载模型前拒绝。原清单及工程 checkpoint 的身份约定继续保留。


正式训练现要求 `--experiment` 指向与准备配置SHA256匹配的冻结配方，检查数据、基底、步数、学习率、有效batch、seed、LoRA rank、checkpoint策略及六任务材料；checkpoint身份加入实验SHA256。新训练首步即保存，随后按固定间隔记录latest；保留step1与最近指定份数，loss采用追加记录。已有历史checkpoint须使用其对应源码版本。
