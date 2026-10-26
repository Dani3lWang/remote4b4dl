# A3 / seed42 非有限损失现场诊断

日期：2026-10-26。范围为 `training_logs/core_geometry_20261022` 的原始 A3 / seed42 失败现场、部署源码 `source-1d23109`、冻结配置和 `step-4750.pt`。本轮未启动或恢复真实训练，未修改原训练目录、原权重、数据或冻结实验文件。

结论：已确定失败阶段并排除 step4750 的已存参数/优化器损坏；四个候选输入的完整 CPU 前向均有限。**原始首个非有限运算仍无法唯一确认**，本轮完成现场保留和更新保护修复，不宣称原数值根因已修复。

## 已核实的故障位置

**4953 是最后成功更新；实际失败发生在尝试更新 4954 的前向 loss 检查。** 原始堆栈定位到 `b4dl_core.b3.train_step → teacher_loss → model(...).loss`，随后在 `b3.py:125` 主动抛出 `ValueError: nonfinite training loss`。这不是已记录的梯度检查、裁剪、AdamW 或 GradScaler 检查失败。

| 项目 | 本次核验 |
|---|---|
| 成功更新 | 日志连续 1–4953，共 19,812 条呈现；总预算 13,851 次更新、55,404 条 train |
| 最后成功 loss / gradient norm | 0.3579063583 / 0.08305604756 |
| 原始退出 | 2026-10-23 15:10:34 北京时间，退出 1；`completion.json` 不存在 |
| 最近保存点 | step 4750；4751–4953 共 203 次成功更新未保存 |
| 检查点 SHA256 | `682ee9febfc0f1c2aa4b5dea7c219e5e146460ca574d38b95760550df4af5a3d`，与原 latest 一致 |
| 原实现 SHA256 | `aa072e07d5696d40d131999eb3e8377511ee0b333c7c779c328e8ea9ca51216f` |
| 原实验文件 SHA256 | `153ae2cf00e7f1c674554663d84c6784854eb5eaaf3ceada2a1f452a155ff2c4` |
| 原配方 | A3 / seed42，batch1 × accum4，恒定 lr1e-4；冻结 B3 FP16，可训练主参数 FP32，GradScaler 初值 1024 |

原审计在 CPU 检查了 38 个可训练参数及 Adam 状态，共 152 个张量，全部有限，Adam step 均为 4750；scaler scale=4096，growth tracker=750。本次再次实际读取检查点并核对文件哈希、参数与配置。**该结论证明 4750 的完整性，不证明丢失的 4953 状态仍然正常。**

原日志没有失败微批次编号、logits、展开后的 labels、首个异常层或失败模型快照。仅凭最后成功 loss、有限的旧检查点和四个候选输入，不能唯一识别原始首个非有限运算，也不能将 FP16 溢出、SDPA 内核、输入损坏或学习率认定为已证实根因。降低 GradScaler scale 不会修复已经发生的前向非有限 loss。

## 按原采样函数还原候选输入

固定 `batch_indices(55404, 4953, micro, 1, 4, 42)`，对应第 4954 次更新；没有重新抽样、跳题或修改标签。

| micro（从 0 起） | index | request_id | scene_id | 任务 | 特征绝对值最大值 |
|---|---:|---|---|---|---:|
| 0 | 53992 | f992ab846b65539584aa6870 | 006403812 | position / right | 8.31594 |
| 1 | 19242 | 5a4338a002bdd07758c6f63a | 008534740 | motion / farther | 10.30861 |
| 2 | 25259 | 764675f3ae485dcc245f2735 | 003979666 | existence | 8.61109 |
| 3 | 39763 | b8699b64d71eb16922bf9b01 | 003833660 | existence | 10.27612 |

四个输入通过数据、场景包、刚体位姿和有限值检查；在实际 step4750 的 FP32 时序适配器后仍有限。FP16 视觉投影的绝对值最大值分别为 **34.84375、38.1875、38.9375、36.5**。因此本次没有复现输入或适配器/投影溢出。

## CPU 诊断与边界

本次远端没有可用 GPU，容器限额为 2 GiB 内存、0.5 CPU，不能直接装入完整 7B 模型。`mllm/scripts/diagnose_core_nonfinite.py` 使用 mmap 读取基底分片、逐个线性层恢复原 B3 LoRA、还原多模态展开和 assistant-only labels，执行四个候选输入的 FP16 CPU 前向。只保留当前线性权重，不创建优化器或执行 backward。

LoRA 按实际 PEFT 0.13.2 的 CPU 合并语义处理：适配器转 FP32 计算 delta，再原位加到 FP16 基底；没有 INT8 近似。位置/RoPE、SDPA 因果注意力、RMSNorm、SwiGLU、残差及输出词表扩展保持原路径。新增小型 CPU 回归核对该解码层与原生 SDPA 层的逐值等价。

早期内存不足的诊断退出与最终诊断分别保存，不把不完整运行视为成功。完整 CPU 结果与测试回执见文末的验收记录。CPU 结果只能说明 step4750 对这些输入的行为，不能代替原 GPU 内核、训练模式及已丢失的 step4953 状态。

最终 v3 诊断退出 0：四个输入共 128 个 decoder 记录及所有线性层输入/输出均有限，四个输出头及 FP32 交叉熵也均有限：

| micro | 展开序列长度 | logits 绝对值最大值 | teacher-forcing loss |
|---|---:|---:|---:|
| 0 | 293 | 35.0625 | 1.35330808 |
| 1 | 362 | 33.8125 | 1.22040701 |
| 2 | 213 | 32.78125 | 0.02633323 |
| 3 | 213 | 32.65625 | 0.02859573 |

以上为失败后旧检查点的诊断量，不是新的性能评测或真实失败 micro 定位。原始执行脚本保存在证据目录的 `diagnose_core_nonfinite_v3.py`，SHA256 为 `b05b791dc366db3c54437bafbd4a8de49db5635a1c16a7513adeaff728c7dea5`；完整 JSONL 的 SHA256 为 `84c910ef6033182845ec5a0095ca45bf1ffcde73898b0a2f3592d3d2f8dfe0c8`，迁回后逐字节一致。提交版提取同一 decoder 运算为辅助函数，以便执行原生层等价回归。

## 最小修复

本轮修复针对已经证实的现场记录缺失及独立发现的更新保护缺口，不根据猜测改动精度、学习率、样本顺序或训练预算：

1. `b3.py` 增加结构化 `NonfiniteTrainingError`：区分前向 loss、反缩放梯度和梯度总范数异常；前向异常记录 loss/logits 摘要、准确 micro、前面已完成的有限 micro loss、输入各张量摘要与监督 token 数。
2. `train` 保存 `failure.json`，记录 attempted step、已完成更新数、各 micro 的 index/request_id/scene_id/input_fingerprint、原配方及身份。记录先于快照写盘，异常时清理部分累积梯度，不执行 optimizer update，不产生 completion 或推进 latest。
3. 保存当次真实扩展参数、优化器、scaler 和 RNG 为 `failure-state.pt`，带 `diagnostic_only=true` 及 SHA256。它是前向/反向失败后的诊断现场，不是累积边界的恢复端点；`load_checkpoint` 在应用任何参数之前拒绝诊断快照。
4. 裁剪启用 `error_if_nonfinite=True`，拒绝“每个梯度有限，但 FP32 总范数归约溢出”的情况。这是独立防护缺口，原日志最后 gradient norm 有限，**没有证据表明它导致了本次前向 loss 失败**。
5. `validate_experiment` 现在核对原计划已经登记的 `verified_implementation_sha256` 与当前实际源码身份；此前该字段未在正式门控中使用。这项缺口没有导致已核验身份一致的原始运行失败，但修复后必须阻止新源码沿用旧冻结计划从头启动。旧检查点加载的严格身份校验不放宽。
6. 增加可复跑的 CPU 逐层诊断脚本和针对性回归。

## 恢复训练前的核验条件

- **资产与资源**：恢复可用 CUDA 设备，核对 torch/Transformers/PEFT、GPU 拓扑和原共享锁；原部署环境路径已经失效，实际可用 wqlc 为 `/root/autodl-tmp/.conda-stuff/envs/wqlc`。核对原 B3 各组件、SST 来源、数据 ID、实验文件与六任务协议的实际哈希，不能仅复制旧回执。
- **源码与迁移**：本次代码变更会改变实现身份。原 `--resume step-4750.pt` 在新源码下应被拒绝；不能清空门控、篡改旧 checkpoint 身份或放宽全局校验。需另行形成明确的父 checkpoint SHA256、原/新实现与配置哈希、差异范围、优化器/scaler/RNG 保留策略和新运行身份的受控迁移记录。本轮未生成迁移 checkpoint 或新实验计划。
- **真实模型数值检查**：用原基底及 step4750，按真实 CUDA 训练精度、`train()`、非重入 gradient checkpointing 和原累积顺序，核对从 4750 恢复的下一组输入，以及 4954 的四个候选 micro；逐段检查输入、适配器、投影、decoder、logits、有效 labels、loss、scaler 反缩放后梯度及总范数。先只做有界前向/反向，优化器更新为 0；失败时保留首个异常运算和实际模型快照。CPU 有限性不是这项检查通过的依据。
- **记录与数据呈现**：从 step4750 开始，下一次更新为 4751，数据位置为 `4750×4=19000`；总终点仍 13851，共剩 9101 次更新。原 4751–4953 的 203 次成功更新需要重做，不能从 4954 跳接或给预算追加一轮。确认重新处理 4954 四个候选时不跳题。
- **独立输出与验收**：使用新作业/输出目录，保留原失败现场和中途检查点评测；核实当前实际进程，不能把旧 `status.json` 的 running 当作存活。本次旧训练及评测 PID 均已不存在，旧评测状态未刷新。恢复后的首个获准更新须核验实际 Adam 计数、参数/scaler/RNG、有限 loss/梯度及保存重载；有新来源的正式准备回执后才执行后续预算。最终 completion、固定终点评测和六任务回归仍是独立验收。

**当前不满足直接恢复训练的条件。** 本轮没有启动训练，也没有以修复提交、CPU 回放或中途检查点声明训练完成或模型性能提升。

## 证据位置与验收记录

本地运行证据：`training_logs/core_geometry_20261022/nonfinite-20261026/`，含 `live-scene.json`、逐层 CPU 日志和测试回执。原始现场位于远端 `/root/autodl-tmp/mmb4dl-core-sft-20261023-r1/{training,jobs}/A3-seed42`；本次独立 CPU 诊断目录为 `/root/autodl-tmp/mmb4dl-core-nonfinite-20261026-r1`。运行产物不入 Git，方法说明、源码和回归检查一起提交。

验收：本地 wqlc Python 3.10.20 / torch 2.8.0+cpu / Transformers 4.47.0 / PEFT 0.13.2 执行全部 50 项核心检查，48 项通过，2 项因未安装 nuScenes/pyquaternion 跳过。新增 6 项全部通过：原生 SDPA 解码层逐值等价、后续 micro 失败清理部分梯度、非有限梯度具名拒绝、有限梯度的总范数溢出拒绝、有限 logits/空监督导致非有限 loss 的区分、真实请求失败记录和诊断快照禁用。原有保存/恢复/采样/正式训练门控回归继续通过，新增对未登记实现的拒绝检查。六个 Python 变更的 Ruff 未定义名称检查及 Git diff 空白检查通过。本轮检查仅在小型 CPU fixture 上执行优化器更新；真实 B3 的优化器更新为 0。修复只在本地工作树验收，未部署为新的远端训练实现。
