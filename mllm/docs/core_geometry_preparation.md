# B4DL 本体更新：独立模块、数据与评测准备

更新日期：2026-10-23。状态：**原始 B3 已接入，CPU 回归及真实 7B GPU 等价、两步工程训练与保存恢复验收通过；正式训练待数据与协议门控完成**。新增接口、命令和保存恢复约定见[接入说明](core_b3_integration.md)。

这份实现独立于现有分割路线。模块接口、几何监督和评测准备已经完成；当前基底固定为原始历史三轮 B3。真实训练与效果对比仍需核验 SST 编码器和完整数据，并固定正式实验协议。

## 1. 交付位置

- 分支：`codex/b4dl-core-geometry`，从 `a89f61b` 建立。
- 独立工作树：`D:/Github/mmb4dl/training_logs/core_geometry_20261022/source`。
- 运行产物目录：`D:/Github/mmb4dl/training_logs/core_geometry_20261022`。
- 最终数据：上述目录的 `mini-v2`；`mini-v1` 为早期验证记录，保留溯源。
- 原 `4D_seg` 工作树的未提交修改保持原样，本分支未合并、未推送。
- 论文依据单独维护在 [论文依据与实现映射](../../docs/research/B4DL本体更新_论文依据与实现映射.md)。

| 文件 | 已实现内容 |
|---|---|
| `mllm/b4dl_core/temporal.py` | 可关闭的帧时序适配器、相对时间/自车位姿偏置、零初始化残差、配置严格加载、projector 接口 |
| `data.py` / `geometry.py` | 原始 nuScenes 关键帧对齐、场景划分、存在/方位/距离/相对运动监督 |
| `inputs.py` | 模型输入白名单、哈希检查、变长组批、真实特征干预 |
| `rewards.py` | 严格 JSON 解析、答案正确性、数值证据与自洽性奖励 |
| `evaluation.py` | 完整分母、分任务/谓词指标、场景 bootstrap、相反答案配对、选项轮换、修复/退化统计、train-only 多数先验 |
| `audit.py` | 独立逆齐次变换复算数值、同一实例与起点最近性核验 |
| `__main__.py` | `prepare/preflight/audit/smoke/controls/prior/evaluate/compare` 命令 |
| `mllm/configs/core_geometry_preparation.json` | 已选定原始 B3 的准备配置和 A0–A3 消融约定；是实验约定，尚非训练启动配置 |
| `mllm/configs/core_geometry_baseline_original_b3.json` | 语言底座、Stage 1 projector 与历史三轮 B3 的路径、逐文件/整体指纹及核验来源 |
| `mllm/tests/test_core_geometry.py` | 25 项单元测试，含真实格式的合成原始表端到端检查 |

## 2. 模型接口与边界

数据流：整场景 SST `[B,T,768]` → 可选帧适配器 → 原 `mm_projector` → 原语言模型。

默认配置为 2 层、8 头、FFN 1536，总参数 10,061,008。保留原投影层的维度映射职责。每帧采用真实时间戳和 `global_from_ego` 齐次位姿；注意力偏置使用 `R_i^T(p_j-p_i)/10m`、相对 yaw 的 sin/cos，以及 `(t_j-t_i)/4s`。先按 float64 做坐标/时间差分，再转为计算精度；支持右侧补齐，屏蔽无效查询与键。

位置输入与成对关系均经过 raw/sin/cos 编码和可学习映射；这些是本项目设计，并非某篇论文的原式复现。时间模式 `use_geometry=False` 保留同一时序结构，但只使用相对时间。

输出残差线性层初始为零，因此有效帧初始输出等于原 SST；关闭适配器时直接调用原 projector。零初始化使第一步内部 attention 梯度为零，输出层更新后梯度传入内部层；测试明确覆盖这一过程。补齐位置仍须由后续多模态展开逻辑按 mask 排除。

`TemporalProjector` 保留为独立接口，不直接替换现有 `mm_projector`。本次已在 VTimeLLM 首次视觉 token 展开前接入 `frame_times/frame_geometry/frame_valid_mask`，挂载独立 `frame_adapter`；生成先展开再建立缓存，后续解码不重复插入。CPU 小模型等价检查已通过，真实 B3 的 CUDA token 等价仍待验收。

位姿仅描述自车/传感器帧关系。全局标注中心、实例 ID、最近实例选择结果只用于监督与审计，不传给模型。该模块没有恢复物体级轨迹、点级空间结构或实例掩码。

## 3. 数据定义

输入为九张原始 nuScenes 表、B4DL `scene_metadata.json`、场景 SST `.npy` 和显式官方 split 清单。只接受 keyframe `LIDAR_TOP`，核对文件名→sample_data→sample→scene、首尾帧、双向时间链、特征行数、pose 时间戳和唯一实例。

坐标为自车 x 向前、y 向左、z 向上，单位米。距离使用标注框中心在地平面的距离，不是最近 LiDAR 点距离。只使用 `num_lidar_pts>0` 的标注；某类仅有零点标注时，该类存在性为未知并跳过，不生成不存在监督。各类最近实例出现等距并列时跳过相关关系题。

- 存在性：某帧是否存在有 LiDAR 点支持的类别实例；证据 `count`。
- 方位：最近同类实例的八个 45° 扇区；证据 `x,y`。每个有效实例生成真实扇区和一个确定性错误扇区。
- 距离排序：四个类别各自最近实例的距离，保留四次循环选项排列；证据 `distance_<class>`。
- 相对运动：从起点选择最近同类实例，严格跟踪同一 `instance_token` 到第 8 个后续关键帧；终点缺失或无 LiDAR 点时跳过。两个端点分别在当时自车坐标中测量，记录 `d0,d1,delta_d,delta_y`。

运动阈值：靠近 `delta_d<=-2m`，远离 `>=2m`，向左 `delta_y>=1m`，向右 `<=-1m`；similar 要求 `abs(delta_d)<1m` 且 `abs(delta_y)<0.5m`。这衡量**自车相对变化**，静止世界物体也可能被标为靠近，不能改写成物体绝对速度。第 8 个关键帧不是硬编码 4 秒，真实时间差另行记录。

按 `split/task/predicate` 对二分类 Yes/No 做平衡，每标签默认上限 256；不声称类别内也平衡。距离题每个语义问题保留四个轮换，总语义问题默认每 split 上限 256。缺少某标签的桶记录于 `shortages`，不补造样本。

官方 validation 作为 test；按 seed=42 对**完整官方 train 场景名单**排序后取 20% 为 internal_dev，剩余 train。子集准备不会改变场景归属。internal_dev 可能已被旧 B3 预训练见过；这不是证明完全未见数据的测试集。

`requests.jsonl` 仅含公开问题及输入定位信息；`supervision.jsonl` 保存答案、数值证据和标注出处。三个 split JSONL 包含 request/target 供监督使用，**不能将 target 或参考 assistant turn 拼入推理提示**。`GeometryDataset.model_input()` 仅返回 prompt、features、frame_times、frame_geometry、frame_valid_mask。文件、源表、生成器代码与场景特征均有 SHA256；中断导致缺少最终 manifest 的目录不可加载。

## 4. 奖励与评测协议

模型输出必须是单个 JSON 对象，顶层仅 `answer` 和 `evidence`。二分类严格为 Yes/No，多选为当前 A/B/C/D；额外正文、重复 JSON 键、非法数值、bool 伪装数字、错误字段均不作宽松补救。答案与证据有效性分开统计：答案有效但证据错误仍可获得答案分数。

`R_answer = 1[答案正确]`。数值字段得分为 `max(0,1-|预测-真值|/1m)`，count 要求整数精确一致；若证据无法推出模型自己的答案，证据奖励为 0。运动还检查 `d1-d0≈delta_d`（0.11m 内，仅容忍输出小数舍入）。字段平均形成 `R_evidence`；联合奖励 `0.7 R_answer+0.3 R_evidence`。容差/权重是待消融的项目约定，不是已证明最优的常数，也不是已实施的 GRPO 训练。

评测要求数据 ID、split、condition、model_id、checkpoint 清单 SHA256 和 prediction_kind。真实模型缺少合法 SHA256 会拒绝；fixture/prior 明确分开。该身份字段是调用方声明，无法单独证明实际加载了哪些权重，后续推理器须产生加载回执。

所有请求都进入分母：missing/error/unavailable/truncated 均失败；重复/未知请求或错指纹直接报错。数值 MAE 只在有效证据上计算，同时报告有效率与 MAE 样本数，不能孤立汇报条件 MAE。按任务及谓词汇报准确率、观察到的标签平衡准确率，缺失标签由覆盖报告揭示。总体 micro 不能代替各任务结果。

场景 bootstrap 默认 2,000 次、seed42，整场景重采样，再按问题数求 micro；少于两场景不给区间。mini 的两场景测试区间仅供代码检查，不适合泛化结论。

相反答案配对要求同 split、task/predicate、完整 prompt 和特征长度，且不同场景、相反标签；确定性最大二分匹配，每条请求最多使用一次，完全不读取预测挑选配对。报告双正确率及有效答案变化率。距离题将字母映射回语义类别，报告四个排列的一致率与全正确率。

对照输入确实执行数值干预：identity、全零、重复首帧、非恒等特征乱序、跨场景同 split 同长度特征替换。时间/位姿和文字保持不变，因此这是 **SST 特征分支消融，不能称为完全无传感器输入**。完全不读取特征/位姿的诊断另用 train-only 多数先验或固定答案 prior。跨场景供体不足记 unavailable；比较效果时须同时报告供体覆盖，不能将缺供体造成的失败解释成模型退化。

`compare` 在相同数据、split、完整请求集合上统计净变化、修复、退化、双对、双错与场景 bootstrap 差值。条件是否真正应用，需要未来推理器调用 `model_input(request, control)` 并记录输入回执；仅在 run JSON 改 condition 名称没有干预作用。

## 5. 复跑命令

在 wqlc Python 3.10 中运行。当前可用解释器及目录：

```powershell
$src = 'D:/Github/mmb4dl/training_logs/core_geometry_20261022/source'
$out = 'D:/Github/mmb4dl/training_logs/core_geometry_20261022'
$py = 'D:/Github/mmb4dl/training_logs/e2q_20261019/runtime-py310/wqlc/Scripts/python.exe'
Set-Location $src
$env:PYTHONPATH = Join-Path $src 'mllm'
& $py -m unittest discover -s mllm/tests -p test_core_geometry.py -v
& $py -m b4dl_core --help
```

准备新数据目录（工具拒绝覆盖已有输出；重跑时更换目录/文件名）：

```powershell
& $py -m b4dl_core prepare --tables "$out/metadata" --scene-metadata 'D:/Github/mmb4dl/datageneration/data/b4dl-official/metadata/scene_metadata.json' --features 'D:/Github/mmb4dl/datageneration/data/b3-mini-answers/20261021-001/features' --splits "$out/official-mini-splits.json" --output "$out/mini-rebuild"
& $py -m b4dl_core preflight --dataset "$out/mini-rebuild" --output "$out/preflight-rebuild.json"
& $py -m b4dl_core audit --dataset "$out/mini-rebuild" --tables "$out/metadata" --output "$out/audit-rebuild.json"
& $py -m b4dl_core smoke --dataset "$out/mini-rebuild" --output "$out/smoke-rebuild.json"
& $py -m b4dl_core controls --dataset "$out/mini-rebuild" --split test --condition shuffle_features --output "$out/shuffle-rebuild.jsonl"
& $py -m b4dl_core prior --dataset "$out/mini-rebuild" --split test --policy train_majority --output "$out/prior-rebuild.json"
```

数据搬迁时运行时只需要 bundle 自身；`audit` 还需要保留 manifest 中记录的原始源路径或在原环境复算。默认训练配置不自动读取：适配器构造时显式传 `TemporalConfig(**config['frame_adapter'])`。

未来推理输出每行精确四字段：

```json
{"request_id":"来自请求清单","input_fingerprint":"来自请求清单","status":"ok","text":"{\"answer\":\"Yes\",\"evidence\":{\"count\":1}}"}
```

`run.json` 精确六字段（下列占位值必须替换）：

```json
{"dataset_id":"数据manifest的dataset_id","split":"test","condition":"identity","model_id":"已选定的基底及适配器身份","checkpoint_sha256":"权重文件清单的64位SHA256","prediction_kind":"model"}
```

```powershell
& $py -m b4dl_core evaluate --dataset "$out/mini-v2" --split test --predictions "$out/predictions.jsonl" --run "$out/run.json" --output "$out/model-report.json"
& $py -m b4dl_core compare --base "$out/base-report.json" --candidate "$out/model-report.json" --output "$out/paired-report.json"
```

smoke 使用两条真实场景特征和随机初始化适配器，在 CPU 上做两次人工目标梯度检查，随后丢弃参数。它不读取 B3 权重、不使用 QA 监督训练、不提供任务准确率。

## 6. 已完成验证与真实数据覆盖

最终数据 ID：`e5a11cc901ccdbe39bdfdafa4703df2b181aea183dfb405577a4c293f2b68daa`。

| split | 场景 | 存在 | 方位 | 距离 | 相对运动 | 合计 | 严格相反答案对 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 6 | 512 | 2216 | 848 | 2326 | 5902 | 332 |
| internal_dev | 2 | 512 | 400 | 224 | 250 | 1386 | 0 |
| test | 2 | 512 | 660 | 288 | 584 | 2044 | 0 |
| 合计 | 10 | 1536 | 3276 | 1360 | 3160 | 9332 | 332 |

原候选 15,554 条，选后 9,332 条。距离题为 340 个语义问题的四种排列，不能当作 1,360 个独立几何事件。测试集 `motion/similar` 有 236 个 No、0 个 Yes，该桶未纳入平衡集合；开发/测试没有严格相反答案配对。跨场景特征替换仅 4,514 条具备供体，4,818 条不可用。以上缺口必须随数据一起报告。

真实原始表→404 帧→26,168 个数值证据独立复算全部通过。生成器采用旋转转置变换，审计器采用逆齐次变换并独立构造四元数旋转，逐题复核标注身份、最近性、姿态和时间戳。此项证明数据内部一致性，不证明标注没有测量误差或模型正确。

25 项测试通过：同一实例跟踪、最近目标切换、零点未知、等距并列、阈值边界、场景隔离、错误帧/时间/位姿/重复标注拒绝、补齐一致性、全局刚体变换不变性、初始等价与后续梯度、保存加载、解析反例、奖励自洽约束、完整分母、最大配对、选项语义映射、真正特征变化、哈希篡改拒绝。

工程回执为 `unit-tests-final.txt`、`preflight-final.json`、`audit-final.json`、`smoke-final.json`、`prior-final.json`、`shuffle-plan-final.jsonl`。train-majority prior 在测试集的混合任务 micro 为 46.48%，二分类 50%、四选一 25%；证据有效率 0。它只是评测器的无传感器诊断结果，**不是 B3 成绩**。

另执行空预测 fixture 的 CLI 负向检查：`empty-fixture-report.json` 保留全部 2,044 个请求、准确率为 0；`fixture-compare-report.json` 检查对比命令。fixture 的输出不构成模型实验。

## 7. 已选基底与后续事项

2026-10-23 用户明确选用原始 B3，本项目将其固定为此前完整 mini FP16 采集使用的 `_protected_b3_3ep_final_20260910`。服务器 `autodl3` 上的三个模型组件已只读复算哈希，与历史采集身份全部一致：

| 组件 | 服务器路径 | artifact SHA256 |
|---|---|---|
| Vicuna 语言底座 | `/root/autodl-tmp/mmb4dl/mllm/base_model/vicuna-v1-5-7b` | `df6f77dc7d40970217e5da0de42576a112ffa2b87b9591889eb2a067dff11830` |
| Stage 1 projector | `/root/autodl-tmp/mmb4dl/mllm/checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin` | `646a52aab39764f68341de5fb72e3f1dff2e06e2c8fb9457e97c1b1efe0ee267` |
| 原始 B3 Stage 2 | `/root/autodl-tmp/mmb4dl/mllm/checkpoints/_protected_b3_3ep_final_20260910` | `2704d1e1e7aebe4e2f2ff43fefe6adb8acc0594726a6af42bd4d47198c179814` |

目录 artifact 指纹使用既有 `answerseg.runtime.artifact_identity` 规则，清单记录全部逐文件 SHA256 与序列化规则；单文件 projector 则直接计算文件 SHA256。核验回执为运行产物目录的 `baseline-live-verification-20261023.jsonl`。新两轮训练权重不作为本轮基底；初始真实评测沿用 FP16、不量化，未来训练精度另行固定。

历史 `mini-v2/manifest.json` 和此前工程回执中的 `baseline_checkpoint=null` 保留，表示生成时尚未选基底；当前实验选择由配置和独立模型清单表达，不重写历史数据 ID。A0–A3 均应从这组相同模型资产开始，SST 编码器来源仍为未核验。

1. 已完成用户基底选择及 base LLM/projector/LoRA/non-LoRA 当前文件指纹核验；仍需确认 SST 编码器和预处理身份。
2. VTimeLLM 代码已接入时间/位姿/mask，新增专用训练、推理、保存恢复入口，34 项核心检查通过；原始 7B 权重的 GPU 等价、变长生成、两步梯度及保存恢复验证均已通过。
3. 准备完整 trainval，检查每个任务/谓词/类别、严格配对和供体覆盖；本轮仅真实 mini 跑通，全量内存/吞吐尚未测量。
4. 冻结种子、算力、SFT 数据和原六任务回归输入。A0 原基线；A1 仅新监督；A2 加时间模块；A3 再加自车几何；本轮尚无任何消融效果结果。
5. 后续 RL 在 `rf-grpo` 开展，从同一 SFT 终点比较答案奖励与联合奖励；固定时序模块以保持参考模型一致。本轮只有离线 scorer，没有 GRPO trainer。

基底锁定阶段仅进行了只读核验。后续接入已在 AutoDL3 完成真实 B3 FP16 验收，工程更新共 2 步，未启动正式 SFT/RL；实际身份、失败修正和回执见[接入说明](core_b3_integration.md)。该验证不产生模型效果提升结论。
