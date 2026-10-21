# B3 回答与实例分割的在线接口

独立入口为 `mllm/scripts/run_answerseg.py`，实现位于 `mllm/answerseg/`。原 B3 的 loader、模型、训练及评测入口保持原有实现。此接口面向 **已知输入帧的单帧回答指代分割**；语言上下文可包含同场景多帧特征，但输出仍是指定帧的原始点序掩码，不提供自由时间定位或跨帧跟踪。

2026-10-21 的真实权重同步、CPU卸载/空间CUDA工程冒烟及自由生成验收缺口见[远端冒烟报告](answerseg_smoke_20261021.md)。

## 训练链与参数

```text
同场景 SST 特征 + 问题 → 恢复后的 B3 → 回答中的 <SEG>/<NOOBJ> 隐状态
                                                    ↓ 在线对齐头
原始点云 → MaskPLS → 预测实例 query / 掩码 → 实例选择或明确无目标
```

- 先恢复、合并原 B3 adapter，再增加 `<SEG>/<NOOBJ>`，避免把扩词表后的模型直接送入原 adapter 恢复流程。保留 `<4DLiDAR>/<meta>`。
- `binding`：冻结 B3 和 MaskPLS，只训练对齐头。此阶段使用人工审核的回答标记作内部 probe。对齐头训练完成后，可直接用于“自由文本→分割→后插入标记”输出；不要求冻结的 B3 学会生成新 token。
- `joint`：在线读取 B3 隐状态，更新新增 q/v LoRA、两个新 token 的输入/输出行和对齐头。`--tune-spatial` 才允许更新 MaskPLS。新增 token 行单独以 FP32 训练，旧词表冻结，不为整张词表建立 Adam 状态。
- `--init-bundle` 仅支持从 `binding` 进入 `joint`，继承已训练的对齐头，重置优化器。只读取小型对齐权重，不同时加载第二个 7B 模型。当前没有断点续训入口；每次训练使用新目录。
- 联合损失为 `L_text + L_bind + L_mask`。对齐 CE 以实际预测实例与 GT 掩码的几何一对一匹配为目标；mask 损失为候选软掩码加权后的 BCE + Dice。`binding` 不计算文本梯度。推理使用离散选择实例的原始点序掩码。
- 候选沿用空间模型现有类别、置信度、竞争和过滤规则。覆盖不足的正例记为 `skipped_uncovered`，不伪装成 NOOBJ；只有人工确认的 `no_target` 才训练拒绝状态。日志单列候选数、正例数、覆盖数和有效绑定数。整个训练没有有效绑定监督时失败退出。

在线 forward 直接使用 `prepare_inputs_labels_for_multimodal()` 返回的展开后 labels 定位 assistant 的 SEG/NOOBJ，支持至多八个不重叠短语。问题、系统提示及视觉位置不计文本损失；上下文超长直接报错，避免目标被静默截断。自由生成后再执行一次完整 forward 获取标记隐状态，第二次 forward 不读取真实回答。

## 环境与配置

在现有 `wqlc` 环境中使用 PyTorch 2.8、Transformers 4.47.0、PEFT 0.13.2，以及现有 MaskPLS 所需的 MinkowskiEngine、SciPy、nuScenes SDK。补充语言依赖可执行：

```bash
conda run -n wqlc python -m pip install -r mllm/answerseg/requirements.txt
```

复制 `mllm/scripts/answerseg_runtime.example.json` 和 `answerseg_data.example.json`，替换为实际绝对路径。runtime 的 `stage2` 是原 `load_pretrained_model()` 第一个 adapter 参数，可将完整 B3 adapter 放在这里；如果 B3 由两层 adapter 构成，依次填写 `stage2/stage3`。不要将已合并模型目录填入 adapter 字段。空间路径要求原生、可经 `load_native()` 校验的 checkpoint，数据配置与 runtime 必须指向同一权重。

B3 原训练路径的环境、数据和特征闭合仍按 `mllm/scripts/preflight_b3.py` 检查。回答分割另有下面的 `preflight`，检查其实际绑定数据。当前接口未声明 7B + 稀疏骨干在特定显存上的峰值；应先用真实权重、真实一帧执行最小训练预算，再决定正式训练预算。

共享 GPU 空闲显存不足时，runtime 可显式增加 `"language_device":"cpu"`：7B 语言部分和其 LoRA 位于 CPU，空间模型与对齐头仍在 `--device` 指定的 GPU。在线隐状态和文本损失会跨设备传递，保留反向梯度；CPU 模式需要额外主存并降低速度。未填写时两端沿用同一设备。测试前应读取当前 GPU 占用并给已有任务留下余量，不自动结束其他任务。

## 请求与特征映射

`requests.jsonl` 每行给出问题和真实输入帧，**不含真实回答或分割目标**。`question` 可以原样包含既有 `<video>/<4DLiDAR>/<meta>` 文本；没有 `<video>` 时自动添加一个。元数据由调用者按 B3 的既有协议构造，此入口不会从答案恢复帧或元数据。

```json
{"request_id":"train-q001","scene_token":"actual-scene-token","sample_data_token":"actual-LIDAR_TOP-token","split":"train","question":"Which vehicle obstructs the route?","features":{"path":"scene.npy","sha256":"actual-64-hex-sha256","indices":[0,1],"frame_tokens":["frame-token-0","actual-LIDAR_TOP-token"],"input_policy":"explicit_frames","encoder_checkpoint_id":"actual-SST-weight-hash"}}
```

`path` 相对 feature root，文件必须为已有 SST 的 `[T,768]` NPY。`indices` 指定本次输入的原特征行，不填则取全部；`frame_tokens` 与选出的行逐一对应，允许如实记录重复采样帧。特征哈希和 encoder 身份必须保存；不猜测 NPY 行与 nuScenes 帧的对应，不把 MaskPLS 的 256 维 query 当成 B3 的 768 维特征。

`input_policy` 只允许 `explicit_frames/whole_scene`，拒绝 `answer_frames`。监督数据准入还从原始 metadata 核对所有上下文帧均属于同一场景、来自 LIDAR_TOP，防止训练场景中混入开发场景特征。映射本身仍需要从实际特征提取记录建立；仅有哈希不能证明特征编码正确。

## 采集与审核导入

```bash
conda run -n wqlc python mllm/scripts/run_answerseg.py collect \
  --runtime /path/runtime.json --requests /path/requests.jsonl \
  --feature-root /path/features --output-dir /path/new-generated
```

只采集 B3 回答，记录实际模型文件哈希、输入特征、是否正常 EOS 结束和完整记录指纹。生成文件为 `generated.jsonl`，状态始终 pending，不自动进入训练。截断回答不能导入监督。

人工核对问题、回答、相机/点云及唯一实例后，另建 `reviews.jsonl`。`span` 是**原始回答**的零基字符半开区间；不能手工在原始回答中预插 token：

```json
{"request_id":"train-q001","record_fingerprint":"copy-from-generated-record","decision":"accept","reviewer":"actual-reviewer","reviewed_at":"2026-10-21T12:00:00+08:00","evidence":"实际核对过的帧与指代证据","targets":[{"span":[4,9],"status":"matched","instance_token":"actual-unique-instance-token"}]}
```

多个对象提供各自不重叠的 span。确认无对象时用 `status=no_target, instance_token=null`，并保留审核证据；不确定、遮挡和候选漏检不能自动变成这个状态。审核只接受原指纹的原回答，改动回答需要重新采集/审核。

`feature-index.jsonl` 每行包含 `scene_token/sample_data_token/features`，其 features 与采集输入完全一致。它也用于导入已有 `b4dl-answer-binding-v1` 的人工接受记录：

```bash
conda run -n wqlc python mllm/scripts/run_answerseg.py prepare \
  --data /path/data.json --source /path/new-generated/generated.jsonl \
  --reviews /path/reviews.jsonl --feature-index /path/feature-index.jsonl \
  --split train --output-dir /path/reviewed-train
```

已有接受记录可把 `--source` 指向原 `binding/train.jsonl`，不再传 `--reviews`；原 `candidates.jsonl` 的 pending 记录会被拒绝。同一 QA/回答/帧的不同已审核 span 合并为一个多目标样本，原回答保持原样，重叠指代拒绝。内部开发集单独运行 `--split internal_dev`，不得重标为 train。

`prepare` 会将语言绑定与现有完整空间 manifest 逐帧连接，复用 MaskPLS/Grounded4D 的 panoptic、原始点序、类别和文件哈希校验；原数据保持原样。全部校验成功才将 `.part` 发布为 `reviewed.jsonl`。开发/训练路径分别写入 data 配置；只需提供各自实际用到的配置字段。

## 训练、保存与推理

```bash
conda run -n wqlc python mllm/scripts/run_answerseg.py preflight \
  --data /path/data.json --output-dir /path/new-preflight

conda run -n wqlc python mllm/scripts/run_answerseg.py train \
  --runtime /path/runtime.json --data /path/data.json --stage binding \
  --epochs 1 --output-dir /path/new-binding

conda run -n wqlc python mllm/scripts/run_answerseg.py train \
  --runtime /path/runtime.json --data /path/data.json --stage joint \
  --init-bundle /path/new-binding/epoch-1 --epochs 1 --lr 0.0001 \
  --output-dir /path/new-joint
```

这里的 1 epoch 仅演示入口，不是已验证的训练配方。默认 batch=1，无自动扩大预算；候选覆盖不足先查空间模型。每轮记录训练损失、有效更新、未覆盖数和 teacher-forcing 开发诊断；开发数据与训练数据按场景隔离。训练完成不等于自由生成或性能验收。

每轮 `epoch-N/` 保存 tokenizer、对齐头、新 token 行、新语言 LoRA、可选空间权重、优化器和数据摘要。`bundle.json` 记录原 B3/空间权重和本接口实现哈希。加载时先核对源权重、扩展文件、tokenizer、token ID 和参数形状，再恢复扩展；需要同时提供匹配的原始权重。扩展不会覆盖原 B3 或原空间 checkpoint。

预测请求在上述 requests 字段上增加原始点云 `point_file={path,sha256}`，path 相对 point root；请求不需要 panoptic、实例 token 或真实回答：

```bash
conda run -n wqlc python mllm/scripts/run_answerseg.py predict \
  --runtime /path/runtime.json --bundle /path/new-joint/epoch-1 \
  --requests /path/predict-requests.jsonl --feature-root /path/features \
  --point-root /path/nuScenes --output-dir /path/new-predictions

conda run -n wqlc python mllm/scripts/run_answerseg.py evaluate \
  --runtime /path/runtime.json --data /path/data.json \
  --bundle /path/new-joint/epoch-1 --output-dir /path/new-evaluation
```

输出 JSON 包含回答、每个标记的选择状态、帧内 query/object ID 和置信度；NPZ 包含 `[目标数,原始点数]` bool masks、原始 point_indices 和点云哈希。帧内 query ID 不作为跨帧身份。

默认先自由回答再分割，不以回答中是否存在 SEG 作为准入条件。生成截断、选择器拒绝、低置信度与空掩码分别记录；失败不插入输出标记。旧的原生 token 诊断保留为 model.predict_native()，它单独区分缺少 token、NOOBJ 等状态，不使用随机向量或最近实例兜底。所有失败正例都计入 Recall 分母；评测输出 Recall/Precision@0.5、每个目标的最佳 IoU 均值和明确 NOOBJ 准确率，不将后者命名为标准 cIoU/gIoU。评测范围为审核过的 internal_dev 绑定；mini 或已暴露场景不能据此宣称独立泛化。B3 六任务回归需使用本接口恢复模型，再通过 `model.bridge.generate()` 以相同问题和特征生成文本，并沿用原六任务指标计算；不能直接将本bundle作为旧入口的stage2/stage3 adapter。本轮尚未执行该回归。

## 本地验证与边界

### 自由回答后插入 SEG（2026-10-21）

默认 `model.predict()` 和 CLI `predict/evaluate` 的顺序为：原问题及原始输入 → B3 自由纯文本回答 → 对回答所指对象推理分割 → 对成功的非空掩码插入 `<SEG>` → 一起输出文本和分割结果。自由回答使用普通 `make_prompt(question)`，无需模型生成特殊 token，也不改成几何模板问题。

分割内部将回答文本的指代位置编码成 SEG probe，第二次 forward 提取其隐状态，交给已训练的对齐头选择真实 MaskPLS 候选。这些内部标记不作为输出成功的证据；只有通过置信度门槛且确有非空原始点序掩码的对象，才由渲染函数将 `<SEG>` 插入最终文本。默认单目标，使用完整回答（保留后置描述，略过末尾标点）作为一个指代；多个目标可调用 `segment_answer(..., spans=[[start,end], ...])` 指定纯文本字符位置，最多八个，不接受目标实例 ID、query 或 GT mask。该默认单目标约定不宣称能自动解析任意多对象回答。

Python 返回 `plain_answer`、含标记的 `answer`、`objects` 和实际 `masks`；每个对象含原文 `text_span/phrase`、状态、`seg_index`、`mask_index` 和输出文本中的 `output_marker_span`。拒绝对象的索引为 null，不占掩码行；前一个对象失败不会使后一个对象的标记与掩码错位。原文字符和标点保留，只新增成功标记。`seg_token_source=post_segmentation` 明确说明标记由分割后渲染产生，`seg_output_rate` 为输出标记比例，不冒充模型自然 token 生成率。CLI 的 JSON 与 NPZ 一起输出，用 `mask_file/mask_index` 连接；NPZ 继续保存完整原始点序及源点云哈希。

若自由回答已由外部 B3 生成，使用新增入口，不必再生成回答：

```bash
conda run -n wqlc python mllm/scripts/run_answerseg.py segment-answer \
  --runtime /path/runtime.json --bundle /path/trained-binding/epoch-1 \
  --requests /path/free-answers.jsonl --feature-root /path/features \
  --point-root /path/nuScenes --output-dir /path/segmented-answers
```

请求在原 `predict` 格式上增加 `answer`，可选 `answer_spans`。例如原文 `The car ahead.` 成功后输出 `The car ahead <SEG>.`；若对齐拒绝或低置信度则保留原文，不硬塞 SEG，不输出伪掩码。普通 `predict` 仍自动完成“生成→分割→渲染”全链路。人工审核绑定准入保持不变，几何启动模块及其 CLI 已依用户澄清移除。

`train` 另外支持 `--max-steps` 作为优化器更新上限、`--free-dev-samples` 进行独立自由回答后分割检查；部分 epoch 保存为 `step-N` 并如实记录。没有有效绑定监督则失败，不将纯文本更新称为分割训练。当前正式已审核 train/dev 仍为空，不能将工程权重或 CPU fixture 当成已训练的回答绑定模型。代码改变实现指纹，旧 bundle 不可静默加载。

```bash
conda run -n wqlc python -m unittest discover -s mllm/answerseg/tests -v
```

运行前使 `mllm/` 位于 `PYTHONPATH`。测试使用真实 VTimeLLM、Transformers、PEFT、小型本地 checkpoint 和显式 CPU 空间 fixture，覆盖视觉展开、LoRA 梯度、新 token 行、冻结状态、原始点序、NOOBJ/漏检区别、指纹与划分、多个对象、训练入口和 bundle 重载。CPU 检查不替代真实 7B 权重、MinkowskiEngine CUDA、mini 数据训练、自由生成效果、多种子或真正留出验收；本轮没有启动这些训练。
