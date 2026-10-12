# 第三阶段分割训练数据候选构建器

`tools/build_stage3_seg_candidates.py` 是独立的 CPU / Python 3.10+ 标准库工具。
它筛选“回答类别提及 → 场景 → 场景帧 → nuScenes 前景 3D 框标注”的候选关联，输出给后续掩码生成和指代验证流程。
它不导入训练模型，不修改 B3 QA、特征、loader、训练参数或训练入口，也不生成分割监督。

## 实际数据字段与路径

已依据仓库代码和 README 链接的官方数据核对：

| 来源 | 关键字段与含义 |
| --- | --- |
| `datageneration/generate_dataset.py::preprocessing` | `scene_id, scene_token, sequence_id, start_index, end_index, conversations`；两个 turn 分别是 `human/gpt` |
| HF `dataset/train/stage2.json, stage3.json` | `split, scene_token, scene_id, human_annotation, question, answer`；发布版没有 `sequence_id` 或精确输入帧 |
| HF `dataset/test/*.json` | `split, scene_token, scene_id, question, answer`，按六任务分文件；`binary/comprehensive` 与预测中的任务名有别名映射 |
| `mllm/scripts/build_stage2_full_train.py` | 将平铺 QA 转成 `conversations`；不会恢复缺失的 sequence 标识 |
| HF `metadata/scene_metadata.json` | 850 场景，`scene_id` 为字符串，应保留前导零；`paths.PATH_LIDAR_TOP.PATH_000` 是场景零基帧路径；split 为 `train/val` |
| HF `metadata/sequence_metadata.json` | 5,100 序列；`indices` 与 `frames` 逐项对应；帧内有 `sample_token, TOKEN_LIDAR_TOP, PATH_LIDAR_TOP` |
| `a Best b3/predictions.json` | 六任务的 `questions/ground_truths/predictions` 平行数组，共 30,145 条，完全没有场景或帧标识 |

**官方 `frames[].frame_id` 是随机标识符，不是帧号。** 关联必须使用 `indices`；不能把 `frame_id` 转为场景帧索引。
生成代码示例中 `frame_id` 又采用序列局部编号，因此两种版本都不能依赖它。
`scene_metadata` 的 `val` 对应 B4DL QA 的 `test`，不能当作训练集。

现有配置优先使用 `B4DL_NUSCENES_ROOT`；否则原始 nuScenes 默认根目录是仓库同级的 `nuScenes/`。
本地核查时 `D:\Github\nuScenes`、旧 `mllm/b4dl_dataset` 和 `encoders/lidarclip/annotations` 均不存在。
本工具不会从 `.npy` 场景特征反推点级标注，也不会重新生成随机 scene/sequence ID 来替代官方元数据。

## 下载官方生成 QA 与元数据

仓库根 README 指向 [ccho4702/nuScenes-B4DL](https://huggingface.co/datasets/ccho4702/nuScenes-B4DL)。
本次核查并下载的固定版本为 `438bd768356c366ea7e3bd5775e1cbaaa5df7d1b`。
从仓库根目录执行：

```powershell
python datageneration/tools/download_b4dl_candidate_sources.py
```

下载脚本只下载该数据集的 README、两份训练 QA、六份测试 QA 和两份元数据。
文件位于已被 Git 忽略的 `datageneration/data/b4dl-official/`，不涉及模型或原始 nuScenes 下载。
`download_manifest.json` 保存固定版本 URL、字节数和 SHA-256；已有缓存须与 manifest 哈希一致才复用，未知文件不会被覆盖。
通过 `--revision <40位commit SHA>` 和 `--destination <新目录>` 可选择其他快照。

## 使用你提供的 B3 回答做审计

```powershell
$qaFiles = Get-ChildItem datageneration/data/b4dl-official/dataset/test/*.json | ForEach-Object { $_.FullName }
python datageneration/tools/build_stage3_seg_candidates.py --input "a Best b3/predictions.json" --answer-source prediction --original-qa $qaFiles --scene-metadata datageneration/data/b4dl-official/metadata/scene_metadata.json --sequence-metadata datageneration/data/b4dl-official/metadata/sequence_metadata.json --output-dir datageneration/data/stage3-seg-candidates/predictions-audit
```

这里提取的是 **模型预测回答** 的类别，而不是 GT 类别。GT 只用于与原始 QA 配对。
关联键为“规范任务名 + 清理 `<video>/<4DLiDAR>/<meta>` 后的问题 + GT 回答”，只统一大小写和空白，保留标点和帧数字。
必须唯一命中：零命中拒绝，重复命中也拒绝；不会依靠数组顺序、挑第一个或消耗重复项队列猜测场景。
`--answer-source ground_truth` 可显式改为提取原始 GT 回答，来源字段也随之改变。

默认 `--expected-split train` 会排除所有已关联的测试场景。
`--expected-split test` 只用于测试集关联审计，其输出始终 `eligible_for_training=false`。
测试预测和 GT 都不能作为本构建器的训练来源；真正的训练候选应从官方训练 QA 构建。

## 构建第三阶段训练 QA 候选

```powershell
python datageneration/tools/build_stage3_seg_candidates.py --input datageneration/data/b4dl-official/dataset/train/stage3.json --scene-metadata datageneration/data/b4dl-official/metadata/scene_metadata.json --sequence-metadata datageneration/data/b4dl-official/metadata/sequence_metadata.json --nuscenes-root /path/to/nuScenes --output-dir datageneration/data/stage3-seg-candidates/train-candidates
```

也支持单轮 `conversations` QA 数组。默认 `--answer-source answer`；不支持把多轮回答任意拼接成一个分割查询。
默认使用仓库 `tools/assets/train_scene_tokens.json` 的 699 个训练场景白名单，可以显式指定 `--train-scene-tokens`。
原始表默认位于 `<nuscenes-root>/v1.0-trainval/`，可以通过 `--nuscenes-version` 切换。
每次输出使用新目录，已有清单不会被覆盖。运行失败保留 `.part` 文件，只有完整运行才发布最终 JSONL 和 summary。

前景类别采用 nuScenes 十类检测类别，以及 QA 词表中的 `wheelchair/stroller/debris`，具体叶类别映射也写入 summary。
大小写、复数和明确别名受支持，例如 `SUV → vehicle.car`、`trafficcone → movable_object.trafficcone`、`construction vehicle → vehicle.construction`。
独立的 `construction`、泛称 `vehicle`、`bike`、建筑物／道路／植被等不会硬映射为可用前景实例标注。
`pedestrian` 不包含 wheelchair/stroller 等其他叶类别。关键词按单词边界匹配，避免 `scar/cargo/busyness` 等子串误命中。
否定／假设提及采用保守的局部规则过滤，规则可能多拒绝一些复杂句子；它不提供语言理解或指代正确性的保证。
类别只从所选回答提取：问题中的 “car” 和回答 “Yes” 不会被补成已识别的 car。

## 关联门槛与输出

1. 校验 `scene_id/scene_token` 一致，QA 与场景的 split 一致，训练场景在白名单内。
2. 帧优先取问题的明确引用，问题没有引用时取所选回答的明确引用，记录 `frame_source`。
   范围使用场景帧号闭区间；若提供准确 `sequence_id/feat_indices`，则只保留对应采样帧。
   没有文本引用时，只有准确的 `sequence_id/feat_indices` 才可提供采样上下文；`start_index/end_index/feat_range` 不足以恢复精确采样。
   不回退 frame 0、整场景、最近序列或最大重叠序列，不悄悄裁剪越界引用。
3. 用原始 `scene.json/sample.json` 的链表重建场景帧序号，核对 scene 长度和 metadata token/path。
   原始 `sample_data.json` 本身没有 SDK 的 `sample.data` 字典：本工具通过 `calibrated_sensor → sensor` 校验 keyframe 的 `LIDAR_TOP` channel。
4. 校验 LiDAR 文件位于根目录内、存在且满足 nuScenes float32 × 5 的字节布局。
   用 `sample_annotation → instance → category` 核对真实叶类别，并要求 `num_lidar_pts > 0` 和有限、有效的框几何。
   多个同类实例全部保留，标记 `multiple_instance_candidates=true`；不会挑选“最近对象”充当已验证指代。
5. 可附带同一 `sample_data_token` 的 `lidarseg/panoptic` 来源路径。
   `--require-point-labels` 可要求点标注文件存在；**文件存在不等于内容、类别、实例或点顺序已验证**。

| 文件 | 内容 |
| --- | --- |
| `audit.jsonl` | 每条输入一个审计记录，保留所选回答、GT、来源文件 SHA-256、原始行号、匹配数量、类别文本 span、场景和帧来源、拒绝原因 |
| `candidates.jsonl` | 每个通过门槛的 QA × 类别 × 帧一条候选；含确定性的 candidate ID、sample/LiDAR token、文件路径、框标注 token、instance/category 和全局框几何 |
| `rejected.jsonl` | 拒绝的 QA 或类别／帧候选及原因，保留来源便于重建与排查 |
| `summary.json` | 输入、代码和原始表哈希，类别映射，参数、数量、拒绝原因计数，以及缺失原始表清单 |

所有条目都保留 `mask_status=not_generated`、`mask_verified=false`、`grounding_verified=false`、`supervision_ready=false`。
通过候选的关联等级仅为 `answer_class_present_in_frame`，框来源明确是 `annotation_kind=3d_box`。
这是同类标注在该帧存在的证据；它不能确认回答中的方向、运动、时间、对象指代或实例选择正确。
点 mask 的生成方式、框内点伪标签误差、点顺序、实例 ID 与可见性、查询指代一致性，都需要后续流程验证。

主要拒绝原因包括：

| 原因 | 含义 |
| --- | --- |
| `ambiguous_original_qa / original_qa_not_found` | 旧预测不能唯一关联原始 QA |
| `excluded_split / qa_metadata_split_mismatch / scene_not_in_training_whitelist` | 测试集隔离或来源划分不一致 |
| `scene_id_token_mismatch / unknown_scene_*` | 场景关联无效 |
| `no_supported_answer_category / only_negated_or_uncertain_mentions` | 回答没有支持的正向类别提及 |
| `missing_exact_frame_reference / invalid_frame_reference` | 无法精确关联帧，或引用越界／倒序 |
| `sequence_scene_mismatch / frame_reference_outside_context` | 序列或采样上下文冲突 |
| `nuscenes_tables_unavailable` | 原始 nuScenes 标注表未提供；仍输出审计，但不会发布候选 |
| `metadata_nuscenes_frame_mismatch / missing_lidar_file / unsafe_data_path` | 原始帧、文件身份或路径边界不满足要求 |
| `class_annotation_not_found / class_has_no_lidar_points / invalid_annotation_box_geometry` | 没有对应类别的有效 LiDAR 框证据 |
| `missing_point_label_file` | 显式要求点标注文件，但该帧没有可用文件 |

计数按输入行统计，一个输入可有多个原因，因此拒绝原因之和不等于输入总数。
缺少原始 nuScenes 时进程可正常完成来源审计，summary 的 `nuscenes_tables_available=false` 和零候选明确表示关联证据尚未闭合。

## 验证

```powershell
python -m unittest discover -s datageneration/tests -p test_stage3_seg_candidates.py -v
```

测试使用原始 JSON 表结构和小型合成 LiDAR 文件，覆盖来源唯一性、split 隔离、采样索引、路径边界、类别标注和掩码未验证状态。
这验证候选构建逻辑，不代表真实 nuScenes 的完整标注闭合或分割训练效果。
