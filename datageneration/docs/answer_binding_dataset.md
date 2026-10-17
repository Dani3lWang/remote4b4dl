# 回答绑定候选与点云标签集

`tools/build_answer_binding_dataset.py` 将官方训练 QA 与已经通过准入的单帧 panoptic 数据集关联，输出可复核的回答绑定候选和完整空间标签包。保持原有训练/开发场景划分；不读取测试预测，不启动模型训练，不改写 B3 路径。

## 两种不同的训练准入

- `spatial/train.jsonl`、`spatial/internal_dev.jsonl` 与 `spatial/masks/*.npz`：每帧完整 Thing 实例监督，可由现有 Grounded4D/MaskPLS 数据加载器读取。原始点和官方 panoptic 文件继续从数据根目录读取。
- `binding/candidates.jsonl`：问题、数据集回答、词面提及、暂定目标句段、帧引用和所有同类实例候选，每个对象保留 `instance_token`、`mask_row`、类别、点数和中心。即使只有一个候选也不自动确认指代。
- `binding/train.jsonl` 与 `binding/internal_dev.jsonl`：只发布经过明确人工审核的绑定。没有审核输入时两者为空，这是有意的训练准入约束。

NPZ `labels` 为 `[K,N]` int8，1=目标、0=非目标、-1=官方忽略点，点轴完整保留原始 nuScenes 行序。ROI 外的标签仍保留，空间训练加载器应用配置 ROI；默认范围为 `[-51.2,-51.2,-5,51.2,51.2,3)`，含1点实例。候选只枚举现有空间标签中在 ROI 内有点的实例，因此没有候选不能推断为无物体。

来源已有空间准备器提供 panoptic 实例到 annotation 的几何匹配证据。本工具重新核对原始点/标签哈希、点序、全部实例/类别/忽略标签、样本链和标注身份，并重算所选 annotation 框内点比例；不把几何匹配当成人工语言审核。

官方 QA 缺少明确帧引用时拒绝关联，不默认选择第一帧；只交集已有空间帧。多帧句段保存完整引用范围、可用帧范围和覆盖标志，不把某一帧候选当作完成了时序指代。由答案提取的帧范围记录为 `answer_frame_reference`，不能当作无答案辅助的部署定位效果。

## 运行

在 wqlc 环境执行，依赖 NumPy，JSON 数组使用标准库流式读取：

```bash
python datageneration/tools/build_answer_binding_dataset.py \
  --qa /path/stage2.json /path/stage3.json \
  --scene-metadata /path/scene_metadata.json \
  --sequence-metadata /path/sequence_metadata.json \
  --train-scene-tokens datageneration/tools/assets/train_scene_tokens.json \
  --label-root /path/existing-spatial-labels \
  --train-manifest /path/existing-spatial-labels/train.jsonl \
  --dev-manifest /path/existing-spatial-labels/internal_dev.jsonl \
  --point-root /path/nuScenes \
  --panoptic-root /path/nuScenes/nuScenes-panoptic-v1.0-all \
  --metadata-root /path/nuScenes/v1.0-trainval \
  --output-dir /path/new-independent-dataset
python datageneration/tools/audit_answer_binding_dataset.py /path/new-independent-dataset
```

必须使用新输出目录。只有 `summary.json` 的 `complete=true` 才代表构建成功；中断目录不可用于训练。磁盘审计另写 `audit.json`，核对每条候选到实际 NPZ 的实例行、点数、类别和划分，并检查待审核样本没有进入绑定训练文件。

`summary.json` 记录输入与工具哈希、源数据根目录、数量和排除原因。`objects.jsonl` 是完整实例索引，其掩码路径相对 `spatial/`。原始点云不复制到标签包，消费端必须配置 `source_roots.points`；移动数据后按相同相对路径重映射根目录。

## 人工审核

`binding/review.tsv` 是文本审核表，可以用文本编辑器或表格软件打开。相同回答中的每个类别提及独立记录；`target_phrase` 只是提及所在句段，不能视为已经解析出的唯一指代短语。审计生成的 `review_first300.jsonl` 优先列出最多300条训练集单帧、单候选记录，仍需人工核实方向、属性、运动和问答上下文。

审核通过的记录另存 JSONL，保留候选原始 ID 和指纹。例如：

```json
{"candidate_id":"实际ID","candidate_fingerprint":"实际指纹","decision":"accept","instance_token":"实际候选token","target_span":[0,8],"reviewer":"审核人","reviewed_at":"ISO日期时间","evidence":"检查过的点云/相机证据及选择理由"}
```

`target_span` 为原回答的零基字符半开区间，必须包含对应类别提及。每条记录明确绑定一个实例；多对象使用不同记录，不把所有同类实例合并。无法唯一确定、未看清或缺少对应帧的记录保留 pending，不改成 NOOBJ。重新运行构建器时提供 `--reviews reviewed.jsonl` 和新输出目录；接受记录自动写入其原划分，开发集不能进入训练。

当前发布的是官方训练数据集回答，`answer_source=dataset_ground_truth`，不代表当前 B4DL 自由生成结果。之后采集模型训练场景回答时必须保存场景/帧、模型身份与输出来源，再单独审核映射。

## 验证

```bash
python -m unittest discover -s datageneration/tests -p test_answer_binding_dataset.py -v
```

覆盖点序篡改、原始文件哈希、错误掩码、开发场景隔离、审核身份/证据和大记录流式解析。实际数据构建和磁盘审计的结果在独立运行目录记录，不把合成测试通过等同于真实语言绑定正确。
