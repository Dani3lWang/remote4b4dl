# 数据生成管线（datageneration/）

调用 GPT-4o API，从 nuScenes 相机图像生成 LiDAR 视角的场景描述（Step 1），再转化为 6 类任务的问答对（Step 2）。对应论文 §3 的数据集构建流程。

## 运行前准备

- 设置环境变量：`OPENAI_API_KEY`（必填）、`OPENAI_BASE_URL`（可选，默认官方 API）、`B4DL_GPT_MODEL`（可选，默认 `gpt-4o`）
- 或通过 `--api_key` 参数直接传入（两个主脚本均必填）
- 下载 nuScenes v1.0-trainval，通过 `--nuscenes_root` 指定路径

## Step 1 — 4D LiDAR 上下文提取

```bash
cd datageneration
bash scripts/generate_description.sh
# 或
python generate_description.py \
    --start_index 10 --end_index 20 \
    --api_key YOUR_API_KEY \
    --nuscenes_root /path/to/nuScenes \
    --dataroot ./data
```

**处理流程**：
- 每段序列取 `LOAD_N_FRAMES=5` 帧（`FRAME_INTERVAL=2` 间隔，如 index [0,2,4,6,8]）
- 6 个相机视图分两组：**前组**（FRONT/FRONT_LEFT/FRONT_RIGHT，15 张图）、**后组**（BACK/BACK_LEFT/BACK_RIGHT，15 张图），分别 base64 编码后各调一次 GPT-4o
- Prompt 要求从 LiDAR 点云视角描述（忽略颜色/文字），输出场景描述 + 关键时序变化 + 重要物体与事件
- 同时从 nuScenes `scene.json` 读取人工场景描述写入 `gt_caption`（论文 §3.2 的 structured human annotations）

**输出** `data/generated_description/generated_description_{start}_{end}.json`：

```json
{"scene_token": "...", "sequence_id": "1234567",
 "description_front": "...", "description_back": "...",
 "gt_caption": "人工场景描述", "start_index": 0, "end_index": 8}
```

## Step 2 — 上下文转 QA 数据集

```bash
bash scripts/generate_dataset.sh
# 或
python generate_dataset.py \
    --start_index 0 --end_index 10 \
    --api_key YOUR_API_KEY \
    --task existence \
    --dataroot ./data
```

读取 Step 1 描述 JSON → 按 `--task` 选择 prompt → GPT-4o 返回 `Q:/A:` 文本 → 正则切分解析 → 组装 Vicuna 对话格式。

**输出** `data/generated_dataset/{task}/generated_{task}_dataset_{start}_{end}.json`：

```json
{"scene_id": "...", "scene_token": "...", "sequence_id": "...",
 "start_index": 0, "end_index": 8,
 "conversations": [{"from": "human", "value": "Q..."},
                   {"from": "gpt", "value": "A..."}]}
```

## 六类任务与 QA 形式

`--task` 支持 6 个规范值 + 2 个别名（`binary_qa→binary`、`temporal_understanding→temporal`）：

| task | 每序列请求量 | QA 形式 | 示例 |
|------|------------|---------|------|
| `existence` | 5 条 | 物体在某帧/帧段是否存在，或哪类物体存在 | `Q: Was a pedestrian present in frame 004? A: Yes.` |
| `binary` | 10 条 | 仅限 14 类物体的 Yes/No 问题（存在、移动、交互） | `Q: Was a bus in front of the ego vehicle between frame 002 and frame 006? A: No.` |
| `time_grounding` | 5 条 | "何时发生"，答案必须是单个时间跨度，**精确格式 `from frame 000 to frame 000.`**（三位零填充+句点） | `A: from frame 000 to frame 008.` |
| `description` | 5 条 | 问题固定为 `Describe the lidar-sequence.`，回答整体描述 | — |
| `temporal` | 10 条 | 简单问答，问题必须含帧号时间信息 | `Q: When did the ego vehicle change lanes? A: from frame 004 to frame 010.` |
| `comprehensive` | 10 条 | 综合推理（空间关系、物体交互、动态演变） | — |

简单任务（existence/binary/time_grounding）与复杂任务（description/temporal/comprehensive）在训练时分别用于 Stage2 与 Stage3。

## 主要 CLI 参数

**generate_description.py**：`--api_key`（必填）、`--gpt_model`、`--frame_interval`（2）、`--load_n_frames`（5）、`--generate_n_sets`（6）、`--nuscenes_root`、`--nuscenes_version`（v1.0-trainval）、`--dataroot`（./data）、`--start_index/--end_index`（0/1000）

**generate_dataset.py**：`--api_key`（必填）、`--description_dir`（data/generated_description）、`--task`、`--start_index/--end_index`、`--dataroot`

## tools/ 辅助脚本

| 脚本 | 用途 |
|------|------|
| `create_metadata.py` | 从 nuScenes 构建元数据（每 scene 均匀采样 6 段序列的 `sequence_metadata.json` + scene 划分的 `scene_metadata.json`）。论文所用 metadata 已在 HF 发布，此脚本仅作参考；直接运行需补齐 `Config.from_args()` 等缺失配置 |
| `build_stage1_from_lidarllm.py` | 把 HF 数据集 `Senqiao/LiDAR-LLM-Nu-Caption`（161,845 条）过滤到训练 scene，转为 Stage-1 训练数据（scene_id=sample_token），对齐论文 162K。需本地 nuScenes sample.json |
| `build_stage1_from_lidarllm_official.py` | 同上用途的官方参考版：内置 `assets/` 映射表，不依赖 nuScenes，支持自动从 HF 下载，输出 `stage1_train.json` |
| `generate_stage1_caption.py` | 单帧双视角 caption 参考脚本（论文 Stage-1 实际使用 LiDAR-LLM 数据，此脚本未使用；运行需补齐缺失配置） |

## 注意事项

- **`start_index` / `end_index` 必须是 `SAVE_TERM`(10) 的倍数**，否则脚本直接退出（description 版打印提示 exit，dataset 版抛 ValueError）
- **time_grounding 有格式过滤**：答案不含 `"from frame"` 短语的 QA 对直接丢弃；GPT 解析本身也有落地损失（请求 10 条、落地约 5.7 条的经验比率）
- 输出文件名由 `START_INDEX + i*SAVE_TERM` 计算且读取时按文件名末尾数字排序，**文件名格式不能改动**
- Prompt 中反复强调视角约束："后视图中左侧物体对应自车右侧"（LiDAR 视角）
- Git 历史中曾有硬编码 API key（已改为环境变量读取），**公开发布前需清理历史**
