# B4DL 4D LiDAR 可视化 Demo

该 Demo 直接读取本地 nuScenes 场景，提供当前帧点云的 3D/BEV 视图、六路相机切换、真值框和截至当前帧的历史轨迹。它不读取或修改训练数据。

## 安装

推荐新建 Python 3.10 环境（例如 `wqlc`），安装独立 Demo 依赖：

```bash
cd mllm
pip install -r requirements-demo.txt
```

该清单仅安装 Gradio、Plotly、nuScenes SDK 等查看依赖，不安装 PyTorch、
DeepSpeed 或 flash-attn。nuScenes 1.1.11 依赖旧版 matplotlib，因此使用
NumPy 1.x；请与使用 NumPy 2.x 的训练环境分开。纯查看模式不加载模型权重，
也不要求 CUDA。

## 纯查看模式

```bash
cd mllm
python -m vtimellm.demo_gradio \
  --nuscenes_root /path/to/nuScenes \
  --nuscenes_version v1.0-trainval \
  --scene_metadata /path/to/scene_metadata.json
```

`--scene_metadata` 可省略。省略后仍可浏览 nuScenes 原生场景，但由于无法解析 B4DL 的随机 `scene_id`，模型问答不会启用。数据根目录也可以通过环境变量 `B4DL_NUSCENES_ROOT` 设置。

## 查看并启用模型问答

在独立 Demo 环境中增加推理依赖：

```bash
pip install -r requirements-inference.txt
```

推理清单使用项目说明中的 PyTorch 2.8、transformers 4.47、PEFT 0.13.2
和 accelerate 1.3，不需要编译 flash-attn。RTX 4090 推荐使用默认的
`--dtype auto --attn_implementation sdpa`：支持时选择 BF16；可显式设置
`--dtype float16` 以复核原 FP16 评测行为。改变精度后应重新评测，不能假设
逐样本输出完全相同。

以下四项必须成套提供，`--stage3` 可选：

```bash
python -m vtimellm.demo_gradio \
  --nuscenes_root /path/to/nuScenes \
  --scene_metadata /path/to/scene_metadata.json \
  --model_base /path/to/vicuna-7b-v1.5 \
  --pretrain_mm_mlp_adapter /path/to/mm_projector.bin \
  --stage2 /path/to/stage2-adapter \
  --feat_folder /path/to/stage2_features
```

模型始终读取完整场景的 `[N, 768]` 特征，并检查 `N` 是否与 nuScenes 场景关键帧数一致。任何缺失或错位都会只禁用当前场景的问答，不会静默使用错误特征。

常用参数：

- `--max_points 30000`：每帧发送给浏览器的最大点数。
- `--gpu_id 0`：CUDA 逻辑设备编号；遵循 `CUDA_VISIBLE_DEVICES`。
- `--max_new_tokens 512`：单次答案生成上限。
- `--max_context_tokens 4096`：上下文预算，实际还会受模型配置限制。
- `--attn_implementation eager`：原生 SDPA 的兼容性排查选项。
- `--server_name 0.0.0.0`：允许局域网访问。
- `--server_port 7860`：修改监听端口。
- `--share`：启用 Gradio 分享链接。

模型请求按单并发执行；场景特征缓存在 CPU，当前推理时才传入 GPU。
上下文预算包含 `<video>` 展开后的 LiDAR 特征；过长对话会移除最早的完整
问答轮次并保留整场景特征，单个问题仍过长时会明确提示。生成失败不会改写
原会话状态。4090 的 24 GB 显存通常可作为 7B 半精度单模型推理的起点，
实际余量受 KV cache、其他 GPU 进程和权重影响；此处不构成显存实测结果。

## 训练曲线

增加 `--trainer_state /path/to/checkpoint/trainer_state.json` 可展示 training
loss、已记录的 eval_loss 和 learning rate。横轴为优化器更新步数，重复步数
保留该指标最后一条记录，非有限数值会跳过。此功能只读取已有日志，不会
启动训练；重新启动 Demo 可载入后续 checkpoint 的更新。

```bash
python -m vtimellm.demo_gradio \
  --nuscenes_root /path/to/nuScenes \
  --trainer_state /path/to/checkpoint-1000/trainer_state.json \
  --predictions /path/to/predictions.json \
  --metrics /path/to/metrics.json
```

### RTX 4090 的 B3 训练入口

在已经通过 `scripts/preflight_b3.py` 的训练 `wqlc` 环境中使用：

```bash
cd mllm
B4DL_ENV_PREFIX=/path/to/training/wqlc \
B4DL_TRAIN_PROFILE=rtx4090 \
B4DL_GPU_ID=0 \
bash scripts/run_b3.sh
```

4090 配置采用 micro-batch 1、gradient accumulation 128、现有 ZeRO-3 CPU
offload 配置；单 GPU 的有效 batch 仍为 128。空闲显存门槛为 20000 MiB，
检查选定 GPU，checkpoint、日志和评测目录使用 `-rtx4090` 后缀，以便独立
恢复及对比。2 epochs、学习率和 LoRA 参数保持 B3 配方。CPU offload 需要
足够的主机内存且可能明显降低速度；这些是配置建议，仍需真实 4090 数据
训练测量峰值显存和吞吐。

未指定配置时保留原 B3 参数（8 × 16、28000 MiB 门槛）。若显存总量小于
门槛，入口会立即报错并提示选择 4090 配置，避免等待 72 小时。可通过
`B4DL_TRAIN_MIN_FREE_MB` 调整空闲显存门槛；门槛只用于等待设备空闲，
不是模型峰值显存的保证。仅重新评测 4090 checkpoint 时使用相同变量并运行
`bash scripts/run_b3.sh 2`。

## 测试

核心测试不需要 Gradio、nuScenes 数据集或 GPU：

```bash
cd mllm
python -m unittest tests.test_lidar_visualizer -v
```

完整验收建议先使用 `v1.0-mini`，确认点云、相机、框和轨迹在同一帧对齐，再切换到 `v1.0-trainval` 和真实 B4DL 权重。

## 模型效果看板

效果看板读取已经完成的评测结果，不重新运行模型，也不要求 CUDA。新版
`evaluation/test_b4dl.py` 会在 `predictions.json` 中保存每条预测对应的
`scene_id`；旧版预测可通过原始测试集安全补全场景关联。

```bash
cd mllm
python -m vtimellm.demo_gradio \
  --nuscenes_root /path/to/nuScenes \
  --scene_metadata ../encoders/lidarclip/annotations/scene_metadata.json \
  --predictions ./eval_results/stage2_full_seqv3_mixed_b3/predictions.json \
  --metrics ./eval_results/stage2_full_seqv3_mixed_b3/metrics.json \
  --test_data ./b4dl_dataset/test_qa.json
```

页面增加三个区域：

- **效果总览**：最终指标与论文 Table 3 对比、per-task 指标、分类混淆矩阵、
  TG IoU 分布和开始帧散点图。
- **样本诊断**：按任务、状态、关键词和得分分页筛选；选择样本后显示点云、
  BEV、相机、问题、Ground Truth、预测和任务对应的单样本得分。
- **论文案例图**：从当前评测页载入一个样本，选择 2–8 个场景帧（默认均匀
  选择 5 帧），生成同步的前视、后视和 LiDAR BEV 三行视图；可编辑两组
  模型答案与黄色/绿色高亮短语，并下载 180 DPI PNG 和 PDF。

时间定位时间轴使用数据集原始 0 基帧号。界面的 `DATASET FRAME 006 ·
POSITION 7/40` 表示数据集帧号为 6，同时它是场景中的第 7 帧。

### 输入组合

| 参数 | 效果 |
|---|---|
| 仅 `--metrics` | 只显示汇总指标；没有样本列表 |
| `--predictions` | 显示文本和单样本诊断；新版结果可直接关联场景 |
| 旧 `--predictions` + `--test_data` | 校验问题和 GT 后补全 `scene_id` |
| 再提供完整模型参数 | 在效果看板之外同时启用自由问答 |

若旧结果无法与 `test_qa.json` 唯一匹配，页面保留文本诊断并明确显示未关联，
不会猜测或跳转到错误场景。

同一“问题 + GT”对应不同场景或输入帧时均视为歧义，不按预测顺序猜测。
新版结果也可仅用 `scene_token` 关联 nuScenes；同时提供两种场景键时会检查
一致性。未关联样本的论文案例场景选择会清空，需手动选择后才能导出。

看板显示 `answer_frames` 的 oracle 输入选择和 METEOR 后端信息。论文柱状图
只是参考数值，比较前须核对测试集、特征构造、输入选择和指标后端。

## 单独导出论文案例图

不启动 Gradio 时，可直接调用独立渲染器。`--frames` 留空会在整段场景中
均匀选择 5 帧；高亮短语支持用逗号、分号或换行分隔。

```bash
cd mllm
python vtimellm/paper_case_visualizer.py \
  --nuscenes-root /path/to/nuScenes \
  --scene-metadata ../encoders/lidarclip/annotations/scene_metadata.json \
  --scene-token <nuscenes-scene-token> \
  --frames "0, 10, 20, 30, 39" \
  --question "What dynamic movement is observed throughout the frames?" \
  --baseline-label "VTimeLLM" \
  --baseline-answer "Vehicles in front move forward." \
  --b4dl-answer "Front vehicles move forward while rear vehicles move away." \
  --baseline-highlights "Vehicles in front" \
  --b4dl-highlights "Front vehicles,rear vehicles" \
  --output-dir ./paper_cases
```

相机行使用 nuScenes 标定将真值 3D 框投影到画面；LiDAR 行是适合打印的
静态 BEV，并可叠加真值框和历史轨迹。该流程只读数据与评测输出，不加载或
修改训练代码。

论文案例图的 CPU 单元测试：

```bash
python -m unittest tests.test_paper_case_visualizer -v
```

运行全部 CPU 回归测试（包括真实模块 CLI、场景关联、上下文预算及 Gradio
构建）使用：

```bash
python -m unittest discover -s tests -v
```
