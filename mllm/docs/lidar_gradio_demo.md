# B4DL 4D LiDAR 可视化 Demo

该 Demo 直接读取本地 nuScenes 场景，提供当前帧点云的 3D/BEV 视图、六路相机切换、真值框和截至当前帧的历史轨迹。它不读取或修改训练数据。

## 安装

在项目的推理环境中安装独立 Demo 依赖：

```bash
cd mllm
pip install -r requirements-demo.txt
```

该文件在原有推理依赖之上追加 Plotly 和 nuScenes SDK，但不会修改原始依赖清单或训练代码。纯查看模式不加载模型权重，也不要求 CUDA。

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
- `--server_name 0.0.0.0`：允许局域网访问。
- `--server_port 7860`：修改监听端口。
- `--share`：启用 Gradio 分享链接。

## 测试

核心测试不需要 Gradio、nuScenes 数据集或 GPU：

```bash
cd mllm
python -m unittest tests.test_lidar_visualizer -v
```

完整验收建议先使用 `v1.0-mini`，确认点云、相机、框和轨迹在同一帧对齐，再切换到 `v1.0-trainval` 和真实 B4DL 权重。
