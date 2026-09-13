# 架构

```text
nuScenes
├─ datageneration     相机图像 → 场景描述 → 六类 QA
├─ encoders/lidarclip 点云序列 → SST → 768 维逐帧特征
└─ mllm               特征 → mm_projector → VTimeLLM → 文本答案
```

训练数据中的 `<video>` token 会在 `prepare_inputs_labels_for_multimodal()` 中被投影后的 LiDAR 特征替换，对应 label 设为 `IGNORE_INDEX`。B3 使用 `--whole_scene`，每条 QA 输入该场景的完整特征序列。

核心文件：

- `mllm/vtimellm/model/vtimellm_arch.py`
- `mllm/vtimellm/train/dataset.py`
- `mllm/vtimellm/train/train.py`
- `mllm/evaluation/test_b4dl.py`
