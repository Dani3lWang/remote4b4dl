# Data Generation Pipeline
**You should make your own OpenAI API key before running the code.**
```bash
cd datageneration
```

## 4D LiDAR Context Extraction Step
Please download the `nuScenes` dataset and set the `nuscenes_root` argument to the download path.

Run the following commands:
```bash
bash scripts/generate_description.sh
```
or you can run the python code directly
```bash
python3 generate_description.py \
    --start_index 10 \
    --end_index 20 \
    --api_key {your openai api key} \
    --nuscenes_root /mnt/nfs_shared_data/dataset/cch/nuScenes \
    --dataroot ./data
```

## Context-to-QA Transformation Step
```bash
bash scripts/generate_dataset.sh
```

or you can run the python code directly
```bash
python3 generate_dataset.py \
    --start_index 0 \
    --end_index 10 \
    --api_key {your openai api key} \
    --nuscenes_root /mnt/nfs_shared_data/dataset/cch/nuScenes \
    --dataroot ./data \
    --task existence
```

## Stage 3 segmentation candidates

独立的候选构建器可从平铺 QA、单轮 conversations 或 B4DL 预测回答识别前景类别，
核对场景、帧和 nuScenes 3D 框标注，输出来源与拒绝原因清单。
类别提及和框关联均不表示掩码已验证；该工具不接入或改动 B3 training path。
官方 QA 下载、预测归属恢复、测试集隔离、原始标注要求与运行方法见
[第三阶段分割候选构建说明](docs/stage3_seg_candidates.md)。
