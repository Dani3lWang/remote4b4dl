# 环境搭建

## Conda 环境

项目统一使用名为 **`wqlc`** 的 Conda 环境（Python 3.10），所有 Python 命令都必须在该环境中执行。

```bash
conda create -n wqlc python=3.10 -y && conda activate wqlc
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r mllm/requirements.txt
```

## ⚠️ 依赖文件选择

- **权威依赖文件是 `mllm/requirements.txt`**。
- 根目录的 `requirements.txt` **版本过新且与 mllm 模块冲突，不要使用**。
- 编码器模块另有 `encoders/lidarclip/requirements.txt`（mmcv_full 1.3.9、mmdet 2.14、pytorch_lightning 1.6.2、openai_clip 等）。

## 关键版本锁定及原因

| 包 | 锁定版本 | 原因 |
|----|---------|------|
| torch / torchvision | 2.5.1 / 0.20.1 | CUDA 12.4 构建基线 |
| transformers | 4.31.0 | VTimeLLM 依赖旧版 Trainer/generation 内部 API 与 monkey-patch；新版会破坏兼容 |
| peft | 0.4.0 | Stage3 的 `merge_and_unload()` + 重新加 LoRA 模式依赖旧式 API |
| deepspeed | 0.16.4 | ZeRO-3 训练；`train.py` 的 `no_sync` monkey-patch 针对该版本 |
| flash-attn | 2.7.0.post2 | `llama_flash_attn_monkey_patch.py` 依赖 `flash_attn_unpadded_qkvpacked_func` |
| decord | 0.6.0 | 推理时视频抽帧 |
| clip | 0.2.0 | 特征提取 / 推理时加载 CLIP ViT-L/14 |

**flash-attn 注意**：flash-attn 2.x 不支持 RTX 5090（sm_120）。`train_mem.py` 已做 try/except，ImportError 时自动回退 transformers 原生 SDPA（实际训练以 SDPA 为主）。另本机 `clip.load` 不转 fp16，`train.py` 已加 `clip.model.convert_weights()` 修复。

## METEOR 评测的 Java 依赖

评测的 METEOR 指标走 pycocoevalcap 的 Meteor-1.5 jar 后端，需要系统 Java：

```bash
apt-get install -y --no-install-recommends default-jre-headless
```

缺 jar 时评测代码会回退 NLTK METEOR-1.0 式实现（系统性偏高，与论文不可比）。

## 数据与模型准备

| 资源 | 来源 | 说明 |
|------|------|------|
| nuScenes v1.0-trainval | [nuscenes.org](https://www.nuscenes.com/) | 相机图像 + LiDAR 点云 + metadata JSON，需自行下载 |
| B4DL 数据集 | [HF ccho4702/nuScenes-B4DL](https://huggingface.co/datasets/ccho4702/nuScenes-B4DL) | 官方发布的 QA 数据与 metadata |
| Vicuna-7B v1.5 | [lmsys/vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) | 放到 `mllm/base_model/vicuna-v1-5-7b/` |
| CLIP ViT-L/14 | OpenAI 权重 | 推理时经 `--clip_path` 指定（默认 `checkpoints/clip/ViT-L-14.pt`） |
| OpenAI API Key | 环境变量 | `OPENAI_API_KEY`（必填）、`OPENAI_BASE_URL`（可选，默认官方）；模型名可经 `B4DL_GPT_MODEL` 覆盖，默认 `gpt-4o` |

## 硬件参考

训练显存上限参考：CLIP ViT-L/14 + batch 32 约占满 2×RTX 5090（32GB），超出可用梯度累积或改用 ViT-B/32（输出 512 维，需同步修改 mm_projector 输入维度）。
