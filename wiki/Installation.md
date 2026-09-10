# 环境搭建

## Conda 环境

项目统一使用名为 **`wqlc`** 的 Conda 环境（Python 3.10），所有 Python 命令都必须在该环境中执行。

```bash
conda create -n wqlc python=3.10 -y && conda activate wqlc
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r mllm/requirements.txt
```

## ⚠️ 依赖文件选择

- **权威依赖文件是 `mllm/requirements.txt`**。
- 根目录的 `requirements.txt` **版本过新且与 mllm 模块冲突，不要使用**。
- 编码器模块另有 `encoders/lidarclip/requirements.txt`（mmcv_full 1.3.9、mmdet 2.14、pytorch_lightning 1.6.2、openai_clip 等）。

## 关键版本（2026-09-08 实测运行版本，CLAUDE.md 已同步）

| 包 | 实测版本 | 说明 |
|----|---------|------|
| torch / torchvision | 2.8.0+cu128 / 0.23.0+cu128 | sm_120（RTX 5090）原生支持；早期文档的 2.5.1/cu124 为历史锁定 |
| transformers | 4.47.0 | 原生 SDPA 需要 ≥4.45（4.31 旧锁定已随 sm_120 升级作废）；DoRA（peft 侧）、NEFTune、现代 logprob API 均可用 |
| peft | 0.13.2 | `merge_and_unload`/`get_peft_model`/`LoraConfig(use_dora=True)` 实测可用（B4a 训练在 4.47/0.13.2 栈上跑通为证） |
| deepspeed | 0.16.4 | ZeRO-3 训练；`train.py` 的 `no_sync` monkey-patch 针对该版本 |
| accelerate | 1.3.0 | TRL 要求 ≥1.1（RL 引进时满足） |
| flash-attn | 未安装 | sm_120 不支持 flash-attn 2.x；`train_mem.py` try/except，ImportError 时自动回退 transformers 原生 SDPA（实际训练以 SDPA 为主） |
| decord / clip | 0.6.0 / 0.2.0 | 推理视频抽帧 / 加载 CLIP ViT-L/14 |
| nltk / pycocoevalcap | 3.10.0 / 1.2 | METEOR dual 双后端（NLTK-2005 主口径 + jar 参考，见下） |

另本机 `clip.load` 不转 fp16，`train.py` 已加 `clip.model.convert_weights()` 修复。

## METEOR 评测的 Java 依赖

评测的 METEOR 自 2026-09-07 起默认 **dual 双后端**：

- `meteor`（主口径）= **NLTK-2005**（论文引用 [2] 的忠实实现，**无需 Java**），与论文对比一律用它
- `meteor_pycocoevalcap`（参考）= pycocoevalcap 的 Meteor-1.5 jar 后端，仅用于衔接 B0-B3 旧表，需要系统 Java：

```bash
apt-get install -y --no-install-recommends default-jre-headless
```

缺 Java/jar 时只有参考字段不可用，不影响主口径 `meteor`。

## 数据与模型准备

| 资源 | 来源 | 说明 |
|------|------|------|
| nuScenes v1.0-trainval | [nuscenes.org](https://www.nuscenes.com/) | 相机图像 + LiDAR 点云 + metadata JSON，需自行下载 |
| B4DL 数据集 | [HF ccho4702/nuScenes-B4DL](https://huggingface.co/datasets/ccho4702/nuScenes-B4DL) | 官方发布的 QA 数据与 metadata |
| Vicuna-7B v1.5 | [lmsys/vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) | 放到 `mllm/base_model/vicuna-v1-5-7b/` |
| CLIP ViT-L/14 | OpenAI 权重 | 推理时经 `--clip_path` 指定（默认 `checkpoints/clip/ViT-L-14.pt`） |
| OpenAI API Key | 环境变量 | `OPENAI_API_KEY`（必填）、`OPENAI_BASE_URL`（可选，默认官方）；模型名可经 `B4DL_GPT_MODEL` 覆盖，默认 `gpt-4o` |

## 硬件参考

- **MLLM 训练**：单卡 RTX 5090（32GB）+ ZeRO-3，实测 3 epoch 24-35h/代（24-36 s/step）；管线脚本内置 28GB 空闲显存门控（与 CoRViD 等任务共用 GPU 时自动等待，见 [[Training]]）
- **编码器训练**：ViT-L/14 + batch 32 约占满 32GB（超出用梯度累积或 ViT-B/32——输出 512 维，需同步修改 mm_projector 输入维度）
