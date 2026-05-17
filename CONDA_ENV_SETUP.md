# B4DL Conda 环境配置指南

screen -r kelly

## 项目环境总览

本项目包含四个子系统，需要 **2 个独立 conda 环境**：

| 环境名 | Python | CUDA | 用途 | 是否需要 GPU |
| --- | --- | --- | --- | --- |
| `b4dl` | 3.10 | 12.4 | LiDARCLIP 训练 + VTimeLLM 训练/推理 + 评估 | 是 (Ampere+) |
| `b4dl_datagen` | 3.10 | - | 数据生成管线 (调用云端 LLM API) | 否 |

> **说明**: 评估模块 (`requirements_b4dl_eval.txt`) 的依赖已合并到 `b4dl` 主环境中，无需单独环境。

---

## 两环境合并为单环境

`b4dl_datagen` 的依赖非常轻量（`openai`、`httpx`、`Pillow`、`numpy`、`tqdm`），与 `b4dl` 主环境 **无任何版本冲突**：

| 包 | b4dl_datagen 要求 | b4dl 实际安装 | 兼容? |
| --- | --- | --- | --- |
| `numpy` | ≥2.0.0 | 2.3.5 | ✓ |
| `Pillow` | ≥10.0.0 | 12.0.0 | ✓ |
| `tqdm` | ≥4.60.0 | 4.67.1 | ✓ |
| `openai` | ≥1.30.0 | (无) | ✓ 仅 datagen 用 |
| `httpx` | ≥0.27.0 | (无) | ✓ 仅 datagen 用 |

**如果希望只维护一个环境**，在 `b4dl` 中追加安装即可：

```bash
conda activate b4dl
pip install openai>=1.30.0 httpx>=0.27.0
```

合并后所有操作都在 `b4dl` 一个环境中完成。

**两环境 vs 单环境的选择**：

| 场景 | 推荐方案 |
| --- | --- |
| 同一台机器上既训练又生成数据 | **合并为单环境**，省去切换 |
| 专用训练服务器 (GPU) + 数据生成在另一台机器 (CPU) | **两环境分开**，datagen 机器无需装 CUDA/PyTorch |
| 只想快速跑数据生成，不涉及深度学习 | **只用 `b4dl_datagen`**，安装快、体积小 |

**结论**：如果数据生成和训练在同一台 GPU 机器上进行，直接合并到 `b4dl` 即可，`b4dl_datagen` 不是必需的。

---

## 硬件要求

| 组件 | 要求 |
| --- | --- |
| GPU | NVIDIA RTX 30/40 系列或更高 (Ampere/Ada 架构) |
| CUDA | 12.4 |
| 显存 | 训练: ≥24GB (推荐 A100 40/80GB 用于 DeepSpeed ZeRO-3 多卡) |
| Java | OpenJDK 11+ (评估模块 `pycocoevalcap` 需要) |

---

## 第一步：安装 CUDA 12.4 与 OpenJDK (系统级)

```bash

# === OpenJDK 11 (评估需要) ===
conda install -c conda-forge openjdk=11 -y
```

---
# === CUDA 12.4 (如未安装) ===
# 从 NVIDIA 官网下载: https://developer.nvidia.com/cuda-12-4-0-download-archive
# 或通过 conda 安装 cuda-toolkit:
conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit -y



## 第二步：创建 b4dl 主环境

```bash
# 1. 创建环境并安装 PyTorch
conda create -n b4dl python=3.10 -y
conda activate b4dl

# 2. 安装 PyTorch 2.5.1 (CUDA 12.4 版本)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# 3. 安装 VTimeLLM 语言模型核心依赖
pip install transformers==4.31.0
pip install tokenizers==0.13.2
pip install peft==0.4.0
pip install accelerate==0.21.0

# 4. 安装训练加速组件
pip install deepspeed==0.16.4
pip install flash_attn==2.7.0.post2
pip install einops==0.8.1

# 5. 安装 OpenAI CLIP (PyPI 包名为 openai_clip)
pip install openai_clip==1.0.1

# 6. 安装视频处理
pip install decord==0.6.0

# 7. 安装 LiDARCLIP 编码器训练依赖
pip install pytorch_lightning==1.6.2
pip install mmengine==0.10.6

# 8. 安装 nuScenes 数据处理
pip install nuscenes_devkit==1.1.10
pip install pyquaternion==0.9.9
pip install opencv_python==4.11.0.86
pip install Shapely==1.8.5

# 9. 安装 ChatGLM 中文版依赖 (可选)
pip install sentencepiece==0.2.0
pip install cpm_kernels==1.0.11

# 10. 安装通用工具
pip install numpy==2.3.5
pip install scipy==1.15.0
pip install Pillow==12.0.0
pip install tqdm==4.67.1
pip install easydict==1.13
pip install Requests==2.32.5
pip install 'setuptools>=68.0.0'

# 11. 安装 Gradio Web Demo (可选)
pip install gradio==6.0.1

# 12. 安装评估指标依赖
pip install pycocoevalcap==1.2
pip install bert_score==0.3.13
pip install moverscore==1.0.3
```

### 一键安装 (b4dl 主环境)

```bash
conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit -y
conda create -n b4dl python=3.10 -y && \
conda activate b4dl && \
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 && \
pip install -r requirements_b4dl.txt && \
pip install -r requirements_b4dl_eval.txt
```

---

## 第三步：创建 b4dl_datagen 数据生成环境

```bash
# 1. 创建环境
conda create -n b4dl_datagen python=3.10 -y
conda activate b4dl_datagen

# 2. 安装依赖
pip install openai>=1.30.0
pip install httpx>=0.27.0
pip install Pillow>=10.0.0
pip install numpy>=2.0.0
pip install tqdm>=4.60.0
```

### 一键安装 (b4dl_datagen 环境)

```bash
conda create -n b4dl_datagen python=3.10 -y && \
conda activate b4dl_datagen && \
pip install -r requirements_b4dl_datagen.txt
```

---

## 第四步：验证环境

```bash
# 切换到主环境
conda activate b4dl

# 验证 PyTorch + CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# 验证 flash_attn
python -c "from flash_attn import flash_attn_func; print('flash_attn OK')"

# 验证 deepspeed
python -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"

# 验证 LiDARCLIP 编码器
python -c "import mmengine; import pytorch_lightning; print('LiDARCLIP deps OK')"

# 验证 OpenAI CLIP
python -c "import clip; print(f'CLIP available, model: ViT-L/14')"

# 验证 nuScenes
python -c "from nuscenes.nuscenes import NuScenes; print('nuScenes devkit OK')"

# 验证评估指标 (需要 Java)
python -c "from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer; print('pycocoevalcap OK')"
python -c "import bert_score; print('bert_score OK')"
```

---

## 版本锁定说明

以下包的版本被严格锁定，不能随意升级：

| 包 | 锁定版本 | 锁定原因 |
| --- | --- | --- |
| `transformers` | 4.31.0 | `llama_flash_attn_monkey_patch.py` 对 `LlamaAttention.forward` 做了 monkey-patch |
| `peft` | 0.4.0 | `builder.py` 的 LoRA 加载/合并依赖旧式 API (≥0.5.0 接口变更) |
| `deepspeed` | 0.16.4 | ZeRO-3 配置与训练脚本兼容性 |
| `flash_attn` | 2.7.0.post2 | monkey-patch 兼容性 |
| `pytorch_lightning` | 1.6.2 | LiDARCLIP 训练代码基于旧版 API |
| `mmengine` | 0.10.6 | SST 配置/注册器兼容 mmcv 2.x |

---

## LiDARCLIP 额外说明

`encoders/lidarclip/sst/` 目录包含了 vendored SST 检测器代码，原始依赖 `mmcv 1.x` + `mmdet 2.x`。`lidarclip/model/sst.py` 中通过大量 shim 代码 (mock 模块) 使其兼容新版 `mmengine 0.10.6`。**因此使用 `requirements_b4dl.txt` 安装即可，无需安装 `mmcv` / `mmdet`**。

---

## 环境使用场景对照

| 操作 | 环境 | 关键脚本/命令 |
| --- | --- | --- |
| 生成场景描述 | `b4dl_datagen` | `bash scripts/generate_description.sh` |
| 生成 QA 数据集 | `b4dl_datagen` | `bash scripts/generate_dataset.sh` |
| 训练 LiDARCLIP | `b4dl` | `python encoders/lidarclip/train.py` |
| 提取点云特征 | `b4dl` | `python encoders/lidarclip/extract_pc_features.py` |
| Stage 1 对齐训练 | `b4dl` | `bash mllm/scripts/stage1.sh` |
| Stage 2 LoRA 微调 | `b4dl` | `bash mllm/scripts/stage2.sh` |
| 完整训练流程 | `b4dl` | `bash mllm/run_stages.sh` |
| 模型推理 | `b4dl` | `python mllm/vtimellm/inference.py` |
| 模型评估 | `b4dl` | `python mllm/vtimellm/eval/eval.py` |
| Gradio Web 演示 | `b4dl` | `python mllm/vtimellm/demo_gradio.py` |
