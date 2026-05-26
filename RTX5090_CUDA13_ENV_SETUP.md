# B4DL RTX 5090 (Blackwell sm_120) 环境配置指南

## 硬件与系统环境

| 项目 | 当前状态 |
|------|---------|
| GPU | 2x NVIDIA GeForce RTX 5090 |
| 计算能力 | sm_120 (Blackwell) |
| 驱动版本 | 580.142 |
| 系统 CUDA | /usr/local/cuda-12.8 (nvcc 12.8.93) |
| 已装 PyTorch (base) | 2.11.0+cu130 |

## 核心问题分析

### 为什么原版 requirements 不能在 RTX 5090 上运行

| 包 | 原版版本 | 问题 |
|----|---------|------|
| **torch** | 2.5.1+cu124 | 预编译二进制**不含 sm_120 内核** |
| **flash_attn** | 2.7.0.post2 | **不支持 Blackwell**，需要 3.x（API 不兼容） |
| **transformers** | 4.31.0 | 无原生 SDPA，依赖 flash_attn monkey-patch |
| **deepspeed** | 0.16.4 | 预编译 wheel 可能不带 sm_120 的 op |

### 解决策略

1. **PyTorch 升级** → `>=2.7.0` + `cu128` 或 `cu130`（含 sm_120 内核）
2. **移除 flash_attn** → 用 PyTorch 原生 `scaled_dot_product_attention` (SDPA) 替代
3. **transformers 升级** → `>=4.45.0`（内置 SDPA，无需 monkey-patch）
4. **peft 同步升级** → `>=0.13.0`（兼容新版 transformers）
5. **放弃 flash_attn monkey-patch** → 直接使用 transformers 原生注意力

---

## 环境搭建步骤

### Step 1: 创建 conda 环境

```bash
conda create -n b4dl-rtx5090 python=3.10 -y
conda activate b4dl-rtx5090
```

### Step 2: 安装 CUDA 工具链 (nvcc)

```bash
# 系统已有 cuda-12.8，直接设置环境变量
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH

# 验证
nvcc --version  # 应输出: Cuda compilation tools, release 12.8
```

### Step 3: 安装 PyTorch (含 sm_120 支持)

PyTorch >= 2.7.0 的 cu128/cu130 预编译包包含 sm_120 内核。

```bash
# 推荐: PyTorch 2.8.0 + CUDA 12.8（稳定且广泛测试）
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# 备选: 使用系统已有的 CUDA 13.0 路线
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

验证 sm_120 支持：

```bash
python3 -c "
import torch
print('CUDA:', torch.version.cuda)
print('Capability:', torch.cuda.get_device_capability())
print('sm_120 in arch list:', 'sm_120' in torch.cuda.get_arch_list())
t = torch.zeros(1, device='cuda', dtype=torch.bfloat16)
print('bf16 on CUDA OK:', t.device, t.dtype)
print('SDPA available:', hasattr(torch.nn.functional, 'scaled_dot_product_attention'))
"
```

### Step 4: 安装核心依赖

```bash
# ---- Transformers 生态 (升级版) ----
pip install transformers==4.47.0         # 内置 SDPA，取代 flash_attn
pip install tokenizers>=0.13.0,<0.20
pip install peft==0.13.2                 # 兼容新版 transformers
pip install accelerate==1.3.0            # deepspeed 训练适配层

# ---- DeepSpeed ----
# ⚠️ 不要装预编译 wheel，从源码编译以确保 sm_120 op 兼容
pip install deepspeed==0.16.4 --no-binary deepspeed

# ---- 无需安装 flash_attn ----
# 用 PyTorch 原生 SDPA 替代（见下方代码修改）

# ---- LiDARCLIP 编码器 ----
pip install einops==0.8.1
pip install pytorch_lightning==1.6.2
pip install mmengine==0.10.6
pip install openai_clip==1.0.1

# ---- nuScenes 数据处理 ----
pip install nuscenes_devkit==1.1.10
pip install pyquaternion==0.9.9
pip install opencv_python==4.11.0.86
pip install Shapely==1.8.5

# ---- 工具包 ----
pip install numpy==2.3.5
pip install scipy==1.15.0
pip install Pillow==12.0.0
pip install tqdm==4.67.1
pip install easydict==1.13
pip install Requests==2.32.5
pip install sentencepiece==0.2.0
pip install decord==0.6.0
pip install setuptools>=68.0.0

# ---- ChatGLM 支持 (可选，仅 ChatGLM 路线需要) ----
pip install cpm_kernels==1.0.11

# ---- Gradio Web Demo (可选) ----
pip install gradio==6.0.1

# ---- 评估指标 (可选) ----
pip install pycocoevalcap==1.2
pip install bert_score==0.3.13
pip install moverscore==1.0.3
pip install dvc-eval                # ⚠️ 原 requirements 遗漏的包
```

### Step 5: 下载 CLIP 权重 (LiDARCLIP 特征提取需要)

```bash
mkdir -p ./encoders/lidarclip/pretrained
# 下载 ViT-L-14.pt 到 ./encoders/lidarclip/pretrained/
wget -P ./encoders/lidarclip/pretrained/ \
  https://openaipublic.azureedge.net/clip/models/b8cca856fdcd45cf6f6f6f6f6fcaf6f6f/ViT-L-14.pt
```

---

## 代码修改（适配 RTX 5090）

### 修改 1: 禁用 flash_attn monkey-patch

**文件**: `mllm/vtimellm/train/train_mem.py`

```python
# 原版（注释掉 flash_attn patch）
# from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
# replace_llama_attn_with_flash_attn()

from train import train

if __name__ == "__main__":
    train()
```

> **说明**: transformers >= 4.45 内置了 `scaled_dot_product_attention` (SDPA)，LlamaModel 会自动使用它。
> 不再需要 flash_attn 的 monkey-patch。

### 修改 2: builder.py LoRA 加载适配新版 peft

**文件**: `mllm/vtimellm/model/builder.py`

如果 PeftModel.from_pretrained 报错，需要更新参数名：

```python
# 新版 peft (>0.5.0) 的参数名可能有变化
# 将 model = PeftModel.from_pretrained(model, lora_path)
# 尝试改为:
model = PeftModel.from_pretrained(model, lora_path, is_trainable=False)
```

大多数情况下 peft 0.13.2 的 API 仍兼容旧的调用方式，无需修改。

### 修改 3: DeepSpeed Zero 配置 (可选)

**文件**: `mllm/scripts/zero3.json`

RTX 5090 有 32GB 显存，可以尝试减小 offload：

```json
{
    "bf16": {
        "enabled": "auto"
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6
    }
}
```

### 修改 4: run_stages.sh CUDA 检测更新

**文件**: `mllm/run_stages.sh`

在第 29 行的版本列表中加入 12.8：

```bash
for _ver in 12.8 12 12.6 12.5 12.4 12.3 12.2 12.1 11.8; do
```

---

## 验证流程

### 1. 基础 CUDA 可用性

```bash
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'GPU: {torch.cuda.get_device_name(0)}')
cap = torch.cuda.get_device_capability(0)
print(f'Compute Capability: sm_{cap[0]}{cap[1]}')
assert (cap[0], cap[1]) == (12, 0), f'Expected sm_120, got sm_{cap[0]}{cap[1]}'
print('[OK] sm_120 confirmed')
"
```

### 2. SDPA 可用性

```bash
python3 -c "
import torch
from torch.nn.functional import scaled_dot_product_attention
x = torch.randn(1, 8, 128, 64, device='cuda', dtype=torch.bfloat16)
y = scaled_dot_product_attention(x, x, x, is_causal=True)
print(f'[OK] SDPA output shape: {y.shape}')
"
```

### 3. 模型加载测试（不需要数据）

```bash
cd mllm
python3 -c "
import torch, transformers
print(f'transformers: {transformers.__version__}')
from transformers import LlamaForCausalLM, AutoTokenizer
# 只测 tokenizer 加载和 config
tokenizer = AutoTokenizer.from_pretrained('./base_model/vicuna-v1-5-7b', use_fast=False)
print(f'[OK] Tokenizer loaded, vocab_size={tokenizer.vocab_size}')
"
```

### 4. DeepSpeed 状态检查

```bash
ds_report
```

确保 JIT 编译的 op (transformer, inference_core_ops, cutlass_ops) 状态为 `[OKAY]`。

### 5. LiDARCLIP 特征提取测试

```bash
cd encoders/lidarclip
python3 -c "
from lidarclip.model.sst import LidarEncoderSST
import torch
model = LidarEncoderSST()
x = torch.randn(2, 2500, 5, device='cuda')
with torch.no_grad():
    y = model(x)
print(f'[OK] SST encoder output shape: {y.shape}')
"
```

---

## 一键安装脚本

另存为 `setup_rtx5090.sh`:

```bash
#!/bin/bash
set -e

ENV_NAME="${1:-b4dl-rtx5090}"

echo "=== 创建 conda 环境: $ENV_NAME ==="
conda create -n "$ENV_NAME" python=3.10 -y

echo "=== 设置 CUDA 环境变量 ==="
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH

echo "=== 激活环境并安装 PyTorch ==="
conda run -n "$ENV_NAME" pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128

echo "=== 安装 transformers 生态（升级版）==="
conda run -n "$ENV_NAME" pip install \
  transformers==4.47.0 \
  "tokenizers>=0.13.0,<0.20" \
  peft==0.13.2 \
  accelerate==1.3.0

echo "=== 从源码编译安装 deepspeed ==="
conda run -n "$ENV_NAME" pip install deepspeed==0.16.4 --no-binary deepspeed

echo "=== 安装 LiDARCLIP 依赖 ==="
conda run -n "$ENV_NAME" pip install \
  einops==0.8.1 \
  pytorch_lightning==1.6.2 \
  mmengine==0.10.6 \
  openai_clip==1.0.1 \
  nuscenes_devkit==1.1.10 \
  pyquaternion==0.9.9 \
  opencv_python==4.11.0.86 \
  Shapely==1.8.5

echo "=== 安装工具包 ==="
conda run -n "$ENV_NAME" pip install \
  numpy==2.3.5 \
  scipy==1.15.0 \
  Pillow==12.0.0 \
  tqdm==4.67.1 \
  easydict==1.13 \
  Requests==2.32.5 \
  sentencepiece==0.2.0 \
  decord==0.6.0 \
  "setuptools>=68.0.0"

echo "=== 验证安装 ==="
conda run -n "$ENV_NAME" python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'SDPA: {hasattr(torch.nn.functional, \"scaled_dot_product_attention\")}')
t = torch.zeros(1, device='cuda', dtype=torch.bfloat16)
print(f'bf16 test: OK')
print(f'sm_120 in arch list: {\"sm_120\" in torch.cuda.get_arch_list()}')
"

echo "=== 环境 $ENV_NAME 创建完成 ==="
echo "请执行: conda activate $ENV_NAME"
echo "然后完成上方文档中的 3 处代码修改后再运行训练。"
```

---

## 依赖版本对照表

| 包 | 原版 (不兼容) | 新版 (RTX 5090) | 原因 |
|----|-------------|-----------------|------|
| torch | 2.5.1+cu124 | **2.8.0+cu128** | 需要 sm_120 内核 |
| flash_attn | 2.7.0.post2 | **移除** | 不支持 sm_120；用 SDPA 替代 |
| transformers | 4.31.0 | **4.47.0** | 内置 SDPA，替代 flash_attn |
| peft | 0.4.0 | **0.13.2** | 兼容新版 transformers |
| accelerate | 0.21.0 | **1.3.0** | 兼容新版 transformers |
| deepspeed | 0.16.4 | 0.16.4 (**源码编译**) | 需用 CUDA 12.8 重新编译 op |
| 其他 | - | 不变 | 无 sm_120 冲突 |
