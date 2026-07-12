# Gradio Demo 启动记录

> 日期：2026-05-31  
> 分支：remote  
> 背景：尝试在 RTX 5090 服务器上启动 VTimeLLM Gradio Web Demo

---

## 环境

| 项目 | 状态 |
|------|------|
| 环境名 | `wqlc` (conda) |
| Python | 3.10.20 |
| PyTorch | 2.8.0+cu128 |
| CUDA | 12.8 |
| GPU | RTX 5090 (sm_120) |
| 服务器 | autodl-tmp |

---

## 问题 1: 缺少 --model_base 参数

**错误**: `demo_gradio.py: error: the following arguments are required: --model_base`

**原因**: `--model_base` 是必填参数，指向 vicuna-7b-v1.5 huggingface checkpoint 路径

**解决**: 指定 `--model_base /root/autodl-tmp/ljq/mmb4dl-main/base_model/vicuna-7b-v1.5`

---

## 问题 2: root_dir 路径解析 Bug

**错误**: `FileNotFoundError: .../checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin`

**原因**: `demo_gradio.py:6` 中 `root_dir = os.path.join(os.getcwd(), "..")`，从 `mllm/` 运行时 `root_dir` 指向 repo 根目录的父级。但 checkpoints 实际位于 `mllm/checkpoints/` 下，导致默认路径全部错误。

**需要显式指定的参数**:
| 参数 | 默认路径（错误） | 实际路径 |
|------|-----------------|---------|
| `--clip_path` | `../checkpoints/clip/ViT-L-14.pt` | 使用模型名 `ViT-L/14` 让 clip 库自动下载 |
| `--pretrain_mm_mlp_adapter` | `../checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin` | `mllm/checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin` |
| `--stage2` | `../checkpoints/vtimellm-vicuna-v1-5-7b-stage2` | `mllm/checkpoints/vtimellm-vicuna-v1-5-7b-stage2` |
| `--stage3` | `../checkpoints/vtimellm-vicuna-v1-5-7b-stage3` | `mllm/checkpoints/vtimellm-vicuna-v1-5-7b-stage3` |

---

## 问题 3: CLIP checkpoint 文件不兼容

**使用本地文件时报错**: `RuntimeError: PytorchStreamReader failed locating file constants.pkl: file not found` → `EOFError`

**原因**: 本地找到的 `ViT-L-14.pt` 文件（来自多个备份路径）与当前 PyTorch 2.8.0 序列化格式不兼容

**解决**: 传入模型名 `ViT-L/14`（而非文件路径），让 `clip` 库自动下载兼容版本到 `~/.cache/clip/`

---

## 正确的启动命令

```bash
conda run -n wqlc python vtimellm/demo_gradio.py \
  --model_base /root/autodl-tmp/ljq/mmb4dl-main/base_model/vicuna-7b-v1.5 \
  --clip_path ViT-L/14 \
  --pretrain_mm_mlp_adapter /root/autodl-tmp/wql/mmb4dl/mllm/checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
  --stage2 /root/autodl-tmp/wql/mmb4dl/mllm/checkpoints/vtimellm-vicuna-v1-5-7b-stage2 \
  --stage3 /root/autodl-tmp/wql/mmb4dl/mllm/checkpoints/vtimellm-vicuna-v1-5-7b-stage3
```

---

## 模型加载流程

1. 加载 vicuna-7b-v1.5 基座模型 + tokenizer（约 14GB，2 个 safetensors shard）
2. 加载 Stage1 mm_projector（`mm_projector.bin`，Linear 128→4096）
3. 加载并合并 Stage2 LoRA 权重
4. 加载并合并 Stage3 LoRA 权重
5. 加载 CLIP ViT-L/14（约 428MB，从 ~/.cache/clip/ 加载）
