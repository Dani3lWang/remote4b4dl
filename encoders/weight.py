import torch
import clip

# 检查是否有可用的 GPU，否则使用 CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 自动下载并加载预训练模型（例如 ViT-B/32 版本）
model, preprocess = clip.load("ViT-L/14", device=device)

print("模型加载成功！")