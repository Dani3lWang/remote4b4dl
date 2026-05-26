import torch
import os

# ================= 配置区域 =================
MODEL_PATH = "/root/autodl-tmp/wql/mmb4dl/encoders/lidarclip/pretrained/ViT-L-14.pt"          # 你的模型文件路径
WEIGHT_PATH = "/root/autodl-tmp/wql/mmb4dl/encoders/lidarclip/lidarclip/checkpoint/vit_l_14.ckpt"       # 你的权重文件路径
OUTPUT_PATH = "/root/autodl-tmp/wql/mmb4dl/encoders/lidarclip/pretrained/ViT-L-14-mer.pt"  # 整合后保存的新模型路径
# ==========================================

print("🚀 开始加载模型与权重...")

# 1. 加载 TorchScript 模型
try:
    # 对于 TorchScript 模型，建议使用 torch.jit.load 加载，或者 torch.load(..., weights_only=False)
    model = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    print(f"✅ 成功加载 TorchScript 模型: {MODEL_PATH}")
except Exception as e:
    print(f"❌ 加载模型失败: {e}")
    exit()

# 2. 加载预训练权重 (vit_l_14.ckpt)
try:
    checkpoint = torch.load(WEIGHT_PATH, map_location='cpu', weights_only=False)
    print(f"✅ 成功加载权重文件: {WEIGHT_PATH}")
except Exception as e:
    print(f"❌ 加载权重失败: {e}")
    exit()

# 3. 提取真正的权重字典
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
elif isinstance(checkpoint, dict) and 'model' in checkpoint:
    state_dict = checkpoint['model']
else:
    state_dict = checkpoint

# 4. 处理键名前缀 (visual., module. 等)
new_state_dict = {}
prefixes_to_remove = ['visual.', 'module.', 'vision_model.', 'backbone.']
for k, v in state_dict.items():
    new_key = k
    for prefix in prefixes_to_remove:
        if k.startswith(prefix):
            new_key = k[len(prefix):]
            break
    new_state_dict[new_key] = v

print("🔧 正在将权重映射到 TorchScript 模型中...")
# 5. 将权重加载到模型里
# 注意：TorchScript 模型加载权重时，strict=False 可能不完全适用，视具体版本而定
try:
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    if missing_keys:
        print(f"⚠️ 缺失的键: {missing_keys}")
    if unexpected_keys:
        print(f"⚠️ 多余的键: {unexpected_keys}")
    if not missing_keys and not unexpected_keys:
        print("✅ 权重完美匹配！")
except Exception as e:
    print(f"⚠️ 加载权重时遇到警告或错误（TorchScript 模型限制）: {e}")
    print("我们将尝试直接保存处理好的权重字典...")

# 6. 核心修改：只保存权重字典（state_dict），不保存整个 TorchScript 模型对象
output_dir = os.path.dirname(OUTPUT_PATH)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# 保存处理好的、已经去掉前缀的纯净权重
torch.save(new_state_dict, OUTPUT_PATH)
print(f"🎉 整合完成！合并后的纯净权重已保存至: {OUTPUT_PATH}")
print("💡 提示：以后你可以先加载 ViT-L-14.pt 模型，再使用 load_state_dict 加载这个 merged_weights.pt 文件。")