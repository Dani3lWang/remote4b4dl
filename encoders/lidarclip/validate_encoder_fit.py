"""编码器适配性验证：新训 nuScenes 编码器 vs 旧 ONCE 权重
同一批 nuScenes 数据上，比较两者与 CLIP 图像特征的对齐质量：
  ① MSE（训练目标本身的量）
  ② 成对余弦相似度结构相关性（语义结构是否与 CLIP 空间一致——下游 LLM 真正依赖的属性）
  ③ 特征范数（供 projector 重训参考）
"""
import os
import sys
import time
import importlib.util

import numpy as np
import torch

os.chdir("/root/autodl-tmp/wql/mmb4dl/encoders/lidarclip")
sys.path.insert(0, os.getcwd())
spec = importlib.util.spec_from_file_location("lc_train", "train.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
import clip
from lidarclip.model.sst import LidarEncoderSST
from lidarclip.loader import build_loader

N_BATCHES, BATCH = 100, 16


def load_encoder(ckpt_path):
    enc = LidarEncoderSST("lidarclip/model/sst_encoder_only_config.py", 768)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = {k.replace("lidar_encoder.", ""): v for k, v in ckpt["state_dict"].items()
             if k.startswith("lidar_encoder.")}
    missing, unexpected = enc.load_state_dict(state, strict=False)
    print(f"  {os.path.basename(ckpt_path)}: loaded {len(state)} keys, "
          f"missing={len(missing)}, unexpected={len(unexpected)}", flush=True)
    return enc.cuda().eval()


print("== CLIP ViT-L/14 (fp32, 对齐参照) ==", flush=True)
clip_model, pre = clip.load("ViT-L/14", jit=False)
clip_model = clip_model.float().cuda().eval()

print("== 加载两个编码器 ==", flush=True)
enc_new = load_encoder("ckpt_nuscenes/lidarclip_mm/epoch=3-step=19500.ckpt")
enc_old = load_encoder("lidarclip/checkpoint/vit_l_14.ckpt")

print("== 数据（与训练同分布，shuffle=False） ==", flush=True)
loader = build_loader("/root/autodl-tmp/Datasets/nuScenes", pre, batch_size=BATCH,
                      num_workers=8, split="trainval", dataset_name="nuscenes")


@torch.no_grad()
def evaluate(enc):
    mses, sims, norms = [], [], []
    it = iter(loader)
    for i in range(N_BATCHES):
        img, pcs = next(it)
        img = img.cuda()
        pcs = [p.cuda() for p in pcs]
        img_f = clip_model.encode_image(img).float()
        lid_f, _ = enc(pcs)
        lid_f = lid_f.float()
        mses.append(torch.mean((img_f - lid_f) ** 2).item())
        norms.append(lid_f.norm(dim=1).mean().item())
        a = torch.nn.functional.normalize(img_f, dim=1)
        b = torch.nn.functional.normalize(lid_f, dim=1)
        sa = (a @ a.T)[~torch.eye(len(a), dtype=torch.bool, device=a.device)]
        sb = (b @ b.T)[~torch.eye(len(b), dtype=torch.bool, device=b.device)]
        r = np.corrcoef(sa.cpu().numpy(), sb.cpu().numpy())[0, 1]
        sims.append(r)
    return (float(np.mean(mses)), float(np.mean(sims)), float(np.mean(norms)))


for name, enc in [("新 nuScenes(step19500)", enc_new), ("旧 ONCE 权重", enc_old)]:
    t0 = time.time()
    mse, sim, norm = evaluate(enc)
    print(f"{name}:  MSE={mse:.5f}  相似度结构相关={sim:.4f}  特征范数={norm:.3f}  ({time.time()-t0:.0f}s)", flush=True)
print("DONE", flush=True)
