"""验证向量化 dynamic scatter：数值等价 + 梯度流 + 速度"""
import os
import sys
import time
import importlib.util

import torch

sys.path.insert(0, "/root/autodl-tmp/wql/mmb4dl/encoders/lidarclip")
os.chdir("/root/autodl-tmp/wql/mmb4dl/encoders/lidarclip")

spec = importlib.util.spec_from_file_location("lc_train", "train.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
compat = sys.modules["lidarclip.model._mmdet3d_compat"]

# ── 旧循环实现（照抄改前版本，作对照） ──
def old_forward(feats, coors, reduce_type="max"):
    coors_flat = coors[:, 0].long()
    for d in range(1, coors.size(1)):
        coors_flat = coors_flat * (coors[:, d].max() + 1) + coors[:, d].long()
    unique_coors, inverse_indices, counts = torch.unique(coors_flat, return_inverse=True, return_counts=True)
    n_voxels, device = unique_coors.size(0), feats.device
    voxel_feats = feats.new_zeros((n_voxels, feats.shape[1]))
    voxel_coors_out = torch.zeros((n_voxels, coors.size(1)), dtype=coors.dtype, device=device)
    for v in range(n_voxels):
        mask = inverse_indices == v
        pts = feats[mask]
        if reduce_type == "max":
            voxel_feats[v] = pts.max(dim=0)[0]
        elif reduce_type == "sum":
            voxel_feats[v] = pts.sum(dim=0)
        elif reduce_type == "mean":
            voxel_feats[v] = pts.mean(dim=0)
        first_idx = torch.where(mask)[0][0]
        voxel_coors_out[v] = coors[first_idx]
    return voxel_feats, voxel_coors_out, inverse_indices, counts.int()

torch.manual_seed(0)
dev = "cuda"
for trial in range(3):
    N, C = torch.randint(3000, 8000, (1,)).item(), 64
    feats = torch.randn(N, C, device=dev)
    coors = torch.stack([
        torch.randint(0, 80, (N,), device=dev),
        torch.randint(0, 80, (N,), device=dev),
        torch.randint(0, 1, (N,), device=dev)], dim=1).int()
    for rt in ("max", "sum", "mean"):
        t0 = time.time()
        of, oc, om, ocnt = old_forward(feats, coors, rt)
        t_old = time.time() - t0
        t0 = time.time()
        nf, nc, nm, ncnt = compat._dynamic_point_to_voxel_forward_py(feats, coors, rt)
        t_new = time.time() - t0
        d_feat = (of - nf).abs().max().item()
        same_c = torch.equal(oc, nc)
        same_m = torch.equal(om, nm)
        same_cnt = torch.equal(ocnt, ncnt)
        print(f"trial{trial} N={N} {rt}: maxdiff={d_feat:.2e} coors={same_c} map={same_m} cnt={same_cnt}  old {t_old:.2f}s new {t_new*1000:.1f}ms", flush=True)

# ── 梯度流测试：真实 SST 编码器 ──
print("\n== SST 梯度流 + 速度 ==", flush=True)
spec2 = importlib.util.spec_from_file_location("lc_train", "train.py")
m = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(m)
import clip
from lidarclip.loader import build_loader

clip_model, pre = clip.load("ViT-L/14", jit=False)
clip.model.convert_weights(clip_model); clip_model.eval()
enc = m.LidarEncoderSST("lidarclip/model/sst_encoder_only_config.py", 768).cuda().train()
loader = build_loader("/root/autodl-tmp/Datasets/nuScenes", pre, batch_size=8, num_workers=8, split="trainval", dataset_name="nuscenes")
it = iter(loader)
opt = torch.optim.Adam(enc.parameters(), lr=1e-5)
for trial in range(3):
    t0 = time.time()
    image, pcs = next(it)
    image = image.cuda(); pcs = [p.cuda() for p in pcs]
    feats, _ = enc(pcs)
    loss = feats.mean()
    loss.backward()
    torch.cuda.synchronize()
    t1 = time.time()
    # point 级层梯度（DynamicVFE 的 vfe_layers，位于 scatter 之前）
    grads = []
    for name, p in enc.named_parameters():
        if "vfe_layers" in name and p.grad is not None:
            grads.append((name, p.grad.abs().mean().item()))
    nz = sum(1 for _, g in grads if g > 0)
    sample_g = f"{grads[0][1]:.2e}" if grads else "N/A"
    print(f"trial {trial}: fwd+bwd {t1-t0:.2f}s  vfe_layers(scatter前) 参数有梯度 {nz}/{len(grads)}  sample grad {sample_g}", flush=True)
    opt.step(); opt.zero_grad()
print("VALIDATION DONE", flush=True)
