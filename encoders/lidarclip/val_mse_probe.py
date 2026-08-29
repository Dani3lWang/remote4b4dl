"""val_mse_probe.py — 编码器 val MSE 探针（2026-08-29 退火链用）

在 val/test 场景（编码器从未训练过的 150 个场景）的确定性子集上，仅 forward 计算
LiDAR→CLIP 对齐 MSE，作为收敛/选型判据。训练 loss 来自 train 集，既不能判断
泛化也无法发现过拟合（审查 2026-08-29：训练侧无任何验证信号）。

模式约定（重要）：lidar_encoder 保持 **train 模式**（BN 用 batch 统计 + 训练版
token 随机丢弃）——与 extract_pc_features.py 的提取约定一致，探针分数才与下游
特征行为可比；固定 --seed + shuffle=False 保证跨 ckpt 可比（BN/丢 token 的随机性
在相同 seed 与相同 batch 顺序下可复现）。

用法:
  python val_mse_probe.py --checkpoint ckpt_nuscenes/lidarclip_mm/last.ckpt --tag baseline
  python val_mse_probe.py --checkpoint <ckpt> --max-batches 2 --batch-size 8   # 冒烟
结果追加写入 logs/val_mse_probe.csv（列: 时间,tag,ckpt,n_batches,n_pairs,mean,std）
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F
import clip
from pytorch_lightning import seed_everything

from lidarclip.loader import build_loader
from lidarclip.model.sst import LidarEncoderSST


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default="/root/autodl-tmp/Datasets/nuScenes")
    p.add_argument("--clip-model", default="ViT-L/14")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-scenes", type=int, default=15,
                   help="val 场景子集大小（sorted 前 N 个；0=全部 150 个）")
    p.add_argument("--max-batches", type=int, default=0,
                   help="0=跑满子集；冒烟测试用小值")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--output", default="logs/val_mse_probe.csv")
    args = p.parse_args()

    seed_everything(args.seed, workers=True)
    clip_model, clip_preprocess = clip.load(args.clip_model, jit=False)
    clip.model.convert_weights(clip_model)  # 与 train.py 相同：本机 clip.load 不转 fp16
    clip_model.eval().cuda()

    lidar_encoder = LidarEncoderSST(
        "lidarclip/model/sst_encoder_only_config.py", clip_model.visual.output_dim)
    # PyTorch 2.6+ 需要 weights_only=False（PL ckpt 含非张量对象），与 extract 脚本一致
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    lidar_state = {k.replace("lidar_encoder.", ""): v
                   for k, v in ckpt["state_dict"].items()
                   if k.startswith("lidar_encoder.")}
    ret = lidar_encoder.load_state_dict(lidar_state, strict=False)
    print(f"Loaded {len(lidar_state)} params; missing={len(ret.missing_keys)} "
          f"unexpected={len(ret.unexpected_keys)}")
    assert len(lidar_state) > 0, "checkpoint 里没有 lidar_encoder.* 权重"
    assert not ret.missing_keys, f"模型参数缺失: {ret.missing_keys[:5]}"
    if ret.unexpected_keys:
        # 历史格式兼容（旧 ckpt 的 bbox_head 键），打印留痕不阻断
        print(f"WARN unexpected keys ({len(ret.unexpected_keys)}): "
              f"{ret.unexpected_keys[:5]}")
    # 保持 train 模式：与训练/特征提取约定一致（见模块 docstring）
    lidar_encoder = lidar_encoder.cuda().train()

    loader = build_loader(
        args.data_dir, clip_preprocess, batch_size=args.batch_size,
        num_workers=args.workers, split="trainval", shuffle=False,
        dataset_name="nuscenes", val_mode=True, val_max_scenes=args.max_scenes,
    )
    print(f"val probe: {len(loader.dataset)} pairs, {len(loader)} batches "
          f"(scenes<={args.max_scenes})")

    losses = []
    t0 = time.time()
    for i, (image, point_cloud) in enumerate(loader):
        if args.max_batches and i >= args.max_batches:
            break
        image = image.cuda(non_blocking=True)
        # PL 训练时自动搬 batch（含变长点云列表），探针需手动搬
        point_cloud = [pc.cuda(non_blocking=True) for pc in point_cloud]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            img_feat = clip_model.encode_image(image)
            lidar_feat, _ = lidar_encoder(point_cloud)
            loss = F.mse_loss(img_feat.float(), lidar_feat.float())
        losses.append(float(loss))
        if (i + 1) % 20 == 0:
            print(f"  batch {i+1}: running mean {sum(losses)/len(losses):.6f}", flush=True)

    n = len(losses)
    mean = sum(losses) / n
    std = (sum((x - mean) ** 2 for x in losses) / n) ** 0.5
    print(f"== val MSE: mean={mean:.6f} std={std:.6f} over {n} batches "
          f"({time.time()-t0:.0f}s) ==", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{args.tag},{args.checkpoint},"
                f"{n},{n * args.batch_size},{mean:.6f},{std:.6f}\n")


if __name__ == "__main__":
    main()
