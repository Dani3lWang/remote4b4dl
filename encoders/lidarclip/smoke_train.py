"""冒烟：用真实 PL Trainer 验证 nuScenes 编码器训练（limit 3 batch × 1 epoch）"""
import os
import sys
import time
import importlib.util

import torch

sys.path.insert(0, "/root/autodl-tmp/wql/mmb4dl/encoders/lidarclip")
os.chdir("/root/autodl-tmp/wql/mmb4dl/encoders/lidarclip")

spec = importlib.util.spec_from_file_location(
    "lc_train", "/root/autodl-tmp/wql/mmb4dl/encoders/lidarclip/train.py"
)
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)

import clip
from lidarclip.loader import build_loader

DATA_DIR = "/root/autodl-tmp/Datasets/nuScenes"
BATCH = 8

print("== loading CLIP ViT-L/14 ==", flush=True)
clip_model, clip_preprocess = clip.load("ViT-L/14", jit=False)
clip.model.convert_weights(clip_model)  # 与 train.py 中修复一致
clip_model.eval()
clip_embed_dim = clip_model.visual.output_dim

print("== building SST encoder ==", flush=True)
lidar_encoder = train_mod.LidarEncoderSST(
    "lidarclip/model/sst_encoder_only_config.py", clip_embed_dim
)

print("== building nuScenes loader (700-scene filter) ==", flush=True)
t0 = time.time()
train_loader = build_loader(
    DATA_DIR,
    clip_preprocess,
    batch_size=BATCH,
    num_workers=4,
    split="trainval",
    dataset_name="nuscenes",
)
print(f"loader ready in {time.time()-t0:.0f}s, batches: {len(train_loader)}", flush=True)

model = train_mod.LidarClip(
    lidar_encoder, clip_model, BATCH, len(train_loader) / 1, "mse"
)

import pytorch_lightning as pl

trainer = pl.Trainer(
    precision=16,
    accelerator="gpu",
    devices=1,
    limit_train_batches=3,
    max_epochs=1,
    logger=False,
    enable_checkpointing=False,
    enable_progress_bar=True,
)
print("== running 3 training steps via Trainer ==", flush=True)
t0 = time.time()
trainer.fit(model=model, train_dataloaders=train_loader)
free, total = torch.cuda.mem_get_info()
print(
    f"SMOKE OK  loss={trainer.callback_metrics.get('train_loss', float('nan')):.4f}  "
    f"vram free {free/1e9:.1f}/{total/1e9:.1f}GB  elapsed {time.time()-t0:.0f}s",
    flush=True,
)
