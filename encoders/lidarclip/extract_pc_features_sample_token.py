#!/usr/bin/env python3
"""
按官方 stage1 约定提取特征：每帧一个 .npy，文件名 = sample_token。

与 extract_pc_features.py 的区别:
    - 旧脚本按 sequence_metadata 的 frame_id 键控输出（{frame_id}.npy）
    - 本脚本按 nuScenes sample_token 键控输出（{sample_token}.npy），
      与 build_stage1_from_lidarllm.py 生成的 stage1_train.json (scene_id=sample_token)
      对应；dataset.py 用 scene_id 拼特征路径，二者必须一致

点云处理复用 loader (NuscenesImageLidarDataset_with_path):
    - 遍历 nuScenes trainval 全部关键帧
    - 点云变换到 CAM_BACK_RIGHT 相机系并裁剪（与旧特征同分布）
    - 仅保存属于 scene_metadata 训练 scene (split == 'train') 的帧

用法:
    conda run -n wqlc python extract_pc_features_sample_token.py \
        --checkpoint ./lidarclip/checkpoint/vit_l_14.ckpt \
        --scene_metadata ./annotations/scene_metadata.json \
        --sample_json /root/autodl-tmp/Datasets/nuScenes/v1.0-trainval/sample.json \
        --data_path /root/autodl-tmp/Datasets/nuScenes \
        --save_dir ./b4dl/stage1_features_sample
"""
import os
import json
import argparse
import shutil

import numpy as np
import torch
from tqdm import tqdm

import clip

from lidarclip.loader import build_loader
from lidarclip.model.sst import LidarEncoderSST


def create_clean_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path)


def load_model(args):
    assert torch.cuda.is_available()

    clip_model, clip_preprocess = clip.load(args.clip_version)
    lidar_encoder = LidarEncoderSST(
        "lidarclip/model/sst_encoder_only_config.py", clip_model.visual.output_dim
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    lidar_state = {
        k.replace("lidar_encoder.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("lidar_encoder.")
    }
    lidar_encoder.load_state_dict(lidar_state, strict=False)
    print(f"Loaded {len(lidar_state)} lidar_encoder parameters from checkpoint")

    class _EncoderWrapper:
        def __init__(self, encoder):
            self.lidar_encoder = encoder.cuda()

    model = _EncoderWrapper(lidar_encoder)
    return model, clip_preprocess


def main(args):
    model, clip_preprocess = load_model(args)
    loader = build_loader(
        args.data_path,
        clip_preprocess,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split="trainval",
        dataset_name="with_path",
    )

    # sample_data.json: filename(相对) → sample_token
    with open(os.path.join(args.data_path, "v1.0-trainval", "sample_data.json")) as f:
        sample_data = json.load(f)
    filename_to_sample = {
        x["filename"]: x["sample_token"]
        for x in sample_data if "LIDAR_TOP" in x.get("filename", "")
    }

    # sample.json: sample_token → scene_token
    with open(args.sample_json) as f:
        samples = json.load(f)
    sample_to_scene = {s["token"]: s["scene_token"] for s in samples}

    # scene_metadata: 训练 scene_token 集合
    with open(args.scene_metadata) as f:
        scenes = json.load(f)
    train_scene_tokens = {s["scene_token"] for s in scenes if s.get("split") == "train"}
    print(f"训练 scene: {len(train_scene_tokens)} 个")

    # 需要保存的 sample_token 集合（700 train scenes 的全部关键帧）
    keep_samples = {
        tok for tok, sc in sample_to_scene.items() if sc in train_scene_tokens
    }
    print(f"需提取的关键帧: {len(keep_samples)}")

    create_clean_directory(args.save_dir)
    data_root = os.path.join(args.data_path, "")

    saved, skipped = 0, 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):
            if args.max_batches and batch_idx >= args.max_batches:
                break
            _, point_clouds, pc_paths = batch[:3]
            point_clouds = [pc.to("cuda") for pc in point_clouds]
            lidar_features, _ = model.lidar_encoder(point_clouds)
            for lidar_feat, full_path in zip(lidar_features, pc_paths):
                # full_path = dataroot + filename → filename → sample_token
                filename = str(full_path)
                if filename.startswith(data_root):
                    filename = filename[len(data_root):]
                sample_token = filename_to_sample.get(filename)
                if sample_token is None or sample_token not in keep_samples:
                    skipped += 1
                    continue
                feat = lidar_feat.unsqueeze(0).cpu().numpy()
                np.save(os.path.join(args.save_dir, sample_token + ".npy"), feat)
                saved += 1

    print(f"完成: 保存 {saved} 帧, 跳过 {skipped} 帧 → {args.save_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="./lidarclip/checkpoint/vit_l_14.ckpt")
    parser.add_argument("--clip-version", type=str, default="ViT-L/14")
    parser.add_argument("--data-path", type=str,
                        default="/root/autodl-tmp/Datasets/nuScenes/")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--sample-json", type=str,
                        default="/root/autodl-tmp/Datasets/nuScenes/v1.0-trainval/sample.json")
    parser.add_argument("--max-batches", type=int, default=0,
                        help="冒烟测试: 只跑前 N 个 batch (0 = 全量)")
    parser.add_argument("--scene-metadata", type=str,
                        default="./annotations/scene_metadata.json")
    parser.add_argument("--save-dir", type=str,
                        default="./b4dl/stage1_features_sample")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
