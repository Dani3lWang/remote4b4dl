import torch
import argparse
import os, json
import numpy as np
import shutil

from tqdm import tqdm

import clip

from lidarclip.anno_loader import build_anno_loader
from lidarclip.loader import build_loader as build_dataonly_loader
from lidarclip.model.sst import LidarEncoderSST


def create_clean_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)  # Remove the existing directory and its contents
    os.makedirs(directory_path)  # Create a fresh directory


DEFAULT_DATA_PATHS = {
    "once": "/proj/nlp4adas/datasets/once",
    "nuscenes": "/root/autodl-tmp/Datasets/nuScenes/",
    "with_path": "/root/autodl-tmp/Datasets/nuScenes/"
}


def load_model(args):
    assert torch.cuda.is_available()

    clip_model, clip_preprocess = clip.load(args.clip_version)
    lidar_encoder = LidarEncoderSST(
        "lidarclip/model/sst_encoder_only_config.py", clip_model.visual.output_dim
    )

    # PyTorch 2.6+ defaults to weights_only=True for torch.load, but the
    # PyTorch Lightning checkpoint contains non-tensor objects (scheduler, etc.).
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    lidar_state = {
        k.replace("lidar_encoder.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("lidar_encoder.")
    }
    # strict=False: checkpoint may contain old bbox_head keys that our
    # encoder-only SSTEncoder does not have.
    lidar_encoder.load_state_dict(lidar_state, strict=False)
    print(f"Loaded {len(lidar_state)} lidar_encoder parameters from checkpoint")

    # Wrap in a thin wrapper that exposes .lidar_encoder
    class _EncoderWrapper:
        def __init__(self, encoder):
            self.lidar_encoder = encoder.cuda()

    model = _EncoderWrapper(lidar_encoder)
    return model, clip_preprocess


def main(args):
    model, clip_preprocess = load_model(args)
    build_loader = build_anno_loader if args.use_anno_loader else build_dataonly_loader
    loader = build_loader(
        args.data_path,
        clip_preprocess,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split=args.split,
        dataset_name=args.dataset_name,
    )
    
    with open(args.scene_json_path, "r") as file:
        token_data = json.load(file)
        
    with open(args.frame_json_path, "r") as file:
        frame_data = json.load(file)

#     lidar_path = f"{args.prefix}lidar.pt"

#     if os.path.exists(lidar_path):
#         print("Found existing files, skipping")
#         return

    lidar_dict = {}
    with torch.no_grad():
        #formulate lidar feature dictionary matching with special tokens
        for batch in tqdm(loader):
            _, point_clouds, pc_path = batch[:3]
            point_clouds = [pc.to("cuda") for pc in point_clouds]
            lidar_features, _ = model.lidar_encoder(point_clouds)
            for lidar_feat, lidar_path in zip(lidar_features, pc_path):
                lidar_dict[lidar_path] = lidar_feat.unsqueeze(0)

#         IF extracting for stage 1
        create_clean_directory(args.stage1_save_dir)
        for d in tqdm(frame_data):
            for f in d["frames"]:
                frame_id = f["frame_id"]
                frame_path = f["PATH_LIDAR_TOP"]
                full_frame_path = os.path.join(args.data_path, frame_path)
                
                frame_feature = lidar_dict[full_frame_path]
                
                frame_feature_save_path = os.path.join(args.stage1_save_dir, frame_id+".npy")
                
                numpy_feature = frame_feature.cpu().detach().numpy()
                np.save(frame_feature_save_path, numpy_feature)
                
#         IF extracting for stage 2 or 3
        create_clean_directory(args.stage2_save_dir)
        for d in tqdm(token_data):
            scene_id = d["scene_id"]
            # scene_token = d["scene_token"]
            scene_length = d["num_frames"]
            # frames = d["frames"]
            frames = d["paths"]["PATH_LIDAR_TOP"]
            feature_list = []
            for i in range(scene_length):
                frame_key = "PATH_{:03d}".format(i)
                frame_path = frames[frame_key]
                full_frame_path = os.path.join(args.data_path, frame_path)

                frame_feature = lidar_dict[full_frame_path]
                feature_list.append(frame_feature)

            concat_feature_path = os.path.join(args.stage2_save_dir, scene_id+".npy")
            concat_lidar_feature = torch.cat(feature_list, dim=0)

            numpy_feature = concat_lidar_feature.cpu().detach().numpy()
            np.save(concat_feature_path, numpy_feature)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, default="./ckpt/trained/.ckpt file", help="Full path to the checkpoint file"
    )
    parser.add_argument("--clip-version", type=str, default="ViT-L/14")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--split", type=str, default="trainval")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use-anno-loader", action="store_true")
    parser.add_argument("--dataset-name", type=str, default="with_path", choices=["once", "nuscenes", "with_path"])
    parser.add_argument("--scene-json-path", type=str, default="./annotations/scene_metadata.json")
    parser.add_argument("--frame-json-path", type=str, default="./annotations/sequence_metadata.json")

    parser.add_argument("--stage1-save-dir", type=str, default="lidarclip/extracted_features/stage1_features/")
    parser.add_argument("--stage2-save-dir", type=str, default="lidarclip/extracted_features/stage2_features/")
    
    args = parser.parse_args()
    
    if not args.data_path:
        args.data_path = DEFAULT_DATA_PATHS[args.dataset_name]
    
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
