#!/usr/bin/env python3
"""
将 LiDAR-LLM-Nu-Caption (Senqiao/LiDAR-LLM-Nu-Caption) 转换为 mllm Stage1 训练格式。

映射链路:
    LiDAR-LLM sample_token
      → nuScenes sample_data.json (LIDAR_TOP: token → sample_token)
      → sequence_metadata.json (TOKEN_LIDAR_TOP → frame_id)
      → stage1_features/{frame_id}.npy 特征文件

输出格式 (与 mllm/vtimellm/train/dataset.py 兼容):
    {"scene_id": frame_id, "scene_token": sample_token,
     "conversations": [{"from": "human", "value": "<video>\\n{question}"},
                       {"from": "gpt", "value": "{answer_lidar}"}]}

注意: dataset.py 用 source['scene_id'] 拼特征路径 {feat_folder}/{scene_id}.npy,
所以 scene_id 必须等于 stage1_features 的文件名 (frame_id)。

用法:
    python convert_lidarllm_to_stage1.py \
        --nuscenes_root /root/autodl-tmp/Datasets/nuScenes \
        --metadata ./b4dl_dataset/metadata/sequence_metadata.json \
        --feat_folder ../encoders/lidarclip/b4dl/stage1_features \
        --output_dir ./b4dl_dataset
"""
import os
import json
import argparse
from collections import defaultdict


def build_lidar_token_to_sample(sample_data_path: str) -> dict:
    """sample_data.json → {LIDAR_TOP token: sample_token}"""
    with open(sample_data_path) as f:
        sd = json.load(f)
    mapping = {}
    for x in sd:
        if 'LIDAR_TOP' in x.get('filename', ''):
            mapping[x['token']] = x['sample_token']
    return mapping


def build_token_to_frame_id(seq_metadata_path: str) -> dict:
    """sequence_metadata.json → {TOKEN_LIDAR_TOP: frame_id}"""
    with open(seq_metadata_path) as f:
        seqs = json.load(f)
    mapping = {}
    for seq in seqs:
        for frame in seq.get('frames', []):
            tok = frame.get('TOKEN_LIDAR_TOP')
            fid = frame.get('frame_id')
            if tok and fid:
                mapping[tok] = fid
    return mapping


def convert(llm_path: str, sample_to_frame: dict, feat_folder: str,
            available_feats: set) -> tuple:
    """转换单个 LiDAR-LLM 文件，返回 (输出列表, 统计)。"""
    with open(llm_path) as f:
        items = json.load(f)

    out, stats = [], {'total': len(items), 'mapped': 0, 'no_feature': 0, 'no_mapping': 0}
    for it in items:
        sample_token = it['sample_token']
        frame_id = sample_to_frame.get(sample_token)
        if frame_id is None:
            stats['no_mapping'] += 1
            continue
        if frame_id not in available_feats:
            stats['no_feature'] += 1
            continue

        question = it['question']
        answer = it.get('answer_lidar') or it.get('answer', '')
        out.append({
            "scene_id": frame_id,
            "scene_token": sample_token,
            "conversations": [
                {"from": "human", "value": f"<video>\n{question}"},
                {"from": "gpt", "value": answer},
            ],
        })
        stats['mapped'] += 1
    return out, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nuscenes_root", required=True,
                        help="nuScenes 根目录 (含 v1.0-trainval/sample_data.json)")
    parser.add_argument("--metadata", required=True,
                        help="sequence_metadata.json 路径")
    parser.add_argument("--feat_folder", required=True,
                        help="stage1_features 目录")
    parser.add_argument("--output_dir", default="./b4dl_dataset")
    parser.add_argument("--llm_train", default="./data/lidarllm_train.json")
    parser.add_argument("--llm_val", default="./data/lidarllm_val.json")
    args = parser.parse_args()

    # 构建映射
    print("构建映射...")
    lidar_tok_to_sample = build_lidar_token_to_sample(
        os.path.join(args.nuscenes_root, 'v1.0-trainval', 'sample_data.json'))
    tok_to_frame = build_token_to_frame_id(args.metadata)

    # sample_token → frame_id
    sample_to_frame = {}
    for lidar_tok, sample_tok in lidar_tok_to_sample.items():
        if lidar_tok in tok_to_frame:
            sample_to_frame[sample_tok] = tok_to_frame[lidar_tok]
    print(f"  LIDAR_TOP sample_data: {len(lidar_tok_to_sample)} 帧")
    print(f"  sequence_metadata 帧: {len(tok_to_frame)} 帧")
    print(f"  可映射 sample: {len(sample_to_frame)}")

    # 特征文件全集
    available_feats = {f[:-4] for f in os.listdir(args.feat_folder)
                       if f.endswith('.npy')}
    print(f"  特征文件: {len(available_feats)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 转换 train / val
    for llm_path, out_name in [(args.llm_train, 'stage1_train.json'),
                               (args.llm_val, 'stage1_val.json')]:
        out, stats = convert(llm_path, sample_to_frame, args.feat_folder,
                             available_feats)
        out_path = os.path.join(args.output_dir, out_name)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n{out_name}: 写入 {len(out)} 条 → {out_path}")
        print(f"  统计: {stats}")


if __name__ == "__main__":
    main()
