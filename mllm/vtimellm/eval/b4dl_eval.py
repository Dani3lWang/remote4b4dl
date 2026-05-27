#!/usr/bin/env python3
"""B4DL evaluation script for Stage2 (QA) and Stage3 (captioning)."""
import os
import sys
import json
import argparse

import torch
import numpy as np
from tqdm import tqdm

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.append(root_dir)

from vtimellm.model.builder import load_pretrained_model
from vtimellm.utils import disable_torch_init
from vtimellm.inference import inference


def classify_answer(gt: str) -> str:
    gt_lower = gt.strip().lower().rstrip('.')
    if gt_lower in ('yes', 'no'):
        return 'binary'
    import re
    if re.match(r'^from frame \d+ to frame \d+$', gt_lower):
        return 'frame_range'
    return 'categorical'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_base", type=str, required=True)
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, required=True)
    parser.add_argument("--stage2", type=str, default=None)
    parser.add_argument("--stage3", type=str, default=None)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--feat_folder", type=str, required=True)
    parser.add_argument("--log_path", type=str, default="eval_log.jsonl")
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    disable_torch_init()

    print("Loading model...")
    tokenizer, model, _ = load_pretrained_model(args, args.stage2, args.stage3)
    model = model.cuda()
    model.to(torch.float16)
    tokenizer.pad_token = tokenizer.unk_token
    print("Model loaded.")

    data = json.load(open(args.data_path, "r"))
    if args.max_samples:
        data = data[:args.max_samples]

    os.makedirs(os.path.dirname(args.log_path) or ".", exist_ok=True)

    with open(args.log_path, "w") as log_f:
        for item in tqdm(data, desc="Evaluating"):
            scene_id = item['scene_id']
            query = item['conversations'][0]['value']
            gt = item['conversations'][1]['value']

            feat_path = os.path.join(args.feat_folder, f"{scene_id}.npy")
            if not os.path.exists(feat_path):
                print(f"[WARN] Missing feature: {scene_id}")
                continue
            features = torch.from_numpy(np.load(feat_path)).to(torch.float16)

            pred = inference(model, features, query, tokenizer)

            log_entry = {
                "scene_id": scene_id,
                "query": query,
                "gt": gt,
                "pred": pred,
                "answer_type": classify_answer(gt)
            }
            log_f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    print(f"Done. Results saved to {args.log_path}")


if __name__ == "__main__":
    main()
