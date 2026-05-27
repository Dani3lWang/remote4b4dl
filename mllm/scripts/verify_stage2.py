#!/usr/bin/env python3
"""Quick verification script for Stage2 model on B4DL data."""
import os
import sys
import json
import argparse
import random

import torch
import numpy as np
from tqdm import tqdm

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(root_dir)

from vtimellm.model.builder import load_pretrained_model
from vtimellm.utils import disable_torch_init
from vtimellm.inference import inference


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_base", type=str, default="./base_model/vicuna-v1-5-7b")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str,
                        default="./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin")
    parser.add_argument("--stage2", type=str,
                        default="./checkpoints/vtimellm-vicuna-v1-5-7b-stage2")
    parser.add_argument("--data_path", type=str,
                        default="./b4dl_dataset/stage2_conversations.json")
    parser.add_argument("--feat_folder", type=str,
                        default="../encoders/lidarclip/b4dl/stage2_features")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--output_path", type=str, default="./eval_results/verify_stage2.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    disable_torch_init()
    print("Loading model...")
    tokenizer, model, _ = load_pretrained_model(args, args.stage2, None)
    model = model.cuda()
    model.to(torch.float16)
    tokenizer.pad_token = tokenizer.unk_token
    print("Model loaded.")

    data = json.load(open(args.data_path, "r"))

    binary_samples = [d for d in data
                      if d['conversations'][1]['value'].strip().lower() in ('yes.', 'no.')]
    other_samples = [d for d in data
                     if d['conversations'][1]['value'].strip().lower() not in ('yes.', 'no.')]

    selected = random.sample(binary_samples, min(10, len(binary_samples)))
    selected += random.sample(other_samples, min(args.num_samples - len(selected), len(other_samples)))
    random.shuffle(selected)

    results = []
    correct = 0

    for item in tqdm(selected, desc="Inference"):
        scene_id = item['scene_id']
        query = item['conversations'][0]['value']
        gt = item['conversations'][1]['value']

        feat_path = os.path.join(args.feat_folder, f"{scene_id}.npy")
        if not os.path.exists(feat_path):
            print(f"[WARN] Missing feature: {scene_id}")
            continue
        features = torch.from_numpy(np.load(feat_path)).to(torch.float16)

        pred = inference(model, features, query, tokenizer)

        is_correct = pred.strip().lower() == gt.strip().lower()
        if is_correct:
            correct += 1

        results.append({
            "scene_id": scene_id,
            "question": query,
            "ground_truth": gt,
            "prediction": pred,
            "correct": is_correct
        })

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"Results saved to {args.output_path}")
    print(f"Accuracy: {correct}/{len(results)} = {correct/len(results)*100:.1f}%")
    print(f"{'='*60}")
    for r in results:
        mark = "PASS" if r['correct'] else "FAIL"
        print(f"\n[{mark}] scene={r['scene_id']}")
        print(f"  Q: {r['question']}")
        print(f"  GT: {r['ground_truth']}")
        print(f"  PR: {r['prediction']}")


if __name__ == "__main__":
    main()
