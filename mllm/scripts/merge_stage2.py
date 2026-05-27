#!/usr/bin/env python3
"""Merge Stage1 projector + Stage2 LoRA into a full model checkpoint for Stage3 training."""
import os, sys, argparse, shutil
import torch

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(root_dir)

from vtimellm.model.builder import load_pretrained_model
from vtimellm.utils import disable_torch_init
from easydict import EasyDict as edict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_base", type=str, default="./base_model/vicuna-v1-5-7b")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str,
                        default="./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin")
    parser.add_argument("--stage2", type=str,
                        default="./checkpoints/vtimellm-vicuna-v1-5-7b-stage2")
    parser.add_argument("--output_dir", type=str,
                        default="./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-merged")
    return parser.parse_args()


def main():
    args = parse_args()
    disable_torch_init()

    print("Loading and merging model...")
    tokenizer, model, _ = load_pretrained_model(args, args.stage2, None)
    # model is already float16 after load_pretrained_model

    os.makedirs(args.output_dir, exist_ok=True)

    print("Saving merged model...")
    # Fix generation config validation error in newer transformers
    if hasattr(model, 'generation_config'):
        model.generation_config.do_sample = True
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Copy mm_projector for stage3 reference
    shutil.copy(args.pretrain_mm_mlp_adapter,
                os.path.join(args.output_dir, "mm_projector.bin"))

    print(f"Merged model saved to {args.output_dir}")
    print("Use --model_name_or_path {}/ for stage3 training".format(args.output_dir))


if __name__ == "__main__":
    main()
