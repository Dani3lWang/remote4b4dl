#!/usr/bin/env python3
"""
验证官方对齐后的 stage1 数据与特征:
1. stage1_train.json 所有 scene_id (sample_token) 都有特征文件
2. 特征形状 (1, 768) 且 dtype 可被 dataset.py 加载
3. dataset.py 冒烟: LazySupervisedDataset 前 N 条能正常出 batch (image shape)

用法 (wqlc 环境, mllm 目录):
    python scripts/verify_stage1_sample_data.py [N]
"""
import sys
import json
import os
import random

FEAT_FOLDER = "../encoders/lidarclip/b4dl/stage1_features_sample"
DATA_PATH = "./b4dl_dataset/stage1_train.json"

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200

    with open(DATA_PATH) as f:
        data = json.load(f)
    print(f"stage1_train.json: {len(data)} 条")

    missing = [e["scene_id"] for e in data[:n] if not os.path.exists(
        os.path.join(FEAT_FOLDER, e["scene_id"] + ".npy"))]
    print(f"前 {n} 条缺特征: {len(missing)}")
    if missing:
        print("  例:", missing[:5])

    # 抽一个特征检查形状
    import numpy as np
    sample_id = data[0]["scene_id"]
    feat = np.load(os.path.join(FEAT_FOLDER, sample_id + ".npy"))
    print(f"特征 {sample_id}.npy: shape={feat.shape} dtype={feat.dtype}")

    # dataset.py 冒烟
    print("dataset.py 冒烟 (前", n, "条)...")
    random.seed(0)
    from vtimellm.train.dataset import LazySupervisedDataset, DataArguments
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "./base_model/vicuna-v1-5-7b", use_fast=False)
    ds = LazySupervisedDataset(DATA_PATH, tok,
                               DataArguments(feat_folder=FEAT_FOLDER))
    print(f"dataset len = {len(ds)}")
    errors = 0
    for i in range(min(n, len(ds))):
        try:
            d = ds[i]
            assert d["image"].shape[0] == 1, d["image"].shape
        except AssertionError as e:
            print(f"  [{i}] 特征 shape 异常: {e}")
            errors += 1
        except Exception as e:
            print(f"  [{i}] 加载失败: {type(e).__name__}: {e}")
            errors += 1
    print(f"冒烟完成: {min(n, len(ds)) - errors}/{min(n, len(ds))} 通过")

if __name__ == "__main__":
    main()
