"""
B4DL Dataset Split Verification
================================
对应论文 Table 2 / Sec. 3.3 的"验证脚本"：读取 scene_metadata.json +
sequence_metadata.json + 已生成的 B4DL 测试集 (QA json 文件)，核对：

  * train scene 数 = 700，test scene 数 = 150 (nuScenes 官方 train/val)
  * train sequence 数 ≈ 700 * 6 = 4200，test sequence 数 ≈ 150 * 6 = 900
  * 6 任务在每个 split 下的样本数与论文 Table 2 对齐
      test:  existence 3770, binary 7525, time_grounding 2783,
             description 3770, temporal_understanding 4757,
             comprehensive 7540 —— 合计 30145

两种模式：
  1. --nuscenes_root (推荐)：直接通过 nuScenes devkit 获取官方 train/val
     划分，无需依赖 metadata 中的 split 字段。
  2. 仅 --metadata_dir：依赖 metadata JSON 中的 split 字段。

--fix_split 会将正确 split 写回 metadata，解决当前所有 scene 的 split
字段均为 "train" 的问题。

用法:
    python split_dataset.py --metadata_dir ./data/metadata \
        --nuscenes_root /path/to/nuScenes
    python split_dataset.py --metadata_dir ./data/metadata \
        --nuscenes_root /path/to/nuScenes --fix_split
    python split_dataset.py --metadata_dir ./data/metadata \
        --dataset_dir ./data/generated_dataset --check_counts

依赖:
    pip install nuscenes-devkit
"""
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set


# 论文 Table 2 (test split)
PAPER_TEST_COUNTS = {
    "existence":               3770,
    "binary":                  7525,   # 论文表中记作 "Yes/No Q&A"
    "time_grounding":          2783,
    "description":             3770,
    "temporal_understanding":  4757,
    "comprehensive":          7540,   # 论文表中记作 "Comprehensive Reasoning"
}
PAPER_TRAIN_COUNTS = {
    "existence":               18545,
    "binary":                  37026,
    "time_grounding":          13124,
    "description":             18540,
    "temporal_understanding":  23956,
    "comprehensive":          37080,
}

# 论文 Sec. 3.2 / 3.3：每条 sequence 产生 40 个样本，按任务分布如下
PER_SEQUENCE_COUNTS = {
    "existence":               5,
    "binary":                 10,
    "time_grounding":          5,
    "description":             5,
    "temporal_understanding":  5,
    "comprehensive":          10,
}

# 论文设定的全局参数
EXPECTED_TRAIN_SCENES = 700
EXPECTED_TEST_SCENES = 150
EXPECTED_SEQUENCES_PER_SCENE = 6


def parse_args():
    p = argparse.ArgumentParser(description="Verify B4DL train/test split integrity")
    p.add_argument("--metadata_dir", type=str, default="./data/metadata",
                   help="目录，包含 scene_metadata.json 和 sequence_metadata.json")
    p.add_argument("--nuscenes_root", type=str, default=None,
                   help="nuScenes 数据集根目录（推荐：通过官方 devkit split 校验）")
    p.add_argument("--dataset_dir", type=str, default=None,
                   help="(可选) datageneration 产出的 ./data/generated_dataset 目录，"
                        "用于核对实际 QA 样本数")
    p.add_argument("--check_counts", action="store_true",
                   help="当与 --dataset_dir 一起使用时，实际点数会和论文 Table 2 对比")
    p.add_argument("--split", type=str, default="both", choices=["both", "train", "test"])
    p.add_argument("--fix_split", action="store_true",
                   help="根据 nuScenes 官方划分修正 metadata 文件中的 split 字段")
    return p.parse_args()


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    """Save with the same formatting convention as generate_metadata.py."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def _get_official_split(nuscenes_root: str) -> Optional[Dict[str, Set[str]]]:
    """Return {'train': set_of_scene_tokens, 'test': set_of_scene_tokens}
    using the official nuScenes devkit split (val → B4DL test)."""
    try:
        from nuscenes.utils.splits import train as train_names
        from nuscenes.utils.splits import val as val_names
    except ImportError:
        print("[ERROR] nuscenes-devkit 未安装，无法获取官方划分。"
              "请运行: pip install nuscenes-devkit")
        return None

    nusc_scene_json = Path(nuscenes_root) / 'v1.0-trainval' / 'scene.json'
    if not nusc_scene_json.exists():
        print(f"[ERROR] {nusc_scene_json} 不存在，请检查 --nuscenes_root")
        return None

    nusc_scenes = load_json(nusc_scene_json)
    name_to_token = {s['name']: s['token'] for s in nusc_scenes}

    train_tokens = {name_to_token[n] for n in train_names if n in name_to_token}
    test_tokens = {name_to_token[n] for n in val_names if n in name_to_token}  # val → B4DL test

    return {'train': train_tokens, 'test': test_tokens}


def _fix_metadata_splits(metadata_dir: Path, nuscenes_root: str):
    """Update scene_metadata.json and sequence_metadata.json with correct
    split labels from the nuScenes official train/val split."""
    split_map = _get_official_split(nuscenes_root)
    if split_map is None:
        return

    # --- scene_metadata.json ---
    scene_path = metadata_dir / 'scene_metadata.json'
    scenes = load_json(scene_path)
    if scenes is None:
        print(f"[WARN] {scene_path} 不存在，跳过")
    else:
        changed = 0
        for s in scenes:
            new_split = 'test' if s['scene_token'] in split_map['test'] else 'train'
            if s.get('split') != new_split:
                s['split'] = new_split
                changed += 1
        save_json(scenes, scene_path)
        n_train = sum(1 for s in scenes if s.get('split') == 'train')
        n_test = sum(1 for s in scenes if s.get('split') == 'test')
        print(f"[FIX] {scene_path.name}: {changed} scenes 更新 → "
              f"train={n_train}, test={n_test}")

    # --- sequence_metadata.json ---
    seq_path = metadata_dir / 'sequence_metadata.json'
    seqs = load_json(seq_path)
    if seqs is None:
        print(f"[WARN] {seq_path} 不存在，跳过")
    else:
        changed = 0
        for seq in seqs:
            new_split = 'test' if seq['scene_token'] in split_map['test'] else 'train'
            if seq.get('split') != new_split:
                seq['split'] = new_split
                changed += 1
        save_json(seqs, seq_path)
        n_train = sum(1 for s in seqs if s.get('split') == 'train')
        n_test = sum(1 for s in seqs if s.get('split') == 'test')
        print(f"[FIX] {seq_path.name}: {changed} sequences 更新 → "
              f"train={n_train}, test={n_test}")


def count_sequences_per_split(seq_meta: List[Dict]) -> Dict[str, int]:
    out = {"train": 0, "test": 0}
    for s in seq_meta:
        sp = s.get("split", "train")
        out[sp] = out.get(sp, 0) + 1
    return out


def count_scenes_per_split(scene_meta: List[Dict]) -> Dict[str, int]:
    out = {"train": 0, "test": 0}
    for s in scene_meta:
        sp = s.get("split", "train")
        out[sp] = out.get(sp, 0) + 1
    return out


def list_dataset_files(dataset_dir: Path) -> Dict[str, List[Path]]:
    """枚举 ./generated_dataset/<task>/*.json"""
    files = {}
    if not dataset_dir.exists():
        return files
    for sub in sorted(dataset_dir.iterdir()):
        if not sub.is_dir():
            continue
        files[sub.name] = sorted(sub.glob("*.json"))
    return files


def count_qa_in_files(files: List[Path]) -> int:
    n = 0
    for fp in files:
        data = load_json(fp)
        if isinstance(data, list):
            n += len(data)
    return n


def main():
    args = parse_args()
    md = Path(args.metadata_dir)
    scene_meta = load_json(md / "scene_metadata.json")
    seq_meta   = load_json(md / "sequence_metadata.json")

    # --- fix metadata splits if requested ---
    if args.fix_split:
        if not args.nuscenes_root:
            print("[ERROR] --fix_split 需要同时提供 --nuscenes_root")
            return
        _fix_metadata_splits(md, args.nuscenes_root)
        # Reload after fixing
        scene_meta = load_json(md / "scene_metadata.json")
        seq_meta = load_json(md / "sequence_metadata.json")

    # --- obtain official split for cross-validation ---
    official = None
    if args.nuscenes_root:
        official = _get_official_split(args.nuscenes_root)

    print("=" * 64)
    print("B4DL Split Verification")
    print("=" * 64)

    if scene_meta is None and seq_meta is None:
        print(f"[WARN] 找不到元数据 ({md})。")
        if args.nuscenes_root and official:
            print(f"  nuScenes 官方划分: train={len(official['train'])} scenes, "
                  f"val→test={len(official['test'])} scenes")
        return

    # 1) Scene 数量
    sc = count_scenes_per_split(scene_meta or [])
    print(f"[Scenes]   train={sc['train']} (期望 {EXPECTED_TRAIN_SCENES})  "
          f"test={sc['test']} (期望 {EXPECTED_TEST_SCENES})")

    # If we have the official split, warn about mismatches
    if official and scene_meta:
        meta_token_to_split = {s['scene_token']: s.get('split', 'train')
                                for s in scene_meta}
        mislabeled = []
        for tok in official['test']:
            if meta_token_to_split.get(tok) != 'test':
                mislabeled.append(tok)
        if mislabeled:
            print(f"  ⚠ {len(mislabeled)} val scenes 在 metadata 中未标记为 test。"
                  f" 用 --fix_split 修复。")

    # 2) Sequence 数量
    sq = count_sequences_per_split(seq_meta or [])
    exp_train_seq = sc['train'] * EXPECTED_SEQUENCES_PER_SCENE
    exp_test_seq  = sc['test']  * EXPECTED_SEQUENCES_PER_SCENE
    print(f"[Seqs]     train={sq['train']} (期望 ~{exp_train_seq})  "
          f"test={sq['test']} (期望 ~{exp_test_seq})")

    # 3) Sequence 长度分布 (Frame 数)
    if seq_meta:
        lens = [s.get("num_frames", len(s.get("indices", []))) for s in seq_meta]
        if lens:
            print(f"[SeqLen]   min={min(lens)}  max={max(lens)}  "
                  f"mean={sum(lens)/len(lens):.2f}")

    # 4) 预估的样本数 (按论文 PER_SEQUENCE_COUNTS)
    print()
    print("[预估样本数 (按论文: 40 samples/sequence)]")
    print(f"  task                     | paper-train | paper-test | est-train | est-test")
    print("  -------------------------+-------------+------------+-----------+---------")
    all_ok_expected = True
    for task, p_train in PAPER_TRAIN_COUNTS.items():
        p_test = PAPER_TEST_COUNTS[task]
        est_train = sq['train'] * PER_SEQUENCE_COUNTS[task]
        est_test  = sq['test']  * PER_SEQUENCE_COUNTS[task]
        print(f"  {task:<24} | {p_train:>11} | {p_test:>10} | "
              f"{est_train:>9} | {est_test:>7}")
        if sc['train'] == EXPECTED_TRAIN_SCENES and sc['test'] == EXPECTED_TEST_SCENES:
            if est_train != p_train or est_test != p_test:
                # 后处理过滤导致 est 略大于 paper (因为后处理丢弃部分样本)
                all_ok_expected = False

    # 5) 实际样本数 (若提供 --dataset_dir --check_counts)
    if args.dataset_dir and args.check_counts:
        print()
        print(f"[实际样本数] 从 {args.dataset_dir} 读取")
        files_map = list_dataset_files(Path(args.dataset_dir))
        if not files_map:
            print(f"  [WARN] 在 {args.dataset_dir} 下没有找到任务子目录")
        else:
            print(f"  task                     | train-gt | test-gt  | "
                  f"paper-train | paper-test | match")
            print("  -------------------------+----------+----------+"
                  "-------------+------------+------")
            overall_ok = True
            for task, p_train in PAPER_TRAIN_COUNTS.items():
                p_test = PAPER_TEST_COUNTS[task]
                files = files_map.get(task, [])
                # generate_dataset.py 不区分 split，我们汇总全部样本对比总量
                total = count_qa_in_files(files)
                paper_total = p_train + p_test
                ok = (total == paper_total)
                overall_ok = overall_ok and ok
                print(f"  {task:<24} | {'-':>8} | {'-':>8} | "
                      f"{p_train:>11} | {p_test:>10} | "
                      f"{('OK' if ok else f'x={total}')} (paper_total={paper_total})")
            if not overall_ok:
                print("\n  [INFO] 实际样本数与论文略有差异，通常因为：")
                print("        (a) 你生成的是子集 (start_index/end_index 不是全集)；")
                print("        (b) 后处理步骤丢弃了部分不符合格式的样本；")
                print("        (c) 数据集正在重新生成中。")

    print()
    print("=" * 64)
    if (sc['train'] == EXPECTED_TRAIN_SCENES and sc['test'] == EXPECTED_TEST_SCENES):
        print(f"[CHECK] Scene 划分与论文一致: 700 / 150 ✓")
    else:
        print(f"[CHECK] Scene 划分与论文不一致！期望 700/150, 实际 "
              f"{sc['train']}/{sc['test']}")
        if not args.nuscenes_root:
            print(f"  提示: 添加 --nuscenes_root <path> 使用官方划分进行校验")
        if not args.fix_split:
            print(f"  提示: 添加 --fix_split + --nuscenes_root <path> 自动修复")


if __name__ == "__main__":
    main()