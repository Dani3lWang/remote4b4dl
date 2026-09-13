"""GRPO stack for B4DL.

Single 32GB GPU cannot hold a merged bf16 7B actor, a second frozen reference
copy and ZeRO-3 activations at once, so the three roles run as separate
processes and exchange JSONL:

    rollout.py    sample G completions per prompt + the sampling-time logp
    logprob.py    one forward over prompt+completion -> per-token logp
    reward.py     score via evaluation.evaluate_model, then group-normalise

Every correctness constraint that is not obvious from the code is documented
where it bites; the review that produced them is in
`docs/learn docs/B4DL_GRPO实施方案_M1审计后定稿_20260913.md`.
"""

import os
import sys

MLLM_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
if MLLM_DIR not in sys.path:
    sys.path.insert(0, MLLM_DIR)

DEFAULT_TRAIN_JSON = os.path.join(MLLM_DIR, 'b4dl_dataset',
                                  'stage2_full_train_seqv3_meta2_148k.json')
DEFAULT_MODEL_BASE = os.path.join(MLLM_DIR, 'base_model', 'vicuna-v1-5-7b')
DEFAULT_MM_ADAPTER = os.path.join(
    MLLM_DIR, 'checkpoints', 'vtimellm-vicuna-v1-5-7b-stage1', 'mm_projector.bin')
DEFAULT_B3 = os.path.join(
    MLLM_DIR, 'checkpoints', 'vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3')
DEFAULT_FEAT_FOLDER = os.path.abspath(
    os.path.join(MLLM_DIR, os.pardir, 'encoders', 'lidarclip', 'b4dl', 'stage2_features'))
