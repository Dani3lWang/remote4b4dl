# 2-Epoch Frame-Position Experiments

These two Stage2 entries use the same B3 mixed-data recipe and differ only in
the optional frame-position module:

```bash
cd mllm

# 2-Epoch baseline: frame position disabled
bash scripts/run_stage2_full_seqv3_mixed_paper2ep.sh

# 2-Epoch improvement: zero-initialized absolute scene-frame embedding
bash scripts/run_stage2_full_seqv3_mixed_framepos2ep.sh
```

Both commands use 148,271 mixed QA pairs, whole-scene input, BF16, LoRA
`r=64/alpha=128/dropout=0.05`, micro-batch 8, gradient accumulation 16,
learning rate `1e-4`, cosine decay, 3% warmup, zero weight decay, sequence
length 2048, gradient checkpointing, and ZeRO-3 CPU offload. The only
experiment variable is `--use_frame_position_embedding`.

The frame-position run stores `use_frame_position_embedding=true` and
`frame_position_max=64` in its checkpoint config. Evaluation reads those
fields automatically and restores the extra non-LoRA parameter from
`non_lora_trainables.bin`.

Before a GPU run, use the CPU smoke test in the training environment:

```bash
python scripts/verify_frame_position.py
```

Evaluate either output with the same frozen B3 protocol; only change the
`--stage2` directory and output directory between the two runs:

```bash
python evaluation/test_b4dl.py \
  --model_base ./base_model/vicuna-v1-5-7b \
  --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
  --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-framepos2ep \
  --feat_folder ../encoders/lidarclip/b4dl/stage2_features \
  --test_data ./b4dl_dataset/test_qa.json \
  --ego_meta ./b4dl_dataset/ego_metadata.json \
  --frame_motion ./b4dl_dataset/ego_frame_motion.json \
  --whole_scene --per_sequence --answer_frames \
  --output ./eval_results/framepos2ep/predictions.json \
  --metrics_output ./eval_results/framepos2ep/metrics.json
```

The 2-epoch value is an official-code-compatible setting; the paper itself
does not explicitly disclose its epoch count.
