#!/bin/bash
cd /root/autodl-tmp/wql/mmb4dl/mllm
eval "$(/root/autodl-tmp/miniconda3/bin/conda shell.bash hook)"
conda activate wqlc
echo "===== BASELINE EVAL (no per_sequence) ====="
echo "Start: $(date)"
python evaluation/test_b4dl.py     --model_base ./base_model/vicuna-v1-5-7b     --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin     --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full     --feat_folder ../encoders/lidarclip/b4dl/stage2_features     --test_data ./b4dl_dataset/test_qa.json     --ego_meta ./b4dl_dataset/ego_metadata.json     --output ./evaluation/predictions_baseline.json     --metrics_output ./evaluation/eval_baseline_metrics.json     2>&1 | tee ./evaluation/baseline_eval.log
echo "End: $(date)"
