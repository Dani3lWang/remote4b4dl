# Training Script

```shell
bash runner.sh \
     --s1_data ./lidarllm_only_dataset/stage1_lidarllm_mm.json \
     --s1_feat ./lidarclip/stage1_features \
     --s2_data ./b4dl_dataset/stage2.json \
     --s2_feat ./b4dl/stage2_features \
     --model_name_or_path ./base_model/vicuna-v1-5-7b
