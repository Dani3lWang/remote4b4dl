# B4DL 训练与评测

当前 `upcore` 分支围绕 B4DL 主模型更新开展。代码审查、执行顺序及验收边界见 [upcore 开发方向](../docs/upcore/README.md)。

- 本体接入与检查：`python -m b4dl_core.b3`，说明见 [接入文档](docs/core_b3_integration.md)。
- 原六任务回归：`python -m b4dl_core.six_task`；几何诊断入口：`python -m b4dl_core`。
- 历史训练配置与已知中断：见 [训练记录](docs/core_geometry_training.md)。新实现不能直接绕过身份校验恢复旧实验。

运行以上命令时使当前 `mllm/` 位于 `PYTHONPATH`，并使用已验证的 `wqlc` 环境。入口存在不代表当前冻结配置、真实数据和权重已满足运行条件。

## 原项目基础训练入口

Before running, please download [this file](https://huggingface.co/lmsys/vicuna-7b-v1.5/tree/main) and place it under ./base_model/

```shell
bash run_stages.sh \
     --s1_data ./b4dl_dataset/stage1_lidarllm_mm.json \
     --s1_feat ./b4dl/stage1_features \
     --s2_data ./b4dl_dataset/stage2.json \
     --s2_feat ./b4dl/stage2_features \
     --model_name_or_path ./base_model/vicuna-v1-5-7b
```
