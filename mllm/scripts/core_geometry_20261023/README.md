# 2026-10-23 B4DL 本体实验的云端脚本

2026-10-25 从 `root@connect.bjb1.seetacloud.com:21358` 采集并原样保存，用于复核首轮 A3 / seed42 训练及中途检查点评测。模型、训练、推理与指标实现位于 `mllm/b4dl_core/`；这里补齐此前只保存在云端的运行编排与审计脚本。

## 来源与作用

云端实验根目录为 `/root/autodl-tmp/mmb4dl-core-sft-20261023-r1`。

| 本地文件 | 云端来源 | 作用 |
| --- | --- | --- |
| `audit_training.py` | 实验根目录下同名文件；`evaluation-20261023-r1/` 内副本内容相同 | 核对训练日志步数、最后检查点哈希、参数与优化器有限值及模型/数据/实验身份，导出失败更新的候选 microbatch |
| `run_evaluation.py` | 实验根目录下同名文件；`evaluation-20261023-r1/` 内副本内容相同 | 串行执行 A3-step4750 与 A0 的几何 dev/test、配对比较、先验及原六任务评测，记录阶段命令、退出状态与完成状态 |
| `snapshot_progress.py` | `evaluation-20261023-r1/snapshot_progress.py` | 冻结并评分当前开发集预测前缀，明确记录样本数、完整分母及临时结果限制 |
| `legacy_b3_2epoch.sh` | `/root/autodl-tmp/mmb4dl/mllm/scripts/run_stage2_full_seqv3_mixed_b3.sh` | 保留服务器尚未提交的两轮 B3 训练脚本；当前维护入口 `mllm/scripts/train_b3.sh` 已采用两轮配置 |

四个文件的原始字节 SHA256 分别为：

```text
e11ed690ac1726c1be53bde82c5304b030bcbc714b7367795620dd77a4373c3f  audit_training.py
5e28cd02ca71c2f72a88464a8c235c00957f16d6aa4bd5c06d1638fcfdd257d0  run_evaluation.py
e4f0e7eecd65b54545b6a701f1161c1a31536148e9bb714e63d704c4da190e53  snapshot_progress.py
762361f6199c15a6c4b3d9fcca95788ee6b6f9f6970f1ae09cdb95f702bf8cfa  legacy_b3_2epoch.sh
```

## 实验路径与运行约定

脚本保留原实验的绝对路径和 step4750 检查点选择。Python 脚本包含顶层执行逻辑：审计和进度脚本会写对应回执，评测脚本会启动完整 GPU 推理。复核时使用原部署布局、原 `wqlc` 环境及对应数据/权重；`run_evaluation.py` 要求审计脚本位于 `evaluation-20261023-r1/audit_training.py`，阶段日志以新建方式打开，已有输出会导致退出。

旧 B3 shell 脚本移入本目录后，运行时须通过 `B4DL_ROOT` 指定项目根目录，并按需设置 `B4DL_ENV_PREFIX`。日常 B3 复现继续使用维护中的 `mllm/scripts/run_b3.sh` 与 `train_b3.sh`。

本次只采集脚本和核对源码，未启动训练或评测，未修改云端源码、实验配方、权重与运行任务。采集时，云端 `source-1d23109` 的 61 个部署源码文件与本地提交 `1d23109` 的 Git 内容在统一 CRLF/LF 后逐一匹配；额外两个文件是部署身份清单。归档不改变已有训练失败记录，也不构成效果验收。

本地核对记录位于忽略目录 `training_logs/b4dl_remote_submit_20261025/`，包含云端快照、原始哈希和逐文件比较结果。预测、日志、数据集与检查点继续留在运行目录。
