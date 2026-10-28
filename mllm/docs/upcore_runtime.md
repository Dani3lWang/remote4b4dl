# upcore 环境与推理恢复

更新日期：2026-10-28。本说明对应当前 `upcore` 源码；历史部署继续使用其原代码、环境和冻结身份。

## 基础依赖

`mllm/requirements.txt` 现在声明基于预计算 LiDAR 特征的本体运行依赖：Python 3.10、Torch 2.8.0、Transformers 4.47.0、PEFT 0.13.2、Accelerate 1.15.0、NumPy 1.26.4、SentencePiece 0.2.2。Torch 接受所选 CPU/CUDA wheel 的本地版本后缀。默认使用 SDPA，基础清单不安装 flash-attn、DeepSpeed、视频解码或空间 CUDA 扩展。

| 清单 | 用途与边界 |
| --- | --- |
| `requirements-cpu.txt` | 基础依赖及 Torch 2.8.0+cpu，用于 CPU 工程检查 |
| `requirements-cuda.txt` | 基础依赖及 Torch 2.8.0+cu128；需要另行核验驱动、GPU、实际 FP16 前向与训练 |
| `requirements-eval.txt` | 原六任务的指标包；NLTK 资源、RoBERTa 权重、Java/METEOR 及冻结协议单独准备 |
| `requirements-demo.txt` | Gradio、图像/视频与可视化依赖，不属于本体推理前置条件 |
| `requirements-legacy.txt` | 原训练器、ChatGLM 和视频路径的可选依赖；DeepSpeed 等需独立 Linux/CUDA 验证 |
| `requirements-maskpls.txt`、`answerseg/requirements.txt` | 历史空间扩展，继续保留各自环境说明，不能覆盖已验收的主模型环境 |

新环境中按用途选择一个 Torch 清单，随后安装所需扩展。例如在仓库根目录、独立 Python 3.10 环境执行：

```bash
python -m pip install -r mllm/requirements-cpu.txt
# 使用 CUDA 12.8 wheel 时，将上面的文件换为 requirements-cuda.txt。
export PYTHONPATH="$PWD/mllm"
python -m b4dl_core.runtime
# GPU 环境还需执行：python -m b4dl_core.runtime --require-cuda
python -B -m unittest discover -s mllm/tests -p 'test_core_*.py' -v
```

PowerShell 使用 `$env:PYTHONPATH = (Join-Path (Get-Location) 'mllm')`。离线检查可将 `HF_HOME` 指向可写目录，并设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。

`b4dl_core.runtime` 在不下载或加载大权重的情况下核对版本、包导入、Llama 的 `cache_position` 接口、SDPA 和可选 CUDA 可用性。核心基底加载和原 `scripts/preflight_b3.py` 数据闭包检查都调用它。旧 Transformers 4.31.0 会得到明确拒绝；不通过忽略参数来改变生成语义。

本次实测使用既有 `wqlc`，Python 3.10.20 / Torch 2.8.0+cpu，其他版本与基础清单一致。81 项核心检查中 79 项通过、2 项因缺 nuScenes/pyquaternion 跳过；包含微型 VTimeLLM 前向、生成、保存重载和真实训练入口恢复检查。Pillow/decord 改为在使用图像/视频功能时导入，六任务的文本 token 化不再被这些可选包阻断。

这些清单不是所有传递依赖的完整锁文件。本次未联网重建干净环境、未安装或验证可选扩展，也没有完成 CUDA、真实 7B 或任务质量验收。已有远端环境（例如历史 Accelerate 1.3.0）不自动视为本次清单通过；环境迁移需独立验证，不能在旧实验上直接升级依赖。

## 几何推理续跑

首次运行仍需新输出目录。中断后，在**相同代码、环境、数据、权重、参数和输出目录**的原命令后增加 `--resume-inference`：

```bash
python -m b4dl_core.b3 infer --dataset DATA --baseline BASELINE.json \
  --preparation PREPARATION.json --arm A0 --split test --output predictions

python -m b4dl_core.b3 infer --dataset DATA --baseline BASELINE.json \
  --preparation PREPARATION.json --arm A0 --split test --output predictions \
  --resume-inference
```

A1/A2/A3 还需要相同的扩展 checkpoint；`--resume CHECKPOINT` 继续专用于训练，不与推理恢复混用。

- `resume.json` 在第一批生成前记录实现、基底/扩展权重、准备配置、请求顺序/指纹、控制计划、batch、解码设置、运行库和设备等身份；`run.json` 保持原几何评分字段。
- `inference-journal.jsonl` 每行提交一批预测及对应输入回执，附运行身份和内容摘要。每批序列化完成后追加并 flush/fsync；中断时只重做尚未提交的批次。
- `predictions.jsonl` 和 `input-receipts.jsonl` 是兼容评测、进度工具的导出文件。正常运行按批追加；恢复时从已校验日志各重建一次，修复日志提交后导出中断造成的不一致，不再每批重写累计全量结果。
- 完成前重读日志和导出文件，核验覆盖、顺序及内容，原子写出含三份文件 SHA256 的 `completion.json`。已完成且哈希一致的恢复不会重新生成；完成产物被改动时拒绝恢复。

历史几何输出没有共同提交日志或完整续跑身份，不能直接由新入口接管。旧 checkpoint/冻结实验的实现身份也不放宽；新旧迁移仍需单独的核验与记录。

## 六任务与文件恢复规则

六任务沿用 `python -m b4dl_core.six_task infer ... --resume`。新的运行清单增加配置、请求、运行库和设备绑定，保持原协议和答案辅助元数据含义。恢复只接受确定性解码。

完整但无末尾换行的 JSON 记录会补齐换行。只有最后一行没有换行且内容可确认是合法 JSON 对象的未完成前缀时，才舍弃该未提交尾行；包括字符串、数字、字面量及 UTF-8 字符的中途截断。修复前先校验所有完整记录的身份，再将**原文件完整字节**保存为 `*.before-recovery-<sha256>.bak`，然后原子发布修复文件。中间损坏、非法尾行、重复/错序 ID、指纹不符及多余记录均报错，原文件保留。

六任务在结束时重读全部预测并核验数量、顺序、ID 和指纹，再写入带预测 SHA256 的完成凭证。OOM 保留已完成前缀并退出；其他生成错误继续记录为空回答，保留原评测分母。两个入口都通过操作系统文件锁防止同一目录并发写入；进程退出后锁自动释放，不需要删除锁文件。

恢复测试通过不代表模型回答正确。CPU 故障注入验证的是文件和控制流契约，不是历史 CUDA 非有限损失的根因复现，也未启动或恢复真实训练。
