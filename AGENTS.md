# 仓库开发约定

## 沟通与版本管理

- 尽量使用中文回复。
- 文件发生较大改动时，在对应 Git 仓库创建提交。提交标题采用 `type(scope): 中文目的`，类别使用标准英文（如 `feat`、`fix`、`docs`、`refactor`）；提交正文用中文列出具体改动及目的。
- 暂存时列出具体文件，保留用户已有改动；运行日志、预测、数据、权重和临时核验产物不纳入提交。

## 当前主线：upcore

- 当前目标是更新 B4DL 主模型及其数据、训练、推理和评测链路，提升基于 LiDAR 证据的回答质量与可复现性。范围和验收顺序见 [upcore 开发方向](docs/upcore/README.md)。
- 主线代码是 `mllm/b4dl_core/`、`mllm/vtimellm/`，以及相应的数据与评测入口。时空理解和时间定位继续属于 B4DL 六任务。
- MaskPLS、Grounded4D、AnswerSeg 和伪掩码相关内容保留为历史研究及可选扩展；分割召回、点掩码或跨帧实例跟踪不作为 upcore 的默认开发目标。
- `core_geometry_*`、数据 schema 和历史实验路径保留其已有含义。分支改名不应触发模型模块、冻结配置、权重或历史记录的批量改名。
- 模型效果以完整、同口径的 B4DL 回归和输入干预实验为依据。训练 loss、CPU 小样例和历史进度快照各自注明边界。
- 功能或开发方向变更同步维护 [项目方法与落实情况](docs/项目方法与落实情况.md)。历史实验条目保留原日期、分支和结论。

## 环境与验证

- 使用已验证的 `wqlc` 环境。当前本体 CPU 检查栈为 Python 3.10、PyTorch 2.8.0、Transformers 4.47.0、PEFT 0.13.2；CUDA 环境另行按实际硬件验证。
- `mllm/requirements.txt` 已对齐本体 CPU 检查栈；CPU/CUDA 与可选扩展分层安装，先运行 `python -m b4dl_core.runtime`。具体步骤及未验收边界见 [环境与恢复说明](mllm/docs/upcore_runtime.md)，安装成功不等同于 CUDA 或模型效果验收。
- 根据实际改动运行相关检查；CPU 通过、依赖跳过、GPU 验证和完整性能评测分别报告。审查结论见 [2026-10-27 代码审查](docs/research/upcore代码审查_20261027.md)。
