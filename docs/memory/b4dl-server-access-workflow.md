---
name: b4dl-server-access-workflow
description: 用户 Daniel 的开发工作流——Windows 本地仓库 + SSH 到 AutoDL GPU 服务器（wqlc conda
  环境、paramiko 上传）
metadata:
  node_type: memory
  type: user
  originSessionId: sess_57bc4f46-cf94-4f89-9779-568442660f70
---

用户（Daniel，GitHub Dani3lWang）的工作流：Windows 本地仓库 `D:\Github\remote4b4dl`（只有代码，数据/特征/权重全部 gitignore 不在本地）+ SSH 到 AutoDL GPU 服务器实际跑训练评测。

- 服务器：RTX 5090 32GB，PyTorch 2.5.1 CUDA 12.4，DeepSpeed ZeRO-3 单卡。SSH 地址随实例变化（出现过 `connect.westc.seetacloud.com:23224` 和 `connect.westd.seetacloud.com:13143`），root 登录，密码用户会在对话中直接给出（2026-08-23 时为 Xc/LBsx20E55）。
- 远端主仓库：`/root/autodl-tmp/wql/mmb4dl`（与本地 remote4b4dl 同源但内容更全）；nuScenes 原始数据在 `/root/autodl-tmp/ljq/mmb4dl-main/nuscenes`。
- Windows 无 sshpass，用 Python paramiko 建立非交互 SSH/SFTP 执行命令和上传文件。
- 远端 conda 环境名 `wqlc`（需 `eval "$(conda shell.bash hook)" && conda activate wqlc`，直接 `conda activate` 会 command not found）。不同实例 conda 安装位置不同（见过 `/root/autodl-tmp/miniconda3` 和 `/root/autodl-tmp/.conda-stuff/envs/wqlc`），非交互脚本可直接用 env 的 bin/python 绝对路径兜底。
- 用户会把关键对话导出成 md 文件作为操作记录让 agent 阅读、写入记忆；中文交流。
- Windows 侧其他路径：官方上游仓库副本 `D:\tmp\B4DL`、HF 数据集下载处 `D:\tmp\data\nuScenes-B4DL`、Qoder 工作区 `C:\Users\Daniel\.qoderworkcn\workspace\*`。Windows GBK 控制台 print ✓/⚠/中文会 UnicodeEncodeError——paramiko 脚本应把输出写文件再读，不要直接 print。

相关：[[b4dl-project-overview]]
