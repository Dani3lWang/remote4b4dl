---
name: b4dl-modify-only-project-folder
description: 写操作只允许在当前项目文件夹 /root/autodl-tmp/wql/mmb4dl 内进行，禁止改动其他文件夹（服务器上多项目共用）
metadata:
  node_type: memory
  type: feedback
  originSessionId: sess_b8c4a691-1a06-4fb5-a14e-00f177f3b543
---

2026-08-30 用户明确要求：不要修改其他文件夹的内容，修改只允许在当前项目文件夹（/root/autodl-tmp/wql/mmb4dl）中。

**Why:** 这台 AutoDL 5090 服务器是多项目共用的（如 /root/autodl-tmp/ymt/CNANET、CoRViD 等），改动其他项目的文件会干扰他人工作；用户对此有明确边界要求。

**How to apply:** 一切写操作（改文件、建脚本、下载数据、装环境、git 操作）仅限项目目录内；排查跨项目问题（如占用 GPU 的其他训练进程）时只做只读检查（ps / nvidia-smi / tail 日志 / grep 代码），绝不写入或改动。参见 [[b4dl-server-access-workflow]]。
