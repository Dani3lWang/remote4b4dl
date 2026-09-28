# 选项 A 三臂串行编排的**权威副本**（正在跑的那份是它的拷贝，放在 gitignore 的
# mllm/training_logs/temporal/ 下）。预注册与判据见
# docs/learn docs/B4DL_ReasonSeg多帧A_预注册_20260927.md。
#
# 两道保守闸：(1) 等 A2 的 CHAIN DONE，超时不启动；(2) 等 a2_verdict.txt 落盘
# （它由另一个 watcher 写），拿不到就停止 —— 绝不把"文件不存在"当成"失效门通过"。
