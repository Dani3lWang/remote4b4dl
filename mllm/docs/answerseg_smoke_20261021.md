# 2026-10-21：AnswerSeg 远端同步与链路冒烟

代码已同步，真实权重的加载、在线梯度、一次优化、保存重载及受控点云掩码回放通过。**自由回答分割尚未通过**：自由生成缺少 `<SEG>`，现有已审核绑定训练/开发文件均为 0 条。本记录属于工程验收，不是分割效果或正式训练结果。

## 部署与原任务保护

| 项目 | 实际记录 |
| --- | --- |
| 服务器 | `ssh -p 21358 root@connect.bjb1.seetacloud.com` |
| 独立目录 | `/root/autodl-tmp/mmb4dl-answerseg-20261021-922e4a8` |
| 当前功能源码 | 上述目录的 `source-r2/`，提交 `dead805158e5b68a9e5f2820bb0826c3971a4b38` |
| 初版接口 | `922e4a8b3354565a2f7f2fbdd844f381718f4082`，保留在 `source/` |
| 新版源码归档 SHA-256 | `a8c0bee28c4a821740736304512b824312465470cad9aa66e36aed20bdb2a5d9` |
| 源码核验 | 156 个文件逐一比对，无差异；本地另与 Git 归档逐文件核对通过 |
| Python | `/root/autodl-tmp/mmb4dl-maskpls-20261016/runtime/wqlc/bin/python` |
| 实测环境 | PyTorch 2.8.0+cu128、Transformers 4.47.0、PEFT 0.13.2 |

原目录 `/root/autodl-tmp/mmb4dl` 正在进行 B3 两轮训练，PID 为 72247。其 HEAD 仍为 `d4c3e7fe2364c05e864da7ce85e71839acfb794a`，原有脚本修改和未跟踪目录保持原状。本次没有在该目录切换分支、更新源码、修改依赖或改写 checkpoint，也没有停止原任务。

## 资源安排

现场 RTX 4090 的整卡占用为 8423 MiB，空闲 15670 MiB；原训练进程占用 8372 MiB，GPU 利用率多次处于 70%–95%。因此采用 `runtime.language_device=cpu`：真实 7B 语言部分及其 LoRA 在 CPU，空间模型和对齐头在 CUDA，隐状态与损失跨设备保留梯度。

工程进程以低 CPU 优先级、2 个 CPU 计算线程运行，设置约 3 GiB 的 PyTorch 进程显存上限和 20 分钟超时。实测峰值 allocated 为 **1463.86 MiB**，reserved 为 **1840 MiB（约 1.80 GiB）**；这两个数不包含全部 CUDA 上下文开销。测试结束后显存恢复到 8423 MiB，GPU 进程列表仅剩原 B3 任务。本轮没有进行完整 7B 同设备 GPU 训练测试。

## 已完成验证

| 检查 | 结果与适用范围 |
| --- | --- |
| 远端 CPU 检查 | 新接口 29 项、既有 MaskPLS 65 项、回答绑定数据 8 项，共 **102 项通过** |
| B3 原闭包检查 | `preflight_b3.py --project-root /root/autodl-tmp/mmb4dl --require-cuda` 退出 0 |
| 真实语言权重 | 使用受保护的历史三轮 B3；没有读取正在写入的两轮训练 checkpoint，不能将历史权重写成当前两轮结果 |
| 空间权重 | 原生 MaskPLS E2-P P0 的 `epoch-06.pt`，实现哈希一致；仅作工程输入，没有替换已保留的正式基线 |
| 真实数据输入 | nuScenes **trainval 单帧** 34592 点、对应真实 panoptic 标签，同场景 SST 特征 `[40,768]`；不是 mini 全量测试 |
| 空间预测 | 5 个导出候选，2 个 GT 实例满足几何覆盖门槛；选择其中一个作为明确标记的 oracle 工程样例 |
| 在线反向与更新 | 一次更新约 21.63 秒（不含加载）；LoRA、对齐头均有非零梯度，137 个扩展参数张量改变，旧输入/输出词表参数哈希不变 |
| 诊断损失 | 总损失 9.557273，文本 6.721296、绑定 2.041778、mask 0.794200；不能作为模型效果指标 |
| 保存与重载 | 保存工程 bundle 后释放模型，重新恢复真实基础权重及扩展，扩展参数逐张量哈希一致 |

这一例的问题、回答 span 和实例身份是**特权工程样例**，用于证明张量与梯度连接，不是人工审核过的语言指代。未将它写入正式 binding 数据，正式训练没有启动。测试生成的 `evidence/smoke-002/engineering-bundle/` 仅供复现冒烟，不作为已验收模型。

## 自由生成与受控回放分开记录

自由生成正常返回文本 `The pedestrian.`，状态为 `missing_seg_token`；没有分割对象，导出掩码形状为 `[0,34592]`。因此，本轮**没有通过自由回答 → SEG → 唯一实例 → 点掩码的验收**。工程提示没有建立可评价的真实 QA 绑定，不据此给出 B3 问答准确率。

另一次回放明确控制了 assistant token 序列和实例选择，保留真实语言 forward 与真实空间预测。回放选择帧内 query 102，输出 186 个前景点，原始点序完整。此处不是自由生成结果，也不是语言绑定成功率。

本地下载了该帧源 GT 掩码，核对 SHA-256、稳定实例身份、原始 point_indices 和点云哈希引用，并独立复算：交集 153、并集 222，IoU=`153/222=0.6891891892`，与服务器记录一致。该单例 IoU 只证明受控接口及点序连接，不能作为整体分割指标。原始 LiDAR 文件没有下载；其文件内容哈希由服务器的数据加载器校验。

## 首次失败与修复

`smoke-001` 已完成真实 7B 加载，在 KeOps 空间前向失败，退出 1。隔离缓存首次编译时链接器报 `cannot find -lnvrtc`，没有生成 `nvrtc_jit.so`。

修复仅作用于本任务：把已有已编译 KeOps 缓存复制到任务缓存，为子进程补充 NVRTC/runtime 库路径。极小 CUDA 插值探针通过后，在新目录运行 `smoke-002`，退出 0。旧失败日志、旧输出目录和旧源码副本均保留，没有修改全局库、原训练环境或插值算法。

## 回执与下一步

远端证据位于部署目录的 `evidence/`；最终完成回执为 `evidence/smoke-002/completion.json`，源码核验及原任务存续记录为 `evidence/final-verification.json`。工程 bundle、优化器和原始基础权重留在服务器，未声称本地迁回完整权重。

紧凑交付归档共 37 项日志、JSON、脚本和掩码，SHA-256 为 `d99e8f4436f42a7e469e52ae139deb04beaba3ff4d181c7fea3ef7680e03c671`。本地已经校验归档内全部文件：

- [本地交付核验](../../training_logs/answerseg_20261021/deployment/local-receipt.json)
- [原始点序与 IoU 独立复算](../../training_logs/answerseg_20261021/deployment/local-mask-audit.json)
- [远端完整冒烟回执](../../training_logs/answerseg_20261021/deployment/remote-delivery/evidence/smoke-002/completion.json)
- [使用接口与配置](answerseg.md)

最高优先级是审核一小批唯一实例绑定，分别填入训练与开发文件，再执行真实 `joint` 格式微调并验证自由生成。冻结 B3 的 `binding` 阶段不能让模型学会输出新 token。当前审核文件仍为空；未开展这一步，也未开展 mini 全量、六任务回归或跨帧验收。
