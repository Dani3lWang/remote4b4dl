# 原始 B3 的首轮完整几何 SFT

日期：2026-10-23。用户要求完成所有准备并开始训练。本轮正式训练前的来源、数据、预算、种子、六任务输入、指标资源和 GPU 接口检查已完成。训练完成与模型效果验收分别记录，不能由启动回执代替。

**实时更新（2026-10-26）：训练已失败，未恢复。** 4953是最后成功更新，失败发生在尝试更新4954的前向loss；正式completion不存在。step4750的38个参数及Adam状态共152个张量、文件哈希和四项身份重新核验通过；四个候选输入在完整CPU前向中均有限，原始首个非有限运算尚未唯一确认。本轮补齐失败记录/实际状态快照及梯度范数保护，48项CPU检查通过、2项依赖缺失跳过，未改动原冻结配方或放宽恢复身份。旧训练和评测PID均已不存在，旧评测状态文件中的running不能作为存活证据。恢复条件、证据与限制见[非有限损失现场诊断](../../docs/research/A3_nonfinite现场诊断_20261026.md)。下文2026-10-23启动与评测说明均为历史快照。

## 固定配方

2026-10-26晚间跟进：独立只读进度核验确认A3开发集实际保留4562/13602条预测（4558ok、4truncated），各条请求身份匹配，但没有阶段completion，旧running状态过期；其余五个推理阶段没有预测。训练仍失败，GPU仍不可用。工具使用、最新汇总及后续顺序见[本体实验进度与改进](../../docs/research/B4DL本体实验进度与改进_20261026.md)。本工具不改变训练源码或冻结实现身份。

| 项目 | 本轮约定 |
|---|---|
| 基底 | 受保护的历史三轮 B3，绑定已核验 nuScenes 退火 SST；四组件身份不变 |
| 数据 | `3c8f568806797d265119592eea219bdbe3b7180634ad05a3345fe9a9af8b4e03`，按真实右后视野支持修订的完整 trainval |
| 首个正式运行 | A3 / seed 42 |
| 训练预算 | 55,404 条 train，1 epoch，13,851 次 optimizer update；batch 1 × accumulation 4，恰好每题一次 |
| 参数 | FP32 projector＋2层/768维/8头时序几何适配器；原语言模型冻结；新增 LoRA rank=0 |
| 优化 | AdamW，lr=1e-4，weight_decay=0.01，betas=(0.9,0.999)，eps=1e-8，恒定学习率，clip norm=1 |
| 精度 | 冻结 B3 FP16，trainable master FP32，FP16 autocast，GradScaler 初值1024；非重入 gradient checkpointing |
| 保存 | 第1步、每250步及最终步保存；保留step1＋最近2个checkpoint，独立输出，不覆盖原B3 |
| 终点选择 | 固定最后一步，不用 test 选择 checkpoint 或超参数 |
| 对照安排 | A1/A2 同数据顺序、步数、有效batch及rank；seed42/43/44已登记。此次只启动A3/42，其余对照和重复尚未执行 |

这是首轮新增几何监督的1轮预算，不是原B3两轮复现配方。不同结构的参数量和FLOPs不同，不能写作等参数或等FLOPs实验；比较时报告相同数据呈现次数、优化步数和实测资源。

不可变实验配置：`mllm/configs/core_geometry_experiment_sft_v1.json`。实验ID为 `bb4b46d916bbd403a6d9a9e776b4d4dffbd1d753299a0aff5408bd939b476f46`，配置SHA256为 `153ae2cf00e7f1c674554663d84c6784854eb5eaaf3ceada2a1f452a155ff2c4`。正式入口检查命令与配方、数据、基底及六任务输入哈希一致；手动清空门控不能代替这些检查。

## 六任务回归已经接通

原六任务共30,145条，覆盖150个官方validation场景：existence3770、binary_qa7525、time_grounding2783、description3770、temporal_understanding4757、comprehensive_reasoning7540。冻结后的query、原始目标、场景包身份、顺序、指标代码和资源共394个文件有哈希记录。

协议沿用原B3的whole_scene输入、per_sequence元数据及answer_frames。后者依据GT答案恢复时间范围的元数据，属于oracle-assisted口径，必须随报告说明。A0和训练后模型使用完全相同的提示与场景输入；六任务提示不增加几何JSON指令。

`b4dl_core.six_task` 使用真实帧时间和位姿直接调用新生成接口，解码continuation，不使用旧脚本按原输入长度切片的方法。错误、截断输出保留记录和分母；OOM直接停止，不悄悄跳样。支持按固定输入指纹及checkpoint哈希恢复预测。

固定greedy、单beam、cache、max_new_tokens=1024。METEOR主报NLTK-2005，次报Java Meteor-1.5；BLEU为COCO corpus实现，ROUGE-L沿用原stem配置，BERTScore使用本地RoBERTa-large第17层。GPT评分关闭。

最终协议ID：`d8d2885f0fa52456a6070b0ade710e4bc6a178707eaa042cb21decde9163edd1`，远端文件为 `protocol-v2/six-task-manifest-final.json`。Java绑定版继承同一份已验证query文件，只增加独立Java运行时身份；不更改QA或生成设置。

## 实际准备检查

- 既有 `scripts/preflight_b3.py --require-cuda` 在原项目资产目录通过，Stage1、B3、原测试集、特征、模型、评测闭包完整。
- 全量文本审计：55,404条训练问答与30,145条回归输入均检查。训练展开最长389 token；六任务提示＋视觉最长241，加1024生成预算仍低于4096模型上限。
- 44项远端核心检查通过，含冻结配方与材料篡改拒绝、六任务错误分母、首步/最近checkpoint保留和无关文件保留。
- 真实7B GPU检查：六类提示各8-token生成，原B3与零初始化A3逐token一致；优化器更新0次，此项为接口验收，不是任务准确率。
- 主指标fixture通过后发现Java次级后端缺失，最终补齐并严格验证双后端。NLTK四包由官方索引SHA256校验，单独部署于本次assets目录；JRE使用Azul官方OpenJDK8发行包及官方SHA256，未修改系统环境。分词、WordNet访问、主次METEOR及RoBERTa-L17实际计算通过，fixture使用合成正确答案，不能当作B3成绩。
- 源码与实际部署逐文件核验，训练实现SHA256为 `aa072e07d5696d40d131999eb3e8377511ee0b333c7c779c328e8ea9ca51216f`。准备证据位于本地 `training_logs/core_geometry_20261022/sft-preparation-evidence`。

## 运行与恢复

远端运行根：`/root/autodl-tmp/mmb4dl-core-sft-20261023-r1`。正式训练使用提交后核验部署的源码；作业回执在 `jobs/A3-seed42`，训练输出在 `training/A3-seed42`。`job`仅启动一个明确预算的作业，不自动扩展配方或连续启动其他实验。

```bash
# 在最终部署源码目录内运行，PYTHONPATH指向本目录mllm。
PY=/root/autodl-tmp/mmb4dl-maskpls-20261016/runtime/wqlc/bin/python
RUN=/root/autodl-tmp/mmb4dl-core-sft-20261023-r1
DATA=/root/autodl-tmp/mmb4dl-core-trainval-20261023-r1/trainval-v1
BASE=mllm/configs/core_geometry_baseline_original_b3_trainval.json
PREP=mllm/configs/core_geometry_preparation.json
PLAN=mllm/configs/core_geometry_experiment_sft_v1.json

"$PY" -m b4dl_core.job start --job-dir "$RUN/jobs/A3-seed42" --dataset "$DATA" --baseline "$BASE" --preparation "$PREP" --experiment "$PLAN" --arm A3 --seed 42 --output "$RUN/training/A3-seed42"
```

启动后以training.log、runtime.json、loss.jsonl及latest.json确认模型恢复、实际优化步和checkpoint。job的process.json记录PID，exit.json记录最终退出码；completion.json仅在固定训练预算完成后生成。GPU全程使用共享独占锁。

从中断恢复时，使用相同源码、数据、基底、计划、seed、lr、batch及累积配置，运行 `b4dl_core.b3 train ... --resume 原checkpoint --output 新目录`，总steps仍为13,851；不把恢复当成追加一轮。扩展checkpoint保存optimizer/scaler/RNG及实验SHA256，身份变化会拒绝。

2026-10-26修复会改变实现身份，因此上述命令只适用于身份完全一致的恢复；不能用新代码直接恢复旧step4750。需先完成诊断报告中的CUDA数值检查、资产核验及显式受控迁移记录，本轮未创建迁移权重或启动训练。`failure-state.pt`只供故障分析，加载入口拒绝其用于恢复或推理。

训练完成后还需运行固定终点的geometry dev/test和六任务A0/扩展比较。geometry推理传同一 `--experiment`，预算256；六任务用 `six_task infer`、`six_task metrics` 并传最终协议及同一实验文件。此时才能报告泛化、退化或相对改进。右后视野的方位缺口、train59条缺同长度供体、internal_dev可能被历史基底见过等数据限制继续有效。


## 正式启动回执（2026-10-23）

北京时间 2026-10-23 14:12:41 启动作业，训练PID344870、watcher PID344869。实际运行源码为 `source-1d23109`，代码提交 `1d231095a5d3f0530eb74277a7bfbaa0b1e05b3e`；61文件逐一核验。部署保留已验收的运行字节，并同时核对Git规范化内容，记录39个Windows换行转换，避免改变配置或实现SHA256。

截至迁回的启动回执，已完成297/13,851次真实更新，呈现1,188条样例，epoch=0.0214425，`engineering_only=false`。可训练参数13,210,832，冻结参数6,738,432,000；第二步内部attention梯度绝对值和13.1761，观察到的loss/梯度norm均有限。近期100步均耗时0.6211秒，按该窗口估计剩余2.34小时，仅为运行时估计。

首步 `step-1.pt` 共158,581,579 bytes，SHA256 `852ee92ffb434ed0d9eff688a1f975c86eeeae1e2b186d47d0755409fef1e215`；38个FP32参数张量、38份optimizer state均有限，Adam step均为1，实验/数据/基底/实现身份与运行一致。启动快照的latest已到step 250。显存使用14,155MiB。检查只在CPU读取权重，没有打断训练。

本地证据为 `training_logs/core_geometry_20261022/sft-startup-evidence`，运行产物持续保存在远端 `training/A3-seed42` 与 `jobs/A3-seed42`。代码和准备配置仍是不可变的启动快照；实时状态查看PID、loss、latest、job exit与最终completion。此启动快照采集时训练进行中；后续已因非有限loss失败，见文首实时更新。首步与loss下降均不是性能提升证据。
