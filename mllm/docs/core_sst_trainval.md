# SST 来源核对与完整 trainval

日期：2026-10-23。**来源核对、实际输入范围修订、完整 trainval 构建与审计均已完成。** 本轮没有训练模型。

## 1. 实际使用的 SST 来源

| 项目 | 核验结果 |
|---|---|
| 编码器 | `/root/autodl-tmp/mmb4dl/encoders/lidarclip/ckpt_anneal/lidarclip_mm/last.ckpt` |
| checkpoint SHA256 | `9707de49b2a6bda21ddf72f4c9a51dc26080c7ee6d67b73180fc0e53e9e63eda` |
| 训练元信息 | epoch 2 / global_step 15750，119 个 LiDAR encoder 张量；历史 nuScenes 训练、退火和提取日志相互对应 |
| 历史特征 | `encoders/lidarclip/b4dl/stage2_features`，850 个 float32 数组，共 34,149×768；全部形状、有限性、官方场景覆盖及逐文件 SHA256 通过 |
| 数值复算 | 数据开头、中部、末尾各一个 8 帧批次，共 24 帧；FP32/eval、无梯度，最大绝对误差 `1.9073486328125e-6`，最低余弦 `0.9999997615814209` |
| 原 mini 一致性 | 原 mini 的 10 份 `source_feature_sha256` 全部匹配此次特征清单 |
| 来源 ID | `dd1b870dbc10203056c666ee8a2fe60c9573562f4aac76582b03bdb25b4dbee1` |

数值比较预设阈值为 atol=5e-4、rtol=1e-4、cosine>0.999999；实际误差远低于该阈值。原始提取 RNG 未归档，因此此证据是“全文件哈希盘点＋24 帧原始数据抽样复算”，不写成全部特征逐位重提。复算 GPU 峰值 allocated 519.81 MiB。

当前特征对应 **CAM_BACK_RIGHT 视野裁剪**。历史 loader 将 LiDAR 经标定和各传感器时间戳的自车位姿变换到右后相机系，以深度>0.5 m、居中的正方形像素范围裁剪，再转换为 camera-z、negative-camera-x、negative-camera-y，强度除以 255。SST 的体素范围为 `[0,-20,-2,40,20,4]`，体素大小为 `(0.5,0.5,6)`。这不是 360° 全点云表示。

服务器没有相机图片；LiDAR 分支不消费图片像素，复算采用官方 `sample_data.width/height` 提供裁剪尺寸。24 帧中独立预处理输出与历史 loader 的点数组逐值一致。首轮因历史 loader 尝试打开缺失图片退出；第二轮因收集动态模块虚拟文件路径退出；第三轮修正后复算和回执均通过。失败日志保留。

## 2. 按用户选择修订监督

用户选择保留原始 B3 特征。新增 `sst_source.py` 重放相同的传感器变换、相机裁剪和体素范围，为 34,149 帧逐一计算实际保留点。仅当原始标注有 LiDAR 支持、且旋转标注框内至少包含一个保留点时，实例才进入对应帧的几何对象集合。框内判断使用 1e-5 m 容差，候选由 KD-tree 加速。

| 全量输入支持统计 | 数量 |
|---|---:|
| 原始关键帧点 | 1,185,656,192 |
| 裁剪与体素范围内保留点 | 69,651,839 |
| 已检查标注 | 1,166,187 |
| 含保留点的旋转框 | 65,489 |
| 原本有 LiDAR 点、但实际输入中无保留点的标注 | 869,505 |

这是旋转框内点的几何支持判断，不是 panoptic 点到实例真值。零原始 LiDAR 点仍按原协议保守处理为未知。存在性统计限定为输入支持的实例数；最近目标在该集合内选择；运动问题要求同一实例在两端都受输入点支持。方位和距离继续在各端点的自车坐标系中度量。所有新问题显式声明右后裁剪范围，避免将视野之外的物体当成当前输入可观察的监督。

原始 SST、B3、原 mini 及其历史回执保持原样；不将本次框内几何支持转成分割监督。

## 3. 完整数据及覆盖

数据 ID：`3c8f568806797d265119592eea219bdbe3b7180634ad05a3345fe9a9af8b4e03`。

| split | 场景 | 存在 | 方位 | 距离 | 相对运动 | 合计 | 严格相反答案对 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 560 | 41,570 | 8,524 | 796 | 4,514 | 55,404 | 13,569 |
| internal_dev | 140 | 9,964 | 1,920 | 388 | 1,330 | 13,602 | 2,803 |
| test | 150 | 11,028 | 2,122 | 328 | 1,468 | 14,946 | 3,151 |
| 总计 | 850 | 62,562 | 12,566 | 1,512 | 7,312 | 83,952 | 19,523 |

官方 train 700 场景按 seed 42 分为 560/140；官方 validation 150 场景作为本实验 test，不是 nuScenes 隐藏测试集。所有 850 场景都有入选题目，物理场景互不交叉。历史 B3/编码器可能见过 internal_dev，因此该集合只对本轮新增监督留出。

总候选 383,258 条。沿用按 split/task/predicate 平衡 Yes/No 的规则，将 `max_per_label` 设为 1,000,000，实际没有触及上限；不沿用 mini 每桶 256 的工程限额。距离题是 378 个语义事件的四种选项排列，分别为 train199/dev97/test82，不能视为 1,512 个独立几何事件。

12 个 split/方位桶没有正例：dev 的 back/back-left/front/front-left/left，test 的 back-left/front/front-left/left，train 的 back-left/front-left/left。其候选计数记录在 manifest 的 shortages 中，未用单一负标签伪造平衡集合。右后裁剪输入不能支撑完整八方位均衡能力结论。

## 4. 已执行的审计

- 从原始姿态、标定、标注及实例身份，使用独立逆齐次变换复算全部 83,952 题的 122,990 个数值，通过；34,149 帧的姿态和时间戳全部核对。
- 类别、任务、谓词、答案、逐场景题数及稀缺桶记录在 `coverage.json`。严格配对数使用同 prompt/长度、异场景、相反答案的精确最大容量；其计数公式通过 80 组随机图与显式最大匹配对照。
- zero/repeat_first/shuffle 三种控制在 850 场景实际改变特征且保留 prompt/时间/位姿；cross_scene 在 849 场景可用。train 有 59 题无同长度供体，dev/test 全部有供体；不可用项保留在分母。
- 全量准备及审计进程峰值 RSS 4,492.84 MiB；当前服务器内存限额 128 GiB。原先 mini 的 2 GiB 环境约束不是本次现场资源额度。
- 最终远端 41 项核心检查全部通过。本地 41 项运行通过，其中 2 项因本地未装 nuScenes/pyquaternion 而跳过；这两项已在远端执行通过。
- 原 B3 三组件保持原身份，连同 SST 第四组件实时哈希核验通过。SST 清单与数据绑定检查、数据生成器代码身份核验以及正式训练准入负向检查通过。

## 5. 位置与复用

- 远端运行根：`/root/autodl-tmp/mmb4dl-core-trainval-20261023-r1`，完整数据为 `trainval-v1`，最终运行源码为 `source-r4`。
- 本地完整副本：`D:/Github/mmb4dl/training_logs/core_geometry_20261022/trainval-data/trainval-v1`。原始支持记录与来源回执在同级 `trainval-data` 目录。
- 本地紧凑回执：`training_logs/core_geometry_20261022/trainval-evidence`；本地全量哈希验证为 `trainval-local-copy-verification.json`。全部 JSONL 与 850 个场景包已本地重读核验，生成器源码与本地一致。
- 最终门控及部署回执：`training_logs/core_geometry_20261022/trainval-final-evidence/readiness.json`；46 个部署文件与本地一致，实现 SHA256 为 `995a647f9d290a3175cb01e63cbb83abcd39a940d0d3ceed10cddd914d7ca30a`。
- 全量数据归档 160,663,969 bytes，SHA256：`d9843fec7219a584155c63635294382f728767800d7ad398594a3a697572ab56`。
- 新配置：`mllm/configs/core_geometry_baseline_original_b3_trainval.json` 绑定相同 B3 三组件与 SST 来源；`core_geometry_preparation.json` 指向完整数据，仅保留预算/种子、六任务回归协议两个门控。

旧 `core_geometry_baseline_original_b3.json` 是接入工程 checkpoint 的历史身份清单，保持不改。新的来源绑定清单具有新的 SHA256；旧工程 checkpoint 恢复仍应使用原清单和对应实现快照，不跨版本绕过身份检查。

复跑数据准备时先新建独立根目录，依次使用 `python -m b4dl_core.sst_source inventory/support/reproduce` 和 `python -m b4dl_core.trainval`。参数完整记录在源文件 CLI 与运行日志中；支持记录、来源回执及 split 文件的绝对路径参与当前数据 ID，移动后复算 ID 时应作为新版本处理，不修改现有 manifest。

后续已冻结 A0–A3 的新 LoRA rank、参数预算、步数/种子与原六任务协议，并补齐实际指标运行依赖，详见[首轮训练说明](core_geometry_training.md)。此次来源核验和数据审计不构成模型效果结果。
