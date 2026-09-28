# 选项 A 三臂串行编排的**权威副本**。正在跑的那份是它的拷贝，放在 gitignore 的
# mllm/training_logs/temporal/ 下；没有这份，A 程序在仓库里不可复现。
# 预注册与判据见 docs/learn docs/B4DL_ReasonSeg多帧A_预注册_20260927.md。
#
# 两道保守闸：(1) 等 A2 的 CHAIN DONE，超时不启动；(2) 等 a2_verdict.txt 落盘
# （它由另一个 watcher 写），拿不到就停止 —— 绝不把"文件不存在"当成"失效门通过"。
#!/bin/bash
# A2 -> A1 -> A3 串行。失效门：a2_verdict.txt 里那句预言若不成立，就不启动后两臂。
cd /root/autodl-tmp/mmb4dl/mllm || exit 1
L=./training_logs/temporal
D=$L/a2_repeat_driver.log
echo "[$(date '+%F %T')] 链启动，等 A2 收尾" | tee -a "$L/chain_driver.log"
for i in $(seq 1 500); do
  grep -q 'CHAIN DONE' "$D" 2>/dev/null && break
  sleep 60
done
grep -q 'CHAIN DONE' "$D" 2>/dev/null || { echo "[$(date '+%F %T')] 等 A2 超时，不启动后两臂" | tee -a "$L/chain_driver.log"; exit 1; }
# verdict 由另一个 watcher 写，可能还没落盘：最多再等 10 分钟。
# 拿不到就保守停止——绝不把“文件不存在”当成“失效门通过”。
for j in $(seq 1 10); do
  [ -s "$L/a2_verdict.txt" ] && grep -q '^DONE$' "$L/a2_verdict.txt" && break
  sleep 60
done
if ! grep -q '^DONE$' "$L/a2_verdict.txt" 2>/dev/null; then
  echo "[$(date '+%F %T')] 拿不到 a2_verdict => 保守起见，不启动后两臂" | tee -a "$L/chain_driver.log"
  exit 1
fi
if grep -q '不成立' "$L/a2_verdict.txt"; then
  echo "[$(date '+%F %T')] 失效门触发（a2 test AUC 掉出单帧带）=> 停止，不跑 A1/A3" | tee -a "$L/chain_driver.log"
  exit 1
fi
for arm in a1_compensated a3_naive; do
  echo "[$(date '+%F %T')] 启动 $arm" | tee -a "$L/chain_driver.log"
  bash scripts/reasonseg_experiments/run_temporal_arm.sh "$arm" 3 20 >> "$L/${arm}_chain.log" 2>&1
  echo "[$(date '+%F %T')] $arm rc=$?" | tee -a "$L/chain_driver.log"
done
echo "[$(date '+%F %T')] === 三臂链全部结束 ===" | tee -a "$L/chain_driver.log"
