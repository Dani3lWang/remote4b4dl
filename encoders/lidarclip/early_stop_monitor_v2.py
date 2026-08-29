"""LiDAR-CLIP nuScenes 编码器续训自动停止监控 v2（2026-08-29）

与 v1 规则一致（用户授权）：loss 连续 500 步降幅 < 1% → 停止。
v2 修正：数据源改为 train_loss.csv（LossCSVCallback 每 50 步写入的真值），
不再使用 ckpt 的 current_score（只在刷新最优时更新，存在滞后假象）。
范围：step < 26,375（再训满 1 epoch）不判停；step ≥ 31,550（+2 epoch）硬性保底。
"""
import json
import os
import signal
import subprocess
import time

CSV = "/root/autodl-tmp/wql/mmb4dl/encoders/lidarclip/logs/train_loss.csv"
LOG = "/root/autodl-tmp/wql/mmb4dl/encoders/lidarclip/logs/early_stop_monitor_v2.log"
STATE = "/root/autodl-tmp/wql/mmb4dl/encoders/lidarclip/logs/early_stop_state_v2.json"
INTERVAL = 600
WINDOW = 500
THRESH = 0.01
FLOOR = 26375          # +1 epoch（自 21,000 起）
HARD_CAP = 31550       # +2 epoch
TRAIN_PAT = "wqlc/bin/python train.py --name lidarclip_nuscenes"


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def find_train_pids():
    r = subprocess.run(["pgrep", "-f", TRAIN_PAT], capture_output=True, text=True)
    return [int(p) for p in r.stdout.split()]


def read_csv_tail(n=60):
    """返回 [(step, loss)] 最近 n 行"""
    try:
        with open(CSV) as f:
            rows = [ln.strip().split(",") for ln in f if ln.strip()]
        return [(int(s), float(l)) for s, l in rows[-n:]]
    except FileNotFoundError:
        return []


def mean(xs):
    return sum(xs) / len(xs)


def stop_training(reason):
    log(f"=== 触发停止：{reason} ===")
    pids = find_train_pids()
    log(f"训练进程: {pids}")
    for p in pids:
        try:
            os.kill(p, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + 120
    while time.time() < deadline:
        if not find_train_pids():
            break
        time.sleep(5)
    for p in find_train_pids():
        log(f"SIGKILL 残留 {p}")
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(10)
    log(f"停止后残留: {find_train_pids() or '无'}")
    tail = read_csv_tail(10)
    with open(STATE, "w") as f:
        json.dump({"stopped_by_monitor": True, "reason": reason,
                   "stopped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "last_csv_rows": tail}, f, ensure_ascii=False, indent=1)
    log(f"状态已写入 {STATE}；监控退出。下一步（重提特征）由用户指示后进行。")


def main():
    log(f"监控v2启动 规则: {WINDOW}步降幅<{THRESH:.0%} floor={FLOOR} hard_cap={HARD_CAP} 间隔={INTERVAL}s 数据源={os.path.basename(CSV)}")
    consecutive = 0
    while True:
        time.sleep(INTERVAL)
        if not find_train_pids():
            if os.path.exists(STATE):
                log("训练已停止，退出。")
            else:
                log("训练进程消失（疑似崩溃/外部终止），记录后退出。")
                with open(STATE, "w") as f:
                    json.dump({"stopped_by_monitor": False, "reason": "vanished",
                               "stopped_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
            return
        rows = read_csv_tail(80)
        if len(rows) < 16:
            log(f"CSV 点不足（{len(rows)} 行），等待")
            continue
        cur_step = rows[-1][0]
        cur = mean([l for _, l in rows[-5:]])                      # 最近 ~250 步均值
        ref = [l for s, l in rows if cur_step - WINDOW - 250 <= s <= cur_step - WINDOW]
        if len(ref) < 3:
            log(f"step={cur_step} 参照窗不足，等待")
            continue
        refm = mean(ref)
        improvement = (refm - cur) / refm
        log(f"step={cur_step} cur(250步均值)={cur:.5f} ref(-500步)={refm:.5f} 降幅={improvement:.2%}")
        if cur_step >= HARD_CAP:
            stop_training(f"硬性保底 step≥{HARD_CAP}")
            return
        if cur_step < FLOOR:
            log("  未达判停下限，继续")
            continue
        if improvement < THRESH:
            consecutive += 1
            log(f"  满足判停条件（连续 {consecutive}/2）")
            if consecutive >= 2:
                stop_training(f"loss 连续 {WINDOW} 步降幅 <1%（CSV 真值，连续 2 次确认）")
                return
        else:
            if consecutive:
                log("  条件中断，计数清零")
            consecutive = 0


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        log(f"监控异常退出: {e}\n{traceback.format_exc()}")
