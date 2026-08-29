"""LiDAR-CLIP nuScenes 编码器训练自动停止监控（用户授权规则，2026-08-29）

规则：loss 连续 500 步降幅 < 1% → 停止训练。
数据源：last.ckpt 内 ModelCheckpoint state 的 current_score（每 250 步保存时的 train_loss）
       + 文件名/ckpt global_step 作为进度真值（wandb offline 历史已确认不可用——0 行）。
保护：① 下限 step≥5,275（1 epoch）才允许判停；② 需连续 2 次检查（间隔 10 min）满足才停，
      且用 4 点均值抗单步噪声；③ step≥26,375（5 epoch）硬性保底停止。
停止动作：SIGTERM 训练进程树 → 最多等 120s → 残留 SIGKILL → 校验 last.ckpt 与 GPU 释放。
"""
import glob
import json
import os
import re
import signal
import subprocess
import time

import torch

CKPT_DIR = "/root/autodl-tmp/wql/mmb4dl/encoders/lidarclip/ckpt_nuscenes/lidarclip_mm"
LOG = "/root/autodl-tmp/wql/mmb4dl/encoders/lidarclip/logs/early_stop_monitor.log"
STATE = "/root/autodl-tmp/wql/mmb4dl/encoders/lidarclip/logs/early_stop_state.json"
INTERVAL = 600          # 检查间隔（秒）
WINDOW = 500            # 规则窗口（步）
THRESH = 0.01           # 降幅阈值 1%
MIN_STEP = 5275         # 允许判停的下限（1 epoch）
HARD_CAP = 26375        # 硬性保底（5 epoch）
TRAIN_PAT = "wqlc/bin/python train.py --name lidarclip_nuscenes"


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def find_train_pids():
    r = subprocess.run(["pgrep", "-f", TRAIN_PAT], capture_output=True, text=True)
    return [int(p) for p in r.stdout.split()]


def read_last_ckpt():
    """返回 (global_step, epoch, train_loss or None)；ckpt 加载失败返回 None"""
    path = os.path.join(CKPT_DIR, "last.ckpt")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        log(f"WARN last.ckpt 加载失败（可能正在写入，下轮重试）: {e}")
        return None
    step = int(ckpt.get("global_step", -1))
    epoch = int(ckpt.get("epoch", -1))
    loss = None
    for k, v in ckpt.get("callbacks", {}).items():
        if "ModelCheckpoint" in k and isinstance(v, dict):
            cs = v.get("current_score")
            if cs is not None:
                loss = float(cs)
            break
    del ckpt
    return step, epoch, loss


def max_ckpt_step():
    steps = []
    for p in glob.glob(os.path.join(CKPT_DIR, "epoch=*-step=*.ckpt")):
        m = re.search(r"step=(\d+)", p)
        if m:
            steps.append(int(m.group(1)))
    return max(steps) if steps else -1


def stop_training(reason, points):
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
        alive = find_train_pids()
        if not alive:
            break
        time.sleep(5)
    for p in find_train_pids():
        log(f"SIGTERM 120s 未退出，SIGKILL {p}")
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(10)
    alive = find_train_pids()
    log(f"停止后残留进程: {alive if alive else '无'}")
    last_step, last_epoch, last_loss = (None, None, None)
    try:
        s, e, l = read_last_ckpt()
        last_step, last_epoch, last_loss = s, e, l
        log(f"最终 ckpt: step={s} epoch={e} loss={l}")
    except Exception as e:
        log(f"WARN 最终 ckpt 读取失败: {e}")
    state = {
        "stopped_by_monitor": True,
        "reason": reason,
        "stopped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "final_step": last_step,
        "final_epoch": last_epoch,
        "final_loss": last_loss,
        "loss_points": points[-20:],
    }
    with open(STATE, "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)
    log(f"状态已写入 {STATE}；监控退出。后续（重提特征/重训）由下一个会话接手。")


def main():
    log(f"监控启动 规则: {WINDOW} 步降幅<{THRESH:.0%} 下限step={MIN_STEP} 保底step={HARD_CAP} 间隔={INTERVAL}s")
    points = []          # [(step, loss)]
    consecutive = 0
    while True:
        time.sleep(INTERVAL)
        pids = find_train_pids()
        if not pids:
            if os.path.exists(STATE):
                log("训练已由本监控停止，退出。")
            else:
                log("训练进程消失且非本监控所为（疑似崩溃或外部终止）——记录状态后退出。")
                with open(STATE, "w") as f:
                    json.dump({"stopped_by_monitor": False,
                               "reason": "training process vanished",
                               "stopped_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=1)
            return
        r = read_last_ckpt()
        if r is None:
            continue
        step, epoch, loss = r
        if loss is None:
            log(f"step={step} epoch={epoch} loss 缺失，跳过")
            continue
        if not points or step > points[-1][0]:
            points.append((step, loss))
        if len(points) < 4:
            log(f"step={step} epoch={epoch} loss={loss:.4f}（累计 {len(points)} 点，未满 4 点）")
            continue
        cur = sum(l for _, l in points[-4:]) / 4.0
        ref_pts = [(s, l) for s, l in points if WINDOW <= (step - s) <= WINDOW + 1200]
        if not ref_pts:
            log(f"step={step} loss={loss:.4f} 暂无 ≥{WINDOW} 步前的参照点（点数 {len(points)}）")
            continue
        ref = sum(l for _, l in ref_pts) / len(ref_pts)
        improvement = (ref - cur) / ref
        log(f"step={step} epoch={epoch} cur(4点均值)={cur:.5f} ref({len(ref_pts)}点@-{WINDOW}+步)={ref:.5f} 降幅={improvement:.2%}")
        if step >= HARD_CAP:
            stop_training(f"硬性保底 step≥{HARD_CAP}", points)
            return
        if step < MIN_STEP:
            continue
        if improvement < THRESH:
            consecutive += 1
            log(f"  满足判停条件（连续 {consecutive}/2）")
            if consecutive >= 2:
                stop_training(f"loss 连续 {WINDOW} 步降幅 <1%（连续 2 次检查确认）", points)
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
