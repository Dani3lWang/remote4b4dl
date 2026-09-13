"""Stage gate for scripts/run_grpo_tg.sh: validate the latest grpo_log.jsonl.

Kept as a module (not an inline `python -c`) so the chain script stays free of
nested-quote escaping and the checks are independently runnable.

    python -m vtimellm.rl.check_gate checkpoints/grpo_tg_m31/grpo_log.jsonl m31
"""

import argparse
import json
import math


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('stage', choices=['m31', 'm32', 'full'])
    a = ap.parse_args()

    with open(a.log, encoding='utf-8') as fh:
        recs = [json.loads(l) for l in fh if l.strip()]
    if not recs:
        print('  gate FAIL: no log records')
        return 1
    r = recs[-1]

    def bad(x):
        return x is None or (isinstance(x, float) and math.isnan(x))

    if a.stage == 'm31':
        # Mechanical correctness of one update: gradient flowed (grad_norm>0),
        # path B is self-consistent in-trainer (ratio_dev~0 => old==actor at init,
        # the G2.4c invariant), and the fp32 master actually moved (lora_delta>0,
        # which pure-bf16 at lr=1e-5 could not -- it would round to zero).
        ok = (r.get('grad_norm', 0) > 0
              and abs(r.get('ratio_dev', 9)) < 1e-3
              and r.get('lora_delta', 0) > 0
              and not bad(r.get('loss')))
        print(f"  m31: grad_norm={r.get('grad_norm')} ratio_dev={r.get('ratio_dev')} "
              f"lora_delta={r.get('lora_delta')} loss={r.get('loss')} "
              f"peak={r.get('peak_gib')}G -> {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1

    if a.stage == 'm32':
        n = len(recs)
        losses = [x.get('loss') for x in recs if x.get('loss') is not None]
        nan = any(bad(l) for l in losses)
        rw = [x.get('reward_mean', 0.0) for x in recs]
        degen = [x.get('degenerate_rate', 0.0) for x in recs]
        ok = n >= 100 and not nan
        print(f"  m32: {n} steps, NaN={nan}, reward {rw[0]:.3f}->{rw[-1]:.3f}, "
              f"degenerate_rate last={degen[-1]:.3f}, center_std last="
              f"{r.get('center_std')} -> {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1

    print(f"  full: {len(recs)} steps logged, last step={r.get('step')}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
