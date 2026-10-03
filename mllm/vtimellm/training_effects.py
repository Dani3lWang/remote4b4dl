"""Read Trainer history without importing transformers or touching checkpoints."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Tuple


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


@dataclass(frozen=True)
class TrainingHistory:
    source: str
    global_step: int
    epoch: float | None
    logs: Tuple[Mapping, ...]

    @classmethod
    def from_file(cls, path):
        source = Path(path).expanduser()
        with source.open(encoding="utf-8") as handle:
            state = json.load(handle)
        if not isinstance(state, Mapping) or not isinstance(state.get("log_history"), list):
            raise ValueError("trainer_state.json 必须包含 log_history 数组")
        logs = tuple(
            dict(row) for row in state["log_history"]
            if isinstance(row, Mapping) and finite_number(row.get("step")) is not None
            and row["step"] >= 0
        )
        global_step = finite_number(state.get("global_step"))
        if global_step is None or global_step < 0:
            raise ValueError("trainer_state.global_step 必须是非负有限数字")
        return cls(str(source), int(global_step), finite_number(state.get("epoch")), logs)

    def series(self, metric):
        # Resume can write the same step twice; latest record wins per metric.
        by_step = {
            row["step"]: finite_number(row.get(metric))
            for row in self.logs if finite_number(row.get(metric)) is not None
        }
        return tuple(sorted(by_step.items()))
