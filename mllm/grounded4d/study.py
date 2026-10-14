"""Bounded, sequential spatial experiments with fixed data and an expansion gate."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from .train import atomic_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def utc():
    return datetime.now(timezone.utc).isoformat()


def gate(control, candidate):
    if control["gt_instances"] != candidate["gt_instances"]:
        raise ValueError("cannot compare different GT scopes")
    return dict(passed=(candidate["recall_iou50"] >= control["recall_iou50"] + 0.05
                        and candidate["mean_gt_mask_iou"] >= control["mean_gt_mask_iou"] + 0.01
                        and candidate["f1_iou50"] >= control["f1_iou50"]),
                recall_gain=candidate["recall_iou50"] - control["recall_iou50"],
                mean_gt_iou_gain=candidate["mean_gt_mask_iou"] - control["mean_gt_mask_iou"],
                f1_gain=candidate["f1_iou50"] - control["f1_iou50"],
                definition="pilot_screen: recall_gain>=0.05 AND mean_GT_IoU_gain>=0.01 AND F1_nonregression; not final acceptance")


class SpatialStudy:
    def __init__(self, args):
        self.args = args
        self.root = args.study_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "commands").mkdir()
        self.repo = Path(__file__).resolve().parents[2]
        self.base_config = read(args.config)
        self.state = dict(status="running", started_utc=utc(), steps=[], pilot={}, expanded={},
                          input_identity=dict(initial_checkpoint_sha256=hashlib.sha256(args.init_from.read_bytes()).hexdigest(),
                            subset_manifest_sha256=hashlib.sha256((args.subset_root / "train.jsonl").read_bytes()).hexdigest(),
                            full_train_manifest_sha256=hashlib.sha256((args.label_root / "train.jsonl").read_bytes()).hexdigest(),
                            config=self.base_config),
                          thresholds=dict(score=0.25, mask=0.5, iou=0.5),
                          budgets=dict(pilot_epochs=50, pilot_optimizer_steps=400, full_epochs=5),
                          model_performance_accepted=False, final_generalization_acceptance=False,
                          expansion_gate=gate(dict(gt_instances=255, recall_iou50=0., mean_gt_mask_iou=0., f1_iou50=0.),
                                              dict(gt_instances=255, recall_iou50=0., mean_gt_mask_iou=0., f1_iou50=0.))["definition"])
        self.save()

    def save(self):
        atomic_json(self.root / "study_state.json", self.state)

    def command(self, name, command):
        step = dict(name=name, command=[str(value) for value in command], started_utc=utc(), status="running")
        self.state["steps"].append(step)
        self.save()
        print(json.dumps(dict(event="study_step_start", **step)), flush=True)
        with (self.root / "commands" / (name + ".log")).open("w", encoding="utf-8") as handle:
            result = subprocess.run(step["command"], cwd=self.repo, stdout=handle, stderr=subprocess.STDOUT)
        step.update(completed_utc=utc(), exit_code=result.returncode, status="complete" if result.returncode == 0 else "failed")
        self.save()
        if result.returncode:
            raise RuntimeError(f"{name} failed: exit={result.returncode}; inspect its command log")

    def common_data(self, subset=False):
        a = self.args
        source = a.subset_root if subset else a.label_root
        return ["--nuscenes-root", a.nuscenes_root, "--label-root", a.label_root,
                "--panoptic-root", a.panoptic_root, "--train-scene-tokens", source / "train_scene_tokens.json",
                "--min-target-points", "1"]

    def evaluate(self, name, checkpoint, subset=False, dev=False):
        a = self.args
        source = a.subset_root if subset else a.label_root
        output = self.root / name / "evaluation"
        command = [sys.executable, "-u", self.repo / "mllm/scripts/evaluate_grounded4d.py", *self.common_data(subset),
                   "--checkpoint", checkpoint, "--split", "internal_dev" if dev else "train_probe",
                   "--manifest", source / ("internal_dev.jsonl" if dev else "train.jsonl"),
                   "--dev-scene-tokens", source / "dev_scene_tokens.json", "--output-dir", output,
                   "--score-threshold", ".25", "--mask-threshold", ".5", "--iou-threshold", ".5", "--query-diagnostics"]
        self.command(name + "-evaluate", command)
        result = read(output / "metrics.json")
        if not result["evaluation_complete"] or result["evaluated_frames"] != (8 if subset else 81 if dev else 323):
            raise ValueError("unexpected or incomplete evaluation scope")
        return dict(metrics=result["metrics"], identity=result["identity"], evaluation_dir=str(output),
                    query_diagnostics=read(output / "query_diagnostics.json"))

    def train(self, name, bce_mode, memory_points, subset):
        a = self.args
        source = a.subset_root if subset else a.label_root
        config = dict(self.base_config, memory_points=memory_points)
        config_path = self.root / (name + "-config.json")
        atomic_json(config_path, config)
        output = self.root / name / "training"
        epochs, accumulation = (50, 1) if subset else (5, 4)
        command = [sys.executable, "-u", self.repo / "mllm/scripts/train_grounded4d.py", *self.common_data(subset),
                   "--train-manifest", source / "train.jsonl", "--stage", "spatial", "--supervision", "panoptic",
                   "--config", config_path, "--output-dir", output, "--init-from", a.init_from,
                   "--epochs", str(epochs), "--gradient-accumulation", str(accumulation),
                   "--learning-rate", "0.0001", "--seed", "2026", "--log-every", "8",
                   "--mask-bce-mode", bce_mode]
        if memory_points != self.base_config["memory_points"]:
            command.append("--allow-memory-points-change")
        self.command(name + "-preflight", [*command, "--preflight-only"])
        self.command(name + "-train", command)
        progress = read(output / "progress.json")
        if progress["completed_epoch"] != epochs or progress["global_step"] != (400 if subset else 405):
            raise ValueError("training progress differs from fixed experiment budget")
        return output / "last.pt"

    def run(self):
        try:
            self.state["initial_prediction"] = self.evaluate("initial", self.args.init_from, subset=True)
            self.save()
            control_checkpoint = self.train("mean", "mean", self.base_config["memory_points"], True)
            control = self.evaluate("mean", control_checkpoint, subset=True)
            self.state["pilot"]["mean"] = control
            self.save()
            balanced_checkpoint = self.train("balanced", "foreground_balanced", self.base_config["memory_points"], True)
            balanced = self.evaluate("balanced", balanced_checkpoint, subset=True)
            balanced["comparison_to_mean"] = gate(control["metrics"], balanced["metrics"])
            self.state["pilot"]["balanced"] = balanced
            selected, points = ("balanced", self.base_config["memory_points"]) if balanced["comparison_to_mean"]["passed"] else (None, None)
            self.save()
            if selected is None:
                memory_checkpoint = self.train("balanced_memory8192", "foreground_balanced", 8192, True)
                memory = self.evaluate("balanced_memory8192", memory_checkpoint, subset=True)
                memory["comparison_to_mean"] = gate(control["metrics"], memory["metrics"])
                memory["comparison_to_balanced"] = gate(balanced["metrics"], memory["metrics"])
                self.state["pilot"]["balanced_memory8192"] = memory
                if memory["comparison_to_mean"]["passed"]:
                    selected, points = "balanced_memory8192", 8192
                self.save()
            self.state["selected_recipe"] = selected
            if selected:
                full_control = self.train("full_mean", "mean", self.base_config["memory_points"], False)
                full_candidate = self.train("full_" + selected, "foreground_balanced", points, False)
                for name, checkpoint in (("full_mean", full_control), ("full_" + selected, full_candidate)):
                    self.state["expanded"][name] = dict(train=self.evaluate(name + "_train", checkpoint),
                                                       internal_dev=self.evaluate(name + "_dev", checkpoint, dev=True))
                    self.save()
                self.state["conclusion"] = "pilot_screen_passed; paired_full_mini_results_require_separate_review; not final acceptance"
            else:
                self.state["conclusion"] = "pilot_screen_failed; no expanded training; no temporal or language training"
            self.state.update(status="complete", completed_utc=utc())
            self.save()
            print(json.dumps(dict(event="study_complete", selected_recipe=selected, conclusion=self.state["conclusion"])), flush=True)
        except Exception as error:
            self.state.update(status="failed", failed_utc=utc(), error=str(error))
            self.save()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study-dir", "config", "subset-root", "label-root", "nuscenes-root", "panoptic-root", "init-from"):
        parser.add_argument("--" + name, required=True, type=Path)
    SpatialStudy(parser.parse_args()).run()


if __name__ == "__main__":
    main()
