"""One first-layer attention pilot followed by a gated paired full-mini study."""
import argparse
from pathlib import Path

from .study import SpatialStudy, gate, read, utc


class FirstAttentionStudy(SpatialStudy):
    def __init__(self, args):
        super().__init__(args)
        self.state.update(study_type="first_layer_attention_control",
                          budgets=dict(pilot_epochs=50, pilot_optimizer_steps=400, pilot_runs=1,
                                       expanded_epochs_per_arm=5, expanded_optimizer_steps_per_arm=405),
                          changed_factor="first_decoder_layer_hard_attention_exclusions_only",
                          unchanged=dict(mask_bce_mode="mean", memory_points=4096, seed=2026, learning_rate=1e-4))
        self.save()

    def run(self):
        try:
            previous = read(self.args.control_study / "study_state.json")
            if previous["status"] != "complete" or self.base_config["memory_points"] != 4096:
                raise ValueError("attention study requires the completed 4096-point mean control")
            for field in ("initial_checkpoint_sha256", "subset_manifest_sha256", "full_train_manifest_sha256", "config"):
                if self.state["input_identity"][field] != previous["input_identity"][field]:
                    raise ValueError("attention/control input identity differs: " + field)
            reference = previous["pilot"]["mean"]
            control_checkpoint = Path(reference["identity"]["checkpoint"])
            # Verify the default behavior with the actual legacy checkpoint before training.
            control = self.evaluate("masked_control", control_checkpoint, subset=True)
            if control["metrics"] != reference["metrics"]:
                raise ValueError("default attention failed exact legacy metric compatibility")
            control.update(training_reused_from=str(self.args.control_study / "mean/training"),
                           training_executed_this_study=False, compatibility_metrics_exact=True)
            self.state["pilot"]["masked_control"] = control
            self.save()
            candidate_checkpoint = self.train("first_full", "mean", 4096, True, first_layer_attention="full")
            candidate = self.evaluate("first_full", candidate_checkpoint, subset=True)
            candidate["comparison_to_masked"] = gate(control["metrics"], candidate["metrics"])
            self.state["pilot"]["first_full"] = candidate
            self.state["selected_recipe"] = "first_full" if candidate["comparison_to_masked"]["passed"] else None
            self.save()
            if self.state["selected_recipe"]:
                for name, mode in (("full_masked", "masked"), ("full_first_full", "full")):
                    checkpoint = self.train(name, "mean", 4096, False, first_layer_attention=mode)
                    self.state["expanded"][name] = dict(train=self.evaluate(name + "_train", checkpoint),
                                                       internal_dev=self.evaluate(name + "_dev", checkpoint, dev=True))
                    self.save()
                self.state["conclusion"] = "attention_pilot_screen_passed; paired_full_mini_completed; independent_acceptance_still_required"
            else:
                self.state["conclusion"] = "attention_pilot_screen_failed; full_mini_not_started; inspect_encoder_next"
            self.state.update(status="complete", completed_utc=utc())
            self.save()
        except Exception as error:
            self.state.update(status="failed", failed_utc=utc(), error=str(error))
            self.save()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study-dir", "control-study", "config", "subset-root", "label-root", "nuscenes-root", "panoptic-root", "init-from"):
        parser.add_argument("--" + name, required=True, type=Path)
    FirstAttentionStudy(parser.parse_args()).run()


if __name__ == "__main__":
    main()
