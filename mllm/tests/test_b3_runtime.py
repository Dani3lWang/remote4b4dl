import os
import subprocess
import unittest
from pathlib import Path

MLLM_ROOT = Path(__file__).resolve().parents[1]
RUNTIME = MLLM_ROOT / "scripts" / "b3_runtime.sh"


class B3RuntimeTests(unittest.TestCase):
    def run_profile(self, profile, script, **extra):
        env = {key: value for key, value in os.environ.items() if not key.startswith("B4DL_")}
        env.update(B4DL_TRAIN_PROFILE=profile, **extra)
        return subprocess.run(
            ["bash", "-c", 'source "$1"; b3_configure_runtime || exit 1; ' + script,
             "test", str(RUNTIME)], env=env, capture_output=True, text=True,
        )

    def test_4090_preserves_effective_batch_and_uses_separate_outputs(self):
        result = self.run_profile("rtx4090", 'printf "%s %s %s %s" "$B3_MICRO_BATCH" "$B3_GRAD_ACCUM" "$B3_ZERO_CONFIG" "$B3_OUTPUT_SUFFIX"')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "1 128 ./scripts/zero3_offload.json -rtx4090")
        baseline = self.run_profile("baseline", 'echo "$((B3_MICRO_BATCH * B3_GRAD_ACCUM))"')
        self.assertEqual(baseline.stdout.strip(), "128")

    def test_impossible_gate_fails_before_waiting(self):
        result = self.run_profile("baseline", 'nvidia-smi() { echo 24564; }; b3_check_gpu_capacity "$B3_MIN_FREE_MB"')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("不能等待满足", result.stderr)
        self.assertIn("rtx4090", result.stderr)

    def test_gpu_query_targets_selected_device(self):
        result = self.run_profile("rtx4090", 'nvidia-smi() { printf "%s\\n" "$*" >&2; echo 24564; }; b3_check_gpu_capacity "$B3_MIN_FREE_MB"', B4DL_GPU_ID="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--id=1", result.stderr)

    def test_invalid_profile_or_gpu_is_rejected(self):
        self.assertNotEqual(self.run_profile("bad", ':').returncode, 0)
        self.assertNotEqual(self.run_profile("rtx4090", ':', B4DL_GPU_ID="-1").returncode, 0)


if __name__ == "__main__":
    unittest.main()
