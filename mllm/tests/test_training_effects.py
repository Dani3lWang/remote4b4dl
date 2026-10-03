import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vtimellm"))
from training_effects import TrainingHistory
from effect_visualizer import training_history_figure


class TrainingHistoryTests(unittest.TestCase):
    def test_resume_logs_and_missing_values_produce_aligned_curves(self):
        state = {"global_step": 20, "epoch": 0.5, "log_history": [
            {"step": 10, "loss": 2.0, "learning_rate": 1e-4},
            {"step": 20, "loss": 1.0, "learning_rate": 5e-5},
            {"step": 10, "loss": 1.5},
            {"step": 20, "eval_loss": 1.2},
            {"step": 30, "loss": float("nan")},
            {"loss": 9.0},
        ]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trainer_state.json"
            path.write_text(json.dumps(state))
            history = TrainingHistory.from_file(path)
        self.assertEqual(history.series("loss"), ((10, 1.5), (20, 1.0)))
        figure = training_history_figure(history)
        self.assertEqual(list(figure.data[0].x), [10, 20])
        self.assertEqual(list(figure.data[1].y), [1.2])
        self.assertEqual(list(figure.data[2].y), [1e-4, 5e-5])

    def test_invalid_history_fails_with_actionable_message(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trainer_state.json"
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, "log_history"):
                TrainingHistory.from_file(path)


if __name__ == "__main__":
    unittest.main()
