import json
import unittest
from copy import deepcopy
from pathlib import Path

from helpers.pipeline import run_pipeline


class PipelineTests(unittest.TestCase):
    def configs(self):
        root = Path(__file__).resolve().parents[1] / "configs"
        for name in ("training2d.json", "training3d.json", "training3dnormal.json"):
            yield json.loads((root / name).read_text(encoding="utf-8"))

    def run_stages(self, config):
        calls = []

        def record(stage):
            def record_call(current, include_training=True):
                calls.append((stage, deepcopy(current)))
            return record_call

        run_pipeline(
            config,
            train=record("train"),
            predict=record("predict"),
            evaluate=record("evaluate"),
            plot=record("plot"),
        )
        return calls

    def test_all_pipelines_keep_validation_and_test_separate(self):
        for config in self.configs():
            config["pipeline"] = dict(train=True, predict=True, evaluate=True, test=True, plot=True)
            with self.subTest(run=config["run_id"]):
                original = deepcopy(config)
                calls = self.run_stages(config)
                self.assertEqual([stage for stage, _ in calls],
                                 ["train", "predict", "evaluate", "predict", "evaluate", "plot", "plot"])
                self.assertEqual(config, original)
                for index in (0, 1, 2, 5):
                    self.assertEqual(calls[index][1], original)
                for index in (3, 4, 6):
                    current = calls[index][1]
                    self.assertEqual(current["run_id"], config["run_id"])
                    self.assertEqual(current["inference"]["partition"], "test")
                    self.assertEqual(current["inference"]["checkpoint"], "best")
                    self.assertEqual(current["inference"]["prediction_id"], config["testing"]["prediction_id"])
                    self.assertEqual(current["graphs"]["figure_id"], config["testing"]["figure_id"])

    def test_test_only_does_not_train(self):
        config = next(self.configs())
        config["pipeline"] = dict(train=False, predict=False, evaluate=False, test=True, plot=False)
        calls = self.run_stages(config)
        self.assertEqual([stage for stage, _ in calls], ["predict", "evaluate"])
        self.assertTrue(all(current["inference"]["partition"] == "test" for _, current in calls))

    def test_old_configs_and_disabled_test_keep_original_sequence(self):
        for enabled in (None, False):
            config = next(self.configs())
            if enabled is None:
                del config["pipeline"]["test"]
            else:
                config["pipeline"]["test"] = enabled
            self.assertEqual([stage for stage, _ in self.run_stages(config)],
                             ["train", "predict", "evaluate", "plot"])

if __name__ == "__main__":
    unittest.main()
