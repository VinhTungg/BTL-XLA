import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from tacs.losses import TACSLoss
from tacs.model import TACSModel
from tacs.train import ExperimentConfig, evaluate, parse_args, run_experiment, train_one_epoch


class TACSLoopTest(unittest.TestCase):
    def test_train_and_evaluate_support_tacs_outputs_and_loss(self) -> None:
        torch.manual_seed(7)
        model = TACSModel(num_classes=4, embedding_dim=8)
        records = [
            {
                "query": torch.randn(3, 24, 24),
                "candidates": torch.randn(4, 3, 24, 24),
                "target": torch.tensor(index % 4),
            }
            for index in range(4)
        ]
        loader = DataLoader(records, batch_size=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        criterion = TACSLoss()

        train_result = train_one_epoch(model, loader, optimizer, criterion, torch.device("cpu"))
        eval_result = evaluate(model, loader, criterion, torch.device("cpu"))

        self.assertGreaterEqual(train_result["accuracy"], 0.0)
        self.assertLessEqual(train_result["accuracy"], 1.0)
        self.assertGreaterEqual(eval_result["accuracy"], 0.0)
        self.assertLessEqual(eval_result["accuracy"], 1.0)
        self.assertGreater(train_result["loss"], 0)
        self.assertGreater(eval_result["loss"], 0)
        for result in (train_result, eval_result):
            self.assertGreater(result["throughput"], 0)

    def test_tacs_and_two_stage_dry_runs_save_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            for baseline in ("tacs", "two_stage"):
                with self.subTest(baseline=baseline):
                    output_dir = Path(tmp_dir) / baseline
                    config = ExperimentConfig(
                        baseline=baseline,
                        seed=42,
                        epochs=1,
                        batch_size=2,
                        embedding_dim=8,
                        device="cpu",
                        output_dir=str(output_dir),
                        standard_encoder_path=str(Path(tmp_dir) / "missing_encoder.pt"),
                        topk_candidates=4,
                        dry_run=True,
                        max_train_samples=4,
                        max_val_samples=4,
                    )

                    result = run_experiment(config)

                    self.assertEqual(result["baseline"], baseline)
                    self.assertTrue((output_dir / "metrics.json").is_file())
                    self.assertTrue((output_dir / "checkpoint_best.pt").is_file())
                    self.assertTrue((output_dir / "best_encoder.pt").is_file())

    def test_cli_parses_two_stage_topk_option(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["tacs.train", "--baseline", "two_stage", "--topk-candidates", "7", "--device", "cpu"],
        ):
            config = parse_args()

        self.assertEqual(config.baseline, "two_stage")
        self.assertEqual(config.topk_candidates, 7)
        self.assertEqual(config.device, "cpu")

    def test_cli_rejects_nonpositive_topk(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["tacs.train", "--baseline", "two_stage", "--topk-candidates", "0"],
        ):
            with self.assertRaises(SystemExit) as error:
                parse_args()

        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
