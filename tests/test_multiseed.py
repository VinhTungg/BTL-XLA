import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tacs.multiseed import (
    BASELINES,
    RunnerConfig,
    aggregate_metrics,
    build_train_command,
    is_complete_run,
    ordered_baselines,
    parse_args,
    run_all,
    write_summaries,
)
from tacs.train import METRICS_SCHEMA_VERSION


def write_complete_run(
    config: RunnerConfig,
    seed: int,
    baseline: str,
    *,
    val_accuracy: float = 0.8,
    test_accuracy: float = 0.75,
) -> None:
    run_dir = config.run_dir(seed, baseline)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "baseline": baseline,
        "seed": seed,
        "config": {
            "dataset": config.dataset,
            "backbone": config.backbone,
            "pretrained_backbone": config.pretrained_backbone,
            "seed": seed,
            "batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "lr": config.lr,
            "min_lr": config.min_lr,
            "warmup_epochs": config.warmup_epochs,
            "weight_decay": config.weight_decay,
            "split_file": str(config.resolved_split_file),
            "topk_candidates": config.topk_candidates,
            "selector_temperature": config.selector_temperature,
            "policy_weight": config.policy_weight,
            "dry_run": config.dry_run,
            "num_workers": config.num_workers,
            "eval_test": config.eval_test,
            "amp": config.amp,
        },
        "training_recipe": {"executed_epochs": config.executed_epochs},
        "best_val_acc": val_accuracy,
        "test_result": {"accuracy": test_accuracy} if config.eval_test else None,
        "end_to_end_time_s": 100.0 + seed,
        "memory": {
            "peak_rss_mb": 500.0 + seed,
            "peak_vram_allocated_mb": 1000.0 + seed,
        },
        "history": [
            {
                "val_latency_ms_per_query": 10.0 + seed,
                "val_throughput_qps": 100.0 - seed,
            }
        ],
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    (run_dir / "checkpoint_best.pt").touch()
    (run_dir / "best_encoder.pt").touch()


class MultiSeedRunnerTest(unittest.TestCase):
    def test_defaults_target_paper_scale_three_seed_experiment(self) -> None:
        config = parse_args([])
        self.assertEqual(config.dataset, "cifar100")
        self.assertEqual(config.backbone, "dinov3_vits16")
        self.assertEqual(config.seeds, (42, 43, 44))
        self.assertEqual(config.baselines, BASELINES)
        self.assertTrue(config.eval_test)
        self.assertTrue(config.amp)
        self.assertTrue(config.resume)
        self.assertEqual(config.configured_epochs, 100)
        self.assertEqual(config.gradient_accumulation_steps, 1)
        self.assertEqual(config.warmup_epochs, 0)

    def test_command_uses_isolated_seed_paths_and_standard_encoder(self) -> None:
        config = RunnerConfig(
            dataset="cifar10",
            backbone="tiny",
            outputs_root=Path("results"),
            epochs=3,
            gradient_accumulation_steps=4,
            warmup_epochs=1,
            dry_run=True,
            pretrained_backbone=False,
        )
        command = build_train_command(config, 43, "two_stage")

        self.assertIn("--standard-encoder", command)
        encoder_index = command.index("--standard-encoder") + 1
        self.assertEqual(
            Path(command[encoder_index]),
            Path("results/cifar10/tiny/seed_43/no_context/best_encoder.pt"),
        )
        self.assertIn("--dry-run", command)
        self.assertIn("--no-pretrained-backbone", command)
        accumulation_index = command.index("--gradient-accumulation-steps") + 1
        warmup_index = command.index("--warmup-epochs") + 1
        self.assertEqual(command[accumulation_index], "4")
        self.assertEqual(command[warmup_index], "1")
        self.assertLess(
            ordered_baselines(("two_stage", "no_context")).index("no_context"),
            ordered_baselines(("two_stage", "no_context")).index("two_stage"),
        )

        restart_command = build_train_command(
            RunnerConfig(force=True), 42, "no_context"
        )
        self.assertIn("--no-resume", restart_command)

    def test_complete_run_requires_matching_config_test_and_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = RunnerConfig(
                dataset="cifar10",
                backbone="tiny",
                seeds=(42,),
                baselines=("no_context",),
                outputs_root=Path(temp_dir),
                epochs=2,
            )
            write_complete_run(config, 42, "no_context")
            self.assertTrue(is_complete_run(config, 42, "no_context"))

            metrics_path = config.run_dir(42, "no_context") / "metrics.json"
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            payload["config"]["batch_size"] = 999
            metrics_path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(is_complete_run(config, 42, "no_context"))

    def test_aggregates_sample_standard_deviation_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = RunnerConfig(
                dataset="cifar10",
                backbone="tiny",
                seeds=(42, 43, 44),
                baselines=("no_context",),
                outputs_root=Path(temp_dir),
                epochs=2,
            )
            for seed, accuracy in zip(config.seeds, (0.7, 0.8, 0.9)):
                write_complete_run(
                    config,
                    seed,
                    "no_context",
                    val_accuracy=accuracy,
                    test_accuracy=accuracy - 0.05,
                )

            document, rows = aggregate_metrics(config)
            self.assertEqual(len(document["runs"]), 3)
            self.assertAlmostEqual(rows[0]["best_val_accuracy_mean"], 0.8)
            self.assertAlmostEqual(rows[0]["best_val_accuracy_std"], 0.1)

            json_path, csv_path, markdown_path = write_summaries(config)
            self.assertEqual(
                json.loads(json_path.read_text(encoding="utf-8"))["baselines"][0][
                    "num_runs"
                ],
                3,
            )
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                self.assertEqual(next(csv.DictReader(handle))["baseline"], "no_context")
            self.assertIn("0.8000 ± 0.1000", markdown_path.read_text(encoding="utf-8"))

    @patch("tacs.multiseed.subprocess.run")
    def test_resume_skips_complete_run(self, subprocess_run) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = RunnerConfig(
                dataset="cifar10",
                backbone="tiny",
                seeds=(42,),
                baselines=("no_context",),
                outputs_root=Path(temp_dir),
                epochs=2,
            )
            write_complete_run(config, 42, "no_context")
            run_all(config)
            subprocess_run.assert_not_called()

    @patch("tacs.multiseed.subprocess.run")
    def test_plan_only_does_not_require_existing_standard_encoder(
        self, subprocess_run
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = RunnerConfig(
                dataset="cifar10",
                backbone="tiny",
                seeds=(42,),
                baselines=("two_stage",),
                outputs_root=Path(temp_dir),
                plan_only=True,
            )
            self.assertIsNone(run_all(config))
            subprocess_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
