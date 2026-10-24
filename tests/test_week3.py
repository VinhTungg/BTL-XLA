import sys
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from tacs.losses import TACSLoss
from tacs.model import TACSModel, TinyVisionEncoder
from tacs.train import (
    ENCODER_CHECKPOINT_SCHEMA_VERSION,
    ExperimentConfig,
    _load_standard_encoder,
    evaluate,
    parse_args,
    run_experiment,
    train_one_epoch,
)


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
            generator = torch.Generator().manual_seed(42)

            def dataset(size: int) -> TensorDataset:
                return TensorDataset(
                    torch.randn(size, 3, 32, 32, generator=generator),
                    torch.randint(0, 10, (size,), generator=generator),
                )

            cifar_fixture = SimpleNamespace(
                query_train=dataset(8),
                candidate_pool=dataset(8),
                validation=dataset(8),
                test=dataset(8),
            )
            split_path = Path(tmp_dir) / "split.json"
            split_path.write_text('{"seed": 42}', encoding="utf-8")
            split_sha256 = sha256(split_path.read_bytes()).hexdigest()
            standard_encoder_path = Path(tmp_dir) / "standard_encoder.pt"
            standard_encoder = TinyVisionEncoder(embedding_dim=8)
            torch.save(
                {
                    "encoder_state_dict": standard_encoder.state_dict(),
                    "metadata": {
                        "schema_version": ENCODER_CHECKPOINT_SCHEMA_VERSION,
                        "baseline": "no_context",
                        "embedding_dim": 8,
                        "data": {
                            "source": "cifar10",
                            "split_sha256": split_sha256,
                            "full_sizes": {
                                "query_train": 8,
                                "candidate_pool": 8,
                                "validation": 8,
                                "test": 8,
                            },
                            "effective_sizes": {
                                "query_train": 4,
                                "candidate_pool": 8,
                                "validation": 4,
                                "test": 8,
                            },
                        },
                    },
                },
                standard_encoder_path,
            )

            # Cache legacy/sai kích thước phải bị Two-Stage thay thế.
            cache_dir = Path(tmp_dir) / "cache"
            cache_dir.mkdir()
            cache_path = cache_dir / "candidate_pool_embeddings.pt"
            torch.save(
                {
                    "embeddings": torch.randn(2, 8),
                    "targets": torch.zeros(2, dtype=torch.long),
                    "indices": torch.arange(2),
                    "embedding_dim": 8,
                    "num_candidates": 2,
                },
                cache_path,
            )

            with patch("tacs.train.build_cifar10", return_value=cifar_fixture):
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
                            split_file=str(split_path),
                            output_dir=str(output_dir),
                            standard_encoder_path=str(standard_encoder_path),
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
                        self.assertEqual(result["data"]["source"], "cifar10")
                        self.assertEqual(result["data"]["effective_sizes"]["query_train"], 4)
                        self.assertIn("train_latency_ms_per_query", result["history"][0])
                        self.assertIn("val_latency_ms_per_query", result["history"][0])
                        self.assertIn(
                            "train_selection_precompute_time_s",
                            result["history"][0],
                        )
                        self.assertIn(
                            "val_selection_precompute_time_s",
                            result["history"][0],
                        )
                        if baseline == "tacs":
                            self.assertEqual(result["selection"]["strategy"], "full_pool_tacs")
                            self.assertEqual(result["selection"]["selector_candidates"], 8)
                            self.assertEqual(
                                result["selection"]["candidate_bank_refresh"],
                                "once_per_train_or_evaluation_epoch",
                            )
                        else:
                            self.assertEqual(
                                result["selection"]["strategy"],
                                "cosine_topk_then_tacs",
                            )
                            self.assertEqual(result["selection"]["retrieval_candidates_scanned"], 8)
                            self.assertEqual(result["selection"]["selector_candidates"], 4)
                            self.assertEqual(
                                result["selection"]["candidate_bank_refresh"],
                                "once_per_train_or_evaluation_epoch",
                            )
                            self.assertEqual(
                                result["selection"]["candidate_batch_representation"],
                                "topk_indices_into_shared_embedding_bank",
                            )

            rebuilt_cache = torch.load(cache_path, map_location="cpu", weights_only=True)
            self.assertEqual(rebuilt_cache["num_candidates"], 8)
            self.assertEqual(rebuilt_cache["metadata"]["candidate_count"], 8)
            self.assertEqual(rebuilt_cache["metadata"]["embedding_dim"], 8)

    def test_standard_encoder_must_exist_and_have_cifar10_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_config = ExperimentConfig(
                baseline="two_stage",
                embedding_dim=8,
                standard_encoder_path=str(Path(tmp_dir) / "missing.pt"),
            )
            with self.assertRaisesRegex(RuntimeError, "không tìm thấy"):
                _load_standard_encoder(missing_config, torch.device("cpu"))

            legacy_path = Path(tmp_dir) / "legacy.pt"
            torch.save(TinyVisionEncoder(8).state_dict(), legacy_path)
            legacy_config = ExperimentConfig(
                baseline="two_stage",
                embedding_dim=8,
                standard_encoder_path=str(legacy_path),
            )
            with self.assertRaisesRegex(RuntimeError, "legacy"):
                _load_standard_encoder(legacy_config, torch.device("cpu"))

            dry_run_path = Path(tmp_dir) / "dry-run.pt"
            torch.save(
                {
                    "encoder_state_dict": TinyVisionEncoder(8).state_dict(),
                    "metadata": {
                        "schema_version": ENCODER_CHECKPOINT_SCHEMA_VERSION,
                        "baseline": "no_context",
                        "embedding_dim": 8,
                        "data": {
                            "source": "cifar10",
                            "full_sizes": {"query_train": 20, "validation": 10},
                            "effective_sizes": {"query_train": 16, "validation": 8},
                        },
                    },
                },
                dry_run_path,
            )
            production_config = ExperimentConfig(
                baseline="two_stage",
                embedding_dim=8,
                dry_run=False,
                standard_encoder_path=str(dry_run_path),
            )
            with self.assertRaisesRegex(RuntimeError, "dry-run"):
                _load_standard_encoder(production_config, torch.device("cpu"))

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
        self.assertEqual(config.selector_temperature, 0.1)
        self.assertEqual(config.policy_weight, 1.0)

    def test_cli_parses_and_validates_tacs_objective_options(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "tacs.train",
                "--baseline",
                "tacs",
                "--selector-temperature",
                "0.25",
                "--policy-weight",
                "0.5",
            ],
        ):
            config = parse_args()

        self.assertEqual(config.selector_temperature, 0.25)
        self.assertEqual(config.policy_weight, 0.5)

        for option, value in (
            ("--selector-temperature", "0"),
            ("--policy-weight", "-1"),
        ):
            with self.subTest(option=option):
                with patch.object(sys, "argv", ["tacs.train", option, value]):
                    with self.assertRaises(SystemExit) as error:
                        parse_args()
                self.assertEqual(error.exception.code, 2)

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
