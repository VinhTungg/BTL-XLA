import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tacs.baselines import BaselineModel
from tacs.train import (
    ExperimentConfig,
    build_dataloaders,
    evaluate,
    run_experiment,
    train_one_epoch,
)


class BaselineModelTest(unittest.TestCase):
    """Kiểm tra kiến trúc mạng và forward pass của BaselineModel."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.query = torch.randn(4, 3, 32, 32)
        self.candidates_5d = torch.randn(4, 1, 3, 32, 32)
        self.candidates_4d = torch.randn(4, 3, 32, 32)

    def test_forward_shapes_all_baselines(self) -> None:
        for b_type in ("no_context", "random", "similarity"):
            model = BaselineModel(num_classes=10, embedding_dim=32, baseline_type=b_type)
            logits = model(self.query, self.candidates_5d)
            self.assertEqual(logits.shape, (4, 10))

    def test_no_context_ignores_candidates(self) -> None:
        model = BaselineModel(num_classes=10, embedding_dim=32, baseline_type="no_context")
        model.eval()
        with torch.no_grad():
            out_with_cand = model(self.query, self.candidates_5d)
            out_without_cand = model(self.query, None)
        self.assertTrue(torch.allclose(out_with_cand, out_without_cand, atol=1e-6))

    def test_extract_embedding_is_l2_normalized(self) -> None:
        model = BaselineModel(num_classes=10, embedding_dim=32)
        embs = model.extract_embedding(self.query)
        norms = torch.norm(embs, p=2, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones(4), atol=1e-5))


class TrainingInfrastructureTest(unittest.TestCase):
    """Kiểm tra hạ tầng huấn luyện, gradient, checkpoint và tính tái lập."""

    def test_config_save_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path = Path(tmp_dir) / "test_cfg.json"
            orig = ExperimentConfig(baseline="random", epochs=3, lr=5e-4, seed=123)
            orig.save(cfg_path)
            loaded = ExperimentConfig.load(cfg_path)
            self.assertEqual(orig, loaded)

    def test_validation_does_not_update_gradients(self) -> None:
        model = BaselineModel(num_classes=5, embedding_dim=16)
        images = torch.randn(8, 3, 32, 32)
        targets = torch.randint(0, 5, (8,))
        loader = DataLoader(TensorDataset(images, targets), batch_size=4)
        criterion = nn.CrossEntropyLoss()

        evaluate(model, loader, criterion, torch.device("cpu"))

        for param in model.parameters():
            self.assertIsNone(param.grad)

    def test_checkpoint_roundtrip_preserves_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "ckpt.pt"
            model1 = BaselineModel(num_classes=5, embedding_dim=16)
            torch.save({"model_state_dict": model1.state_dict()}, ckpt_path)

            model2 = BaselineModel(num_classes=5, embedding_dim=16)
            data = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model2.load_state_dict(data["model_state_dict"])

            x = torch.randn(2, 3, 32, 32)
            model1.eval()
            model2.eval()
            with torch.no_grad():
                self.assertTrue(torch.allclose(model1(x), model2(x), atol=1e-6))

    def test_dry_run_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out1 = Path(tmp_dir) / "run1"
            out2 = Path(tmp_dir) / "run2"
            split_path = Path(tmp_dir) / "split.json"
            split_path.write_text('{"seed": 42}', encoding="utf-8")
            generator = torch.Generator().manual_seed(42)

            def dataset(size: int) -> TensorDataset:
                return TensorDataset(
                    torch.randn(size, 3, 32, 32, generator=generator),
                    torch.randint(0, 10, (size,), generator=generator),
                )

            cifar_fixture = SimpleNamespace(
                query_train=dataset(20),
                candidate_pool=dataset(10),
                validation=dataset(12),
                test=dataset(14),
            )

            cfg1 = ExperimentConfig(
                baseline="no_context",
                seed=42,
                dry_run=True,
                split_file=str(split_path),
                output_dir=str(out1),
            )
            cfg2 = ExperimentConfig(
                baseline="no_context",
                seed=42,
                dry_run=True,
                split_file=str(split_path),
                output_dir=str(out2),
            )

            with patch("tacs.train.build_cifar10", return_value=cifar_fixture):
                res1 = run_experiment(cfg1)
                res2 = run_experiment(cfg2)

            self.assertEqual(res1["best_val_acc"], res2["best_val_acc"])
            self.assertEqual(res1["history"][0]["train_loss"], res2["history"][0]["train_loss"])
            self.assertEqual(res1["data"]["source"], "cifar10")
            self.assertEqual(res1["metrics_schema_version"], 2)
            self.assertEqual(res1["config"]["baseline"], "no_context")
            self.assertEqual(res1["selection"]["strategy"], "no_context")
            self.assertGreater(res1["memory"]["peak_rss_mb"], 0)
            self.assertIsNone(res1["memory"]["peak_vram_allocated_mb"])
            self.assertIn("train_peak_rss_mb", res1["history"][0])
            self.assertEqual(
                res1["data"]["full_sizes"],
                {
                    "query_train": 20,
                    "candidate_pool": 10,
                    "validation": 12,
                    "test": 14,
                },
            )
            encoder_payload = torch.load(
                out1 / "best_encoder.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertIn("encoder_state_dict", encoder_payload)
            self.assertEqual(encoder_payload["metadata"]["baseline"], "no_context")
            self.assertEqual(encoder_payload["metadata"]["data"]["source"], "cifar10")

    def test_real_data_failure_does_not_fallback_or_write_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "failed-run"
            config = ExperimentConfig(
                baseline="no_context",
                device="cpu",
                output_dir=str(output_dir),
            )

            with patch(
                "tacs.train.build_cifar10",
                side_effect=FileNotFoundError("missing CIFAR-10"),
            ):
                with self.assertRaisesRegex(RuntimeError, "Không thể nạp CIFAR-10"):
                    run_experiment(config)

            self.assertFalse(output_dir.exists())

    def test_real_data_provenance_records_sizes_and_split_hash(self) -> None:
        generator = torch.Generator().manual_seed(9)

        def dataset(size: int) -> TensorDataset:
            return TensorDataset(
                torch.randn(size, 3, 32, 32, generator=generator),
                torch.randint(0, 10, (size,), generator=generator),
            )

        fake_cifar = SimpleNamespace(
            query_train=dataset(20),
            candidate_pool=dataset(10),
            validation=dataset(12),
            test=dataset(14),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            split_path = Path(tmp_dir) / "split.json"
            split_content = b'{"seed": 42}'
            split_path.write_bytes(split_content)
            config = ExperimentConfig(
                baseline="no_context",
                dry_run=True,
                data_root=str(Path(tmp_dir) / "data"),
                split_file=str(split_path),
            )

            with patch("tacs.train.build_cifar10", return_value=fake_cifar):
                *_, data_info = build_dataloaders(config, torch.device("cpu"))

        self.assertEqual(data_info["source"], "cifar10")
        self.assertEqual(
            data_info["full_sizes"],
            {
                "query_train": 20,
                "candidate_pool": 10,
                "validation": 12,
                "test": 14,
            },
        )
        self.assertEqual(data_info["effective_sizes"]["query_train"], 16)
        self.assertEqual(data_info["effective_sizes"]["validation"], 8)
        self.assertEqual(data_info["split_sha256"], sha256(split_content).hexdigest())


if __name__ == "__main__":
    unittest.main()
