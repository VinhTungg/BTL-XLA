import tempfile
import unittest
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tacs.baselines import BaselineModel
from tacs.model import TinyPatchTokenTaskNetwork
from tacs.train import (
    ExperimentConfig,
    _restore_training_checkpoint,
    build_optimizer_and_scheduler,
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

    def test_all_baselines_use_the_patch_token_task_network(self) -> None:
        for baseline_type, expected_tokens in (
            ("no_context", 65),
            ("random", 129),
            ("similarity", 129),
        ):
            with self.subTest(baseline_type=baseline_type):
                model = BaselineModel(
                    num_classes=10,
                    embedding_dim=32,
                    baseline_type=baseline_type,
                )
                self.assertIsInstance(
                    model.task_network,
                    TinyPatchTokenTaskNetwork,
                )
                sequence_lengths: list[int] = []

                def record_shape(_module, inputs) -> None:
                    sequence_lengths.append(inputs[0].shape[1])

                handle = model.task_network.blocks.layers[0].register_forward_pre_hook(
                    record_shape
                )
                try:
                    model(self.query, self.candidates_5d)
                finally:
                    handle.remove()
                self.assertEqual(sequence_lengths, [expected_tokens])

    def test_retrieval_encoder_is_frozen_while_task_network_trains(self) -> None:
        model = BaselineModel(
            num_classes=10,
            embedding_dim=32,
            baseline_type="random",
        )
        model.train()
        self.assertFalse(model.encoder.training)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.encoder.parameters())
        )

        model(self.query, self.candidates_5d).sum().backward()

        self.assertTrue(
            all(parameter.grad is None for parameter in model.encoder.parameters())
        )
        task_gradient = sum(
            parameter.grad.abs().sum().item()
            for parameter in model.task_network.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(task_gradient, 0.0)


class TrainingInfrastructureTest(unittest.TestCase):
    """Kiểm tra hạ tầng huấn luyện, gradient, checkpoint và tính tái lập."""

    def test_config_save_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path = Path(tmp_dir) / "test_cfg.json"
            orig = ExperimentConfig(baseline="random", epochs=3, lr=5e-4, seed=123)
            orig.save(cfg_path)
            loaded = ExperimentConfig.load(cfg_path)
            self.assertEqual(orig, loaded)

    def test_cifar100_config_derives_safe_dataset_defaults(self) -> None:
        config = ExperimentConfig(dataset="cifar100", baseline="tacs", seed=7)

        self.assertEqual(config.num_classes, 100)
        self.assertEqual(config.split_file, "data/splits/cifar100_seed42.json")
        self.assertEqual(config.output_dir, "outputs/cifar100/tacs")
        self.assertEqual(
            config.standard_encoder_path,
            "outputs/cifar100/no_context/best_encoder.pt",
        )
        with self.assertRaisesRegex(ValueError, "num_classes=100"):
            ExperimentConfig(dataset="cifar100", num_classes=10)

    def test_dinov3_config_derives_paper_backbone_defaults(self) -> None:
        config = ExperimentConfig(
            dataset="cifar100",
            backbone="dinov3_vits16",
            baseline="no_context",
        )

        self.assertEqual(config.embedding_dim, 384)
        self.assertEqual(config.image_size, 256)
        self.assertEqual(config.epochs, 100)
        self.assertEqual(
            config.output_dir,
            "outputs/cifar100/dinov3_vits16/no_context",
        )
        with self.assertRaisesRegex(ValueError, "embedding_dim=384"):
            ExperimentConfig(backbone="dinov3_vits16", embedding_dim=64)

    def test_training_recipe_uses_adamw_and_cosine_annealing(self) -> None:
        model = BaselineModel(num_classes=10, embedding_dim=16)
        config = ExperimentConfig(epochs=4, lr=1e-3, min_lr=1e-5)

        optimizer, scheduler = build_optimizer_and_scheduler(model, config)

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertIsInstance(
            scheduler,
            torch.optim.lr_scheduler.CosineAnnealingLR,
        )
        self.assertEqual(scheduler.T_max, 4)
        self.assertEqual(scheduler.eta_min, 1e-5)
        learning_rates = [optimizer.param_groups[0]["lr"]]
        for _ in range(4):
            optimizer.step()
            scheduler.step()
            learning_rates.append(optimizer.param_groups[0]["lr"])
        self.assertEqual(learning_rates[0], 1e-3)
        self.assertAlmostEqual(learning_rates[-1], 1e-5)
        self.assertTrue(
            all(
                current >= following
                for current, following in zip(learning_rates, learning_rates[1:])
            )
        )

    def test_training_config_rejects_invalid_schedule_values(self) -> None:
        for kwargs in (
            {"epochs": 0},
            {"lr": 0.0},
            {"lr": 1e-3, "min_lr": 1e-3},
            {"weight_decay": -1e-4},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ExperimentConfig(**kwargs)

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

    def test_resume_restores_model_optimizer_scheduler_and_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "checkpoint_latest.pt"
            config = ExperimentConfig(
                baseline="no_context",
                epochs=3,
                output_dir=tmp_dir,
                device="cpu",
            )
            source_model = BaselineModel(num_classes=10, embedding_dim=64)
            source_optimizer, source_scheduler = build_optimizer_and_scheduler(
                source_model, config
            )
            source_optimizer.step()
            source_scheduler.step()
            with torch.no_grad():
                next(source_model.parameters()).fill_(0.25)
            torch.save(
                {
                    "epoch": 1,
                    "model_state_dict": source_model.state_dict(),
                    "optimizer_state_dict": source_optimizer.state_dict(),
                    "scheduler_state_dict": source_scheduler.state_dict(),
                    "val_accuracy": 0.4,
                    "best_val_accuracy": 0.5,
                    "config": asdict(config),
                    "history": [{"epoch": 1}],
                    "elapsed_training_time_s": 12.5,
                    "scaler_state_dict": {},
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state_all": None,
                },
                checkpoint_path,
            )

            target_model = BaselineModel(num_classes=10, embedding_dim=64)
            target_optimizer, target_scheduler = build_optimizer_and_scheduler(
                target_model, config
            )
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            start_epoch, best_acc, history, elapsed = _restore_training_checkpoint(
                checkpoint_path,
                target_model,
                target_optimizer,
                target_scheduler,
                scaler,
                config,
                torch.device("cpu"),
            )

            self.assertEqual(start_epoch, 2)
            self.assertEqual(best_acc, 0.5)
            self.assertEqual(history, [{"epoch": 1}])
            self.assertEqual(elapsed, 12.5)
            self.assertTrue(
                torch.equal(
                    next(source_model.parameters()),
                    next(target_model.parameters()),
                )
            )
            self.assertEqual(
                target_scheduler.state_dict(), source_scheduler.state_dict()
            )

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
                device="cpu",
                split_file=str(split_path),
                output_dir=str(out1),
            )
            cfg2 = ExperimentConfig(
                baseline="no_context",
                seed=42,
                dry_run=True,
                device="cpu",
                split_file=str(split_path),
                output_dir=str(out2),
            )

            with patch("tacs.train.build_cifar10", return_value=cifar_fixture):
                res1 = run_experiment(cfg1)
                res2 = run_experiment(cfg2)

            self.assertEqual(res1["best_val_acc"], res2["best_val_acc"])
            self.assertEqual(res1["history"][0]["train_loss"], res2["history"][0]["train_loss"])
            self.assertEqual(res1["data"]["source"], "cifar10")
            self.assertEqual(res1["metrics_schema_version"], 5)
            self.assertEqual(res1["config"]["baseline"], "no_context")
            self.assertEqual(res1["selection"]["strategy"], "no_context")
            self.assertEqual(
                res1["selection"]["downstream_input"],
                "query_patch_tokens",
            )
            self.assertEqual(
                res1["model_architecture"]["downstream_task_network"],
                "patch_token_transformer",
            )
            self.assertGreater(res1["memory"]["peak_rss_mb"], 0)
            self.assertIsNone(res1["memory"]["peak_vram_allocated_mb"])
            self.assertIn("train_peak_rss_mb", res1["history"][0])
            self.assertEqual(res1["training_recipe"]["optimizer"], "AdamW")
            self.assertEqual(
                res1["training_recipe"]["scheduler"],
                "CosineAnnealingLR",
            )
            self.assertEqual(res1["history"][0]["learning_rate"], 1e-3)
            self.assertIn("next_learning_rate", res1["history"][0])
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
            checkpoint = torch.load(
                out1 / "checkpoint_latest.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertIn("scheduler_state_dict", checkpoint)
            self.assertIn("scaler_state_dict", checkpoint)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertFalse(res1["training_recipe"]["amp_enabled"])
            self.assertIsNone(res1["training_recipe"]["resumed_from_epoch"])

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

    def test_cifar100_uses_real_dataset_route_and_records_provenance(self) -> None:
        generator = torch.Generator().manual_seed(11)

        def dataset(size: int) -> TensorDataset:
            return TensorDataset(
                torch.randn(size, 3, 32, 32, generator=generator),
                torch.randint(0, 100, (size,), generator=generator),
            )

        fake_cifar100 = SimpleNamespace(
            query_train=dataset(20),
            candidate_pool=dataset(10),
            validation=dataset(12),
            test=dataset(14),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            split_path = Path(tmp_dir) / "cifar100_split.json"
            split_path.write_text('{"dataset": "cifar100"}', encoding="utf-8")
            config = ExperimentConfig(
                dataset="cifar100",
                baseline="no_context",
                dry_run=True,
                split_file=str(split_path),
                output_dir=str(Path(tmp_dir) / "output"),
            )
            with patch("tacs.train.build_cifar100", return_value=fake_cifar100) as builder:
                *_, data_info = build_dataloaders(config, torch.device("cpu"))

        builder.assert_called_once()
        self.assertEqual(config.num_classes, 100)
        self.assertEqual(data_info["source"], "cifar100")
        self.assertEqual(data_info["effective_sizes"]["query_train"], 16)


if __name__ == "__main__":
    unittest.main()
