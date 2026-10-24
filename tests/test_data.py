import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset, TensorDataset

from tacs import TACSLoss, TACSModel
from tacs.data import (
    CIFAR_NORMALIZATION,
    ContextDataset,
    FullPoolContextDataset,
    SplitIndices,
    cifar_num_classes,
    cifar_transforms,
    normalize_cifar_name,
    stratified_split,
)


class StratifiedSplitTest(unittest.TestCase):
    """Kiểm tra việc chia dữ liệu theo lớp."""

    def setUp(self) -> None:
        # 10 lớp, mỗi lớp có 100 mẫu.
        # Tổng cộng có 1.000 mẫu.
        self.labels = [
            label
            for label in range(10)
            for _ in range(100)
        ]

    def test_split_is_balanced_and_reproducible(
        self,
    ) -> None:
        first = stratified_split(
            labels=self.labels,
            validation_size=100,
            candidate_ratio=0.20,
            seed=42,
        )

        second = stratified_split(
            labels=self.labels,
            validation_size=100,
            candidate_ratio=0.20,
            seed=42,
        )

        # 1.000 ảnh:
        # 100 validation
        # 20% của 900 = 180 candidate
        # Còn lại 720 query
        self.assertEqual(
            len(first.query_train),
            720,
        )

        self.assertEqual(
            len(first.candidate_pool),
            180,
        )

        self.assertEqual(
            len(first.validation),
            100,
        )

        # Hàm validate kiểm tra trùng, thiếu
        # và index ngoài phạm vi.
        first.validate(
            dataset_size=len(self.labels)
        )

        # Cùng seed phải cho cùng split.
        self.assertEqual(first, second)

        # Kiểm tra từng tập vẫn cân bằng lớp.
        for indices in (
            first.query_train,
            first.candidate_pool,
            first.validation,
        ):
            class_counts = [
                sum(
                    self.labels[index] == label
                    for index in indices
                )
                for label in range(10)
            ]

            self.assertEqual(
                len(set(class_counts)),
                1,
            )

    def test_split_save_and_load(self) -> None:
        original = stratified_split(
            labels=self.labels,
            validation_size=100,
            candidate_ratio=0.20,
            seed=42,
        )

        with tempfile.TemporaryDirectory() as directory:
            split_path = (
                Path(directory)
                / "splits"
                / "test_split.json"
            )

            original.save(split_path)

            loaded = SplitIndices.load(split_path)

            self.assertTrue(split_path.exists())
            self.assertEqual(original, loaded)

            loaded.validate(
                dataset_size=len(self.labels)
            )

    def test_cifar100_split_has_expected_sizes_and_provenance(self) -> None:
        labels = [label for label in range(100) for _ in range(500)]
        split = stratified_split(
            labels=labels,
            validation_size=5_000,
            candidate_ratio=0.20,
            seed=42,
            dataset_name="CIFAR-100",
        )

        self.assertEqual(len(split.query_train), 36_000)
        self.assertEqual(len(split.candidate_pool), 9_000)
        self.assertEqual(len(split.validation), 5_000)
        self.assertEqual(split.dataset, "cifar100")
        self.assertEqual(split.dataset_size, 50_000)
        split.validate(50_000, "cifar100")
        with self.assertRaisesRegex(ValueError, "không thể dùng"):
            split.validate(50_000, "cifar10")

    def test_cifar_dataset_metadata_is_canonical(self) -> None:
        self.assertEqual(normalize_cifar_name("CIFAR-100"), "cifar100")
        self.assertEqual(cifar_num_classes("cifar100"), 100)
        self.assertNotEqual(
            CIFAR_NORMALIZATION["cifar10"],
            CIFAR_NORMALIZATION["cifar100"],
        )
        with self.assertRaisesRegex(ValueError, "không được hỗ trợ"):
            normalize_cifar_name("imagenet")

    def test_dinov3_evaluation_transform_produces_256_pixel_images(self) -> None:
        _, evaluation_transform = cifar_transforms(
            "cifar100",
            image_size=256,
            normalization="dinov3",
        )
        image = Image.new("RGB", (32, 32), color=(127, 64, 32))
        transformed = evaluation_transform(image)

        self.assertEqual(transformed.shape, (3, 256, 256))


class ContextDatasetTest(unittest.TestCase):
    """Kiểm tra candidate sampling và tích hợp TACS."""

    def setUp(self) -> None:
        generator = torch.Generator()
        generator.manual_seed(42)

        images = torch.randn(
            30,
            3,
            32,
            32,
            generator=generator,
        )

        labels = torch.arange(30) % 10

        self.source = TensorDataset(
            images,
            labels,
        )

        # 20 ảnh đầu làm query.
        self.query_dataset = Subset(
            self.source,
            list(range(20)),
        )

        # 10 ảnh cuối làm candidate pool.
        self.candidate_dataset = Subset(
            self.source,
            list(range(20, 30)),
        )

    def test_sampling_is_reproducible(
        self,
    ) -> None:
        dataset = ContextDataset(
            query_dataset=self.query_dataset,
            candidate_dataset=self.candidate_dataset,
            num_candidates=8,
            seed=42,
        )

        first = dataset[0]
        second = dataset[0]

        # Cùng query và cùng epoch phải có
        # cùng candidate indices.
        self.assertTrue(
            torch.equal(
                first["candidate_indices"],
                second["candidate_indices"],
            )
        )

        query_index = first[
            "query_index"
        ].item()

        self.assertNotIn(
            query_index,
            first["candidate_indices"].tolist(),
        )

        # Sang epoch mới, candidates phải đổi.
        dataset.set_epoch(1)

        third = dataset[0]

        self.assertFalse(
            torch.equal(
                first["candidate_indices"],
                third["candidate_indices"],
            )
        )

    def test_batch_integrates_with_tacs(
        self,
    ) -> None:
        dataset = ContextDataset(
            query_dataset=self.query_dataset,
            candidate_dataset=self.candidate_dataset,
            num_candidates=8,
            seed=42,
        )

        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )

        batch = next(iter(loader))

        self.assertEqual(
            batch["query"].shape,
            (4, 3, 32, 32),
        )

        self.assertEqual(
            batch["candidates"].shape,
            (4, 8, 3, 32, 32),
        )

        self.assertEqual(
            batch["target"].shape,
            (4,),
        )

        model = TACSModel(
            num_classes=10,
            embedding_dim=16,
        )

        output = model(
            batch["query"],
            batch["candidates"],
        )

        self.assertEqual(
            output.context_logits.shape,
            (4, 10),
        )

        loss = TACSLoss()(
            output,
            batch["target"],
        )

        loss.total.backward()

        selector_gradients = [
            parameter.grad
            for parameter
            in model.selector.parameters()
            if parameter.grad is not None
        ]

        self.assertTrue(selector_gradients)

        total_gradient = sum(
            gradient.abs().sum().item()
            for gradient in selector_gradients
        )

        self.assertGreater(
            total_gradient,
            0.0,
        )

    def test_full_pool_wrapper_does_not_duplicate_candidate_images(self) -> None:
        dataset = FullPoolContextDataset(
            query_dataset=self.query_dataset,
            candidate_dataset=self.candidate_dataset,
        )
        batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False)))

        self.assertEqual(batch["query"].shape, (4, 3, 32, 32))
        self.assertNotIn("candidates", batch)
        self.assertIs(dataset.candidate_dataset, self.candidate_dataset)
        self.assertEqual(len(dataset.candidate_dataset), 10)


if __name__ == "__main__":
    unittest.main()
