"""Công cụ chia dữ liệu và tạo candidate pool cho TACS"""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset, Subset


@dataclass(frozen=True)
class SplitIndices:
    """Các index thuộc 3 tập con của tập train"""

    query_train: list[int]
    candidate_pool: list[int]
    validation: list[int]
    seed: int

    def validate(self, dataset_size: int) -> None:
        query = set(self.query_train)
        candidate = set(self.candidate_pool)
        validation = set(self.validation)

        if len(query) != len(self.query_train):
            raise ValueError("Query-training chứa index bị lặp")

        if len(candidate) != len(self.candidate_pool):
            raise ValueError("Candidate pool chứa index bị lặp")

        if len(validation) != len(self.validation):
            raise ValueError("Validation chứa index bị lặp")

        if query & candidate:
            raise ValueError("Query-training trùng candidate pool")

        if query & validation:
            raise ValueError("Query-training trùng validation")

        if candidate & validation:
            raise ValueError("Candidate pool trùng validation")

        all_indices = query | candidate | validation

        if all_indices != set(range(dataset_size)):
            raise ValueError(
                "Các tập phải bao phủ toàn bộ dữ liệu train đúng một lần"
            )

    def save(self, path: str | Path) -> None:
        """Lưu các index vào file JSON."""
        output_path = Path(path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        content = asdict(self)

        output_path.write_text(
            json.dumps(
                content,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "SplitIndices":
        """Đọc file JSON và khôi phục SplitIndices."""

        input_path = Path(path)

        content = json.loads(
            input_path.read_text(encoding="utf-8")
        )

        return cls(
            query_train=content["query_train"],
            candidate_pool=content["candidate_pool"],
            validation=content["validation"],
            seed=content["seed"],
        )



def stratified_split(
        labels: Sequence[int],
        validation_size: int = 5_000,
        candidate_ratio: float = 0.20,
        seed: int = 42,
) -> SplitIndices:
    """Chia tập train theo lớp thành query, candidate và validation.

    Candidate ratio được tính sau khi đã tách validation.

    Với CIFAR-10:
        50.000 ảnh ban đầu
        5.000 ảnh validation
        20% của 45.000 ảnh còn lại = 9.000 candidates
        36.000 ảnh còn lại = query-training
    """

    dataset_size = len(labels)

    if dataset_size == 0:
        raise ValueError("Danh sách labels không được rỗng")

    if not 0 < validation_size < dataset_size:
        raise ValueError(
            "validation_size phải lớn hơn 0 và nhỏ hơn kích thước dataset"
        )

    if not 0 < candidate_ratio < 1:
        raise ValueError("candidate_ratio phải nằm trong khoảng (0, 1)")

    labels_tensor = torch.as_tensor(labels, dtype=torch.long)
    classes = torch.unique(labels_tensor, sorted=True)

    num_classes = len(classes)

    if validation_size % num_classes != 0:
        raise ValueError(
            "validation_size phải chia hết cho số lớp "
            "để mỗi lớp có số mẫu validation bằng nhau"
        )

    validation_per_class = validation_size // num_classes

    generator = torch.Generator()
    generator.manual_seed(seed)

    query_indices: list[int] = []
    candidate_indices: list[int] = []
    validation_indices: list[int] = []

    for class_label in classes:
        # Tìm tất cả vị trí có nhãn bằng class_label.
        class_indices = torch.where(
            labels_tensor == class_label
        )[0]

        if validation_per_class >= len(class_indices):
            raise ValueError(
                f"Lớp {class_label.item()} không đủ dữ liệu "
                "sau khi tách validation"
            )

        # Xáo trộn index của riêng lớp hiện tại.
        permutation = torch.randperm(
            len(class_indices),
            generator=generator,
        )

        shuffled_indices = class_indices[permutation]

        # Phần đầu dành cho validation.
        class_validation = shuffled_indices[
            :validation_per_class
        ]

        # Phần còn lại sẽ được chia thành candidate và query.
        class_remaining = shuffled_indices[
            validation_per_class:
        ]

        candidate_count = round(
            len(class_remaining) * candidate_ratio
        )

        class_candidates = class_remaining[
            :candidate_count
        ]

        class_queries = class_remaining[
            candidate_count:
        ]

        validation_indices.extend(
            class_validation.tolist()
        )

        candidate_indices.extend(
            class_candidates.tolist()
        )

        query_indices.extend(
            class_queries.tolist()
        )

    split = SplitIndices(
        query_train=query_indices,
        candidate_pool=candidate_indices,
        validation=validation_indices,
        seed=seed,
    )

    split.validate(dataset_size)

    return split

def cifar10_transforms():
    """Tạo transform dành cho train và evaluation."""

    from torchvision import transforms

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(
                size=32,
                padding=4,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )

    evaluation_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )

    return train_transform, evaluation_transform

@dataclass
class CIFAR10Datasets:
    """Bốn tập dữ liệu sử dụng trong thí nghiệm."""

    query_train: Dataset
    candidate_pool: Dataset
    validation: Dataset
    test: Dataset
    split: SplitIndices

def build_cifar10(
    root: str | Path,
    split_file: str | Path,
    validation_size: int = 5_000,
    candidate_ratio: float = 0.20,
    seed: int = 42,
    download: bool = False,
) -> CIFAR10Datasets:
    """Tạo bốn tập CIFAR-10 dùng trong dự án."""

    from torchvision.datasets import CIFAR10

    train_transform, evaluation_transform = (
        cifar10_transforms()
    )

    # Dataset này chỉ dùng để lấy danh sách nhãn.
    raw_train = CIFAR10(
        root=str(root),
        train=True,
        download=download,
    )

    split_path = Path(split_file)

    if split_path.exists():
        split = SplitIndices.load(split_path)
        split.validate(len(raw_train))
    else:
        split = stratified_split(
            labels=raw_train.targets,
            validation_size=validation_size,
            candidate_ratio=candidate_ratio,
            seed=seed,
        )

        split.save(split_path)

    augmented_train = CIFAR10(
        root=str(root),
        train=True,
        transform=train_transform,
        download=False,
    )

    evaluation_train = CIFAR10(
        root=str(root),
        train=True,
        transform=evaluation_transform,
        download=False,
    )

    test_dataset = CIFAR10(
        root=str(root),
        train=False,
        transform=evaluation_transform,
        download=download,
    )

    return CIFAR10Datasets(
        query_train=Subset(
            augmented_train,
            split.query_train,
        ),
        candidate_pool=Subset(
            evaluation_train,
            split.candidate_pool,
        ),
        validation=Subset(
            evaluation_train,
            split.validation,
        ),
        test=test_dataset,
        split=split,
    )

class ContextDataset(Dataset):
    """Ghép mỗi query với một nhóm ảnh trong candidate pool."""

    def __init__(
        self,
        query_dataset: Dataset,
        candidate_dataset: Dataset,
        num_candidates: int = 8,
        seed: int = 42,
    ) -> None:
        if num_candidates <= 0:
            raise ValueError(
                "num_candidates phải lớn hơn 0"
            )

        if num_candidates > len(candidate_dataset):
            raise ValueError(
                "num_candidates lớn hơn kích thước candidate pool"
            )

        self.query_dataset = query_dataset
        self.candidate_dataset = candidate_dataset
        self.num_candidates = num_candidates
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.query_dataset)

    def set_epoch(self, epoch: int) -> None:
        """Thay đổi candidates theo epoch nhưng vẫn tái lập được."""

        self.epoch = epoch

    def _sample_candidate_positions(
        self,
        query_position: int,
    ) -> list[int]:
        """Chọn vị trí candidates cho một query."""

        sample_seed = (
            self.seed
            + self.epoch * len(self)
            + query_position
        )

        random_generator = random.Random(sample_seed)

        return random_generator.sample(
            range(len(self.candidate_dataset)),
            k=self.num_candidates,
        )

    def __getitem__(
        self,
        query_position: int,
    ) -> dict[str, torch.Tensor]:
        query_image, query_target = (
            self.query_dataset[query_position]
        )

        candidate_positions = (
            self._sample_candidate_positions(
                query_position
            )
        )

        candidate_samples = [
            self.candidate_dataset[position]
            for position in candidate_positions
        ]

        candidate_images = torch.stack(
            [
                image
                for image, _ in candidate_samples
            ]
        )

        candidate_targets = torch.tensor(
            [
                target
                for _, target in candidate_samples
            ],
            dtype=torch.long,
        )

        candidate_indices = torch.tensor(
            [
                self.candidate_dataset.indices[position]
                if isinstance(
                    self.candidate_dataset,
                    Subset,
                )
                else position
                for position in candidate_positions
            ],
            dtype=torch.long,
        )

        query_index = (
            self.query_dataset.indices[query_position]
            if isinstance(self.query_dataset, Subset)
            else query_position
        )

        return {
            "query": query_image,
            "candidates": candidate_images,
            "target": torch.as_tensor(
                query_target,
                dtype=torch.long,
            ),
            "query_index": torch.tensor(
                query_index,
                dtype=torch.long,
            ),
            "candidate_indices": candidate_indices,
            "candidate_targets": candidate_targets,
        }