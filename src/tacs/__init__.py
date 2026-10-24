"""Minimal, testable TACS implementation."""

from .baselines import BaselineModel
from .data import (
    CIFARDatasets,
    FullPoolContextDataset,
    build_cifar,
    build_cifar10,
    build_cifar100,
)
from .losses import TACSLoss, TACSLossOutput
from .model import (
    DINOv3VisionEncoder,
    TACSModel,
    TACSOutput,
    TinyVisionEncoder,
    create_vision_encoder,
)
from .retrieval import (
    CosineSimilarityRetriever,
    SimilarityContextDataset,
    TopKContextDataset,
    precompute_candidate_embeddings,
)


def __getattr__(name: str):
    if name in ("ExperimentConfig", "run_experiment"):
        from .train import ExperimentConfig, run_experiment

        return {"ExperimentConfig": ExperimentConfig, "run_experiment": run_experiment}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TACSModel",
    "TACSOutput",
    "TinyVisionEncoder",
    "DINOv3VisionEncoder",
    "create_vision_encoder",
    "TACSLoss",
    "TACSLossOutput",
    "BaselineModel",
    "FullPoolContextDataset",
    "CIFARDatasets",
    "build_cifar",
    "build_cifar10",
    "build_cifar100",
    "CosineSimilarityRetriever",
    "SimilarityContextDataset",
    "TopKContextDataset",
    "precompute_candidate_embeddings",
    "ExperimentConfig",
    "run_experiment",
]
