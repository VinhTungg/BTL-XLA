"""Minimal, testable TACS implementation."""

from .baselines import BaselineModel
from .losses import TACSLoss, TACSLossOutput
from .model import TACSModel, TACSOutput, TinyVisionEncoder
from .retrieval import (
    CosineSimilarityRetriever,
    SimilarityContextDataset,
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
    "TACSLoss",
    "TACSLossOutput",
    "BaselineModel",
    "CosineSimilarityRetriever",
    "SimilarityContextDataset",
    "precompute_candidate_embeddings",
    "ExperimentConfig",
    "run_experiment",
]
