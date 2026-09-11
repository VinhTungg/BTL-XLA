"""Minimal, testable TACS implementation."""

from .losses import TACSLoss, TACSLossOutput
from .model import TACSModel, TACSOutput

__all__ = ["TACSModel", "TACSOutput", "TACSLoss", "TACSLossOutput"]

