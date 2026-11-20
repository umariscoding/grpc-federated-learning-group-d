"""Aggregation strategies (server side)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from .models import ModelUpdate


class Strategy(ABC):
    """How the server combines node updates into a new global model."""

    @abstractmethod
    def aggregate(self, updates: Sequence[ModelUpdate]) -> float:
        ...


class FedAvgStrategy(Strategy):
    """Federated Averaging: mean of local models weighted by sample count.

    With equal sample counts this is the plain average, e.g. (10 + 20 + 30) / 3 = 20.
    """

    def aggregate(self, updates: Sequence[ModelUpdate]) -> float:
        if not updates:
            raise ValueError("cannot aggregate an empty list of updates")
        total_samples = sum(u.num_samples for u in updates)
        return sum(u.local_model * u.num_samples for u in updates) / total_samples
