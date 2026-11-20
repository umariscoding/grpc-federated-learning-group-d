"""Worker node (participant role): keeps its data private and trains locally."""
from __future__ import annotations

from typing import Sequence

from .models import ModelUpdate, RoundConfig


class WorkerNode:
    """Simulated participant whose 'model' is a single number.

    Local training moves the received global model toward the mean of the
    node's private data:

        local = global + learning_rate * (mean(data) - global)

    Only the resulting number and the sample count ever leave this class.
    """

    def __init__(self, node_id: str, data: Sequence[float]):
        if not node_id or not node_id.strip():
            raise ValueError("node_id must not be empty")
        if len(data) == 0:
            raise ValueError("a node needs a non-empty local dataset")
        self.node_id = node_id
        self._data = [float(x) for x in data]

    @property
    def num_samples(self) -> int:
        return len(self._data)

    def train(self, round_config: RoundConfig) -> ModelUpdate:
        local_optimum = sum(self._data) / len(self._data)
        local_model = round_config.global_model + round_config.learning_rate * (
            local_optimum - round_config.global_model
        )
        return ModelUpdate(
            experiment_id=round_config.experiment_id,
            round_number=round_config.round_number,
            node_id=self.node_id,
            local_model=local_model,
            num_samples=len(self._data),
        )

    def __repr__(self) -> str:  # never print the private data
        return f"WorkerNode(node_id={self.node_id!r}, num_samples={self.num_samples})"
