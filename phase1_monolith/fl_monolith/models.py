"""Data structures exchanged between the components.

Each class here is the in-process version of a message that Phase 2 will
define in Protocol Buffers:

    ExperimentConfig -> message ExperimentConfig
    RoundConfig      -> message TrainingTask      (GetTrainingTask reply)
    ModelUpdate      -> message TrainingResult    (SubmitTrainingResult request)
    Experiment       -> message ExperimentStatus  (GetTrainingStatus reply)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class ExperimentStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class ExperimentConfig:
    """User-chosen settings for one experiment."""
    rounds: int = 1
    # 1.0 => a node's local model jumps straight to its local optimum.
    learning_rate: float = 1.0

    def __post_init__(self) -> None:
        if self.rounds < 1:
            raise ValueError("rounds must be >= 1")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")


@dataclass(frozen=True)
class RoundConfig:
    """Sent to every worker node at the start of a round."""
    experiment_id: str
    round_number: int
    global_model: float
    learning_rate: float


@dataclass(frozen=True)
class ModelUpdate:
    """Returned by a worker node after local training."""
    experiment_id: str
    round_number: int
    node_id: str
    local_model: float
    num_samples: int

    def __post_init__(self) -> None:
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")


@dataclass
class Experiment:
    """Full state of one experiment, owned by the ExperimentManager."""
    experiment_id: str
    config: ExperimentConfig
    status: ExperimentStatus = ExperimentStatus.CREATED
    node_ids: List[str] = field(default_factory=list)
    current_round: int = 0
    global_model: float = 0.0
    history: List[float] = field(default_factory=list)  # global model after each round
    updates: Dict[int, Dict[str, ModelUpdate]] = field(default_factory=dict)  # round -> node -> update
    error: Optional[str] = None
