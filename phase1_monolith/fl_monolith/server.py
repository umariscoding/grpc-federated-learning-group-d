"""FLServer: the coordinator role. A thin facade over registry, manager and strategy.

Its methods correspond to the Phase 2 gRPC operations:

    register_node     -> RegisterParticipant
    create_experiment }
    start_experiment  } -> StartTraining
    get_status        -> GetTrainingStatus
"""
from __future__ import annotations

from typing import List, Optional

from .experiment_manager import ExperimentManager
from .models import Experiment, ExperimentConfig
from .node_registry import NodeRegistry
from .strategy import FedAvgStrategy, Strategy
from .worker_node import WorkerNode


class FLServer:
    def __init__(self, strategy: Optional[Strategy] = None):
        self._registry = NodeRegistry()
        self._manager = ExperimentManager(self._registry, strategy or FedAvgStrategy())

    def register_node(self, node: WorkerNode) -> None:
        self._registry.register_node(node)

    def get_active_nodes(self) -> List[str]:
        return [n.node_id for n in self._registry.get_active_nodes()]

    def create_experiment(self, config: Optional[ExperimentConfig] = None) -> Experiment:
        return self._manager.create_experiment(config)

    def start_experiment(self, experiment_id: str) -> Experiment:
        return self._manager.start_experiment(experiment_id)

    def get_status(self, experiment_id: str) -> Experiment:
        return self._manager.get_status(experiment_id)

    def list_experiments(self) -> List[Experiment]:
        return self._manager.list_experiments()
