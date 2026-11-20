"""Controls the experiment lifecycle and runs the training rounds."""
from __future__ import annotations

import copy
import itertools
import logging
from typing import Dict, List, Optional

from .models import (Experiment, ExperimentConfig, ExperimentStatus, ModelUpdate,
                     RoundConfig)
from .node_registry import NodeRegistry
from .strategy import Strategy
from .worker_node import WorkerNode

log = logging.getLogger("fl.experiment")


class ExperimentManager:
    """Owns all experiments. Public methods return *snapshots*, never live state."""

    def __init__(self, registry: NodeRegistry, strategy: Strategy):
        self._registry = registry
        self._strategy = strategy
        self._experiments: Dict[str, Experiment] = {}
        self._ids = itertools.count(1)

    # -- public API -------------------------------------------------------
    def create_experiment(self, config: Optional[ExperimentConfig] = None) -> Experiment:
        experiment = Experiment(f"exp-{next(self._ids)}", config or ExperimentConfig())
        self._experiments[experiment.experiment_id] = experiment
        log.info("created %s (rounds=%d, lr=%s)", experiment.experiment_id,
                 experiment.config.rounds, experiment.config.learning_rate)
        return copy.deepcopy(experiment)

    def start_experiment(self, experiment_id: str) -> Experiment:
        """Run every round to completion. A failing node marks the experiment FAILED."""
        experiment = self._get(experiment_id)
        if experiment.status is not ExperimentStatus.CREATED:
            raise ValueError(f"{experiment_id} cannot be started (status={experiment.status.value})")
        nodes = self._registry.get_active_nodes()
        if not nodes:
            raise ValueError("cannot start an experiment: no nodes are registered")

        experiment.node_ids = [n.node_id for n in nodes]
        experiment.status = ExperimentStatus.RUNNING
        try:
            for round_number in range(1, experiment.config.rounds + 1):
                experiment.current_round = round_number
                self._run_round(experiment, nodes)
            experiment.status = ExperimentStatus.COMPLETED
            log.info("%s COMPLETED, global model = %s", experiment_id, experiment.global_model)
        except Exception as exc:  # node code is arbitrary; any error fails the experiment
            experiment.status = ExperimentStatus.FAILED
            experiment.error = f"{type(exc).__name__}: {exc}"
            log.error("%s FAILED: %s", experiment_id, experiment.error)
        return copy.deepcopy(experiment)

    def get_status(self, experiment_id: str) -> Experiment:
        return copy.deepcopy(self._get(experiment_id))

    def list_experiments(self) -> List[Experiment]:
        return [copy.deepcopy(e) for e in self._experiments.values()]

    # -- internals --------------------------------------------------------
    def _run_round(self, experiment: Experiment, nodes: List[WorkerNode]) -> None:
        round_config = RoundConfig(
            experiment_id=experiment.experiment_id,
            round_number=experiment.current_round,
            global_model=experiment.global_model,
            learning_rate=experiment.config.learning_rate,
        )
        log.info("%s round %d: sending RoundConfig(global_model=%s) to %s",
                 experiment.experiment_id, round_config.round_number,
                 round_config.global_model, ", ".join(experiment.node_ids))

        round_updates = experiment.updates.setdefault(round_config.round_number, {})
        for node in nodes:
            update = node.train(round_config)
            self._validate(update, node, round_config)
            round_updates[node.node_id] = update
            log.info("%s round %d: ModelUpdate from %s (local_model=%s, samples=%d)",
                     experiment.experiment_id, round_config.round_number,
                     update.node_id, update.local_model, update.num_samples)

        experiment.global_model = self._strategy.aggregate(list(round_updates.values()))
        experiment.history.append(experiment.global_model)
        log.info("%s round %d: aggregated -> global_model=%s", experiment.experiment_id,
                 round_config.round_number, experiment.global_model)

    @staticmethod
    def _validate(update: ModelUpdate, node: WorkerNode, round_config: RoundConfig) -> None:
        """Reject updates that don't belong to this node/experiment/round."""
        if (update.node_id != node.node_id
                or update.experiment_id != round_config.experiment_id
                or update.round_number != round_config.round_number):
            raise ValueError(f"invalid update from node '{node.node_id}'")

    def _get(self, experiment_id: str) -> Experiment:
        try:
            return self._experiments[experiment_id]
        except KeyError:
            raise KeyError(f"unknown experiment '{experiment_id}'") from None
