"""Phase 1: monolithic federated learning prototype (single process).

Components follow the architecture diagram:
AdminConsole -> FLServer (NodeRegistry, ExperimentManager, FedAvgStrategy) -> WorkerNode

Messages (RoundConfig, ModelUpdate) are immutable, like Protobuf messages.
Calls marked [RPC] become gRPC calls between separate processes in Phase 2.

  python3 phase1_monolith/monolith.py          run the spec scenario (10, 20, 30 -> 20)
  python3 phase1_monolith/monolith.py --cli    interactive console
"""
from __future__ import annotations

import copy
import itertools
import logging
import math
import sys
import threading
from dataclasses import dataclass, field
from enum import Enum

Weights = tuple[float, ...]


# --------- data structures -----------
class ExperimentStatus(Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class RoundConfig:
    """Coordinator -> participant: what to train on."""
    exp_id: str
    round_num: int
    learning_rate: float
    global_weights: Weights


@dataclass(frozen=True)
class ModelUpdate:
    """Participant -> coordinator: the result of local training."""
    exp_id: str
    node_id: str
    round_num: int
    weights: Weights
    num_samples: int


@dataclass
class Experiment:
    exp_id: str
    total_rounds: int
    learning_rate: float
    status: ExperimentStatus = ExperimentStatus.CREATED
    current_round: int = 0
    global_weights: Weights = (0.0,)
    node_ids: list[str] = field(default_factory=list)
    history: list[Weights] = field(default_factory=list)                    # global model per round
    updates: dict[int, dict[str, ModelUpdate]] = field(default_factory=dict)  # round -> node -> update
    error: str = ""


# ----------- WorkerNode -----------------
class WorkerNode:
    """A participant. Its data is private: only a ModelUpdate ever leaves it."""

    def __init__(self, node_id: str, address: str, data: list[float]):
        if not node_id:
            raise ValueError("node_id must not be empty")
        if not data:
            raise ValueError("a worker needs at least one data sample")
        if not all(math.isfinite(x) for x in data):
            raise ValueError("data samples must be finite numbers")
        self.node_id = node_id
        self.address = address            # unused here; Phase 2 pushes Train requests to it
        self._data = list(data)
        self._log = logging.getLogger("fl.worker")

    def train(self, config: RoundConfig) -> ModelUpdate:
        """Simulated training: move every weight toward the mean of the local data."""
        target = sum(self._data) / len(self._data)
        weights = tuple(w + config.learning_rate * (target - w) for w in config.global_weights)
        self._log.info("%s round %d: trained on %d samples -> %s", self.node_id,
                       config.round_num, len(self._data), _fmt(weights))
        return ModelUpdate(config.exp_id, self.node_id, config.round_num, weights, len(self._data))


# ------------ NodeRegistry ------------------------
class NodeRegistry:
    def __init__(self):
        self._nodes: dict[str, WorkerNode] = {}
        self._lock = threading.Lock()

    def register(self, node: WorkerNode) -> bool:
        with self._lock:
            if node.node_id in self._nodes:
                return False
            self._nodes[node.node_id] = node
        logging.getLogger("fl.registry").info("registered node %s (%s)", node.node_id, node.address)
        return True

    def get_active_nodes(self) -> list[WorkerNode]:
        with self._lock:
            return list(self._nodes.values())


# -------------- FedAvgStrategy ------------------
class FedAvgStrategy:
    def aggregate(self, updates: list[ModelUpdate]) -> Weights:
        """Mean of the local weights, weighted by num_samples."""
        if not updates:
            raise ValueError("cannot aggregate an empty list of updates")
        size = len(updates[0].weights)
        if any(len(u.weights) != size for u in updates):
            raise ValueError("all updates must have the same number of weights")
        total = sum(u.num_samples for u in updates)
        if total <= 0:
            raise ValueError("total number of samples must be positive")
        return tuple(sum(u.weights[i] * u.num_samples for u in updates) / total
                     for i in range(size))


# ------------ ExperimentManager ------------
class ExperimentManager:
    """Owns all experiments. Returns snapshots (copies), never live state.

    Training runs outside the lock, so get_status() works while an experiment is
    RUNNING. This is what lets Phase 2 run training in a background thread.
    """

    def __init__(self, registry: NodeRegistry, strategy: FedAvgStrategy):
        self._registry = registry
        self._strategy = strategy
        self._experiments: dict[str, Experiment] = {}
        self._ids = itertools.count(1)
        self._lock = threading.RLock()
        self._log = logging.getLogger("fl.manager")

    def create_experiment(self, rounds: int, lr: float) -> str:
        if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
            raise ValueError("rounds must be an integer >= 1")
        if not 0 < lr <= 1:
            raise ValueError("learning rate must be in (0, 1]")
        with self._lock:
            exp_id = f"exp-{next(self._ids)}"
            self._experiments[exp_id] = Experiment(exp_id, rounds, lr)
        self._log.info("created %s (rounds=%d, lr=%s)", exp_id, rounds, lr)
        return exp_id

    def start_experiment(self, exp_id: str) -> Experiment:
        """Run all rounds. A failing participant makes the experiment FAILED."""
        with self._lock:
            exp = self._get(exp_id)
            if exp.status is not ExperimentStatus.CREATED:
                raise ValueError(f"{exp_id} cannot be started (status={exp.status.value})")
            nodes = self._registry.get_active_nodes()
            if not nodes:
                raise ValueError("cannot start an experiment: no participants registered")
            exp.node_ids = [n.node_id for n in nodes]
            exp.status = ExperimentStatus.RUNNING
        self._log.info("started %s with %d participant(s)", exp_id, len(nodes))
        try:
            for round_num in range(1, exp.total_rounds + 1):
                self._run_round(exp, nodes, round_num)
            with self._lock:
                exp.status = ExperimentStatus.COMPLETED
            self._log.info("%s COMPLETED, global model = %s", exp_id, _fmt(exp.global_weights))
        except Exception as e:  # participant code is arbitrary: any error fails the experiment
            with self._lock:
                exp.status = ExperimentStatus.FAILED
                exp.error = f"{type(e).__name__}: {e}"
            self._log.error("%s FAILED: %s", exp_id, exp.error)
        return self.get_status(exp_id)

    def get_status(self, exp_id: str) -> Experiment:
        with self._lock:
            return copy.deepcopy(self._get(exp_id))

    def list_experiments(self) -> list[Experiment]:
        with self._lock:
            return [copy.deepcopy(e) for e in self._experiments.values()]

    def _run_round(self, exp: Experiment, nodes: list[WorkerNode], round_num: int) -> None:
        """Distribute -> train -> validate -> aggregate."""
        with self._lock:
            exp.current_round = round_num
            config = RoundConfig(exp.exp_id, round_num, exp.learning_rate, exp.global_weights)
        self._log.info("%s round %d: sending RoundConfig(global=%s) to %s", exp.exp_id,
                       round_num, _fmt(config.global_weights), ", ".join(exp.node_ids))
        updates: dict[str, ModelUpdate] = {}
        for node in nodes:
            update = node.train(config)                       # [RPC] Train
            self._validate(update, node, config)
            updates[node.node_id] = update
        new_global = self._strategy.aggregate(list(updates.values()))
        with self._lock:
            exp.updates[round_num] = updates
            exp.global_weights = new_global
            exp.history.append(new_global)
        self._log.info("%s round %d: aggregated -> global=%s", exp.exp_id, round_num, _fmt(new_global))

    @staticmethod
    def _validate(update: ModelUpdate, node: WorkerNode, config: RoundConfig) -> None:
        """Reject an update that does not answer the request that was sent."""
        if (update.node_id != node.node_id
                or update.exp_id != config.exp_id
                or update.round_num != config.round_num
                or len(update.weights) != len(config.global_weights)
                or update.num_samples <= 0
                or not all(math.isfinite(w) for w in update.weights)):
            raise ValueError(f"invalid update from node '{node.node_id}'")

    def _get(self, exp_id: str) -> Experiment:
        if exp_id not in self._experiments:
            raise KeyError(f"unknown experiment '{exp_id}'")
        return self._experiments[exp_id]


# ------ FLServer --------
class FLServer:
    """Coordinator facade. Its methods become the Coordinator RPCs in Phase 2."""

    def __init__(self):
        self.registry = NodeRegistry()
        self.manager = ExperimentManager(self.registry, FedAvgStrategy())

    def register_node(self, node: WorkerNode) -> bool:                  # [RPC] RegisterParticipant
        return self.registry.register(node)

    def list_nodes(self) -> list[WorkerNode]:
        return self.registry.get_active_nodes()

    def create_experiment(self, rounds: int, lr: float) -> str:         # [RPC] CreateExperiment
        return self.manager.create_experiment(rounds, lr)

    def start_experiment(self, exp_id: str) -> Experiment:              # [RPC] StartTraining
        return self.manager.start_experiment(exp_id)

    def get_status(self, exp_id: str) -> Experiment:                    # [RPC] GetTrainingStatus
        return self.manager.get_status(exp_id)

    def list_experiments(self) -> list[Experiment]:
        return self.manager.list_experiments()


# ------ AdminConsole --------
HELP = """commands:
  register <node_id> <v1,v2,...> [address]   add a participant holding this local data
  nodes                                       list registered participants
  create [rounds=N] [lr=X]                    create an experiment (defaults: rounds=1 lr=1.0)
  start <exp_id>                              run the experiment
  status <exp_id>                             show status and results
  list                                        list experiments
  help | quit"""


class AdminConsole:
    """User entry point. Becomes the Management Client (CLI) in Phase 2."""

    def __init__(self, server: FLServer):
        self._server = server
        self._commands = {"register": self._register, "nodes": self._nodes,
                          "create": self._create, "start": self._start,
                          "status": self._status, "list": self._list,
                          "help": lambda args: HELP}

    def execute(self, line: str) -> str:
        """Run one command and return its output. Never raises on bad input."""
        parts = line.split()
        if not parts:
            return ""
        handler = self._commands.get(parts[0].lower())
        if handler is None:
            return f"unknown command '{parts[0]}', type 'help'"
        try:
            return handler(parts[1:])
        except (ValueError, KeyError) as e:
            return f"error: {e.args[0] if isinstance(e, KeyError) else e}"

    def _register(self, args: list[str]) -> str:
        if len(args) not in (2, 3):
            raise ValueError("usage: register <node_id> <v1,v2,...> [address]")
        data = [float(x) for x in args[1].split(",")]
        address = args[2] if len(args) == 3 else f"localhost:{50051 + len(self._server.list_nodes())}"
        node = WorkerNode(args[0], address, data)
        if not self._server.register_node(node):
            return f"node {node.node_id} is already registered"
        return f"registered node {node.node_id} ({len(data)} samples, {address})"

    def _nodes(self, args: list[str]) -> str:
        nodes = self._server.list_nodes()
        return "nodes: " + ", ".join(n.node_id for n in nodes) if nodes else "no nodes registered"

    def _create(self, args: list[str]) -> str:
        options = {"rounds": "1", "lr": "1.0"}
        for arg in args:
            key, sep, value = arg.partition("=")
            if not sep or key not in options:
                raise ValueError("usage: create [rounds=N] [lr=X]")
            options[key] = value
        return f"created {self._server.create_experiment(int(options['rounds']), float(options['lr']))}"

    def _start(self, args: list[str]) -> str:
        if len(args) != 1:
            raise ValueError("usage: start <exp_id>")
        exp = self._server.start_experiment(args[0])
        return f"{exp.exp_id} finished with status {exp.status.value}" + (f": {exp.error}" if exp.error else "")

    def _status(self, args: list[str]) -> str:
        if len(args) != 1:
            raise ValueError("usage: status <exp_id>")
        e = self._server.get_status(args[0])
        lines = [f"experiment:   {e.exp_id}",
                 f"status:       {e.status.value}",
                 f"nodes:        {', '.join(e.node_ids) or '-'}",
                 f"round:        {e.current_round}/{e.total_rounds}",
                 f"global_model: {_fmt(e.global_weights)}",
                 f"history:      {[_fmt(h) for h in e.history]}"]
        for r, ups in sorted(e.updates.items()):
            lines.append(f"  round {r} updates: " + ", ".join(f"{n}={_fmt(u.weights)}" for n, u in ups.items()))
        if e.error:
            lines.append(f"error:        {e.error}")
        return "\n".join(lines)

    def _list(self, args: list[str]) -> str:
        exps = self._server.list_experiments()
        return "\n".join(f"{e.exp_id}  {e.status.value}  round {e.current_round}/{e.total_rounds}"
                         for e in exps) if exps else "no experiments"

    def run_demo(self) -> None:
        """Spec scenario: A=10, B=20, C=30 -> global model 20."""
        for line in ("register A 10,10,10", "register B 20,20,20", "register C 30,30,30",
                     "create rounds=1 lr=1.0", "start exp-1", "status exp-1"):
            print(f"fl> {line}")
            print(self.execute(line))

    def start_cli(self) -> None:
        print(HELP)
        while True:
            try:
                line = input("fl> ")
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if line.strip().lower() in ("quit", "exit"):
                return
            output = self.execute(line)
            if output:
                print(output)


def _fmt(weights: Weights) -> list[float]:
    """Readable weights for logs and output (hides floating-point noise)."""
    return [round(w, 4) for w in weights]


def main(argv: list[str]) -> None:
    logging.basicConfig(level=logging.INFO, format="    [%(name)s] %(message)s", stream=sys.stdout)
    console = AdminConsole(FLServer())
    console.start_cli() if "--cli" in argv else console.run_demo()


if __name__ == "__main__":
    main(sys.argv[1:])
