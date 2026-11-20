"""AdminConsole: the management CLI. Talks to the FLServer only through its public API."""
from __future__ import annotations

import cmd
from typing import IO, Optional

from .models import Experiment, ExperimentConfig
from .server import FLServer
from .worker_node import WorkerNode


def format_experiment(exp: Experiment) -> str:
    lines = [
        f"experiment:   {exp.experiment_id}",
        f"status:       {exp.status.value}",
        f"nodes:        {', '.join(exp.node_ids) or '-'}",
        f"round:        {exp.current_round}/{exp.config.rounds}",
        f"global_model: {exp.global_model}",
        f"history:      {exp.history}",
    ]
    for round_number, updates in sorted(exp.updates.items()):
        parts = ", ".join(f"{nid}={u.local_model}" for nid, u in updates.items())
        lines.append(f"  round {round_number} updates: {parts}")
    if exp.error:
        lines.append(f"error:        {exp.error}")
    return "\n".join(lines)


class AdminConsole(cmd.Cmd):
    prompt = "fl> "
    intro = "Federated Learning admin console. Type 'help' for commands."

    def __init__(self, server: FLServer, stdin: Optional[IO[str]] = None,
                 stdout: Optional[IO[str]] = None):
        super().__init__(stdin=stdin, stdout=stdout)
        self.use_rawinput = stdin is None
        self._server = server

    # -- plumbing ---------------------------------------------------------
    def _say(self, text: str) -> None:
        print(text, file=self.stdout)

    def onecmd(self, line: str) -> bool:
        """Report user errors instead of crashing the console."""
        try:
            return super().onecmd(line)
        except (ValueError, KeyError) as exc:
            self._say(f"error: {exc.args[0] if exc.args else exc}")
            return False

    def emptyline(self) -> bool:
        return False

    def default(self, line: str) -> None:
        self._say(f"unknown command: {line.split()[0]!r} (type 'help')")

    # -- commands ---------------------------------------------------------
    def do_register(self, arg: str) -> None:
        """register <node_id> <value> [value ...]  - register a node with its private data."""
        parts = arg.replace(",", " ").split()
        if len(parts) < 2:
            raise ValueError("usage: register <node_id> <value> [value ...]")
        try:
            data = [float(p) for p in parts[1:]]
        except ValueError:
            raise ValueError("data values must be numbers") from None
        self._server.register_node(WorkerNode(parts[0], data))
        self._say(f"registered node {parts[0]} ({len(data)} samples)")

    def do_nodes(self, arg: str) -> None:
        """nodes  - list registered nodes."""
        nodes = self._server.get_active_nodes()
        self._say("nodes: " + (", ".join(nodes) if nodes else "(none)"))

    def do_create(self, arg: str) -> None:
        """create [rounds=N] [lr=X]  - create an experiment (defaults: rounds=1 lr=1.0)."""
        options = {"rounds": 1, "lr": 1.0}
        for token in arg.split():
            key, sep, value = token.partition("=")
            if not sep or key not in options:
                raise ValueError("usage: create [rounds=N] [lr=X]")
            try:
                options[key] = int(value) if key == "rounds" else float(value)
            except ValueError:
                raise ValueError(f"invalid value for {key}: {value!r}") from None
        exp = self._server.create_experiment(
            ExperimentConfig(rounds=options["rounds"], learning_rate=options["lr"]))
        self._say(f"created {exp.experiment_id}")

    def do_start(self, arg: str) -> None:
        """start <experiment_id>  - run the experiment to completion."""
        exp = self._server.start_experiment(self._one_arg(arg, "start"))
        self._say(f"{exp.experiment_id} finished with status {exp.status.value}")

    def do_status(self, arg: str) -> None:
        """status <experiment_id>  - show an experiment's status and result."""
        self._say(format_experiment(self._server.get_status(self._one_arg(arg, "status"))))

    def do_list(self, arg: str) -> None:
        """list  - list all experiments."""
        experiments = self._server.list_experiments()
        if not experiments:
            self._say("(no experiments)")
        for exp in experiments:
            self._say(f"{exp.experiment_id}  {exp.status.value}  global_model={exp.global_model}")

    def do_quit(self, arg: str) -> bool:
        """quit  - leave the console."""
        return True

    do_exit = do_quit
    do_EOF = do_quit

    @staticmethod
    def _one_arg(arg: str, command: str) -> str:
        parts = arg.split()
        if len(parts) != 1:
            raise ValueError(f"usage: {command} <experiment_id>")
        return parts[0]
