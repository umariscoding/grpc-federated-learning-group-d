import dataclasses
import logging
import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase1_monolith"))
from monolith import (AdminConsole, ExperimentStatus, FedAvgStrategy, FLServer,  # noqa: E402
                      ModelUpdate, WorkerNode)

NAN = float("nan")


def setUpModule():
    logging.disable(logging.CRITICAL)       # keep test output clean


def make_server(**nodes):
    s = FLServer()
    for node_id, data in nodes.items():
        s.register_node(WorkerNode(node_id, "localhost", data))
    return s


def run(s, rounds=1, lr=1.0):
    return s.start_experiment(s.create_experiment(rounds, lr))


class FailingNode(WorkerNode):
    """Raises during training in a given round."""
    def __init__(self, *args, fail_round=1):
        super().__init__(*args)
        self.fail_round = fail_round

    def train(self, config):
        if config.round_num == self.fail_round:
            raise RuntimeError("disk on fire")
        return super().train(config)


class BadReplyNode(WorkerNode):
    """Answers with a tampered ModelUpdate."""
    def __init__(self, *args, tweak):
        super().__init__(*args)
        self.tweak = tweak

    def train(self, config):
        return dataclasses.replace(super().train(config), **self.tweak)


class GatedNode(WorkerNode):
    """Blocks inside train() until released, so a test can observe RUNNING."""
    def __init__(self, *args):
        super().__init__(*args)
        self.entered, self.release = threading.Event(), threading.Event()

    def train(self, config):
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("test timeout")
        return super().train(config)


class TrainingTests(unittest.TestCase):
    def test_spec_example(self):
        exp = run(make_server(A=[10.0], B=[20.0], C=[30.0]))
        self.assertEqual(exp.status, ExperimentStatus.COMPLETED)
        self.assertAlmostEqual(exp.global_weights[0], 20.0)

    def test_weighted_average(self):
        exp = run(make_server(A=[10.0], B=[40.0, 40.0, 40.0]))
        self.assertAlmostEqual(exp.global_weights[0], 32.5)

    def test_multi_round_converges_and_keeps_history(self):
        exp = run(make_server(A=[10.0], B=[20.0], C=[30.0]), rounds=3, lr=0.5)
        self.assertEqual([round(h[0], 4) for h in exp.history], [10.0, 15.0, 17.5])
        self.assertEqual(exp.current_round, 3)

    def test_updates_are_stored_per_round_and_node(self):
        exp = run(make_server(A=[10.0], B=[20.0]), rounds=2, lr=0.5)
        self.assertEqual(sorted(exp.updates), [1, 2])
        self.assertEqual(sorted(exp.updates[1]), ["A", "B"])
        self.assertEqual(exp.node_ids, ["A", "B"])

    def test_participant_data_is_not_exposed_in_results(self):
        exp = run(make_server(A=[1.0, 2.0, 3.0]))
        self.assertNotIn("_data", dataclasses.asdict(exp.updates[1]["A"]))
        self.assertEqual(exp.updates[1]["A"].num_samples, 3)

    def test_experiment_ids_are_sequential(self):
        s = make_server(A=[1.0])
        self.assertEqual([s.create_experiment(1, 1.0), s.create_experiment(1, 1.0)], ["exp-1", "exp-2"])


class ValidationTests(unittest.TestCase):
    def test_invalid_experiment_parameters(self):
        s = FLServer()
        for rounds, lr in [(0, 1.0), (-1, 1.0), (True, 1.0), (1.5, 1.0),
                           (1, 0), (1, -0.1), (1, 1.5), (1, NAN)]:
            with self.subTest(rounds=rounds, lr=lr), self.assertRaises(ValueError):
                s.create_experiment(rounds, lr)

    def test_invalid_workers(self):
        for node_id, data in [("", [1.0]), ("X", []), ("X", [NAN]), ("X", [float("inf")])]:
            with self.subTest(node_id=node_id, data=data), self.assertRaises(ValueError):
                WorkerNode(node_id, "localhost", data)

    def test_cannot_start_without_participants(self):
        s = FLServer()
        with self.assertRaises(ValueError):
            s.start_experiment(s.create_experiment(1, 1.0))

    def test_cannot_start_twice(self):
        s = make_server(A=[1.0])
        exp_id = s.create_experiment(1, 1.0)
        s.start_experiment(exp_id)
        with self.assertRaises(ValueError):
            s.start_experiment(exp_id)

    def test_duplicate_participant_rejected_and_original_kept(self):
        s = make_server(A=[1.0])
        self.assertFalse(s.register_node(WorkerNode("A", "other", [9.0])))
        self.assertEqual([n.address for n in s.list_nodes()], ["localhost"])

    def test_unknown_experiment(self):
        s = FLServer()
        for call in (s.get_status, s.start_experiment):
            with self.assertRaises(KeyError):
                call("nope")


class FailureTests(unittest.TestCase):
    def test_crashing_participant_fails_experiment_and_keeps_earlier_rounds(self):
        s = FLServer()
        s.register_node(WorkerNode("A", "localhost", [10.0]))
        s.register_node(FailingNode("B", "localhost", [20.0], fail_round=2))
        exp = run(s, rounds=3, lr=0.5)
        self.assertEqual(exp.status, ExperimentStatus.FAILED)
        self.assertIn("RuntimeError: disk on fire", exp.error)
        self.assertEqual(len(exp.history), 1)           # round 1 finished, round 2 did not

    def test_tampered_updates_are_rejected(self):
        tweaks = [{"node_id": "Z"}, {"round_num": 99}, {"exp_id": "other"},
                  {"weights": (1.0, 2.0)}, {"num_samples": 0}, {"weights": (NAN,)}]
        for tweak in tweaks:
            with self.subTest(tweak=tweak):
                s = FLServer()
                s.register_node(BadReplyNode("A", "localhost", [10.0], tweak=tweak))
                exp = run(s)
                self.assertEqual(exp.status, ExperimentStatus.FAILED)
                self.assertIn("invalid update from node 'A'", exp.error)
                self.assertEqual(exp.history, [])

    def test_failed_experiment_cannot_be_restarted(self):
        s = FLServer()
        s.register_node(FailingNode("A", "localhost", [1.0]))
        exp_id = s.create_experiment(1, 1.0)
        s.start_experiment(exp_id)
        with self.assertRaises(ValueError):
            s.start_experiment(exp_id)


class IsolationTests(unittest.TestCase):
    def test_status_returns_a_snapshot(self):
        s = make_server(A=[10.0])
        exp_id = s.create_experiment(1, 1.0)
        snapshot = s.get_status(exp_id)
        snapshot.status = ExperimentStatus.COMPLETED
        snapshot.node_ids.append("hacker")
        fresh = s.get_status(exp_id)
        self.assertEqual(fresh.status, ExperimentStatus.CREATED)
        self.assertEqual(fresh.node_ids, [])

    def test_status_is_readable_while_running(self):
        node = GatedNode("A", "localhost", [10.0])
        s = FLServer()
        s.register_node(node)
        exp_id = s.create_experiment(1, 1.0)
        t = threading.Thread(target=s.start_experiment, args=(exp_id,))
        t.start()
        self.assertTrue(node.entered.wait(5))
        status = s.get_status(exp_id)                   # must not block
        self.assertEqual((status.status, status.current_round), (ExperimentStatus.RUNNING, 1))
        with self.assertRaises(ValueError):             # cannot be started again while running
            s.start_experiment(exp_id)
        node.release.set()
        t.join(5)
        self.assertEqual(s.get_status(exp_id).status, ExperimentStatus.COMPLETED)


class StrategyTests(unittest.TestCase):
    def update(self, weights, n):
        return ModelUpdate("e", "x", 1, tuple(weights), n)

    def test_weighted_vector_average(self):
        result = FedAvgStrategy().aggregate([self.update([1.0, 10.0], 1), self.update([3.0, 20.0], 3)])
        self.assertEqual(result, (2.5, 17.5))

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            FedAvgStrategy().aggregate([])
        with self.assertRaises(ValueError):
            FedAvgStrategy().aggregate([self.update([1.0], 1), self.update([1.0, 2.0], 1)])


class ConsoleTests(unittest.TestCase):
    def setUp(self):
        self.console = AdminConsole(FLServer())

    def test_full_cli_flow(self):
        c = self.console
        for line in ("register A 10", "register B 20", "register C 30"):
            self.assertTrue(c.execute(line).startswith("registered node"))
        self.assertEqual(c.execute("nodes"), "nodes: A, B, C")
        self.assertEqual(c.execute("create rounds=1 lr=1.0"), "created exp-1")
        self.assertEqual(c.execute("start exp-1"), "exp-1 finished with status COMPLETED")
        status = c.execute("status exp-1")
        self.assertIn("status:       COMPLETED", status)
        self.assertIn("global_model: [20.0]", status)
        self.assertIn("exp-1  COMPLETED", c.execute("list"))

    def test_defaults_and_duplicates(self):
        c = self.console
        c.execute("register A 10")
        self.assertEqual(c.execute("register A 99"), "node A is already registered")
        self.assertEqual(c.execute("create"), "created exp-1")

    def test_bad_input_gives_messages_not_crashes(self):
        c = self.console
        for line, expected in [("bogus", "unknown command"), ("register A", "usage: register"),
                               ("register A x,y", "error:"), ("create rounds=0", "error:"),
                               ("create foo=1", "usage: create"), ("start", "usage: start"),
                               ("start exp-9", "unknown experiment 'exp-9'"),
                               ("status exp-9", "unknown experiment 'exp-9'")]:
            with self.subTest(line=line):
                self.assertIn(expected, c.execute(line))
        self.assertEqual(c.execute("   "), "")
        self.assertEqual(c.execute("nodes"), "no nodes registered")
        self.assertEqual(c.execute("list"), "no experiments")

    def test_start_without_nodes_reports_error(self):
        c = self.console
        c.execute("create")
        self.assertIn("no participants registered", c.execute("start exp-1"))

    def test_failed_experiment_shows_error(self):
        server = FLServer()
        server.register_node(FailingNode("A", "localhost", [1.0]))
        c = AdminConsole(server)
        c.execute("create")
        self.assertIn("FAILED: RuntimeError: disk on fire", c.execute("start exp-1"))
        self.assertIn("error:        RuntimeError: disk on fire", c.execute("status exp-1"))


if __name__ == "__main__":
    unittest.main()
