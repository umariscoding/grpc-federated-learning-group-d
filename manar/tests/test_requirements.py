"""One test per Phase 1 requirement of the project statement."""
import logging
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase1_monolith"))
from monolith import AdminConsole, ExperimentStatus, FLServer, WorkerNode  # noqa: E402


def setUpModule():
    logging.disable(logging.CRITICAL)


class SpyNode(WorkerNode):
    """A normal participant that records every RoundConfig it receives."""
    def __init__(self, *args):
        super().__init__(*args)
        self.received = []

    def train(self, config):
        self.received.append(config)
        return super().train(config)


def server_with(**data):
    s = FLServer()
    nodes = {n: SpyNode(n, "localhost", d) for n, d in data.items()}
    for node in nodes.values():
        s.register_node(node)
    return s, nodes


class Phase1Requirements(unittest.TestCase):
    def test_R1_register_different_participants(self):
        s, _ = server_with(A=[10.0], B=[20.0], C=[30.0])
        self.assertEqual([n.node_id for n in s.list_nodes()], ["A", "B", "C"])

    def test_R2_start_a_federated_learning_job(self):
        s, _ = server_with(A=[10.0])
        exp_id = s.create_experiment(1, 1.0)
        self.assertEqual(s.get_status(exp_id).status, ExperimentStatus.CREATED)
        self.assertEqual(s.start_experiment(exp_id).status, ExperimentStatus.COMPLETED)

    def test_R3_distribute_training_configuration_to_all_participants(self):
        s, nodes = server_with(A=[10.0], B=[20.0], C=[30.0])
        exp = s.start_experiment(s.create_experiment(2, 0.5))
        for node in nodes.values():
            self.assertEqual(len(node.received), 2)                    # one config per round
        for r in (0, 1):                                              # same config for everyone
            self.assertEqual(len({n.received[r] for n in nodes.values()}), 1)
        self.assertEqual(nodes["A"].received[0].global_weights, (0.0,))
        self.assertEqual(nodes["A"].received[1].global_weights, exp.history[0])   # round 2 starts from round 1 result

    def test_R4_local_training_runs_on_each_participant_with_its_own_data(self):
        s, nodes = server_with(A=[10.0, 10.0], B=[20.0, 40.0, 60.0])
        exp = s.start_experiment(s.create_experiment(1, 1.0))
        self.assertEqual(exp.updates[1]["A"].weights, (10.0,))        # mean of A's data
        self.assertEqual(exp.updates[1]["B"].weights, (40.0,))        # mean of B's data

    def test_R5_receive_local_training_results(self):
        s, _ = server_with(A=[10.0], B=[20.0, 20.0, 20.0])
        exp = s.start_experiment(s.create_experiment(1, 1.0))
        self.assertEqual(sorted(exp.updates[1]), ["A", "B"])
        self.assertEqual((exp.updates[1]["A"].num_samples, exp.updates[1]["B"].num_samples), (1, 3))

    def test_R6_aggregate_results_into_a_global_model(self):
        s, _ = server_with(A=[10.0], B=[20.0], C=[30.0])
        exp = s.start_experiment(s.create_experiment(1, 1.0))
        self.assertAlmostEqual(exp.global_weights[0], 20.0)           # (10 + 20 + 30) / 3

    def test_R7_query_status_and_result(self):
        s, _ = server_with(A=[10.0], B=[20.0], C=[30.0])
        exp_id = s.create_experiment(1, 1.0)
        s.start_experiment(exp_id)
        status = s.get_status(exp_id)
        self.assertEqual((status.status, status.current_round), (ExperimentStatus.COMPLETED, 1))
        self.assertAlmostEqual(status.global_weights[0], 20.0)

    def test_example_scenario_through_the_console(self):
        c = AdminConsole(FLServer())
        for line in ("register A 10", "register B 20", "register C 30", "create", "start exp-1"):
            c.execute(line)
        self.assertIn("global_model: [20.0]", c.execute("status exp-1"))


if __name__ == "__main__":
    unittest.main()
