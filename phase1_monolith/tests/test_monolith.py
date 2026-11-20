import io
import unittest

from fl_monolith import (AdminConsole, ExperimentConfig, ExperimentManager,
                         ExperimentStatus, FedAvgStrategy, FLServer, ModelUpdate,
                         NodeRegistry, RoundConfig, WorkerNode)


def make_server(values=(10, 20, 30)):
    server = FLServer()
    for node_id, v in zip("ABC", values):
        server.register_node(WorkerNode(node_id, [v]))
    return server


class FaultyNode(WorkerNode):
    def train(self, round_config):
        raise RuntimeError("disk on fire")


class ImpostorNode(WorkerNode):
    """Answers with an update that claims to come from someone else."""
    def train(self, round_config):
        real = super().train(round_config)
        return ModelUpdate(real.experiment_id, real.round_number, "someone-else",
                           real.local_model, real.num_samples)


class ModelTests(unittest.TestCase):
    def test_config_validation(self):
        for kwargs in ({"rounds": 0}, {"learning_rate": 0}, {"learning_rate": 1.5}):
            with self.assertRaises(ValueError):
                ExperimentConfig(**kwargs)

    def test_update_needs_samples(self):
        with self.assertRaises(ValueError):
            ModelUpdate("e", 1, "A", 1.0, 0)


class StrategyTests(unittest.TestCase):
    def test_simple_average(self):
        ups = [ModelUpdate("e", 1, n, v, 1) for n, v in (("A", 10), ("B", 20), ("C", 30))]
        self.assertEqual(FedAvgStrategy().aggregate(ups), 20)

    def test_weighted_average(self):
        ups = [ModelUpdate("e", 1, "A", 10, 1), ModelUpdate("e", 1, "B", 40, 3)]
        self.assertEqual(FedAvgStrategy().aggregate(ups), 32.5)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            FedAvgStrategy().aggregate([])


class WorkerNodeTests(unittest.TestCase):
    def test_full_step_goes_to_local_mean(self):
        update = WorkerNode("A", [8, 10, 12]).train(RoundConfig("e", 1, 0.0, 1.0))
        self.assertEqual((update.local_model, update.num_samples), (10.0, 3))

    def test_partial_step(self):
        update = WorkerNode("A", [20]).train(RoundConfig("e", 1, 0.0, 0.5))
        self.assertEqual(update.local_model, 10.0)

    def test_invalid_nodes_rejected(self):
        with self.assertRaises(ValueError):
            WorkerNode("A", [])
        with self.assertRaises(ValueError):
            WorkerNode(" ", [1])

    def test_repr_hides_data(self):
        self.assertNotIn("42", repr(WorkerNode("A", [42])))


class RegistryTests(unittest.TestCase):
    def test_register_and_list(self):
        reg = NodeRegistry()
        reg.register_node(WorkerNode("A", [1]))
        reg.register_node(WorkerNode("B", [2]))
        self.assertEqual([n.node_id for n in reg.get_active_nodes()], ["A", "B"])
        self.assertEqual(len(reg), 2)

    def test_duplicate_rejected(self):
        reg = NodeRegistry()
        reg.register_node(WorkerNode("A", [1]))
        with self.assertRaises(ValueError):
            reg.register_node(WorkerNode("A", [2]))


class ExperimentTests(unittest.TestCase):
    def test_assignment_example_gives_20(self):
        server = make_server()
        exp = server.create_experiment()
        self.assertEqual(exp.status, ExperimentStatus.CREATED)
        server.start_experiment(exp.experiment_id)
        result = server.get_status(exp.experiment_id)
        self.assertEqual(result.status, ExperimentStatus.COMPLETED)
        self.assertEqual(result.global_model, 20)
        self.assertEqual(result.node_ids, ["A", "B", "C"])
        self.assertEqual({n: u.local_model for n, u in result.updates[1].items()},
                         {"A": 10, "B": 20, "C": 30})

    def test_weighted_by_sample_count_end_to_end(self):
        server = FLServer()
        server.register_node(WorkerNode("A", [10]))
        server.register_node(WorkerNode("B", [40, 40, 40]))
        exp = server.create_experiment()
        self.assertEqual(server.start_experiment(exp.experiment_id).global_model, 32.5)

    def test_multi_round_with_partial_learning_rate(self):
        server = make_server()
        exp = server.create_experiment(ExperimentConfig(rounds=5, learning_rate=0.5))
        result = server.start_experiment(exp.experiment_id)
        self.assertEqual(result.history[:2], [10.0, 15.0])
        self.assertAlmostEqual(result.global_model, 20 * (1 - 0.5 ** 5))
        self.assertEqual(result.current_round, 5)

    def test_cannot_start_without_nodes(self):
        server = FLServer()
        exp = server.create_experiment()
        with self.assertRaises(ValueError):
            server.start_experiment(exp.experiment_id)

    def test_cannot_start_twice(self):
        server = make_server()
        exp = server.create_experiment()
        server.start_experiment(exp.experiment_id)
        with self.assertRaises(ValueError):
            server.start_experiment(exp.experiment_id)

    def test_unknown_experiment(self):
        with self.assertRaises(KeyError):
            make_server().get_status("nope")

    def test_status_returns_snapshot(self):
        server = make_server()
        exp = server.create_experiment()
        server.start_experiment(exp.experiment_id)
        server.get_status(exp.experiment_id).history.append(999)
        self.assertEqual(server.get_status(exp.experiment_id).history, [20.0])

    def test_failing_node_fails_experiment(self):
        server = make_server()
        server.register_node(FaultyNode("D", [1]))
        exp = server.create_experiment()
        result = server.start_experiment(exp.experiment_id)
        self.assertEqual(result.status, ExperimentStatus.FAILED)
        self.assertIn("disk on fire", result.error)
        self.assertEqual(result.history, [])  # nothing was aggregated

    def test_impostor_update_rejected(self):
        server = make_server()
        server.register_node(ImpostorNode("D", [1]))
        result = server.start_experiment(server.create_experiment().experiment_id)
        self.assertEqual(result.status, ExperimentStatus.FAILED)
        self.assertIn("invalid update", result.error)

    def test_experiments_are_independent(self):
        server = make_server()
        e1 = server.create_experiment().experiment_id
        e2 = server.create_experiment(ExperimentConfig(rounds=2, learning_rate=0.5)).experiment_id
        server.start_experiment(e1)
        server.start_experiment(e2)
        self.assertEqual(server.get_status(e1).global_model, 20)
        self.assertEqual(server.get_status(e2).global_model, 15.0)
        self.assertEqual([e.experiment_id for e in server.list_experiments()], [e1, e2])

    def test_manager_accepts_custom_strategy(self):
        class Max(FedAvgStrategy):
            def aggregate(self, updates):
                return max(u.local_model for u in updates)
        server = FLServer(strategy=Max())
        for n, v in zip("AB", (5, 9)):
            server.register_node(WorkerNode(n, [v]))
        exp = server.create_experiment()
        self.assertEqual(server.start_experiment(exp.experiment_id).global_model, 9)


class ConsoleTests(unittest.TestCase):
    def run_console(self, *lines):
        out = io.StringIO()
        console = AdminConsole(FLServer(), stdout=out)
        for line in lines:
            console.onecmd(line)
        return out.getvalue()

    def test_full_flow(self):
        out = self.run_console("register A 10", "register B 20", "register C 30",
                               "create", "start exp-1", "status exp-1")
        self.assertIn("exp-1 finished with status COMPLETED", out)
        self.assertIn("global_model: 20.0", out)

    def test_multiple_values_and_commas(self):
        out = self.run_console("register A 1,2,3", "nodes")
        self.assertIn("registered node A (3 samples)", out)
        self.assertIn("nodes: A", out)

    def test_errors_are_reported_not_raised(self):
        out = self.run_console("register A", "register A x", "start exp-9", "create rounds=0",
                               "create foo=1", "status", "bogus")
        self.assertEqual(out.count("error:"), 6)
        self.assertIn("unknown command", out)

    def test_quit_stops_loop(self):
        console = AdminConsole(FLServer(), stdout=io.StringIO())
        self.assertTrue(console.onecmd("quit"))


if __name__ == "__main__":
    unittest.main()
