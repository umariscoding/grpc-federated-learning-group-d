# Federated Learning with gRPC

A minimal federated learning platform, built in two phases:
a monolithic prototype (Phase 1), then a distributed version where independent
processes communicate only through gRPC (Phase 2).

| Phase | Content | Status |
|---|---|---|
| Phase 1 | Monolithic design and implementation (Python, single process) | Done |
| Phase 2 | Distributed version: Python Coordinator and CLI, Node.js Participants, gRPC and Protocol Buffers | Planned |

## What it does

Several participants each hold private local data. A coordinator sends them the
current global model, each participant trains locally and returns only its updated
parameters, and the coordinator averages them into a new global model. This repeats
for a configurable number of rounds. The data never leaves the participant.

Example from the project statement: participants A, B and C produce local
results 10, 20 and 30, so the global model is (10 + 20 + 30) / 3 = 20.

## Project structure

    .
    |-- README.md
    |-- phase1_monolith/
    |   `-- monolith.py                  the whole monolithic application
    `-- tests/
        |-- test_monolith.py             behavior, validation, failure and console tests
        `-- test_requirements.py         one test per requirement of the statement

The architecture and class diagrams are part of the final submission and are
added to the project at the end.

## Requirements

- Python 3.9 or newer
- No external packages

## Quick start

Run every command from the folder that contains this README.

Run the example scenario (A=10, B=20, C=30, result 20):

    python3 phase1_monolith/monolith.py

Run the tests:

    python3 -m unittest discover -s tests -v

Start the interactive console:

    python3 phase1_monolith/monolith.py --cli

### Console commands

| Command | Description |
|---|---|
| `register <node_id> <v1,v2,...> [address]` | Add a participant holding this local data |
| `nodes` | List registered participants |
| `create [rounds=N] [lr=X]` | Create an experiment (defaults: rounds=1, lr=1.0) |
| `start <exp_id>` | Run the experiment |
| `status <exp_id>` | Show status, global model, history and per-round updates |
| `list` | List experiments |
| `help`, `quit` | Help and exit |

Example session (the final global model is 24.375):

    fl> register A 10
    fl> register B 40,40,40
    fl> create rounds=2 lr=0.5
    fl> start exp-1
    fl> status exp-1

## Architecture

| Component | Responsibility |
|---|---|
| `AdminConsole` | User entry point: registers participants, creates and starts experiments, shows status |
| `FLServer` | Coordinator facade; its methods become the Coordinator RPCs in Phase 2 |
| `NodeRegistry` | Keeps track of registered participants |
| `ExperimentManager` | Experiment lifecycle and the round loop: distribute, train, validate, aggregate |
| `FedAvgStrategy` | Federated averaging, weighted by number of samples |
| `WorkerNode` | Participant: private data and local training |

Flow of one experiment:

1. `register` adds participants to the registry.
2. `create` and `start` create the experiment and begin the rounds.
3. For each round, a `RoundConfig` with the current global model is sent to every participant.
4. Each participant trains locally and returns a `ModelUpdate`.
5. The manager validates each update, then FedAvg aggregates them into the new global model.
6. After the last round the experiment is `COMPLETED`; `status` shows the result.

## Data structures

| Structure | Fields |
|---|---|
| `ExperimentStatus` | `CREATED`, `RUNNING`, `COMPLETED`, `FAILED` |
| `RoundConfig` (immutable) | `exp_id`, `round_num`, `learning_rate`, `global_weights` |
| `ModelUpdate` (immutable) | `exp_id`, `node_id`, `round_num`, `weights`, `num_samples` |
| `Experiment` | `exp_id`, `total_rounds`, `learning_rate`, `status`, `current_round`, `global_weights`, `node_ids`, `history`, `updates`, `error` |
| `WorkerNode` | `node_id`, `address`, private data |

## Design decisions

- **Push model.** The coordinator sends `RoundConfig` to each participant and receives a `ModelUpdate` back. In Phase 2 each participant is a gRPC server.
- **Simulated training.** `local = global + lr * (mean(data) - global)`. With `lr = 1` the result is the mean of the local data. With `lr < 1` and several rounds the global model converges gradually.
- **Weighted FedAvg.** Local models are averaged weighted by `num_samples`. With equal sample counts this is the plain mean.
- **Validation.** Experiment parameters (`rounds` is an integer >= 1, `lr` in (0, 1]), participant data (not empty, finite) and every returned update (right node, experiment, round, number of weights, samples > 0, finite values).
- **Failure handling.** A failing participant or an invalid update marks the experiment `FAILED` and stores the error. Earlier rounds are kept.
- **Snapshots.** Status queries return copies, never live state, like an RPC response.
- **Thread safety.** Training runs outside the lock, so `get_status` works while an experiment is `RUNNING`. This is needed in Phase 2, where training runs in a background thread.

## Testing

32 tests, all passing. `tests/test_requirements.py` maps each requirement of the statement to a test:

| Requirement | Test |
|---|---|
| Register different participants | `test_R1_register_different_participants` |
| Start a federated learning job | `test_R2_start_a_federated_learning_job` |
| Distribute a training configuration | `test_R3_distribute_training_configuration_to_all_participants` |
| Execute local training on each participant | `test_R4_local_training_runs_on_each_participant_with_its_own_data` |
| Receive the local results | `test_R5_receive_local_training_results` |
| Aggregate into a global model | `test_R6_aggregate_results_into_a_global_model` |
| Query status and result | `test_R7_query_status_and_result` |
| Example scenario 10, 20, 30 gives 20 | `test_example_scenario_through_the_console` |

`tests/test_monolith.py` also covers weighted averaging, multi-round convergence,
invalid input, crashing participants, tampered updates, snapshots, reading status while
an experiment runs, and the console commands.

## Phase 2 plan

| Phase 1 | Phase 2 |
|---|---|
| `AdminConsole` | Management Client: Python CLI, gRPC client of the Coordinator |
| `FLServer`, `NodeRegistry`, `ExperimentManager`, `FedAvgStrategy` | Coordinator Service: Python gRPC server |
| `WorkerNode` | Participant Service: Node.js gRPC server |
| `register_node` | `RegisterParticipant` |
| `create_experiment` | `CreateExperiment` |
| `start_experiment` | `StartTraining` |
| `get_status` | `GetTrainingStatus` |
| `WorkerNode.train` | `Train` (Coordinator calls the Participant) |

The proposed `GetTrainingTask` and `SubmitTrainingResult` operations are replaced
by a single `Train` call, because the coordinator pushes work to participants.
`RoundConfig`, `ModelUpdate`, `Experiment` and `ExperimentStatus` become Protocol Buffers
messages; `weights` becomes `ModelParameters`, and `node_id` plus `address` becomes `NodeInfo`.

## Limitations

- Training is simulated and the model is a plain list of numbers, as allowed by the statement.
- No persistence: everything lives in memory.
- A failing participant fails the whole experiment (no dropout tolerance).

