# Phase 1 - Monolithic Federated Learning

Everything runs in **one Python process** (standard library only, Python 3.8+).
The components follow the team's architecture diagram.

## Run

```bash
cd phase1_monolith
python3 main.py                       # scripted demo: A=10, B=20, C=30 -> global model 20
python3 main.py --rounds 5 --lr 0.5   # multi-round demo (converges toward 20)
python3 main.py shell                 # interactive admin console (add -v to see server logs)
python3 -m unittest -v                # run the tests
```

### Console commands

| Command | Meaning |
|---|---|
| `register <node_id> <v> [v ...]` | register a worker node with its private data |
| `nodes` | list registered nodes |
| `create [rounds=N] [lr=X]` | create an experiment |
| `start <experiment_id>` | run it to completion |
| `status <experiment_id>` | show status, per-round updates and the global model |
| `list` | list experiments |
| `quit` | exit |

## How to see that it works

`python3 main.py` must end with `status: COMPLETED` and `global_model: 20.0`,
with the three local updates `A=10.0, B=20.0, C=30.0` in the log.

## Components

| Diagram | Code | Phase 2 equivalent |
|---|---|---|
| AdminConsole | `fl_monolith/admin_console.py` | Management Client |
| FLServer | `fl_monolith/server.py` | Coordinator Service |
| NodeRegistry | `fl_monolith/node_registry.py` | `RegisterParticipant` |
| ExperimentManager | `fl_monolith/experiment_manager.py` | `StartTraining`, `GetTrainingStatus` |
| FedAvgStrategy | `fl_monolith/strategy.py` | aggregation inside the Coordinator |
| WorkerNode | `fl_monolith/worker_node.py` | Participant Service |
| RoundConfig | `models.RoundConfig` | `GetTrainingTask` reply |
| ModelUpdate | `models.ModelUpdate` | `SubmitTrainingResult` request |
| Experiment | `models.Experiment` | `GetTrainingStatus` reply |

Terminology: *node* = participant, *experiment* = training job.
