# gRPC Federated Learning (Group D)

A small federated learning system built in two phases.

- **Phase 1 – Monolith:** a single-process simulator. An admin console registers
  nodes, creates experiments and starts training. The server sends a round
  config to each node, collects their model updates and aggregates them with
  FedAvg.
- **Phase 2 – Distributed (gRPC):** the same logic split into a Coordinator and
  independent Participant processes that talk over gRPC
  (RegisterParticipant, StartTraining, GetTrainingTask,
  SubmitTrainingResult, GetTrainingStatus).

Example: three nodes holding 10, 20 and 30 produce a global model of 20.

## Branches

`main` holds only project documentation. Code lives on feature branches and is
merged once the team agrees on it.

| Branch | Contents |
|---|---|
| `umar-phase1-monolith` | Phase 1 monolith (Python) |

## Team

Group D
