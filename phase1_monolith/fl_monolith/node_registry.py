"""Tracks which worker nodes take part in the federation."""
from __future__ import annotations

import logging
from typing import Dict, List

from .worker_node import WorkerNode

log = logging.getLogger("fl.registry")


class NodeRegistry:
    def __init__(self) -> None:
        self._nodes: Dict[str, WorkerNode] = {}

    def register_node(self, node: WorkerNode) -> None:
        if node.node_id in self._nodes:
            raise ValueError(f"node '{node.node_id}' is already registered")
        self._nodes[node.node_id] = node
        log.info("registered node %s (%d samples)", node.node_id, node.num_samples)

    def get_active_nodes(self) -> List[WorkerNode]:
        """All registered nodes, in registration order."""
        return list(self._nodes.values())

    def __len__(self) -> int:
        return len(self._nodes)
