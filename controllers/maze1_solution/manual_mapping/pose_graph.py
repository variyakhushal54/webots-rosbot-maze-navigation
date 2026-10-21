# This file stores the poses collected while the robot moves through the maze.
# Edges describe measured movement between poses or a match with an older location.
# The graph optimizer uses this information to reduce drift in the final map.

import math
import numpy as np

class PoseNode:
    """Stores a robot pose and the lidar scan taken at that pose."""
    __slots__ = ('node_id', 'x', 'y', 'theta', 'scan_ranges', 'scan_fov', 'timestamp')

    def __init__(self, node_id: int, x: float, y: float, theta: float,
                 scan_ranges, scan_fov: float, timestamp: float):
        self.node_id    = node_id
        self.x          = x
        self.y          = y
        self.theta      = theta
        self.scan_ranges = np.array(scan_ranges, dtype=np.float32)
        self.scan_fov   = float(scan_fov)
        self.timestamp  = float(timestamp)

class PoseEdge:
    """Docstring."""
    __slots__ = ('from_id', 'to_id', 'dx', 'dy', 'dtheta', 'info_matrix', 'kind')

    def __init__(self, from_id: int, to_id: int,
                 dx: float, dy: float, dtheta: float,
                 info_matrix: np.ndarray,
                 kind: str = 'odometry'):
        self.from_id     = from_id
        self.to_id       = to_id
        self.dx          = float(dx)
        self.dy          = float(dy)
        self.dtheta      = float(dtheta)
        self.info_matrix = np.asarray(info_matrix, dtype=np.float64)
        self.kind        = kind



def _odometry_info(sigma_xy: float = 0.02, sigma_theta: float = 0.01) -> np.ndarray:
    """Diagonal information matrix for dead-reckoned odometry."""
    return np.diag([1.0/sigma_xy**2, 1.0/sigma_xy**2, 1.0/sigma_theta**2])

def _loop_info(fitness: float,
               sigma_xy: float = 0.04, sigma_theta: float = 0.02) -> np.ndarray:
    """Docstring."""
    scale = max(0.1, fitness)
    return np.diag([scale/sigma_xy**2, scale/sigma_xy**2, scale/sigma_theta**2])

class PoseGraph:
    """Docstring."""

    def __init__(self):
        self._nodes: dict[int, PoseNode] = {}
        self._edges: list[PoseEdge]      = []
        self._next_id: int               = 0

        self._opt_poses: dict[int, tuple] = {}



    def add_node(self, x: float, y: float, theta: float,
                 scan_ranges, fov: float, timestamp: float = 0.0) -> int:
        nid = self._next_id
        self._next_id += 1
        node = PoseNode(nid, x, y, theta, scan_ranges, fov, timestamp)
        self._nodes[nid] = node
        self._opt_poses[nid] = (x, y, theta)
        return nid

    def add_odometry_edge(self, from_id: int, to_id: int,
                          dx: float, dy: float, dtheta: float,
                          sigma_xy: float = 0.02, sigma_theta: float = 0.01):
        info = _odometry_info(sigma_xy, sigma_theta)
        self._edges.append(PoseEdge(from_id, to_id, dx, dy, dtheta, info, 'odometry'))

    def add_loop_closure_edge(self, from_id: int, to_id: int,
                              dx: float, dy: float, dtheta: float,
                              fitness: float):
        info = _loop_info(fitness)
        self._edges.append(PoseEdge(from_id, to_id, dx, dy, dtheta, info, 'loop'))
        print(f"[PoseGraph] Loop closure: node {from_id} ↔ {to_id}  "
              f"Δ=({dx:.3f},{dy:.3f},{math.degrees(dtheta):.1f}°)  "
              f"fitness={fitness:.3f}")



    def get_node(self, node_id: int) -> PoseNode:
        return self._nodes[node_id]

    def get_all_nodes(self) -> list:
        return list(self._nodes.values())

    def get_edges(self) -> list:
        return self._edges

    def get_optimized_poses(self) -> dict:
        """Returns {node_id: (x, y, theta)}  updated by apply_optimized_poses."""
        return dict(self._opt_poses)

    def apply_optimized_poses(self, poses_dict: dict):
        """Called by graph_optimizer to write back corrected poses."""
        for nid, (x, y, theta) in poses_dict.items():
            self._opt_poses[nid] = (x, y, theta)

    def node_count(self) -> int:
        return len(self._nodes)
