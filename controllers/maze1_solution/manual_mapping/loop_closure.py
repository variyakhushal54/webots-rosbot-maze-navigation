# This file checks whether the robot has returned to a place it visited earlier.
# Nearby old scans are compared with the current scan to find a reliable match.
# A successful match adds a graph connection that can correct accumulated drift.

import math
import numpy as np
from scan_matcher import match_scans

class LoopClosureDetector:
    """Detect loops and trigger closure optimization."""

    def __init__(self, pose_graph, scan_matcher_fn, config):
        self.graph          = pose_graph
        self.match_fn       = scan_matcher_fn
        self.radius         = getattr(config, 'LOOP_RADIUS_M',       0.5)
        self.time_gap       = getattr(config, 'LOOP_TIME_GAP',        30)
        self.fitness_thresh = getattr(config, 'LOOP_FITNESS_THRESH',  0.5)
        self._attempt_count = 0
        self._closure_count = 0



    def maybe_close_loop(self, current_node_id: int):
        """Check for loop closure candidates and optimize."""
        current_node = self.graph.get_node(current_node_id)
        opt_poses    = self.graph.get_optimized_poses()

        cx, cy = opt_poses[current_node_id][:2]
        candidates = []

        for node in self.graph.get_all_nodes():

            if current_node_id - node.node_id < self.time_gap:
                continue

            ox, oy = opt_poses[node.node_id][:2]
            dist = math.hypot(cx - ox, cy - oy)
            if dist <= self.radius:
                candidates.append((dist, node))

        if not candidates:
            return None


        candidates.sort(key=lambda t: t[0])

        best_edge   = None
        best_fitness = 0.0

        for dist, cand_node in candidates[:5]:
            self._attempt_count += 1


            ox, oy, oth = opt_poses[cand_node.node_id]

            dx_world = cx - ox
            dy_world = cy - oy
            c, s = math.cos(-oth), math.sin(-oth)
            dx_local =  c * dx_world - s * dy_world
            dy_local =  s * dx_world + c * dy_world
            dth_local = _wrap_angle(opt_poses[current_node_id][2] - oth)

            ref_scan   = (cand_node.scan_ranges,    cand_node.scan_fov)
            query_scan = (current_node.scan_ranges,  current_node.scan_fov)

            dx, dy, dtheta, fitness, info = self.match_fn(
                ref_scan, query_scan,
                initial_guess=(dx_local, dy_local, dth_local)
            )

            if fitness >= self.fitness_thresh and fitness > best_fitness:
                best_fitness = fitness
                best_edge = (cand_node.node_id, dx, dy, dtheta, fitness, info)

        if best_edge is None:
            return None


        from_id, dx, dy, dtheta, fitness, info = best_edge
        self.graph.add_loop_closure_edge(from_id, current_node_id, dx, dy, dtheta, fitness)
        self._closure_count += 1
        print(f"[LoopClosure] Closed! #{self._closure_count}  "
              f"attempts={self._attempt_count}  fitness={fitness:.3f}")


        return self.graph.get_edges()[-1]

def _wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))
