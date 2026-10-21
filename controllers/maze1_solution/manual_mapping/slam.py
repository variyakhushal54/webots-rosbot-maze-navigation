# This file builds an occupancy map while tracking where the robot is located.
# Lidar rays mark free space and walls, while other modules add special obstacles.
# Pose graph corrections are applied so older map data stays aligned after loop closure.

import math
import time
import numpy as np
import cv2
from config import *



class PoseGraphSLAM:
    """Docstring."""

    def __init__(self, map_size, map_res, sensors, loc, perception, robot, timestep):

        self.map_size    = map_size
        self.map_res     = map_res
        self.sensors     = sensors
        self.loc         = loc
        self.perception  = perception
        self.robot       = robot
        self.timestep    = timestep
        self.log_map     = np.zeros((map_size, map_size), dtype=np.float32)
        self.green_mask_global = np.zeros((map_size, map_size), dtype=np.uint8)
        self.floating_wall_mask_global = np.zeros((map_size, map_size), dtype=np.uint8)
        self.floating_wall_inferred_mask = np.zeros((map_size, map_size), dtype=np.uint8)
        self.floating_wall_last_seen = np.zeros((map_size, map_size), dtype=np.int32)


        self.floating_wall_vote_map   = np.zeros((map_size, map_size), dtype=np.int16)
        self.floating_wall_vote_frame = np.zeros((map_size, map_size), dtype=np.int32)
        self.visit_count = np.zeros((map_size, map_size), dtype=np.uint8)
        self.tunnel_map  = np.zeros((map_size, map_size), dtype=np.float32)
        self.overlay     = np.full((map_size, map_size, 3), 127, dtype=np.uint8)
        self.origin      = map_size // 2
        self.wall_geometry: list = []
        self.point_cloud: list   = []
        self.MAX_CLOUD_POINTS    = 5000
        self.frame_iter          = 0
        self.junctions: list     = []
        self.is_turning          = False


        from pose_graph    import PoseGraph
        from scan_matcher  import match_scans
        from loop_closure  import LoopClosureDetector
        from graph_optimizer import optimize_pose_graph

        self._graph         = PoseGraph()
        self._match_scans   = match_scans
        self._optimize      = optimize_pose_graph


        class _Cfg:
            LOOP_RADIUS_M       = getattr(__import__('config'), 'LOOP_RADIUS_M',      0.5)
            LOOP_TIME_GAP       = getattr(__import__('config'), 'LOOP_TIME_GAP',       30)
            LOOP_FITNESS_THRESH = getattr(__import__('config'), 'LOOP_FITNESS_THRESH', 0.5)
        self._lcd = LoopClosureDetector(self._graph, match_scans, _Cfg())


        self._last_node_id: int | None = None
        self._last_node_pos = (0.0, 0.0)
        self._node_spacing  = getattr(
            __import__('config'), 'NODE_SPACING_M', 0.20)


        self._last_theta = 0.0
        self._last_scan_pos = (0.0, 0.0)


        self._pending_rebuild = False
        self._last_opt_time   = 0.0
        self._opt_min_interval = 1.0

    def stamp_green_points(self, world_points):
        """Permanently burns mapped world points into global green mask and SLAM grid."""
        radius = int(getattr(__import__('config'), 'GREEN_STAMP_RADIUS', 1))
        for wx, wy in world_points:
            mx0 = self.origin + int(round(wx / self.map_res))
            my0 = self.origin - int(round(wy / self.map_res))
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx*dx + dy*dy > radius*radius:
                        continue
                    mx, my = mx0 + dx, my0 + dy
                    if 0 <= mx < self.map_size and 0 <= my < self.map_size:

                        if (self.log_map[my, mx] > WALL_THRESH
                                and self.green_mask_global[my, mx] == 0
                                and self.floating_wall_mask_global[my, mx] == 0):
                            continue
                        self.green_mask_global[my, mx] = 1
                        self.log_map[my, mx] = LOG_CLIP
                        self.overlay[my, mx] = (0, 180, 0)

    def _protect_green_zone(self):
        """Enforces all globally confirmed green cells to remain LOG_CLIP every frame."""
        self.log_map[self.green_mask_global > 0] = LOG_CLIP

    def stamp_floating_wall_points(self, world_points, log_increment=None, radius_cells=None,
                                   inferred=False):
        """Docstring."""
        world_points = np.asarray(world_points)
        if world_points.ndim != 2 or world_points.shape[0] == 0:
            return

        if radius_cells is None:
            if inferred:
                radius_cells = getattr(__import__('config'), 'INFERRED_FLOATING_RADIUS_CELLS', 3)
            else:
                radius_cells = getattr(__import__('config'), 'FLOATING_WALL_RADIUS_CELLS', 2)


        wxs = world_points[:, 0]
        wys = world_points[:, 1]
        mxs0 = np.round(self.origin + wxs / self.map_res).astype(np.int32)
        mys0 = np.round(self.origin - wys / self.map_res).astype(np.int32)


        offsets = []
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if dx*dx + dy*dy <= radius_cells*radius_cells:
                    offsets.append((dx, dy))

        mxs_all = []
        mys_all = []
        for dx, dy in offsets:
            mxs_all.append(mxs0 + dx)
            mys_all.append(mys0 + dy)

        mxs = np.concatenate(mxs_all)
        mys = np.concatenate(mys_all)


        valid = (mxs >= 0) & (mxs < self.map_size) & (mys >= 0) & (mys < self.map_size)
        mxs = mxs[valid]
        mys = mys[valid]

        if len(mxs) == 0:
            return


        indices = mys * self.map_size + mxs
        unique_indices = np.unique(indices)
        mys = unique_indices // self.map_size
        mxs = unique_indices % self.map_size



        is_lidar_wall = (self.log_map[mys, mxs] > WALL_THRESH) & (~self.floating_wall_mask[mys, mxs])
        valid_cells = ~is_lidar_wall
        mxs = mxs[valid_cells]
        mys = mys[valid_cells]

        if len(mxs) == 0:
            return

        if inferred:
            self.log_map[mys, mxs] = LOG_CLIP
            self.floating_wall_inferred_mask[mys, mxs] = 1
            self.floating_wall_last_seen[mys, mxs] = self.frame_iter
            return



        self.floating_wall_vote_map[mys, mxs] = np.clip(
            self.floating_wall_vote_map[mys, mxs] + 1, 0, 127)
        self.floating_wall_vote_frame[mys, mxs] = self.frame_iter
        self._promote_floating_wall_votes()

    def _promote_floating_wall_votes(self):
        """Promote pending vote clusters to confirmed floating walls when they have"""
        import config as _cfg
        thresh       = getattr(_cfg, 'FLOATING_VOTE_THRESHOLD',   3)
        decay_frames = getattr(_cfg, 'FLOATING_VOTE_DECAY_FRAMES', 50)
        min_width_m  = getattr(_cfg, 'FLOATING_MIN_WALL_WIDTH_M',  0.03)
        min_cells    = max(2, int(math.ceil(min_width_m / self.map_res)))


        stale = (self.floating_wall_vote_map > 0) & (
            (self.frame_iter - self.floating_wall_vote_frame) > decay_frames)
        if stale.any():
            self.floating_wall_vote_map[stale] = 0

        candidate = (self.floating_wall_vote_map >= thresh).astype(np.uint8)
        if not candidate.any():
            return

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidate, connectivity=8)

        promoted = False
        for lbl in range(1, n_labels):
            w = int(stats[lbl, cv2.CC_STAT_WIDTH])
            h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            if max(w, h) < min_cells:
                continue

            blob = (labels == lbl)
            self.floating_wall_mask_global[blob] = 1
            self.floating_wall_last_seen[blob]   = self.frame_iter
            self.log_map[blob] = LOG_CLIP
            self.overlay[blob] = (0, 0, 255)
            self.floating_wall_vote_map[blob] = 0
            promoted = True
            print("FLOATING_PROMOTE",
                  f"confirmed wall blob {lbl}: {stats[lbl, cv2.CC_STAT_AREA]} cells, "
                  f"bbox {w}x{h} (min {min_cells} cells = {min_width_m*100:.0f} cm)")

        if promoted:

            gap_size = getattr(__import__('config'), 'FLOATING_WALL_CONNECT_GAP', 5)
            if gap_size > 0:
                mask = self.floating_wall_mask_global
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap_size, gap_size))
                closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                newly = (closed > 0) & (mask == 0)
                if newly.any():
                    self.floating_wall_mask_global[newly] = 1
                    self.log_map[newly] = LOG_CLIP
                    self.floating_wall_last_seen[newly] = self.frame_iter
                    self.overlay[newly] = (0, 0, 255)
            self._prune_noise_blobs()

    def _prune_noise_blobs(self):
        """Remove floating-wall blobs too small or too circular to be wall line segments."""
        import config as _cfg
        min_area    = getattr(_cfg, 'FLOATING_NOISE_MIN_AREA_CELLS', 20)
        blob_thresh = getattr(_cfg, 'FLOATING_NOISE_BLOB_THRESH',    60)
        min_aspect  = getattr(_cfg, 'FLOATING_NOISE_MIN_ASPECT',     2.0)

        mask = self.floating_wall_mask_global
        if not mask.any():
            return

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8)

        removed = 0
        for lbl in range(1, n_labels):
            area   = int(stats[lbl, cv2.CC_STAT_AREA])
            w      = int(stats[lbl, cv2.CC_STAT_WIDTH])
            h      = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            aspect = max(w, h) / max(1, min(w, h))
            if area < min_area or (area < blob_thresh and aspect < min_aspect):
                blob = (labels == lbl)
                self.floating_wall_mask_global[blob] = 0
                self.floating_wall_last_seen[blob]   = 0
                self.log_map[blob] = 0.0
                o = self.overlay
                red = (o[:, :, 0] == 0) & (o[:, :, 1] == 0) & (o[:, :, 2] == 255)
                self.overlay[blob & red] = 127
                removed += 1

        if removed:
            print("FLOATING_PRUNE", f"removed {removed} noise blob(s)")

    @property
    def floating_wall_mask(self):
        return ((self.floating_wall_mask_global > 0)
                | (self.floating_wall_inferred_mask > 0))

    def _protect_floating_walls(self):
        """Enforces all globally confirmed floating wall cells to remain LOG_CLIP every frame."""
        confirmed = (self.floating_wall_mask_global > 0)
        inferred  = (self.floating_wall_inferred_mask > 0)
        self.log_map[confirmed] = LOG_CLIP
        self.log_map[inferred]  = LOG_CLIP

        self.overlay[confirmed] = (0, 0, 255)
        self.overlay[inferred]  = (0, 0, 180)

    def _decay_floating_walls(self):
        """Clears floating_wall cells not re-confirmed within DECAY_FRAMES."""
        pass

    def _strip_floating_duplicate_near_lidar(self):
        """Docstring."""
        import config as _cfg
        margin = int(getattr(_cfg, 'FLOATING_LIDAR_STRIP_MARGIN_CELLS', 4))
        if margin < 1:
            return


        pending_votes = (self.floating_wall_vote_map > 0) & (self.floating_wall_mask_global == 0)
        if not pending_votes.any():
            return


        lidar_wall = (self.log_map > WALL_THRESH).astype(np.uint8)
        if not lidar_wall.any():
            return


        to_clear = pending_votes & (lidar_wall > 0)

        if not to_clear.any():
            return

        n_cleared = int(to_clear.sum())
        self.floating_wall_vote_map[to_clear] = 0
        self.floating_wall_vote_frame[to_clear] = 0

        if n_cleared:
            print("LIDAR_WINS",
                  f"cleared {n_cleared} pre-confirmation vote cells"
                  f" that overlapped direct lidar walls")

    def cleanup_stale_floating_walls(self):
        return

    def _clear_confirmed_pillar_walls(self):
        """Docstring."""
        if not hasattr(self, 'perception') or self.perception is None:
            return

        r_margin = getattr(__import__('config'), 'PILLAR_RADIUS_M', 0.10) + 0.03
        r_cells = max(1, int(r_margin / self.map_res))

        for color in ['BLUE', 'YELLOW']:
            cx, cy = None, None
            if color in self.perception.cam_confirmed:
                cx, cy = self.perception.cam_confirmed[color][:2]
            else:
                buf = self.perception.cam_pillar_buf.get(color, [])
                if buf:
                    recent = buf[-10:]
                    xs = sorted(p[0] for p in recent)
                    ys = sorted(p[1] for p in recent)
                    cx, cy = xs[len(xs)//2], ys[len(ys)//2]

            if cx is not None and cy is not None:
                px = self.origin + int(round(cx / self.map_res))
                py = self.origin - int(round(cy / self.map_res))

                y0 = max(0, py - r_cells)
                y1 = min(self.map_size, py + r_cells + 1)
                x0 = max(0, px - r_cells)
                x1 = min(self.map_size, px + r_cells + 1)

                for my in range(y0, y1):
                    for mx in range(x0, x1):
                        if math.hypot(mx - px, my - py) <= r_cells:
                            if self.log_map[my, mx] > 0:
                                self.log_map[my, mx] = 0.0
                                if (self.overlay[my, mx] == [0, 0, 0]).all():
                                    self.overlay[my, mx] = [127, 127, 127]

    def update_map(self):
        self.frame_iter += 1
        if self.frame_iter == 90 and not getattr(self, '_cleanup_done', False):
            self.cleanup_stale_floating_walls()
            self._cleanup_done = True

        if self.frame_iter % 50 == 0:
            self._update_junctions()

        self._protect_green_zone()
        self._protect_floating_walls()
        self._clear_confirmed_pillar_walls()

        ranges, w, fov = self.sensors.get_lidar()


        if not hasattr(self, '_last_theta'): self._last_theta = self.loc.theta
        raw_dtheta = abs(self.loc.theta - self._last_theta)


        if raw_dtheta > math.pi: raw_dtheta = 2 * math.pi - raw_dtheta
        dtheta = raw_dtheta



        is_turning = dtheta > 0.050
        self._last_theta = self.loc.theta
        self.is_turning = is_turning



        settle_val = int(getattr(__import__('config'), 'LIDAR_POST_TURN_SETTLE', 0))
        if not hasattr(self, '_post_turn_frames'): self._post_turn_frames = 0
        if is_turning:
            self._post_turn_frames = settle_val if settle_val > 0 else 0
            return
        elif self._post_turn_frames > 0:
            self._post_turn_frames -= 1
            return


        self._raytrace(ranges, w, fov, is_turning=False)
        self._strip_floating_duplicate_near_lidar()




        dist_since_node = math.hypot(
            self.loc.x - self._last_node_pos[0],
            self.loc.y - self._last_node_pos[1])

        if dist_since_node >= self._node_spacing:
            t = self.robot.getTime()
            node_id = self._graph.add_node(
                self.loc.x, self.loc.y, self.loc.theta,
                ranges, fov, t)


            if self._last_node_id is not None:
                prev = self._graph.get_node(self._last_node_id)
                dx = self.loc.x - prev.x
                dy = self.loc.y - prev.y
                dth = _wrap(self.loc.theta - prev.theta)

                c, s = math.cos(prev.theta), math.sin(prev.theta)
                dx_l =  c * dx + s * dy
                dy_l = -s * dx + c * dy
                self._graph.add_odometry_edge(
                    self._last_node_id, node_id, dx_l, dy_l, dth)

            self._last_node_id  = node_id
            self._last_node_pos = (self.loc.x, self.loc.y)


            edge = self._lcd.maybe_close_loop(node_id)
            if edge is not None:
                self._pending_rebuild = True


        if self._pending_rebuild:
            now = self.robot.getTime()
            if now - self._last_opt_time >= self._opt_min_interval:
                self._run_optimisation()
                self._last_opt_time   = now
                self._pending_rebuild = False
        self._decay_floating_walls()



    def _run_optimisation(self):
        opt = self._optimize(self._graph)
        self._graph.apply_optimized_poses(opt)
        self._rebuild_grid()

    def _rebuild_grid(self):
        """Docstring."""
        t0 = time.perf_counter()
        self.log_map[:] = 0.0
        self.overlay[:]  = 127
        self._protect_green_zone()
        self._protect_floating_walls()
        self.overlay[self.green_mask_global > 0] = (0, 180, 0)
        self.overlay[self.floating_wall_mask_global > 0] = (0, 0, 255)
        opt_poses = self._graph.get_optimized_poses()

        lidar_offset = (LIDAR_OFFSET_X, LIDAR_OFFSET_Y,
                        LIDAR_OFFSET_Z,  LIDAR_OFFSET_YAW)

        for node in self._graph.get_all_nodes():
            ox, oy, otheta = opt_poses[node.node_id]
            ranges = node.scan_ranges
            fov    = node.scan_fov
            w      = len(ranges)
            self._raytrace_at_pose(ranges, w, fov, ox, oy, otheta,
                                   is_turning=False, lidar_offset=lidar_offset)



        elapsed = (time.perf_counter() - t0) * 1000
        print(f"[PoseGraphSLAM] Grid rebuilt from {self._graph.node_count()} nodes "
              f"in {elapsed:.1f} ms")



    def _raytrace(self, ranges, w, fov, is_turning):
        """Ray-trace the current scan at the current dead-reckoned pose."""
        self._raytrace_at_pose(
            ranges, w, fov,
            self.loc.x, self.loc.y, self.loc.theta,
            is_turning)

    def _raytrace_at_pose(self, ranges, w, fov,
                          px, py, theta,
                          is_turning, lidar_offset=None):
        if lidar_offset is None:
            lidar_offset = (LIDAR_OFFSET_X, LIDAR_OFFSET_Y,
                            LIDAR_OFFSET_Z,  LIDAR_OFFSET_YAW)
        sx, sy, sz, syaw = lidar_offset
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        sensor_wx = px + sx * cos_t - sy * sin_t
        sensor_wy = py + sx * sin_t + sy * cos_t
        FREE_LIMIT = getattr(__import__('config'), 'LIDAR_FRONTIER_RANGE', 2.5)
        free_decrement = getattr(__import__('config'), 'LIDAR_FREE_DECREMENT', 0.30)
        wall_increment_base = getattr(__import__('config'), 'LIDAR_WALL_INCREMENT', 1.5)
        MAX_FREE_TRACE_M = 4.0

        ranges_arr = np.asarray(ranges, dtype=np.float32)
        subsample = getattr(__import__('config'), 'LIDAR_SUBSAMPLE', 1)

        for i in range(0, w, subsample):
            d = ranges[i]
            if math.isnan(d) or d < 0.15: continue

            hit = not math.isinf(d) and d < MAX_RANGE * 0.98


            if not hit:
                neighbor_idx = [i-1, i+1, i-2, i+2]
                close_neighbors = 0
                for n_idx in neighbor_idx:
                    if 0 <= n_idx < w:
                        nd = ranges[n_idx]
                        if not math.isinf(nd) and not math.isnan(nd) and nd < 1.5:
                            close_neighbors += 1
                if close_neighbors >= 1:
                    continue

            if not hit and getattr(__import__('config'), 'LIDAR_SKIP_INF_RAYS', True):
                continue

            free_len = min(d - MAP_RES, MAX_FREE_TRACE_M) if hit else FREE_LIMIT
            bearing  = fov / 2 - i * fov / w
            ray_yaw  = theta + syaw + bearing
            ca, sa   = math.cos(ray_yaw), math.sin(ray_yaw)


            n_steps = int(free_len / MAP_RES)
            if n_steps > 1:
                steps = np.arange(1, n_steps, dtype=np.float32)
                pxs = np.round(self.origin + (sensor_wx + steps * MAP_RES * ca) / MAP_RES).astype(np.int32)
                pys = np.round(self.origin - (sensor_wy + steps * MAP_RES * sa) / MAP_RES).astype(np.int32)

                in_map = (pxs >= 0) & (pxs < MAP_SIZE) & (pys >= 0) & (pys < MAP_SIZE)
                pxs_v = pxs[in_map]
                pys_v = pys[in_map]

                if len(pxs_v):
                    blocked = (
                        (self.green_mask_global[pys_v, pxs_v] > 0) |
                        (self.floating_wall_mask_global[pys_v, pxs_v] > 0)
                    )
                    if blocked.any():
                        stop = int(np.argmax(blocked))
                        pxs_v = pxs_v[:stop]
                        pys_v = pys_v[:stop]

                    if len(pxs_v):
                        unlocked = self.log_map[pys_v, pxs_v] < WALL_LOCK

                        not_floating = (
                            (self.floating_wall_mask_global[pys_v, pxs_v] == 0) &
                            (self.floating_wall_inferred_mask[pys_v, pxs_v] == 0)
                        )
                        ux, uy = pxs_v[unlocked & not_floating], pys_v[unlocked & not_floating]
                        if len(ux):
                            self.log_map[uy, ux] -= free_decrement
                            self.overlay[uy, ux] = (255, 255, 255)


            if hit and not is_turning:
                wx = sensor_wx + d * ca
                wy = sensor_wy + d * sa
                mpx = self.origin + int(round(wx / MAP_RES))
                mpy = self.origin - int(round(wy / MAP_RES))


                on_pillar = False
                if hasattr(self, 'perception') and self.perception is not None:
                    r_margin = getattr(__import__('config'), 'PILLAR_RADIUS_M', 0.10) + 0.03
                    for color in ['BLUE', 'YELLOW']:
                        if color in self.perception.cam_confirmed:
                            cx, cy = self.perception.cam_confirmed[color][:2]
                            if math.hypot(wx - cx, wy - cy) < r_margin:
                                on_pillar = True
                                break
                        buf = self.perception.cam_pillar_buf.get(color, [])
                        if buf:
                            recent = buf[-10:]
                            xs = sorted(p[0] for p in recent)
                            ys = sorted(p[1] for p in recent)
                            cx, cy = xs[len(xs)//2], ys[len(ys)//2]
                            if math.hypot(wx - cx, wy - cy) < r_margin:
                                on_pillar = True
                                break

                if not on_pillar and 0 <= mpx < MAP_SIZE and 0 <= mpy < MAP_SIZE:
                    if self.green_mask_global[mpy, mpx] == 0 and self.floating_wall_mask_global[mpy, mpx] == 0:

                        near_wx = sensor_wx + (d - MAP_RES) * ca
                        near_wy = sensor_wy + (d - MAP_RES) * sa
                        near_px = self.origin + int(round(near_wx / MAP_RES))
                        near_py = self.origin - int(round(near_wy / MAP_RES))
                        if 0 <= near_px < MAP_SIZE and 0 <= near_py < MAP_SIZE:
                            if self.log_map[near_py, near_px] >= WALL_LOCK:
                                continue

                        current_val = self.log_map[mpy, mpx]
                        if current_val < WALL_LOCK:
                            increment = wall_increment_base
                        elif current_val < 3.0:
                            increment = wall_increment_base * 0.5
                        elif current_val < 5.0:
                            increment = wall_increment_base * 0.15
                        else:
                            increment = 0.05
                        self.log_map[mpy, mpx] = min(current_val + increment, LOG_CLIP)
                        if self.log_map[mpy, mpx] > WALL_THRESH:
                            self.overlay[mpy, mpx] = (0, 0, 0)
                if i % 4 == 0:
                    wz = sz
                    self._append_cloud_point(wx, wy, wz, 'lidar')



    def _update_junctions(self):
        """Docstring."""

        free = (self.log_map < -2.0).astype(np.uint8)
        if free.sum() < 400: return


        free = cv2.morphologyEx(free, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        free = cv2.morphologyEx(free, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))


        dist_field = cv2.distanceTransform(free, cv2.DIST_L2, 3)


        skel = self._skeletonize(free)


        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        neighbors = cv2.filter2D(skel, -1, kernel)


        junction_pixels = (skel > 0) & (neighbors >= 3) & (dist_field >= 5)

        coords = np.column_stack(np.where(junction_pixels))


        new_junctions = []
        for y, x in coords:
            if not any(math.hypot(y-jy, x-jx) < 25 for jy, jx in new_junctions):
                new_junctions.append((int(y), int(x)))

        self.junctions = new_junctions

    def _skeletonize(self, img):
        """Standard iterative skeletonization."""
        skel = np.zeros(img.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False
        temp = img.copy()
        while not done:
            eroded = cv2.erode(temp, element)
            temp_open = cv2.dilate(eroded, element)
            temp_open = cv2.subtract(temp, temp_open)
            skel = cv2.bitwise_or(skel, temp_open)
            temp = eroded.copy()
            if cv2.countNonZero(temp) == 0: done = True
        return skel

    def _filter_ghost_obstacles(self, min_area=3):
        """Remove isolated black wall clusters; floating-wall cells are NEVER touched."""
        occ = (np.all(self.overlay == [0, 0, 0], axis=-1)).astype(np.uint8)
        floating_confirmed = (self.floating_wall_mask_global > 0) | (self.floating_wall_inferred_mask > 0)
        occ[floating_confirmed] = 0
        if occ.sum() == 0: return
        n, labels, stats, _ = cv2.connectedComponentsWithStats(occ, connectivity=8)
        for lbl in range(1, n):
            if stats[lbl, cv2.CC_STAT_AREA] < min_area:
                cluster = (labels == lbl)
                if floating_confirmed[cluster].any():
                    continue
                is_free = self.log_map[cluster].mean() < FREE_THRESH
                self.overlay[cluster] = [255, 255, 255] if is_free else [127, 127, 127]
        if ENABLE_GAP_CLOSING and self.frame_iter % 50 == 0:
            self._close_narrow_gaps()

    def _close_narrow_gaps(self):
        threshold_m = PATH_DRAW_WIDTH * NARROW_GAP_THRESHOLD
        k_size = int(threshold_m / MAP_RES)
        if k_size < 3 or k_size % 2 == 0: k_size = max(3, k_size + 1)
        occ = (self.log_map > WALL_THRESH).astype(np.uint8)
        if occ.sum() == 0: return
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        closed = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, kernel)
        new_walls = (closed > 0) & (occ == 0)
        self.log_map[new_walls] = LOG_CLIP
        self.overlay[new_walls] = (70, 70, 70)

    def _stamp_green_obstacle(self):
        if 'GREEN' not in self.perception.cam_confirmed: return
        entry = self.perception.cam_confirmed['GREEN']
        GREEN_BGR = (0, 180, 0)
        if len(entry) >= 4 and isinstance(entry[3], list):
            corners_world = entry[3]
            pts = []
            for wx, wy in corners_world:
                mx = max(0, min(MAP_SIZE - 1, self.origin + int(wx / MAP_RES)))
                my = max(0, min(MAP_SIZE - 1, self.origin - int(wy / MAP_RES)))
                pts.append([mx, my])
            poly = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(self.overlay, [poly], GREEN_BGR)
            mask = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 1)
            self.log_map[mask > 0] = LOG_CLIP
            lx = min(p[0] for p in pts)
            ty = min(p[1] for p in pts)
            cv2.putText(self.overlay, "GREEN-ZONE", (lx + 2, ty + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, GREEN_BGR, 1)
        else:
            rx, ry, _ = entry[:3]
            h = GREEN_HALF
            px0 = max(0, min(MAP_SIZE - 1, self.origin + int((rx - h) / MAP_RES)))
            px1 = max(0, min(MAP_SIZE - 1, self.origin + int((rx + h) / MAP_RES)))
            py0 = max(0, min(MAP_SIZE - 1, self.origin - int((ry + h) / MAP_RES)))
            py1 = max(0, min(MAP_SIZE - 1, self.origin - int((ry - h) / MAP_RES)))
            self.overlay[py0:py1 + 1, px0:px1 + 1] = GREEN_BGR
            self.log_map[py0:py1 + 1, px0:px1 + 1] = LOG_CLIP
            cv2.putText(self.overlay, "GREEN-ZONE", (px0 + 3, py0 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, GREEN_BGR, 1)

    def _world_from_sensor(self, sensor_offset, distance, sensor_bearing=0.0):
        sx, sy, sz, syaw = sensor_offset
        cos_t, sin_t = math.cos(self.loc.theta), math.sin(self.loc.theta)
        sensor_wx = self.loc.x + sx * cos_t - sy * sin_t
        sensor_wy = self.loc.y + sx * sin_t + sy * cos_t
        ray_yaw = self.loc.theta + syaw + sensor_bearing
        wx = sensor_wx + distance * math.cos(ray_yaw)
        wy = sensor_wy + distance * math.sin(ray_yaw)
        return (wx, wy, sz)

    def _append_cloud_point(self, wx, wy, wz, source):
        self.point_cloud.append((wx, wy, wz, source))
        if len(self.point_cloud) > self.MAX_CLOUD_POINTS:
            drop = self.MAX_CLOUD_POINTS // 10
            self.point_cloud = self.point_cloud[drop:]

    def _thin_lidar_walls(self):
        """Morphological wall thinning to keep lidar walls 1 cell thin."""
        lidar_only = (
            (self.log_map > WALL_THRESH)
            & (self.floating_wall_mask_global == 0)
            & (self.floating_wall_inferred_mask == 0)
            & (self.green_mask_global == 0)
        ).astype(np.uint8)

        if lidar_only.sum() < 10:
            return

        kernel = np.ones((3, 3), np.uint8)
        interior = cv2.erode(lidar_only, kernel)

        if not interior.any():
            return

        self.log_map[interior > 0] = max(0.0, WALL_THRESH * 0.4)
        self.overlay[interior > 0] = (70, 70, 70)

    def save_debug_maps(self):
        cv2.imwrite("maze_overlay.png", self.overlay)
        debug_grid = np.full((MAP_SIZE, MAP_SIZE), 127, dtype=np.uint8)
        debug_grid[self.log_map > WALL_THRESH] = 0
        debug_grid[self.log_map < FREE_THRESH] = 255
        cv2.imwrite("maze_occupancy_debug.png", debug_grid)
        print("MAP-VERIFY", "Debug images saved.")



def make_slam(map_size, map_res, sensors, loc, perception, robot, timestep):
    """Docstring."""
    return PoseGraphSLAM(map_size, map_res, sensors, loc, perception, robot, timestep)

def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))
