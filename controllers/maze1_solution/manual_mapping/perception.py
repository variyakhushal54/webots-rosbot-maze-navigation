# This file turns camera and depth images into features the robot can use.
# It looks for pillars, green ground areas, and obstacles that lidar may not detect.
# Detected points are converted from image coordinates into positions in the map.

import math
import numpy as np
import cv2
from config import *

class PillarDetector:
    def __init__(self, sensors):
        self.sensors = sensors
        self.cam_confirmed = {}
        self.cam_pillar_buf = {'YELLOW': [], 'BLUE': [], 'GREEN': []}
        self.tiny_sighting  = None
        self.green_pending  = False
        self.green_pending_pos = None
        self._last_theta    = None
        self.green_world_points = np.empty((0, 2), dtype=np.float32)
        self.floating_wall_points = np.empty((0, 2), dtype=np.float32)


        self.checked_lidar_pillars = []
        self.unidentified_lidar_pillars = []

        self._cached_rays = None



    def _pixel_rays(self, width, height, fov_h):
        """Pre-calculate and cache pixel ray vectors in camera frame."""
        fx = width / (2.0 * math.tan(fov_h / 2.0))
        cx, cy = width / 2.0, height / 2.0

        u = np.arange(width)
        v = np.arange(height)
        uu, vv = np.meshgrid(u, v)

        rx = uu - cx
        ry = vv - cy
        rz = np.full_like(rx, fx, dtype=np.float32)

        rays = np.stack([rx, ry, rz], axis=-1)
        norms = np.linalg.norm(rays, axis=-1, keepdims=True)
        return rays / norms

    def _camera_to_world(self, rays_cam, offset_x=None, offset_y=None, offset_z=None):
        """Rigid-body transformation from camera to world frame."""
        pitch = getattr(__import__('config'), 'CAMERA_OFFSET_PITCH', 0.0)
        yaw   = getattr(__import__('config'), 'CAMERA_OFFSET_YAW', 0.0)


        rx0 = rays_cam[:, 2]
        ry0 = -rays_cam[:, 0]
        rz0 = -rays_cam[:, 1]


        cos_p, sin_p = math.cos(pitch), math.sin(pitch)
        rx_p = rx0 * cos_p - rz0 * sin_p
        ry_p = ry0
        rz_p = rx0 * sin_p + rz0 * cos_p


        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        rx_r = rx_p * cos_y - ry_p * sin_y
        ry_r = rx_p * sin_y + ry_p * cos_y
        rz_r = rz_p


        theta = self.loc.theta
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        dir_x = rx_r * cos_t - ry_r * sin_t
        dir_y = rx_r * sin_t + ry_r * cos_t
        dir_z = rz_r
        dirs_world = np.stack([dir_x, dir_y, dir_z], axis=-1)


        ox_r = offset_x if offset_x is not None else getattr(__import__('config'), 'CAMERA_OFFSET_X', 0.00)
        oy_r = offset_y if offset_y is not None else getattr(__import__('config'), 'CAMERA_OFFSET_Y', 0.011)
        oz_r = offset_z if offset_z is not None else getattr(__import__('config'), 'CAMERA_OFFSET_Z', 0.199)

        ox_w = ox_r * cos_t - oy_r * sin_t + self.loc.x
        oy_w = ox_r * sin_t + oy_r * cos_t + self.loc.y
        oz_w = oz_r

        origins_world = np.tile(np.array([ox_w, oy_w, oz_w], dtype=np.float32), (len(rays_cam), 1))
        return origins_world, dirs_world

    def _intersect_ground_plane(self, origins, dirs):
        """Vectorized intersection with flat z=0 ground plane."""
        max_dist = getattr(__import__('config'), 'MAX_GROUND_DIST', 3.0)

        oz, dz = origins[:, 2], dirs[:, 2]
        valid = (dz < -1e-4)
        t = np.zeros_like(dz)
        t[valid] = -oz[valid] / dz[valid]

        valid = valid & (t >= 0) & (t <= max_dist)

        hits = np.zeros_like(origins)
        hits[valid, 0] = origins[valid, 0] + t[valid] * dirs[valid, 0]
        hits[valid, 1] = origins[valid, 1] + t[valid] * dirs[valid, 1]
        hits[valid, 2] = 0.0
        return hits, valid

    def _detect_green_ground(self, bgr):
        """Detect green ground using HSV and IPM."""
        import config as _cfg


        lo = np.array(getattr(_cfg, 'GREEN_HSV_LOW',  (38, 80, 50)),  dtype=np.uint8)
        hi = np.array(getattr(_cfg, 'GREEN_HSV_HIGH', (90, 255, 255)), dtype=np.uint8)
        hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lo, hi)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0:
            self.green_world_points = np.empty((0, 2), dtype=np.float32)
            return


        img_h = bgr.shape[0]
        row_min = int(getattr(_cfg, 'GREEN_ROW_MIN_FRAC', 0.55) * img_h)
        row_max = int(getattr(_cfg, 'GREEN_ROW_MAX_FRAC', 0.92) * img_h)
        row_band = (y_indices >= row_min) & (y_indices <= row_max)
        y_indices = y_indices[row_band]
        x_indices = x_indices[row_band]

        if len(y_indices) == 0:
            self.green_world_points = np.empty((0, 2), dtype=np.float32)
            return


        stride = int(getattr(_cfg, 'GREEN_SUBSAMPLE_STRIDE', 3))
        sub_mask = (y_indices % stride == 0) & (x_indices % stride == 0)
        y_indices = y_indices[sub_mask]
        x_indices = x_indices[sub_mask]

        if len(y_indices) == 0:
            self.green_world_points = np.empty((0, 2), dtype=np.float32)
            return


        width, height = self.sensors.cam_w, self.sensors.cam_h
        if self._cached_rays is None or self._cached_rays.shape[:2] != (height, width):
            self._cached_rays = self._pixel_rays(width, height, self.sensors.cam_fov)


        rays_cam = self._cached_rays[y_indices, x_indices]
        origins_world, dirs_world = self._camera_to_world(rays_cam)
        hits_plane, valid_plane   = self._intersect_ground_plane(origins_world, dirs_world)


        min_dist = float(getattr(_cfg, 'GREEN_MIN_STAMP_DIST', 0.20))
        max_dist = float(getattr(_cfg, 'GREEN_MAX_STAMP_DIST', 1.5))
        if valid_plane.any():
            hx    = hits_plane[valid_plane, 0]
            hy    = hits_plane[valid_plane, 1]
            dist2 = (hx - self.loc.x) ** 2 + (hy - self.loc.y) ** 2
            in_range = (dist2 >= min_dist ** 2) & (dist2 <= max_dist ** 2)
            valid_indices = np.where(valid_plane)[0][in_range]
            valid_plane_filtered = np.zeros(len(valid_plane), dtype=bool)
            valid_plane_filtered[valid_indices] = True
        else:
            valid_plane_filtered = valid_plane

        if not valid_plane_filtered.any():
            self.green_world_points = np.empty((0, 2), dtype=np.float32)
            return


        world_pts = hits_plane[valid_plane_filtered, :2].astype(np.float32)


        bias = float(getattr(_cfg, 'GREEN_DEPTH_BIAS', 0.0))
        if bias != 0.0:
            cos_t = math.cos(self.loc.theta)
            sin_t = math.sin(self.loc.theta)
            world_pts[:, 0] += bias * cos_t
            world_pts[:, 1] += bias * sin_t

        self.green_world_points = world_pts


        if not hasattr(self, '_green_vote_map'):
            self._green_vote_map   = {}
            self._green_vote_frame = 0

        vote_thresh = int(getattr(_cfg, 'GREEN_VOTE_THRESHOLD', 3))
        vote_decay  = int(getattr(_cfg, 'GREEN_VOTE_DECAY',     60))
        map_res     = self.slam.map_res  if hasattr(self, 'slam') and self.slam else 0.02
        origin      = self.slam.origin   if hasattr(self, 'slam') and self.slam else 500

        self._green_vote_frame += 1


        if self._green_vote_frame % vote_decay == 0:
            keys_to_del = [k for k, v in self._green_vote_map.items() if v <= 1]
            for k in keys_to_del:
                del self._green_vote_map[k]
            for k in self._green_vote_map:
                self._green_vote_map[k] = max(0, self._green_vote_map[k] - 1)


        confirmed_this_frame = []
        for wx, wy in world_pts:
            mx = origin + int(wx / map_res)
            my = origin - int(wy / map_res)
            key = (mx, my)
            self._green_vote_map[key] = self._green_vote_map.get(key, 0) + 1
            if self._green_vote_map[key] >= vote_thresh:
                confirmed_this_frame.append((wx, wy))


        if confirmed_this_frame and hasattr(self, 'slam') and self.slam is not None:
            confirmed_arr = np.array(confirmed_this_frame, dtype=np.float32)
            self.slam.stamp_green_points(confirmed_arr)


            centroid_x = float(np.mean(confirmed_arr[:, 0]))
            centroid_y = float(np.mean(confirmed_arr[:, 1]))
            if 'GREEN' not in self.cam_confirmed:
                self.cam_confirmed['GREEN'] = (centroid_x, centroid_y, (0, 180, 0))
                if hasattr(self, 'planner') and self.planner is not None:
                    self.planner.nav_path = []
                    print("[GREEN-OBSTACLE] Confirmed via voted IPM & forcing path replan")

    def detect_pillars_camera(self, bgr, ranges, lidar_w, lidar_fov):
        """Detect pillars from camera."""
        if bgr is None:
            return


        TURN_GATE_GREEN = 0.040 * 8.0
        if self._last_theta is None:
            self._last_theta = self.loc.theta
        dtheta = abs(self.loc.theta - self._last_theta)
        if dtheta > math.pi:
            dtheta = abs(dtheta - 2 * math.pi)
        is_rotating_fast = dtheta > TURN_GATE_GREEN
        self._last_theta = self.loc.theta

        if not is_rotating_fast:
            self._detect_green_ground(bgr)
        else:
            self.green_world_points = np.empty((0, 2), dtype=np.float32)

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        debug = bgr.copy()

        specs = {
            'YELLOW': (np.array([20, 120, 80]),  np.array([38, 255, 255]),  (0, 220, 255), 'lidar'),
            'BLUE':   (np.array([95, 90, 50]),   np.array([135, 255, 255]), (255, 60, 0),  'lidar'),
        }

        for name, (lo, hi, col, mode) in specs.items():
            if name in self.cam_confirmed:
                wx, wy, c = self.cam_confirmed[name][:3]
                px = self.slam.origin + int(wx / MAP_RES)
                py = self.slam.origin - int(wy / MAP_RES)
                cv2.circle(self.slam.overlay, (px, py), 9, c, -1)
                cv2.putText(self.slam.overlay, name, (px + 11, py + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1)
                continue

            mask = cv2.inRange(hsv, lo, hi)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
            debug[mask > 0] = [int(v) for v in col[::-1]]
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in cnts:
                if cv2.contourArea(cnt) < 120:
                    continue
                M = cv2.moments(cnt)
                if M['m00'] == 0:
                    continue
                cx_px = int(M['m10'] / M['m00'])
                bear = self.sensors.cam_fov * (0.5 - cx_px / self.sensors.cam_w)

                idx = max(0, min(lidar_w - 1,
                          int((lidar_fov / 2 - bear) / lidar_fov * lidar_w)))
                d = ranges[idx]
                if math.isinf(d) or math.isnan(d) or d < 0.20:
                    continue
                if d > PILLAR_DETECT_RANGE:
                    continue

                wb = self.loc.theta + bear
                wx = self.loc.x + d * math.cos(wb)
                wy = self.loc.y + d * math.sin(wb)

                buf = self.cam_pillar_buf[name]
                buf.append((wx, wy, self.loc.x, self.loc.y))
                if len(buf) > 60:
                    buf.pop(0)

                if len(buf) >= 5:
                    est_x = sum(p[0] for p in buf[-5:]) / 5
                    est_y = sum(p[1] for p in buf[-5:]) / 5
                    epx = self.slam.origin + int(est_x / MAP_RES)
                    epy = self.slam.origin - int(est_y / MAP_RES)
                    if 0 <= epx < MAP_SIZE and 0 <= epy < MAP_SIZE:
                        cv2.circle(self.slam.overlay, (epx, epy), 4, col, 1)

                if len(buf) >= 15:
                    recent = buf[-15:]
                    lm_mx = sum(p[0] for p in recent) / 15
                    lm_my = sum(p[1] for p in recent) / 15
                    stable = all(math.hypot(p[0] - lm_mx, p[1] - lm_my) < 0.30
                                 for p in recent)
                    if stable:
                        robot_dist = math.hypot(self.loc.x - lm_mx, self.loc.y - lm_my)
                        if robot_dist <= PILLAR_CONFIRM_DIST:
                            self.cam_confirmed[name] = (lm_mx, lm_my, col)
                            print(f"[CONFIRMED] {name} @ ({lm_mx:.2f},{lm_my:.2f})"
                                  f"  robot_dist={robot_dist:.2f}m")

        if getattr(__import__('config'), 'SHOW_CAMERA_DEBUG', False):
            cv2.imshow("CAMERA DEBUG", cv2.resize(debug, (480, 240)))
            cv2.waitKey(1)




        self.tiny_sighting = None
        min_px = getattr(__import__('config'), 'PILLAR_TINY_MIN_PX', 4)
        for name in ('BLUE', 'YELLOW'):
            if name in self.cam_confirmed:
                continue
            lo, hi, col, _ = specs[name]
            tiny_mask = cv2.inRange(hsv, lo, hi)
            px_count = cv2.countNonZero(tiny_mask)
            if px_count >= min_px:
                M = cv2.moments(tiny_mask)
                if M['m00'] > 0:
                    cx_px = int(M['m10'] / M['m00'])
                    bear = self.sensors.cam_fov * (0.5 - cx_px / self.sensors.cam_w)

                    if (self.tiny_sighting is None or
                            px_count > self.tiny_sighting['area']):
                        self.tiny_sighting = {
                            'color':   name,
                            'bearing': bear,
                            'area':    px_count,
                        }

    def _detect_floating_walls(self, bgr, depth):
        """OpenCV wood texture segmentation + Vectorized Depth Camera 3D projection & Z filtering."""
        if bgr is None or depth is None:
            self.floating_wall_points = np.empty((0, 2), dtype=np.float32)
            return

        wood_low = getattr(__import__('config'), 'WOOD_HSV_LOW', (8, 60, 70))
        wood_high = getattr(__import__('config'), 'WOOD_HSV_HIGH', (28, 255, 255))
        z_floor_guard = getattr(__import__('config'), 'FLOATING_Z_FLOOR_GUARD', 0.06)
        z_ceiling = getattr(__import__('config'), 'FLOATING_Z_CEILING', 0.26)
        z_ceiling_relaxed = getattr(__import__('config'), 'FLOATING_Z_CEILING_RELAXED', 0.80)
        z_blocking_threshold = getattr(__import__('config'), 'FLOATING_Z_BLOCKING_THRESHOLD', 0.21)
        min_depth = getattr(__import__('config'), 'FLOATING_WALL_DANGER_MIN', 0.06)
        max_depth = getattr(__import__('config'), 'FLOATING_MAX_DIST', 1.8)

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(wood_low), np.array(wood_high))


        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        wood_pixels_count = np.sum(mask > 0)
        self.wood_weak_pixels = int(wood_pixels_count)


        R = getattr(__import__('config'), 'FLOATING_VERTICAL_K_PIXELS', 6)
        valid_vertical = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((R, 1), np.uint8)).astype(bool)

        n_vertical = int(mask.sum() - valid_vertical.sum())
        y_indices, x_indices = np.where(valid_vertical > 0)
        n_pixels = len(y_indices)
        n_vert = n_vertical

        if n_pixels == 0:
            self.floating_wall_points = np.empty((0, 2), dtype=np.float32)
            return


        sub_mask = (y_indices % 4 == 0) & (x_indices % 4 == 0)
        y_indices = y_indices[sub_mask]
        x_indices = x_indices[sub_mask]
        subsampled_pixels = len(y_indices)

        if subsampled_pixels == 0:
            self.floating_wall_points = np.empty((0, 2), dtype=np.float32)
            return


        if depth.shape != bgr.shape[:2]:
            depth = cv2.resize(depth, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

        depths = depth[y_indices, x_indices]


        valid_depth = (~np.isnan(depths)) & (~np.isinf(depths)) & (depths >= min_depth) & (depths <= max_depth)
        n_far = np.sum(~valid_depth)

        y_indices = y_indices[valid_depth]
        x_indices = x_indices[valid_depth]
        depths = depths[valid_depth]


        bias = getattr(__import__('config'), 'FLOATING_DEPTH_BIAS', 0.0)
        depths = np.maximum(0.01, depths + bias)

        valid_depth_pixels = len(y_indices)

        if valid_depth_pixels == 0:
            self.floating_wall_points = np.empty((0, 2), dtype=np.float32)
            return


        width, height = self.sensors.cam_w, self.sensors.cam_h
        if self._cached_rays is None or self._cached_rays.shape[:2] != (height, width):
            self._cached_rays = self._pixel_rays(width, height, self.sensors.cam_fov)

        rays_cam = self._cached_rays[y_indices, x_indices]



        rz_cam = rays_cam[:, 2]
        hits_cam = rays_cam * (depths / rz_cam)[:, np.newaxis]


        depth_x = getattr(__import__('config'), 'DEPTH_OFFSET_X', 0.000)
        depth_y = getattr(__import__('config'), 'DEPTH_OFFSET_Y', 0.037)
        depth_z = getattr(__import__('config'), 'DEPTH_OFFSET_Z', 0.199)
        origins_world, dirs_world = self._camera_to_world(hits_cam, depth_x, depth_y, depth_z)
        hits_world = origins_world + dirs_world


        mxs = self.slam.origin + (hits_world[:, 0] / self.slam.map_res).astype(np.int32)
        mys = self.slam.origin - (hits_world[:, 1] / self.slam.map_res).astype(np.int32)

        in_bounds = (mxs >= 0) & (mxs < self.slam.map_size) & (mys >= 0) & (mys < self.slam.map_size)
        mxs = mxs[in_bounds]
        mys = mys[in_bounds]
        z_world = hits_world[in_bounds, 2]
        hits_world_filtered = hits_world[in_bounds]
        depths_filtered = depths[in_bounds]



        dx = hits_world_filtered[:, 0] - self.loc.x
        dy = hits_world_filtered[:, 1] - self.loc.y
        dist = np.sqrt(dx*dx + dy*dy)
        bearing = np.arctan2(dy, dx) - self.loc.theta


        bearing = (bearing + np.pi) % (2*np.pi) - np.pi

        fov_guard = getattr(__import__('config'), 'FLOATING_FOV_GUARD', 0.7)
        max_dist = getattr(__import__('config'), 'FLOATING_MAX_DIST', 1.5)
        in_cone = (np.abs(bearing) < fov_guard) & (dist < max_dist)

        mxs = mxs[in_cone]
        mys = mys[in_cone]
        z_world = z_world[in_cone]
        hits_world_filtered = hits_world_filtered[in_cone]
        depths_filtered = depths_filtered[in_cone]

        if len(mxs) == 0:
            self.floating_wall_points = np.empty((0, 2), dtype=np.float32)
            return


        is_blocking = np.sum((z_world >= z_floor_guard) & (z_world < z_blocking_threshold)) >= 8
        if not is_blocking:
            self.floating_wall_points = np.empty((0, 2), dtype=np.float32)
            return
        z_ceil_val = z_blocking_threshold

        is_floor = z_world <= z_floor_guard
        is_ceiling = z_world >= z_ceil_val

        n_floor = np.sum(is_floor)
        n_ceiling = np.sum(is_ceiling)


        is_low = (z_world < z_floor_guard) & (z_world > -0.10)
        low_cells_set = set(zip(mxs[is_low], mys[is_low]))



        lidar_walls = (self.slam.log_map > 0.4) & (self.slam.floating_wall_mask_global == 0)
        lidar_walls_u8 = lidar_walls.astype(np.uint8)




        r_margin = getattr(__import__('config'), 'PILLAR_RADIUS_M', 0.10) + 0.03
        margin_cells_pillar = int(math.ceil(r_margin / self.slam.map_res))

        pillar_coords = []
        for color in ['BLUE', 'YELLOW', 'GREEN']:
            if color in self.cam_confirmed:
                pillar_coords.append(self.cam_confirmed[color][:2])
            else:
                buf = self.cam_pillar_buf.get(color, [])
                if len(buf) >= 3:
                    pillar_coords.append(buf[-1][:2])

        pillar_coords.extend(self.checked_lidar_pillars)
        pillar_coords.extend(self.unidentified_lidar_pillars)

        safety_margin_m = getattr(__import__('config'), 'FLOATING_WALL_MIN_DIST_M', 0.12)
        margin_cells = int(math.ceil(safety_margin_m / self.slam.map_res))

        kernel_sz = 2 * margin_cells + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_sz, kernel_sz))
        lidar_dilated = cv2.dilate(lidar_walls_u8, kernel)


        for px, py in pillar_coords:
            pmx = self.slam.origin + int(px / self.slam.map_res)
            pmy = self.slam.origin - int(py / self.slam.map_res)
            cv2.circle(lidar_dilated, (pmx, pmy), margin_cells_pillar, 0, -1)



        z_ok = (z_world >= z_floor_guard) & (z_world <= z_ceil_val)

        not_lidar = (lidar_dilated[mys[z_ok], mxs[z_ok]] == 0)
        n_lidar_skip = int((~not_lidar).sum())

        not_low = ~is_low[z_ok]
        valid_mask = not_lidar & not_low
        floating_pts_arr = hits_world_filtered[z_ok][valid_mask, :2]
        n_accepted = len(floating_pts_arr)

        min_pts = getattr(__import__('config'), 'FLOATING_WALL_MIN_POINTS', 30)
        if n_accepted >= min_pts:
            self.floating_wall_points = floating_pts_arr.astype(np.float32)
        else:
            self.floating_wall_points = np.empty((0, 2), dtype=np.float32)
            n_accepted = 0


        if not hasattr(self, '_detect_cycle'):
            self._detect_cycle = 0
        self._detect_cycle += 1

        if self._detect_cycle % 30 == 0:
            print("FLOATING_WALL",
                  f"frame={self.slam.frame_iter} detected_pixels={n_pixels} accepted={n_accepted} "
                  f"floor_rejected={n_floor} ceiling_rejected={n_ceiling} "
                  f"far_rejected={n_far} vertical_rejected={n_vert} "
                  f"lidar_override={n_lidar_skip}")

    def detect_unmapped_lidar_pillars(self, ranges, w, fov):
        """Detect unmapped pillars from LiDAR."""
        if ranges is None: return

        r_arr = np.array(ranges)


        start_scan = int(w * 0.1)
        end_scan = int(w * 0.9)
        r_center = r_arr[start_scan:end_scan]


        valid_mask = (~np.isinf(r_center)) & (~np.isnan(r_center)) & (r_center > 0.1) & (r_center < 2.5)
        clean_r = np.where(valid_mask, r_center, 10.0)


        diffs = np.diff(clean_r)



        starts = np.where(diffs < -0.35)[0] + 1
        ends = np.where(diffs > 0.35)[0]

        pillars_found = []

        for s in starts:

            valid_ends = ends[ends >= s]
            if len(valid_ends) == 0:
                continue
            e = valid_ends[0]


            cluster_len = e - s + 1
            if cluster_len < 2 or cluster_len > 15:
                continue



            if np.mean(valid_mask[s:e+1]) < 0.95:
                continue


            idx1, idx2 = start_scan + s, start_scan + e
            d1, d2 = r_arr[idx1], r_arr[idx2]

            a1 = fov/2 - idx1 * fov / w
            x1, y1 = d1 * math.cos(a1), d1 * math.sin(a1)

            a2 = fov/2 - idx2 * fov / w
            x2, y2 = d2 * math.cos(a2), d2 * math.sin(a2)

            width = math.hypot(x2 - x1, y2 - y1)



            if 0.14 <= width <= 0.25:

                c_idx = start_scan + s + cluster_len // 2
                d_c = r_arr[c_idx]
                a_c = fov/2 - c_idx * fov / w

                if not hasattr(self, 'loc') or self.loc is None: continue

                ray_yaw = self.loc.theta + getattr(__import__('config'), 'LIDAR_OFFSET_YAW', 0.0) + a_c
                sx, sy = getattr(__import__('config'), 'LIDAR_OFFSET_X', 0.0), getattr(__import__('config'), 'LIDAR_OFFSET_Y', 0.0)

                cos_t, sin_t = math.cos(self.loc.theta), math.sin(self.loc.theta)
                sensor_wx = self.loc.x + sx * cos_t - sy * sin_t
                sensor_wy = self.loc.y + sx * sin_t + sy * cos_t

                wx = sensor_wx + d_c * math.cos(ray_yaw)
                wy = sensor_wy + d_c * math.sin(ray_yaw)

                pillars_found.append((wx, wy))


        for (wx, wy) in pillars_found:
            is_mapped = False


            if hasattr(self, 'slam') and self.slam is not None:
                mx = self.slam.origin + int(wx / getattr(__import__('config'), 'MAP_RES', 0.02))
                my = self.slam.origin - int(wy / getattr(__import__('config'), 'MAP_RES', 0.02))
                if 0 <= mx < getattr(__import__('config'), 'MAP_SIZE', 1000) and 0 <= my < getattr(__import__('config'), 'MAP_SIZE', 1000):
                    if self.slam.log_map[my, mx] > 0.5:
                        is_mapped = True

            if not is_mapped:
                for color, pos in self.cam_confirmed.items():
                    if math.hypot(wx - pos[0], wy - pos[1]) < 0.4:
                        is_mapped = True
                        break

            if not is_mapped:
                for (cx, cy) in self.checked_lidar_pillars:
                    if math.hypot(wx - cx, wy - cy) < 0.4:
                        is_mapped = True
                        break

            if not is_mapped:
                for (ux, uy) in self.unidentified_lidar_pillars:
                    if math.hypot(wx - ux, wy - uy) < 0.4:
                        is_mapped = True
                        break

            if not is_mapped:
                self.unidentified_lidar_pillars.append((wx, wy))
                print(f"LIDAR_PILLAR: New unidentified round object detected at ({wx:.2f}, {wy:.2f})")
