import math
import numpy as np
import cv2
from config import *


class MotionController:
    def __init__(self, robot, sensors, slam, loc):
        self.robot   = robot
        self.sensors = sensors
        self.slam    = slam
        self.loc     = loc
        self.motors  = sensors.motors
        self._looked_frontiers = {}
        self._current_look_target = None
        self._look_start_time = None

    def _set_motors(self, vx, vy=0.0, om=0.0):
        """Differential drive motor control."""
        if isinstance(vy, (int, float)) and om == 0.0:
            om = vy
            vy = 0.0

        if vx > 0.05 and hasattr(self, 'recovery') and self.recovery is not None:
            if hasattr(self.recovery, 'note_commanded_motion'):
                self.recovery.note_commanded_motion(vx, om)

        # Cache imminent floating wall checks to save CPU.
        if vx > 0:
            _ifw_frame = getattr(self, '_ifw_cache_frame', -99)
            _ifw_result = getattr(self, '_ifw_cache_result', False)
            if (self.sensors.robot.getTime() - getattr(self, '_ifw_cache_t', -99)) > (0.032 * 3):
                _ifw_result = self._check_imminent_floating_wall(self.sensors.get_camera_depth())
                self._ifw_cache_result = _ifw_result
                self._ifw_cache_t = self.sensors.robot.getTime()
            if _ifw_result:
                vx = 0.0
                self._floating_wall_block_count = getattr(self, '_floating_wall_block_count', 0) + 1
                if self._floating_wall_block_count % 30 == 1:
                    print("MOTION", "Floating wall ahead — forward motion blocked")

        if not hasattr(self, '_tactile_last_pos'):
            self._tactile_last_pos = (self.loc.x, self.loc.y)
            self._tactile_stall_ticks = 0

        if vx > 0.05 and abs(om) < 0.8:
            dist_moved = math.hypot(self.loc.x - self._tactile_last_pos[0],
                                    self.loc.y - self._tactile_last_pos[1])
            if dist_moved < 0.005:
                self._tactile_stall_ticks += 1
            else:
                self._tactile_stall_ticks = 0
                self._tactile_last_pos = (self.loc.x, self.loc.y)
            
            if self._tactile_stall_ticks >= 12:
                print("TACTILE", "Robot is blocked while driving forward.")
                fusion = getattr(self, 'floating_fusion', None)
                if fusion is not None:
                    fusion.note_tactile_stall()
                
                self._tactile_stall_ticks = 0
                self._tactile_last_pos = (self.loc.x, self.loc.y)
        else:
            self._tactile_stall_ticks = 0
            self._tactile_last_pos = (self.loc.x, self.loc.y)

        # Detect wheel slip when hitting an obstacle.
        f, l, r, rear = self.sensors.get_distances_at_corners()
        is_stuck_front = (vx > 0.02 and f < 0.12)
        is_stuck_rear  = (vx < -0.02 and rear < 0.12)

        if is_stuck_front or is_stuck_rear:
            self._slip_ticks = getattr(self, '_slip_ticks', 0) + 1
        else:
            # Fallback if encoders are completely stalled
            if abs(vx) > 0.04 or abs(om) > 0.2:
                curr_encs = self.sensors.get_encoders()
                last_encs = getattr(self, '_last_slip_encs', curr_encs)
                enc_diff = sum(abs(c - l) for c, l in zip(curr_encs, last_encs))
                
                if enc_diff < 0.05:
                    self._slip_ticks = getattr(self, '_slip_ticks', 0) + 1
                else:
                    self._slip_ticks = 0
                self._last_slip_encs = curr_encs
            else:
                self._slip_ticks = 0
                self._last_slip_encs = self.sensors.get_encoders()

        self._last_slip_x = self.loc.x
        self._last_slip_y = self.loc.y

        # Flag wheel slip to the localizer
        self.loc.slip_active = (getattr(self, '_slip_ticks', 0) >= 4)

        if vx == 0.0 and om == 0.0:
            for m in self.motors:
                m.setVelocity(0.0)
            return

        vx = max(-CTRL_MAX_VX, min(CTRL_MAX_VX, vx))
        om = max(-CTRL_MAX_OM, min(CTRL_MAX_OM, om))

        vL = vx - (WHEEL_TREAD / 2) * om
        vR = vx + (WHEEL_TREAD / 2) * om
        omegaL = max(-CTRL_MAX_WHEEL_RAD, min(CTRL_MAX_WHEEL_RAD, vL / WHEEL_RADIUS))
        omegaR = max(-CTRL_MAX_WHEEL_RAD, min(CTRL_MAX_WHEEL_RAD, vR / WHEEL_RADIUS))

        if not hasattr(self, '_motor_log_tick'): self._motor_log_tick = 0
        self._motor_log_tick += 1
        if self._motor_log_tick % 30 == 0 and (abs(vx) > 0.01 or abs(om) > 0.05):
            f, l, r, b = self.sensors.get_distances_at_corners()
            print("MOTOR", f"v={vx:+.2f} om={om:+.2f} | F={f:.2f} L={l:.2f} R={r:.2f} rear={b:.2f}")

        if len(self.motors) >= 4:
            self.motors[0].setVelocity(omegaL)
            self.motors[1].setVelocity(omegaR)
            self.motors[2].setVelocity(omegaL)
            self.motors[3].setVelocity(omegaR)
        elif len(self.motors) == 2:
            self.motors[0].setVelocity(omegaL)
            self.motors[1].setVelocity(omegaR)

    def _check_imminent_floating_wall(self, depth):
        """Check for floating walls immediately in front."""
        if depth is None:
            return False

        # Load config parameters
        danger_min = getattr(__import__('config'), 'FLOATING_WALL_DANGER_MIN', 0.20)
        danger_max = getattr(__import__('config'), 'FLOATING_WALL_DANGER_MAX', 0.35)
        danger_count = getattr(__import__('config'), 'FLOATING_WALL_DANGER_COUNT', 50)
        z_floor_guard = getattr(__import__('config'), 'FLOATING_Z_FLOOR_GUARD', 0.06)
        z_ceiling = getattr(__import__('config'), 'FLOATING_Z_CEILING', 0.26)
        z_ceiling_relaxed = getattr(__import__('config'), 'FLOATING_Z_CEILING_RELAXED', 0.80)
        z_blocking_threshold = getattr(__import__('config'), 'FLOATING_Z_BLOCKING_THRESHOLD', 0.21)

        # Get depth image dimensions
        H, W = depth.shape
        cy_pix, cx_pix = H // 2, W // 2

        # Region of interest for checking obstacles.
        r_start = max(0, int(cy_pix - 0.15 * H))
        r_end = min(H, int(cy_pix + 0.15 * H))
        c_start = max(0, int(cx_pix - 0.20 * W))
        c_end = min(W, int(cx_pix + 0.20 * W))

        roi_depth = depth[r_start:r_end, c_start:c_end]
        if roi_depth.size == 0:
            return False

        roi_mask = (roi_depth >= danger_min) & (roi_depth <= danger_max) & (~np.isnan(roi_depth)) & (~np.isinf(roi_depth))
        y_loc, x_loc = np.where(roi_mask)
        if len(y_loc) == 0:
            return False

        y_indices = y_loc + r_start
        x_indices = x_loc + c_start
        depths = roi_depth[roi_mask]
        
        # Offset depth bias
        bias = getattr(__import__('config'), 'FLOATING_DEPTH_BIAS', 0.0)
        depths = np.maximum(0.01, depths + bias)

        width, height = self.sensors.cam_w, self.sensors.cam_h
        perception = None
        if hasattr(self, 'planner') and self.planner is not None:
            perception = getattr(self.planner, 'perception', None)

        if perception is not None and getattr(perception, '_cached_rays', None) is not None:
            cached_rays = perception._cached_rays
        else:
            fx = width / (2.0 * math.tan(self.sensors.cam_fov / 2.0))
            cx, cy = width / 2.0, height / 2.0
            u = np.arange(width)
            v = np.arange(height)
            uu, vv = np.meshgrid(u, v)
            rx = uu - cx
            ry = vv - cy
            rz = np.full_like(rx, fx, dtype=np.float32)
            rays = np.stack([rx, ry, rz], axis=-1)
            norms = np.linalg.norm(rays, axis=-1, keepdims=True)
            cached_rays = rays / norms
            if perception is not None:
                perception._cached_rays = cached_rays

        if (H, W) == (height, width):
            rays_cam = cached_rays[y_indices, x_indices]
        else:
            y_cam = (y_indices * (height - 1) / (H - 1)).astype(np.int32)
            x_cam = (x_indices * (width - 1) / (W - 1)).astype(np.int32)
            rays_cam = cached_rays[y_cam, x_cam]

        hits_cam = rays_cam * depths[:, np.newaxis]

        if perception is not None:
            origins_world, dirs_world = perception._camera_to_world(hits_cam)
        else:
            pitch = getattr(__import__('config'), 'CAMERA_OFFSET_PITCH', 0.0)
            yaw   = getattr(__import__('config'), 'CAMERA_OFFSET_YAW', 0.0)
            rx0 = hits_cam[:, 2]
            ry0 = -hits_cam[:, 0]
            rz0 = -hits_cam[:, 1]
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
            dirs_world = np.stack([
                rx_r * cos_t - ry_r * sin_t,
                rx_r * sin_t + ry_r * cos_t,
                rz_r
            ], axis=-1)
            ox_r = getattr(__import__('config'), 'DEPTH_OFFSET_X', 0.000)
            oy_r = getattr(__import__('config'), 'DEPTH_OFFSET_Y', 0.037)
            oz_w = getattr(__import__('config'), 'DEPTH_OFFSET_Z', 0.199)
            ox_w = self.loc.x + ox_r * cos_t - oy_r * sin_t
            oy_w = self.loc.y + ox_r * sin_t + oy_r * cos_t
            origins_world = np.tile(np.array([ox_w, oy_w, oz_w], dtype=np.float32), (len(hits_cam), 1))

        hits_world = origins_world + dirs_world
        z_world = hits_world[:, 2]
        # Determine if wall block is in front of the robot
        is_blocking = np.sum((z_world >= z_floor_guard) & (z_world < z_blocking_threshold)) >= 8
        if not is_blocking:
            return False
        z_ceil_val = z_blocking_threshold
        
        z_valid = (z_world >= z_floor_guard) & (z_world <= z_ceil_val)

        danger_pixel_count = np.sum(z_valid)
        is_endangered = danger_pixel_count >= danger_count

        if is_endangered:
            # Stamp new coordinates that are not yet mapped
            danger_pts = hits_world[z_valid, :2]
            try:
                import config as _cfg
                import cv2 as _cv2
                min_dist_m = getattr(_cfg, 'FLOATING_WALL_MIN_DIST_M', 0.10)
                margin_cells = max(1, int(math.ceil(min_dist_m / self.slam.map_res)))
                lidar_walls = (self.slam.log_map > getattr(_cfg, 'WALL_THRESH', 3.0)) \
                              & (self.slam.floating_wall_mask_global == 0)
                ksz = 2 * margin_cells + 1
                kernel = _cv2.getStructuringElement(_cv2.MORPH_RECT, (ksz, ksz))
                lidar_dilated = _cv2.dilate(lidar_walls.astype(np.uint8), kernel)
                # Convert points to map cells
                mxs_d = (self.slam.origin + (danger_pts[:, 0] / self.slam.map_res)).astype(np.int32)
                mys_d = (self.slam.origin - (danger_pts[:, 1] / self.slam.map_res)).astype(np.int32)
                in_b = (mxs_d >= 0) & (mxs_d < self.slam.map_size) & \
                       (mys_d >= 0) & (mys_d < self.slam.map_size)
                not_lidar = np.zeros(len(danger_pts), dtype=bool)
                not_lidar[in_b] = (lidar_dilated[mys_d[in_b], mxs_d[in_b]] == 0)
                danger_pts = danger_pts[not_lidar]
            except Exception:
                pass
            if len(danger_pts) > 0:
                self.slam.stamp_floating_wall_points(danger_pts)

        return is_endangered

    def _face_and_approach(self, tx, ty, stop_dist=CLOSE_CONFIRM_DIST):
        """Align and approach target."""
        dist = math.hypot(tx - self.loc.x, ty - self.loc.y)
        if dist < stop_dist:
            self._set_motors(0.0, 0.0)
            return 'ARRIVED'

        goal_yaw = math.atan2(ty - self.loc.y, tx - self.loc.x)
        err = (goal_yaw - self.loc.theta + math.pi) % (2 * math.pi) - math.pi

        f_dist, l_side, r_side, _ = self.sensors.get_distances_at_corners()
        side_clear = min(l_side, r_side)

        import config as _cfg
        expected_surface_dist = dist - _cfg.ROBOT_HALF_L - getattr(_cfg, 'PILLAR_RADIUS_M', 0.10)
        if abs(err) < 0.30 and f_dist < (expected_surface_dist - 0.15):
            vx, _, om_s = self._apply_safety_guard(-0.15, 0.0, 0.0)
            self._set_motors(vx, om_s)
            return 'BLOCKED'

        if abs(err) > 0.10:
            max_om = APPROACH_MAX_OM
            if side_clear < (ROBOT_HALF_W * 0.5) + 0.05:
                max_om = APPROACH_TIGHT_OM
            om = max(-max_om, min(max_om, err * APPROACH_OM_GAIN))
            print("APPROACH", f"Pillar-Alignment: Err={err:.3f} rad, Output-Om={om:.2f}, Clearance={side_clear:.2f}m")
            _, _, om_s = self._apply_safety_guard(0.0, 0.0, om)
            self._set_motors(0.0, om_s)
            return 'TURNING'
        else:
            speed = max(0.10, min(0.35, dist * 0.4))
            guard = 0.0
            if l_side < (ROBOT_HALF_W * 0.5) + 0.02: guard -= 1.5
            if r_side < (ROBOT_HALF_W * 0.5) + 0.02: guard += 1.5
            tot_steer = (err * APPROACH_STEER_GAIN) + guard
            print("APPROACH", f"Final Forward: Dist={dist:.2f}m, Speed={speed:.2f}, Steer={tot_steer:.2f}")
            vx, _, om_s = self._apply_safety_guard(speed, 0.0, tot_steer)
            self._set_motors(vx, om_s)
            return 'DRIVING'

    def _apply_safety_guard(self, vx, vy=0.0, om=0.0):
        """Apply safety limits to avoid wall contact."""
        orig_vx = vx
        f, l, r, b = self.sensors.get_distances_at_corners()

        # Forward motion safety check
        if vx > 0:
            SENSOR_STOP = getattr(__import__('config'), 'HARD_STOP', 0.04)
            SENSOR_WARN = max(getattr(__import__('config'), 'FWD_MARGIN', 0.08) + 0.07, 0.15)
            if f < SENSOR_STOP:
                vx = 0.0
            elif f < SENSOR_WARN:
                vx *= max(0.0, (f - SENSOR_STOP) / (SENSOR_WARN - SENSOR_STOP))

            # Safety margins for map walls
            MAP_STOP_FW = 0.12
            MAP_STOP_LW = 0.04
            MAP_WARN    = 0.24
            if vx > 0 and hasattr(self, 'slam') and self.slam is not None:
                import config as _cfg, math as _math, numpy as _np
                theta   = self.loc.theta
                cos_t   = _math.cos(theta)
                sin_t   = _math.sin(theta)
                lcos    = _math.cos(theta + _math.pi / 2)
                lsin    = _math.sin(theta + _math.pi / 2)
                origin, res = self.slam.origin, self.slam.map_res
                # Scan ahead for obstacles
                nearest_fw  = MAP_WARN + 1.0
                nearest_lw  = MAP_WARN + 1.0
                for dist in _np.linspace(0.05, MAP_WARN, 10):
                    for side in [-0.06, 0.0, 0.06]:
                        cx = self.loc.x + dist * cos_t + side * lcos
                        cy = self.loc.y + dist * sin_t + side * lsin
                        mx = int(origin + cx / res)
                        my = int(origin - cy / res)
                        if 0 <= mx < _cfg.MAP_SIZE and 0 <= my < _cfg.MAP_SIZE:
                            cell_log = self.slam.log_map[my, mx]
                            if cell_log > _cfg.WALL_THRESH:
                                is_float = (self.slam.floating_wall_mask_global[my, mx] > 0)
                                if is_float:
                                    nearest_fw = min(nearest_fw, dist)
                                else:
                                    nearest_lw = min(nearest_lw, dist)
                # Apply deceleration scaling
                if nearest_fw <= MAP_STOP_FW:
                    vx = 0.0
                elif nearest_fw <= MAP_WARN:
                    fw_scale = max(0.0, (nearest_fw - MAP_STOP_FW) / (MAP_WARN - MAP_STOP_FW))
                    vx = min(vx, vx * fw_scale)
                if nearest_lw <= MAP_STOP_LW:
                    vx = min(vx, 0.04)
                elif nearest_lw <= MAP_WARN:
                    lw_scale = max(0.0, (nearest_lw - MAP_STOP_LW) / (MAP_WARN - MAP_STOP_LW))
                    vx = min(vx, max(0.04, vx * lw_scale))

        # Backward safety check
        if vx < 0:
            is_recovering = (hasattr(self, 'recovery') and self.recovery is not None and self.recovery.recovery_substate is not None)
            stop_dist = 0.06 if is_recovering else 0.05
            warn_dist = 0.12 if is_recovering else 0.16
            
            # Map-based safety check for backing up
            map_ok = True
            if hasattr(self, 'recovery') and self.recovery is not None:
                map_ok = self.recovery._map_open_behind(reach=0.15)
                
            if b < stop_dist or not map_ok:
                vx = 0.0
            elif b < warn_dist:
                scale = max(0.0, (b - stop_dist) / (warn_dist - stop_dist))
                vx *= scale

        # Rotational safety check
        if abs(om) > 0.05:
            min_clear = self.sensors.get_exact_chassis_clearance()
            is_recovering = (
                hasattr(self, 'recovery') and self.recovery is not None
                and self.recovery.recovery_substate is not None
            )
            turn_stop = getattr(__import__('config'), 'IR_CORNER_HARD_STOP', 0.03)
            turn_warn = max(getattr(__import__('config'), 'IR_CORNER_WARN_LIMIT', 0.08), turn_stop + 0.03)
            min_scale = 0.30 if is_recovering else 0.18

            if min_clear < turn_stop:
                scale = min_scale
            elif min_clear < turn_warn:
                scale = (min_clear - turn_stop) / (turn_warn - turn_stop)
                scale = max(min_scale, scale)
            else:
                scale = 1.0

            # Check rotation safety over predicted trajectory
            if hasattr(self, 'recovery') and self.recovery is not None:
                horizon = 0.12 if abs(orig_vx) < 0.02 else 0.08
                yaw_step = math.copysign(min(abs(om) * horizon, math.radians(14)), om)
                margin = getattr(__import__('config'), 'TURN_MARGIN', 0.05)
                if not self.recovery.check_rotation_safety(yaw_step, margin=margin):
                    predictive_scale = min_scale
                    for candidate in (0.75, 0.55, 0.35):
                        test_step = yaw_step * candidate
                        if self.recovery.check_rotation_safety(test_step, margin=margin):
                            predictive_scale = candidate
                            break
                    scale = min(scale, max(min_scale, predictive_scale))

            om *= scale

        return vx, vy, om

    def _follow_path(self):
        """Pure Pursuit path follower."""
        if not self.planner.nav_path:
            return True

        if not hasattr(self, '_fp_last_pos'):   self._fp_last_pos = (self.loc.x, self.loc.y)
        if not hasattr(self, '_fp_stall_ticks'): self._fp_stall_ticks = 0

        # Check if grinding wall and increase threshold
        f_dist_fp, l_dist_fp, r_dist_fp, rear_dist_fp = self.sensors.get_distances_at_corners()
        min_clear_fp = min(f_dist_fp, l_dist_fp, r_dist_fp, rear_dist_fp)
        stall_dist_thresh = 0.04 if min_clear_fp < 0.06 else 0.02

        if math.hypot(self.loc.x - self._fp_last_pos[0],
                      self.loc.y - self._fp_last_pos[1]) < stall_dist_thresh:
            self._fp_stall_ticks += 1
        else:
            self._fp_stall_ticks = 0
            self._fp_last_pos = (self.loc.x, self.loc.y)

        if not hasattr(self, '_fp_mini_loop_pos'):
            self._fp_mini_loop_pos = []
            self._fp_mini_loop_t = self.robot.getTime()

        current_t = self.robot.getTime()
        if current_t - self._fp_mini_loop_t > 0.5:
            self._fp_mini_loop_pos.append((self.loc.x, self.loc.y))
            if len(self._fp_mini_loop_pos) > 6:
                self._fp_mini_loop_pos.pop(0)
            self._fp_mini_loop_t = current_t

            if len(self._fp_mini_loop_pos) == 6:
                path_len = sum(math.hypot(self._fp_mini_loop_pos[i][0] - self._fp_mini_loop_pos[i-1][0], 
                                          self._fp_mini_loop_pos[i][1] - self._fp_mini_loop_pos[i-1][1]) 
                               for i in range(1, 6))
                displacement = math.hypot(self._fp_mini_loop_pos[-1][0] - self._fp_mini_loop_pos[0][0],
                                          self._fp_mini_loop_pos[-1][1] - self._fp_mini_loop_pos[0][1])
                
                # Check corner oscillation
                if path_len > 0.12 and displacement < 0.06:
                    print("PATH_FOLLOW", f"Corner oscillation detected! path={path_len:.2f}m, disp={displacement:.2f}m")
                    self.planner.nav_path = []
                    self._fp_stall_ticks = 0
                    self._fp_last_pos = (self.loc.x, self.loc.y)
                    self._fp_mini_loop_pos.clear()
                    self.recovery._execute_stuck_recovery()
                    return True

        if self._fp_stall_ticks < 20 and not self.planner._path_still_valid():
            print("PATH_FOLLOW", "Wall on path — discarding and re-planning.")
            self.planner.nav_path = []
            self._set_motors(0.0, 0.0)
            return True

        # Reset stall counter while spinning
        if getattr(self, '_spinning', False):
            self._fp_stall_ticks = 0
            self._fp_last_pos = (self.loc.x, self.loc.y)

        if self._fp_stall_ticks >= RECOVERY_STUCK_TICKS:
            print("PATH_FOLLOW", f"Stalled {self._fp_stall_ticks} ticks — forcing recovery NOW")
            self.planner.nav_path = []
            self._fp_stall_ticks = 0
            self._fp_last_pos = (self.loc.x, self.loc.y)
            self.recovery._execute_stuck_recovery()
            return True

        tight        = getattr(self.planner, 'last_path_kernel', 13) <= 11
        arrive_tol   = PATH_ARRIVE_TOL
        skip_tol     = max(0.08, 1.0 * MAP_RES) if tight else PATH_SKIP_TOL
        base_lookahead = FOLLOW_LOOKAHEAD_TIGHT if tight else FOLLOW_LOOKAHEAD_BASE

        def cell_to_world(cell):
            mx, my = cell
            return (mx - self.slam.origin) * MAP_RES, (self.slam.origin - my) * MAP_RES

        heading_x = math.cos(self.loc.theta)
        heading_y = math.sin(self.loc.theta)
        while len(self.planner.nav_path) > 1:
            wx, wy = cell_to_world(self.planner.nav_path[0])
            dx, dy = wx - self.loc.x, wy - self.loc.y
            d = math.hypot(dx, dy)
            behind = (dx * heading_x + dy * heading_y) < 0.0 and d < (skip_tol * 2.0)
            if d < skip_tol or behind:
                self.planner.nav_path.pop(0)
            else:
                break

        if len(self.planner.nav_path) == 1:
            gx, gy = cell_to_world(self.planner.nav_path[0])
            if math.hypot(gx - self.loc.x, gy - self.loc.y) < arrive_tol:
                self.planner.nav_path.pop(0)
                self._set_motors(0.0, 0.0)
                return True

        turn_angle = 0.0
        upcoming_sharp_turn = False
        if len(self.planner.nav_path) >= 3:
            p0 = cell_to_world(self.planner.nav_path[0])
            p1 = cell_to_world(self.planner.nav_path[1])
            p2 = cell_to_world(self.planner.nav_path[2])
            v1 = (p1[0] - p0[0], p1[1] - p0[1])
            v2 = (p2[0] - p1[0], p2[1] - p1[1])
            angle1 = math.atan2(v1[1], v1[0])
            angle2 = math.atan2(v2[1], v2[0])
            turn_angle = abs(((angle2 - angle1 + math.pi) % (2 * math.pi)) - math.pi)
            if turn_angle > 1.0:
                upcoming_sharp_turn = True

        if turn_angle > 1.4:
            lookahead = base_lookahead * 0.5
        elif turn_angle > 0.9:
            lookahead = base_lookahead * 0.7
        else:
            lookahead = base_lookahead

        target_cell = self.planner.nav_path[-1]
        acc = 0.0
        prev_x, prev_y = self.loc.x, self.loc.y
        for cell in self.planner.nav_path:
            wx, wy = cell_to_world(cell)
            acc += math.hypot(wx - prev_x, wy - prev_y)
            prev_x, prev_y = wx, wy
            if acc >= lookahead:
                target_cell = cell
                break

        tx, ty = cell_to_world(target_cell)
        dx, dy = tx - self.loc.x, ty - self.loc.y
        target_dist = math.hypot(dx, dy)
        if target_dist < 0.01:
            return False

        desired_heading = math.atan2(dy, dx)
        
        waypoint_heading_error = (desired_heading - self.loc.theta + math.pi) % (2 * math.pi) - math.pi

        # Look toward frontiers when spinning
        active_frontier_look_heading = None
        # Look only once per session
        if getattr(self, '_spinning', False) and not getattr(self, '_has_looked_during_current_spin', False) and getattr(self, 'planner', None) is not None:
            nearby_frontiers = self._find_nearby_selected_frontiers(max_dist=2.2)
            if nearby_frontiers:
                (f_cell, (fx_w, fy_w), f_dist) = nearby_frontiers[0]
                frontier_heading = math.atan2(fy_w - self.loc.y, fx_w - self.loc.x)
                frontier_error = (frontier_heading - self.loc.theta + math.pi) % (2 * math.pi) - math.pi
                current_time = self.sensors.robot.getTime()
                
                if self._current_look_target != f_cell:
                    self._current_look_target = f_cell
                    self._look_start_time = current_time
                
                if current_time - self._look_start_time > 2.0:
                    self._looked_frontiers[f_cell] = current_time
                    self._current_look_target = None
                    self._has_looked_during_current_spin = True
                else:
                    active_frontier_look_heading = frontier_heading
                    if abs(frontier_error) < 0.15:
                        self._looked_frontiers[f_cell] = current_time
                        self._current_look_target = None
                        self._has_looked_during_current_spin = True

        heading_error = (desired_heading - self.loc.theta + math.pi) % (2 * math.pi) - math.pi

        f_dist, l_dist, r_dist, rear_dist = self.sensors.get_distances_at_corners()

        # Back up if heading error is large and space is tight
        backup_heading_thresh = 0.45 if f_dist < 0.15 else 0.75
        need_backup = (not getattr(self, '_spinning', False) and abs(waypoint_heading_error) > backup_heading_thresh) or \
                      (getattr(self, '_spinning', False) and f_dist < 0.11 and abs(waypoint_heading_error) > 0.25)
        
        if need_backup and f_dist < 0.17 and rear_dist > 0.09:
            back_vx, _, back_om = self._apply_safety_guard(-0.12, 0.0, math.copysign(1.0, waypoint_heading_error))
            if abs(back_vx) < 0.03:
                if not getattr(self, '_spinning', False):
                    self._spinning = True
                    self._has_looked_during_current_spin = False
                    print("MOTION", "Backup blocked by safety. Transitioning to pure spin.")
            else:
                if getattr(self, '_spinning', False):
                    self._spinning = False
                print("MOTION", f"Backing up to turn: err={math.degrees(waypoint_heading_error):.1f}°, F={f_dist:.2f}m, B={rear_dist:.2f}m")
                self._set_motors(back_vx, back_om)
                return False

        CRITICAL_FRONT = 0.05
        CRITICAL_SIDE  = 0.06
        WARNING_FRONT  = 0.16
        WARNING_SIDE   = 0.13

        base_speed   = FOLLOW_SPEED_TIGHT if tight else FOLLOW_SPEED_BASE
        if f_dist < CRITICAL_FRONT:
            base_speed = 0.06

        speed_factor = 1.0

        SPIN_ENTER    = 0.50
        SPIN_EXIT     = 0.15
        SPIN_REENTER  = 0.40
        SPIN_COOLDOWN = 25

        if not hasattr(self, '_spinning'):          self._spinning = False
        if not hasattr(self, '_spin_cooldown_rem'): self._spin_cooldown_rem = 0

        # Cooldown check
        if self._spin_cooldown_rem > 0:
            self._spin_cooldown_rem -= 1

        # Update spin status
        if self._spinning:
            pass
        elif self._spin_cooldown_rem == 0 and abs(waypoint_heading_error) > SPIN_ENTER:
            # Enter spin mode
            self._has_looked_during_current_spin = False
            self._spinning = True
        elif self._spin_cooldown_rem > 0 and abs(waypoint_heading_error) > SPIN_REENTER:
            # Force re-entry
            self._has_looked_during_current_spin = False
            self._spinning = True

        if self._spinning:
            spin_target_error = waypoint_heading_error

            if abs(spin_target_error) < SPIN_EXIT:
                # Align complete
                self._spinning = False
                self._spin_cooldown_rem = SPIN_COOLDOWN
            else:
                # Nudge forward if close to wall
                _, _, om_safe = self._apply_safety_guard(0.0, 0.0,
                    math.copysign(min(3.0 * abs(spin_target_error), CTRL_MAX_OM), spin_target_error))

                # Creep forward if side space is narrow
                side_min = min(l_dist, r_dist)
                if side_min < (ROBOT_HALF_W + 0.08):
                    # Shift away from the nearest side wall
                    away = -1.0 if l_dist < r_dist else 1.0
                    om_safe = om_safe if math.copysign(1.0, om_safe) == away else om_safe * 0.45
                    creep = 0.04 if f_dist > 0.14 else 0.0
                    nudge_vx, _, _ = self._apply_safety_guard(creep, 0.0, 0.0)
                    self._set_motors(nudge_vx, om_safe)
                else:
                    self._set_motors(0.0, om_safe)
                return False

        if   abs(heading_error) > 1.2: speed_factor *= 0.45
        elif abs(heading_error) > 1.0: speed_factor *= 0.55
        elif abs(heading_error) > 0.6: speed_factor *= 0.70
        elif abs(heading_error) > 0.3: speed_factor *= 0.85

        if f_dist < WARNING_FRONT:
            front_factor = (f_dist - CRITICAL_FRONT) / (WARNING_FRONT - CRITICAL_FRONT)
            speed_factor *= max(0.40, front_factor)

        min_side = min(l_dist, r_dist)
        if min_side < WARNING_SIDE:
            side_factor = (min_side - CRITICAL_SIDE) / (WARNING_SIDE - CRITICAL_SIDE)
            speed_factor *= max(0.50, side_factor)

        if upcoming_sharp_turn:
            speed_factor *= 0.60
            if min_side < 0.15:
                speed_factor *= 0.75

        vx = base_speed * speed_factor

        om = heading_error * FOLLOW_OM_GAIN

        # Align to corridor center
        if l_dist < 0.22 and r_dist < 0.22:
            side_diff = l_dist - r_dist
            centering_gain = 1.0 if tight else 0.5
            om += side_diff * centering_gain

        max_om = 2.5 if vx > 0.25 else 3.5
        om = max(-max_om, min(max_om, om))

        # Slow down for obstacles
        depth_threat = False
        if getattr(self.sensors, 'rangefinder', None):
            raw_depth = self.sensors.rangefinder.getRangeImage()
            d_img  = np.array(raw_depth).reshape(self.sensors.rf_h, self.sensors.rf_w)
            d_swath = d_img[int(self.sensors.rf_h * 0.4):int(self.sensors.rf_h * 0.6),
                            int(self.sensors.rf_w * 0.4):int(self.sensors.rf_w * 0.6)]
            if np.min(d_swath) < 0.15:
                depth_threat = True

        if f_dist < 0.15 or depth_threat:
            vx *= 0.3
            if depth_threat:
                print("SENSORS", "Smart-Slowdown: Low-profile obstacle detected.")

        vx, _, om = self._apply_safety_guard(vx, 0.0, om)
        self._set_motors(vx, om)
        return False

    def _has_line_of_sight(self, rx, ry, fx, fy):
        """Check line of sight to target."""
        gmx = np.clip(self.slam.origin + int(round(fx / MAP_RES)), 0, MAP_SIZE - 1)
        gmy = np.clip(self.slam.origin - int(round(fy / MAP_RES)), 0, MAP_SIZE - 1)
        rmx = np.clip(self.slam.origin + int(round(rx / MAP_RES)), 0, MAP_SIZE - 1)
        rmy = np.clip(self.slam.origin - int(round(ry / MAP_RES)), 0, MAP_SIZE - 1)
        steps = max(abs(gmx - rmx), abs(gmy - rmy))
        if steps == 0:
            return True
        xs = np.clip(np.linspace(rmx, gmx, steps + 1).astype(np.int32), 0, MAP_SIZE - 1)
        ys = np.clip(np.linspace(rmy, gmy, steps + 1).astype(np.int32), 0, MAP_SIZE - 1)
        return not (self.slam.log_map[ys, xs] > WALL_THRESH).any()

    def _find_nearby_selected_frontiers(self, max_dist=2.2):
        """Find candidate frontiers nearby."""
        current_time = self.sensors.robot.getTime()
        frontiers = self.planner._find_frontiers()
        if not frontiers:
            return []
            
        nearby = []
        for mx, my in frontiers[:10]:
            wx = (mx - self.slam.origin) * MAP_RES
            wy = (self.slam.origin - my) * MAP_RES
            dist = math.hypot(wx - self.loc.x, wy - self.loc.y)
            if dist < max_dist:
                last_look = self._looked_frontiers.get((mx, my), 0.0)
                if current_time - last_look > 8.0:
                    if self._has_line_of_sight(self.loc.x, self.loc.y, wx, wy):
                        nearby.append(((mx, my), (wx, wy), dist))
                        
        nearby.sort(key=lambda x: x[2])
        return nearby
