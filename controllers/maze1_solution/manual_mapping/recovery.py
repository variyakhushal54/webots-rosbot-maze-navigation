# This file handles cases where the robot cannot continue along its planned path.
# It checks whether turning or reversing is safe before trying an escape movement.
# Recovery becomes more forceful in stages and returns control when the robot is free.

import math
import numpy as np
import cv2
from config import *

class RecoveryManager:
    def note_commanded_motion(self, vx, om=0.0):
        self._last_cmd_vx = vx

    def check_rotation_safety(self, target_yaw_diff, margin=0.03):
        """Docstring."""
        ranges, lw, fov = self.sensors.get_lidar()
        if lw == 0 or not ranges:
            return True

        mid = lw // 2
        angle_step = fov / lw


        points = []
        lidar_x = getattr(__import__('config'), 'LIDAR_OFFSET_X', 0.02)
        for i, r in enumerate(ranges):
            if math.isinf(r) or math.isnan(r) or r <= 0.0:
                continue
            theta = (i - mid) * angle_step
            px = lidar_x + r * math.cos(theta)
            py = r * math.sin(theta)
            points.append((px, py))


        import config as _cfg
        for key, s_list in self.sensors.ir.items():
            for idx, sensor in enumerate(s_list):
                val = sensor.getValue()
                if math.isinf(val) or math.isnan(val) or val <= 0.0 or val >= 4.0:
                    continue
                if key == 'front':
                    s_name = 'fl' if idx == 0 else 'fr'
                else:
                    s_name = 'rl' if idx == 0 else 'rr'

                if s_name in _cfg.IR_OFFSETS:
                    x_off, y_off, _, yaw_off = _cfg.IR_OFFSETS[s_name]
                    px = x_off + val * math.cos(yaw_off)
                    py = y_off + val * math.sin(yaw_off)
                    points.append((px, py))

        if not points:
            return True


        steps = int(max(3, abs(target_yaw_diff) / math.radians(5)))
        angles = np.linspace(0, target_yaw_diff, steps)

        half_l = _cfg.ROBOT_HALF_L + margin
        half_w = _cfg.ROBOT_HALF_W + margin

        for phi in angles:
            cos_phi = math.cos(phi)
            sin_phi = math.sin(phi)
            for px, py in points:

                rx = px * cos_phi + py * sin_phi
                ry = -px * sin_phi + py * cos_phi
                if abs(rx) < half_l and abs(ry) < half_w:
                    return False

        return True

    def get_lidar_clearance_in_sector(self, angle_rad, sector_width_rad=math.radians(30)):
        """Docstring."""
        ranges, lw, fov = self.sensors.get_lidar()
        if lw == 0 or not ranges:
            return 0.0

        mid = lw // 2
        angle_step = fov / lw

        min_r = 4.0
        half_sector = sector_width_rad / 2.0

        for i, r in enumerate(ranges):
            if math.isinf(r) or math.isnan(r) or r <= 0.0:
                continue
            theta = (i - mid) * angle_step

            diff = (theta - angle_rad + math.pi) % (2 * math.pi) - math.pi
            if abs(diff) <= half_sector:
                if r < min_r:
                    min_r = r
        return min_r

    def _lidar_open_ahead(self):
        f_dist, _, _, _ = self.sensors.get_distances_at_corners()
        return f_dist >= getattr(__import__('config'), 'FLOATING_STUCK_LIDAR_OPEN', 0.30)

    def _map_open_ahead(self):
        if not hasattr(self, 'slam') or self.slam is None: return True
        import config as _cfg
        import math, numpy as np
        theta, cos_t, sin_t = self.loc.theta, math.cos(self.loc.theta), math.sin(self.loc.theta)
        origin, res, reach = self.slam.origin, self.slam.map_res, getattr(_cfg, 'INFERRED_FLOATING_FORWARD_REACH', 0.28)
        for dist in np.linspace(0.08, reach, 4):
            mx, my = origin + int((self.loc.x + dist * cos_t) / res), origin - int((self.loc.y + dist * sin_t) / res)
            if 0 <= mx < _cfg.MAP_SIZE and 0 <= my < _cfg.MAP_SIZE and self.slam.log_map[my, mx] > 3.0: return False
        return True

    def _map_open_behind(self, reach=0.18):
        if not hasattr(self, 'slam') or self.slam is None: return True
        import config as _cfg
        import math, numpy as np
        theta, cos_t, sin_t = self.loc.theta, math.cos(self.loc.theta), math.sin(self.loc.theta)
        origin, res = self.slam.origin, self.slam.map_res



        left_cos, left_sin = math.cos(theta + math.pi/2), math.sin(theta + math.pi/2)

        for dist in np.linspace(0.05, reach, 5):
            for side_offset in [-0.05, 0.0, 0.05]:
                cx = self.loc.x - dist * cos_t + side_offset * left_cos
                cy = self.loc.y - dist * sin_t + side_offset * left_sin
                mx = origin + int(cx / res)
                my = origin - int(cy / res)
                if 0 <= mx < _cfg.MAP_SIZE and 0 <= my < _cfg.MAP_SIZE:
                    if self.slam.log_map[my, mx] > getattr(_cfg, 'WALL_THRESH', 2.0):
                        return False
        return True

    def _map_open_ahead_nudge(self, reach=0.08):
        if not hasattr(self, 'slam') or self.slam is None: return True
        import config as _cfg
        import math, numpy as np
        theta, cos_t, sin_t = self.loc.theta, math.cos(self.loc.theta), math.sin(self.loc.theta)
        origin, res = self.slam.origin, self.slam.map_res



        left_cos, left_sin = math.cos(theta + math.pi/2), math.sin(theta + math.pi/2)

        for dist in np.linspace(0.05, reach, 5):
            for side_offset in [-0.05, 0.0, 0.05]:
                cx = self.loc.x + dist * cos_t + side_offset * left_cos
                cy = self.loc.y + dist * sin_t + side_offset * left_sin
                mx = origin + int(cx / res)
                my = origin - int(cy / res)
                if 0 <= mx < _cfg.MAP_SIZE and 0 <= my < _cfg.MAP_SIZE:
                    if self.slam.log_map[my, mx] > getattr(_cfg, 'WALL_THRESH', 2.0):
                        return False
        return True

    def _is_floating_wall_stuck(self):
        return (getattr(self, '_last_cmd_vx', 0.0) > 0.05 and self._lidar_open_ahead() and self._map_open_ahead())
    def __init__(self, motion, sensors, planner, loc):
        self.motion = motion
        self.sensors = sensors
        self.planner = planner
        self.loc = loc
        self.recovery_level = 0
        self.recovery_substate = None
        self.stuck_counter = 0
        self.panic_count = 0
        self.last_progress_time = 0.0
        self.last_nav_x = 0.0
        self.last_nav_y = 0.0



    def _execute_stuck_recovery(self):
        """Docstring."""
        if not hasattr(self, 'recovery_level'):    self.recovery_level = 0
        if not hasattr(self, 'recovery_substate'): self.recovery_substate = None
        if not hasattr(self, 'recovery_anchor'):   self.recovery_anchor = None

        if self.recovery_substate is None:
            self.recovery_level += 1
            if self.recovery_level >= 4:
                sm = self.state_machine
                if hasattr(sm, 'current_frontier_target') and sm.current_frontier_target:
                    print("RECOVERY", f"Aborting impossible goal {sm.current_frontier_target} (L4 escalation)")
                    if hasattr(sm, 'failed_goals'):
                        sm.failed_goals.add(sm.current_frontier_target)
                    sm.current_frontier_target = None

            self.planner.nav_path = []
            self.recovery_anchor = (self.loc.x, self.loc.y, self.loc.theta)


            candidates = [math.pi/2, -math.pi/2, math.pi, -math.pi]
            rated = []
            for angle in candidates:
                safe = self.check_rotation_safety(angle, margin=0.03)
                clearance = self.get_lidar_clearance_in_sector(angle)
                rated.append((safe, clearance, angle))


            rated.sort(key=lambda x: (1 if x[0] else 0, x[1]), reverse=True)


            idx = min(len(rated) - 1, max(0, self.recovery_level - 1))
            best_angle = rated[idx][2]
            best_safe = rated[idx][0]
            best_clearance = rated[idx][1]

            self._recovery_target_yaw_diff = best_angle
            self._recovery_start_theta = self.loc.theta
            self._recovery_start_time = self.robot.getTime()
            self._recovery_phase = 'ROTATE' if best_safe else 'NUDGE'

            print("RECOVERY", f"LEVEL {self.recovery_level} selected target angle={math.degrees(best_angle):+.0f}°, phase={self._recovery_phase}, clearance={best_clearance:.2f}m")

            if   self.recovery_level == 1: self.recovery_substate = 'L1_TURN'
            elif self.recovery_level == 2: self.recovery_substate = 'L2_TURN'
            elif self.recovery_level == 3: self.recovery_substate = 'L3_TURN'
            else:                          self.recovery_substate = 'L4_FRONTIER'

        f_dist, l_s, r_s, rear = self.sensors.get_distances_at_corners()
        ax, ay, ath = self.recovery_anchor


        if self.recovery_substate in ['L1_TURN', 'L2_TURN', 'L3_TURN']:

            if self._recovery_phase == 'NUDGE':
                if self.check_rotation_safety(self._recovery_target_yaw_diff, margin=0.03):
                    print("RECOVERY", "Nudge complete: turning is now safe. Transitioning to ROTATE phase.")
                    self._recovery_phase = 'ROTATE'
                    self._recovery_start_theta = self.loc.theta
                else:
                    if f_dist > rear:

                        if f_dist < 0.05 or not self._map_open_ahead_nudge(reach=0.08):
                            print("RECOVERY", f"Cannot nudge forward further (front={f_dist:.2f}m). Forcing ROTATE phase.")
                            self._recovery_phase = 'ROTATE'
                            self._recovery_start_theta = self.loc.theta
                        else:
                            vx_safe, _, _ = self.motion._apply_safety_guard(0.10, 0.0, 0.0)
                            if vx_safe == 0.0:
                                print("RECOVERY", "Nudge forward blocked by safety guard. Forcing ROTATE phase.")
                                self._recovery_phase = 'ROTATE'
                                self._recovery_start_theta = self.loc.theta
                            else:
                                self.motion._set_motors(vx_safe, 0.0)
                                return False
                    else:

                        if rear < 0.05 or not self._map_open_behind(reach=0.08):
                            print("RECOVERY", f"Cannot nudge backward further (rear={rear:.2f}m). Forcing ROTATE phase.")
                            self._recovery_phase = 'ROTATE'
                            self._recovery_start_theta = self.loc.theta
                        else:
                            vx_safe, _, _ = self.motion._apply_safety_guard(-0.10, 0.0, 0.0)
                            if vx_safe == 0.0:
                                print("RECOVERY", "Nudge backward blocked by safety guard. Forcing ROTATE phase.")
                                self._recovery_phase = 'ROTATE'
                                self._recovery_start_theta = self.loc.theta
                            else:
                                self.motion._set_motors(vx_safe, 0.0)
                                return False


            if self._recovery_phase == 'ROTATE':
                rotated = abs(((self.loc.theta - self._recovery_start_theta + math.pi) % (2 * math.pi)) - math.pi)
                target_rotated = abs(self._recovery_target_yaw_diff)
                elapsed = self.robot.getTime() - self._recovery_start_time

                if rotated >= target_rotated - 0.05 or elapsed > 4.5:
                    print("RECOVERY", f"Turn recovery complete: rotated={math.degrees(rotated):.0f}° target={math.degrees(target_rotated):.0f}° elapsed={elapsed:.1f}s")
                    self.motion._set_motors(0.0, 0.0)
                    self.recovery_substate = None
                    self.stuck_counter = 0
                    self.last_nav_x, self.last_nav_y = self.loc.x, self.loc.y
                    self.last_progress_time = self.robot.getTime()
                    return True

                om_cmd = math.copysign(2.5, self._recovery_target_yaw_diff)
                _, _, om_safe = self.motion._apply_safety_guard(0.0, 0.0, om_cmd)
                self.motion._set_motors(0.0, om_safe)
                return False


        if self.recovery_substate == 'L4_FRONTIER':
            rmx = self.slam.origin + int(self.loc.x / MAP_RES)
            rmy = self.slam.origin - int(self.loc.y / MAP_RES)
            self.planner.force_kernel = 0
            found = False

            for fx, fy in self.planner._find_frontiers()[:15]:
                p = self.planner._astar(rmx, rmy, fx, fy)
                if p:
                    self.planner.nav_path = p
                    print("RECOVERY", f"L4 escape via frontier ({fx},{fy})")
                    found = True
                    break

            if not found and len(self.history.coord_history) > 10:
                print("RECOVERY",
                      f"L4 no live frontier — trying {len(self.history.coord_history)} "
                      f"saved waypoints as escape targets")
                stuck_x, stuck_y = self.loc.x, self.loc.y
                candidates = sorted(
                    self.history.coord_history,
                    key=lambda p: -math.hypot(p['x'] - stuck_x, p['y'] - stuck_y)
                )
                for entry in candidates[:30]:
                    ex, ey = entry['x'], entry['y']
                    if math.hypot(ex - stuck_x, ey - stuck_y) < 0.30:
                        continue
                    gmx = self.slam.origin + int(ex / MAP_RES)
                    gmy = self.slam.origin - int(ey / MAP_RES)
                    p = self.planner._astar(rmx, rmy, gmx, gmy)
                    if p:
                        self.planner.nav_path = p
                        print("RECOVERY",
                              f"L4 escape via saved waypoint ({ex:.2f},{ey:.2f}) "
                              f"dist={math.hypot(ex-stuck_x,ey-stuck_y):.2f}m")
                        found = True
                        break
                if not found:
                    print("RECOVERY", "L4 exhausted: frontiers and saved waypoints both failed")

            self.planner.force_kernel = None
            self.motion._set_motors(0.0, 0.0)
            self.recovery_substate = None
            self.recovery_level    = 0
            self.stuck_counter     = 0
            self.last_nav_x, self.last_nav_y = self.loc.x, self.loc.y
            self.last_progress_time = self.robot.getTime()


            self._recovery_cooldown_ticks = 100
            self._recovery_escape_stuck_pos = (self.loc.x, self.loc.y)
            if hasattr(self, 'state_machine') and self.state_machine is not None:
                self.state_machine._recovery_escape_active = True
                self.state_machine._recovery_escape_origin = (self.loc.x, self.loc.y)
            return True


        self.recovery_substate = None
        return True
