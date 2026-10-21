# This file controls the order of tasks the robot performs during the mission.
# It moves between scanning, exploring, finding pillars, navigating, and recovery states.
# Each update chooses the next target or motion command from the latest sensor results.

import math
import numpy as np
import cv2
import heapq
from config import *
import config

class MissionStateMachine:
    def __init__(self, perception, planner, motion, recovery, slam, loc, history):
        self.perception = perception
        self.planner = planner
        self.motion = motion
        self.recovery = recovery
        self.slam = slam
        self.loc = loc
        self.history = history
        self.state = STATE_STARTUP
        self.nav_state = 'DRIVE'
        self.nav_back_end = 0.0
        self.failed_goals = set()
        self.current_frontier_target = None
        self.t_start = None
        self._blue_visited = False
        self._yellow_visited = False

        self.completed_360_scans = []

    def run(self, frame, t):
        """Docstring."""

        if self.nav_state == 'DRIVE' and self.state in [STATE_EXPLORE, STATE_NAV_BLUE, STATE_NAV_YELLOW]:
            cooldown = getattr(self, '_as_cooldown_until', 0.0)
            if t > cooldown:

                wood_px = getattr(self.perception, 'wood_weak_pixels', 0)
                n_strong = len(getattr(self.perception, 'floating_wall_points', []))

                if wood_px >= 120 or n_strong >= 8:

                    nearest_dist = float('inf')
                    if n_strong > 0:
                        pts = self.perception.floating_wall_points
                        dists = np.hypot(pts[:, 0] - self.loc.x, pts[:, 1] - self.loc.y)
                        nearest_dist = np.min(dists)

                    confirm_dist = getattr(__import__('config'), 'FLOATING_WALL_CONFIRM_DIST_M', 0.55)

                    if nearest_dist <= confirm_dist + 0.10:
                        is_stamped_ahead = False
                        for dist in np.arange(0.1, 0.45, 0.05):
                            wx = self.loc.x + dist * math.cos(self.loc.theta)
                            wy = self.loc.y + dist * math.sin(self.loc.theta)
                            mx = self.slam.origin + int(wx / MAP_RES)
                            my = self.slam.origin - int(wy / MAP_RES)
                            if 0 <= mx < MAP_SIZE and 0 <= my < MAP_SIZE:
                                if self.slam.floating_wall_mask_global[my, mx] > 0 or self.slam.floating_wall_inferred_mask[my, mx] > 0:
                                    is_stamped_ahead = True
                                    break

                        if not is_stamped_ahead:
                            self.nav_state = 'ACTIVE_SENSING'
                            if hasattr(self, '_as_start_frame'): delattr(self, '_as_start_frame')
                            return

        rmx, rmy = self.slam.origin + int(self.loc.x/MAP_RES), self.slam.origin - int(self.loc.y/MAP_RES)


        if not hasattr(self, 'recent_positions'):
            self.recent_positions = []
            self.last_loop_check_t = t

        if t - self.last_loop_check_t > 2.0:
            self.recent_positions.append((self.loc.x, self.loc.y))
            if len(self.recent_positions) > 10:
                self.recent_positions.pop(0)
            self.last_loop_check_t = t

            if len(self.recent_positions) == 10 and self.state in [STATE_EXPLORE, STATE_NAV_BLUE, STATE_NAV_YELLOW]:
                path_len = sum(math.hypot(self.recent_positions[i][0] - self.recent_positions[i-1][0],
                                          self.recent_positions[i][1] - self.recent_positions[i-1][1])
                               for i in range(1, 10))
                displacement = math.hypot(self.recent_positions[-1][0] - self.recent_positions[0][0],
                                          self.recent_positions[-1][1] - self.recent_positions[0][1])


                if path_len > 1.5 and displacement < 0.4:
                    print("LOOP_DETECTED", "Robot is stuck in a loop (high travel, low displacement)! Forcing recovery.")
                    self.recovery.panic_count = max(self.recovery.panic_count, 3)
                    self.planner.nav_path = []
                    self.current_frontier_target = None
                    self.recent_positions.clear()

        if self.nav_state == 'SCAN':

            SCAN_COOLDOWN_M = 0.45
            for sx, sy in self.completed_360_scans:
                dist_to_scan = math.hypot(self.loc.x - sx, self.loc.y - sy)
                if dist_to_scan < SCAN_COOLDOWN_M:
                    print(f"SCAN: Suppressed — only {dist_to_scan:.2f}m from previous 360° scan at "
                          f"({sx:.2f}, {sy:.2f}). Skipping.")
                    self.nav_state = 'DRIVE'; self.planner.nav_path = []
                    return

            if not hasattr(self, '_scan_rotated'):
                self._scan_rotated = 0.0
                self._scan_last_theta = self.loc.theta
                self._scan_start_theta = self.loc.theta


            d = ((self.loc.theta - self._scan_last_theta + math.pi) % (2 * math.pi)) - math.pi
            self._scan_rotated += abs(d)
            self._scan_last_theta = self.loc.theta


            self.motion._set_motors(0.0, 1.5)

            if self._scan_rotated >= 2*math.pi - 0.2 or t > self.nav_back_end:

                self.completed_360_scans.append((self.loc.x, self.loc.y))
                self.last_360_scan_world = (self.loc.x, self.loc.y)
                print(f"SCAN: 360° complete at world ({self.loc.x:.2f}, {self.loc.y:.2f})")
                self.nav_state = 'DRIVE'; self.planner.nav_path = []
                for attr in ['_scan_rotated', '_scan_last_theta', '_scan_start_theta']:
                    if hasattr(self, attr): delattr(self, attr)
            return





        if self.nav_state == 'VISUAL_LOCK':
            color    = getattr(self, '_vlock_color',    None)
            tgt_yaw  = getattr(self, '_vlock_target_yaw', self.loc.theta)
            deadline = getattr(self, '_vlock_deadline', t)


            if color and color in self.perception.cam_confirmed:
                print("VLOCK", f"{color} CONFIRMED during visual lock. Resuming exploration.")
                self.nav_state = 'DRIVE'
                return


            if t > deadline:
                print("VLOCK", f"{color} not confirmed after {VISUAL_LOCK_TIMEOUT:.0f}s. Resuming.")
                self._vlock_cooldown_until = t + VISUAL_LOCK_COOLDOWN
                self.nav_state = 'DRIVE'
                return


            ts = getattr(self.perception, 'tiny_sighting', None)
            if ts and ts['color'] == color:
                tgt_yaw = self.loc.theta + ts['bearing']
                self._vlock_target_yaw = tgt_yaw


            err = (tgt_yaw - self.loc.theta + math.pi) % (2 * math.pi) - math.pi
            if abs(err) > 0.10:

                om = max(-2.0, min(2.0, err * 3.5))
                _, _, om_safe = self.motion._apply_safety_guard(0.0, 0.0, om)
                self.motion._set_motors(0.0, om_safe)
            else:

                om = max(-1.0, min(1.0, err * 2.0))
                vx, _, om_s = self.motion._apply_safety_guard(0.25, 0.0, om)
                self.motion._set_motors(vx, om_s)
            return





        if self.nav_state == 'ACTIVE_SENSING':
            if not hasattr(self, '_as_start_frame'):
                self._as_start_frame = frame
                self._as_original_pose = (self.loc.x, self.loc.y, self.loc.theta)
                print("ACTIVE_SENSING", "Triggered active sensing to localize floating wall!")


            is_stamped_ahead = False
            for dist in np.arange(0.1, 0.45, 0.05):
                wx = self.loc.x + dist * math.cos(self.loc.theta)
                wy = self.loc.y + dist * math.sin(self.loc.theta)
                mx = self.slam.origin + int(wx / MAP_RES)
                my = self.slam.origin - int(wy / MAP_RES)
                if 0 <= mx < MAP_SIZE and 0 <= my < MAP_SIZE:
                    if self.slam.floating_wall_mask_global[my, mx] > 0 or self.slam.floating_wall_inferred_mask[my, mx] > 0:
                        is_stamped_ahead = True
                        break

            if is_stamped_ahead:
                print("ACTIVE_SENSING", "Floating wall mapped successfully! Resuming.")
                self.nav_state = 'DRIVE'
                self.planner.nav_path = []
                for attr in ['_as_start_frame', '_as_original_pose']:
                    if hasattr(self, attr): delattr(self, attr)
                return


            elapsed = frame - self._as_start_frame
            if elapsed > 40:
                print("ACTIVE_SENSING", "Active sensing timed out. Resuming.")
                self.nav_state = 'DRIVE'
                self._as_cooldown_until = t + 5.0
                for attr in ['_as_start_frame', '_as_original_pose']:
                    if hasattr(self, attr): delattr(self, attr)
                return


            if elapsed <= 8:

                self.motion._set_motors(0.0, 0.0)
            elif elapsed <= 16:

                vx, _, om_s = self.motion._apply_safety_guard(0.07, 0.0, 0.0)
                self.motion._set_motors(vx, om_s)
            elif elapsed <= 24:

                vx, _, om_s = self.motion._apply_safety_guard(-0.07, 0.0, 0.0)
                self.motion._set_motors(vx, om_s)
            else:

                wiggle_om = 0.6 if (elapsed % 8 < 4) else -0.6
                _, _, om_s = self.motion._apply_safety_guard(0.0, 0.0, wiggle_om)
                self.motion._set_motors(0.0, om_s)
            return





        if self.nav_state == 'DRIVE' and self.state in [STATE_EXPLORE, STATE_SEARCH_TARGET, STATE_NAV_BLUE, STATE_NAV_YELLOW]:
            if getattr(self.perception, 'unidentified_lidar_pillars', []):

                lx, ly = self.perception.unidentified_lidar_pillars.pop(0)
                self._lidar_look_target = (lx, ly)
                self._lidar_look_start = t
                self.nav_state = 'LIDAR_PILLAR_LOOK'
                print(f"LIDAR_PILLAR: Interrupting navigation to look at candidate at ({lx:.2f}, {ly:.2f})")
                self.motion._set_motors(0.0, 0.0)
                return

        if self.nav_state == 'LIDAR_PILLAR_LOOK':
            lx, ly = self._lidar_look_target
            elapsed = t - self._lidar_look_start


            if elapsed > 2.5:
                print(f"LIDAR_PILLAR: Look timed out. Marking as checked.")
                self.perception.checked_lidar_pillars.append((lx, ly))
                self.nav_state = 'DRIVE'
                return


            for color, pos in self.perception.cam_confirmed.items():
                if math.hypot(lx - pos[0], ly - pos[1]) < 0.4:
                    print(f"LIDAR_PILLAR: Camera successfully confirmed {color} pillar!")
                    self.nav_state = 'DRIVE'
                    return


            target_yaw = math.atan2(ly - self.loc.y, lx - self.loc.x)
            err = (target_yaw - self.loc.theta + math.pi) % (2 * math.pi) - math.pi

            if abs(err) > 0.08:

                om = max(-1.5, min(1.5, err * 3.0))
                self.motion._set_motors(0.0, om)
            else:

                self.motion._set_motors(0.0, 0.0)


                if not hasattr(self, '_lidar_look_aligned_start'):
                    self._lidar_look_aligned_start = t
                elif t - self._lidar_look_aligned_start > 0.8:
                    print(f"LIDAR_PILLAR: Camera saw nothing. Marking as checked.")
                    self.perception.checked_lidar_pillars.append((lx, ly))
                    delattr(self, '_lidar_look_aligned_start')
                    self.nav_state = 'DRIVE'
            return

        if self.state != STATE_STARTUP and self.t_start is None:
            self.t_start = t


        self._update_visual_coverage()

        if self.state == STATE_STARTUP:
            self.motion._set_motors(0.0, 0.0)
            if frame >= 50:
                print("STATE", "Startup complete. Entering AUTO-EXPLORE mode.")
                self.state = STATE_EXPLORE

        elif self.state == STATE_EXPLORE:









            blue_buf   = self.perception.cam_pillar_buf.get('BLUE',   [])
            yellow_buf = self.perception.cam_pillar_buf.get('YELLOW', [])
            blue_confirmed   = ('BLUE'   in self.perception.cam_confirmed) or (len(blue_buf)   >= PILLAR_NAV_MIN_BUF)
            yellow_confirmed = ('YELLOW' in self.perception.cam_confirmed) or (len(yellow_buf) >= PILLAR_NAV_MIN_BUF)





            EXPLORE_HARD_TIMEOUT = 120.0
            if not hasattr(self, '_explore_start_time'):
                self._explore_start_time = t
            explore_elapsed = t - self._explore_start_time
            if explore_elapsed > EXPLORE_HARD_TIMEOUT:
                if not blue_confirmed and len(self.perception.cam_pillar_buf.get('BLUE', [])) >= 2:
                    blue_confirmed = True
                    print("EXPLORE", "Hard timeout — accepting weak BLUE sighting")
                if not yellow_confirmed and len(self.perception.cam_pillar_buf.get('YELLOW', [])) >= 2:
                    yellow_confirmed = True
                    print("EXPLORE", "Hard timeout — accepting weak YELLOW sighting")


            if not hasattr(self, '_explore_gate_tick'): self._explore_gate_tick = 0
            self._explore_gate_tick += 1
            if self._explore_gate_tick % 150 == 0 or not hasattr(self, '_map_fully_explored'):
                self._map_fully_explored = (len(self.planner._find_frontiers()) == 0)




            if blue_confirmed and yellow_confirmed and not getattr(self, '_blue_visited', False):

                _sp_deadline = getattr(self, '_safe_path_cache_deadline', 0.0)
                if t >= _sp_deadline:
                    has_safe_path = self._has_safe_explored_path_to_blue()
                    self._safe_path_cache = has_safe_path
                    self._safe_path_cache_deadline = t + 3.0
                else:
                    has_safe_path = getattr(self, '_safe_path_cache', False)
                if has_safe_path or self._map_fully_explored:
                    print("STATE", f"BOTH PILLARS LOCATED (SafePath={has_safe_path}, MapDone={self._map_fully_explored})! Transitioning to NAV_BLUE.")
                    self.state = STATE_NAV_BLUE
                    self.planner.nav_path = []
                    self.current_frontier_target = None
                    self.recovery.panic_count = 0
                    self.recovery.recovery_level = 0
                    self.recovery.recovery_substate = None
                    if hasattr(self, 'failed_goals'): self.failed_goals.clear()
                    if hasattr(self, '_goal_retry_count'): self._goal_retry_count = 0
                    return
                else:
                    if frame % 150 == 0:
                        print("EXPLORE", "Both pillars located, but path to BLUE is narrow/unexplored. Continuing exploration to open up the true corridor.")

            if blue_confirmed and yellow_confirmed and getattr(self, '_blue_visited', False):

                _sp_deadline = getattr(self, '_safe_path_yellow_cache_deadline', 0.0)
                if t >= _sp_deadline:
                    has_safe_path = self._has_safe_explored_path_to_yellow()
                    self._safe_path_yellow_cache = has_safe_path
                    self._safe_path_yellow_cache_deadline = t + 3.0
                else:
                    has_safe_path = getattr(self, '_safe_path_yellow_cache', False)
                if has_safe_path or self._map_fully_explored:
                    print("STATE", f"SAFE PATH TO YELLOW FOUND (SafePath={has_safe_path}, MapDone={self._map_fully_explored})! Transitioning to NAV_YELLOW.")
                    self.state = STATE_NAV_YELLOW
                    self.planner.nav_path = []
                    self.current_frontier_target = None
                    self.recovery.panic_count = 0
                    self.recovery.recovery_level = 0
                    self.recovery.recovery_substate = None
                    if hasattr(self, 'failed_goals'): self.failed_goals.clear()
                    if hasattr(self, '_goal_retry_count'): self._goal_retry_count = 0
                    return
                else:
                    if frame % 150 == 0:
                        print("EXPLORE", "Both pillars located and BLUE visited, but path to YELLOW is narrow/unexplored. Continuing exploration.")

            if self._map_fully_explored:


                if len(self.planner._find_frontiers()) == 0:
                    print("STATE", "Map fully explored but pillars missing. Switching to SEARCH_TARGET.")
                    self.state = STATE_SEARCH_TARGET
                    self.planner.nav_path = []
                    self.current_frontier_target = None
                    self.failed_goals.clear()
                    return





            if (getattr(self.perception, 'green_pending', False)
                    and 'GREEN' not in self.perception.cam_confirmed):
                gpos = self.perception.green_pending_pos
                if gpos is not None:

                    if not hasattr(self, '_green_confirm_start'):
                        self._green_confirm_start = t
                        self.motion._set_motors(0.0, 0.0)
                        print(f"GREEN-CONFIRM: Stopping to face green at ({gpos[0]:.2f},{gpos[1]:.2f})")

                    elapsed_gc = t - self._green_confirm_start
                    if elapsed_gc < 1.5:

                        angle_to_green = math.atan2(gpos[1] - self.loc.y,
                                                     gpos[0] - self.loc.x)
                        heading_err = ((angle_to_green - self.loc.theta + math.pi) %
                                       (2 * math.pi)) - math.pi
                        if abs(heading_err) > 0.08:
                            turn_dir = 1.0 if heading_err > 0 else -1.0
                            self.motion._set_motors(0.0, turn_dir * 1.2)
                        else:
                            self.motion._set_motors(0.0, 0.0)
                        return
                    else:

                        print("GREEN-CONFIRM: Timeout — resuming exploration")
                        self.perception.green_pending = False
                        self.perception.green_pending_pos = None
                        if hasattr(self, '_green_confirm_start'):
                            del self._green_confirm_start
                        self.planner.nav_path = []

            elif hasattr(self, '_green_confirm_start'):

                del self._green_confirm_start
                self.planner.nav_path = []




            if frame % 300 == 0:
                frontier_count = len(self.planner._find_frontiers())
                print("EXPLORE",
                    f"Gate status: BLUE={'✓' if blue_confirmed else '✗'} "
                    f"YELLOW={'✓' if yellow_confirmed else '✗'} "
                    f"MapDone={'✓' if self._map_fully_explored else f'✗ ({frontier_count} frontiers left)'}")


            if self.planner.nav_path:
                self.recovery.panic_count = 0







                ts = getattr(self.perception, 'tiny_sighting', None)
                min_px  = getattr(config, 'PILLAR_TINY_MIN_PX',  4)
                max_px  = getattr(config, 'PILLAR_VLOCK_MAX_PX', 300)
                cooldown_ok = t >= getattr(self, '_vlock_cooldown_until', 0.0)
                if (ts is not None
                        and min_px <= ts['area'] <= max_px
                        and ts['color'] not in self.perception.cam_confirmed
                        and self.recovery.recovery_substate is None
                        and cooldown_ok):
                    world_yaw = self.loc.theta + ts['bearing']
                    self._vlock_target_yaw = world_yaw
                    self._vlock_color      = ts['color']
                    self._vlock_deadline   = t + VISUAL_LOCK_TIMEOUT
                    self.nav_state = 'VISUAL_LOCK'
                    self.motion._set_motors(0.0, 0.0)
                    print("VLOCK", f"Spotted {ts['color']} ({ts['area']} px) at "
                          f"{math.degrees(ts['bearing']):+.1f}°. Locking on.")
                    return


                if self._check_and_trigger_junction_scan(t):
                    return

                if self.motion._follow_path():
                    self.planner.nav_path = []
                    self._goal_retry_count = 0




                    arrived_at_target = False
                    if self.current_frontier_target:
                        tx, ty = self.current_frontier_target
                        dist_px = math.hypot(tx - (self.slam.origin + int(self.loc.x/MAP_RES)),
                                             ty - (self.slam.origin - int(self.loc.y/MAP_RES)))
                        arrived_at_target = dist_px * MAP_RES < 0.20




                    rmx_dfs = self.slam.origin + int(self.loc.x / MAP_RES)
                    rmy_dfs = self.slam.origin - int(self.loc.y / MAP_RES)
                    ext = self._extend_corridor(rmx_dfs, rmy_dfs)
                    if ext:
                        self.planner.nav_path = ext

                        print("EXPLORE_DFS", f"Extending deeper into corridor → ({ext[-1][0]},{ext[-1][1]})")
                        return

                    if arrived_at_target:


                        print("EXPLORE", f"Arrived at frontier target. Scanning for new frontiers.")
                        self.current_frontier_target = None
                        self.nav_state = 'SCAN'
                        self.nav_back_end = t + 1.2
                        return
                    elif self.current_frontier_target:


                        tx, ty = self.current_frontier_target
                        new_path = self.planner._astar(rmx_dfs, rmy_dfs, tx, ty)
                        if new_path:
                            self.planner.nav_path = new_path
                            print("EXPLORE", f"Re-pathing to same frontier ({tx},{ty}) after waypoint.")
                        else:

                            self.current_frontier_target = None
                            self.nav_state = 'SCAN'
                            self.nav_back_end = t + 1.0
                        return
                    else:

                        self.nav_state = 'SCAN'
                        self.nav_back_end = t + 1.0
                        return


            else:
                rmx, rmy = self.slam.origin + int(self.loc.x/MAP_RES), self.slam.origin - int(self.loc.y/MAP_RES)


                if self.current_frontier_target:
                    tx, ty = self.current_frontier_target
                    dist_to_goal = math.hypot(tx - rmx, ty - rmy) * MAP_RES


                    if dist_to_goal < 0.20:
                        print("GOAL", f"ARRIVED: Reached frontier goal ({tx},{ty}), dist={dist_to_goal:.2f}m. Clearing.")
                        self.current_frontier_target = None
                        self._goal_retry_count = 0
                    else:

                        self.planner.nav_path = self.planner._astar(rmx, rmy, tx, ty)
                        if self.planner.nav_path:
                            print("GOAL", f"LATCHED: Re-pathing to ({tx},{ty}), {dist_to_goal:.1f}m away")
                            self._goal_retry_count = 0
                        else:
                            if not hasattr(self, '_goal_retry_count'): self._goal_retry_count = 0
                            self._goal_retry_count += 1
                            if self._goal_retry_count < 5:
                                print("GOAL", f"BLOCKED: Retry {self._goal_retry_count}/5 for goal ({tx},{ty})")
                                self.motion._set_motors(0.0, 0.4)
                                return
                            else:
                                print("GOAL", f"ABANDONED: Goal ({tx},{ty}) consistently unreachable.")
                                self.current_frontier_target = None
                                if hasattr(self, 'failed_goals'): self.failed_goals.add((tx, ty))
                                self._goal_retry_count = 0


                if not self.planner.nav_path:
                    if not hasattr(self, 'failed_goals'): self.failed_goals = set()



                    bias_target = None



                    if not blue_confirmed or not yellow_confirmed:
                        missing = 'BLUE' if not blue_confirmed else 'YELLOW'
                        guess = self._resolve_blue_target() if missing == 'BLUE' else self._resolve_yellow_target()
                        if guess:
                            gx, gy = guess
                            gmx, gmy = self.slam.origin + int(gx/MAP_RES), self.slam.origin - int(gy/MAP_RES)
                            self.planner.nav_path = self.planner._astar(rmx, rmy, gmx, gmy)
                            if self.planner.nav_path:
                                print("GOAL", f"PURSUE: Going to unconfirmed {missing} at ({gx:.2f}, {gy:.2f})")
                                self.current_frontier_target = (gmx, gmy)
                                return

                    if blue_confirmed and yellow_confirmed:
                        if getattr(self, '_blue_visited', False):

                            target = self._resolve_yellow_target(min_buf=3)
                            if target:
                                yx, yy = target
                                yellow_mx = self.slam.origin + int(yx / MAP_RES)
                                yellow_my = self.slam.origin - int(yy / MAP_RES)
                                bias_target = (yellow_mx, yellow_my)
                        else:

                            target = self._resolve_blue_target(min_buf=3)
                            if target:
                                bx, by = target
                                blue_mx = self.slam.origin + int(bx / MAP_RES)
                                blue_my = self.slam.origin - int(by / MAP_RES)
                                bias_target = (blue_mx, blue_my)
                    elif blue_confirmed and not yellow_confirmed:
                        target = self._resolve_blue_target(min_buf=3)
                        if target:
                            bx, by = target
                            blue_mx = self.slam.origin + int(bx / MAP_RES)
                            blue_my = self.slam.origin - int(by / MAP_RES)

                            bias_target = (2 * rmx - blue_mx, 2 * rmy - blue_my)
                    elif yellow_confirmed and not blue_confirmed:
                        target = self._resolve_yellow_target(min_buf=3)
                        if target:
                            yx, yy = target
                            yellow_mx = self.slam.origin + int(yx / MAP_RES)
                            yellow_my = self.slam.origin - int(yy / MAP_RES)

                            bias_target = (2 * rmx - yellow_mx, 2 * rmy - yellow_my)

                    if bias_target is not None:
                        frontiers = self.planner._find_frontiers(target_mx=bias_target[0],
                                                         target_my=bias_target[1])[:15]
                        print("EXPLORE",
                            f"Biasing frontier search toward missing pillar "
                            f"(opposite cell {bias_target})")
                    else:
                        frontiers = self.planner._find_frontiers()[:15]
                    valid_frontiers = [f for f in frontiers if f not in self.failed_goals]

                    found_path = False
                    for fx, fy in valid_frontiers:
                        p = self.planner._astar(rmx, rmy, fx, fy)
                        if p:
                            self.planner.nav_path = p
                            self.current_frontier_target = (fx, fy)
                            print("GOAL", f"NEW TARGET: Selected frontier ({fx},{fy})")
                            found_path = True
                            break
                        else:


                            self.failed_goals.add((fx, fy))
                            for ofx, ofy in valid_frontiers:
                                if abs(ofx - fx) <= EXPLORE_CLUSTER_R and abs(ofy - fy) <= EXPLORE_CLUSTER_R:
                                    self.failed_goals.add((ofx, ofy))

                    if not found_path:


                        has_any_reachable = False
                        for fx, fy in valid_frontiers:
                            p = self.planner._astar(rmx, rmy, fx, fy)
                            if p:
                                has_any_reachable = True
                                break
                        if not has_any_reachable:
                            print("EXPLORE", "No reachable frontiers left inside the walls! Map is fully explored.")
                            self._map_fully_explored = True
                            self.planner.nav_path = []
                            self.current_frontier_target = None
                            self.failed_goals.clear()
                            if blue_confirmed and yellow_confirmed:
                                self.state = STATE_NAV_BLUE
                            else:
                                self.state = STATE_SEARCH_TARGET
                            return

                        self.recovery.panic_count += 1
                        if self.recovery.panic_count >= 3:


                            green_pos = self.perception.cam_confirmed.get('GREEN')
                            if green_pos:
                                dist_to_green = math.hypot(self.loc.x - green_pos[0],
                                                           self.loc.y - green_pos[1])
                                if dist_to_green < 1.5:
                                    self._burn_green_exclusion_zone()
                            if not self.recovery._execute_stuck_recovery():
                                return
                            self.recovery.panic_count = 0; self.failed_goals.clear()
                        else:

                            self.motion._set_motors(0.0, 0.8)
                            self.nav_state = 'SCAN'; self.nav_back_end = t + 2.5
            if self.planner.nav_path:
                self.recovery.panic_count = 0
                self.motion._follow_path()

        elif self.state == STATE_SEARCH_TARGET:

            blue_buf   = self.perception.cam_pillar_buf.get('BLUE',   [])
            yellow_buf = self.perception.cam_pillar_buf.get('YELLOW', [])
            blue_confirmed   = ('BLUE'   in self.perception.cam_confirmed) or (len(blue_buf)   >= PILLAR_NAV_MIN_BUF)
            yellow_confirmed = ('YELLOW' in self.perception.cam_confirmed) or (len(yellow_buf) >= PILLAR_NAV_MIN_BUF)
            if blue_confirmed and yellow_confirmed:
                has_safe_path = self._has_safe_explored_path_to_blue()
                if has_safe_path or self._map_fully_explored:
                    print("STATE", f"BOTH PILLARS FOUND during search (SafePath={has_safe_path})! Switching to NAV_BLUE.")
                    self.state = STATE_NAV_BLUE
                    self.planner.nav_path = []; self.current_frontier_target = None
                    return

            if self.planner.nav_path:
                if self._check_and_trigger_junction_scan(t):
                    return
                if self.motion._follow_path():
                    self.planner.nav_path = []
                    self.nav_state = 'SCAN'
                    self.nav_back_end = t + 4.2
            else:

                target_wp = self._find_largest_uncovered_viewpoint()
                if target_wp:
                    gmx, gmy = target_wp
                    path = self.planner._astar(rmx, rmy, gmx, gmy)
                    if path:
                        self.planner.nav_path = path
                        print(f"SEARCH_TARGET: Navigating to largest uncovered patch at ({gmx},{gmy})")
                        return


                target_j = None
                if hasattr(self, 'persistent_junctions'):
                    unscanned_indices = [i for i in range(len(self.persistent_junctions))
                                         if i not in self.scanned_junctions]
                    if unscanned_indices:
                        target_j = self.persistent_junctions[unscanned_indices[0]]

                if target_j:
                    wx, wy = target_j
                    gmx = self.slam.origin + int(wx / MAP_RES)
                    gmy = self.slam.origin - int(wy / MAP_RES)
                    self.planner.nav_path = self.planner._astar(rmx, rmy, gmx, gmy)
                else:
                    print("SEARCH_TARGET: All areas checked. Slow sweep...")
                    self.motion._set_motors(0.15, 0.3)

        elif self.state == STATE_NAV_BLUE:
            self._navigate_to_pillar('BLUE', t, frame)

        elif self.state == STATE_NAV_YELLOW:
            self._navigate_to_pillar('YELLOW', t, frame)

        elif self.state == STATE_DONE:
            self.motion._set_motors(0.0, 0.0)

            if not getattr(self, '_final_hist_saved', False):
                self.history._record_position()
                self.history._save_coord_history()
                self.slam.save_debug_maps()

                if self.t_start is not None and hasattr(self, 'mission_start_time'):
                    time_to_blue = self.mission_start_time - self.t_start
                    time_to_yellow = t - self.mission_start_time
                    total_time = t - self.t_start
                    print("=" * 60)
                    print("                   MISSION ACCOMPLISHED!")
                    print(f"   TIMER 1 (SIMULATION START -> BLUE):  {time_to_blue:8.2f}s")
                    print(f"   TIMER 2 (BLUE -> YELLOW):            {time_to_yellow:8.2f}s")
                    print(f"   TOTAL TIME (SIMULATION -> YELLOW):   {total_time:8.2f}s")
                    print("=" * 60)

                self._final_hist_saved = True

        if frame % 200 == 0:
            tracked = {k: f"({sum(p[0] for p in v[-5:])/max(1,len(v[-5:])):.2f},"
                          f"{sum(p[1] for p in v[-5:])/max(1,len(v[-5:])):.2f})"
                       for k, v in self.perception.cam_pillar_buf.items() if v}
            print(f"State={self.state} pos=({self.loc.x:.2f},{self.loc.y:.2f}) "
                  f"confirmed={list(self.perception.cam_confirmed.keys())} "
                  f"tracking={tracked}")

    def _resolve_blue_target(self, min_buf=5):
        if 'BLUE' in self.perception.cam_confirmed:
            return self.perception.cam_confirmed['BLUE'][:2]
        buf = self.perception.cam_pillar_buf.get('BLUE', [])
        if len(buf) >= min_buf:

            recent = buf[-min(len(buf), 15):]
            xs = sorted(p[0] for p in recent)
            ys = sorted(p[1] for p in recent)
            n = len(xs)
            med_x = xs[n // 2] if n % 2 else 0.5 * (xs[n//2 - 1] + xs[n//2])
            med_y = ys[n // 2] if n % 2 else 0.5 * (ys[n//2 - 1] + ys[n//2])
            return (med_x, med_y)
        return None

    def _resolve_yellow_target(self, min_buf=5):
        if 'YELLOW' in self.perception.cam_confirmed:
            return self.perception.cam_confirmed['YELLOW'][:2]
        buf = self.perception.cam_pillar_buf.get('YELLOW', [])
        if len(buf) >= min_buf:

            recent = buf[-min(len(buf), 15):]
            xs = sorted(p[0] for p in recent)
            ys = sorted(p[1] for p in recent)
            n = len(xs)
            med_x = xs[n // 2] if n % 2 else 0.5 * (xs[n//2 - 1] + xs[n//2])
            med_y = ys[n // 2] if n % 2 else 0.5 * (ys[n//2 - 1] + ys[n//2])
            return (med_x, med_y)
        return None

    def _has_safe_explored_path_to_blue(self):
        target = self._resolve_blue_target(min_buf=5)
        if target is None:
            return False
        tx, ty = target


        rmx, rmy = self.slam.origin + int(self.loc.x/MAP_RES), self.slam.origin - int(self.loc.y/MAP_RES)
        has_los = (math.hypot(self.loc.x - tx, self.loc.y - ty) < VISUAL_RANGE and self._has_line_of_sight(tx, ty))
        if has_los:
            ax, ay = self._pillar_approach_point((tx, ty))
            gmx, gmy = self.slam.origin + int(ax/MAP_RES), self.slam.origin - int(ay/MAP_RES)
        else:
            raw_gmx = self.slam.origin + int(tx/MAP_RES)
            raw_gmy = self.slam.origin - int(ty/MAP_RES)
            if (0 <= raw_gmx < MAP_SIZE and 0 <= raw_gmy < MAP_SIZE
                    and self.slam.log_map[raw_gmy, raw_gmx] < FREE_THRESH):
                gmx, gmy = raw_gmx, raw_gmy
            else:
                snapped = self._bfs_nearest_free(raw_gmx, raw_gmy, max_radius=30)
                gmx, gmy = snapped if snapped else (raw_gmx, raw_gmy)


        orig_force = getattr(self.planner, 'force_kernel', None)

        self.planner.force_kernel = 7


        self.planner.block_unknown = True
        path = self.planner._astar(rmx, rmy, gmx, gmy)
        self.planner.block_unknown = False

        self.planner.force_kernel = orig_force

        return len(path) > 0

    def _has_safe_explored_path_to_yellow(self):
        target = self._resolve_yellow_target(min_buf=5)
        if target is None:
            return False
        tx, ty = target


        rmx, rmy = self.slam.origin + int(self.loc.x/MAP_RES), self.slam.origin - int(self.loc.y/MAP_RES)
        has_los = (math.hypot(self.loc.x - tx, self.loc.y - ty) < VISUAL_RANGE and self._has_line_of_sight(tx, ty))
        if has_los:
            ax, ay = self._pillar_approach_point((tx, ty))
            gmx, gmy = self.slam.origin + int(ax/MAP_RES), self.slam.origin - int(ay/MAP_RES)
        else:
            raw_gmx = self.slam.origin + int(tx/MAP_RES)
            raw_gmy = self.slam.origin - int(ty/MAP_RES)
            if (0 <= raw_gmx < MAP_SIZE and 0 <= raw_gmy < MAP_SIZE
                    and self.slam.log_map[raw_gmy, raw_gmx] < FREE_THRESH):
                gmx, gmy = raw_gmx, raw_gmy
            else:
                snapped = self._bfs_nearest_free(raw_gmx, raw_gmy, max_radius=30)
                gmx, gmy = snapped if snapped else (raw_gmx, raw_gmy)


        orig_force = getattr(self.planner, 'force_kernel', None)

        self.planner.force_kernel = 7


        self.planner.block_unknown = True
        path = self.planner._astar(rmx, rmy, gmx, gmy)
        self.planner.block_unknown = False

        self.planner.force_kernel = orig_force

        return len(path) > 0

    def _has_line_of_sight(self, target_x, target_y):
        """Docstring."""
        rmx_v = self.slam.origin + int(self.loc.x / MAP_RES)
        rmy_v = self.slam.origin - int(self.loc.y / MAP_RES)
        tmx_v = self.slam.origin + int(target_x / MAP_RES)
        tmy_v = self.slam.origin - int(target_y / MAP_RES)
        walls = (self.slam.log_map > WALL_THRESH)
        steps = max(abs(tmx_v - rmx_v), abs(tmy_v - rmy_v))
        if steps == 0: return True
        for s in range(1, steps + 1):
            frac = s / steps
            cx = int(rmx_v + (tmx_v - rmx_v) * frac)
            cy = int(rmy_v + (tmy_v - rmy_v) * frac)
            if 0 <= cy < MAP_SIZE and 0 <= cx < MAP_SIZE and walls[cy, cx]:
                return False
        return True

    def _navigate_to_pillar(self, color, t, frame):
        """Docstring."""
        replan_attr = f'_nav_{color.lower()}_last_replan'
        strikes_attr = f'_nav_{color.lower()}_bfs_strikes'
        if not hasattr(self, replan_attr): setattr(self, replan_attr, -9999)
        if not hasattr(self, strikes_attr): setattr(self, strikes_attr, 0)

        target = self._resolve_blue_target(min_buf=5) if color == 'BLUE' else self._resolve_yellow_target(min_buf=5)
        if target is None:
            if color == 'YELLOW' and getattr(config, 'USE_DFS_LIGHTHOUSE_EXPLORATION', False):
                print("STATE", "Yellow pillar missing! Reverting to DFS Lighthouse Exploration.")
                self.state = STATE_EXPLORE; self.planner.nav_path = []
            else:
                self.motion._set_motors(0.08, 0.4 if color == 'BLUE' else 0.6)
            return

        tx, ty = target
        dist = math.hypot(self.loc.x - tx, self.loc.y - ty)
        has_los = (dist < VISUAL_RANGE and self._has_line_of_sight(tx, ty))


        if dist < PILLAR_ARRIVE_DIST + 0.05:
            if color == 'BLUE':
                self._handle_blue_touch_confirm(tx, ty, t)
            else:
                self._handle_pillar_arrival(color, t)
            return


        if dist < PILLAR_ARRIVE_DIST + 0.15 and self.recovery.panic_count >= 1 and has_los:
            print(f"NAV_{color}", f"LOS-arrival: dist={dist:.2f}m, stalled — accepting as reached")
            if color == 'BLUE':
                self._handle_blue_touch_confirm(tx, ty, t)
            else:
                self._handle_pillar_arrival(color, t)
            return


        if (dist < CLOSE_APPROACH_DIST or (has_los and self.recovery.panic_count >= 1)) and t > getattr(self, 'los_block_time', 0):
            result = self.motion._face_and_approach(tx, ty, stop_dist=CLOSE_CONFIRM_DIST)
            if result == 'ARRIVED':
                if color == 'BLUE':
                    self._handle_blue_touch_confirm(tx, ty, t)
                elif color in self.perception.cam_confirmed:
                    self._handle_pillar_arrival(color, t)
            elif result == 'BLOCKED':
                self.los_block_time = t + 3.0
            return






        if has_los:
            ax, ay = self._pillar_approach_point((tx, ty))
            gmx, gmy = self.slam.origin + int(ax/MAP_RES), self.slam.origin - int(ay/MAP_RES)
        else:

            raw_gmx = self.slam.origin + int(tx/MAP_RES)
            raw_gmy = self.slam.origin - int(ty/MAP_RES)
            if (0 <= raw_gmx < MAP_SIZE and 0 <= raw_gmy < MAP_SIZE
                    and self.slam.log_map[raw_gmy, raw_gmx] < FREE_THRESH):

                gmx, gmy = raw_gmx, raw_gmy
            else:

                snapped = self._bfs_nearest_free(raw_gmx, raw_gmy, max_radius=30)
                gmx, gmy = snapped if snapped else (raw_gmx, raw_gmy)


        strikes = getattr(self, strikes_attr)
        if strikes >= BFS_STRIKE_THRESHOLD and dist < VISUAL_RANGE and has_los:
            print(f"NAV_{color}: BFS-loop trap — forcing direct approach")
            result = self.motion._face_and_approach(tx, ty, stop_dist=CLOSE_CONFIRM_DIST)
            if result == 'ARRIVED':
                self._handle_pillar_arrival(color, t)
                setattr(self, strikes_attr, 0)
            return

        rmx, rmy = self.slam.origin + int(self.loc.x/MAP_RES), self.slam.origin - int(self.loc.y/MAP_RES)
        REPLAN_FREQ = 60 if color == 'BLUE' else 40






        _escape_active = getattr(self, '_recovery_escape_active', False)
        if _escape_active:
            escape_origin = getattr(self, '_recovery_escape_origin', (self.loc.x, self.loc.y))
            dist_escaped = math.hypot(self.loc.x - escape_origin[0], self.loc.y - escape_origin[1])
            if dist_escaped > 0.20 or not self.planner.nav_path:
                self._recovery_escape_active = False
                _escape_active = False

        if not _escape_active and (not self.planner.nav_path or (frame - getattr(self, replan_attr) >= REPLAN_FREQ)):
            new_path = self.planner._astar(rmx, rmy, gmx, gmy)
            used_surrogate = False
            if not new_path:
                bfs_t = self._bfs_nearest_free(gmx, gmy)
                if bfs_t:
                    new_path = self.planner._astar(rmx, rmy, bfs_t[0], bfs_t[1])
                    if new_path: used_surrogate = True
            if not new_path:
                self.planner.force_kernel = 0
                new_path = self.planner._astar(rmx, rmy, gmx, gmy)
                self.planner.force_kernel = None
                if new_path: used_surrogate = True

            if new_path:
                self.planner.nav_path = new_path
                setattr(self, replan_attr, frame)


                if hasattr(self.motion, '_fp_stall_ticks'):
                    self.motion._fp_stall_ticks = 0
                    self.motion._fp_last_pos = (self.loc.x, self.loc.y)
                if used_surrogate: setattr(self, strikes_attr, strikes + 1)
                else:
                    setattr(self, strikes_attr, 0)
                    self.recovery.panic_count = 0
            else:
                self.recovery.panic_count += 1
                setattr(self, strikes_attr, strikes + 1)



                frontiers = self.planner._find_frontiers()
                if len(frontiers) > 0 and self.recovery.panic_count >= 3:
                    print(f"NAV_{color}", "Cannot plan path to pillar (no route). Reverting to EXPLORE phase.")
                    self.state = STATE_EXPLORE
                    self.planner.nav_path = []
                    self.current_frontier_target = None
                    self.recovery.panic_count = 0
                    return

                if self.recovery.panic_count >= 2:
                    if not self.recovery._execute_stuck_recovery(): return
                    self.recovery.panic_count = 0
                else:
                    self.nav_state = 'SCAN'; self.nav_back_end = t + 2.0






        import config as _smcfg
        NAV_REPLAN_LOOP_LIMIT    = 5
        NAV_GRACE_TICKS          = 60
        NAV_LOOP_BREAKER_ESCALATE = getattr(_smcfg, 'NAV_LOOP_BREAKER_ESCALATE', 3)
        loop_attr    = f'_nav_{color.lower()}_replan_loop_count'
        grace_attr   = f'_nav_{color.lower()}_grace_remaining'
        breaker_attr = f'_nav_{color.lower()}_breaker_fires'
        if not hasattr(self, loop_attr):    setattr(self, loop_attr, 0)
        if not hasattr(self, grace_attr):   setattr(self, grace_attr, 0)
        if not hasattr(self, breaker_attr): setattr(self, breaker_attr, 0)

        grace_rem = getattr(self, grace_attr)
        if grace_rem > 0:

            self.planner._path_validity_grace = True
            setattr(self, grace_attr, grace_rem - 1)
            if grace_rem == 1:
                self.planner._path_validity_grace = False
        else:
            self.planner._path_validity_grace = False

        if self.planner.nav_path:
            old_path_len = len(self.planner.nav_path)
            if self.motion._follow_path():
                self.planner.nav_path = []
                if old_path_len > 1:
                    loop_count = getattr(self, loop_attr) + 1
                    setattr(self, loop_attr, loop_count)
                    if loop_count >= NAV_REPLAN_LOOP_LIMIT:
                        fires = getattr(self, breaker_attr) + 1
                        setattr(self, breaker_attr, fires)
                        setattr(self, loop_attr, 0)
                        if fires >= NAV_LOOP_BREAKER_ESCALATE:

                            print(f"NAV_{color}", f"REPLAN-LOOP BREAKER escalated (fire {fires}) "
                                  f"— calling stuck recovery directly")
                            setattr(self, breaker_attr, 0)
                            setattr(self, grace_attr, 0)
                            self.recovery._execute_stuck_recovery()
                        else:
                            print(f"NAV_{color}", f"REPLAN-LOOP BREAKER: {loop_count} consecutive "
                                  f"invalidations — granting {NAV_GRACE_TICKS}-tick grace "
                                  f"(fire {fires}/{NAV_LOOP_BREAKER_ESCALATE})")
                            setattr(self, grace_attr, NAV_GRACE_TICKS)
                else:
                    setattr(self, loop_attr, 0)
            else:

                setattr(self, loop_attr, 0)
                setattr(self, breaker_attr, 0)

    def _handle_blue_touch_confirm(self, tx, ty, t):
        """Docstring."""
        import config as _cfg
        TOUCH_DIST    = getattr(_cfg, 'BLUE_TOUCH_SENSOR_DIST', 0.05)
        TOUCH_TIMEOUT = getattr(_cfg, 'BLUE_TOUCH_TIMEOUT',      6.0)
        TOUCH_SPEED   = 0.08

        if not hasattr(self, '_blue_touch_start_t'):
            self._blue_touch_start_t = t
            print("BLUE_TOUCH", "Entering touch-confirm phase. Creeping toward blue pillar…")

        f_dist, _, _, _ = self.motion.sensors.get_distances_at_corners()

        if f_dist <= TOUCH_DIST:
            elapsed = t - self._blue_touch_start_t
            print(f"BLUE_TOUCH", f"Front sensor confirms TOUCH! f_dist={f_dist:.3f}m (elapsed={elapsed:.1f}s). "
                  f"Declaring BLUE reached.")
            self.motion._set_motors(0.0, 0.0)
            delattr(self, '_blue_touch_start_t')
            self._handle_pillar_arrival('BLUE', t)
            return

        if t - self._blue_touch_start_t >= TOUCH_TIMEOUT:
            print("BLUE_TOUCH", f"Timeout after {TOUCH_TIMEOUT}s — accepting BLUE as touched (f_dist={f_dist:.3f}m).")
            self.motion._set_motors(0.0, 0.0)
            delattr(self, '_blue_touch_start_t')
            self._handle_pillar_arrival('BLUE', t)
            return

        goal_yaw = math.atan2(ty - self.loc.y, tx - self.loc.x)
        err = (goal_yaw - self.loc.theta + math.pi) % (2 * math.pi) - math.pi
        if abs(err) > 0.15:
            om = max(-2.0, min(2.0, err * 3.0))
            self.motion._set_motors(0.0, om)
        else:
            vx, _, om_s = self.motion._apply_safety_guard(TOUCH_SPEED, 0.0, err * 1.5)
            if f_dist > getattr(_cfg, 'HARD_STOP', 0.03) + 0.01:
                self.motion._set_motors(TOUCH_SPEED, om_s)
            else:
                print("BLUE_TOUCH", f"Hard-stop limit reached (f_dist={f_dist:.3f}m) — declaring BLUE touched.")
                self.motion._set_motors(0.0, 0.0)
                delattr(self, '_blue_touch_start_t')
                self._handle_pillar_arrival('BLUE', t)

    def _handle_pillar_arrival(self, color, t):
        if color == 'BLUE':
            print(">>> MISSION START: At Blue Pillar. Starting Timer. <<<")
            self.mission_start_time = t
            self._blue_visited = True
            self.planner.nav_path = []
            self.recovery.panic_count = 0
            self.motion._set_motors(0.0, 0.0)

            self.slam.visit_count.fill(0)
            self.planner.trail_weight = 10.0

            if self._has_safe_explored_path_to_yellow():
                print("STATE", "Safe path to YELLOW exists! Transitioning to NAV_YELLOW.")
                self.state = STATE_NAV_YELLOW
            else:
                print("STATE", "No safe path to YELLOW yet. Reverting to EXPLORE phase.")
                self.state = STATE_EXPLORE
        else:
            print("STATE", "YELLOW confirmed/reached → MISSION DONE")
            self.state = STATE_DONE
            self.planner.nav_path = []
            self.motion._set_motors(0.0, 0.0)
            self.slam.save_debug_maps()

    def _pillar_approach_point(self, pillar_world_xy):
        """Docstring."""
        px, py = pillar_world_xy
        rx, ry = self.loc.x, self.loc.y
        dx, dy = rx - px, ry - py
        d = math.hypot(dx, dy)
        if d < 1e-3:
            return (px, py)

        robot_angle = math.atan2(dy, dx)
        step_m = 0.02
        best_point = None
        best_score = float('inf')


        for i in range(24):
            angle = robot_angle + (i - 12) * (math.pi / 12)
            for ring in range(20):
                target_dist = PILLAR_ARRIVE_DIST + ring * step_m
                cx = px + math.cos(angle) * target_dist
                cy = py + math.sin(angle) * target_dist
                mx = self.slam.origin + int(cx / MAP_RES)
                my = self.slam.origin - int(cy / MAP_RES)
                if not (0 <= mx < MAP_SIZE and 0 <= my < MAP_SIZE):
                    break
                if self.slam.log_map[my, mx] < FREE_THRESH:

                    score = math.hypot(rx - cx, ry - cy)

                    dot = math.cos(angle) * (-dx/d) + math.sin(angle) * (-dy/d)
                    score -= dot * 0.5
                    if score < best_score:
                        best_score = score
                        best_point = (cx, cy)
                    break

        if best_point is not None:
            return best_point


        ux, uy = dx / d, dy / d
        return (px + ux * PILLAR_ARRIVE_DIST, py + uy * PILLAR_ARRIVE_DIST)

    def _bfs_nearest_free(self, start_mx, start_my, max_radius=80):
        """Docstring."""
        rx, ry = self.slam.origin + int(self.loc.x / MAP_RES), self.slam.origin - int(self.loc.y / MAP_RES)


        pq = [(0, start_mx, start_my)]
        visited = {(start_mx, start_my)}
        DIRS = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

        while pq:
            d_robot, cx, cy = heapq.heappop(pq)
            if abs(cx - start_mx) > max_radius or abs(cy - start_my) > max_radius:
                continue
            if (0 <= cx < MAP_SIZE and 0 <= cy < MAP_SIZE
                    and self.slam.log_map[cy, cx] < FREE_THRESH):
                return (cx, cy)

            for dx, dy in DIRS:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) not in visited:
                    if 0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE:

                        if self.slam.log_map[ny, nx] > WALL_THRESH:

                            if (nx - start_mx)**2 + (ny - start_my)**2 <= 4: pass
                            else: continue
                        visited.add((nx, ny))

                        dist_r = (nx - rx)**2 + (ny - ry)**2
                        heapq.heappush(pq, (dist_r, nx, ny))
        return None

    def _find_largest_uncovered_viewpoint(self):
        """Docstring."""
        if not hasattr(self, 'visual_coverage_map'):
            return None


        free_mask   = (self.slam.log_map < FREE_THRESH).astype(np.uint8)
        unseen_mask = (free_mask & ~self.visual_coverage_map.astype(np.uint8))

        if unseen_mask.sum() == 0:
            return None


        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            unseen_mask, connectivity=8)


        best_area = 0
        best_cx, best_cy = -1, -1
        for lbl in range(1, n_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area > best_area:
                best_area = area
                best_cx = int(centroids[lbl, 0])
                best_cy = int(centroids[lbl, 1])

        if best_cx < 0:
            return None


        if not hasattr(MissionStateMachine, '_radial_offsets'):
            offsets = []
            for dx in range(-20, 21):
                for dy in range(-20, 21):
                    offsets.append((dx*dx + dy*dy, dx, dy))
            MissionStateMachine._radial_offsets = [(dx, dy) for d2, dx, dy in sorted(offsets)]

        for dx, dy in MissionStateMachine._radial_offsets:
            nx, ny = best_cx + dx, best_cy + dy
            if 0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE:
                if self.slam.log_map[ny, nx] < FREE_THRESH:
                    return (nx, ny)
        return None

    def _burn_green_exclusion_zone(self):
        """Docstring."""
        if 'GREEN' not in self.perception.cam_confirmed:
            return
        rx, ry, _ = self.perception.cam_confirmed['GREEN']


        EXCL_R_M = 0.18
        excl_px = int(EXCL_R_M / MAP_RES)
        cx = self.slam.origin + int(rx / MAP_RES)
        cy = self.slam.origin - int(ry / MAP_RES)
        for dy in range(-excl_px, excl_px + 1):
            for dx in range(-excl_px, excl_px + 1):
                if dx*dx + dy*dy <= excl_px*excl_px:
                    px, py = cx + dx, cy + dy
                    if 0 <= px < MAP_SIZE and 0 <= py < MAP_SIZE:

                        self.slam.log_map[py, px] = max(
                            self.slam.log_map[py, px], WALL_THRESH + 1.0)
        print("GREEN_EXCL", f"Permanent exclusion zone burned at ({rx:.2f},{ry:.2f}), radius={EXCL_R_M}m")

    def _update_visual_coverage(self, cam_fov_rad=1.04, max_depth_m=2.5):
        """Docstring."""
        if not hasattr(self, 'visual_coverage_map'):
            self.visual_coverage_map = np.zeros_like(self.slam.log_map, dtype=bool)


        if not hasattr(self, '_vis_cov_tick'): self._vis_cov_tick = 0
        self._vis_cov_tick += 1
        if self._vis_cov_tick % 3 != 0: return

        rx, ry, rtheta = self.loc.x, self.loc.y, self.loc.theta
        num_rays = 15


        angles = rtheta - (cam_fov_rad / 2) + np.linspace(0, cam_fov_rad, num_rays)
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)


        for i in range(num_rays):
            for d in np.arange(0.10, max_depth_m, MAP_RES):
                wx = rx + d * cos_a[i]
                wy = ry + d * sin_a[i]
                mx = self.slam.origin + int(wx / MAP_RES)
                my = self.slam.origin - int(wy / MAP_RES)

                if not (0 <= mx < MAP_SIZE and 0 <= my < MAP_SIZE): break
                self.visual_coverage_map[my, mx] = True
                if self.slam.log_map[my, mx] >= 0.0: break

    def _check_and_trigger_junction_scan(self, t):


        import config as _cfg
        if not getattr(_cfg, 'ENABLE_JUNCTION_SCAN', False):
            return False

        if not hasattr(self, 'scanned_junctions'):
            self.scanned_junctions = set()
        if not hasattr(self, 'persistent_junctions'):
            self.persistent_junctions = []

        if hasattr(self.slam, 'junctions'):

            for jy, jx in self.slam.junctions:

                wx = (jx - self.slam.origin) * MAP_RES
                wy = (self.slam.origin - jy) * MAP_RES


                idx = -1
                for i, (px, py) in enumerate(self.persistent_junctions):
                    if math.hypot(wx - px, wy - py) < 0.60:
                        idx = i; break

                if idx == -1:
                    idx = len(self.persistent_junctions)
                    self.persistent_junctions.append((wx, wy))


                if idx not in self.scanned_junctions:

                    if self.slam.log_map[jy, jx] >= WALL_THRESH - 1.0:
                        print(f"Junction {idx} is on a wall. Marking as visited.")
                        self.scanned_junctions.add(idx)

                    green_pos = self.perception.cam_confirmed.get('GREEN')
                    if green_pos:
                        if math.hypot(wx - green_pos[0], wy - green_pos[1]) < 0.40:
                            print(f"Junction {idx} is near GREEN obstacle. Marking as visited.")
                            self.scanned_junctions.add(idx)


            unscanned_indices = [i for i in range(len(self.persistent_junctions))
                                 if i not in self.scanned_junctions]
            for i in unscanned_indices:
                jx, jy = self.persistent_junctions[i]
                dist = math.hypot(self.loc.x - jx, self.loc.y - jy)
                if dist < JUNCTION_REACH_DIST:
                    self.scanned_junctions.add(i)
                    self.nav_state = 'SCAN'
                    self._scan_rotated = 0.0
                    self._scan_last_theta = self.loc.theta
                    self._scan_start_theta = self.loc.theta
                    self.nav_back_end = t + 5.2
                    print(f"Junction {i} reached at ({jx:.2f}, {jy:.2f}). Initiating 360° sweep.")
                    return True
        return False

    def _extend_corridor(self, rmx, rmy, max_steps=60):
        """Docstring."""

        DIRS = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

        cx, cy = rmx, rmy

        rtheta = self.loc.theta
        fwd_x = math.cos(rtheta)
        fwd_y = -math.sin(rtheta)

        def heading_align(d):
            return -(d[0]*fwd_x + d[1]*fwd_y)

        visited_local = set()
        visited_local.add((cx, cy))
        best_target = None

        for _ in range(max_steps):

            free_neighbors = []
            for dx, dy in sorted(DIRS, key=heading_align):
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in visited_local:
                    continue
                if not (0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE):
                    continue

                if self.slam.log_map[ny, nx] >= 0.0:
                    continue

                if self.slam.visit_count[ny, nx] > 5:
                    continue
                free_neighbors.append((nx, ny))

            if not free_neighbors:

                break

            if len(free_neighbors) >= 3:

                best_target = (cx, cy)
                break


            nx, ny = free_neighbors[0]
            visited_local.add((nx, ny))
            best_target = (nx, ny)
            cx, cy = nx, ny

        if best_target is None or best_target == (rmx, rmy):
            return None


        path = self.planner._astar(rmx, rmy, best_target[0], best_target[1])
        return path if path else None
