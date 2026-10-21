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
        # Save every scanned position in world coordinates (x, y) during entire run
        self.completed_360_scans = []

    def run(self, frame, t):
        """
        The Main execution loop of the entire codebase.
        This handles sensor data injection, triggers SLAM updating, and hosts the
        Primary State Machine (STARTUP -> EXPLORE -> NAV_BLUE -> NAV_YELLOW -> DONE).

        KEY POLICY (v39.0):
          A* navigation toward BLUE or YELLOW pillars is BLOCKED until ALL three
          conditions are simultaneously satisfied:
            1. cam_confirmed['BLUE']   — exact BLUE coordinates known
            2. cam_confirmed['YELLOW'] — exact YELLOW coordinates known
            3. _is_map_fully_explored() returns True — no frontiers remain
          Until then the robot keeps exploring and passively buffers pillar sightings.
        """
        # --- ACTIVE PERCEPTION TRIGGER ---
        if self.nav_state == 'DRIVE' and self.state in [STATE_EXPLORE, STATE_NAV_BLUE, STATE_NAV_YELLOW]:
            cooldown = getattr(self, '_as_cooldown_until', 0.0)
            if t > cooldown:
                # Check if camera has unconfirmed floating wall cues in front
                wood_px = getattr(self.perception, 'wood_weak_pixels', 0)
                n_strong = len(getattr(self.perception, 'floating_wall_points', []))
                
                if wood_px >= 120 or n_strong >= 8:
                    # Determine distance to nearest detected floating wall point
                    nearest_dist = float('inf')
                    if n_strong > 0:
                        pts = self.perception.floating_wall_points
                        dists = np.hypot(pts[:, 0] - self.loc.x, pts[:, 1] - self.loc.y)
                        nearest_dist = np.min(dists)
                    
                    confirm_dist = getattr(__import__('config'), 'FLOATING_WALL_CONFIRM_DIST_M', 0.55)
                    # Only trigger active sensing if the nearest point is close enough to confirm (with a 10cm buffer)
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
        
        # --- LOOP / STUCK DETECTION ---
        if not hasattr(self, 'recent_positions'):
            self.recent_positions = []
            self.last_loop_check_t = t
            
        if t - self.last_loop_check_t > 2.0:
            self.recent_positions.append((self.loc.x, self.loc.y))
            if len(self.recent_positions) > 10:  # track over 20 seconds
                self.recent_positions.pop(0)
            self.last_loop_check_t = t
            
            if len(self.recent_positions) == 10 and self.state in [STATE_EXPLORE, STATE_NAV_BLUE, STATE_NAV_YELLOW]:
                path_len = sum(math.hypot(self.recent_positions[i][0] - self.recent_positions[i-1][0], 
                                          self.recent_positions[i][1] - self.recent_positions[i-1][1]) 
                               for i in range(1, 10))
                displacement = math.hypot(self.recent_positions[-1][0] - self.recent_positions[0][0],
                                          self.recent_positions[-1][1] - self.recent_positions[0][1])
                
                # If robot travels > 1.5m but ends up within 0.4m of start, it's looping
                if path_len > 1.5 and displacement < 0.4:
                    print("LOOP_DETECTED", "Robot is stuck in a loop (high travel, low displacement)! Forcing recovery.")
                    self.recovery.panic_count = max(self.recovery.panic_count, 3)
                    self.planner.nav_path = []
                    self.current_frontier_target = None
                    self.recent_positions.clear()

        if self.nav_state == 'SCAN':
            # ── Proximity guard: skip if within 45 cm of ANY previously completed 360° scan ─
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

            # Integrate |delta theta| correctly across wrap-around
            d = ((self.loc.theta - self._scan_last_theta + math.pi) % (2 * math.pi)) - math.pi
            self._scan_rotated += abs(d)
            self._scan_last_theta = self.loc.theta

            # Rotate slowly to let camera confirm pillars
            _, _, om_s = self.motion._apply_safety_guard(0.0, 0.0, 1.5)
            self.motion._set_motors(0.0, om_s)
            # Finish if full rotation achieved OR timeout safety net
            if self._scan_rotated >= 2*math.pi - 0.2 or t > self.nav_back_end:
                # ── Record completed 360° scan world position ───────────
                self.completed_360_scans.append((self.loc.x, self.loc.y))
                self.last_360_scan_world = (self.loc.x, self.loc.y)
                print(f"SCAN: 360° complete at world ({self.loc.x:.2f}, {self.loc.y:.2f})")
                self.nav_state = 'DRIVE'; self.planner.nav_path = []
                for attr in ['_scan_rotated', '_scan_last_theta', '_scan_start_theta']:
                    if hasattr(self, attr): delattr(self, attr)
            return

        # ── VISUAL LOCK-ON ──────────────────────────────────────────────────
        # A tiny sighting of BLUE or YELLOW was detected.  The robot rotates to
        # face the bearing, then creeps forward until the pillar is confirmed or
        # the timeout expires.  nav_path and current_frontier_target are PRESERVED.
        if self.nav_state == 'VISUAL_LOCK':
            color    = getattr(self, '_vlock_color',    None)
            tgt_yaw  = getattr(self, '_vlock_target_yaw', self.loc.theta)
            deadline = getattr(self, '_vlock_deadline', t)

            # Exit 1: pillar confirmed by main tracker — best outcome
            if color and color in self.perception.cam_confirmed:
                print("VLOCK", f"{color} CONFIRMED during visual lock. Resuming exploration.")
                self.nav_state = 'DRIVE'
                return

            # Exit 2: timeout — set cooldown so we don't immediately re-trigger
            if t > deadline:
                print("VLOCK", f"{color} not confirmed after {VISUAL_LOCK_TIMEOUT:.0f}s. Resuming.")
                self._vlock_cooldown_until = t + VISUAL_LOCK_COOLDOWN
                self.nav_state = 'DRIVE'
                return

            # Update bearing if we still see the pillar (track within frame)
            ts = getattr(self.perception, 'tiny_sighting', None)
            if ts and ts['color'] == color:
                tgt_yaw = self.loc.theta + ts['bearing']
                self._vlock_target_yaw = tgt_yaw

            # Rotate to face bearing; once aligned, hold still to confirm, DO NOT approach
            err = (tgt_yaw - self.loc.theta + math.pi) % (2 * math.pi) - math.pi
            if abs(err) > 0.10:
                # Still turning — rotate only
                om = max(-2.0, min(2.0, err * 3.5))
                _, _, om_safe = self.motion._apply_safety_guard(0.0, 0.0, om)
                self.motion._set_motors(0.0, om_safe)
                if hasattr(self, '_vlock_aligned_start'):
                    delattr(self, '_vlock_aligned_start')
            else:
                # Aligned — hold still to let camera process
                self.motion._set_motors(0.0, 0.0)
                # If we've been aligned for 1.5 seconds without confirmation, abort
                if not hasattr(self, '_vlock_aligned_start'):
                    self._vlock_aligned_start = t
                elif t - self._vlock_aligned_start > 1.5:
                    print("VLOCK", f"Aligned but {color} not confirmed. Resuming path.")
                    self._vlock_cooldown_until = t + VISUAL_LOCK_COOLDOWN
                    delattr(self, '_vlock_aligned_start')
                    self.nav_state = 'DRIVE'
            return

        # ── ACTIVE SENSING FOR FLOATING WALLS ───────────────────────────────
        # If the camera detects wood texture or partial floating wall cues,
        # but no floating wall is yet stamped in front of us, the robot pauses,
        # wiggles, and creeps forward/backward to get clean camera/depth alignments.
        if self.nav_state == 'ACTIVE_SENSING':
            if not hasattr(self, '_as_start_frame'):
                self._as_start_frame = frame
                self._as_original_pose = (self.loc.x, self.loc.y, self.loc.theta)
                print("ACTIVE_SENSING", "Triggered active sensing to localize floating wall!")

            # 1. Check if the wall is now successfully mapped/stamped ahead
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
                self.planner.nav_path = []  # Force replan around the newly mapped wall
                for attr in ['_as_start_frame', '_as_original_pose']:
                    if hasattr(self, attr): delattr(self, attr)
                return

            # 2. Timeout: if we have scanned for 40 ticks without stamp, abort
            elapsed = frame - self._as_start_frame
            if elapsed > 40:
                print("ACTIVE_SENSING", "Active sensing timed out. Resuming.")
                self.nav_state = 'DRIVE'
                self._as_cooldown_until = t + 5.0  # 5-second cooldown
                for attr in ['_as_start_frame', '_as_original_pose']:
                    if hasattr(self, attr): delattr(self, attr)
                return

            # 3. Active sensing choreography: stop, creep, and pivot
            if elapsed <= 8:
                # Phase 1: Stop and look (let image stabilize)
                self.motion._set_motors(0.0, 0.0)
            elif elapsed <= 16:
                # Phase 2: Creep forward slowly
                vx, _, om_s = self.motion._apply_safety_guard(0.07, 0.0, 0.0)
                self.motion._set_motors(vx, om_s)
            elif elapsed <= 24:
                # Phase 3: Creep backward slowly
                vx, _, om_s = self.motion._apply_safety_guard(-0.07, 0.0, 0.0)
                self.motion._set_motors(vx, om_s)
            else:
                # Phase 4: Small yaw wiggle to expand camera coverage
                wiggle_om = 0.6 if (elapsed % 8 < 4) else -0.6
                _, _, om_s = self.motion._apply_safety_guard(0.0, 0.0, wiggle_om)
                self.motion._set_motors(0.0, om_s)
            return

        # ── LIDAR PILLAR CANDIDATE LOOK ──────────────────────────────────────────
        # The Lidar has detected a round object that has not been mapped.
        # We suspend navigation and spin to face it for a short time to let the
        # camera classify it.
        if self.nav_state == 'DRIVE' and self.state in [STATE_EXPLORE, STATE_SEARCH_TARGET, STATE_NAV_BLUE, STATE_NAV_YELLOW]:
            if getattr(self.perception, 'unidentified_lidar_pillars', []):
                # Pop the first candidate
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
            
            # Timeout after 2.5 seconds max
            if elapsed > 2.5:
                print(f"LIDAR_PILLAR: Look timed out. Marking as checked.")
                self.perception.checked_lidar_pillars.append((lx, ly))
                self.nav_state = 'DRIVE'
                return
                
            # If camera confirms a pillar, we can exit early!
            for color, pos in self.perception.cam_confirmed.items():
                if math.hypot(lx - pos[0], ly - pos[1]) < 0.4:
                    print(f"LIDAR_PILLAR: Camera successfully confirmed {color} pillar!")
                    self.nav_state = 'DRIVE'
                    return
            
            # Spin to face the target
            target_yaw = math.atan2(ly - self.loc.y, lx - self.loc.x)
            err = (target_yaw - self.loc.theta + math.pi) % (2 * math.pi) - math.pi
            
            if abs(err) > 0.08:
                # Still spinning
                om = max(-1.5, min(1.5, err * 3.0))
                _, _, om_s = self.motion._apply_safety_guard(0.0, 0.0, om)
                self.motion._set_motors(0.0, om_s)
            else:
                # Aligned, hold still to let camera process
                self.motion._set_motors(0.0, 0.0)
                # If we've been aligned for at least 0.8 seconds without a camera confirmation,
                # it's probably not a colored pillar (just a weird wall/leg).
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

        # Update the visual coverage map every frame in ALL states
        self._update_visual_coverage()

        if self.state == STATE_STARTUP:
            self.motion._set_motors(0.0, 0.0)
            if t >= 5.0: 
                print("STATE", "Startup complete. Entering AUTO-EXPLORE mode.")
                self.state = STATE_EXPLORE

        elif self.state == STATE_EXPLORE:
            # ── v39.0 EXPLORE-FIRST GATE ────────────────────────────────────
            # A* pillar navigation is LOCKED until ALL three conditions pass:
            #   1. BLUE pillar coordinates confirmed (cam_confirmed['BLUE'])
            #   2. YELLOW pillar coordinates confirmed (cam_confirmed['YELLOW'])
            #   3. No unexplored frontiers remain on the map
            # A pillar counts as "located" if EITHER close-range confirmed OR
            # long-range buffered with PILLAR_NAV_MIN_BUF stable readings. The
            # buffer estimate is good enough to know the pillar exists and roughly
            # where it is — close-range confirmation can wait for NAV_BLUE/YELLOW.
            blue_buf   = self.perception.cam_pillar_buf.get('BLUE',   [])
            yellow_buf = self.perception.cam_pillar_buf.get('YELLOW', [])
            blue_confirmed   = ('BLUE'   in self.perception.cam_confirmed) or (len(blue_buf)   >= PILLAR_NAV_MIN_BUF)
            yellow_confirmed = ('YELLOW' in self.perception.cam_confirmed) or (len(yellow_buf) >= PILLAR_NAV_MIN_BUF)

            # Hard timeout: if we've been exploring for too long and at least
            # ONE pillar is still missing from buffers, accept any single-frame
            # sighting as good enough. Prevents infinite exploration when a
            # pillar is in an awkward spot the robot can't reach via frontiers.
            EXPLORE_HARD_TIMEOUT = 120.0  # seconds of simulation time
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

            # Evaluate map completeness (throttled — expensive every 150 frames)
            if not hasattr(self, '_explore_gate_tick'): self._explore_gate_tick = 0
            self._explore_gate_tick += 1
            if self._explore_gate_tick % 150 == 0 or not hasattr(self, '_map_fully_explored'):
                self._map_fully_explored = (len(self.planner._find_frontiers()) == 0)

            if blue_confirmed and yellow_confirmed and not getattr(self, '_blue_visited', False):
                if self.state not in [STATE_NAV_BLUE, STATE_NAV_YELLOW]:
                    if t > getattr(self, '_nav_blue_block_until', 0):
                        print("STATE", f"BOTH PILLARS LOCATED! Transitioning directly to NAV_BLUE.")
                        self.state = STATE_NAV_BLUE
                        self.planner.nav_path = []
                        self.current_frontier_target = None
                        self.recovery.panic_count = 0
                        self.recovery.recovery_level = 0
                        self.recovery.recovery_substate = None
                        if hasattr(self, 'failed_goals'): self.failed_goals.clear()
                        if hasattr(self, '_goal_retry_count'): self._goal_retry_count = 0
                        return

            if blue_confirmed and yellow_confirmed and getattr(self, '_blue_visited', False):
                if self.state not in [STATE_NAV_YELLOW, STATE_DONE]:
                    if t > getattr(self, '_nav_yellow_block_until', 0):
                        print("STATE", f"BLUE VISITED! Transitioning directly to NAV_YELLOW.")
                        self.state = STATE_NAV_YELLOW
                        self.planner.nav_path = []
                        self.current_frontier_target = None
                        self.recovery.panic_count = 0
                        self.recovery.recovery_level = 0
                        self.recovery.recovery_substate = None
                        if hasattr(self, 'failed_goals'): self.failed_goals.clear()
                        if hasattr(self, '_goal_retry_count'): self._goal_retry_count = 0
                        return

            if self._map_fully_explored:
                # Map done but pillars not both found? Keep searching frontiers if any exist.
                # If truly 0 frontiers, then we have no choice but to transition anyway.
                if len(self.planner._find_frontiers()) == 0:
                    print("STATE", "Map fully explored but pillars missing. Switching to SEARCH_TARGET.")
                    self.state = STATE_SEARCH_TARGET
                    self.planner.nav_path = []
                    self.current_frontier_target = None
                    self.failed_goals.clear()
                    return

            # ── GREEN GROUND CONFIRMATION INTERRUPT ──────────────────────────
            # When perception has seen green but hasn't confirmed it yet (robot was
            # moving or spinning), stop and face the estimated green position so
            # the camera gets a clean, static measurement frame.
            if (getattr(self.perception, 'green_pending', False)
                    and 'GREEN' not in self.perception.cam_confirmed):
                gpos = self.perception.green_pending_pos
                if gpos is not None:
                    # Face toward the green estimate and hold still for 1.5 s
                    if not hasattr(self, '_green_confirm_start'):
                        self._green_confirm_start = t
                        self.motion._set_motors(0.0, 0.0)
                        print(f"GREEN-CONFIRM: Stopping to face green at ({gpos[0]:.2f},{gpos[1]:.2f})")

                    elapsed_gc = t - self._green_confirm_start
                    if elapsed_gc < 1.5:
                        # Face the robot toward green estimate
                        angle_to_green = math.atan2(gpos[1] - self.loc.y,
                                                     gpos[0] - self.loc.x)
                        heading_err = ((angle_to_green - self.loc.theta + math.pi) %
                                       (2 * math.pi)) - math.pi
                        if abs(heading_err) > 0.08:
                            turn_dir = 1.0 if heading_err > 0 else -1.0
                            _, _, om_s = self.motion._apply_safety_guard(0.0, 0.0, turn_dir * 1.2)
                            self.motion._set_motors(0.0, om_s)
                        else:
                            self.motion._set_motors(0.0, 0.0)  # hold still, camera accumulates
                        return
                    else:
                        # 1.5 s elapsed — clear interrupt and resume regardless
                        print("GREEN-CONFIRM: Timeout — resuming exploration")
                        self.perception.green_pending = False
                        self.perception.green_pending_pos = None
                        if hasattr(self, '_green_confirm_start'):
                            del self._green_confirm_start
                        self.planner.nav_path = []

            elif hasattr(self, '_green_confirm_start'):
                # Confirmation completed (green_pending cleared by perception on confirm)
                del self._green_confirm_start
                self.planner.nav_path = []

            # ── DFS LIGHTHOUSE EXPLORATION DELEGATION ──
            if getattr(config, 'USE_DFS_LIGHTHOUSE_EXPLORATION', False):
                if not hasattr(self, 'dfs_explorer'):
                    from dfs_lighthouse import DFSLighthouseExploration
                    self.dfs_explorer = DFSLighthouseExploration(
                        self.motion.robot, self.motion.sensors, self.slam, self.planner, self.motion, self.loc, state_machine=self)
                
                res = self.dfs_explorer.run(frame, t)
                if res == 'DONE':
                    self.state = STATE_SEARCH_TARGET
                return

            # ── Gate NOT yet passed: keep exploring the map ─────────────────
            # Log the current gate status periodically so the operator can see
            # which condition(s) are still missing.
            if frame % 300 == 0:
                frontier_count = len(self.planner._find_frontiers())
                print("EXPLORE",
                    f"Gate status: BLUE={'✓' if blue_confirmed else '✗'} "
                    f"YELLOW={'✓' if yellow_confirmed else '✗'} "
                    f"MapDone={'✓' if self._map_fully_explored else f'✗ ({frontier_count} frontiers left)'}")

            # PRIORITY 1: Follow current latched frontier goal
            if self.planner.nav_path:
                self.recovery.panic_count = 0

                # ── Visual lock-on: interrupt navigation for a tiny pillar sighting ──
                # Conditions to fire:
                #   • sighting area BETWEEN min_px and max_px (large sightings handled by main tracker)
                #   • color not yet confirmed
                #   • not in recovery
                #   • cooldown from previous timeout expired
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
                    self.motion._set_motors(0.0, 0.0)   # stop immediately
                    print("VLOCK", f"Spotted {ts['color']} ({ts['area']} px) at "
                          f"{math.degrees(ts['bearing']):+.1f}°. Locking on.")
                    return

                # Intercept normal driving if we pass a new junction
                if self._check_and_trigger_junction_scan(t):
                    return
                    
                if self.motion._follow_path(): 
                    self.planner.nav_path = []
                    self._goal_retry_count = 0
                    
                    # ── Check if we've actually arrived at the frontier target ──
                    # A path segment may end at a waypoint that is NOT the frontier.
                    # Only clear the target when the robot is physically close to it.
                    arrived_at_target = False
                    if self.current_frontier_target:
                        tx, ty = self.current_frontier_target
                        dist_px = math.hypot(tx - (self.slam.origin + int(self.loc.x/MAP_RES)),
                                             ty - (self.slam.origin - int(self.loc.y/MAP_RES)))
                        arrived_at_target = dist_px * MAP_RES < 0.20  # within 20 cm of target

                    # ── DEPTH-FIRST: Try to extend deeper into the corridor ─────
                    # Before picking a new global frontier, follow the current 
                    # corridor to its dead-end (DFS). This avoids backtracking.
                    rmx_dfs = self.slam.origin + int(self.loc.x / MAP_RES)
                    rmy_dfs = self.slam.origin - int(self.loc.y / MAP_RES)
                    ext = self._extend_corridor(rmx_dfs, rmy_dfs)
                    if ext:
                        self.planner.nav_path = ext
                        # Keep same frontier target — we're extending deeper
                        print("EXPLORE_DFS", f"Extending deeper into corridor → ({ext[-1][0]},{ext[-1][1]})")
                        return

                    if arrived_at_target:
                        # Robot is at (or very near) the frontier — true dead-end.
                        # Clear target and do a short look-around scan.
                        print("EXPLORE", f"Arrived at frontier target. Scanning for new frontiers.")
                        self.current_frontier_target = None
                        self.nav_state = 'SCAN'
                        self.nav_back_end = t + 1.2
                        return  # ← prevent fall-through to bottom block
                    elif self.current_frontier_target:
                        # Path segment done but still far from the frontier.
                        # Re-path to the SAME target immediately — no scan needed.
                        tx, ty = self.current_frontier_target
                        new_path = self.planner._astar(rmx_dfs, rmy_dfs, tx, ty)
                        if new_path:
                            self.planner.nav_path = new_path
                            print("EXPLORE", f"Re-pathing to same frontier ({tx},{ty}) after waypoint.")
                        else:
                            # Can't repath — abandon and scan
                            self.current_frontier_target = None
                            self.nav_state = 'SCAN'
                            self.nav_back_end = t + 1.0
                        return
                    else:
                        # No active target — short scan then pick new frontier
                        self.nav_state = 'SCAN'
                        self.nav_back_end = t + 1.0
                        return

            # PRIORITY 2: Select or re-validate frontier goal
            else:
                rmx, rmy = self.slam.origin + int(self.loc.x/MAP_RES), self.slam.origin - int(self.loc.y/MAP_RES)
                
                # Try to maintain the current target if still valid
                if self.current_frontier_target:
                    tx, ty = self.current_frontier_target
                    dist_to_goal = math.hypot(tx - rmx, ty - rmy) * MAP_RES
                    # Only drop the goal when physically close to it (within 0.5m),
                    # NOT merely because we passed nearby during navigation
                    if dist_to_goal < 0.20:
                        print("GOAL", f"ARRIVED: Reached frontier goal ({tx},{ty}), dist={dist_to_goal:.2f}m. Clearing.")
                        self.current_frontier_target = None
                        self._goal_retry_count = 0
                    else:
                        # Repath to the same destination
                        self.planner.nav_path = self.planner._astar(rmx, rmy, tx, ty)
                        if self.planner.nav_path:
                            print("GOAL", f"LATCHED: Re-pathing to ({tx},{ty}), {dist_to_goal:.1f}m away")
                            self._goal_retry_count = 0
                        else:
                            if not hasattr(self, '_goal_retry_count'): self._goal_retry_count = 0
                            self._goal_retry_count += 1
                            if self._goal_retry_count < 5:
                                print("GOAL", f"BLOCKED: Retry {self._goal_retry_count}/5 for goal ({tx},{ty})")
                                _, _, om_s = self.motion._apply_safety_guard(0.0, 0.0, 0.4)
                                self.motion._set_motors(0.0, om_s)
                                return
                            else:
                                print("GOAL", f"ABANDONED: Goal ({tx},{ty}) consistently unreachable.")
                                self.current_frontier_target = None
                                if hasattr(self, 'failed_goals'): self.failed_goals.add((tx, ty))
                                self._goal_retry_count = 0

                # If no active goal, pick a new one
                if not self.planner.nav_path:
                    if not hasattr(self, 'failed_goals'): self.failed_goals = set()
                    # If exactly one pillar is still missing, bias frontier search
                    # toward areas the robot HASN'T scanned in the rough direction
                    # the missing pillar is likely to be (i.e. opposite the seen one).
                    bias_target = None
                    # --- NEW: Pillar Re-Discovery Logic ---
                    # If we are missing a pillar but have a "best guess" in the buffer, 
                    # go there even if it's not a 'frontier'.
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
                            # We visited Blue, now bias frontiers towards Yellow
                            target = self._resolve_yellow_target(min_buf=3)
                            if target:
                                yx, yy = target
                                yellow_mx = self.slam.origin + int(yx / MAP_RES)
                                yellow_my = self.slam.origin - int(yy / MAP_RES)
                                bias_target = (yellow_mx, yellow_my)
                        else:
                            # Bias frontiers towards Blue
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
                            # Reflect through robot's current cell
                            bias_target = (2 * rmx - blue_mx, 2 * rmy - blue_my)
                    elif yellow_confirmed and not blue_confirmed:
                        target = self._resolve_yellow_target(min_buf=3)
                        if target:
                            yx, yy = target
                            yellow_mx = self.slam.origin + int(yx / MAP_RES)
                            yellow_my = self.slam.origin - int(yy / MAP_RES)
                            # Reflect through robot's current cell
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
                            # Cluster blacklist: when one cell fails, blacklist all
                            # frontiers within a small radius to avoid re-selecting the same zone
                            self.failed_goals.add((fx, fy))
                            for ofx, ofy in valid_frontiers:
                                if abs(ofx - fx) <= EXPLORE_CLUSTER_R and abs(ofy - fy) <= EXPLORE_CLUSTER_R:
                                    self.failed_goals.add((ofx, ofy))
                    
                    if not found_path:
                        # Self-healing check: if we have zero reachable frontiers (e.g. they leaked outside walls),
                        # consider the map fully explored and switch to SEARCH_TARGET or NAV_BLUE.
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
                            # Burn a permanent no-go zone around the GREEN obstacle
                            # if the robot keeps getting stuck near it
                            green_pos = self.perception.cam_confirmed.get('GREEN')
                            if green_pos:
                                dist_to_green = math.hypot(self.loc.x - green_pos[0],
                                                           self.loc.y - green_pos[1])
                                if dist_to_green < 1.5:
                                    self._burn_green_exclusion_zone()
                            if not self.recovery._execute_stuck_recovery():
                                return   # recovery still in progress
                            self.recovery.panic_count = 0; self.failed_goals.clear()
                        else:
                            # Standard scan to find new frontiers
                            _, _, om_s = self.motion._apply_safety_guard(0.0, 0.0, 0.8)
                            self.motion._set_motors(0.0, om_s)
                            self.nav_state = 'SCAN'; self.nav_back_end = t + 2.5
            if self.planner.nav_path:
                self.recovery.panic_count = 0
                self.motion._follow_path()   # drive; lifecycle managed by PRIORITY 1 above


        elif self.state == STATE_SEARCH_TARGET:
            # PRIORITY: If both pillars found, start mission immediately
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
                # Navigate to the LARGEST visually uncovered (white but not cyan) patch
                target_wp = self._find_largest_uncovered_viewpoint()
                if target_wp:
                    gmx, gmy = target_wp
                    path = self.planner._astar(rmx, rmy, gmx, gmy)
                    if path:
                        self.planner.nav_path = path
                        print(f"SEARCH_TARGET: Navigating to largest uncovered patch at ({gmx},{gmy})")
                        return

                # Fallback: unvisited junction
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
                    vx, _, om_s = self.motion._apply_safety_guard(0.15, 0.0, 0.3)
                    self.motion._set_motors(vx, om_s)

        elif self.state == STATE_NAV_BLUE:
            self._navigate_to_pillar('BLUE', t, frame)

        elif self.state == STATE_NAV_YELLOW:
            self._navigate_to_yellow_final(t, frame)

        elif self.state == STATE_DONE:
            self.motion._set_motors(0.0, 0.0)
            # Final save on mission completion
            if not getattr(self, '_final_hist_saved', False):
                self.history._record_position()
                self.history._save_coord_history()
                self.slam.save_debug_maps()
                
                t_start = getattr(self, 't_start', 0.0)
                if t_start is None:
                    t_start = 0.0
                t_blue = getattr(self, 'mission_start_time', 0.0)
                if t_blue is None:
                    t_blue = 0.0
                
                timer1 = t_blue - t_start
                timer2 = t - t_blue
                total_time = t - t_start
                
                print("============================================================")
                print("                    MISSION ACCOMPLISHED!                    ")
                print(f"    TIMER 1 (SIMULATION START -> BLUE):    {timer1:.2f}s")
                print(f"    TIMER 2 (BLUE -> YELLOW):              {timer2:.2f}s")
                print(f"    TOTAL TIME (SIMULATION -> YELLOW):   {total_time:.2f}s")
                print("=================================")#
                
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
            # Robust median to filter outliers
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
            # Robust median to filter outliers
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
        
        # Determine the goal coordinates
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
            
        # Try to find a path using only a safe clearance kernel (size >= 7)
        orig_force = getattr(self.planner, 'force_kernel', None)
        # Use a safe kernel (e.g. 7 or greater) to verify that the path is wide and clear
        self.planner.force_kernel = 7
        
        # Block unknown space when validating if we have a safe path to blue
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
        
        # Determine the goal coordinates
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
            
        # Try to find a path using only a safe clearance kernel (size >= 7)
        orig_force = getattr(self.planner, 'force_kernel', None)
        # Use a safe kernel (e.g. 7 or greater) to verify that the path is wide and clear
        self.planner.force_kernel = 7
        
        # Block unknown space when validating if we have a safe path to yellow
        self.planner.block_unknown = True
        path = self.planner._astar(rmx, rmy, gmx, gmy)
        self.planner.block_unknown = False
        
        self.planner.force_kernel = orig_force
        
        return len(path) > 0

    def _resolve_home_target(self):
        # Home is the starting point (0,0)
        return (0.0, 0.0)

    def _has_line_of_sight(self, target_x, target_y):
        """
        Ray-casts on the SLAM log_map to check if a straight line from robot to 
        target is free of confirmed walls.
        """
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

    def _check_and_execute_blue_stuck_recovery(self, t, replan_attr):
        """
        Stuck recovery logic for NAV_BLUE state.
        If the robot is stopped at one position for 15 seconds, takes it backwards very little and then turns.
        """
        if not hasattr(self, '_blue_last_pos'):
            self._blue_last_pos = (self.loc.x, self.loc.y, self.loc.theta)
            self._blue_last_pos_t = t
            self._blue_recovery_state = None
            self._blue_recovery_end_t = 0.0

        # Execute recovery phases if active
        if self._blue_recovery_state is not None:
            if self._blue_recovery_state == 'BACKWARD':
                if t < self._blue_recovery_end_t:
                    self.motion._set_motors(-0.08, 0.0)
                    return True
                else:
                    self._blue_recovery_state = 'ROTATE'
                    self._blue_recovery_end_t = t + 1.5
                    print("BLUE_RECOVERY", "Backward phase complete. Starting rotation phase.")
            
            if self._blue_recovery_state == 'ROTATE':
                if t < self._blue_recovery_end_t:
                    # Spin turn
                    self.motion._set_motors(0.0, 1.2)
                    return True
                else:
                    self._blue_recovery_state = None
                    self._blue_last_pos = (self.loc.x, self.loc.y, self.loc.theta)
                    self._blue_last_pos_t = t
                    print("BLUE_RECOVERY", "Rotation phase complete. Resuming normal A* navigation.")
                    self.planner.nav_path = []
                    setattr(self, replan_attr, -9999)
                    return False

        # Update and check position/stuck timer every 1 second
        if t - self._blue_last_pos_t >= 1.0:
            dx = self.loc.x - self._blue_last_pos[0]
            dy = self.loc.y - self._blue_last_pos[1]
            dtheta = abs(self.loc.theta - self._blue_last_pos[2])
            if dtheta > math.pi:
                dtheta = abs(dtheta - 2 * math.pi)
            
            # If robot hasn't moved (>2cm) or rotated (>0.05 rad), it's stationary
            if math.hypot(dx, dy) < 0.02 and dtheta < 0.05:
                if t - self._blue_last_pos_t >= 15.0:
                    print("BLUE_RECOVERY", "Robot stuck at the same position for 15 seconds! Activating backward and rotation recovery.")
                    self._blue_recovery_state = 'BACKWARD'
                    self._blue_recovery_end_t = t + 1.5  # backup for 1.5 seconds
                    self.motion._set_motors(-0.08, 0.0)
                    return True
            else:
                self._blue_last_pos = (self.loc.x, self.loc.y, self.loc.theta)
                self._blue_last_pos_t = t

        return False


    def _navigate_to_blue_maze1_final(self, t, frame):
        """Maze1-final NAV_BLUE logic copied as a BLUE-only path."""
        color = 'BLUE'
        replan_attr = '_nav_blue_last_replan'
        strikes_attr = '_nav_blue_bfs_strikes'
        if not hasattr(self, replan_attr): setattr(self, replan_attr, -9999)
        if not hasattr(self, strikes_attr): setattr(self, strikes_attr, 0)

        # Trigger stuck check and recovery
        if self._check_and_execute_blue_stuck_recovery(t, replan_attr):
            return

        target = self._resolve_blue_target(min_buf=5)
        if target is None:
            self.motion._set_motors(0.08, 0.4)
            return

        tx, ty = target
        dist = math.hypot(self.loc.x - tx, self.loc.y - ty)
        has_los = (dist < VISUAL_RANGE and self._has_line_of_sight(tx, ty))

        # Physical contact check
        f_dist, _, _, _ = self.motion.sensors.get_distances_at_corners()
        touch_window = PILLAR_ARRIVE_DIST + 0.20
        touch_near = getattr(config, 'BLUE_TOUCH_SENSOR_DIST', 0.05) + 0.03
        if dist <= touch_window and f_dist <= touch_near:
            print("NAV_BLUE", f"front-contact arrival candidate: f_dist={f_dist:.3f}m, dist={dist:.3f}m")
            self._handle_blue_touch_confirm(tx, ty, t)
            return

        if dist < PILLAR_ARRIVE_DIST + 0.05:
            self._handle_blue_touch_confirm(tx, ty, t)
            return

        is_stalled = (self.recovery.panic_count >= 1) or (getattr(self.motion, '_fp_stall_ticks', 0) >= 15) or (self.recovery.stuck_counter >= 1)
        if dist < PILLAR_ARRIVE_DIST + 0.15 and is_stalled and has_los:
            print("NAV_BLUE", f"LOS-arrival: dist={dist:.2f}m, stalled — accepting as reached")
            self._handle_blue_touch_confirm(tx, ty, t)
            return

        if (dist < CLOSE_APPROACH_DIST or (has_los and is_stalled)) and t > getattr(self, 'los_block_time', 0):
            result = self.motion._face_and_approach(tx, ty, stop_dist=CLOSE_CONFIRM_DIST)
            if result == 'ARRIVED':
                self._handle_blue_touch_confirm(tx, ty, t)
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
            print("NAV_BLUE: BFS-loop trap — forcing direct approach")
            result = self.motion._face_and_approach(tx, ty, stop_dist=CLOSE_CONFIRM_DIST)
            if result == 'ARRIVED':
                self._handle_pillar_arrival('BLUE', t)
                setattr(self, strikes_attr, 0)
            return

        rmx, rmy = self.slam.origin + int(self.loc.x/MAP_RES), self.slam.origin - int(self.loc.y/MAP_RES)
        REPLAN_FREQ = 60

        # Stuck check for 15 seconds:
        # Keep track of last position and orientation to see if the robot is "at same place without moving or adjusting itself"
        if not hasattr(self, '_blue_last_pos'):
            self._blue_last_pos = (self.loc.x, self.loc.y, self.loc.theta)
            self._blue_last_pos_t = t
            self._blue_stuck_active = False
            self._blue_temp_blocked = None

        # Check displacement over 1 second interval
        force_different_path = False
        if t - self._blue_last_pos_t >= 1.0:
            dx = self.loc.x - self._blue_last_pos[0]
            dy = self.loc.y - self._blue_last_pos[1]
            dtheta = abs(self.loc.theta - self._blue_last_pos[2])
            if dtheta > math.pi:
                dtheta = abs(dtheta - 2 * math.pi)
            
            # If displacement < 0.02m and orientation change < 0.05 rad, the robot is stationary
            if math.hypot(dx, dy) < 0.02 and dtheta < 0.05:
                # Still stuck. If the total accumulated time is >= 15 seconds, trigger detour
                if t - self._blue_last_pos_t >= 15.0:
                    print("NAV_BLUE", "Robot is stuck at the same place for 15 seconds! Requesting a different path near the robot.")
                    force_different_path = True
                    # Reset the timer
                    self._blue_last_pos = (self.loc.x, self.loc.y, self.loc.theta)
                    self._blue_last_pos_t = t
            else:
                # Robot is moving or adjusting/turning
                self._blue_last_pos = (self.loc.x, self.loc.y, self.loc.theta)
                self._blue_last_pos_t = t

        _escape_active = getattr(self, '_recovery_escape_active', False)
        if _escape_active:
            escape_origin = getattr(self, '_recovery_escape_origin', (self.loc.x, self.loc.y))
            dist_escaped = math.hypot(self.loc.x - escape_origin[0], self.loc.y - escape_origin[1])
            if dist_escaped > 0.20 or not self.planner.nav_path:
                self._recovery_escape_active = False
                _escape_active = False

        # Determine if we should replan
        should_replan = not _escape_active and (not self.planner.nav_path or force_different_path or (frame - getattr(self, replan_attr) >= REPLAN_FREQ))

        if should_replan:
            temp_blocked = None
            if force_different_path and self.planner.nav_path:
                # Find the first few path cells that are near the robot (within 0.4 meters) and temporarily block them
                temp_blocked = set()
                for cell in self.planner.nav_path[:12]:
                    cx, cy = (cell[0] - self.slam.origin) * MAP_RES, (self.slam.origin - cell[1]) * MAP_RES
                    if math.hypot(self.loc.x - cx, self.loc.y - cy) < 0.40:
                        # Block this cell so A* is forced to find a different path starting near the robot
                        temp_blocked.add(cell)

            new_path = self.planner._astar(rmx, rmy, gmx, gmy, temp_blocked=temp_blocked)
            used_surrogate = False
            if not new_path:
                bfs_t = self._bfs_nearest_free(gmx, gmy)
                if bfs_t:
                    new_path = self.planner._astar(rmx, rmy, bfs_t[0], bfs_t[1], temp_blocked=temp_blocked)
                    if new_path: used_surrogate = True
            if not new_path:
                self.planner.force_kernel = 0
                new_path = self.planner._astar(rmx, rmy, gmx, gmy, temp_blocked=temp_blocked)
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
                    print("NAV_BLUE", "Cannot plan path to pillar (no route). Reverting to EXPLORE phase.")
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
        loop_attr    = '_nav_blue_replan_loop_count'
        grace_attr   = '_nav_blue_grace_remaining'
        breaker_attr = '_nav_blue_breaker_fires'
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
                            print("NAV_BLUE", f"REPLAN-LOOP BREAKER escalated (fire {fires}) "
                                  f"— calling stuck recovery directly")
                            setattr(self, breaker_attr, 0)
                            setattr(self, grace_attr, 0)
                            self.recovery._execute_stuck_recovery()
                        else:
                            print("NAV_BLUE", f"REPLAN-LOOP BREAKER: {loop_count} consecutive "
                                  f"invalidations — granting {NAV_GRACE_TICKS}-tick grace "
                                  f"(fire {fires}/{NAV_LOOP_BREAKER_ESCALATE})")
                            setattr(self, grace_attr, NAV_GRACE_TICKS)
                else:
                    setattr(self, loop_attr, 0)
            else:
                setattr(self, loop_attr, 0)
                setattr(self, breaker_attr, 0)

    def _navigate_to_yellow_final(self, t, frame):
        """Maze2-final NAV_YELLOW logic with stuck check and path detour."""
        color = 'YELLOW'
        replan_attr = '_nav_yellow_last_replan'
        strikes_attr = '_nav_yellow_bfs_strikes'
        if not hasattr(self, replan_attr): setattr(self, replan_attr, -9999)
        if not hasattr(self, strikes_attr): setattr(self, strikes_attr, 0)

        target = self._resolve_yellow_target(min_buf=5)
        if target is None:
            if getattr(config, 'USE_DFS_LIGHTHOUSE_EXPLORATION', False):
                print("STATE", "Yellow pillar missing! Reverting to DFS Lighthouse Exploration.")
                self.state = STATE_EXPLORE; self.planner.nav_path = []
            else:
                self.motion._set_motors(0.08, 0.6)
            return

        tx, ty = target
        dist = math.hypot(self.loc.x - tx, self.loc.y - ty)
        has_los = (dist < VISUAL_RANGE and self._has_line_of_sight(tx, ty))

        # Physical contact check
        f_dist, _, _, _ = self.motion.sensors.get_distances_at_corners()
        touch_window = PILLAR_ARRIVE_DIST + 0.20
        touch_near = getattr(config, 'BLUE_TOUCH_SENSOR_DIST', 0.05) + 0.03
        if dist <= touch_window and f_dist <= touch_near:
            print("NAV_YELLOW", f"front-contact arrival candidate: f_dist={f_dist:.3f}m, dist={dist:.3f}m")
            self._handle_pillar_arrival(color, t)
            return

        if dist < PILLAR_ARRIVE_DIST + 0.05:
            self._handle_pillar_arrival(color, t)
            return

        is_stalled = (self.recovery.panic_count >= 1) or (getattr(self.motion, '_fp_stall_ticks', 0) >= 15) or (self.recovery.stuck_counter >= 1)
        if dist < PILLAR_ARRIVE_DIST + 0.15 and is_stalled and has_los:
            print("NAV_YELLOW", f"LOS-arrival: dist={dist:.2f}m, stalled — accepting as reached")
            self._handle_pillar_arrival(color, t)
            return

        if (dist < CLOSE_APPROACH_DIST or (has_los and is_stalled)) and t > getattr(self, 'los_block_time', 0):
            result = self.motion._face_and_approach(tx, ty, stop_dist=CLOSE_CONFIRM_DIST)
            if result == 'ARRIVED':
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
            print("NAV_YELLOW: BFS-loop trap — forcing direct approach")
            result = self.motion._face_and_approach(tx, ty, stop_dist=CLOSE_CONFIRM_DIST)
            if result == 'ARRIVED':
                self._handle_pillar_arrival('YELLOW', t)
                setattr(self, strikes_attr, 0)
            return

        rmx, rmy = self.slam.origin + int(self.loc.x/MAP_RES), self.slam.origin - int(self.loc.y/MAP_RES)
        REPLAN_FREQ = 40

        # Stuck check for 15 seconds:
        # Keep track of last position and orientation to see if the robot is "at same place without moving or adjusting itself"
        if not hasattr(self, '_yellow_last_pos'):
            self._yellow_last_pos = (self.loc.x, self.loc.y, self.loc.theta)
            self._yellow_last_pos_t = t
            self._yellow_stuck_active = False
            self._yellow_temp_blocked = None

        # Check displacement over 1 second interval
        force_different_path = False
        if t - self._yellow_last_pos_t >= 1.0:
            dx = self.loc.x - self._yellow_last_pos[0]
            dy = self.loc.y - self._yellow_last_pos[1]
            dtheta = abs(self.loc.theta - self._yellow_last_pos[2])
            if dtheta > math.pi:
                dtheta = abs(dtheta - 2 * math.pi)
            
            # If displacement < 0.02m and orientation change < 0.05 rad, the robot is stationary
            if math.hypot(dx, dy) < 0.02 and dtheta < 0.05:
                # Still stuck. If the total accumulated time is >= 15 seconds, trigger detour
                if t - self._yellow_last_pos_t >= 15.0:
                    print("NAV_YELLOW", "Robot is stuck at the same place for 15 seconds! Requesting a different path near the robot.")
                    force_different_path = True
                    # Reset the timer
                    self._yellow_last_pos = (self.loc.x, self.loc.y, self.loc.theta)
                    self._yellow_last_pos_t = t
            else:
                # Robot is moving or adjusting/turning
                self._yellow_last_pos = (self.loc.x, self.loc.y, self.loc.theta)
                self._yellow_last_pos_t = t

        _escape_active = getattr(self, '_recovery_escape_active', False)
        if _escape_active:
            escape_origin = getattr(self, '_recovery_escape_origin', (self.loc.x, self.loc.y))
            dist_escaped = math.hypot(self.loc.x - escape_origin[0], self.loc.y - escape_origin[1])
            if dist_escaped > 0.20 or not self.planner.nav_path:
                self._recovery_escape_active = False
                _escape_active = False

        # Determine if we should replan
        should_replan = not _escape_active and (not self.planner.nav_path or force_different_path or (frame - getattr(self, replan_attr) >= REPLAN_FREQ))

        if should_replan:
            temp_blocked = None
            if force_different_path and self.planner.nav_path:
                # Find the first few path cells that are near the robot (within 0.4 meters) and temporarily block them
                temp_blocked = set()
                for cell in self.planner.nav_path[:12]:
                    cx, cy = (cell[0] - self.slam.origin) * MAP_RES, (self.slam.origin - cell[1]) * MAP_RES
                    if math.hypot(self.loc.x - cx, self.loc.y - cy) < 0.40:
                        # Block this cell so A* is forced to find a different path starting near the robot
                        temp_blocked.add(cell)

            new_path = self.planner._astar(rmx, rmy, gmx, gmy, temp_blocked=temp_blocked)
            used_surrogate = False
            if not new_path:
                bfs_t = self._bfs_nearest_free(gmx, gmy)
                if bfs_t:
                    new_path = self.planner._astar(rmx, rmy, bfs_t[0], bfs_t[1], temp_blocked=temp_blocked)
                    if new_path: used_surrogate = True
            if not new_path:
                self.planner.force_kernel = 0
                new_path = self.planner._astar(rmx, rmy, gmx, gmy, temp_blocked=temp_blocked)
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
                    print("NAV_YELLOW", "Cannot plan path to pillar (no route). Reverting to EXPLORE phase.")
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
        loop_attr    = '_nav_yellow_replan_loop_count'
        grace_attr   = '_nav_yellow_grace_remaining'
        breaker_attr = '_nav_yellow_breaker_fires'
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
                            print("NAV_YELLOW", f"REPLAN-LOOP BREAKER escalated (fire {fires}) — calling stuck recovery directly")
                            setattr(self, breaker_attr, 0)
                            setattr(self, grace_attr, 0)
                            self.recovery._execute_stuck_recovery()
                        else:
                            print("NAV_YELLOW", f"REPLAN-LOOP BREAKER: {loop_count} consecutive invalidations — granting {NAV_GRACE_TICKS}-tick grace")
                            setattr(self, grace_attr, NAV_GRACE_TICKS)
                else:
                    setattr(self, loop_attr, 0)
            else:
                setattr(self, loop_attr, 0)
                setattr(self, breaker_attr, 0)


    def _navigate_to_pillar(self, color, t, frame):
        """
        Unified navigation logic for BLUE and YELLOW pillars.
        Handles re-planning, BFS-fallback, LOS-arrival, and stuck recovery.
        """
        if color == 'BLUE':
            self._navigate_to_blue_maze1_final(t, frame)
            return

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
                vx, _, om_s = self.motion._apply_safety_guard(0.08, 0.0, 0.4 if color == 'BLUE' else 0.6)
                self.motion._set_motors(vx, om_s)
            return

        tx, ty = target
        dist = math.hypot(self.loc.x - tx, self.loc.y - ty)
        has_los = (dist < VISUAL_RANGE and self._has_line_of_sight(tx, ty))

        # Physical contact can happen before the center-distance estimate reaches PILLAR_ARRIVE_DIST.
        # Treat a close front hit near the confirmed target as arrival.
        f_dist, _, _, _ = self.motion.sensors.get_distances_at_corners()
        touch_window = PILLAR_ARRIVE_DIST + 0.20
        touch_near = getattr(config, 'BLUE_TOUCH_SENSOR_DIST', 0.05) + 0.03
        if dist <= touch_window and f_dist <= touch_near:
            print(f"NAV_{color}", f"front-contact arrival candidate: f_dist={f_dist:.3f}m, dist={dist:.3f}m")
            if color == 'BLUE':
                self._handle_blue_touch_confirm(tx, ty, t)
            else:
                self._handle_pillar_arrival(color, t)
            return

        # --- ARRIVAL CHECK 1: Close enough to center ---
        if dist < PILLAR_ARRIVE_DIST + 0.05:
            if color == 'BLUE':
                self._handle_blue_touch_confirm(tx, ty, t)
            else:
                self._handle_pillar_arrival(color, t)
            return

        # --- ARRIVAL CHECK 2: Close but stalled with LOS ---
        is_stalled = (self.recovery.panic_count >= 1) or (getattr(self.motion, '_fp_stall_ticks', 0) >= 15) or (self.recovery.stuck_counter >= 1)
        if dist < PILLAR_ARRIVE_DIST + 0.15 and is_stalled and has_los:
            print(f"NAV_{color}", f"LOS-arrival: dist={dist:.2f}m, stalled — accepting as reached")
            if color == 'BLUE':
                self._handle_blue_touch_confirm(tx, ty, t)
            else:
                self._handle_pillar_arrival(color, t)
            return

        # --- FINAL APPROACH: Face-and-approach if very close or stalled with LOS ---
        if (dist < CLOSE_APPROACH_DIST or (has_los and is_stalled)) and t > getattr(self, 'los_block_time', 0):
            result = self.motion._face_and_approach(tx, ty, stop_dist=CLOSE_CONFIRM_DIST)
            if result == 'ARRIVED':
                if color == 'BLUE':
                    self._handle_blue_touch_confirm(tx, ty, t)
                else:
                    self._handle_pillar_arrival(color, t)
            elif result == 'BLOCKED':
                self.los_block_time = t + 3.0
            return
        # --- TARGETING & REPLANNING ---
        # POLICY: When we don't have LOS, we must NOT target the pillar center because it
        # is inside the pillar/wall boundary. Instead, use BFS to snap the goal to the
        # nearest truly free map cell close to the pillar — so A* always has a reachable goal.
        # When we DO have LOS, use the offset approach point for a clean final dock.
        if has_los:
            ax, ay = self._pillar_approach_point((tx, ty))
            gmx, gmy = self.slam.origin + int(ax/MAP_RES), self.slam.origin - int(ay/MAP_RES)
        else:
            # No LOS: snap the goal to the nearest free cell next to the pillar
            raw_gmx = self.slam.origin + int(tx/MAP_RES)
            raw_gmy = self.slam.origin - int(ty/MAP_RES)
            if (0 <= raw_gmx < MAP_SIZE and 0 <= raw_gmy < MAP_SIZE
                    and self.slam.log_map[raw_gmy, raw_gmx] < FREE_THRESH):
                # Pillar center is in a free cell — use it directly
                gmx, gmy = raw_gmx, raw_gmy
            else:
                # Pillar center is blocked — BFS to nearest truly free cell
                snapped = self._bfs_nearest_free(raw_gmx, raw_gmy, max_radius=30)
                gmx, gmy = snapped if snapped else (raw_gmx, raw_gmy)

        # BFS-Loop Trap Detection
        strikes = getattr(self, strikes_attr)
        if strikes >= BFS_STRIKE_THRESHOLD and dist < VISUAL_RANGE and has_los:
            print(f"NAV_{color}: BFS-loop trap — forcing direct approach")
            result = self.motion._face_and_approach(tx, ty, stop_dist=CLOSE_CONFIRM_DIST)
            if result == 'ARRIVED':
                self._handle_pillar_arrival(color, t)
                setattr(self, strikes_attr, 0)
            return

        rmx, rmy = self.slam.origin + int(self.loc.x/MAP_RES), self.slam.origin - int(self.loc.y/MAP_RES)

        # Base replan frequency
        REPLAN_FREQ = 60 if color == 'BLUE' else 40
        
        # Performance fix: If we already have a long healthy path, don't lag the simulation by constantly replanning!
        if self.planner.nav_path and len(self.planner.nav_path) > 30 and self.recovery.panic_count == 0:
            REPLAN_FREQ = 250  # Only recalculate every ~8 seconds if we are on a long, healthy path

        # ── ESCAPE PATH GUARD (Bug 3 fix) ────────────────────────────────────
        # After L4 recovery sets a frontier escape path, _recovery_escape_active is True.
        # Skip pillar-goal replanning until the robot has moved >20 cm from the stuck
        # position OR the escape path has been consumed, so the L4 path can't be
        # immediately overwritten by a new pillar A* path.
        _escape_active = getattr(self, '_recovery_escape_active', False)
        if _escape_active:
            escape_origin = getattr(self, '_recovery_escape_origin', (self.loc.x, self.loc.y))
            dist_escaped = math.hypot(self.loc.x - escape_origin[0], self.loc.y - escape_origin[1])
            if dist_escaped > 0.20 or not self.planner.nav_path:
                self._recovery_escape_active = False
                _escape_active = False

        if not _escape_active and (not self.planner.nav_path or (frame - getattr(self, replan_attr) >= REPLAN_FREQ)):
            new_path = self.planner._astar(rmx, rmy, gmx, gmy, truncate=False)
            used_surrogate = False
            if not new_path:
                bfs_t = self._bfs_nearest_free(gmx, gmy)
                if bfs_t:
                    new_path = self.planner._astar(rmx, rmy, bfs_t[0], bfs_t[1], truncate=False)
                    if new_path: used_surrogate = True
            if not new_path:
                self.planner.force_kernel = 0
                new_path = self.planner._astar(rmx, rmy, gmx, gmy, truncate=False)
                self.planner.force_kernel = None
                if new_path: used_surrogate = True

            if new_path:
                self.planner.nav_path = new_path
                setattr(self, replan_attr, frame)
                # Bug 4 fix (part 2): reset path-follower stall state on each fresh path
                # so stale ticks from the previous path can't fire recovery immediately.
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

                # If we cannot plan a path to the pillar (even after falling back to kernel=0),
                # check if there are frontiers left. If so, revert to EXPLORE phase to find a path.
                frontiers = self.planner._find_frontiers()
                if len(frontiers) > 0 and self.recovery.panic_count >= 3:
                    print(f"NAV_{color}", "Cannot plan path to pillar (no route). Reverting to EXPLORE phase for 10 seconds.")
                    self.state = STATE_EXPLORE
                    if color == 'YELLOW':
                        self._nav_yellow_block_until = t + 10.0
                    else:
                        self._nav_blue_block_until = t + 10.0
                    self.planner.nav_path = []
                    self.current_frontier_target = None
                    self.recovery.panic_count = 0
                    return

                if self.recovery.panic_count >= 2:
                    if not self.recovery._execute_stuck_recovery(): return
                    self.recovery.panic_count = 0
                else:
                    self.nav_state = 'SCAN'; self.nav_back_end = t + 2.0

        # ── REPLAN-LOOP BREAKER (Bug 1 fix) ──────────────────────────────────
        # Tracks consecutive ticks where _follow_path() immediately invalidates the path.
        # After NAV_REPLAN_LOOP_LIMIT consecutive invalidations a grace period is granted.
        # After NAV_LOOP_BREAKER_ESCALATE consecutive grace periods without escape,
        # escalate to a real _execute_stuck_recovery() call instead.
        import config as _smcfg
        NAV_REPLAN_LOOP_LIMIT    = 5
        NAV_GRACE_TICKS          = 60
        NAV_LOOP_BREAKER_ESCALATE = getattr(_smcfg, 'NAV_LOOP_BREAKER_ESCALATE', 3)
        loop_attr    = f'_nav_{color.lower()}_replan_loop_count'
        grace_attr   = f'_nav_{color.lower()}_grace_remaining'
        breaker_attr = f'_nav_{color.lower()}_breaker_fires'   # counts grace grants without escape
        if not hasattr(self, loop_attr):    setattr(self, loop_attr, 0)
        if not hasattr(self, grace_attr):   setattr(self, grace_attr, 0)
        if not hasattr(self, breaker_attr): setattr(self, breaker_attr, 0)

        grace_rem = getattr(self, grace_attr)
        if grace_rem > 0:
            # In grace period — force validity to pass
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
                            # Escalate: enough grace periods wasted — force a real recovery
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
                # Robot actually drove — reset both loop counter and escalation counter
                setattr(self, loop_attr, 0)
                setattr(self, breaker_attr, 0)


    def _handle_blue_touch_confirm(self, tx, ty, t):
        """
        Touch-confirmation sub-state for the BLUE pillar.
        """
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
        """
        Returns a world point PILLAR_ARRIVE_DIST from the pillar centre that:
          1. Is in a free map cell (not inside a wall).
          2. Is on the robot-facing side (angular sector within ±90° of the robot-to-pillar vector).
        Searches a fan of 24 candidate directions and picks the best free cell
        nearest to the robot. Falls back to the naive radial point if nothing is found.
        """
        px, py = pillar_world_xy
        rx, ry = self.loc.x, self.loc.y
        dx, dy = rx - px, ry - py
        d = math.hypot(dx, dy)
        if d < 1e-3:
            return (px, py)

        robot_angle = math.atan2(dy, dx)  # angle from pillar toward robot
        step_m = 0.02
        best_point = None
        best_score = float('inf')

        # Search a full circle of 24 candidate directions in rings of increasing radius
        for i in range(24):
            angle = robot_angle + (i - 12) * (math.pi / 12)  # ±180° around robot direction
            for ring in range(20):  # search from PILLAR_ARRIVE_DIST outward up to +40cm
                target_dist = PILLAR_ARRIVE_DIST + ring * step_m
                cx = px + math.cos(angle) * target_dist
                cy = py + math.sin(angle) * target_dist
                mx = self.slam.origin + int(cx / MAP_RES)
                my = self.slam.origin - int(cy / MAP_RES)
                if not (0 <= mx < MAP_SIZE and 0 <= my < MAP_SIZE):
                    break
                if self.slam.log_map[my, mx] < FREE_THRESH:
                    # Score: how far this candidate is from the robot (prefer nearer)
                    score = math.hypot(rx - cx, ry - cy)
                    # Favor directions facing the robot (dot product > 0)
                    dot = math.cos(angle) * (-dx/d) + math.sin(angle) * (-dy/d)
                    score -= dot * 0.5  # small bias toward robot-facing side
                    if score < best_score:
                        best_score = score
                        best_point = (cx, cy)
                    break  # found free cell in this direction, no need to go further out

        if best_point is not None:
            return best_point

        # Ultimate fallback: naive radial offset
        ux, uy = dx / d, dy / d
        return (px + ux * PILLAR_ARRIVE_DIST, py + uy * PILLAR_ARRIVE_DIST)

    def _bfs_nearest_free(self, start_mx, start_my, max_radius=80):
        """
        BFS expanding outward using a Priority Queue (heapq) to favor cells 
        closer to the robot's current position, preventing unreachable goals.
        """
        rx, ry = self.slam.origin + int(self.loc.x / MAP_RES), self.slam.origin - int(self.loc.y / MAP_RES)
        
        # pq stores (dist_to_robot, mx, my)
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
                        # Wall jump logic for escaping pillar inflation
                        if self.slam.log_map[ny, nx] > WALL_THRESH:
                            # Use squared Euclidean distance for circle check (Bug 3 fix)
                            if (nx - start_mx)**2 + (ny - start_my)**2 <= 4: pass
                            else: continue
                        visited.add((nx, ny))
                        # Bias priority by distance to robot
                        dist_r = (nx - rx)**2 + (ny - ry)**2
                        heapq.heappush(pq, (dist_r, nx, ny))
        return None

    def _find_largest_uncovered_viewpoint(self):
        """
        Finds the centroid of the largest free-space patch that the camera
        has NOT yet looked at (white on SLAM map, but not cyan on coverage map).
        Returns the closest routable (mx, my) cell near that centroid.
        """
        if not hasattr(self, 'visual_coverage_map'):
            return None

        # Build uncovered mask: free space that camera hasn't seen
        free_mask   = (self.slam.log_map < FREE_THRESH).astype(np.uint8)
        unseen_mask = (free_mask & ~self.visual_coverage_map.astype(np.uint8))

        if unseen_mask.sum() == 0:
            return None

        # Find connected regions of unseen free space
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            unseen_mask, connectivity=8)

        # Skip label 0 (background). Find the largest patch.
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

        # Snap centroid to the nearest passable cell using pre-sorted radial offsets
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
        """
        Permanently marks a no-go zone around the confirmed GREEN obstacle
        in the SLAM log_map so the A* planner never routes through it again.
        Called once after recovery loops near GREEN are detected.
        """
        if 'GREEN' not in self.perception.cam_confirmed:
            return
        rx, ry, _ = self.perception.cam_confirmed['GREEN']
        # Burn a tight 0.18m radius hard obstacle into the log_map.
        # Previously 0.40m which was so large it swallowed nearby pillars!
        EXCL_R_M = 0.18
        excl_px = int(EXCL_R_M / MAP_RES)
        cx = self.slam.origin + int(rx / MAP_RES)
        cy = self.slam.origin - int(ry / MAP_RES)
        for dy in range(-excl_px, excl_px + 1):
            for dx in range(-excl_px, excl_px + 1):
                if dx*dx + dy*dy <= excl_px*excl_px:
                    px, py = cx + dx, cy + dy
                    if 0 <= px < MAP_SIZE and 0 <= py < MAP_SIZE:
                        # Lock the cell as a confirmed permanent wall
                        self.slam.log_map[py, px] = max(
                            self.slam.log_map[py, px], WALL_THRESH + 1.0)
        print("GREEN_EXCL", f"Permanent exclusion zone burned at ({rx:.2f},{ry:.2f}), radius={EXCL_R_M}m")

    def _update_visual_coverage(self, cam_fov_rad=1.04, max_depth_m=2.5):
        """
        Throttled visual coverage update using vectorized ray endpoints.
        """
        if not hasattr(self, 'visual_coverage_map'):
            self.visual_coverage_map = np.zeros_like(self.slam.log_map, dtype=bool)

        # Performance: Only update every 3rd frame
        if not hasattr(self, '_vis_cov_tick'): self._vis_cov_tick = 0
        self._vis_cov_tick += 1
        if self._vis_cov_tick % 3 != 0: return

        rx, ry, rtheta = self.loc.x, self.loc.y, self.loc.theta
        num_rays = 15 # Reduced from 20 for speed
        
        # Vectorized angles
        angles = rtheta - (cam_fov_rad / 2) + np.linspace(0, cam_fov_rad, num_rays)
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        
        # Ray casting
        for i in range(num_rays):
            for d in np.arange(0.10, max_depth_m, MAP_RES):
                wx = rx + d * cos_a[i]
                wy = ry + d * sin_a[i]
                mx = self.slam.origin + int(wx / MAP_RES)
                my = self.slam.origin - int(wy / MAP_RES)
                
                if not (0 <= mx < MAP_SIZE and 0 <= my < MAP_SIZE): break
                self.visual_coverage_map[my, mx] = True
                if self.slam.log_map[my, mx] >= 0.0: break # Hit wall or unknown

    def _check_and_trigger_junction_scan(self, t):
        # Junction scans can be globally disabled via config.ENABLE_JUNCTION_SCAN.
        # When disabled the robot keeps driving without interrupting for a 360° sweep.
        import config as _cfg
        if not getattr(_cfg, 'ENABLE_JUNCTION_SCAN', False):
            return False

        if not hasattr(self, 'scanned_junctions'):
            self.scanned_junctions = set()
        if not hasattr(self, 'persistent_junctions'):
            self.persistent_junctions = [] # List of (world_x, world_y)

        if hasattr(self.slam, 'junctions'):
            # 1. Update persistent junctions (Latch new ones, don't move old ones)
            for jy, jx in self.slam.junctions:
                # Convert pixel to world
                wx = (jx - self.slam.origin) * MAP_RES
                wy = (self.slam.origin - jy) * MAP_RES
                
                # Check if this matches any existing persistent junction
                idx = -1
                for i, (px, py) in enumerate(self.persistent_junctions):
                    if math.hypot(wx - px, wy - py) < 0.60: # 60cm cluster
                        idx = i; break
                
                if idx == -1:
                    idx = len(self.persistent_junctions)
                    self.persistent_junctions.append((wx, wy))
                
                # AUTO-MARK: If junction is on a wall or near GREEN ground, mark as scanned
                if idx not in self.scanned_junctions:
                    # Check SLAM map for walls
                    if self.slam.log_map[jy, jx] >= WALL_THRESH - 1.0:
                        print(f"Junction {idx} is on a wall. Marking as visited.")
                        self.scanned_junctions.add(idx)
                    # Check for Green ground nearby
                    green_pos = self.perception.cam_confirmed.get('GREEN')
                    if green_pos:
                        if math.hypot(wx - green_pos[0], wy - green_pos[1]) < 0.40:
                            print(f"Junction {idx} is near GREEN obstacle. Marking as visited.")
                            self.scanned_junctions.add(idx)

            # 2. Check if we are near an UNSCANNED persistent junction
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
        """
        Depth-First Corridor Extension.
        Walks forward from (rmx, rmy) following the unvisited free space in the
        direction the robot is already heading. Stops when:
          - A dead-end is reached (no unvisited free neighbors) → return None
          - A junction is reached (≥3 free neighbors) → return path to deepest cell
          - max_steps is exhausted

        Returns an A* path to the furthest reachable unvisited cell,
        or None if the robot is already at a dead-end / fully explored locally.
        """
        # 8-connected neighbors
        DIRS = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

        cx, cy = rmx, rmy
        # Sort directions by alignment with current robot heading (prefer forward)
        rtheta = self.loc.theta
        fwd_x = math.cos(rtheta)
        fwd_y = -math.sin(rtheta)  # map Y is inverted

        def heading_align(d):
            return -(d[0]*fwd_x + d[1]*fwd_y)  # negative → sorted ascending (best first)

        visited_local = set()
        visited_local.add((cx, cy))
        best_target = None

        for _ in range(max_steps):
            # Find all free unvisited neighbors
            free_neighbors = []
            for dx, dy in sorted(DIRS, key=heading_align):
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in visited_local:
                    continue
                if not (0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE):
                    continue
                # Must be confirmed free (not wall, not unknown)
                if self.slam.log_map[ny, nx] >= 0.0:
                    continue
                # Skip heavily visited cells (robot has been there many times)
                if self.slam.visit_count[ny, nx] > 5:
                    continue
                free_neighbors.append((nx, ny))

            if not free_neighbors:
                # Dead-end: return path to best_target if we moved at all
                break

            if len(free_neighbors) >= 3:
                # Junction: stop here, don't go deeper — the junction scan will handle it
                best_target = (cx, cy)
                break

            # Single corridor — walk forward
            nx, ny = free_neighbors[0]
            visited_local.add((nx, ny))
            best_target = (nx, ny)
            cx, cy = nx, ny

        if best_target is None or best_target == (rmx, rmy):
            return None

        # Use A* to get a safe routable path to that cell
        path = self.planner._astar(rmx, rmy, best_target[0], best_target[1])
        return path if path else None
