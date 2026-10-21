import math
import numpy as np
import cv2
from config import *


class Planner:
    def __init__(self, slam, loc):
        self.slam = slam
        self.loc = loc
        self.nav_path = []
        self.mission_path = None
        self.confirmed_trails = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
        self.narrow_gap_mask = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)
        self.trail_weight = ASTAR_TRAIL_WEIGHT
        # Cache frontier results to save CPU
        self._frontier_cache = None
        self._frontier_cache_frame = -9999
        self._frontier_cache_ttl = 60
        self._costmap_cache = {}

    def _astar(self, sx, sy, gx, gy, truncate=True, temp_blocked=None):
        """Calculate A* path between start and goal cells."""
        import heapq
        
        # Store target coordinates
        orig_gx, orig_gy = gx, gy
        
        # Apply forced kernel if set
        forced = getattr(self, 'force_kernel', None)
        if forced is not None:
            kernels = [forced]
        else:
            last_k = getattr(self, 'last_path_kernel', None)
            if last_k is not None:
                # Put last successful kernel first
                kernels = [last_k] + [k for k in PLANNER_KERNELS if k != last_k]
            else:
                kernels = PLANNER_KERNELS
                
        for kernel_size in kernels:
            costmap = self._build_costmap(k_size=kernel_size)
            
            # Get precomputed distance field
            dist_field = getattr(self, 'last_dist_field', None)
            if dist_field is None:
                wall_bin = costmap.astype(np.uint8)
                dist_field = cv2.distanceTransform(1 - wall_bin, cv2.DIST_L2, 3) * MAP_RES
            max_dist = float(dist_field.max()) if dist_field.max() > 0 else 1.0
            
            def is_passable(mx, my):
                if temp_blocked and (mx, my) in temp_blocked:
                    return False
                return (0 <= mx < MAP_SIZE and 0 <= my < MAP_SIZE and 
                        not costmap[my, mx])
            
            # Snap start position to passable cell
            if not is_passable(sx, sy):
                snapped = False
                for radius in range(1, 6):
                    for dx in range(-radius, radius + 1):
                        for dy in range(-radius, radius + 1):
                            if is_passable(sx + dx, sy + dy):
                                sx, sy = sx + dx, sy + dy
                                snapped = True
                                break
                        if snapped:
                            break
                    if snapped:
                        break

            # Snap goal position to passable cell
            if not is_passable(gx, gy):
                snapped = False
                for radius in range(1, 26):
                    for dx in range(-radius, radius + 1):
                        for dy in range(-radius, radius + 1):
                            if is_passable(gx + dx, gy + dy):
                                gx, gy = gx + dx, gy + dy
                                snapped = True
                                break
                        if snapped:
                            break
                    if snapped:
                        break
            
            # If not passable, try next kernel size
            if not is_passable(sx, sy) or not is_passable(gx, gy):
                continue
                
            print("PLANNER", f"Attempting A* (Kernel={kernel_size}) from ({sx},{sy}) to ({gx},{gy})")

            # Initialize A* search structures
            open_set = []
            heapq.heappush(open_set, (0, 0, sx, sy))
            
            g_score = {(sx, sy): 0}
            came_from = {}
            visited = set()
            
            # 8-connected grid directions
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0), 
                          (1, 1), (1, -1), (-1, 1), (-1, -1)]
            
            # Run search loop
            best_path = None
            best_score = float('inf')
            
            while open_set:
                _, g, current_x, current_y = heapq.heappop(open_set)
                
                if (current_x, current_y) in visited:
                    continue
                    
                visited.add((current_x, current_y))
                
                # Check if goal reached
                if (current_x - gx)**2 + (current_y - gy)**2 <= 5:
                    # Reconstruct path
                    path = []
                    node = (current_x, current_y)
                    while node in came_from:
                        path.append(node)
                        node = came_from[node]
                    path.reverse()
                    
                    import config
                    max_len = getattr(config, 'ASTAR_MAX_PATH_LENGTH', None)
                    if truncate and max_len is not None and len(path) > max_len:
                        path = path[:max_len]
                        
                    # Simplify path waypoints
                    optimized_path = self._optimize_path(path, costmap)
                    
                    # Verify corner safety
                    if kernel_size > 0 and not self._has_safe_corners(optimized_path, k_size=kernel_size):
                        print("PLANNER", f"Kernel={kernel_size} path rejected: unsafe corner for chassis")
                        break
                    self.last_path_kernel = kernel_size
                    self.last_costmap = costmap
                    self.last_path_time = self.robot.getTime()
                    self._path_invalid_votes = 0
                    print("PLANNER", f"Success! Path found with kernel={kernel_size}, length={len(optimized_path)}")
                    return optimized_path
                
                # Limit search depth
                if len(visited) > 40000:
                    break
                    
                # Evaluate neighbor cells
                for dx, dy in directions:
                    nx, ny = current_x + dx, current_y + dy
                    
                    if (nx, ny) in visited or not is_passable(nx, ny):
                        continue

                    # Prevent diagonal movement through walls
                    if dx != 0 and dy != 0:
                        if not is_passable(current_x + dx, current_y) or \
                           not is_passable(current_x, current_y + dy):
                            continue
                        
                    if kernel_size == 0:
                        repulsion = 0.0
                        narrow_penalty = 0.0
                        centering_bonus = 0.0
                    else:
                        d_m = dist_field[ny, nx]

                        # Repel path away from walls
                        repulsion = ASTAR_HUG_WEIGHT * math.exp(
                            -(d_m**2) / (2.0 * (ASTAR_GAUSSIAN_STD**2)))

                        # Pull path to corridor center
                        if kernel_size >= 7:
                            c_sigma = 0.20
                        elif kernel_size >= 3:
                            c_sigma = 0.12
                        else:
                            c_sigma = 0.08

                        centering_bonus = ASTAR_CENTERING_WEIGHT * (1.0 - math.exp(
                            -(d_m**2) / (2.0 * (c_sigma**2))))

                        # Penalty for extreme proximity to wall
                        narrow_penalty = ASTAR_NARROW_WEIGHT if d_m < 0.15 else 0.0

                    move_cost = (1.414 if dx != 0 and dy != 0 else 1.0) + repulsion + narrow_penalty - centering_bonus

                    # Penalty for sharp turns
                    turn_penalty = 0.0
                    parent = came_from.get((current_x, current_y))
                    if parent is not None:
                        pdx = current_x - parent[0]
                        pdy = current_y - parent[1]
                        # Turn angle error
                        dot = pdx*dx + pdy*dy
                        norm = math.hypot(pdx, pdy) * math.hypot(dx, dy)
                        if norm > 0:
                            cosang = dot / norm
                            turn_penalty = max(0.0, (1.0 - cosang)) * ASTAR_TURN_WEIGHT 
                        
                    # Prefer paths parallel to walls
                    geom_bonus = 0.0
                    for (x0, y0, x1, y1) in getattr(self, 'wall_geometry', []):
                        # Check if near wall segment
                        w_mx, w_my = (nx-self.slam.origin)*MAP_RES, (self.slam.origin-ny)*MAP_RES
                        l2 = (x0-x1)**2 + (y0-y1)**2
                        t = max(0, min(1, ((w_mx-x0)*(x1-x0) + (w_my-y0)*(y1-y0)) / l2)) if l2 > 0 else 0
                        dist_to_wall = math.hypot(w_mx - (x0 + t*(x1-x0)), w_my - (y0 + t*(y1-y0)))
                        if 0.15 < dist_to_wall < 0.25:
                            geom_bonus = ASTAR_GEOM_WEIGHT; break
                        
                    # Avoid cutting sharp corners
                    corner_penalty = 0.0
                    if kernel_size > 0:
                        d_m = dist_field[ny, nx]
                        if d_m < 0.16: # Near wall limit
                            corner_penalty = (0.16 - d_m) * ASTAR_CORNER_WEIGHT 

                    # Penalty for already visited cells
                    visit_penalty = self.slam.visit_count[ny, nx] * ASTAR_VISIT_WEIGHT
                    
                    # Bonus for confirmed trail cells
                    trail_weight = getattr(self, 'trail_weight', ASTAR_TRAIL_WEIGHT)
                    trail_bonus = self.confirmed_trails[ny, nx] * trail_weight
                    
                    # Compute total cell cost
                    tentative_g = g + move_cost + turn_penalty + visit_penalty + corner_penalty - trail_bonus - geom_bonus
                    tentative_g = max(0.1, tentative_g) # Clamp to minimum positive value
                    
                    if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                        g_score[(nx, ny)] = tentative_g
                        came_from[(nx, ny)] = (current_x, current_y)
                        
                        # Add heuristic distance estimate
                        h = math.hypot(nx - gx, ny - gy) * 1.2
                        # Prefer straight lines
                        h *= (1.0 + 0.001 * math.hypot(nx - sx, ny - sy))
                        
                        heapq.heappush(open_set, (tentative_g + h, tentative_g, nx, ny))
        
        # If search failed
        print("PLANNER", f"A* failed: No path from ({sx},{sy}) to ({orig_gx},{orig_gy}) after trying all kernel sizes")
        return []

    def _build_narrow_gap_mask(self):
        """Mark narrow corridors as virtual walls."""
        import config as _cfg
        if not getattr(_cfg, 'ENABLE_NARROW_GAP_WALLS', True):
            self.narrow_gap_mask = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)
            return

        # Check frequency throttle
        if not hasattr(self, '_ngm_tick'): self._ngm_tick = 0
        self._ngm_tick += 1
        if self._ngm_tick % 15 != 1:
            return

        # Width of robot in cells
        robot_px = max(3, int(PATH_DRAW_WIDTH / MAP_RES))
        if robot_px % 2 == 0:
            robot_px += 1

        wall_bin = (self.slam.log_map > WALL_THRESH).astype(np.uint8)
        if wall_bin.sum() == 0:
            self.narrow_gap_mask = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)
            return

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (robot_px, robot_px))
        closed = cv2.morphologyEx(wall_bin, cv2.MORPH_CLOSE, kernel)

        # Find newly closed gap cells
        bridged = (closed.astype(bool)) & (~wall_bin.astype(bool))

        # Check if free space
        confirmed_free = self.slam.log_map < FREE_THRESH
        gap_cells = bridged & confirmed_free

        # Save gap cells as walls
        if gap_cells.any():
            self.slam.log_map[gap_cells] = LOG_CLIP
            self.slam.overlay[gap_cells] = (0, 0, 0)

        self.narrow_gap_mask = gap_cells

    def _build_costmap(self, k_size=13):
        """Build a binary costmap from log-odds grid."""
        # Return cache if valid
        block_unk = getattr(self, 'block_unknown', False)
        cache_key = (k_size, block_unk)
        cached = self._costmap_cache.get(cache_key)
        if cached is not None:
            cached_frame, cached_cm, cached_df = cached
            if cached_frame == self.slam.frame_iter:
                self.last_dist_field = cached_df
                return cached_cm
        # Compute narrow gaps first
        self._build_narrow_gap_mask()

        # Check ghost filter periodically
        if not hasattr(self, '_ghost_tick'): self._ghost_tick = 0
        self._ghost_tick += 1
        if self._ghost_tick % 30 == 0:
            self.slam._filter_ghost_obstacles()
        high = (self.slam.log_map > WALL_THRESH).astype(np.uint8) # Confirmed wall cells
        # Add floating wall cells
        fw_mask = (self.slam.floating_wall_mask_global > 0).astype(np.uint8)
        high = np.maximum(high, fw_mask)
        # Placeholders
        low = np.zeros_like(high)
        occ = high

        # Mark unknown cells if required
        if block_unk:
            unknown_or_wall = (self.slam.log_map >= -0.25).astype(np.uint8)
            occ = np.maximum(occ, unknown_or_wall)
        
        # Mark known obstacles

        if 'GREEN' in self.perception.cam_confirmed:
            rx, ry, _ = self.perception.cam_confirmed['GREEN']
            h = GREEN_HALF # Safe margin size
            px0 = max(0, self.slam.origin + int((rx - h) / MAP_RES))
            px1 = min(MAP_SIZE-1, self.slam.origin + int((rx + h) / MAP_RES))
            py0 = max(0, self.slam.origin - int((ry + h) / MAP_RES))
            py1 = min(MAP_SIZE-1, self.slam.origin - int((ry - h) / MAP_RES))
            occ[py0:py1+1, px0:px1+1] = 1 # Fill cells

        # Exclude dynamic tunnels
        tunnel_mask = self.slam.tunnel_map > 2.0
        occ[tunnel_mask] = 0
        
        # Color tunnels
        self.slam.overlay[tunnel_mask] = (255, 120, 255) 

        # Adjust padding size based on kernel
        if k_size >= 11:
            robot_half_w = ROBOT_HALF_W
            base_margin  = 0.03
        elif k_size >= 7:
            robot_half_w = ROBOT_HALF_W - 0.01
            base_margin  = 0.02
        elif k_size >= 3:
            robot_half_w = ROBOT_HALF_W - 0.02
            base_margin  = 0.01
        else:
            robot_half_w = ROBOT_HALF_W - 0.03
            base_margin  = 0.005

        MARGIN_PER_K     = 0.0035
        
        wall_bin = occ.astype(np.uint8)
        dist_px  = cv2.distanceTransform(1 - wall_bin, cv2.DIST_L2, 3)
        dist_m   = dist_px * MAP_RES
        self.last_dist_field = dist_m

        # Check zero padding case
        if k_size == 0:
            return wall_bin.astype(bool)

        pad_radius = robot_half_w + base_margin + MARGIN_PER_K * k_size
        padded = (dist_m <= pad_radius)

        # Inflate floating walls specifically
        fw_confirmed = (self.slam.floating_wall_mask_global > 0).astype(np.uint8)
        if fw_confirmed.any():
            fw_dist_px  = cv2.distanceTransform(1 - fw_confirmed, cv2.DIST_L2, 3)
            fw_dist_m   = fw_dist_px * MAP_RES
            # Set floating wall margins
            if k_size == 1:
                fw_margin = robot_half_w + 0.002
            elif k_size == 3:
                fw_margin = robot_half_w + 0.01
            else:
                fw_margin = robot_half_w * 2.0
            fw_padded   = (fw_dist_m <= fw_margin)
            padded = padded | fw_padded

        result_cm = (padded | low.astype(bool))
        # Save to cache
        self._costmap_cache[cache_key] = (self.slam.frame_iter, result_cm, dist_m)
        return result_cm

    def _find_frontiers(self, target_mx=None, target_my=None):
        """Find and rank search goals near unknown space."""
        # Return cache if valid
        if target_mx is None and target_my is None:
            age = self.slam.frame_iter - self._frontier_cache_frame
            if self._frontier_cache is not None and age < self._frontier_cache_ttl:
                return self._frontier_cache
        rmx, rmy = self.slam.origin + int(self.loc.x/MAP_RES), self.slam.origin - int(self.loc.y/MAP_RES)
        rtheta = self.loc.theta
        visited_mask = (self.slam.visit_count > 0).astype(np.float32)
        unknown_mask = (np.abs(self.slam.log_map) < 0.20).astype(np.uint8)
        unknown_mask_f = unknown_mask.astype(np.float32)
        # Novelty counts unknown cells
        novelty_map  = cv2.boxFilter(unknown_mask_f, -1, (41, 41), normalize=False)
        # Revisit counts explored cells
        revisit_map  = cv2.boxFilter(visited_mask, -1, (51, 51), normalize=False)

        near_unknown = cv2.dilate(unknown_mask, np.ones((19, 19), np.uint8))

        wall_bin = (self.slam.log_map > WALL_THRESH).astype(np.uint8)
        if wall_bin.sum() > 0:
            width_field = cv2.distanceTransform(1 - wall_bin, cv2.DIST_L2, 3)
        else:
            width_field = np.full_like(self.slam.log_map, 50.0, dtype=np.float32)

        WIDTH_WEIGHT    = FRONTIER_WIDTH_WEIGHT
        MOMENTUM_WEIGHT = 8.0

        # Add virtual walls to map
        self._build_narrow_gap_mask()

        # Filter reachable coordinates only
        free_bin = (self.slam.log_map < -0.25).astype(np.uint8)
        reachable_mask = free_bin.copy()
        h_m, w_m = reachable_mask.shape
        ff_mask = np.zeros((h_m+2, w_m+2), np.uint8)

        seed_x = max(0, min(w_m-1, rmx))
        seed_y = max(0, min(h_m-1, rmy))

        if free_bin[seed_y, seed_x] == 1:
            cv2.floodFill(reachable_mask, ff_mask, (seed_x, seed_y), 2)
        else:
            # Try nearby cells if needed
            found = False
            for r in range(1, 6):
                for dy in range(-r, r+1):
                    for dx in range(-r, r+1):
                        ny, nx = seed_y + dy, seed_x + dx
                        if 0 <= ny < h_m and 0 <= nx < w_m and free_bin[ny, nx] == 1:
                            cv2.floodFill(reachable_mask, ff_mask, (nx, ny), 2)
                            found = True
                            break
                    if found: break
                if found: break

        reachable_free = (reachable_mask == 2)

        # Snap window size
        do_center_snap = getattr(__import__('config'), 'FRONTIER_CENTER_SNAP', True)
        SNAP_R = max(3, int(0.30 / MAP_RES))

        frontiers = []
        for my in range(6, MAP_SIZE-6, 6):
            for mx in range(6, MAP_SIZE-6, 6):
                if not reachable_free[my, mx]: continue
                if not near_unknown[my, mx]:   continue
                if self.slam.visit_count[my, mx] > 8 or novelty_map[my, mx] < 15:
                    continue

                clearance = float(width_field[my, mx])
                # Check minimum clearance limit
                min_clearance_px = (PATH_DRAW_WIDTH / 2.0) / MAP_RES
                if clearance < min_clearance_px:
                    continue

                # Centering target check
                if do_center_snap:
                    y0 = max(0, my - SNAP_R)
                    y1 = min(MAP_SIZE, my + SNAP_R + 1)
                    x0 = max(0, mx - SNAP_R)
                    x1 = min(MAP_SIZE, mx + SNAP_R + 1)
                    patch = width_field[y0:y1, x0:x1]
                    # Check free space status
                    free_patch = (self.slam.log_map[y0:y1, x0:x1] < -0.25).astype(np.float32)
                    weighted = patch * free_patch
                    # Fallback check
                    if weighted.max() > 0:
                        flat_idx = np.argmax(weighted)
                        local_r, local_c = divmod(int(flat_idx), weighted.shape[1])
                        cmx = x0 + local_c
                        cmy = y0 + local_r
                        # Accept snap if better
                        if width_field[cmy, cmx] > clearance * 1.05:
                            mx, my = cmx, cmy
                            clearance = float(width_field[cmy, cmx])

                # Alignment check
                dx_f = mx - rmx
                dy_f = my - rmy
                dist_f = math.hypot(dx_f, dy_f) or 1.0
                heading_x = math.cos(rtheta)
                heading_y = -math.sin(rtheta) # Inverted coordinates check
                alignment = (dx_f / dist_f) * heading_x + (dy_f / dist_f) * heading_y

                score = (novelty_map[my, mx]   * FRONTIER_NOVELTY_WEIGHT
                         - dist_f              * FRONTIER_DISTANCE_WEIGHT
                         - revisit_map[my, mx] * 8.0
                         - self.slam.visit_count[my, mx] * 6.0
                         + clearance           * WIDTH_WEIGHT
                         + alignment           * MOMENTUM_WEIGHT)
                if target_mx is not None:
                    score -= math.hypot(mx - target_mx, my - target_my) * 5.0

                # Check if visited before
                # Window check size
                is_new = not self.slam.visit_count[max(0, my-8):min(MAP_SIZE, my+9), 
                                                   max(0, mx-8):min(MAP_SIZE, mx+9)].any()
                frontiers.append((-score, is_new, mx, my))

        # Sort results
        frontiers.sort()

        # Return new cells if any exist
        new_cells      = [(mx, my) for _, is_n, mx, my in frontiers if is_n]
        visited_cells  = [(mx, my) for _, is_n, mx, my in frontiers if not is_n]
        result = new_cells if new_cells else visited_cells

        # Cache current results
        if target_mx is None and target_my is None:
            self._frontier_cache = result
            self._frontier_cache_frame = self.slam.frame_iter

        return result

    def _path_still_valid(self):
        """Check if path is still clear of walls."""
        if not self.nav_path: return True
        
        # Skip check for tightest paths
        k = getattr(self, 'last_path_kernel', 11)
        if k <= 1:
            return True
        
        # Check grace period
        if getattr(self, '_path_validity_grace', False):
            return True
        
        # Skip check for new paths
        path_age = self.robot.getTime() - getattr(self, 'last_path_time', 0.0)
        if path_age < 0.3:
            return True
        
        # Throttle check rate
        current_time = self.robot.getTime()
        if not hasattr(self, '_last_path_valid_check'):
            self._last_path_valid_check = 0.0
        
        if current_time - self._last_path_valid_check < 0.05:
            return True
        
        self._last_path_valid_check = current_time
        
        # Determine check footprint based on kernel
        k = getattr(self, 'last_path_kernel', 11)
        if k >= 11:
            footprint_radius = 4
        elif k >= 9:
            footprint_radius = 3
        elif k >= 7:
            footprint_radius = 2
        elif k >= 5:
            footprint_radius = 1
        else:
            footprint_radius = 0
        
        # Precompute cell offsets
        if footprint_radius > 0:
            _r = footprint_radius
            _offsets = [(dx, dy) for dx in range(-_r, _r+1)
                        for dy in range(-_r, _r+1) if dx*dx+dy*dy <= _r*_r]
            _off_arr = np.array(_offsets, dtype=np.int32)
        else:
            _off_arr = np.zeros((1, 2), dtype=np.int32)

        wall_map = self.slam.log_map > WALL_THRESH

        def footprint_blocked(x0, y0, x1, y1):
            steps = max(abs(x1 - x0), abs(y1 - y0)) * 2
            if steps == 0:
                return False
            # Project points
            ts = np.linspace(0, 1, steps + 1)
            cxs = np.clip((x0 + (x1 - x0) * ts).astype(np.int32), 0, MAP_SIZE - 1)
            cys = np.clip((y0 + (y1 - y0) * ts).astype(np.int32), 0, MAP_SIZE - 1)
            # Check grid footprint
            all_px = np.clip((cxs[:, None] + _off_arr[:, 0]).ravel(), 0, MAP_SIZE - 1)
            all_py = np.clip((cys[:, None] + _off_arr[:, 1]).ravel(), 0, MAP_SIZE - 1)
            return bool(wall_map[all_py, all_px].any())

        # Check intermediate path segments only
        if len(self.nav_path) < 2:
            self._path_invalid_votes = 0
            return True
            
        blocked = False
        prev = self.nav_path[0]
        for wp in self.nav_path[1:]:
            if footprint_blocked(prev[0], prev[1], wp[0], wp[1]):
                blocked = True
                break
            prev = wp
            
        if blocked:
            if not hasattr(self, '_path_invalid_votes'):
                self._path_invalid_votes = 0
            self._path_invalid_votes += 1
            if self._path_invalid_votes >= 1: # Block threshold check
                print("PATH_INVALID", f"Wall detected on path (votes={self._path_invalid_votes}) — discarding immediately.")
                self._path_invalid_votes = 0
                return False
            return True
        else:
            self._path_invalid_votes = 0
            return True

    def stamp_trail(self, x, y):
        """Mark robot trajectory coordinates."""
        mx = self.slam.origin + int(x / MAP_RES)
        my = self.slam.origin - int(y / MAP_RES)
        if 0 <= mx < MAP_SIZE and 0 <= my < MAP_SIZE:
            # Mark cell footprint
            r = 2
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dx*dx + dy*dy > r*r: continue
                    px, py = mx + dx, my + dy
                    if 0 <= py < MAP_SIZE and 0 <= px < MAP_SIZE:
                        self.confirmed_trails[py, px] = min(self.confirmed_trails[py, px] + 0.2, 5.0)
                        if self.slam.log_map[py, px] < -0.25:
                            self.slam.visit_count[py, px] = min(255, self.slam.visit_count[py, px] + 1)

    def _has_safe_corners(self, path, turn_deg=70, clearance_m=None, k_size=None):
        """Check corner safety clearance margins."""
        if clearance_m is None:
            k = k_size if k_size is not None else getattr(self, 'last_path_kernel', 11)
            if k == 0:
                clearance_m = (PATH_DRAW_WIDTH / 2.0) * 0.6 # permissive values
            else:
                margin = 0.005 if k > 3 else 0.0
                clearance_m = (PATH_DRAW_WIDTH / 2.0) + margin

        if len(path) < 3: return True
        r_cells = max(1, int(clearance_m / MAP_RES))
        walls = (self.slam.log_map > WALL_THRESH)
        turn_rad = math.radians(turn_deg)

        # Trajectory footprint
        trail_mask = self.confirmed_trails > 0.5

        for i in range(1, len(path) - 1):
            (x0, y0), (x1, y1), (x2, y2) = path[i-1], path[i], path[i+1]
            a1 = math.atan2(y1 - y0, x1 - x0)
            a2 = math.atan2(y2 - y1, x2 - x1)
            turn = abs(((a2 - a1 + math.pi) % (2 * math.pi)) - math.pi)
            if turn < turn_rad: continue

            # Skip if corner is on trajectory
            y_t_lo = max(0, y1 - 1); y_t_hi = min(MAP_SIZE, y1 + 2)
            x_t_lo = max(0, x1 - 1); x_t_hi = min(MAP_SIZE, x1 + 2)
            if trail_mask[y_t_lo:y_t_hi, x_t_lo:x_t_hi].any():
                continue

            y_lo = max(0, y1 - r_cells); y_hi = min(MAP_SIZE, y1 + r_cells + 1)
            x_lo = max(0, x1 - r_cells); x_hi = min(MAP_SIZE, x1 + r_cells + 1)
            if walls[y_lo:y_hi, x_lo:x_hi].any():
                return False
        return True

    def _optimize_path(self, path, costmap):
        """Smooth path waypoints."""
        if len(path) <= 2:
            return path

        optimized = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self._is_line_of_sight(path[i], path[j], costmap):
                    optimized.append(path[j])
                    i = j
                    break
                j -= 1
            else:
                optimized.append(path[i + 1])
                i += 1

        if len(optimized) <= 2:
            return optimized

        weight_data   = 0.35
        weight_smooth = 0.20
        tolerance     = 1e-4
        max_iter      = 80

        orig = [(float(p[0]), float(p[1])) for p in optimized]
        smooth = [list(p) for p in orig]

        dist_field = getattr(self, 'last_dist_field', None)
        robot_safe_m = (PATH_DRAW_WIDTH / 2.0) + 0.01

        for _ in range(max_iter):
            change = 0.0
            for k in range(1, len(smooth) - 1):
                for dim in range(2):
                    old = smooth[k][dim]
                    new_val = old + weight_data * (orig[k][dim] - old)
                    new_val += weight_smooth * (smooth[k-1][dim] + smooth[k+1][dim] - 2.0 * old)
                    
                    # Calculate next position
                    nx, ny = smooth[k][0], smooth[k][1]
                    if dim == 0: nx = new_val
                    else:        ny = new_val
                    
                    rx = int(round(nx))
                    ry = int(round(ny))
                    rx = max(0, min(MAP_SIZE - 1, rx))
                    ry = max(0, min(MAP_SIZE - 1, ry))
                    
                    is_safe = True
                    # Clearance check
                    if dist_field is not None and dist_field[ry, rx] < robot_safe_m:
                        is_safe = False
                        
                    # Intersection check
                    if is_safe and not self._is_line_of_sight((rx, ry), optimized[k], costmap):
                        is_safe = False
                        
                    if is_safe:
                        smooth[k][dim] = new_val
                        change += abs(new_val - old)
            
            if change < tolerance:
                break

        # Convert back to grid indexes
        result = [optimized[0]]
        for k in range(1, len(smooth) - 1):
            rx, ry = int(round(smooth[k][0])), int(round(smooth[k][1]))
            rx = max(0, min(MAP_SIZE - 1, rx))
            ry = max(0, min(MAP_SIZE - 1, ry))
            result.append((rx, ry))
        result.append(optimized[-1])
        return result

    def _is_line_of_sight(self, start, end, costmap):
        """Check line of sight visibility between cells."""
        x0, y0 = start
        x1, y1 = end

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        n = 1 + dx + dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        error = dx - dy
        dx *= 2
        dy *= 2

        for _ in range(n):
            if costmap[y, x]: # Check cell blockage
                return False

            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx

        return True
