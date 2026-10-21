import math
import os
import numpy as np
import cv2
from config import *


class Visualizer:
    def __init__(self, slam, loc, planner, perception, state_machine=None):
        self.slam = slam
        self.loc = loc
        self.planner = planner
        self.perception = perception
        self.state_machine = state_machine
        self._map_win_title = "MAZE MAP"
        self._map_window_ready = False

    def display(self):
        # Throttle display update rate
        if not hasattr(self, '_display_skip_counter'): self._display_skip_counter = 0
        self._display_skip_counter += 1
        if self._display_skip_counter % 2 != 0: return

        gray = np.full(self.slam.log_map.shape, 200, dtype=np.uint8)
        free_mask = (self.slam.log_map <= FREE_THRESH).astype(np.uint8)
        free_mask = cv2.morphologyEx(free_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        gray[free_mask > 0]  = 255
        gray[self.slam.log_map > WALL_THRESH] = 0
        view = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        view[self.slam.green_mask_global > 0] = (0, 180, 0)

        # Highlight floating walls in red
        cam_fw = self.slam.floating_wall_mask_global > 0
        if cam_fw.any():
            view[cam_fw] = (0, 0, 255)

        inf_fw = getattr(self.slam, 'floating_wall_inferred_mask', None)
        if inf_fw is not None and inf_fw.any():
            iy, ix = np.where(inf_fw > 0)
            dot = ((ix + iy) % 5) < 2
            view[iy[dot], ix[dot]] = (0, 0, 255)

        # Draw trajectory tracks
        if getattr(self.planner, 'nav_path', None):
            # Connect path to robot coordinate
            rx = self.slam.origin + int(self.loc.x / self.slam.map_res)
            ry = self.slam.origin - int(self.loc.y / self.slam.map_res)
            path = [(float(rx), float(ry))] + [(float(wp[0]), float(wp[1])) for wp in self.planner.nav_path]
            n_pts = len(path)
            if n_pts > 1:
                half_width_px = (PATH_DRAW_WIDTH / 2.0) / self.slam.map_res
                left_pts = []
                right_pts = []
                for i in range(n_pts):
                    x, y = path[i]
                    if i == 0:
                        tx = path[1][0] - path[0][0]
                        ty = path[1][1] - path[0][1]
                    elif i == n_pts - 1:
                        tx = path[-1][0] - path[-2][0]
                        ty = path[-1][1] - path[-2][1]
                    else:
                        tx = path[i+1][0] - path[i-1][0]
                        ty = path[i+1][1] - path[i-1][1]
                    
                    length = math.hypot(tx, ty)
                    if length > 0:
                        nx = -ty / length
                        ny = tx / length
                    else:
                        nx, ny = 0.0, 0.0
                    
                    left_pts.append((int(round(x - half_width_px * nx)), int(round(y - half_width_px * ny))))
                    right_pts.append((int(round(x + half_width_px * nx)), int(round(y + half_width_px * ny))))
                
                cv2.polylines(view, [np.array(left_pts, dtype=np.int32)], False, (255, 0, 0), 1)
                cv2.polylines(view, [np.array(right_pts, dtype=np.int32)], False, (255, 0, 0), 1)

        # Draw robot pose
        rx = self.slam.origin + int(self.loc.x / self.slam.map_res)
        ry = self.slam.origin - int(self.loc.y / self.slam.map_res)
        
        cv2.circle(view, (rx, ry), 4, (255, 0, 0), -1)
        hx = rx + int(10 * math.cos(self.loc.theta))
        hy = ry - int(10 * math.sin(self.loc.theta))
        cv2.line(view, (rx, ry), (hx, hy), (0, 255, 255), 2)

        # Draw overlays
        o = self.slam.overlay
        is_bg = (((o[:, :, 0] == 127) & (o[:, :, 1] == 127) & (o[:, :, 2] == 127)) |
                 ((o[:, :, 0] == 255) & (o[:, :, 1] == 255) & (o[:, :, 2] == 255)) |
                 ((o[:, :, 0] == 0) & (o[:, :, 1] == 0) & (o[:, :, 2] == 0)))
        mask = ~is_bg
        view[mask] = self.slam.overlay[mask]

        # Highlight floating walls
        floating_mask = (self.slam.floating_wall_mask_global > 0)
        inf_fw = getattr(self.slam, 'floating_wall_inferred_mask', None)
        if inf_fw is not None:
            floating_mask = floating_mask | (inf_fw > 0)
            
        if floating_mask.any():
            view[floating_mask] = (0, 0, 255)

        # Calculate dynamic crop bounding box
        sig = (self.slam.log_map > WALL_THRESH) | (self.slam.log_map < FREE_THRESH)
        explored = np.where(sig)
        if len(explored[0]) > 0:
            y_min, y_max, x_min, x_max = np.min(explored[0]), np.max(explored[0]), np.min(explored[1]), np.max(explored[1])
            pad = 25
            y_min = min(y_min, ry)
            y_max = max(y_max, ry)
            x_min = min(x_min, rx)
            x_max = max(x_max, rx)
            
            y_min, y_max = max(0, y_min - pad), min(MAP_SIZE - 1, y_max + pad)
            x_min, x_max = max(0, x_min - pad), min(MAP_SIZE - 1, x_max + pad)
            view = view[y_min:y_max + 1, x_min:x_max + 1]

        zoom_view = cv2.resize(view, (900, 900))

        if not self._map_window_ready:
            cv2.namedWindow(self._map_win_title, cv2.WINDOW_NORMAL)
            self._map_window_ready = True
        cv2.imshow(self._map_win_title, zoom_view)
        cv2.waitKey(1)
