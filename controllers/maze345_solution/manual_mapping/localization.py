# Localization keeps the robot position and heading used by every other module.
# Wheel encoders provide local movement and the inertial sensor provides heading.
# Scan matching may add small corrections when lidar geometry gives a reliable
# match, which limits the drift that wheel odometry builds up over time.
import math
import numpy as np
import cv2
from config import *

class Localizer:
    def __init__(self, sensors):
        self.sensors = sensors
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self._last_enc = None

    def update_pose(self, slam=None):
        """
        Calculates the robot's world position (x, y) and heading (theta).
        - Uses the IMU for absolute yaw (theta) to eliminate gyroscopic drift.
        - Uses Wheel Encoders to calculate the linear distance traveled.
        - OPTIONAL: Correlates Lidar hits with the existing map to correct slip/drift.
        """
        prev_theta = self.theta
        self.theta = self.sensors.get_imu_yaw()
        if not self.sensors.encoders:
            return
        curr = self.sensors.get_encoders()
        if self._last_enc is None:
            self._last_enc = curr
            return

        
        raw_dtheta = abs(self.theta - prev_theta)
        if raw_dtheta > math.pi:
            raw_dtheta = 2 * math.pi - raw_dtheta
        
        is_turning = raw_dtheta > 0.015

        
        dist = sum((c - l) * WHEEL_RADIUS for c, l in zip(curr, self._last_enc)) / len(curr)
        if getattr(self, 'slip_active', False):
            dist = 0.0
        self.x += dist * math.cos(self.theta)
        self.y += dist * math.sin(self.theta)
        self._last_enc = curr

        

        
        
        
        
        
        

        
        
        POST_TURN_SETTLE_LOC = 8   
        if not hasattr(self, '_icp_settle_frames'):
            self._icp_settle_frames = 0
        if is_turning:
            self._icp_settle_frames = POST_TURN_SETTLE_LOC
        elif self._icp_settle_frames > 0:
            self._icp_settle_frames -= 1

        icp_ok = (self._icp_settle_frames == 0) and not is_turning
        if icp_ok and slam and hasattr(slam, 'log_map') and np.sum(slam.log_map > WALL_THRESH) > 50:
            self._correct_pose_from_map(slam)

    def _correct_pose_from_map(self, slam):
        """Two-pass high-precision scan-to-map alignment to eliminate wall juggling."""
        ranges, w, fov = self.sensors.get_lidar()

        
        ranges_arr = np.asarray(ranges, dtype=np.float32)
        valid = (~np.isnan(ranges_arr)) & (~np.isinf(ranges_arr)) & (ranges_arr >= 0.2)
        valid_indices = np.where(valid)[0]

        if len(valid_indices) < 5:
            return

        valid_ranges = ranges_arr[valid_indices]

        
        bearings = fov/2 - valid_indices * (fov/w)
        ray_yaws = self.theta + LIDAR_OFFSET_YAW + bearings
        cos_rays = np.cos(ray_yaws)
        sin_rays = np.sin(ray_yaws)

        cos_t = math.cos(self.theta)
        sin_t = math.sin(self.theta)

        
        sx_base = LIDAR_OFFSET_X * cos_t - LIDAR_OFFSET_Y * sin_t
        sy_base = LIDAR_OFFSET_X * sin_t + LIDAR_OFFSET_Y * cos_t

        
        coarse_steps = np.array([-0.03, -0.015, 0.0, 0.015, 0.03], dtype=np.float32)
        
        dx_grid, dy_grid = np.meshgrid(coarse_steps, coarse_steps)
        dxs = dx_grid.flatten() 
        dys = dy_grid.flatten() 

        cxs = self.x + dxs 
        cys = self.y + dys 

        sxs = cxs + sx_base 
        sys = cys + sy_base 

        
        wxs = sxs[:, np.newaxis] + (valid_ranges * cos_rays)[np.newaxis, :] 
        wys = sys[:, np.newaxis] + (valid_ranges * sin_rays)[np.newaxis, :] 

        mxs = np.round(slam.origin + wxs / MAP_RES).astype(np.int32)
        mys = np.round(slam.origin - wys / MAP_RES).astype(np.int32)

        
        in_map = (mxs >= 0) & (mxs < MAP_SIZE) & (mys >= 0) & (mys < MAP_SIZE)
        mxs_clipped = np.clip(mxs, 0, MAP_SIZE - 1)
        mys_clipped = np.clip(mys, 0, MAP_SIZE - 1)

        
        scores = slam.log_map[mys_clipped, mxs_clipped].copy()
        scores[~in_map] = 0.0
        summed_scores = np.sum(scores, axis=1) 

        best_idx = np.argmax(summed_scores)
        best_dx = dxs[best_idx]
        best_dy = dys[best_idx]
        best_score = summed_scores[best_idx]

        
        fine_steps = np.array([-0.01, -0.0075, -0.005, -0.0025, 0.0, 0.0025, 0.005, 0.0075, 0.01], dtype=np.float32)
        fdx_grid, fdy_grid = np.meshgrid(fine_steps, fine_steps)
        fdxs = fdx_grid.flatten() 
        fdys = fdy_grid.flatten() 

        dxs_fine = best_dx + fdxs 
        dys_fine = best_dy + fdys 

        cxs_fine = self.x + dxs_fine 
        cys_fine = self.y + dys_fine 

        sxs_fine = cxs_fine + sx_base 
        sys_fine = cys_fine + sy_base 

        wxs_fine = sxs_fine[:, np.newaxis] + (valid_ranges * cos_rays)[np.newaxis, :] 
        wys_fine = sys_fine[:, np.newaxis] + (valid_ranges * sin_rays)[np.newaxis, :] 

        mxs_fine = np.round(slam.origin + wxs_fine / MAP_RES).astype(np.int32)
        mys_fine = np.round(slam.origin - wys_fine / MAP_RES).astype(np.int32)

        in_map_fine = (mxs_fine >= 0) & (mxs_fine < MAP_SIZE) & (mys_fine >= 0) & (mys_fine < MAP_SIZE)
        mxs_clipped_fine = np.clip(mxs_fine, 0, MAP_SIZE - 1)
        mys_clipped_fine = np.clip(mys_fine, 0, MAP_SIZE - 1)

        scores_fine = slam.log_map[mys_clipped_fine, mxs_clipped_fine].copy()
        scores_fine[~in_map_fine] = 0.0
        summed_scores_fine = np.sum(scores_fine, axis=1) 

        
        best_fine_idx = np.argmax(summed_scores_fine)
        if summed_scores_fine[best_fine_idx] > best_score:
            fine_best_dx = dxs_fine[best_fine_idx]
            fine_best_dy = dys_fine[best_fine_idx]
        else:
            fine_best_dx = best_dx
            fine_best_dy = best_dy

        
        
        self.x += 0.50 * fine_best_dx
        self.y += 0.50 * fine_best_dy
