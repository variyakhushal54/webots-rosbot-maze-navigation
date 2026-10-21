# Floating walls can be missed by lidar because they do not cross its scan plane.
# This module combines camera depth, wood color, lidar clearance, and signs that
# the robot is stuck. It waits for enough supporting evidence before placing an
# obstacle in the map, then removes weak observations that become stale.
import config
"""
Mixed floating-wall fusion — never rely on camera wood texture alone.

Evidence channels (each frame):
  • depth band ahead (robot-height obstacles, no HSV required)
  • camera wood (strong / weak pixel counts)
  • stuck + open lidar + free map ahead
  • tactile forward stall

Solid red map stamp  : depth + (wood OR open-lidar) confirmation
Red dotted stamp     : stuck blind-ahead + at least one non-camera cue
"""
import math
import numpy as np
import cv2
from config import *

class FloatingWallFusion:
  def __init__(self):
    self._last_log_frame = -999
    self.tactile_stall_flag = False
    
    
    self._pending_pts = []

  def reset_tactile(self):
    self.tactile_stall_flag = False

  def note_tactile_stall(self):
    self.tactile_stall_flag = True

  def _score_depth(self, perception):
    n_danger = getattr(perception, 'depth_danger_count', 0)
    n_band = len(getattr(perception, 'depth_band_points', []))
    min_danger = getattr(config, 'FLOATING_FUSION_DEPTH_DANGER_FRAC', 0.4)
    min_pts = getattr(config, 'FLOATING_DEPTH_ONLY_MIN_POINTS', 12)
    thresh = getattr(config, 'FLOATING_WALL_DANGER_COUNT', 50)
    if n_danger >= thresh * min_danger:
      return 1.0
    if n_band >= min_pts:
      return 0.85
    if n_danger > 0:
      return 0.4
    return 0.0

  def _score_wood(self, perception):
    pts = getattr(perception, 'floating_wall_points', None)
    n_strong = len(pts) if pts is not None else 0
    min_strong = getattr(config, 'FLOATING_WALL_MIN_POINTS', 40)
    weak_px = getattr(perception, 'wood_weak_pixels', 0)
    weak_min = getattr(config, 'FLOATING_WEAK_WOOD_PIXELS', 150) 

    
    
    total_px = 640 * 480
    if weak_px > total_px * 0.5:
      weak_px = total_px * 0.2

    if n_strong >= min_strong:
      return 1.0
    if weak_px >= weak_min:
      return 0.55
    if weak_px >= weak_min // 2:
      return 0.25
    return 0.0

  def _score_pillar_occlusion(self, perception):
    """Partial BLUE/YELLOW pillar vs Webots 0.3×0.2 m → obstacle on line of sight."""
    score = getattr(perception, 'pillar_occlusion_score', 0.0)
    vis = getattr(perception, 'pillar_visibility', {})
    for name in ('BLUE', 'YELLOW'):
      info = vis.get(name, {})
      if info.get('seen') and info.get('full') is False:
        score = max(score, 0.7)
      elif info.get('seen') and info.get('full') is None:
        score = max(score, 0.3)
    return min(1.0, score)

  def _score_stuck(self, recovery):
    if recovery is None:
      return 0.0
    s = 0.0
    if recovery._is_floating_wall_stuck():
      s = 1.0
    elif getattr(recovery, 'stuck_counter', 0) >= 2:
      s = 0.6
    if getattr(recovery, '_last_cmd_vx', 0.0) > 0.05 and recovery._lidar_open_ahead():
      s = max(s, 0.35)
    return s

  def evaluate(self, perception, recovery, slam, frame):
    """
    Returns dict:
      stamp_camera: bool — solid red (mixed depth + wood/lidar)
      stamp_inferred: bool — red dotted (stuck fusion)
      camera_points: ndarray (N,2)
    """
    import config as cfg

    w_depth = getattr(cfg, 'FLOATING_FUSION_DEPTH_WEIGHT', 1.0)
    w_wood = getattr(cfg, 'FLOATING_FUSION_WOOD_WEIGHT', 0.9)
    w_stuck = getattr(cfg, 'FLOATING_FUSION_STUCK_WEIGHT', 1.2)
    w_lidar = getattr(cfg, 'FLOATING_FUSION_LIDAR_OPEN_WEIGHT', 0.6)
    w_tactile = getattr(cfg, 'FLOATING_FUSION_TACTILE_WEIGHT', 1.0)
    w_pillar = getattr(cfg, 'FLOATING_FUSION_PILLAR_WEIGHT', 1.1)
    cam_thresh = getattr(cfg, 'FLOATING_FUSION_CAMERA_STAMP_SCORE', 1.85)
    infer_thresh = getattr(cfg, 'FLOATING_FUSION_INFER_STAMP_SCORE', 2.0)

    d = self._score_depth(perception)
    w = self._score_wood(perception)
    p_occ = self._score_pillar_occlusion(perception)
    st = self._score_stuck(recovery)
    lidar_open = recovery is not None and recovery._lidar_open_ahead()
    map_open = recovery is not None and recovery._map_open_ahead()
    tactile = self.tactile_stall_flag

    
    score_cam = d * w_depth
    if w >= 0.55:
      score_cam += w * w_wood
    if d >= 0.85 and lidar_open:
      score_cam += w_lidar * 0.7
    if w >= 1.0 and lidar_open and d >= 0.4:
      score_cam += w_lidar * 0.5
    if p_occ >= 0.7:
      score_cam += w_pillar * p_occ * 0.9
      if lidar_open:
        score_cam += w_lidar * 0.4

    
    
    stamp_camera = (
        (score_cam >= cam_thresh and (d >= 0.4 or p_occ >= 0.7))
        or (w >= 1.0 and (d >= 0.25 or lidar_open))
        or (p_occ >= 0.85 and lidar_open)
    )

    
    score_infer = 0.0
    if st >= 0.6 and lidar_open and map_open:
      score_infer += w_stuck * st
      score_infer += d * w_depth * 0.8
      score_infer += w * w_wood * 0.5
      if tactile:
        score_infer += w_tactile
      if getattr(recovery, 'stuck_counter', 0) >= 3:
        score_infer += 0.4
      score_infer += p_occ * w_pillar

    stamp_inferred = (
        (p_occ >= 0.7 and lidar_open and map_open
         and not getattr(recovery, '_inferred_wall_stamped', False))
        or (
        score_infer >= infer_thresh
        and st >= 0.6
        and lidar_open
        and map_open
        and not getattr(recovery, '_inferred_wall_stamped', False))
    )

    camera_pts = []
    wood_pts = getattr(perception, 'floating_wall_points', None)
    if wood_pts is not None and len(wood_pts):
      camera_pts.append(wood_pts)
    band_pts = getattr(perception, 'depth_band_points', None)
    if band_pts is not None and len(band_pts):
      camera_pts.append(band_pts)
    p_pts = getattr(perception, 'pillar_occlusion_points', None)
    if p_pts is not None and len(p_pts):
      camera_pts.append(p_pts)
    if camera_pts:
      camera_pts = np.vstack(camera_pts).astype(np.float32)
    else:
      camera_pts = np.empty((0, 2), dtype=np.float32)

    if frame - self._last_log_frame >= 45:
      self._last_log_frame = frame
      
      
      
      
      
      

    return {
        'stamp_camera': stamp_camera,
        'stamp_inferred': stamp_inferred,
        'camera_points': camera_pts,
        'score_cam': score_cam,
        'score_infer': score_infer,
    }

  def _flush_pending_near_robot(self, perception, slam, frame):
    """Stamp any pending detections the robot has now approached close enough."""
    import config as _cfg
    confirm_dist = getattr(_cfg, 'FLOATING_WALL_CONFIRM_DIST_M', 0.55)
    max_frames   = getattr(_cfg, 'FLOATING_WALL_PENDING_MAX_FRAMES', 120)
    rx, ry = perception.loc.x, perception.loc.y

    still_pending = []
    for entry in self._pending_pts:
      age = frame - entry['frame']
      if age > max_frames:
        
        continue
      pts = entry['pts']
      
      dists = np.sqrt((pts[:, 0] - rx)**2 + (pts[:, 1] - ry)**2)
      if dists.min() <= confirm_dist:
        slam.stamp_floating_wall_points(pts, inferred=False)
        
        
        
      else:
        still_pending.append(entry)
    self._pending_pts = still_pending

  def apply(self, perception, recovery, slam, frame):
    """Run fusion and stamp map when mixed criteria are met."""
    if slam is None or perception is None:
      return

    
    self._flush_pending_near_robot(perception, slam, frame)

    result = self.evaluate(perception, recovery, slam, frame)

    if result['stamp_camera'] and len(result['camera_points']) > 0:
        
        mxs = slam.origin + (result['camera_points'][:, 0] / slam.map_res).astype(np.int32)
        mys = slam.origin - (result['camera_points'][:, 1] / slam.map_res).astype(np.int32)
        
        lidar_mask = (slam.log_map > WALL_THRESH)

        
        min_dist_cells = max(1, int(getattr(config, 'FLOATING_WALL_MIN_DIST_M', 0.12) / slam.map_res))
        kernel_sz = 2 * min_dist_cells - 1
        if kernel_sz >= 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_sz, kernel_sz))
            lidar_dilated = cv2.dilate(lidar_mask.astype(np.uint8), kernel)
        else:
            lidar_dilated = lidar_mask.astype(np.uint8)

        
        r_margin = getattr(config, 'PILLAR_RADIUS_M', 0.10) + 0.03
        margin_cells_pillar = int(math.ceil(r_margin / slam.map_res))
        pillar_points = []
        if hasattr(slam, 'perception') and slam.perception is not None:
            for color in ['BLUE', 'YELLOW', 'GREEN']:
                if color in slam.perception.cam_confirmed:
                    pillar_points.append(slam.perception.cam_confirmed[color][:2])
                else:
                    buf = slam.perception.cam_pillar_buf.get(color, [])
                    if len(buf) >= 3:
                        recent = buf[-10:]
                        xs = sorted(p[0] for p in recent)
                        ys = sorted(p[1] for p in recent)
                        pillar_points.append((xs[len(xs)//2], ys[len(ys)//2]))
        for cx, cy in pillar_points:
            pmx = slam.origin + int(cx / slam.map_res)
            pmy = slam.origin - int(cy / slam.map_res)
            cv2.circle(lidar_dilated, (pmx, pmy), margin_cells_pillar, 0, -1)

        
        in_bounds = (mxs >= 0) & (mxs < slam.map_size) & (mys >= 0) & (mys < slam.map_size)
        mxs_b = mxs[in_bounds]
        mys_b = mys[in_bounds]
        pts_b = result['camera_points'][in_bounds]
        not_lidar_wall = (lidar_dilated[mys_b, mxs_b] == 0)
        not_confirmed  = (slam.log_map[mys_b, mxs_b] <= WALL_THRESH)
        keep = not_lidar_wall & not_confirmed
        if keep.any():
            filtered_pts = pts_b[keep]

            
            if pillar_points:
                keep_non_pillar = np.ones(len(filtered_pts), dtype=bool)
                for i, pt in enumerate(filtered_pts):
                    for cx, cy in pillar_points:
                        if math.hypot(pt[0]-cx, pt[1]-cy) < r_margin:
                            keep_non_pillar[i] = False
                            break
                filtered_pts = filtered_pts[keep_non_pillar]

            if len(filtered_pts) == 0:
                span = 0.0
            elif len(filtered_pts) >= 2:
                pt_min = filtered_pts.min(axis=0)
                pt_max = filtered_pts.max(axis=0)
                span = float(np.max(pt_max - pt_min))  
            else:
                span = 0.0

            import config as _cfg
            min_span = getattr(_cfg, 'FLOATING_MIN_WALL_WIDTH_M', 0.03)

            if span < min_span:
                pass 
                     
            else:
                
                
                
                
                
                import config as _cfg
                confirm_dist = getattr(_cfg, 'FLOATING_WALL_CONFIRM_DIST_M', 0.55)
                rx, ry = perception.loc.x, perception.loc.y
                dists = np.sqrt((filtered_pts[:, 0] - rx)**2 +
                                (filtered_pts[:, 1] - ry)**2)
                close_mask = dists <= confirm_dist

                if close_mask.any():
                    
                    slam.stamp_floating_wall_points(filtered_pts[close_mask], inferred=False)
                    
                    
                    

                far_pts = filtered_pts[~close_mask]
                if len(far_pts) > 0:
                    self._pending_pts.append({'pts': far_pts, 'frame': frame})
                    
                    
                    
                    

    
    slam._promote_floating_wall_votes()

    
    
    

    
    
    
    
    
    

    
    
    
    

    if hasattr(slam, '_strip_floating_duplicate_near_lidar'):
      slam._strip_floating_duplicate_near_lidar()

    self.reset_tactile()
