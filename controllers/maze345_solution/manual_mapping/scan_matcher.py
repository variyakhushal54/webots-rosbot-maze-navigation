# Scan matching estimates the small transform between two lidar observations.
# Invalid ranges are removed before matching, and repeated alignment steps refine
# position and heading. The returned fitness score allows callers to reject weak
# matches instead of introducing a poor correction into the pose graph.
"""
scan_matcher.py — Point-to-point ICP for 2-D lidar scan matching.

Takes two lidar scans (ranges + fov), converts to 2-D point clouds,
and returns an SE(2) transform via iterative closest point.

No external deps beyond NumPy / SciPy.
"""
import math
import numpy as np

try:
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _scan_to_points(ranges, fov: float) -> np.ndarray:
    """
    Convert a 1-D range array into a (N, 2) array of 2-D Cartesian points.
    Invalid (inf / nan / < 0.10 m) readings are discarded.
    """
    n = len(ranges)
    pts = []
    for i, r in enumerate(ranges):
        if math.isnan(r) or math.isinf(r) or r < 0.10 or r > 4.0:
            continue
        bearing = fov / 2.0 - i * fov / n
        pts.append((r * math.cos(bearing), r * math.sin(bearing)))
    return np.array(pts, dtype=np.float64) if pts else np.empty((0, 2))

def _apply_se2(pts: np.ndarray, dx: float, dy: float, dtheta: float) -> np.ndarray:
    """Apply SE(2) transform (dx, dy, dtheta) to (N, 2) point array."""
    c, s = math.cos(dtheta), math.sin(dtheta)
    R = np.array([[c, -s], [s, c]])
    return (pts @ R.T) + np.array([dx, dy])

def _nearest_neighbours(ref: np.ndarray, query: np.ndarray, max_dist: float):
    """Return (indices_into_ref, distances) for each query point."""
    if _HAS_SCIPY and len(ref) > 0:
        tree = cKDTree(ref)
        dists, idx = tree.query(query, k=1, distance_upper_bound=max_dist * 2)
        return idx, dists
    else:
        idx, dists = _nearest_neighbours_brute(ref, query)
        return idx, dists

def _point_to_point_icp_step(ref: np.ndarray, query: np.ndarray, max_dist: float):
    """
    One ICP iteration.
    Returns (dx, dy, dtheta, rms, n_corr) computed from SVD-based closed-form.
    """
    idx, dists = _nearest_neighbours(ref, query, max_dist)

    
    valid = (dists < max_dist) & (idx < len(ref))
    if valid.sum() < 6:
        return 0.0, 0.0, 0.0, float('inf'), 0

    src = query[valid]
    dst = ref[idx[valid]]

    
    c_src = src.mean(axis=0)
    c_dst = dst.mean(axis=0)
    A = (src - c_src).T @ (dst - c_dst)

    U, _, Vt = np.linalg.svd(A)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = c_dst - R @ c_src
    dtheta = math.atan2(R[1, 0], R[0, 0])
    dx, dy = float(t[0]), float(t[1])

    
    transformed = _apply_se2(src, dx, dy, dtheta)
    rms = float(np.sqrt(np.mean(np.sum((transformed - dst) ** 2, axis=1))))
    return dx, dy, dtheta, rms, int(valid.sum())


def match_scans(ref_scan: tuple, query_scan: tuple,
                initial_guess: tuple = (0.0, 0.0, 0.0),
                max_iter: int = 20,
                max_corr_dist: float = 0.20) -> tuple:
    """
    Point-to-point ICP scan matcher.

    Parameters
    ----------
    ref_scan    : (ranges, fov)  — reference lidar scan (the stored node scan)
    query_scan  : (ranges, fov)  — current query scan
    initial_guess : (dx, dy, dtheta) in metres/radians
    max_iter    : maximum ICP iterations
    max_corr_dist : maximum correspondence distance [m]

    Returns
    -------
    (dx, dy, dtheta, fitness_score, info_matrix)
    fitness_score ∈ [0, 1]  — 1 = perfect overlap.
    info_matrix is 3×3 inverse-covariance estimate.
    Returns (0,0,0, 0.0, eye(3)*0.01) on failure.
    """
    ref_ranges, ref_fov     = ref_scan
    qry_ranges, qry_fov     = query_scan
    init_dx, init_dy, init_dtheta = initial_guess

    ref_pts = _scan_to_points(ref_ranges, ref_fov)
    qry_pts = _scan_to_points(qry_ranges, qry_fov)

    _FAIL = (0.0, 0.0, 0.0, 0.0, np.eye(3) * 0.01)

    if len(ref_pts) < 20 or len(qry_pts) < 20:
        return _FAIL

    
    qry_pts = _apply_se2(qry_pts, init_dx, init_dy, init_dtheta)

    cum_dx, cum_dy, cum_dtheta = init_dx, init_dy, init_dtheta
    prev_rms = float('inf')

    for _ in range(max_iter):
        dx, dy, dtheta, rms, n_corr = _point_to_point_icp_step(
            ref_pts, qry_pts, max_corr_dist)

        
        c, s = math.cos(cum_dtheta), math.sin(cum_dtheta)
        cum_dx    += c * dx - s * dy
        cum_dy    += s * dx + c * dy
        cum_dtheta = _wrap_angle(cum_dtheta + dtheta)

        
        qry_pts = _apply_se2(qry_pts, dx, dy, dtheta)

        
        if abs(prev_rms - rms) < 1e-5:
            break
        prev_rms = rms

    
    corr_ratio = n_corr / max(1, len(qry_pts))
    if prev_rms > 0.10 or corr_ratio < 0.30:
        return _FAIL

    
    
    fitness = float(max(0.0, 1.0 - prev_rms / max_corr_dist) * corr_ratio)

    
    
    sigma_xy    = max(0.01, prev_rms)
    sigma_theta = max(0.005, prev_rms / 0.5)   
    info_matrix = np.diag([
        1.0 / sigma_xy**2,
        1.0 / sigma_xy**2,
        1.0 / sigma_theta**2
    ])

    return (cum_dx, cum_dy, cum_dtheta, fitness, info_matrix)

def _wrap_angle(a: float) -> float:
    """Wrap angle to [-π, π]."""
    return math.atan2(math.sin(a), math.cos(a))
