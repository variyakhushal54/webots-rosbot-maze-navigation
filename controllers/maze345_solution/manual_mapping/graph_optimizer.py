# This module corrects drift after the robot recognizes a previously visited area.
# It solves the pose graph as one connected system instead of adjusting only the
# latest pose. The first pose remains fixed so the map keeps a stable origin while
# position and heading errors are shared across the later poses.
"""
graph_optimizer.py — Pose-graph optimisation via Gauss-Newton (pure NumPy/SciPy).

Minimises the sum of squared, information-weighted edge residuals.
First node is fixed (gauge constraint).

Optional: wraps g2o-python if installed (faster on large graphs).
"""
import math
import time
import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    import g2o
    _HAS_G2O = True
except ImportError:
    _HAS_G2O = False


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))

def _error_vec(x_i, x_j, z_ij):
    """
    Compute 3-vector residual e = R_i^T (t_j - t_i) - t_ij  ‖  wrap(θ_j - θ_i - θ_ij).
    All inputs are (x, y, theta) tuples/arrays.
    """
    xi, yi, thi = x_i
    xj, yj, thj = x_j
    dx_ij, dy_ij, dth_ij = z_ij

    
    dx = xj - xi
    dy = yj - yi
    dth = _wrap(thj - thi)

    
    c, s = math.cos(thi), math.sin(thi)
    ex = c * dx + s * dy - dx_ij
    ey = -s * dx + c * dy - dy_ij
    eth = _wrap(dth - dth_ij)
    return np.array([ex, ey, eth])

def _jacobians(x_i, x_j):
    """
    Analytical Jacobians of e w.r.t. x_i (Ji) and x_j (Jj).
    Both are 3×3 matrices.
    """
    xi, yi, thi = x_i
    xj, yj, thj = x_j
    dx = xj - xi
    dy = yj - yi
    c, s = math.cos(thi), math.sin(thi)

    
    Ji = np.array([
        [-c, -s,  -s * dx + c * dy],
        [ s, -c,  -c * dx - s * dy],
        [ 0,  0,  -1.0            ],
    ])
    
    Jj = np.array([
        [c,  s,  0.0],
        [-s, c,  0.0],
        [0,  0,  1.0],
    ])
    return Ji, Jj


def _gauss_newton(pose_graph, max_iter: int, tol: float) -> dict:
    """
    Gauss-Newton minimisation of the pose-graph error.
    Uses scipy.sparse if available for efficiency; falls back to dense NumPy.
    """
    nodes = pose_graph.get_all_nodes()
    edges = pose_graph.get_edges()
    n = len(nodes)
    if n < 2:
        return pose_graph.get_optimized_poses()

    
    opt_poses = pose_graph.get_optimized_poses()
    id_to_idx = {node.node_id: i for i, node in enumerate(nodes)}
    fixed_id  = nodes[0].node_id          

    
    x = np.array([v for nid, (px, py, pt) in sorted(opt_poses.items())
                  for v in (px, py, pt)], dtype=np.float64)
    n_vars = 3 * n

    for iteration in range(max_iter):
        if _HAS_SCIPY:
            H = sp.lil_matrix((n_vars, n_vars), dtype=np.float64)
            b = np.zeros(n_vars)
        else:
            H = np.zeros((n_vars, n_vars))
            b = np.zeros(n_vars)

        total_err = 0.0

        for edge in edges:
            i_idx = id_to_idx[edge.from_id]
            j_idx = id_to_idx[edge.to_id]

            xi = x[3*i_idx : 3*i_idx+3]
            xj = x[3*j_idx : 3*j_idx+3]
            z  = np.array([edge.dx, edge.dy, edge.dtheta])
            Om = edge.info_matrix   

            e       = _error_vec(xi, xj, z)
            Ji, Jj  = _jacobians(xi, xj)

            total_err += float(e @ Om @ e)

            
            for (row_base, J_r) in [(3*i_idx, Ji), (3*j_idx, Jj)]:
                if id_to_idx[edge.from_id] == id_to_idx[fixed_id] and row_base == 3*i_idx:
                    continue
                for (col_base, J_c) in [(3*i_idx, Ji), (3*j_idx, Jj)]:
                    H[row_base:row_base+3, col_base:col_base+3] += J_r.T @ Om @ J_c
                b[row_base:row_base+3] -= J_r.T @ Om @ e

        
        H[0:3, :] = 0.0
        H[:, 0:3] = 0.0
        H[0:3, 0:3] = np.eye(3)
        b[0:3]  = 0.0

        
        try:
            if _HAS_SCIPY:
                H_csr = H.tocsr()
                dx, _ = spla.cg(H_csr, b, atol=1e-8, maxiter=500)
            else:
                dx = np.linalg.solve(H, b)
        except Exception:
            break   

        
        for k in range(2, n_vars, 3):
            dx[k] = _wrap(dx[k])

        x += dx

        if np.linalg.norm(dx) < tol:
            break

    
    result = {}
    for node in nodes:
        i = id_to_idx[node.node_id]
        px, py, pt = x[3*i], x[3*i+1], x[3*i+2]
        result[node.node_id] = (float(px), float(py), float(_wrap(pt)))
    return result



def optimize_pose_graph(pose_graph, max_iter: int = 20, tol: float = 1e-4) -> dict:
    """
    Optimise the pose graph and return corrected poses.

    Returns
    -------
    {node_id: (x_opt, y_opt, theta_opt)}

    Raises nothing — returns current estimates on failure.
    """
    t0 = time.perf_counter()
    n  = pose_graph.node_count()

    if n < 2:
        return pose_graph.get_optimized_poses()

    try:
        if _HAS_G2O:
            result = _g2o_optimize(pose_graph, max_iter)
        else:
            result = _gauss_newton(pose_graph, max_iter, tol)
    except Exception as err:
        print(f"[GraphOpt] WARNING: optimisation failed ({err}), keeping current poses")
        return pose_graph.get_optimized_poses()

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[GraphOpt] Optimised {n} nodes / {len(pose_graph.get_edges())} edges "
          f"in {elapsed_ms:.1f} ms  (g2o={'yes' if _HAS_G2O else 'no'})")
    return result
