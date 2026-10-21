# This file improves the pose graph after new movement or loop measurements arrive.
# It moves the stored poses until connected measurements agree as closely as possible.
# The optimized poses are then used to build a more consistent map.

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
    """Docstring."""
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
    """Docstring."""
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
    """Docstring."""
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



def _g2o_optimize(pose_graph, max_iter: int) -> dict:
    """Wrap g2o SE2 pose graph optimiser."""
    solver = g2o.SparseOptimizer()
    solver.set_verbose(False)
    algorithm = g2o.OptimizationAlgorithmLevenberg(
        g2o.BlockSolverX(g2o.LinearSolverDenseX()))
    solver.set_algorithm(algorithm)

    opt_poses = pose_graph.get_optimized_poses()
    for node in pose_graph.get_all_nodes():
        x, y, th = opt_poses[node.node_id]
        v = g2o.VertexSE2()
        v.set_id(node.node_id)
        v.set_estimate(g2o.SE2(x, y, th))
        v.set_fixed(node.node_id == 0)
        solver.add_vertex(v)

    for edge in pose_graph.get_edges():
        e = g2o.EdgeSE2()
        e.set_vertex(0, solver.vertex(edge.from_id))
        e.set_vertex(1, solver.vertex(edge.to_id))
        e.set_measurement(g2o.SE2(edge.dx, edge.dy, edge.dtheta))
        e.set_information(edge.info_matrix)
        solver.add_edge(e)

    solver.initialize_optimization()
    solver.optimize(max_iter)

    result = {}
    for node in pose_graph.get_all_nodes():
        est = solver.vertex(node.node_id).estimate()
        result[node.node_id] = (est.translation()[0],
                                est.translation()[1],
                                est.rotation().angle())
    return result



def optimize_pose_graph(pose_graph, max_iter: int = 20, tol: float = 1e-4) -> dict:
    """Docstring."""
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
