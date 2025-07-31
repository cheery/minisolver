import numpy as np
import random
import math

def approximate_jacobian(f, eps=1e-8):
    def jac(xi):
        fi = f(xi)
        m = fi.size
        n = xi.size
        J = np.zeros((m, n), float)
        for j in range(n):
            dx = np.zeros(n)
            dx[j] = eps
            J[:, j] = (f(xi + dx) - fi) / eps
        return J
    return jac

def null_space(A, rcond=None):
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = vh[num:,:].T.conj()
    return Q

def analyze(J, tol=1e-6):
    U, S, Vt = np.linalg.svd(J, full_matrices=True)
    # Count singular values > tol → rank
    rank = np.sum(S > tol)
    # Nullity = #columns - rank
    n = J.shape[1]
    nullity = n - rank
    
    movable = []
    # Nullspace basis = last `nullity` columns of V = rows of Vt.T
    if nullity > 0:
        nullspace = Vt.T[:, rank:]
        # Find which variable indices have any significant component
        # across the nullspace basis vectors
        for i in range(n):
            # Compute the L2 norm of row i of `nullspace`
            comp_norm = np.linalg.norm(nullspace[i, :])
            if comp_norm > tol:
                movable.append(i)
    else:
        nullspace = np.zeros((n, 0))
    
    return int(nullity), movable

class NoConvergence(Exception):
    pass

def solve_soft(f, f_jac, g, g_jac, x, tol=1e-6, max_iterations=100, nudge=1e-2):
    x = constrain(g, g_jac, x, tol, 10, nudge)[0]
    x, f_norm = constrain(f, f_jac, x, tol, max_iterations, nudge)
    if f_norm >= tol:
        raise NoConvergence
    try:
        n = x.size
        fi = f(x)
        gi = g(x)
        f_norm = np.linalg.norm(fi, ord=np.inf)
        g_norm = np.linalg.norm(gi, ord=np.inf)
        x_best = x.copy()
        g_best = g_norm
        for k in range(25):
            if f_norm < tol and g_norm < tol:
                return x
            J_h = f_jac(x)
            J_s = g_jac(x)
            if fi.size > 0:
                dx = perform_solver_step(fi, J_h)
                x += dx
                N = null_space(J_h)
                if N.size > 0:
                    # Project soft constraints into null space
                    J_s_null = J_s @ N
                    # Compute step in null space (minimize soft constraints)
                    gi = gi + J_s.dot(dx)
                    dx_soft = N @ perform_solver_step(gi, J_s_null)
                    x += dx_soft
            else:
                x += perform_solver_step(gi, J_s)

            f_norm = np.linalg.norm(fi := f(x), ord=np.inf)
            g_norm = np.linalg.norm(gi := g(x), ord=np.inf)
            if f_norm < tol and g_norm < g_best:
                x_best[:] = x
    except np.linalg.LinAlgError as e:
        import traceback
        traceback.print_exc()
    return x_best

def constrain(f, jac, x, tol=1e-6, max_iterations=100, nudge=1e-2):
    n = x.size
    fi = f(x)
    f_norm = np.linalg.norm(fi, ord=np.inf)
    f_norm_prev = f_norm
    for i in range(max_iterations):
        if f_norm < tol:
            return x, f_norm
        x += perform_solver_step(fi, jac(x))

        fi = f(x)
        f_norm = np.linalg.norm(fi, ord=np.inf)
        if f_norm >= f_norm_prev:
            x += np.random.uniform(-nudge, +nudge, n)
        f_norm_prev = f_norm
    return x, f_norm

def perform_solver_step(fi, J, lambda_factor=10.0, max_lambda=1e10):
    return np.linalg.lstsq(J, -fi)[0]

def perform_solver_step2(fi, J, f, x):
    alpha0 = 1.0
    rho    = 0.5
    c      = 1e-4
    dx = np.linalg.lstsq(J, -fi)[0]
    f_norm = np.linalg.norm(fi)
    alpha = alpha0
    while True:
        x_new = x + alpha * dx
        if np.linalg.norm(f(x_new)) <= (1 - c * alpha) * f_norm:
            break
        alpha *= rho
    return alpha*dx

    # This would seem interesting but
    # the condition of the matrix explodes as the result.
    # Adaptive Levenberg-Marquardt
    n = J.shape[1]
    lm_lambda = 1e-3
    best_dx = None
    best_norm = np.inf
    
    print("COND (ORIG)", np.linalg.cond(J))
    while lm_lambda < max_lambda:
        J_lm = J.T @ J + lm_lambda * np.eye(n)
        grad = J.T @ fi
        print("COND", np.linalg.cond(J_lm))

        #try:
        dx = np.linalg.solve(J_lm, -grad)
        #except np.linalg.LinAlgError:
        #dx = np.linalg.lstsq(J_lm, -grad, rcond=None)[0]
        
        # Accept step if it reduces residual
        residual_norm = np.linalg.norm(J @ dx + fi, 2)
        if residual_norm < best_norm:
            best_dx = dx
            best_norm = residual_norm
            lm_lambda /= lambda_factor
            break
        else:
            lm_lambda *= lambda_factor
    
    return best_dx if best_dx is not None else dx

# Add this in if needed.
# Armijo
#         alpha = alpha0
#         if damping:
#             f_norm = np.linalg.norm(fi)
#             while True:
#                 x_new = x + alpha * dx
#                 if np.linalg.norm(f(x_new)) <= (1 - c * alpha) * f_norm:
#                     break
#                 alpha *= rho
#         step = alpha * dx if damping else dx
