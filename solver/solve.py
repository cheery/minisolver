from .expressions import NonZero, Eq, SoftEq, print_system, all_variables, JustContext, VectoredContext, dual
import numpy as np
import random
import math

def setup(entities, context):
    hard = []
    soft = []
    for entity in entities:
        for obj in entity.equations:
            if isinstance(obj, (NonZero, Eq)):
                hard.append(obj)
            elif isinstance(obj, SoftEq):
                soft.append(obj)
    if False:
        print_system(hard + soft)
    each, x0 = make_vectored_context(hard + soft, context)
    wrap = lambda x: VectoredContext(x0, x0.variables, {}, x)
    def f(x):
        x = wrap(x)
        return np.array([a.evaluate(x) for a in hard], float)
    def f_jac(x):
        m = len(hard)
        n = len(x)
        x = wrap(x)
        variables = {}
        for sym in x.variables:
            variables[sym] = dual(sym, x[sym])
        y = JustContext(None, variables, {})
        out = np.zeros((m,n), float)
        for i, a in enumerate(hard):
            if isinstance(a, Eq):
                f = (a.lhs - a.rhs).evaluate(y)
                for sym, d in f.partials.items():
                    out[i, x.variables[sym]] += d
        return out
    def g(x):
        x = wrap(x)
        return np.array([a.evaluate(x) for a in soft], float)
    def g_jac(x):
        m = len(soft)
        n = len(x)
        x = wrap(x)
        variables = {}
        for sym in x.variables:
            variables[sym] = dual(sym, x[sym])
        y = JustContext(None, variables, {})
        out = np.zeros((m,n), float)
        for i, a in enumerate(soft):
            if isinstance(a, SoftEq):
                f = (a.lhs - a.rhs).evaluate(y)
                for sym, d in f.partials.items():
                    out[i, x.variables[sym]] += d
        return out
    g_w = np.array([a.weight for a in soft], float)
    return f, f_jac, g, g_jac, g_w, x0.x.copy(), wrap

def make_vectored_context(system, context):
    each, symbols = all_variables(system)
    variables = {}
    vector  = []
    for sym in symbols:
        variables[sym] = len(vector)
        try:
            vector.append(context[sym])
        except KeyError:
            vector.append(0.0)
    return each, VectoredContext(None, variables, {}, np.array(vector, float))

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

def truncated_pinv(A, rcond=None):
    """Compute pseudoinverse with truncated SVD"""
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(A.shape)
    tol = np.amax(s) * rcond
    mask = s > tol
    s_inv = np.zeros_like(s)
    s_inv[mask] = 1.0 / s[mask]
    return Vh.T @ (s_inv[:, None] * U.T)

def null_space(A, rcond=None):
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = vh[num:,:].T.conj()
    return Q

def analyze(J, rcond=None):
    k = J.shape[1]
    u, s, vh = np.linalg.svd(J, full_matrices=True)
    # Count singular values > tol → rank
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    # Nullity = #columns - rank
    if s.size == 0:
        tol = 0
        rank = 0
        nullity = k
    else:
        tol = np.amax(s) * rcond
        rank = np.sum(s > tol, dtype=int)
        nullity = k - rank
    
    movable = []
    # Nullspace basis = last `nullity` columns of V = rows of Vt.T
    if nullity > 0:
        nullspace = vh[rank:,:].T.conj()
        #nullspace = Vt.T[:, rank:]
        # Find which variable indices have any significant component
        # across the nullspace basis vectors
        for i in range(k):
            # Compute the L2 norm of row i of `nullspace`
            comp_norm = np.linalg.norm(nullspace[i, :])
            if comp_norm > tol:
                movable.append(i)
    else:
        nullspace = np.zeros((k, 0))
    
    return int(nullity), movable

class NoConvergence(Exception):
    pass

def solve_soft(f, f_jac, g, g_jac, g_w, x, tol=1e-6, max_iterations=100, nudge=1e-2):
    def H(x):
        return np.linalg.norm(f(x)) + np.linalg.norm(g(x) * g_w)
    def Q(A, B):
        A_T = A.T
        return np.linalg.pinv(A_T @ A) @ A_T @ B
    def R(A, B, w):
        W = np.diag(w)
        A_T = A.T
        Aw = A_T @ W @ A + np.eye(A.shape[1])*0.01
        Bw = A_T @ W @ B
        return np.linalg.pinv(Aw) @ Bw
    fi = f(x)
    g_norm = g_norm_prev = np.linalg.norm(gi := g(x), ord=np.inf)
    for k in range(max_iterations):
        if g_norm < tol:
            return enforce(f, f_jac, x, tol, max_iterations, nudge)
        J_h = f_jac(x)
        J_s = g_jac(x)
        if J_h.size > 0:
            dx = Q(J_h, -fi)
            N = null_space(J_h)
            if N.size > 0:
                # Project soft constraints into null space
                J_s_null = J_s @ N
                gi = gi + J_s @ dx
                dx_soft = N @ R(J_s_null, -gi, g_w)
                dx += dx_soft
            while H(x+dx*0.5) < H(x+dx):
                dx *= 0.5
            x += dx
        else:
            dx = R(J_s, -gi, g_w)
            while H(x+dx*0.5) < H(x+dx):
                dx *= 0.5
            x += dx
        fi = f(x)
        g_norm = np.linalg.norm(gi := g(x), ord=np.inf)
        if g_norm_prev - g_norm < tol:
            return enforce(f, f_jac, x, tol, max_iterations, nudge)
        g_norm_prev = g_norm
    return enforce(f, f_jac, x, tol, max_iterations, nudge)

def enforce(f, f_jac, x, tol=1e-6, max_iterations=100, nudge=1e-2):
    x, f_norm = constrain(f, f_jac, x, tol, max_iterations, nudge)
    if f_norm < tol:
        return x
    else:
        raise NoConvergence

def constrain(f, jac, x, tol=1e-6, max_iterations=100, nudge=1e-2):
    def F(x):
        return np.linalg.norm(f(x))
    n = x.size
    fi = f(x)
    f_norm = np.linalg.norm(fi, ord=np.inf)
    f_norm_prev = f_norm
    for i in range(max_iterations):
        if f_norm < tol:
            return x, f_norm
        J = jac(x)
        J_T = J.T
        A = J_T @ J
        B = J_T @ -fi
        dx = np.linalg.pinv(A) @ B
        while F(x+dx*0.5) < F(x+dx):
            dx *= 0.5
        x += dx
        fi = f(x)
        f_norm = np.linalg.norm(fi, ord=np.inf)
        if f_norm_prev - f_norm < tol:
            x += np.random.uniform(-nudge, +nudge, n)
        f_norm_prev = f_norm
    return x, f_norm
