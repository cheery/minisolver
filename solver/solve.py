from .expressions import NonZero, Eq, SoftEq, print_system, all_variables, JustContext, VectoredContext, EmptyContext, dual, cells
import numpy as np
import random
import math

# TODO: examine stochastic gradient descent

def setup(entities, context, known, knownvec):
    hard = set()
    soft = set()
    for entity in entities:
        for obj in entity.equations:
            if isinstance(obj, (NonZero, Eq)):
                hard.add(obj)
            elif isinstance(obj, SoftEq):
                soft.add(obj)
    if True:
        print_system(hard | soft)
    cx = EmptyContext(context, {})
    hard = set(obj.evaluate(cx) for obj in hard)
    soft = set(obj.evaluate(cx) for obj in soft)

    each, x0 = make_vectored_context(hard | soft, context, known)
    wrap = lambda x: VectoredContext(x0, x0.variables, {}, x)

    memo = {}
    dhard = list(x.diff(memo, x0.variables) for x in hard)
    dsoft = list(x.diff(memo, x0.variables) for x in soft)

    LH = len(hard)
    hard = cells(hard, x0.variables, known)
    LS = len(soft)
    g_w  = np.array([a.weight for a in soft], float)
    soft = cells(soft, x0.variables, known)

    DLH = len(dhard)
    dhard = cells(dhard, x0.variables, known)
    DLS = len(dsoft)
    dsoft = cells(dsoft, x0.variables, known)

    def f(x):
        s = np.zeros(len(hard), float)
        xs = x, x0.x, s, knownvec
        for i, cell in enumerate(hard):
            s[i] = cell(xs)
        return s[-LH:]
    def f_jac(x):
        m = DLH
        n = len(x)
        out = np.zeros((m,n), float)
        s = np.zeros(len(dhard), float)
        xs = x, x0.x, s, knownvec, out
        for i, cell in enumerate(dhard):
            s[i] = cell(xs)
        return out
    def g(x):
        s = np.zeros(len(soft), float)
        xs = x, x0.x, s, knownvec
        for i, cell in enumerate(soft):
            s[i] = cell(xs)
        return s[-LS:]
    def g_jac(x):
        m = DLS
        n = len(x)
        out = np.zeros((m,n), float)
        s = np.zeros(len(dsoft), float)
        xs = x, x0.x, s, knownvec, out
        for i, cell in enumerate(dsoft):
            s[i] = cell(xs)
        return out
    #f_jac = approximate_jacobian(f)
    #g_jac = approximate_jacobian(g)
    return f, f_jac, g, g_jac, g_w, x0.x.copy(), wrap

def make_vectored_context(system, context, known):
    each, symbols = all_variables(system)
    variables = {}
    vector  = []
    for sym in symbols:
        if sym in known:
            continue
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
    lam = 0.01
    def H(x):
        fi = f(x)
        gi = g(x) * g_w
        return np.dot(fi, fi) + np.dot(gi, gi)
    def Q(A, B):
        A_T = A.T
        h = np.diag(np.random.uniform(0, lam, A.shape[1]))
        return (np.linalg.pinv(A_T @ A + h) @ A_T @ B)
    def R(A, B, w):
        W = np.diag(w)
        A_T = A.T
        Aw = A_T @ W @ A + np.eye(A.shape[1]) * (lam * 100)
        Bw = A_T @ W @ B
        return (np.linalg.pinv(Aw) @ Bw)
    fi = f(x)
    g_norm = g_norm_prev = np.linalg.norm(gi := g(x), ord=np.inf)
    for k in range(max_iterations):
        if g_norm < tol:
            return enforce(f, f_jac, x, tol, 20, nudge)
        Sx = np.maximum(np.abs(x), 1e-2)
        J_h = f_jac(x)
        J_s = g_jac(x)
        #scale = 1 / np.maximum(np.maximum(np.max(J_h, axis=0), np.max(J_s, axis=0)), tol)
        if J_h.size > 0:
            dx = Q(J_h * Sx[np.newaxis,:], -fi) * Sx
            N = null_space(J_h)
            if N.size > 0:
                # Project soft constraints into null space
                J_s_null = J_s @ N
                gi = gi + J_s @ dx
                dx_soft = N @ R(J_s_null, -gi, g_w)
                dx += dx_soft
            if H(x+dx) + tol < H(x):
                lam /= 10
            else:
                lam *= 10
            while H(x+dx*0.5) < H(x+dx):
                dx *= 0.5
            x += dx
        else:
            dx = R(J_s, -gi, g_w)
            if H(x+dx) + tol < H(x):
                lam /= 10
            else:
                lam *= 10
            while H(x+dx*0.5) < H(x+dx):
                dx *= 0.5
            x += dx
        fi = f(x)
        g_norm = np.linalg.norm(gi := g(x), ord=np.inf)
        if g_norm_prev - g_norm < tol:
            return enforce(f, f_jac, x, tol, 20, nudge)
        g_norm_prev = g_norm
    return enforce(f, f_jac, x, tol, 20, nudge)

def enforce(f, f_jac, x, tol=1e-6, max_iterations=100, nudge=1e-2):
    x, f_norm = constrain(f, f_jac, x, tol, max_iterations, nudge)
    if f_norm < tol:
        return x
    else:
        print("X =", x)
        print("Y =", f(x))
        print("J =", f_jac(x))
        raise NoConvergence

def constrain(f, jac, x, tol=1e-6, max_iterations=100, nudge=1e-2):
    max_iterations = 100
    def F(x):
        fi = f(x)
        return np.dot(fi, fi)
    n = x.size
    fi = f(x)
    f_norm = np.linalg.norm(fi, ord=np.inf)
    lam = 0.01
    for i in range(max_iterations):
        if f_norm < tol:
            return x, f_norm
        Sx = np.maximum(np.abs(x), 1e-2)
        J = jac(x)
        J = J * Sx[np.newaxis,:]
        J_T = J.T
        A = J_T @ J + np.diag(np.random.uniform(0, lam, J.shape[1]))
        B = J_T @ -fi
        dx = (np.linalg.pinv(A) @ B) * Sx
        if F(x+dx) + tol < F(x):
            lam /= 10
        else:
            lam *= 10
        dx *= 8
        while F(x+dx*0.5) < F(x+dx):
            dx *= 0.5
        x += dx
        fi = f(x)
        f_norm = np.linalg.norm(fi, ord=np.inf)
    return x, f_norm
