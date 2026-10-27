"""Bipartite Bell-scenario utilities.

Conventions
-----------
A bipartite behavior is an array ``p[x, y, a, b] = p(a, b | x, y)``.
A Bell expression is a pair ``(C, offset)`` with value
``sum(C * p) + offset``. All expressions used here are normalized so that
their local bound is 0.
"""
import itertools

import numpy as np
from scipy.optimize import linprog


# ---------------------------------------------------------------------------
# Bell expressions
# ---------------------------------------------------------------------------
def imm22(m):
    """Collins-Gisin I_mm22 inequality (m settings, 2 outcomes), local bound 0.

    I = -pA(0|0) - sum_y (m-1-y) pB(0|y) + sum_{x,y} J[x,y] p(00|x,y),
    with J = 1 if x+y < m, J = -1 if x+y = m, and J = 0 otherwise.
    m = 2 gives CHSH and m = 3 gives I3322.
    """
    C = np.zeros((m, m, 2, 2))
    C[0, 0, 0, :] -= 1.0                       # pA(0|0), read at y = 0
    for y in range(m):
        C[0, y, :, 0] -= (m - 1 - y)           # pB(0|y), read at x = 0
    for x, y in itertools.product(range(m), repeat=2):
        if x + y < m:
            C[x, y, 0, 0] += 1.0
        elif x + y == m:
            C[x, y, 0, 0] -= 1.0
    return C, 0.0


def cglmp(d):
    """CGLMP inequality (2 settings, d outcomes), rescaled to local bound 0.

    With I_d the standard CGLMP expression (local bound 2, no-signaling
    maximum 4), this returns (d-1)/(2d) * (I_d - 2).
    """
    C = np.zeros((2, 2, d, d))

    def add(x, y, rel, shift, w):
        # rel = "ab": P(A_x = B_y + shift); rel = "ba": P(B_y = A_x + shift)
        for a, b in itertools.product(range(d), repeat=2):
            hit = (a - b - shift) % d == 0 if rel == "ab" else (b - a - shift) % d == 0
            if hit:
                C[x, y, a, b] += w

    for k in range(d // 2):
        w = 1.0 - 2.0 * k / (d - 1)
        add(0, 0, "ab", k, w)
        add(1, 0, "ba", k + 1, w)
        add(1, 1, "ab", k, w)
        add(0, 1, "ba", k, w)
        add(0, 0, "ab", -k - 1, -w)
        add(1, 0, "ba", -k, -w)
        add(1, 1, "ab", -k - 1, -w)
        add(0, 1, "ba", -k - 1, -w)
    s = (d - 1) / (2 * d)
    return s * C, -2.0 * s


def bell_value(p, expr):
    C, offset = expr
    return float(np.sum(C * p) + offset)


def cg_coefficients(expr):
    """Rewrite a binary-outcome expression in Collins-Gisin moments.

    Returns (const, cA[x], cB[y], cAB[x, y]) such that the value equals
    const + sum cA <A_x> + sum cB <B_y> + sum cAB <A_x B_y>, where <A_x> is
    the probability of outcome 0 for setting x.
    """
    C, offset = expr
    mA, mB = C.shape[:2]
    const, cA, cB, cAB = offset, np.zeros(mA), np.zeros(mB), np.zeros((mA, mB))
    for x, y in itertools.product(range(mA), range(mB)):
        c00, c01, c10, c11 = C[x, y, 0, 0], C[x, y, 0, 1], C[x, y, 1, 0], C[x, y, 1, 1]
        # p00 = AB, p01 = A - AB, p10 = B - AB, p11 = 1 - A - B + AB
        const += c11
        cA[x] += c01 - c11
        cB[y] += c10 - c11
        cAB[x, y] += c00 - c01 - c10 + c11
    return const, cA, cB, cAB


# ---------------------------------------------------------------------------
# Local polytope
# ---------------------------------------------------------------------------
def deterministic_behaviors(mA, mB, d):
    """All deterministic bipartite behaviors, shape (num, mA, mB, d, d)."""
    out = []
    for fa in itertools.product(range(d), repeat=mA):
        for fb in itertools.product(range(d), repeat=mB):
            D = np.zeros((mA, mB, d, d))
            for x, y in itertools.product(range(mA), range(mB)):
                D[x, y, fa[x], fb[y]] = 1.0
            out.append(D)
    return np.array(out)


def local_bound(expr):
    C, offset = expr
    mA, mB, d, _ = C.shape
    D = deterministic_behaviors(mA, mB, d)
    return float(np.max(np.tensordot(D, C, axes=4)) + offset)


def local_visibility(p):
    """Largest v such that v*p + (1-v)*uniform lies in the local polytope.

    p is local if and only if the returned value is at least 1.
    """
    mA, mB, d, _ = p.shape
    D = deterministic_behaviors(mA, mB, d).reshape(-1, mA * mB * d * d).T
    u = np.full(mA * mB * d * d, 1.0 / d**2)
    pv = p.reshape(-1)
    n = D.shape[1]
    # variables: q (n), v ; maximize v
    A_eq = np.hstack([D, -(pv - u)[:, None]])
    b_eq = u.copy()
    A_eq = np.vstack([A_eq, np.r_[np.ones(n), 0.0]])
    b_eq = np.r_[b_eq, 1.0]
    c = np.r_[np.zeros(n), -1.0]
    res = linprog(c, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0, None)] * n + [(None, 100.0)], method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    return float(-res.fun)
