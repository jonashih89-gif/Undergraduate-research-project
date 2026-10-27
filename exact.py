"""Exact (rational) verification of the main claims.

1. A rational three-qubit realization: integer state vector q and integer
   measurement directions. Its p_AB is computed exactly, p_AB = p_AC is
   checked exactly, and the Bell value is an exact fraction.
2. A lower bound on the mean mismatch of every almost-quantum (hence every
   quantum) extension of that p_AB, obtained from a rounded dual solution.
   For any Z = L L^T >= 0, u >= 0 and multipliers mu, every feasible moment
   vector v (with vec(G) = M v, G >= 0, T v >= 0, A v = c) satisfies
       f.v = r.v + <Z, G> + u.(T v) + mu.c  >=  mu.c - sum_k |r_k|,
   where r = f - M^T vec(Z) - T^T u - A^T mu and |v_k| <= 1 for every moment.
3. An exact perfect no-signaling extension of the same p_AB.
"""
import itertools
from fractions import Fraction

import numpy as np
import scipy.linalg

from quantum import SWAP_BC


# ---------------------------------------------------------------------------
# 1. Rational quantum example
# ---------------------------------------------------------------------------
def integer_direction(theta, scale=10**5):
    return np.array([int(round(np.cos(theta / 2) * scale)), int(round(np.sin(theta / 2) * scale))], dtype=object)


def int_outcome_matrices(v):
    """Integer numerators of the two outcome projectors; both share denominator |v|^2."""
    vp = np.array([-v[1], v[0]], dtype=object)
    return [np.outer(v, v), np.outer(vp, vp)], int(v @ v)


def rationalize_state(psi, scale=10**6):
    s = (np.real(psi) + SWAP_BC @ np.real(psi)) / 2
    q = np.array([int(round(t * scale)) for t in s], dtype=object)
    assert all(q[i] == (SWAP_BC @ q)[i] for i in range(8))
    return q


def exact_pair_marginal(q, dirA, dirB, which="AB"):
    """Exact p(ab|xy) for the pair AB (or AC, with B's directions on C)."""
    m = len(dirA)
    I2 = np.array([[1, 0], [0, 1]], dtype=object)
    qq = int(q @ q)
    p = np.empty((m, m, 2, 2), dtype=object)
    for x, y in itertools.product(range(m), repeat=2):
        MA, nA = int_outcome_matrices(dirA[x])
        MB, nB = int_outcome_matrices(dirB[y])
        for a, b in itertools.product(range(2), repeat=2):
            op = (np.kron(np.kron(MA[a], MB[b]), I2) if which == "AB"
                  else np.kron(np.kron(MA[a], I2), MB[b]))
            p[x, y, a, b] = Fraction(int(q @ op @ q), qq * nA * nB)
    return p


def exact_bell_value(p, expr):
    C, offset = expr
    val = Fraction(offset).limit_denominator(10**6)
    for idx in itertools.product(*map(range, C.shape)):
        coef = Fraction(C[idx]).limit_denominator(10**6)
        if coef:
            val += coef * p[idx]
    return val


# ---------------------------------------------------------------------------
# 2. Exact dual certificate for the almost-quantum mismatch minimum
# ---------------------------------------------------------------------------
def _int_rows(rows, scale):
    """Sparse rows {k: v} -> {k: int(v*scale)}, checking that scaling is exact."""
    out = []
    for row in rows:
        r = {}
        for k, v in row.items():
            iv = int(round(v * scale))
            assert abs(iv - v * scale) < 1e-9, "non-integer coefficient"
            if iv:
                r[k] = iv
        out.append(r)
    return out


def _exact_rhs(tags, pAB):
    rhs = []
    for t in tags:
        if t == "identity":
            rhs.append(Fraction(1))
        elif "=" in t:                                  # extension equalities
            rhs.append(Fraction(0))
        elif t.startswith("pA("):
            x = int(t[3])
            rhs.append(pAB[x, 0, 0, 0] + pAB[x, 0, 0, 1])
        elif t.startswith("pB("):
            y = int(t[3])
            rhs.append(pAB[0, y, 0, 0] + pAB[0, y, 1, 0])
        elif t.startswith("pAB("):
            rhs.append(pAB[int(t[4]), int(t[5]), 0, 0])
        else:
            raise ValueError(t)
    return rhs


def certify_mismatch_lower_bound(R, pAB_exact, bits=30):
    """Rigorous lower bound on the mean mismatch over the relaxation R.

    R must be a solved almost-quantum MomentRelaxation built by
    moments.min_mismatch. In the almost-quantum moment matrix every diagonal
    entry is the expectation of a product of commuting projectors, so
    positivity forces every moment into [-1, 1].
    """
    assert R.level == "AQ"
    m, K = R.m, R.K
    cons = R._last["constraints"]
    Z = np.array(cons[0].dual_value)
    u = np.array(cons[2].dual_value).ravel()

    s2 = 2 ** (2 * bits)
    D = m * s2                                          # common denominator
    f_int = _int_rows([R.mean_mismatch_row()], m)[0]    # f * m
    T_int = _int_rows(R.ineq_rows, 1)
    A_int = _int_rows(R.eq_rows, 1)
    rhs = _exact_rhs(R.eq_tags, pAB_exact)

    # Z -> L L^T with integer L (exactly positive semidefinite)
    w, V = np.linalg.eigh((Z + Z.T) / 2)
    keep = w > 0
    L = V[:, keep] * np.sqrt(w[keep])
    L_int = np.vectorize(lambda t: int(round(t * 2**bits)), otypes=[object])(L)
    Z_int = L_int.dot(L_int.T)                          # Z = Z_int / s2
    u_int = [int(round(max(t, 0.0) * s2)) for t in u]   # u = u_int / s2

    # M^T vec(Z): sum of Z over all entries holding moment k
    Mc = R.M.tocoo()
    Zflat = Z_int.reshape(-1, order="F")
    MtZ = [0] * K
    for f, k in zip(Mc.row, Mc.col):
        MtZ[k] += int(Zflat[f])

    # choose mu by least squares on the stationarity residual
    target = np.zeros(K)
    for k, v in f_int.items():
        target[k] += v / m
    target -= np.array(MtZ, dtype=float) / s2
    for ui, row in zip(u_int, T_int):
        for k, v in row.items():
            target[k] -= v * ui / s2
    A = np.zeros((len(A_int), K))
    for i, row in enumerate(A_int):
        for k, v in row.items():
            A[i, k] = v
    mu, *_ = np.linalg.lstsq(A.T, target, rcond=None)
    mu_int = [int(round(t * D)) for t in mu]            # mu = mu_int / D

    # exact residual r * D = f*D - M^T vec(Z)*D - T^T u*D - A^T mu*D
    r = [0] * K
    for k, v in f_int.items():
        r[k] += v * s2
    for k in range(K):
        r[k] -= m * MtZ[k]
    for ui, row in zip(u_int, T_int):
        if ui:
            for k, v in row.items():
                r[k] -= m * ui * v
    for mi, row in zip(mu_int, A_int):
        if mi:
            for k, v in row.items():
                r[k] -= mi * v

    mu_c = sum((Fraction(mi) * c for mi, c in zip(mu_int, rhs)), Fraction(0)) / D
    res_l1 = Fraction(sum(abs(t) for t in r), D)
    return {"dual_objective": mu_c, "residual_l1": res_l1, "lower_bound": mu_c - res_l1,
            "num_moments": K, "bits": bits}


# ---------------------------------------------------------------------------
# 3. Exact perfect no-signaling extension
# ---------------------------------------------------------------------------
def solve_exact(M, b):
    """Solve a square nonsingular system exactly with Fractions."""
    k = len(b)
    A = [[Fraction(M[i][j]) for j in range(k)] + [Fraction(b[i])] for i in range(k)]
    for col in range(k):
        piv = next(r for r in range(col, k) if A[r][col] != 0)
        A[col], A[piv] = A[piv], A[col]
        pv = A[col][col]
        A[col] = [t / pv for t in A[col]]
        for r in range(k):
            if r != col and A[r][col] != 0:
                fac = A[r][col]
                A[r] = [s - fac * t for s, t in zip(A[r], A[col])]
    return [A[i][k] for i in range(k)]


def exact_ns_extension(T, rows, P_float, pAB_exact, tol=1e-10):
    """Turn a vertex solution of the perfect-extension LP into an exact one."""
    support = [int(k) for k in np.flatnonzero(P_float.reshape(-1) > tol)]
    # exact rhs: rows built by ns_lp in order normalization, NS, extension, fixed AB
    m, d = T.m, T.d
    fixed = [pAB_exact[x, y, a, b] for x, y, a, b in itertools.product(range(m), range(m), range(d), range(d))]
    n_other = len(rows) - len(fixed)
    rhs = [Fraction(int(round(r[1]))) for r in rows[:n_other]] + fixed
    E = np.zeros((len(rows), len(support)))
    for i, (row, _) in enumerate(rows):
        for j, k in enumerate(support):
            E[i, j] = row.get(k, 0.0)
    _, _, piv = scipy.linalg.qr(E.T, pivoting=True)
    rank = np.linalg.matrix_rank(E)
    if rank != len(support):
        raise RuntimeError("support is not a vertex; cannot solve uniquely")
    sel = sorted(piv[:rank])
    sol = solve_exact([[int(round(E[i, j])) for j in range(rank)] for i in sel], [rhs[i] for i in sel])
    P = [Fraction(0)] * T.n
    for k, v in zip(support, sol):
        P[k] = v
    # verify every constraint exactly
    worst = Fraction(0)
    for (row, _), c in zip(rows, rhs):
        lhs = sum((int(round(v)) * P[k] for k, v in row.items()), Fraction(0))
        worst = max(worst, abs(lhs - c))
    mism = sum((P[k] for k in T.mismatch_cells()), Fraction(0))
    return {"P": P, "num_equalities": len(rows), "num_probabilities": T.n,
            "support_size": len(support), "max_equality_violation": worst,
            "min_probability": min(P), "total_mismatch_mass": mism}
