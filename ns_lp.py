"""Linear programs over tripartite no-signaling (NS) distributions.

The variable is P[x, y, z, a, b, c] = P(a, b, c | x, y, z), with C acting as
a candidate copy of B. Supported problems:

* maximize a Bell expression of p_AB, with or without an extension;
* test whether a fixed p_AB has a perfect NS extension;
* minimize the mean mismatch between b and c for a fixed p_AB.
"""
import itertools

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from scenario import bell_value


class TripartiteNS:
    def __init__(self, m, d):
        self.m, self.d = m, d
        self.shape = (m, m, m, d, d, d)
        self.n = int(np.prod(self.shape))
        self.idx = np.arange(self.n).reshape(self.shape)

    # -- constraint builders: each returns a list of (dict{var: coef}, rhs) --
    def _row(self, cells, coef=1.0):
        return {int(k): coef for k in np.ravel(cells)}

    def normalization(self):
        rows = []
        for x, y, z in itertools.product(range(self.m), repeat=3):
            rows.append((self._row(self.idx[x, y, z]), 1.0))
        return rows

    def no_signaling(self):
        m, d, I = self.m, self.d, self.idx
        rows = []
        # Alice: sum_a P(abc|xyz) independent of x
        for x, y, z, b, c in itertools.product(range(1, m), range(m), range(m), range(d), range(d)):
            r = self._row(I[x, y, z, :, b, c])
            for k in np.ravel(I[0, y, z, :, b, c]):
                r[int(k)] = r.get(int(k), 0.0) - 1.0
            rows.append((r, 0.0))
        for x, y, z, a, c in itertools.product(range(m), range(1, m), range(m), range(d), range(d)):
            r = self._row(I[x, y, z, a, :, c])
            for k in np.ravel(I[x, 0, z, a, :, c]):
                r[int(k)] = r.get(int(k), 0.0) - 1.0
            rows.append((r, 0.0))
        for x, y, z, a, b in itertools.product(range(m), range(m), range(1, m), range(d), range(d)):
            r = self._row(I[x, y, z, a, b, :])
            for k in np.ravel(I[x, y, 0, a, b, :]):
                r[int(k)] = r.get(int(k), 0.0) - 1.0
            rows.append((r, 0.0))
        return rows

    def pab_cells(self, x, y, a, b):
        """Cells summing to p_AB(ab|xy) (read at z = 0)."""
        return self.idx[x, y, 0, a, b, :]

    def pac_cells(self, x, z, a, c):
        """Cells summing to p_AC(ac|xz) (read at y = 0)."""
        return self.idx[x, 0, z, a, :, c]

    def ordinary_extension(self):
        rows = []
        for x, y, a, b in itertools.product(range(self.m), range(self.m), range(self.d), range(self.d)):
            r = self._row(self.pab_cells(x, y, a, b))
            for k in np.ravel(self.pac_cells(x, y, a, b)):
                r[int(k)] = r.get(int(k), 0.0) - 1.0
            rows.append((r, 0.0))
        return rows

    def fixed_ab(self, pAB):
        rows = []
        for x, y, a, b in itertools.product(range(self.m), range(self.m), range(self.d), range(self.d)):
            rows.append((self._row(self.pab_cells(x, y, a, b)), float(pAB[x, y, a, b])))
        return rows

    def mismatch_cells(self):
        """Cells with y = z and b != c (any x, a)."""
        cells = []
        for x, y, a, b, c in itertools.product(range(self.m), range(self.m), range(self.d), range(self.d), range(self.d)):
            if b != c:
                cells.append(int(self.idx[x, y, y, a, b, c]))
        return cells

    def mean_mismatch_objective(self):
        """(1/m) sum_y P(b != c | y, z = y), read at x = 0."""
        c = np.zeros(self.n)
        for y, a, b, cc in itertools.product(range(self.m), range(self.d), range(self.d), range(self.d)):
            if b != cc:
                c[self.idx[0, y, y, a, b, cc]] = 1.0 / self.m
        return c

    def bell_objective(self, expr):
        C, offset = expr
        c = np.zeros(self.n)
        for x, y, a, b in itertools.product(range(self.m), range(self.m), range(self.d), range(self.d)):
            for k in np.ravel(self.pab_cells(x, y, a, b)):
                c[k] += C[x, y, a, b]
        return c, offset

    def marginal_ab(self, P):
        return P[:, :, 0].sum(axis=-1)

    def marginal_ac(self, P):
        return P[:, 0, :].sum(axis=-2)

    # -- solver ---------------------------------------------------------------
    def solve(self, c, rows, zero_cells=(), maximize=False, method="highs"):
        A = sp.lil_matrix((len(rows), self.n))
        b = np.zeros(len(rows))
        for i, (r, rhs) in enumerate(rows):
            for k, v in r.items():
                A[i, k] = v
            b[i] = rhs
        bounds = [(0, None)] * self.n
        for k in zero_cells:
            bounds[k] = (0, 0)
        res = linprog(-c if maximize else c, A_eq=A.tocsr(), b_eq=b, bounds=bounds, method=method)
        if res.status != 0:
            return None, {"status": res.message}
        val = -res.fun if maximize else res.fun
        return res.x.reshape(self.shape), {"status": "optimal", "objective": float(val)}


def max_bell_ns(expr, mode):
    """mode: 'ns' (no extension), 'ext' (ordinary), or 'perfect'."""
    C, _ = expr
    m, d = C.shape[0], C.shape[2]
    T = TripartiteNS(m, d)
    rows = T.normalization() + T.no_signaling()
    zeros = ()
    if mode in ("ext", "perfect"):
        rows += T.ordinary_extension()
    if mode == "perfect":
        zeros = T.mismatch_cells()
    c, offset = T.bell_objective(expr)
    P, info = T.solve(c, rows, zero_cells=zeros, maximize=True)
    if P is not None:
        info["objective"] += offset
        pAB = T.marginal_ab(P)
        info["check_bell_value"] = bell_value(pAB, expr)
        info["max_marginal_difference"] = float(np.max(np.abs(pAB - T.marginal_ac(P))))
    return P, info


def min_mismatch_ns(pAB, perfect_only=False):
    """Minimum mean mismatch over NS extensions of a fixed binary p_AB.

    With perfect_only=True, solves the feasibility problem with b = c imposed
    whenever y = z, using a simplex method so that the solution is a vertex.
    """
    m, d = pAB.shape[0], pAB.shape[2]
    T = TripartiteNS(m, d)
    rows = T.normalization() + T.no_signaling() + T.ordinary_extension() + T.fixed_ab(pAB)
    if perfect_only:
        P, info = T.solve(np.zeros(T.n), rows, zero_cells=T.mismatch_cells(), method="highs-ds")
    else:
        P, info = T.solve(T.mean_mismatch_objective(), rows)
    return T, rows, P, info
