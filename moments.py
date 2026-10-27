"""Moment-matrix relaxations of the tripartite quantum set.

Each party (0 = A, 1 = B, 2 = C) has m binary measurements. We use the
projector P_x onto outcome 0 for each setting; outcome 1 is I - P_x.
An operator word is a tuple of (party, setting) pairs. Projectors of
different parties commute and every projector is idempotent.

Supported word sets
-------------------
* level = k (int): all products of at most k projectors (NPA level k);
* level = "AQ": all products with at most one projector per party
  (the almost-quantum set, NPA level 1+AB+AC+BC+ABC).

The moment matrix G[i, j] = <w_i^dagger w_j> is parameterized by one real
variable per distinct moment and is taken to be real and
symmetric. This loses nothing: for any feasible complex moment matrix, its
real part is feasible and gives the same probabilities.
"""
import itertools

import numpy as np
import scipy.sparse as sp
import cvxpy as cp

from scenario import cg_coefficients


# ---------------------------------------------------------------------------
# Word algebra
# ---------------------------------------------------------------------------
def canonical(word):
    """Commute parties into the order A, B, C and merge repeated projectors."""
    per_party = {0: [], 1: [], 2: []}
    for party, s in word:
        seq = per_party[party]
        if seq and seq[-1] == s:
            continue
        seq.append(s)
    return tuple((p, s) for p in (0, 1, 2) for s in per_party[p])


def key(word):
    """Identify a word with its adjoint (valid for real moment matrices)."""
    w = canonical(word)
    r = canonical(tuple(reversed(w)))
    return min(w, r)


def word_set(m, level):
    letters = [(p, s) for p in range(3) for s in range(m)]
    if level == "AQ":
        words = [()]
        for subset in itertools.product(range(m + 1), repeat=3):
            w = tuple((p, s) for p, s in enumerate(subset) if s < m)
            if w:
                words.append(w)
        return sorted(set(words), key=lambda w: (len(w), w))
    words, frontier = {()}, {()}
    for _ in range(level):
        new = set()
        for w in frontier:
            for l in letters:
                c = canonical(w + (l,))
                if c not in words:
                    new.add(c)
        words |= new
        frontier = new
    return sorted(words, key=lambda w: (len(w), w))


# ---------------------------------------------------------------------------
# Relaxation
# ---------------------------------------------------------------------------
class MomentRelaxation:
    def __init__(self, m, level):
        self.m, self.level = m, level
        self.words = word_set(m, level)
        n = self.n = len(self.words)
        # one variable per distinct moment; vec(G) = M v
        self.index = {}               # key -> moment index k
        rows, cols = [], []
        for i in range(n):
            wi_dag = tuple(reversed(self.words[i]))
            for j in range(n):
                k = key(wi_dag + self.words[j])
                if k not in self.index:
                    self.index[k] = len(self.index)
                rows.append(self.flat(i, j))
                cols.append(self.index[k])
        self.K = len(self.index)
        self.M = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n * n, self.K))
        # linear data collected as sparse rows over the moment vector v
        self.eq_rows, self.eq_rhs, self.eq_tags = [], [], []
        self.ineq_rows, self.ineq_rhs, self.ineq_tags = [], [], []   # row . v >= rhs

    # -- indexing -------------------------------------------------------------
    def flat(self, i, j):
        return i + j * self.n

    def moment(self, word):
        """Index of the moment variable <word>."""
        k = key(word)
        if k not in self.index:
            raise KeyError(f"word {word} not in the moment matrix")
        return self.index[k]

    def A(self, x):
        return (0, x)

    def B(self, y):
        return (1, y)

    def C(self, z):
        return (2, z)

    def lin(self, terms):
        """Sparse row from a list of (coefficient, word)."""
        row = {}
        for coef, w in terms:
            f = self.moment(w)
            row[f] = row.get(f, 0.0) + coef
        return row

    # -- constraint helpers ---------------------------------------------------
    def add_eq(self, row, rhs, tag):
        self.eq_rows.append(row)
        self.eq_rhs.append(rhs)
        self.eq_tags.append(tag)

    def add_ineq(self, row, rhs, tag):
        self.ineq_rows.append(row)
        self.ineq_rhs.append(rhs)
        self.ineq_tags.append(tag)

    def probability_terms(self, a, b, c, x, y, z):
        """P(abc|xyz) as a list of (coefficient, word)."""
        factors = []
        for party, outcome, s in ((0, a, x), (1, b, y), (2, c, z)):
            factors.append([(1.0, ((party, s),))] if outcome == 0
                           else [(1.0, ()), (-1.0, ((party, s),))])
        terms = []
        for combo in itertools.product(*factors):
            coef, word = 1.0, ()
            for cf, w in combo:
                coef *= cf
                word += w
            terms.append((coef, word))
        return terms

    def add_standard_constraints(self, nonnegativity=True):
        self.add_eq({self.moment(()): 1.0}, 1.0, "identity")
        if not nonnegativity:
            return
        m = self.m
        for x, y, z, a, b, c in itertools.product(range(m), range(m), range(m), (0, 1), (0, 1), (0, 1)):
            self.add_ineq(self.lin(self.probability_terms(a, b, c, x, y, z)), 0.0,
                          f"P({a}{b}{c}|{x}{y}{z})>=0")

    def add_ordinary_extension(self):
        m = self.m
        for y in range(m):
            self.add_eq(self.lin([(1, (self.B(y),)), (-1, (self.C(y),))]), 0.0, f"pB({y})=pC({y})")
        for x, y in itertools.product(range(m), repeat=2):
            self.add_eq(self.lin([(1, (self.A(x), self.B(y))), (-1, (self.A(x), self.C(y)))]), 0.0,
                        f"pAB({x}{y})=pAC({x}{y})")

    def mismatch_terms(self, y):
        """P(b != c | y, z = y) = <B_y> + <C_y> - 2 <B_y C_y>."""
        return [(1.0, (self.B(y),)), (1.0, (self.C(y),)), (-2.0, (self.B(y), self.C(y)))]

    def add_mismatch_bound(self, eps):
        for y in range(self.m):
            row = self.lin([(-c, w) for c, w in self.mismatch_terms(y)])
            if eps == 0:
                self.add_eq({k: -v for k, v in row.items()}, 0.0, f"e({y})=0")
            else:
                self.add_ineq(row, -eps, f"e({y})<=eps")

    def add_fixed_ab(self, pAB):
        """Fix the Collins-Gisin moments of p_AB (binary outcomes)."""
        m = self.m
        for x in range(m):
            self.add_eq(self.lin([(1, (self.A(x),))]), float(pAB[x, 0, 0, :].sum()), f"pA({x})")
        for y in range(m):
            self.add_eq(self.lin([(1, (self.B(y),))]), float(pAB[0, y, :, 0].sum()), f"pB({y})")
        for x, y in itertools.product(range(m), repeat=2):
            self.add_eq(self.lin([(1, (self.A(x), self.B(y)))]), float(pAB[x, y, 0, 0]), f"pAB({x}{y})")

    def bell_row(self, expr):
        const, cA, cB, cAB = cg_coefficients(expr)
        terms = [(cA[x], (self.A(x),)) for x in range(self.m)]
        terms += [(cB[y], (self.B(y),)) for y in range(self.m)]
        terms += [(cAB[x, y], (self.A(x), self.B(y))) for x in range(self.m) for y in range(self.m)]
        return self.lin(terms), const

    def mean_mismatch_row(self):
        terms = []
        for y in range(self.m):
            terms += [(c / self.m, w) for c, w in self.mismatch_terms(y)]
        return self.lin(terms)

    # -- sparse matrices ------------------------------------------------------
    def sparse(self, rows):
        data, ri, ci = [], [], []
        for r, row in enumerate(rows):
            for k, v in row.items():
                if v != 0:
                    ri.append(r)
                    ci.append(k)
                    data.append(v)
        return sp.csr_matrix((data, (ri, ci)), shape=(len(rows), self.K))

    def moment_matrix(self, v):
        return (self.M @ v).reshape((self.n, self.n), order="F")

    # -- solve ----------------------------------------------------------------
    def solve(self, objective_row, maximize, solver=None, eps=1e-8, time_limit=None, verbose=False):
        n = self.n
        v = cp.Variable(self.K)
        G = cp.reshape(self.M @ v, (n, n), order="F")   # symmetric by construction
        obj_vec = np.zeros(self.K)
        for k, c in objective_row.items():
            obj_vec[k] += c
        Aeq = self.sparse(self.eq_rows)
        cons = [G >> 0, Aeq @ v == np.array(self.eq_rhs)]
        if self.ineq_rows:
            Ain = self.sparse(self.ineq_rows)
            cons.append(Ain @ v >= np.array(self.ineq_rhs))
        expr = obj_vec @ v
        prob = cp.Problem(cp.Maximize(expr) if maximize else cp.Minimize(expr), cons)
        if solver is None:
            solver = "CLARABEL" if n <= 100 else "SCS"
        kw = {}
        if solver == "SCS":
            kw = dict(eps_abs=eps, eps_rel=eps, max_iters=500000)
            if time_limit:
                kw["time_limit_secs"] = time_limit
        elif solver == "CLARABEL":
            kw = dict(tol_gap_abs=eps, tol_gap_rel=eps, tol_feas=eps)
            if time_limit:
                kw["time_limit"] = time_limit
        prob.solve(solver=solver, verbose=verbose, **kw)
        info = {"status": prob.status, "objective": None if prob.value is None else float(prob.value),
                "solver": solver, "matrix_size": n, "num_moments": self.K}
        vv = v.value
        if vv is not None:
            info["min_eigenvalue"] = float(np.linalg.eigvalsh(self.moment_matrix(vv))[0])
            res = [np.max(np.abs(Aeq @ vv - np.array(self.eq_rhs)))]
            if self.ineq_rows:
                res.append(max(0.0, -np.min(Ain @ vv - np.array(self.ineq_rhs))))
            info["max_constraint_violation"] = float(max(res))
        self._last = dict(problem=prob, v=v, constraints=cons)
        return vv, info


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------
def max_bell(m, level, expr, mode, eps_mismatch=None, nonnegativity=True, **solve_kw):
    """mode: 'qb' (no extension), 'ext' (ordinary), 'perfect'.

    nonnegativity=False drops the explicit P(abc|xyz) >= 0 constraints, which
    are implied by positivity of G only when all needed words are present.
    """
    R = MomentRelaxation(m, level)
    R.add_standard_constraints(nonnegativity)
    if mode in ("ext", "perfect"):
        R.add_ordinary_extension()
    if mode == "perfect":
        R.add_mismatch_bound(0.0)
    if eps_mismatch is not None:
        R.add_mismatch_bound(eps_mismatch)
    row, const = R.bell_row(expr)
    G, info = R.solve(row, maximize=True, **solve_kw)
    if info["objective"] is not None:
        info["objective"] += const
    return R, G, info


def min_mismatch(m, level, pAB, **solve_kw):
    R = MomentRelaxation(m, level)
    R.add_standard_constraints()
    R.add_ordinary_extension()
    R.add_fixed_ab(pAB)
    G, info = R.solve(R.mean_mismatch_row(), maximize=False, **solve_kw)
    return R, G, info
