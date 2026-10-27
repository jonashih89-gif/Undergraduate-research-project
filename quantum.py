"""Three-qubit quantum realizations with Bob and Charlie as symmetric copies.

Conventions: projectors are 2x2 arrays P[x] onto outcome 0; outcome 1 is
I - P[x]. The full behavior is P[x, y, z, a, b, c].
"""
import itertools

import numpy as np
from scipy.optimize import minimize

from scenario import bell_value, local_visibility

I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0, -1.0]).astype(complex)

# swap of qubits B and C on A (x) B (x) C
SWAP_BC = np.zeros((8, 8))
for a, b, c in itertools.product(range(2), repeat=3):
    SWAP_BC[4 * a + 2 * c + b, 4 * a + 2 * b + c] = 1.0


# ---------------------------------------------------------------------------
# States and measurements
# ---------------------------------------------------------------------------
def haar_state(dim, rng):
    v = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return v / np.linalg.norm(v)


def haar_unitary(dim, rng):
    Zm = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))) / np.sqrt(2)
    Q, R = np.linalg.qr(Zm)
    return Q * (np.diag(R) / np.abs(np.diag(R)))


def symmetrize_bc(psi):
    """Project onto the B<->C symmetric subspace and renormalize."""
    v = psi + SWAP_BC @ psi
    nrm = np.linalg.norm(v)
    if nrm < 1e-12:
        raise ValueError("state is antisymmetric under B<->C")
    return v / nrm


def random_projectors(m, rng):
    """Projector onto the first column of a Haar-random unitary, per setting."""
    out = []
    for _ in range(m):
        u = haar_unitary(2, rng)[:, 0]
        out.append(np.outer(u, u.conj()))
    return np.array(out)


def planar_projector(theta):
    """Projector onto (cos(theta/2), sin(theta/2)), i.e. Bloch vector in the x-z plane."""
    v = np.array([np.cos(theta / 2), np.sin(theta / 2)])
    return np.outer(v, v)


def outcome_ops(P):
    return [P, I2 - P]


def behavior(rho, A, B, C):
    """Born rule P(abc|xyz) = Tr[rho (A_a|x (x) B_b|y (x) C_c|z)]."""
    m = len(A)
    out = np.zeros((m, m, m, 2, 2, 2))
    for x, y, z in itertools.product(range(m), repeat=3):
        for a, b, c in itertools.product(range(2), repeat=3):
            op = np.kron(np.kron(outcome_ops(A[x])[a], outcome_ops(B[y])[b]), outcome_ops(C[z])[c])
            out[x, y, z, a, b, c] = np.real(np.trace(rho @ op))
    return out


def marginal_ab(P):
    return P[:, :, 0].sum(axis=-1)


def marginal_ac(P):
    return P[:, 0, :].sum(axis=-2)


def mismatch(P):
    """e_y = P(b != c | y, z = y), read at x = 0."""
    m = P.shape[0]
    return np.array([P[0, y, y, :, 0, 1].sum() + P[0, y, y, :, 1, 0].sum() for y in range(m)])


def diagnostics(P):
    m = P.shape[0]
    norm = P.sum(axis=(3, 4, 5))
    ns = 0.0
    for ax_set, ax_out in ((0, 3), (1, 4), (2, 5)):
        marg = P.sum(axis=ax_out)
        ref = np.take(marg, [0], axis=ax_set)
        ns = max(ns, float(np.max(np.abs(marg - ref))))
    return {
        "max_normalization_error": float(np.max(np.abs(norm - 1))),
        "min_probability": float(P.min()),
        "max_no_signaling_error": ns,
        "max_marginal_difference_AB_AC": float(np.max(np.abs(marginal_ab(P) - marginal_ac(P)))),
        "mismatch_per_setting": mismatch(P).tolist(),
        "mean_mismatch": float(mismatch(P).mean()),
    }


# ---------------------------------------------------------------------------
# Stage 1: random sampling
# ---------------------------------------------------------------------------
def random_sampling(n_samples, m, expr, seed):
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(n_samples):
        psi = symmetrize_bc(haar_state(8, rng))
        rho = np.outer(psi, psi.conj())
        A = random_projectors(m, rng)
        B = random_projectors(m, rng)
        P = behavior(rho, A, B, B)
        pAB = marginal_ab(P)
        d = diagnostics(P)
        records.append({
            "bell_value": bell_value(pAB, expr),
            "local_visibility": local_visibility(pAB),
            "max_marginal_difference": d["max_marginal_difference_AB_AC"],
            "mean_mismatch": d["mean_mismatch"],
            "max_no_signaling_error": d["max_no_signaling_error"],
        })
    return records


def independent_measurement_demo(m, seed):
    """Marginal difference when C does / does not share Bob's measurements."""
    rng = np.random.default_rng(seed)
    psi = symmetrize_bc(haar_state(8, rng))
    rho = np.outer(psi, psi.conj())
    A, B, Cind = (random_projectors(m, rng) for _ in range(3))
    shared = behavior(rho, A, B, B)
    indep = behavior(rho, A, B, Cind)
    return (float(np.max(np.abs(marginal_ab(shared) - marginal_ac(shared)))),
            float(np.max(np.abs(marginal_ab(indep) - marginal_ac(indep)))))


# ---------------------------------------------------------------------------
# Stage 2: targeted search
# ---------------------------------------------------------------------------
def symmetric_basis():
    """Orthonormal basis (8 x 6) of the B<->C symmetric subspace."""
    w, V = np.linalg.eigh((np.eye(8) + SWAP_BC) / 2)
    return V[:, w > 0.5]


def bell_operator(expr, A, B):
    C, offset = expr
    m = len(A)
    H = offset * np.eye(8)
    for x, y, a, b in itertools.product(range(m), range(m), range(2), range(2)):
        H = H + C[x, y, a, b] * np.kron(np.kron(outcome_ops(A[x])[a], outcome_ops(B[y])[b]), I2)
    return np.real(H)


def best_symmetric_state(expr, A, B):
    """Leading eigenvector of the Bell operator restricted to the symmetric subspace."""
    V = symmetric_basis()
    H = V.T @ bell_operator(expr, A, B) @ V
    w, U = np.linalg.eigh((H + H.T) / 2)
    return w[-1], V @ U[:, -1]


def targeted_search(expr, m, restarts, seed):
    """Maximize the Bell value over planar measurements (B and C shared)
    and over B<->C symmetric pure states. Returns the best realization."""
    rng = np.random.default_rng(seed)

    def unpack(t):
        return (np.array([planar_projector(v) for v in t[:m]]),
                np.array([planar_projector(v) for v in t[m:]]))

    def neg_value(t):
        A, B = unpack(t)
        return -best_symmetric_state(expr, A, B)[0]

    runs, best = [], None
    for r in range(restarts):
        t0 = rng.uniform(-np.pi, np.pi, 2 * m)
        res = minimize(neg_value, t0, method="BFGS", options={"gtol": 1e-10, "maxiter": 2000})
        runs.append(float(-res.fun))
        if best is None or res.fun < best.fun:
            best = res
    A, B = unpack(best.x)
    value, psi = best_symmetric_state(expr, A, B)
    return {"angles_A": best.x[:m].tolist(), "angles_B": best.x[m:].tolist(),
            "psi": psi, "A": A, "B": B, "value": float(value), "restart_values": runs}
