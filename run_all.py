"""Run the full study.

Usage:
    python run_all.py                 # all stages
    python run_all.py --stage lp      # one stage: lp, sdp, sampling, quantum,
                                      #   exact, noise, tradeoff, summary
    python run_all.py --force         # recompute cached cases

Every case is written to results/<case>.json as soon as it finishes, and
existing files are reused unless --force is given.
"""
import argparse
import itertools
import json
import platform
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy
import cvxpy

from scenario import imm22, cglmp, local_bound, local_visibility, bell_value
from ns_lp import max_bell_ns, min_mismatch_ns
from moments import max_bell, min_mismatch, word_set
import quantum as Q
import exact as E

OUT = Path("results")
FORCE = False
SEED = 20261027


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def clean(v):
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return clean(v.tolist())
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, Fraction):
        return {"fraction": f"{v.numerator}/{v.denominator}", "decimal": float(v)}
    return v


def cached(name, fn):
    path = OUT / f"{name}.json"
    if path.exists() and not FORCE:
        return json.loads(path.read_text())
    t0 = time.time()
    result = fn()
    result["wall_seconds"] = round(time.time() - t0, 2)
    path.write_text(json.dumps(clean(result), indent=2))
    print(f"[done] {name}  ({result['wall_seconds']} s)", flush=True)
    return json.loads(path.read_text())


def load_realization():
    d = np.load(OUT / "quantum_m3.npz")
    return d["psi"], d["A"], d["B"], d["angles_A"], d["angles_B"]


def exact_pab():
    psi, A, B, thA, thB = load_realization()
    q = E.rationalize_state(psi)
    dA = [E.integer_direction(t) for t in thA]
    dB = [E.integer_direction(t) for t in thB]
    return q, dA, dB, E.exact_pair_marginal(q, dA, dB, "AB")


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def stage_lp():
    for m in range(2, 6):
        expr = imm22(m)
        for mode in ("ns", "ext", "perfect"):
            cached(f"lp_imm22_m{m}_{mode}", lambda: {
                "inequality": f"I_{m}{m}22", "mode": mode, "local_bound": local_bound(expr),
                **max_bell_ns(expr, mode)[1]})
    for d in range(2, 6):
        expr = cglmp(d)
        for mode in ("ns", "ext", "perfect"):
            cached(f"lp_cglmp_d{d}_{mode}", lambda: {
                "inequality": f"CGLMP d={d} (rescaled)", "mode": mode, "local_bound": local_bound(expr),
                **max_bell_ns(expr, mode)[1]})


SDP_CASES = [("AQ", 3), ("AQ", 4), (2, 3), (2, 4), ("AQ", 5), (3, 3), (2, 5)]


def stage_sdp():
    # level 2 without explicit nonnegativity of tripartite probabilities
    for mode in ("qb", "ext", "perfect"):
        cached(f"sdp_L2_m3_{mode}_no_nonneg", lambda mode=mode: {
            "level": 2, "m": 3, "mode": mode, "nonnegativity": False,
            **max_bell(3, 2, imm22(3), mode, nonnegativity=False, eps=1e-7)[2]})
    for level, m in SDP_CASES:
        modes = ("perfect", "ext", "qb") if m == 5 or level == 3 else ("qb", "ext", "perfect")
        for mode in modes:
            def run(level=level, m=m, mode=mode):
                n = len(word_set(m, level))
                kw = dict(eps=1e-7) if n <= 100 else dict(eps=1e-6, time_limit=420)
                _, _, info = max_bell(m, level, imm22(m), mode, **kw)
                return {"level": level, "m": m, "mode": mode, **info}
            cached(f"sdp_L{level}_m{m}_{mode}", run)


def stage_sampling():
    def run():
        recs = Q.random_sampling(1000, 3, imm22(3), SEED)
        vis = np.array([r["local_visibility"] for r in recs])
        bell = np.array([r["bell_value"] for r in recs])
        shared, indep = Q.independent_measurement_demo(3, SEED + 1)
        return {
            "num_samples": len(recs),
            "num_nonlocal": int(np.sum(vis < 1 - 1e-9)),
            "num_positive_I3322": int(np.sum(bell > 1e-9)),
            "max_I3322": float(bell.max()),
            "min_local_visibility": float(vis.min()),
            "max_marginal_difference": float(max(r["max_marginal_difference"] for r in recs)),
            "max_no_signaling_error": float(max(r["max_no_signaling_error"] for r in recs)),
            "num_perfect": int(sum(r["mean_mismatch"] < 1e-9 for r in recs)),
            "min_mean_mismatch": float(min(r["mean_mismatch"] for r in recs)),
            "marginal_difference_shared_measurements": shared,
            "marginal_difference_independent_measurements": indep,
        }
    cached("sampling_m3", run)


def stage_quantum():
    for m, restarts in ((3, 40), (4, 60), (5, 60)):
        def run(m=m, restarts=restarts):
            out = Q.targeted_search(imm22(m), m, restarts, SEED + m)
            psi = out["psi"]
            P = Q.behavior(np.outer(psi, psi.conj()), out["A"], out["B"], out["B"])
            pAB = Q.marginal_ab(P)
            np.savez(OUT / f"quantum_m{m}.npz", psi=psi, A=out["A"], B=out["B"], P=P,
                     angles_A=out["angles_A"], angles_B=out["angles_B"])
            vals = np.array(out["restart_values"])
            return {"m": m, "restarts": restarts, "bell_value": bell_value(pAB, imm22(m)),
                    "local_visibility": local_visibility(pAB),
                    "num_restarts_positive": int(np.sum(vals > 1e-6)),
                    "angles_A": out["angles_A"], "angles_B": out["angles_B"],
                    **Q.diagnostics(P)}
        cached(f"quantum_m{m}", run)


def stage_exact():
    def example():
        q, dA, dB, pab = exact_pab()
        pac = E.exact_pair_marginal(q, dA, dB, "AC")
        I = E.exact_bell_value(pab, imm22(3))
        return {"state_integer_vector": [int(t) for t in q],
                "directions_A": [[int(t) for t in v] for v in dA],
                "directions_B_and_C": [[int(t) for t in v] for v in dB],
                "pAB_equals_pAC_exactly": all(pab[i] == pac[i] for i in np.ndindex(pab.shape)),
                "I3322": I, "I3322_greater_than_0.016": I > Fraction(16, 1000),
                "pAB": {f"{x}{y}{a}{b}": pab[x, y, a, b]
                        for x, y, a, b in itertools.product(range(3), range(3), range(2), range(2))}}
    cached("exact_example", example)

    def aq_certificate():
        _, _, _, pab = exact_pab()
        R, _, info = min_mismatch(3, "AQ", np.array(pab, dtype=float), eps=1e-8)
        cert = E.certify_mismatch_lower_bound(R, pab)
        return {"numerical": info, **cert,
                "certified_greater_than_0.0963": cert["lower_bound"] > Fraction(963, 10000)}
    cached("exact_aq_mismatch_certificate", aq_certificate)

    def ns_extension():
        _, _, _, pab = exact_pab()
        pf = np.array(pab, dtype=float)
        _, _, _, info_min = min_mismatch_ns(pf)
        T, rows, P, info = min_mismatch_ns(pf, perfect_only=True)
        ex = E.exact_ns_extension(T, rows, P, pab)
        Pd = {f"{k}": v for k, v in enumerate(ex.pop("P")) if v != 0}
        return {"numerical_min_mean_mismatch": info_min["objective"], **ex,
                "nonzero_probabilities": Pd}
    cached("exact_ns_perfect_extension", ns_extension)

    for level in (2, 3):
        def run(level=level):
            _, _, _, pab = exact_pab()
            kw = dict(eps=1e-7) if level == 2 else dict(eps=1e-6, time_limit=900)
            return {"level": level, **min_mismatch(3, level, np.array(pab, dtype=float), **kw)[2]}
        cached(f"fixed_ab_min_mismatch_L{level}", run)


NOISE_GRID = [1.0, 0.998, 0.996, 0.994, 0.992, 0.990, 0.988, 0.986, 0.985, 0.984]


def stage_noise():
    _, _, _, pab = exact_pab()
    p = np.array(pab, dtype=float)
    I1 = bell_value(p, imm22(3))
    for v in NOISE_GRID:
        def run(v=v):
            pv = v * p + (1 - v) / 4
            _, _, info = min_mismatch(3, "AQ", pv, eps=1e-8)
            return {"visibility": v, "I3322": bell_value(pv, imm22(3)),
                    "local_visibility": local_visibility(pv), "aq_min_mean_mismatch": info}
        cached(f"noise_v{v:.3f}", run)
    cached("noise_threshold", lambda: {"I3322_at_v1": I1, "nonlocality_threshold_v": 1 / (1 + I1)})


EPS_GRID = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3]


def stage_tradeoff():
    for eps in EPS_GRID:
        cached(f"tradeoff_eps{eps:g}", lambda eps=eps: {
            "eps": eps, **max_bell(3, "AQ", imm22(3), "ext", eps_mismatch=eps, eps=1e-8)[2]})


def stage_summary():
    def j(name):
        p = OUT / f"{name}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def fmt(r, key="objective", nd=4):
        if r is None or r.get(key) is None:
            return "n/a"
        v = r[key]
        return f"{0.0 if abs(v) < 5e-7 else v:.{nd}f}"

    lines = ["# Results summary", ""]
    lines += ["## NS bounds (LP)", "", "| m | NS | Ext | Perf-Ext | | d | NS | Ext | Perf-Ext |",
              "|---|---|---|---|---|---|---|---|---|"]
    for m, d in zip(range(2, 6), range(2, 6)):
        a = [fmt(j(f"lp_imm22_m{m}_{x}")) for x in ("ns", "ext", "perfect")]
        b = [fmt(j(f"lp_cglmp_d{d}_{x}")) for x in ("ns", "ext", "perfect")]
        lines.append(f"| {m} | " + " | ".join(a) + f" | | {d} | " + " | ".join(b) + " |")
    lines += ["", "## Moment relaxations (I_mm22)", "", "| relaxation | m | QB | Ext | Perf-Ext | status |",
              "|---|---|---|---|---|---|"]
    for level, m in SDP_CASES:
        rs = [j(f"sdp_L{level}_m{m}_{x}") for x in ("qb", "ext", "perfect")]
        st = ",".join(r["status"] if r else "n/a" for r in rs)
        name = "almost quantum" if level == "AQ" else f"NPA level {level}"
        lines.append(f"| {name} | {m} | " + " | ".join(fmt(r) for r in rs) + f" | {st} |")
    rs = [j(f"sdp_L2_m3_{x}_no_nonneg") for x in ("qb", "ext", "perfect")]
    lines.append("| NPA level 2, no explicit nonnegativity | 3 | " + " | ".join(fmt(r) for r in rs) + " | |")
    s = j("sampling_m3")
    if s:
        lines += ["", "## Random sampling (m = 3)", "", "```", json.dumps(s, indent=2), "```"]
    lines += ["", "## Targeted quantum search", ""]
    for m in (3, 4, 5):
        r = j(f"quantum_m{m}")
        if r:
            lines.append(f"- m = {m}: Bell value {r['bell_value']:.10f}, local visibility "
                         f"{r['local_visibility']:.6f}, mean mismatch {r['mean_mismatch']:.6f}, "
                         f"{r['num_restarts_positive']}/{r['restarts']} restarts positive")
    for name in ("exact_example", "exact_aq_mismatch_certificate", "exact_ns_perfect_extension",
                 "fixed_ab_min_mismatch_L2", "fixed_ab_min_mismatch_L3"):
        r = j(name)
        if r:
            r = {k: v for k, v in r.items() if k not in ("pAB", "nonzero_probabilities")}
            lines += ["", f"## {name}", "", "```", json.dumps(r, indent=2), "```"]
    lines += ["", "## White noise (AQ minimum mean mismatch)", "", "| v | I3322 | local visibility | min mismatch |",
              "|---|---|---|---|"]
    for v in NOISE_GRID:
        r = j(f"noise_v{v:.3f}")
        if r:
            lines.append(f"| {v:.3f} | {r['I3322']:.6f} | {r['local_visibility']:.6f} | "
                         f"{fmt(r['aq_min_mean_mismatch'], nd=6)} |")
    r = j("noise_threshold")
    if r:
        lines.append(f"\nNonlocality threshold for I3322: v = {r['nonlocality_threshold_v']:.6f}")
    lines += ["", "## Bell value vs. allowed mismatch (AQ, ordinary extension)", "", "| eps | max I3322 |", "|---|---|"]
    for eps in EPS_GRID:
        lines.append(f"| {eps:g} | {fmt(j(f'tradeoff_eps{eps:g}'), nd=6)} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


STAGES = {"lp": stage_lp, "sampling": stage_sampling, "quantum": stage_quantum, "exact": stage_exact,
          "noise": stage_noise, "tradeoff": stage_tradeoff, "sdp": stage_sdp, "summary": stage_summary}


def main():
    global FORCE
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=list(STAGES) + ["all"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    FORCE = args.force
    OUT.mkdir(exist_ok=True)
    (OUT / "environment.json").write_text(json.dumps({
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "cvxpy": cvxpy.__version__, "solvers": cvxpy.installed_solvers(), "seed": SEED}, indent=2))
    for name, fn in STAGES.items():
        if args.stage in (name, "all"):
            fn()


if __name__ == "__main__":
    main()
