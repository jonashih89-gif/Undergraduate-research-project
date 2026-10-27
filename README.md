# Tripartite Extendibility of Bipartite Nonlocal Correlations

This package studies the following question. A bipartite behavior
`p_AB(a,b|x,y)` is given. A third party C acts as a candidate copy of B:

* an **ordinary extension** is a tripartite behavior `P(a,b,c|x,y,z)` with
  `p_AB = p_AC`;
* a **perfect extension** additionally requires `b = c` whenever `y = z`.

The question is whether a nonlocal `p_AB` that admits an ordinary extension
must also admit a perfect one, and how the answer depends on the set of
allowed extensions (no-signaling, quantum, or a relaxation of the quantum
set).

## Installation

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
```

## Running

```bash
python run_all.py                  # every stage, results cached in results/
python run_all.py --stage exact    # a single stage
python run_all.py --force          # recompute instead of reusing cached cases
```

Stages, in order:

| Stage | What it does | Typical time |
|---|---|---|
| `lp` | Maximum Bell violation over no-signaling distributions, with no extension, an ordinary extension, and a perfect extension | seconds |
| `sampling` | 1,000 random symmetric three-qubit states with random measurements (B and C share measurements) | about 1 minute |
| `quantum` | Targeted search for an explicit nonlocal quantum example | under 1 minute |
| `exact` | Rational version of the example and exact certificates | under 1 minute |
| `noise` | Effect of white noise on nonlocality and on the minimum mismatch | about 1 minute |
| `tradeoff` | Maximum Bell violation when the mismatch per setting is at most eps | about 1 minute |
| `sdp` | Moment-matrix relaxations (almost-quantum, NPA levels 2 and 3) | tens of minutes |
| `summary` | Writes `results/summary.md` | seconds |

Each finished case is written immediately to `results/<case>.json`.

## Files

| File | Contents |
|---|---|
| `scenario.py` | Bell expressions (`I_mm22`, rescaled CGLMP), deterministic strategies, local bound, locality test by linear programming |
| `ns_lp.py` | Linear programs over tripartite no-signaling distributions |
| `moments.py` | Moment-matrix relaxations: NPA level k and the almost-quantum set |
| `quantum.py` | Haar-random states and measurements, B/C symmetrization, Born rule, random sampling, targeted search |
| `exact.py` | Exact rational verification: quantum example, dual certificate, no-signaling extension |
| `run_all.py` | Driver, caching, and summary |

## Conventions

* Settings and outcomes are zero-indexed.
* Bipartite behaviors are arrays `p[x, y, a, b]`; tripartite behaviors are
  `P[x, y, z, a, b, c]`.
* Every Bell expression is normalized so that its local bound is 0.
  `I_mm22` is the Collins-Gisin family (`m = 2` is CHSH, `m = 3` is I3322).
  The CGLMP expression is rescaled as `(d-1)/(2d) * (I_d - 2)`.
* For binary outcomes, `P_x` denotes the projector onto outcome 0; moments
  such as `<A_x B_y>` are probabilities of outcome 0.
* The mismatch for setting `y` is `e_y = P(b != c | y, z = y)`, and the mean
  mismatch is the average over `y`.

## Method notes

**Moment relaxations.** The moment matrix has one real variable per distinct
moment. Restricting to real matrices loses nothing, because the real part of
any feasible complex moment matrix is feasible and gives the same
probabilities. All relaxations impose nonnegativity of every tripartite
probability `P(a,b,c|x,y,z)`; for NPA level 2 this constraint is not implied
by positivity of the moment matrix, and a variant without it is also run.

**Exact certificate.** For the almost-quantum relaxation with `p_AB` fixed,
the solver's dual solution is rounded to rationals with the positive
semidefinite multiplier kept in the form `Z = L L^T`. For any such
`Z >= 0`, nonnegative `u`, and multipliers `mu`, every feasible moment
vector `v` satisfies

```
f.v = r.v + <Z, G(v)> + u.(T v) + mu.c >= mu.c - sum_k |r_k|,
```

where `r = f - M^T vec(Z) - T^T u - A^T mu` is computed exactly and
`|v_k| <= 1` holds for every almost-quantum moment. The resulting lower
bound does not depend on solver accuracy. Since every quantum behavior lies
in the almost-quantum set, the bound applies to all quantum extensions.

**Exact no-signaling extension.** A vertex of the perfect-extension
feasibility LP is found with a simplex method; the linear system on its
support is then solved in rational arithmetic, and every constraint is
rechecked exactly.

## Limitations

* The targeted search gives an achievable value, not a proven optimum.
* Only the specific rational example carries exact certificates; noise
  scans, trade-off curves, and SDP tables are numerical.
* First-order solver (SCS) results for the largest relaxations are accurate
  only to about `1e-5`; their status is stored in each JSON file.

## Report

`report/` turns the results into LaTeX:

| File | Contents |
|---|---|
| `report_template.tex` | Abstract, methodology subsections, Results, and Discussion with placeholders |
| `fill_report.py` | Fills the placeholders from `results/` and writes `report.tex` |
| `report.tex` | Generated sections, ready to paste into the main report |
| `preview.tex`, `preview.pdf` | Standalone document that compiles `report.tex` with its references |

After rerunning any stage:

```bash
cd report
python fill_report.py
pdflatex preview.tex && pdflatex preview.tex
```

## Cases not yet computed

The largest relaxations were not completed in the bundled run (single CPU):
almost-quantum `m = 5` (QB and Ext), NPA level 2 `m = 5` (all three), and
NPA level 3 `m = 3` (QB and Perf-Ext). `results/sdp_L3_m3_ext.json` comes
from a standalone run of the same function. To finish them, run

```bash
python run_all.py --stage sdp
```

Completed cases are reused, so only the missing ones are computed. Each
takes several minutes. Then regenerate the report as above.
