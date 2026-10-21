"""Probability that an infection establishes from a small inoculum, from exact
simulation and from the branching-process prediction.

Establishment: 100 cumulative infection events before every infected cell and
virion is gone. Two sweeps with the "establishment" preset (T0 = 1000 = T*):
  (a) inoculum V0 for several fixed effector counts E0 (q = d_E = 0, so k E0 is a
      constant extra death rate of infected cells),
  (b) wild-type infectivity beta_w, hence R0, for V0 in {1, 5} with no effectors.
A third sweep checks that the establishment threshold does not matter: for five
infectivities and one virion with no effectors, the probability of reaching 25,
50, 100, 200 and 400 cumulative infections (results/establishment_threshold.csv).
Writes results/establishment.csv and results/establishment.meta.json.
"""

import csv
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

from viraldyn import PRESETS, r0_wild_type, simulate
from viraldyn.ensembles import provenance, run_seeds, write_meta
from viraldyn.ssa import STOP_CUM_INFECTIONS, STOP_EXTINCT
from viraldyn.stats import wilson
from viraldyn.theory import establishment_probability

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "establishment.csv"
N = 2000
T_MAX = 500.0
THRESHOLD = 100
BASE = PRESETS["establishment"].replace(q=0.0, dE=0.0)  # effectors fixed at E0
T0 = 1000
V0_SWEEP = (1, 2, 3, 5, 10, 20)
E0_LEVELS = (0, 5, 20, 50)
BETA_SWEEP = np.logspace(np.log10(5e-5), np.log10(2e-2), 22)
V0_FOR_BETA = (1, 5)
THRESHOLDS = (25, 50, 100, 200, 400)
BETA_THRESHOLD = (2e-4, 4e-4, 1e-3, 3e-3, 2e-2)
N_THRESHOLD = 4000
OUT_THRESHOLD = ROOT / "results" / "establishment_threshold.csv"


def one(seed, p, x0):
    return simulate(p, x0, [0.0, T_MAX], seed_value=seed, stop_on_extinction=True, stop_cum_infections=THRESHOLD).stop_reason


def one_cumulative(seed, p, x0):
    r = simulate(p, x0, [0.0, T_MAX], seed_value=seed, stop_on_extinction=True, stop_cum_infections=max(THRESHOLDS))
    return r.cumulative_infections


def condition(sweep, p, x0, seed_start):
    reasons = np.array(run_seeds(partial(one, p=p, x0=x0), range(seed_start, seed_start + N)))
    k = int((reasons == STOP_CUM_INFECTIONS).sum())
    undetermined = int(((reasons != STOP_CUM_INFECTIONS) & (reasons != STOP_EXTINCT)).sum())
    lo, hi = wilson(k, N)
    return {
        "sweep": sweep,
        "E0": x0[5],
        "V0": x0[3],
        "bw": p.bw,
        "R0_wt": r0_wild_type(p),
        "n": N,
        "k_established": k,
        "k_undetermined": undetermined,
        "p_hat": k / N,
        "wilson_low": float(lo),
        "wilson_high": float(hi),
        "p_theory": establishment_probability(p, x0),
        "seed_start": seed_start,
    }


def main():
    started = time.time()
    one(0, BASE, [T0, 0, 0, 1, 0, 0])  # compile before forking
    rows = []
    seed = 1_000_000
    for E0 in E0_LEVELS:
        for V0 in V0_SWEEP:
            rows.append(condition("V0", BASE, [T0, 0, 0, V0, 0, E0], seed))
            seed += N
    for bw in BETA_SWEEP:
        p = BASE.replace(bw=float(bw), bm=float(bw) * 0.8)
        for V0 in V0_FOR_BETA:
            rows.append(condition("beta", p, [T0, 0, 0, V0, 0, 0], seed))
            seed += N
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    threshold_rows = []
    for bw in BETA_THRESHOLD:
        p = BASE.replace(bw=float(bw), bm=float(bw) * 0.8)
        x0 = [T0, 0, 0, 1, 0, 0]
        cum = np.array(run_seeds(partial(one_cumulative, p=p, x0=x0), range(seed, seed + N_THRESHOLD)))
        threshold_rows.append({"bw": p.bw, "R0_wt": r0_wild_type(p), "n": N_THRESHOLD, "p_theory": establishment_probability(p, x0), "seed_start": seed,
                               **{f"p_reach_{k}": float((cum >= k).mean()) for k in THRESHOLDS}})
        seed += N_THRESHOLD
    with open(OUT_THRESHOLD, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(threshold_rows[0]))
        w.writeheader()
        w.writerows(threshold_rows)
    write_meta(OUT.with_suffix(".meta.json"), provenance(started, script=Path(__file__).name, n_per_condition=N, threshold=THRESHOLD, thresholds_checked=THRESHOLDS, t_max=T_MAX, argv=sys.argv))
    spread = max(max(r[f"p_reach_{k}"] for k in THRESHOLDS) - min(r[f"p_reach_{k}"] for k in THRESHOLDS) for r in threshold_rows)
    print(f"threshold check: largest spread of P(reach k) over k in {THRESHOLDS} is {spread:.4f}")
    worst = max(abs(r["p_hat"] - r["p_theory"]) for r in rows)
    outside = sum(not (r["wilson_low"] <= r["p_theory"] <= r["wilson_high"]) for r in rows)
    print(f"{len(rows)} conditions, {len(rows) * N} runs, {time.time() - started:.0f} s")
    print(f"largest |p_hat - p_theory| = {worst:.3f}; theory outside the 95% interval in {outside}/{len(rows)} conditions")


if __name__ == "__main__":
    main()
