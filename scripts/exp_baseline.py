"""Baseline (WP6): the exact simulation tracks the deterministic solution when
populations are large, and the phase-plane picture of chronic infection.

Chronic preset (T* = 1e6, chronic set point I* = d_E/q = 1000 infected cells),
N realisations from the notebook's state, recorded on a grid to day 60. For each
species the 5th to 95th percentile band and the median are compared with the ODE,
and the largest relative deviation of the median plus the band's coverage of the
ODE are written to results/baseline_validation.csv. One long realisation from the
notebook's phase-plane state is stored with the ODE orbit for Figure 3.
Writes results/baseline_ensemble.npz and the CSV.
"""

import csv
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

from viraldyn import PRESETS, simulate, solve_ode
from viraldyn.ensembles import provenance, run_seeds, write_meta
from viraldyn.model import SPECIES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "baseline_validation.csv"
ENSEMBLE = ROOT / "results" / "baseline_ensemble.npz"
P = PRESETS["chronic"]
X0 = [1000, 5, 0, 30, 0, 10]
T_END = 60.0
GRID = np.linspace(0.0, T_END, 1201)
N = 200
QUANTILES = (5, 25, 50, 75, 95)
PHASE_X0 = [1000, 10, 0, 100, 0, 10]
PHASE_T = 200.0
PHASE_GRID = np.linspace(0.0, PHASE_T, 4001)
PHASE_SEED = 7


def one(seed):
    return simulate(P, X0, GRID, seed_value=seed).X.astype(np.int32)


def main():
    started = time.time()
    one(0)
    runs = np.stack(run_seeds(one, range(5_000_000, 5_000_000 + N)))  # (N, time, species)
    ode = solve_ode(P, X0, GRID)
    q = np.percentile(runs, QUANTILES, axis=0)  # (5, time, species)
    median = q[QUANTILES.index(50)]
    later = GRID >= 1.0
    rows = []
    for j, name in enumerate(SPECIES):
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(median[later, j] / ode[later, j] - 1.0)
            logdev = np.abs(np.log10(np.maximum(median[later, j], 0.5) / np.maximum(ode[later, j], 0.5)))
        inside = (ode[later, j] >= q[0, later, j]) & (ode[later, j] <= q[-1, later, j])
        rows.append({
            "species": name,
            "ode_min_after_day1": float(ode[later, j].min()),
            "ode_max_after_day1": float(ode[later, j].max()),
            "max_rel_dev_median": float(np.nanmax(rel)) if np.isfinite(rel).any() else float("nan"),
            "max_abs_log10_dev_median": float(np.nanmax(logdev)),
            "coverage_5_95": float(inside.mean()),
            "n": N,
            "n_extinct_at_end": int((runs[:, -1, 1:5].sum(axis=1) == 0).sum()),
        })
    phase = simulate(P, PHASE_X0, PHASE_GRID, seed_value=PHASE_SEED)
    phase_ode = solve_ode(P, PHASE_X0, PHASE_GRID)
    np.savez_compressed(
        ENSEMBLE,
        t=GRID, x0=np.array(X0), n=N, quantiles=np.array(QUANTILES), q=q.astype(np.int32), ode=ode,
        phase_t=PHASE_GRID, phase_x0=np.array(PHASE_X0), phase_X=phase.X.astype(np.int32), phase_ode=phase_ode, phase_seed=PHASE_SEED,
    )
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_meta(OUT.with_suffix(".meta.json"), provenance(started, script=Path(__file__).name, preset="chronic", n=N, t_end=T_END, x0=X0, phase_x0=PHASE_X0, argv=sys.argv))
    print(f"{N} realisations, {time.time() - started:.0f} s")
    for r in rows:
        print(f"  {r['species']:3s} ODE range after day 1 [{r['ode_min_after_day1']:.3g}, {r['ode_max_after_day1']:.3g}]  max |median/ODE - 1| = {r['max_rel_dev_median']:.3f}  max |log10 ratio| = {r['max_abs_log10_dev_median']:.3f}  ODE inside 5-95% band {100 * r['coverage_5_95']:.0f}% of the time")


if __name__ == "__main__":
    main()
