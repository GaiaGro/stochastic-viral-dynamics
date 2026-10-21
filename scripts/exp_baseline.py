"""Baseline: the exact simulation tracks the deterministic solution when
populations are large, and the phase-plane picture of chronic infection.

"chronic_large" preset: the resistance preset with every population scaled by 100
(T* = 1e5, chronic set point 5000 infected cells), started from its uninfected
equilibrium with 1000 infected cells and 1000 effectors. For each species the 5th to
95th percentile band and the median of N realisations are compared with the ODE,
and the largest relative deviation of the median plus the band's coverage of the
ODE are written to results/baseline_validation.csv. One 200-day realisation is
stored with the ODE orbit for the phase plane of Figure 3.

The "chronic" preset is checked separately (results/chronic_preset_check.csv): from
its own uninfected equilibrium T* = 1e6 the deterministic infection is cleared and
every realisation dies, so its oscillations exist only when started from T0 = 1000,
a thousandth of equilibrium.

Writes results/baseline_ensemble.npz and the two CSVs.
"""

import csv
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

from viraldyn import PRESETS, simulate, solve_ode, target_cell_equilibrium
from viraldyn.ensembles import provenance, run_seeds, write_meta
from viraldyn.model import SPECIES
from viraldyn.ssa import STOP_EXTINCT

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "baseline_validation.csv"
CHECK = ROOT / "results" / "chronic_preset_check.csv"
ENSEMBLE = ROOT / "results" / "baseline_ensemble.npz"
P = PRESETS["chronic_large"]
X0 = [100_000, 1000, 0, 0, 0, 1000]
T_END = 60.0
GRID = np.linspace(0.0, T_END, 1201)
N = 200
QUANTILES = (5, 25, 50, 75, 95)
PHASE_T = 200.0
PHASE_GRID = np.linspace(0.0, PHASE_T, 4001)
PHASE_SEED = 7


def one(seed):
    return simulate(P, X0, GRID, seed_value=seed).X.astype(np.int32)


def _alive(seed, p, x0, t_end):
    return simulate(p, x0, [0.0, t_end], seed_value=seed, stop_on_extinction=True).stop_reason != STOP_EXTINCT


def chronic_preset_check(seed, n=200, t_end=60.0):
    """The chronic preset from T0 = 1000 and from its uninfected equilibrium T0 = T*."""
    p = PRESETS["chronic"]
    rows = []
    for label, T0 in (("T0_1000", 1000), ("T0_Tstar", int(target_cell_equilibrium(p)))):
        x0 = [T0, 5, 0, 30, 0, 10]
        t = np.linspace(0.0, 120.0, 2401)
        ode = solve_ode(p, x0, t)
        load = ode[:, 1:5].sum(axis=1)
        alive = sum(run_seeds(partial(_alive, p=p, x0=x0, t_end=t_end), range(seed, seed + n)))
        rows.append({
            "start": label, "T0": T0, "T_star": target_cell_equilibrium(p),
            "R_eff_at_start": p.bw * T0 / (p.cw + p.bw * T0) * p.pw * (1 - p.mu) / p.delta,
            "ode_peak_load": float(load.max()), "ode_peak_day": float(t[load.argmax()]),
            "ode_load_day120": float(load[-1]), "ode_T_day120": float(ode[-1, 0]), "ode_E_day120": float(ode[-1, 5]),
            "n": n, "t_end": t_end, "n_alive": int(alive), "seed_start": seed,
        })
        seed += n
    return rows


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
    phase = simulate(P, X0, PHASE_GRID, seed_value=PHASE_SEED)
    phase_ode = solve_ode(P, X0, PHASE_GRID)
    np.savez_compressed(
        ENSEMBLE,
        t=GRID, x0=np.array(X0), n=N, quantiles=np.array(QUANTILES), q=q.astype(np.int32), ode=ode,
        phase_t=PHASE_GRID, phase_x0=np.array(X0), phase_X=phase.X.astype(np.int32), phase_ode=phase_ode, phase_seed=PHASE_SEED,
    )
    check = chronic_preset_check(5_100_000)
    for path, data in ((OUT, rows), (CHECK, check)):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    write_meta(OUT.with_suffix(".meta.json"), provenance(started, script=Path(__file__).name, preset="chronic_large", n=N, t_end=T_END, x0=X0, argv=sys.argv))
    print(f"{N} realisations, {time.time() - started:.0f} s")
    for r in rows:
        print(f"  {r['species']:3s} ODE range after day 1 [{r['ode_min_after_day1']:.3g}, {r['ode_max_after_day1']:.3g}]  max |median/ODE - 1| = {r['max_rel_dev_median']:.3f}  max |log10 ratio| = {r['max_abs_log10_dev_median']:.3f}  ODE inside 5-95% band {100 * r['coverage_5_95']:.0f}% of the time; extinct at end {r['n_extinct_at_end']}")
    for r in check:
        print(f"  chronic preset, {r['start']}: R_eff(0) = {r['R_eff_at_start']:.2f}, ODE peak {r['ode_peak_load']:.3g} at day {r['ode_peak_day']:.1f}, load at day 120 {r['ode_load_day120']:.3g}; alive at day {r['t_end']:g}: {r['n_alive']}/{r['n']}")


if __name__ == "__main__":
    main()
