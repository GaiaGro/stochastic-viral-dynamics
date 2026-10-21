"""Stochastic failure of immune activation.

Full model, "establishment" preset (R0 = 33; the deterministic solution clears the
infection within two days through effector expansion). From one wild-type virion
and E0 effector cells each realisation is classified at T_END:

  bottleneck         virus extinct before 100 cumulative infections
  cleared            virus extinct after establishing, effectors still present
  immune_failure     virus present at T_END and every effector lineage extinct
  persistent_immune  virus present at T_END with effectors present
  other              virus extinct after establishing with no effector left

Target quantity: P(immune_failure | established). The "k0" variant (effectors
cannot kill) removes the feedback of the effectors on the infection; given the
infected-cell path, the effector lineage is then exactly a linear birth-death
process and Kendall's formula applies path by path.

Writes results/immune_activation.csv, .meta.json and
results/immune_activation_trajectories.npz (100 runs for the figure).
"""

import csv
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

from viraldyn import PRESETS, simulate
from viraldyn.ensembles import provenance, run_seeds, write_meta
from viraldyn.ssa import CTL, STOP_EXTINCT
from viraldyn.stats import wilson
from viraldyn.theory import immune_failure_probability, kendall_extinction

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "immune_activation.csv"
TRAJ = ROOT / "results" / "immune_activation_trajectories.npz"
N = 2000
T_END = 60.0
GRID = np.linspace(0.0, T_END, 3001)  # 0.02-day steps, used for the per-path formula
THRESHOLD = 100
E_CAP = 1000  # k0 variant: nothing limits the effectors, so stop once the lineage is this large
E_EXPANDED = 10  # an effector lineage that reached this size had taken off before it died
BASE = PRESETS["establishment"]
T0, V0 = 1000, 1
E0_SWEEP = (1, 2, 3, 5, 10)
DE_LEVELS = (0.1, 0.3, 1.0)  # effector death rates for the E0 sweep; 0.1 is the preset
DE_SWEEP = np.logspace(np.log10(0.03), np.log10(3.0), 9)
Q_SWEEP = np.logspace(-2, 0, 9)  # kept to show that the proliferation rate is a weak lever
FATES = ("bottleneck", "cleared", "immune_failure", "persistent_immune", "other")


def classify(stop_reason, t_end, t_ext_ctl, cum_infections, threshold=THRESHOLD):
    """`t_end` is unused: the virus and the effector lineage cannot die in the same
    event, so an effector death time is always strictly before an extinction stop."""
    established = cum_infections >= threshold
    ctl_died = not np.isnan(t_ext_ctl)
    if stop_reason == STOP_EXTINCT:
        if not established:
            return "bottleneck"
        return "other" if ctl_died else "cleared"
    return "immune_failure" if ctl_died else "persistent_immune"


def one(seed, p, x0, per_path):
    cap = [0, 0, 0, 0, 0, E_CAP] if per_path else None
    r = simulate(p, x0, GRID, seed_value=seed, stop_on_extinction=True, stop_above=cap)
    p0 = np.nan
    if per_path:
        valid = r.X[:, 1] >= 0  # recorded part of the grid; by the cap the lineage is safe
        if valid.sum() >= 2:
            I = (r.X[valid, 1] + r.X[valid, 2]).astype(float)
            p0 = kendall_extinction(GRID[valid], p.q * I, p.dE)[-1]
    return r.stop_reason, r.t_end, r.t_ext[CTL], r.cumulative_infections, p0, r.x_max[5]


def condition(sweep, variant, p, x0, seed_start):
    out = run_seeds(partial(one, p=p, x0=x0, per_path=(variant == "k0")), range(seed_start, seed_start + N))
    fates = [classify(*o[:4]) for o in out]
    counts = {f: fates.count(f) for f in FATES}
    established = N - counts["bottleneck"]
    lo, hi = wilson(counts["immune_failure"], established)
    # two routes to failure: the founding lineage never took off (what Kendall's formula
    # describes) or the response expanded, suppressed the virus and then collapsed
    early = sum(1 for o, f in zip(out, fates) if f == "immune_failure" and o[5] < E_EXPANDED)
    lo_e, hi_e = wilson(early, established)
    p0_path = np.array([o[4] for o, f in zip(out, fates) if f != "bottleneck"])
    return {
        "sweep": sweep,
        "variant": variant,
        "E0": x0[5],
        "q": p.q,
        "dE": p.dE,
        "n": N,
        "n_established": established,
        **{f"n_{f}": counts[f] for f in FATES},
        "n_failure_early": early,
        "n_failure_collapse": counts["immune_failure"] - early,
        "p_fail": counts["immune_failure"] / established,
        "wilson_low": float(lo),
        "wilson_high": float(hi),
        "p_fail_early": early / established,
        "wilson_low_early": float(lo_e),
        "wilson_high_early": float(hi_e),
        "p_theory": immune_failure_probability(p, x0, t_max=T_END),
        "p_theory_per_path": float(np.mean(p0_path ** x0[5])) if variant == "k0" else float("nan"),
        "seed_start": seed_start,
    }


def trajectories(seed_start, n_runs=100, t_end=30.0):
    """Total viral load of n_runs full-model realisations from (V0, E0) = (1, 1), coloured by fate."""
    grid = np.linspace(0.0, t_end, 1501)
    V = np.full((n_runs, grid.size), -1, dtype=np.int64)
    fates = []
    for i in range(n_runs):
        r = simulate(BASE, [T0, 0, 0, V0, 0, 1], grid, seed_value=seed_start + i, stop_on_extinction=True)
        V[i] = np.where(r.X[:, 1] < 0, -1, r.X[:, 1:5].sum(axis=1))
        fates.append(classify(r.stop_reason, r.t_end, r.t_ext[CTL], r.cumulative_infections))
    return grid, V, np.array(fates)


def main():
    started = time.time()
    one(0, BASE, [T0, 0, 0, V0, 0, 1], True)  # compile before forking
    rows = []
    seed = 2_000_000
    for variant, p in (("full", BASE), ("k0", BASE.replace(k=0.0))):
        for dE in DE_LEVELS:
            for E0 in E0_SWEEP:
                rows.append(condition("E0", variant, p.replace(dE=dE), [T0, 0, 0, V0, 0, E0], seed))
                seed += N
        for dE in DE_SWEEP:
            rows.append(condition("dE", variant, p.replace(dE=float(dE)), [T0, 0, 0, V0, 0, 1], seed))
            seed += N
        for q in Q_SWEEP:
            rows.append(condition("q", variant, p.replace(q=float(q)), [T0, 0, 0, V0, 0, 1], seed))
            seed += N
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    grid, V, fates = trajectories(seed)
    np.savez_compressed(TRAJ, t=grid, V=V, fate=fates, seed_start=seed)
    write_meta(OUT.with_suffix(".meta.json"), provenance(started, script=Path(__file__).name, n_per_condition=N, threshold=THRESHOLD, t_end=T_END, argv=sys.argv))
    print(f"{len(rows)} conditions, {len(rows) * N} runs, {time.time() - started:.0f} s")
    for r in rows:
        print(f"  {r['sweep']:2s} {r['variant']:4s} q={r['q']:.4f} dE={r['dE']:.3f} E0={r['E0']:>2}  fail={r['p_fail']:.3f} [{r['wilson_low']:.3f},{r['wilson_high']:.3f}] early={r['p_fail_early']:.3f} theory={r['p_theory']:.3f} per-path={r['p_theory_per_path']:.3f}  fates: " + " ".join(f"{f}={r['n_' + f]}" for f in FATES))
    print("trajectory fates:", {f: int((fates == f).sum()) for f in FATES})


if __name__ == "__main__":
    main()
