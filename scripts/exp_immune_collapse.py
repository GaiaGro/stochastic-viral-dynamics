"""Follow-up: what happens after an expanded effector response collapses.

For every effector death rate of at least 0.1/day in the sweep of
exp_immune_activation.py, reruns the same seeds (full model) to day 200 and records,
for the two routes to activation failure, when the effectors died, how far they had
pushed the infection down, and whether the virus was present at days 60 and 200.
The distribution of the peak effector count among failures is binned to show that
the split at E_EXPANDED = 10 falls in an empty gap. Writes results/immune_collapse.csv
and .meta.json.
"""

import csv
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

from exp_immune_activation import BASE, E_EXPANDED, N, T0, V0, classify
from viraldyn import simulate, solve_ode
from viraldyn.ensembles import provenance, run_seeds, write_meta
from viraldyn.ssa import CTL, STOP_EXTINCT, STOP_T_END

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "immune_activation.csv"
OUT = ROOT / "results" / "immune_collapse.csv"
T_LONG = 200.0
T_CLASSIFY = 60.0
GRID = np.linspace(0.0, T_LONG, 4001)
I60 = int(np.searchsorted(GRID, T_CLASSIFY))


def one(seed, p):
    r = simulate(p, [T0, 0, 0, V0, 0, 1], GRID, seed_value=seed, stop_on_extinction=True)
    load = np.where(r.X[:, 1] < 0, np.nan, r.X[:, 1:5].sum(axis=1)).astype(float)
    return r.stop_reason, r.t_end, r.t_ext[CTL], r.cumulative_infections, int(r.x_max[5]), load


def fate_at_60(o):
    stop, t_end, t_ctl, cum = o[:4]
    if t_end > T_CLASSIFY:
        stop, t_end = STOP_T_END, T_CLASSIFY
    return classify(stop, t_end, t_ctl if t_ctl <= T_CLASSIFY else np.nan, cum)


def summarise(group):
    if not group:
        return {"n": 0}
    t_death = np.array([o[2] for o in group])
    peak = np.array([o[4] for o in group])
    min_load = []
    for o in group:
        m = (GRID > 0.5) & (GRID <= o[2])
        min_load.append(np.nanmin(o[5][m]) if m.any() and np.isfinite(o[5][m]).any() else np.nan)
    load60 = np.array([o[5][I60] for o in group])
    load200 = np.array([o[5][-1] for o in group])
    return {
        "n": len(group),
        "median_t_effector_death": float(np.median(t_death)),
        "median_peak_E": float(np.median(peak)),
        "median_min_load_before_death": float(np.nanmedian(min_load)),
        "median_load_day60": float(np.nanmedian(load60)),
        "n_extinct_by_day200": int(sum(o[0] == STOP_EXTINCT for o in group)),
        "median_load_day200_survivors": float(np.nanmedian(load200)) if np.isfinite(load200).any() else float("nan"),
    }


def main():
    started = time.time()
    src = [r for r in csv.DictReader(open(SRC)) if r["sweep"] == "dE" and r["variant"] == "full" and float(r["dE"]) >= 0.09]
    set_point = solve_ode(BASE.replace(k=0.0), [T0, 0, 0, V0, 0, 0], np.linspace(0, T_LONG, 2001))[-1, 1:5].sum()
    one(0, BASE)
    rows = []
    for r in src:
        p = BASE.replace(dE=float(r["dE"]))
        seed_start = int(r["seed_start"])
        out = run_seeds(partial(one, p=p), range(seed_start, seed_start + N))
        fates = [fate_at_60(o) for o in out]
        cleared = [o for o, f in zip(out, fates) if f == "cleared"]
        early = [o for o, f in zip(out, fates) if f == "immune_failure" and o[4] < E_EXPANDED]
        collapse = [o for o, f in zip(out, fates) if f == "immune_failure" and o[4] >= E_EXPANDED]
        peaks = np.array([o[4] for o, f in zip(out, fates) if f == "immune_failure"])
        row = {
            "dE": p.dE,
            "seed_start": seed_start,
            "n": N,
            "n_cleared": len(cleared),
            "n_cleared_effectors_never_expanded": int(sum(o[4] < E_EXPANDED for o in cleared)),
            "n_fail_peakE_1": int((peaks == 1).sum()),
            "n_fail_peakE_2_to_9": int(((peaks >= 2) & (peaks <= 9)).sum()),
            "n_fail_peakE_10_to_99": int(((peaks >= 10) & (peaks <= 99)).sum()),
            "n_fail_peakE_ge_100": int((peaks >= 100).sum()),
            "no_immune_set_point": float(set_point),
        }
        for name, group in (("early", early), ("collapse", collapse)):
            row.update({f"{name}_{k}": v for k, v in summarise(group).items()})
        rows.append(row)
        print(f"dE={p.dE:.3f}: cleared={len(cleared)} early={len(early)} collapse={len(collapse)}; collapse extinct by day 200: {row.get('collapse_n_extinct_by_day200', 0)}")
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("dE", "seed_start", "n"), k))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    write_meta(OUT.with_suffix(".meta.json"), provenance(started, script=Path(__file__).name, t_long=T_LONG, t_classify=T_CLASSIFY, argv=sys.argv))
    print(f"{len(rows)} conditions, {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
