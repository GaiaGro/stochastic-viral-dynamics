"""Stochastic regime map over wild-type infectivity and effector killing rate.

The deterministic model classifies each cell as extinction (the load never rises
above the inoculum), cleared (it grows and falls below one by the horizon) or
chronic (above one at the horizon). Here each cell of a 25 x 25 log grid gets N exact
realisations from 5 wild-type virions and 10 effectors in the "establishment"
preset (T* = 1000) and two probabilities are recorded:

  establishment   100 cumulative infection events before extinction
  persistence     infected cells or virions still present at day 200

together with the deterministic classification on the same horizon, the
effective reproduction number with the initial effectors present,
R_eff = R0 delta / (delta + k E0), and the branching-process establishment
probability.

Two maps are written. With the preset's proliferation rate q = 0.1 the chronic
set point is I* = d_E / q = 1 infected cell, so every realisation that
establishes is cleared by day 200 and the deterministic chronic region does not
exist stochastically (results/regime_map_original_q.csv). The main map
(results/regime_map.csv) lowers q to 0.002 so that I* = 50 and persistence has a
genuine boundary.
"""

import csv
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

from viraldyn import PRESETS, r0_wild_type, simulate, solve_ode
from viraldyn.ensembles import provenance, run_seeds, write_meta
from viraldyn.model import target_cell_equilibrium
from viraldyn.ssa import STOP_EXTINCT
from viraldyn.stats import wilson
from viraldyn.theory import establishment_probability, establishment_probability_decaying_effectors, infection_probability

ROOT = Path(__file__).resolve().parents[1]
Q_MAIN = 0.002  # chronic set point I* = d_E / q = 50 infected cells
Q_ORIGINAL = 0.1  # the preset's own value: I* = 1
MAPS = {Q_MAIN: ROOT / "results" / "regime_map.csv", Q_ORIGINAL: ROOT / "results" / "regime_map_original_q.csv"}
HORIZON_OUT = ROOT / "results" / "regime_map_horizon.csv"
BASE = PRESETS["establishment"].replace(q=Q_MAIN)
X0 = [1000, 0, 0, 5, 0, 10]  # T0 = T* = lambda / d_T
BETA = np.logspace(-5, -1, 25)
K = np.logspace(-3, 1, 25)
N = 1000
T_END = 200.0
T_LONG = 1000.0  # horizon check for the cells with intermediate persistence
N_LONG = 200
THRESHOLD = 100
E_CAP = 10_000_000  # safety only; nothing in this grid is expected to reach it


def params(beta, k, q=Q_MAIN):
    return BASE.replace(bw=float(beta), bm=0.8 * float(beta), k=float(k), q=float(q))


def r_eff(p, E0):
    """Reproduction number of the wild type with E0 effectors present from the start."""
    return r0_wild_type(p) * p.delta / (p.delta + p.k * E0)


def k_threshold(beta, E0, p=BASE):
    """Killing rate at which R_eff = 1 for a given infectivity; nan where even k = 0 gives R_eff < 1."""
    pi = infection_probability(beta, p.cw, target_cell_equilibrium(p))
    k = (pi * p.pw * (1.0 - p.mu) - p.delta) / E0
    return np.where(k > 0, k, np.nan)


def one(seed, p):
    r = simulate(p, X0, [0.0, T_END], seed_value=seed, stop_on_extinction=True, stop_above=[0, 0, 0, 0, 0, E_CAP])
    return r.cumulative_infections >= THRESHOLD, r.stop_reason != STOP_EXTINCT


def one_long(seed, p):
    """Virus present at day 200 and at day 1000 for the same realisation."""
    r = simulate(p, X0, [0.0, T_END, T_LONG], seed_value=seed, stop_on_extinction=True, stop_above=[0, 0, 0, 0, 0, E_CAP])
    present_200 = r.X[1, 1] >= 0 and r.X[1, 1:5].sum() > 0
    return present_200, r.stop_reason != STOP_EXTINCT


def deterministic(p):
    t = np.linspace(0.0, T_END, 2001)
    ode = solve_ode(p, X0, t)
    load = ode[:, 1:5].sum(axis=1)
    return float(load.max()), float(load[-1])


def classify_ode(peak, end, load0=sum(X0[1:5])):
    """Deterministic class: load never above the inoculum, grown then cleared, or persisting."""
    if end >= 1.0:
        return "chronic"
    return "cleared" if peak > load0 * (1 + 1e-6) else "extinction"  # 1e-6: floating-point rounding on a declining load, not growth


def run_map(q, seed):
    rows = []
    for k in K:
        for beta in BETA:
            p = params(beta, k, q)
            out = run_seeds(partial(one, p=p), range(seed, seed + N))
            est = sum(o[0] for o in out)
            per = sum(o[1] for o in out)
            peak, end = deterministic(p)
            lo_e, hi_e = wilson(est, N)
            lo_p, hi_p = wilson(per, N)
            rows.append({
                "bw": float(beta), "k": float(k), "q": float(q),
                "R0_wt": r_eff(p, 0.0), "R_eff": r_eff(p, X0[5]),
                "n": N, "n_established": est, "n_persist": per,
                "p_est": est / N, "p_est_low": float(lo_e), "p_est_high": float(hi_e),
                "p_persist": per / N, "p_persist_low": float(lo_p), "p_persist_high": float(hi_p),
                "p_est_theory": establishment_probability_decaying_effectors(p, X0),
                "p_est_theory_fixedE": establishment_probability(p, X0),
                "ode_peak_load": peak, "ode_load_end": end, "ode_class": classify_ode(peak, end),
                "seed_start": seed,
            })
            seed += N
    return rows, seed


def horizon_check(rows, seed):
    """Cells of the main map with intermediate persistence, followed to day 1000."""
    out = []
    for r in rows:
        if not 0.02 < r["p_persist"] < 0.98:
            continue
        res = run_seeds(partial(one_long, p=params(r["bw"], r["k"], r["q"])), range(seed, seed + N_LONG))
        k200, k1000 = sum(o[0] for o in res), sum(o[1] for o in res)
        out.append({"bw": r["bw"], "k": r["k"], "q": r["q"], "n": N_LONG, "p_persist_main": r["p_persist"],
                    "p_persist_200": k200 / N_LONG, "p_persist_1000": k1000 / N_LONG, "seed_start": seed})
        seed += N_LONG
    return out, seed


def main():
    started = time.time()
    one(0, params(1e-3, 0.1))
    seed = 4_000_000
    main_rows = None
    for q, out in MAPS.items():
        rows, seed = run_map(q, seed)
        if q == Q_MAIN:
            main_rows = rows
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        write_meta(out.with_suffix(".meta.json"), provenance(started, script=Path(__file__).name, q=q, n_per_cell=N, threshold=THRESHOLD, t_end=T_END, x0=X0, argv=sys.argv))
        classes = {c: sum(r["ode_class"] == c for r in rows) for c in ("extinction", "cleared", "chronic")}
        mixed_est = sum(0.05 < r["p_est"] < 0.95 for r in rows)
        mixed_per = sum(0.05 < r["p_persist"] < 0.95 for r in rows)
        persist_any = sum(r["p_persist"] > 0 for r in rows)
        print(f"q={q}: {len(rows)} cells, {time.time() - started:.0f} s elapsed; deterministic classes {classes}; "
              f"cells with intermediate establishment {mixed_est}, intermediate persistence {mixed_per}, any persistence {persist_any}", flush=True)
    hz, seed = horizon_check(main_rows, seed)
    with open(HORIZON_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hz[0]))
        w.writeheader()
        w.writerows(hz)
    write_meta(HORIZON_OUT.with_suffix(".meta.json"), provenance(started, script=Path(__file__).name, n_per_cell=N_LONG, t_long=T_LONG, argv=sys.argv))
    d = np.array([r["p_persist_1000"] - r["p_persist_200"] for r in hz])
    print(f"horizon check: {len(hz)} cells followed to day {T_LONG:g}; persistence at 1000 minus at 200: mean {d.mean():+.3f}, min {d.min():+.3f}; {time.time() - started:.0f} s total")


if __name__ == "__main__":
    main()
