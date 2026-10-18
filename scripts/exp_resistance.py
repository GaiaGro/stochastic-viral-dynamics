"""WP4: competitive release and the probability that drug resistance emerges.

"resistance" preset: R0 is 4.8 for the wild type and 3.9 for the mutant, so the
drug suppresses the wild type only above efficacy 1 - 1/4.8 = 0.79. It is the
notebook's therapy preset with a slower, weaker effector response, because the
notebook's deterministic chronic infection passes through 3e-8 virions and no
stochastic realisation survives that trough (recorded in
results/preset_viability.csv). Each realisation starts from 10 wild-type-infected
cells and 10 effectors, runs untreated to t_treat, then with an infection-blocking
drug of efficacy eps for H = 120 days.

Emergence: at least 100 mutant infection events during therapy, that is a mutant
lineage that established under treatment. Pre-existing: a mutant cell or virion
present at t_treat; otherwise de novo. Also recorded: mutant and wild type present
at the horizon, mutant dominant at the horizon, and the time at which mutant
virions first outnumber wild-type virions.

Theory per realisation, in the spirit of Alexander and Bonhoeffer (2012): the
environment a rare mutant lineage faces is the deterministic post-treatment path
of the wild type alone (mutant removed, no mutation), because while the lineage is
small it does not change that environment and once it is large its fate is
decided. Lineages present at t_treat survive with the one-type branching
probability at the treatment-start environment (lower bound) or at the most
favourable point of that path (upper bound); de novo mutants arrive as a Poisson
stream from the wild-type load along the path, each founding a lineage with the
probability of its birth environment.

A system-size sweep at one cell scales every population together
(lam, 1/beta, 1/q, 1/k and the initial state by Omega) and shows where the
deterministic prediction of certain takeover becomes right.

Writes results/resistance.csv (one row per grid cell), results/resistance_runs.csv
(one row per realisation), results/resistance_system_size.csv,
results/preset_viability.csv and results/competitive_release_ensemble.npz
(quantiles over time for one cell, used by Figure 6).
"""

import csv
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

from viraldyn import PRESETS, simulate, solve_ode
from viraldyn.ensembles import provenance, run_seeds, write_meta
from viraldyn.model import E, IM, IW, T, VM, VW
from viraldyn.ssa import STOP_EXTINCT
from viraldyn.stats import wilson
from viraldyn.theory import mutant_lineage_probabilities

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "resistance.csv"
RUNS = ROOT / "results" / "resistance_runs.csv"
ENSEMBLE = ROOT / "results" / "competitive_release_ensemble.npz"
VIABILITY = ROOT / "results" / "preset_viability.csv"
SYSTEM_SIZE = ROOT / "results" / "resistance_system_size.csv"
BASE = PRESETS["resistance"]
X0 = [1000, 10, 0, 0, 0, 10]
T_TREAT = (5, 10, 20, 30, 50)
EPS = (0.5, 0.7, 0.8, 0.9, 0.99)
OMEGA = (1, 3, 10, 30, 100)
H = 120.0
N = 1000
THRESHOLD = 100  # mutant infection events during therapy
STEP = 0.1
SHOWCASE = (30, 0.9)  # the cell whose ensemble is stored for Figure 6
QUANTILES = (5, 25, 50, 75, 95)


def grid(t0, t1):
    return np.linspace(t0, t1, int(round((t1 - t0) / STEP)) + 1)


def dominance_time(t, Vw, Vm):
    """First time mutant virions outnumber wild-type virions; nan if never."""
    idx = np.where((Vm > Vw) & (Vm > 0))[0]
    return float(t[idx[0]]) if idx.size else float("nan")


def scaled(omega):
    """Parameters and initial state with every population multiplied by omega."""
    p = BASE.replace(lam=BASE.lam * omega, bw=BASE.bw / omega, bm=BASE.bm / omega, q=BASE.q / omega, k=BASE.k / omega)
    return p, [int(v * omega) for v in X0]


def theory(p_drug, state, t_treat):
    """Emergence probability band for one realisation from its state at treatment start.
    Returns (p_pre_low, p_pre_high, p_denovo, p_low, p_high)."""
    t2 = np.linspace(t_treat, t_treat + H, 241)
    wt_only = np.asarray(state, dtype=float).copy()
    wt_only[IM] = wt_only[VM] = 0.0
    ode = solve_ode(p_drug.replace(mu=0.0), wt_only, t2)
    qV, qI = mutant_lineage_probabilities(p_drug, ode[:, T], ode[:, E])
    rate = p_drug.mu * (1.0 - p_drug.eps_p) * p_drug.pw * ode[:, IW]  # mutant virions produced per day
    p_denovo = 1.0 - np.exp(-np.trapezoid(rate * (1.0 - qV), t2))
    qV0, qI0 = mutant_lineage_probabilities(p_drug, state[T], state[E])
    j = int(np.argmax(1.0 - qV))
    p_pre_lo = 1.0 - float(qV0) ** state[VM] * float(qI0) ** state[IM]
    p_pre_hi = 1.0 - qV[j] ** state[VM] * qI[j] ** state[IM]
    return p_pre_lo, p_pre_hi, p_denovo, 1.0 - (1.0 - p_pre_lo) * (1.0 - p_denovo), 1.0 - (1.0 - p_pre_hi) * (1.0 - p_denovo)


def one(seed, t_treat, eps, keep=False, p=BASE, x0=X0):
    g1 = grid(0.0, t_treat)
    r1 = simulate(p, x0, g1, seed_value=seed, stop_on_extinction=True)
    if r1.stop_reason == STOP_EXTINCT:
        return None
    s = r1.x_end
    p_drug = p.replace(eps_b=eps)
    g2 = grid(t_treat, t_treat + H)
    r2 = simulate(p_drug, s, g2, stop_on_extinction=True)  # continues the seeded stream
    Vw2 = np.where(r2.X[:, VW] < 0, 0, r2.X[:, VW])
    Vm2 = np.where(r2.X[:, VM] < 0, 0, r2.X[:, VM])
    rec = {
        "seed": seed,
        "T0": int(s[T]), "Iw0": int(s[IW]), "Im0": int(s[IM]), "Vw0": int(s[VW]), "Vm0": int(s[VM]), "E0": int(s[E]),
        "pre": int(s[IM] + s[VM] > 0),
        "mut_infections": int(r2.c_end[1]),
        "emerged": int(r2.c_end[1] >= THRESHOLD),
        "Im_h": int(r2.x_end[IM]), "Vm_h": int(r2.x_end[VM]),
        "mut_h": int(r2.x_end[IM] + r2.x_end[VM] > 0),
        "wt_h": int(r2.x_end[IW] + r2.x_end[VW] > 0),
        "dominant_h": int(r2.x_end[VM] > r2.x_end[VW]),
        "t_dom": dominance_time(g2, Vw2, Vm2),
    }
    rec.update(zip(("p_pre_lo", "p_pre_hi", "p_denovo", "p_lo", "p_hi"), theory(p_drug, s.astype(float), t_treat)))
    traj = None
    if keep:
        Vw1 = np.where(r1.X[:, VW] < 0, 0, r1.X[:, VW])
        Vm1 = np.where(r1.X[:, VM] < 0, 0, r1.X[:, VM])
        traj = np.stack([np.concatenate([Vw1, Vw2[1:]]), np.concatenate([Vm1, Vm2[1:]])]).astype(np.int32)
    return rec, traj


def deterministic(t_treat, eps, omega=1):
    p, x0 = scaled(omega)
    s = solve_ode(p, x0, grid(0.0, t_treat))[-1]
    ode = solve_ode(p.replace(eps_b=eps), s, grid(t_treat, t_treat + H))
    return float(ode[-1, VM]), float(ode[-1, VW])


def _survives(seed, p):
    return simulate(p, X0, [0.0, 100.0], seed_value=seed, stop_on_extinction=True).stop_reason != STOP_EXTINCT


def viability(seed_start, n=1000):
    """Untreated infection in the notebook's therapy preset and in the resistance preset:
    deterministic first trough and set point against stochastic survival to day 100."""
    rows = []
    for name in ("therapy", "resistance"):
        p = PRESETS[name]
        t = np.linspace(0.0, 300.0, 30001)
        ode = solve_ode(p, X0, t)
        load = ode[:, IW] + ode[:, VW]
        ipk = int(np.argmax(load))
        itr = ipk + int(np.argmin(load[ipk : ipk + 10000]))
        alive = run_seeds(partial(_survives, p=p), range(seed_start, seed_start + n))
        k = int(sum(alive))
        lo, hi = wilson(k, n)
        rows.append({
            "preset": name, "n": n, "seed_start": seed_start,
            "ode_peak_load": float(load[ipk]), "ode_peak_day": float(t[ipk]),
            "ode_trough_load": float(load[itr]), "ode_trough_day": float(t[itr]),
            "ode_setpoint_Iw": float(ode[-1, IW]), "ode_setpoint_Vw": float(ode[-1, VW]), "ode_setpoint_E": float(ode[-1, E]),
            "n_alive_day100": k, "p_alive_day100": k / n, "wilson_low": float(lo), "wilson_high": float(hi),
        })
        seed_start += n
    return rows


def summarise(t_treat, eps, recs, n_extinct_before, seed_start, omega=1):
    n = len(recs)
    if n == 0:
        raise RuntimeError(f"no realisation reached treatment at t_treat={t_treat}: the pre-treatment infection is not viable")
    k = sum(r["emerged"] for r in recs)
    lo, hi = wilson(k, n)
    pre = [r for r in recs if r["pre"]]
    dn = [r for r in recs if not r["pre"]]
    t_dom = np.array([r["t_dom"] for r in recs]) - t_treat
    ode_vm, ode_vw = deterministic(t_treat, eps, omega)
    return {
        "t_treat": t_treat,
        "eps": eps,
        "omega": omega,
        "n": n,
        "n_extinct_before_treatment": n_extinct_before,
        "n_pre_existing": len(pre),
        "n_emerged": k,
        "n_emerged_pre": sum(r["emerged"] for r in pre),
        "n_emerged_denovo": sum(r["emerged"] for r in dn),
        "n_mutant_at_horizon": sum(r["mut_h"] for r in recs),
        "n_wt_at_horizon": sum(r["wt_h"] for r in recs),
        "n_cure": sum(1 for r in recs if not r["mut_h"] and not r["wt_h"]),
        "n_dominant_at_horizon": sum(r["dominant_h"] for r in recs),
        "p_emerge": k / n,
        "wilson_low": float(lo),
        "wilson_high": float(hi),
        "p_emerge_pre": sum(r["emerged"] for r in pre) / len(pre) if pre else float("nan"),
        "p_emerge_denovo": sum(r["emerged"] for r in dn) / len(dn) if dn else float("nan"),
        "p_theory_low": float(np.mean([r["p_lo"] for r in recs])),
        "p_theory_high": float(np.mean([r["p_hi"] for r in recs])),
        "p_theory_denovo": float(np.mean([r["p_denovo"] for r in recs])),
        "median_t_dominance": float(np.nanmedian(t_dom)) if np.isfinite(t_dom).any() else float("nan"),
        "median_Vm0": float(np.median([r["Vm0"] for r in recs])),
        "median_E0": float(np.median([r["E0"] for r in recs])),
        "median_T0": float(np.median([r["T0"] for r in recs])),
        "ode_Vm_horizon": ode_vm,
        "ode_Vw_horizon": ode_vw,
        "ode_takeover": int(ode_vm > 1.0),
        "seed_start": seed_start,
    }


def main():
    started = time.time()
    one(0, 30, 0.9)
    rows, run_rows = [], []
    seed = 3_000_000
    for t_treat in T_TREAT:
        for eps in EPS:
            keep = (t_treat, eps) == SHOWCASE
            out = run_seeds(partial(one, t_treat=t_treat, eps=eps, keep=keep), range(seed, seed + N))
            recs = [o[0] for o in out if o is not None]
            for r in recs:
                run_rows.append({"t_treat": t_treat, "eps": eps, **r})
            rows.append(summarise(t_treat, eps, recs, sum(o is None for o in out), seed))
            if keep:
                traj = np.stack([o[1] for o in out if o is not None])  # (n, 2, time)
                t = np.concatenate([grid(0.0, t_treat), grid(t_treat, t_treat + H)[1:]])
                np.savez_compressed(
                    ENSEMBLE,
                    t=t, t_treat=t_treat, eps=eps, n=len(recs), quantiles=np.array(QUANTILES),
                    Vw_q=np.percentile(traj[:, 0], QUANTILES, axis=0), Vm_q=np.percentile(traj[:, 1], QUANTILES, axis=0),
                    frac_wt=(traj[:, 0] > 0).mean(axis=0), frac_mut=(traj[:, 1] > 0).mean(axis=0),
                    frac_dominant=(traj[:, 1] > traj[:, 0]).mean(axis=0),
                )
            seed += N
    size_rows = []
    for omega in OMEGA:
        p, x0 = scaled(omega)
        out = run_seeds(partial(one, t_treat=SHOWCASE[0], eps=SHOWCASE[1], p=p, x0=x0), range(seed, seed + N))
        recs = [o[0] for o in out if o is not None]
        size_rows.append(summarise(SHOWCASE[0], SHOWCASE[1], recs, sum(o is None for o in out), seed, omega))
        seed += N
    via = viability(seed)
    for path, data in ((OUT, rows), (RUNS, run_rows), (SYSTEM_SIZE, size_rows), (VIABILITY, via)):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    write_meta(OUT.with_suffix(".meta.json"), provenance(started, script=Path(__file__).name, preset="resistance", n_per_cell=N, threshold=THRESHOLD, horizon=H, x0=X0, argv=sys.argv))
    for v in via:
        print(f"viability [{v['preset']}]: ODE trough {v['ode_trough_load']:.3g} at day {v['ode_trough_day']:.1f}, set point Iw={v['ode_setpoint_Iw']:.1f} Vw={v['ode_setpoint_Vw']:.0f}; stochastic survival to day 100: {v['p_alive_day100']:.3f} [{v['wilson_low']:.3f},{v['wilson_high']:.3f}]")
    print(f"{len(rows)} cells, {len(run_rows)} treated realisations, {time.time() - started:.0f} s")
    print(f"{'t_treat':>7} {'eps':>5} {'P(emerge)':>10} {'95% CI':>15} {'pre-exist':>9} {'P|pre':>6} {'P|denovo':>8} {'theory lo-hi':>13} {'ODE':>4} {'cure':>5} {'t_dom':>6}")
    for r in rows:
        print(f"{r['t_treat']:>7} {r['eps']:>5} {r['p_emerge']:>10.3f} [{r['wilson_low']:.3f},{r['wilson_high']:.3f}] {r['n_pre_existing'] / r['n']:>9.2f} {r['p_emerge_pre']:>6.2f} {r['p_emerge_denovo']:>8.2f} {r['p_theory_low']:>6.3f}-{r['p_theory_high']:.3f} {r['ode_takeover']:>4} {r['n_cure'] / r['n']:>5.2f} {r['median_t_dominance']:>6.1f}")
    print("system size at the showcase cell:")
    for r in size_rows:
        print(f"  omega={r['omega']:>3}  P(emerge)={r['p_emerge']:.3f} [{r['wilson_low']:.3f},{r['wilson_high']:.3f}]  pre-existing {r['n_pre_existing'] / r['n']:.2f}  theory {r['p_theory_low']:.3f}-{r['p_theory_high']:.3f}  ODE takeover {r['ode_takeover']}  cure {r['n_cure'] / r['n']:.2f}  extinct before treatment {r['n_extinct_before_treatment']}")


if __name__ == "__main__":
    main()
