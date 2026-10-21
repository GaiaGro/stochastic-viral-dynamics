"""Competitive release and the probability that drug resistance emerges.

"resistance" preset: R0 is 4.8 for the wild type and 3.9 for the mutant, so the
drug suppresses the wild type only above the efficacy eps* = 0.798 at which
R0(eps) = 1 (not 1 - 1/R0 = 0.79, because of virion consumption). It is the
therapy preset with a slower, weaker effector response, because the therapy
preset's deterministic chronic infection passes through 3e-8 virions and no
stochastic realisation survives that trough (results/preset_viability.csv). Each
realisation starts at the uninfected equilibrium T* = 1000 with 10 wild-type-
infected cells and 10 effectors, a primary infection into a naive host, runs
untreated to t_treat, then with an infection-blocking drug of efficacy eps for
H = 120 days.

Emergence: at least 100 mutant infection events during therapy, that is a mutant
lineage that established under treatment (the probability is the same for any
threshold between 20 and 500; the CSV carries the alternatives). Pre-existing: a
mutant cell or virion present at t_treat; otherwise de novo. Also recorded: mutant
and wild type present at the horizon, mutant dominant at the horizon, the time at
which mutant virions first outnumber wild-type virions, and the wild type's
extinction time under therapy.

Theory per realisation (Alexander and Bonhoeffer 2012 in spirit): the environment
a rare mutant lineage faces is the deterministic post-treatment path of the wild
type alone (mutant removed, no mutation). A mutant-infected-cell lineage is a
linear birth-death process with birth p_m pi_m(T(t)) and death delta + k E(t), and
Kendall's formula gives the extinction probability of a lineage founded at any
time on that path. Pre-existing lineages are those present at t_treat; de novo
mutants arrive at rate mu (1-eps_p) p_w Iw(t) and each founds a surviving lineage
with the probability of its birth time. The arrival integral runs to the horizon
(unconditional) or only while the wild type of that realisation was alive
(conditional), since a wild type that has died out produces no mutants.

Also written: a system-size sweep at one cell (lambda, 1/beta, 1/q, 1/k and the
initial state scaled by Omega), a sensitivity check of the two immune parameters
that define the preset, a longer horizon for the cells where the wild type
persists, and the viability contrast between the therapy preset and the
resistance preset.

Files: results/resistance.csv (one row per grid cell), results/resistance_runs.csv
(one row per realisation), results/resistance_system_size.csv,
results/resistance_preset_sensitivity.csv, results/resistance_horizon.csv,
results/preset_viability.csv and results/competitive_release_ensemble.npz.
"""

import csv
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

from viraldyn import PRESETS, scale_system_size, simulate, solve_ode, suppression_efficacy
from viraldyn.ensembles import provenance, run_seeds, write_meta
from viraldyn.model import E, IM, IW, T, VM, VW, immune_activation_threshold
from viraldyn.ssa import STOP_EXTINCT
from viraldyn.stats import wilson
from viraldyn.theory import infection_probability, lineage_extinction_from, mutant_lineage_rates

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "resistance.csv"
RUNS = ROOT / "results" / "resistance_runs.csv"
ENSEMBLE = ROOT / "results" / "competitive_release_ensemble.npz"
VIABILITY = ROOT / "results" / "preset_viability.csv"
SYSTEM_SIZE = ROOT / "results" / "resistance_system_size.csv"
SENSITIVITY = ROOT / "results" / "resistance_preset_sensitivity.csv"
HORIZON = ROOT / "results" / "resistance_horizon.csv"
BASE = PRESETS["resistance"]
X0 = [1000, 10, 0, 0, 0, 10]
T_TREAT = (5, 10, 20, 30, 50)
EPS = (0.5, 0.7, 0.8, 0.9, 0.99)
OMEGA = (1, 3, 10, 30, 100, 300)
K_LEVELS = (0.001, 0.003, 0.01)  # preset k = 0.003
Q_LEVELS = (0.01, 0.002, 0.0005)  # preset q = 0.002, I* = d_E/q = 10, 50, 200
H = 120.0
H_LONG = 360.0
N = 1000
THRESHOLD = 100  # mutant infection events during therapy
THRESHOLDS_CHECKED = (20, 50, 100, 200, 500)
STEP = 0.1
SHOWCASE = (30, 0.9)  # the cell whose ensemble is stored for Figure 6
QUANTILES = (5, 25, 50, 75, 95)
WT_SURVIVAL_WINDOW = 30.0  # days after treatment start; "survived the drug-induced trough"


def grid(t0, t1):
    return np.linspace(t0, t1, int(round((t1 - t0) / STEP)) + 1)


def scaled(omega):
    """Parameters and initial state with every population multiplied by omega."""
    return scale_system_size(BASE, omega), [int(v * omega) for v in X0]


def dominance_time(t, Vw, Vm):
    """First time mutant virions outnumber wild-type virions; nan if never."""
    idx = np.where((Vm > Vw) & (Vm > 0))[0]
    return float(t[idx[0]]) if idx.size else float("nan")


def theory(p_drug, state, t_treat, horizon=H, t_wt_end=np.inf):
    """Emergence probability for one realisation from its state at treatment start;
    see the module docstring. Returns a dict with the pre-existing term, the de novo
    term (unconditional and conditional on the wild type's survival time) and the
    totals."""
    t2 = grid(t_treat, t_treat + horizon)
    wt_only = np.asarray(state, dtype=float).copy()
    wt_only[IM] = wt_only[VM] = 0.0
    ode = solve_ode(p_drug.replace(mu=0.0), wt_only, t2)
    birth, death = mutant_lineage_rates(p_drug, ode[:, T], ode[:, E])
    p0 = lineage_extinction_from(t2, birth, death)  # extinction of a mutant-cell lineage founded at each time
    u_virion = infection_probability(p_drug.bm, p_drug.cm, ode[:, T]) * (1.0 - p0)
    p_pre = 1.0 - p0[0] ** state[IM] * (1.0 - u_virion[0]) ** state[VM]
    arrivals = p_drug.mu * (1.0 - p_drug.eps_p) * p_drug.pw * ode[:, IW] * u_virion
    lam = float(np.trapezoid(arrivals, t2))
    alive = t2 <= t_wt_end
    lam_cond = float(np.trapezoid(arrivals[alive], t2[alive])) if alive.sum() >= 2 else 0.0
    p_dn, p_dn_cond = 1.0 - np.exp(-lam), 1.0 - np.exp(-lam_cond)
    return {
        "p_pre": p_pre,
        "p_denovo": p_dn,
        "p_denovo_cond": p_dn_cond,
        "p_theory": 1.0 - (1.0 - p_pre) * (1.0 - p_dn),
        "p_theory_cond": 1.0 - (1.0 - p_pre) * (1.0 - p_dn_cond),
    }


def one(seed, t_treat, eps, keep=False, p=BASE, x0=X0, horizon=H):
    g1 = grid(0.0, t_treat)
    r1 = simulate(p, x0, g1, seed_value=seed, stop_on_extinction=True)
    if r1.stop_reason == STOP_EXTINCT:
        return None
    s = r1.x_end
    p_drug = p.replace(eps_b=eps)
    g2 = grid(t_treat, t_treat + horizon)
    r2 = simulate(p_drug, s, g2, stop_on_extinction=True)  # continues the seeded stream
    Vw2 = np.where(r2.X[:, VW] < 0, 0, r2.X[:, VW])
    Vm2 = np.where(r2.X[:, VM] < 0, 0, r2.X[:, VM])
    t_wt_ext = float(r2.t_ext[0]) if s[IW] + s[VW] > 0 else float(t_treat)  # nan if alive at the horizon
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
        "t_wt_ext": t_wt_ext,
        "wt_survived_window": int(not (t_wt_ext <= t_treat + WT_SURVIVAL_WINDOW)),
    }
    rec.update(theory(p_drug, s.astype(float), t_treat, horizon, t_wt_end=t_wt_ext if np.isfinite(t_wt_ext) else np.inf))
    traj = None
    if keep:
        Vw1 = np.where(r1.X[:, VW] < 0, 0, r1.X[:, VW])
        Vm1 = np.where(r1.X[:, VM] < 0, 0, r1.X[:, VM])
        traj = np.stack([np.concatenate([Vw1, Vw2[1:]]), np.concatenate([Vm1, Vm2[1:]])]).astype(np.int32)
    return rec, traj


def deterministic(t_treat, eps, p, x0, horizon=H):
    s = solve_ode(p, x0, grid(0.0, t_treat))[-1]
    ode = solve_ode(p.replace(eps_b=eps), s, grid(t_treat, t_treat + horizon))
    return float(ode[-1, VM]), float(ode[-1, VW])


def run_cell(t_treat, eps, seed, p=BASE, x0=X0, horizon=H, keep=False):
    out = run_seeds(partial(one, t_treat=t_treat, eps=eps, keep=keep, p=p, x0=x0, horizon=horizon), range(seed, seed + N))
    recs = [o[0] for o in out if o is not None]
    trajs = [o[1] for o in out if o is not None and o[1] is not None]
    return recs, sum(o is None for o in out), trajs


def summarise(t_treat, eps, recs, n_extinct_before, seed_start, p=BASE, x0=X0, omega=1, horizon=H):
    n = len(recs)
    if n == 0:
        # the pre-treatment infection is not viable for these parameters: that is the result
        row = {k: float("nan") for k in _COLUMNS}
        row.update({"t_treat": t_treat, "eps": eps, "omega": omega, "horizon": horizon, "k": p.k, "q": p.q,
                    "n": 0, "n_extinct_before_treatment": n_extinct_before, "seed_start": seed_start})
        return row
    k = sum(r["emerged"] for r in recs)
    lo, hi = wilson(k, n)
    pre = [r for r in recs if r["pre"]]
    dn = [r for r in recs if not r["pre"]]
    t_dom = np.array([r["t_dom"] for r in recs]) - t_treat
    mi = np.array([r["mut_infections"] for r in recs])
    ode_vm, ode_vw = deterministic(t_treat, eps, p, x0, horizon)
    return {
        "t_treat": t_treat, "eps": eps, "omega": omega, "horizon": horizon, "k": p.k, "q": p.q,
        "n": n, "n_extinct_before_treatment": n_extinct_before,
        "n_pre_existing": len(pre), "n_emerged": k,
        "n_emerged_pre": sum(r["emerged"] for r in pre), "n_emerged_denovo": sum(r["emerged"] for r in dn),
        "n_mutant_at_horizon": sum(r["mut_h"] for r in recs), "n_wt_at_horizon": sum(r["wt_h"] for r in recs),
        "n_cure": sum(1 for r in recs if not r["mut_h"] and not r["wt_h"]),
        "n_dominant_at_horizon": sum(r["dominant_h"] for r in recs),
        "n_wt_survived_window": sum(r["wt_survived_window"] for r in recs),
        "p_emerge": k / n, "wilson_low": float(lo), "wilson_high": float(hi),
        **{f"p_emerge_threshold_{th}": float((mi >= th).mean()) for th in THRESHOLDS_CHECKED},
        "p_emerge_threshold_scaled": float((mi >= THRESHOLD * omega).mean()),
        "p_emerge_pre": sum(r["emerged"] for r in pre) / len(pre) if pre else float("nan"),
        "p_emerge_denovo": sum(r["emerged"] for r in dn) / len(dn) if dn else float("nan"),
        "p_theory": float(np.mean([r["p_theory"] for r in recs])),
        "p_theory_cond": float(np.mean([r["p_theory_cond"] for r in recs])),
        "p_theory_pre": float(np.mean([r["p_pre"] for r in recs])),
        "p_theory_denovo": float(np.mean([r["p_denovo"] for r in recs])),
        "p_theory_denovo_cond": float(np.mean([r["p_denovo_cond"] for r in recs])),
        "n_dominance": int(np.isfinite(t_dom).sum()),
        "median_t_dominance": float(np.nanmedian(t_dom)) if np.isfinite(t_dom).any() else float("nan"),
        "median_Vm0": float(np.median([r["Vm0"] for r in recs])), "median_E0": float(np.median([r["E0"] for r in recs])), "median_T0": float(np.median([r["T0"] for r in recs])),
        "ode_Vm_horizon": ode_vm, "ode_Vw_horizon": ode_vw, "ode_takeover": int(ode_vm > 1.0),
        "seed_start": seed_start,
    }


_COLUMNS = (
    ["t_treat", "eps", "omega", "horizon", "k", "q", "n", "n_extinct_before_treatment", "n_pre_existing", "n_emerged",
     "n_emerged_pre", "n_emerged_denovo", "n_mutant_at_horizon", "n_wt_at_horizon", "n_cure", "n_dominant_at_horizon",
     "n_wt_survived_window", "p_emerge", "wilson_low", "wilson_high"]
    + [f"p_emerge_threshold_{th}" for th in THRESHOLDS_CHECKED]
    + ["p_emerge_threshold_scaled", "p_emerge_pre", "p_emerge_denovo", "p_theory", "p_theory_cond", "p_theory_pre",
       "p_theory_denovo", "p_theory_denovo_cond", "n_dominance", "median_t_dominance", "median_Vm0", "median_E0",
       "median_T0", "ode_Vm_horizon", "ode_Vw_horizon", "ode_takeover", "seed_start"]
)


def _survives(seed, p):
    return simulate(p, X0, [0.0, 100.0], seed_value=seed, stop_on_extinction=True).stop_reason != STOP_EXTINCT


def viability(seed_start, n=1000):
    """Untreated infection in the therapy preset and in the resistance preset:
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
            "preset": name, "n": n, "seed_start": seed_start, "I_star": immune_activation_threshold(p),
            "ode_peak_load": float(load[ipk]), "ode_peak_day": float(t[ipk]),
            "ode_trough_load": float(load[itr]), "ode_trough_day": float(t[itr]),
            "ode_setpoint_Iw": float(ode[-1, IW]), "ode_setpoint_Vw": float(ode[-1, VW]), "ode_setpoint_E": float(ode[-1, E]),
            "n_alive_day100": k, "p_alive_day100": k / n, "wilson_low": float(lo), "wilson_high": float(hi),
        })
        seed_start += n
    return rows


def write(path, data):
    """Each stage writes its file as soon as it is done, so a failure later loses nothing."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)


def main():
    started = time.time()
    one(0, 30, 0.9)
    seed = 3_000_000
    rows, run_rows = [], []
    for t_treat in T_TREAT:
        for eps in EPS:
            keep = (t_treat, eps) == SHOWCASE
            recs, n_dead, trajs = run_cell(t_treat, eps, seed, keep=keep)
            run_rows += [{"t_treat": t_treat, "eps": eps, **r} for r in recs]
            rows.append(summarise(t_treat, eps, recs, n_dead, seed))
            if keep:
                traj = np.stack(trajs)
                t = np.concatenate([grid(0.0, t_treat), grid(t_treat, t_treat + H)[1:]])
                np.savez_compressed(
                    ENSEMBLE,
                    t=t, t_treat=t_treat, eps=eps, n=len(recs), quantiles=np.array(QUANTILES),
                    Vw_q=np.percentile(traj[:, 0], QUANTILES, axis=0), Vm_q=np.percentile(traj[:, 1], QUANTILES, axis=0),
                    frac_wt=(traj[:, 0] > 0).mean(axis=0), frac_mut=(traj[:, 1] > 0).mean(axis=0),
                    frac_dominant=(traj[:, 1] > traj[:, 0]).mean(axis=0),
                )
            seed += N
    write(OUT, rows)
    write(RUNS, run_rows)
    print(f"grid done, {time.time() - started:.0f} s", flush=True)

    size_rows = []
    for omega in OMEGA:
        p, x0 = scaled(omega)
        recs, n_dead, _ = run_cell(*SHOWCASE, seed, p=p, x0=x0)
        size_rows.append(summarise(*SHOWCASE, recs, n_dead, seed, p=p, x0=x0, omega=omega))
        seed += N
        write(SYSTEM_SIZE, size_rows)
        print(f"omega {omega} done, {time.time() - started:.0f} s", flush=True)

    sens_rows = []
    for k in K_LEVELS:
        for q in Q_LEVELS:
            if k != BASE.k and q != BASE.q:
                continue  # one parameter at a time
            p = BASE.replace(k=k, q=q)
            recs, n_dead, _ = run_cell(*SHOWCASE, seed, p=p)
            sens_rows.append(summarise(*SHOWCASE, recs, n_dead, seed, p=p))
            seed += N
    write(SENSITIVITY, sens_rows)
    print(f"sensitivity done, {time.time() - started:.0f} s", flush=True)

    horizon_rows = []
    for t_treat in T_TREAT:
        recs, n_dead, _ = run_cell(t_treat, EPS[0], seed, horizon=H_LONG)
        horizon_rows.append(summarise(t_treat, EPS[0], recs, n_dead, seed, horizon=H_LONG))
        seed += N
    write(HORIZON, horizon_rows)
    via = viability(seed)
    write(VIABILITY, via)
    write_meta(OUT.with_suffix(".meta.json"), provenance(started, script=Path(__file__).name, preset="resistance", eps_star=suppression_efficacy(BASE), n_per_cell=N, threshold=THRESHOLD, horizon=H, x0=X0, argv=sys.argv))

    for v in via:
        print(f"viability [{v['preset']}]: I* = {v['I_star']:g}; ODE trough {v['ode_trough_load']:.3g} at day {v['ode_trough_day']:.1f}, set point Iw={v['ode_setpoint_Iw']:.1f} Vw={v['ode_setpoint_Vw']:.0f}; stochastic survival to day 100: {v['p_alive_day100']:.3f} [{v['wilson_low']:.3f},{v['wilson_high']:.3f}]")
    print(f"eps* = {suppression_efficacy(BASE):.4f}; {len(rows)} cells, {len(run_rows)} treated realisations, {time.time() - started:.0f} s")
    head = f"{'t_treat':>7} {'eps':>5} {'P(emerge)':>10} {'95% CI':>15} {'pre':>5} {'theory':>7} {'cond':>6} {'WT30d':>6} {'cure':>5} {'n_dom':>5} {'t_dom':>6}"
    print(head)
    for r in rows + size_rows + sens_rows + horizon_rows:
        tag = f"{r['t_treat']:>7} {r['eps']:>5}" if r["omega"] == 1 and r["k"] == BASE.k and r["q"] == BASE.q and r["horizon"] == H else f"om={r['omega']} k={r['k']:g} q={r['q']:g} H={r['horizon']:g}"
        if r["n"] == 0:
            print(f"{tag}  not viable: all {r['n_extinct_before_treatment']} realisations died before treatment")
            continue
        print(f"{tag} {r['p_emerge']:>10.3f} [{r['wilson_low']:.3f},{r['wilson_high']:.3f}] {r['n_pre_existing'] / r['n']:>5.2f} {r['p_theory']:>7.3f} {r['p_theory_cond']:>6.3f} {r['n_wt_survived_window'] / r['n']:>6.2f} {r['n_cure'] / r['n']:>5.2f} {r['n_dominance']:>5} {r['median_t_dominance']:>6.1f}")


if __name__ == "__main__":
    main()
