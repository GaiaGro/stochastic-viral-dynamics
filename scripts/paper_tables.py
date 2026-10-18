"""Generate the LaTeX fragments the manuscript includes: the parameter table of
the presets and a file of macros holding every number quoted in the text, each
read from the code or from a results CSV. Run with `make -C paper generated`.
Writes paper/generated/presets.tex and paper/generated/numbers.tex."""

import csv
from pathlib import Path

import numpy as np

from viraldyn import PRESETS, immune_activation_threshold, r0_mutant, r0_wild_type, suppression_efficacy, target_cell_equilibrium

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "generated"
RESULTS = ROOT / "results"

PARAMS = [
    ("lam", r"$\lambda$", "target-cell production", "cells/day"),
    ("dT", r"$d_T$", "target-cell death", "/day"),
    ("bw", r"$\beta_w$", "wild-type infectivity", "/(virion day)"),
    ("bm", r"$\beta_m$", "mutant infectivity", "/(virion day)"),
    ("pw", r"$p_w$", "wild-type virion production", "/(cell day)"),
    ("pm", r"$p_m$", "mutant virion production", "/(cell day)"),
    ("mu", r"$\mu$", "mutation probability per virion", ""),
    ("cw", r"$c_w$", "wild-type virion clearance", "/day"),
    ("cm", r"$c_m$", "mutant virion clearance", "/day"),
    ("delta", r"$\delta$", "infected-cell death", "/day"),
    ("k", r"$k$", "killing by effectors", "/(cell day)"),
    ("q", r"$q$", "effector proliferation", "/(cell day)"),
    ("dE", r"$d_E$", "effector death", "/day"),
]
PRESET_ORDER = ["acute", "establishment", "resistance", "chronic_large"]
PRESET_LABEL = {"acute": "acute", "establishment": "establishment", "resistance": "resistance", "chronic_large": r"resistance $\times 100$"}


def fmt(v):
    if v == 0:
        return "0"
    if abs(v) >= 1000 or abs(v) < 0.01:
        s = f"{v:.3g}"
        if "e" in s:
            m, e = s.split("e")
            return rf"${m}\times 10^{{{int(e)}}}$"
        return s
    return f"{v:g}"


def rows(csv_name):
    with open(RESULTS / csv_name) as f:
        return list(csv.DictReader(f))


def presets_table():
    lines = [r"\begin{tabular}{llr" + "r" * len(PRESET_ORDER) + "}", r"\toprule",
             "symbol & meaning & unit & " + " & ".join(PRESET_LABEL[n] for n in PRESET_ORDER) + r" \\", r"\midrule"]
    for key, sym, meaning, unit in PARAMS:
        lines.append(f"{sym} & {meaning} & {unit} & " + " & ".join(fmt(getattr(PRESETS[n], key)) for n in PRESET_ORDER) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"$T^* = \lambda/d_T$ & uninfected equilibrium & cells & " + " & ".join(fmt(target_cell_equilibrium(PRESETS[n])) for n in PRESET_ORDER) + r" \\")
    lines.append(r"$R_0$ & wild type / mutant & & " + " & ".join(f"{r0_wild_type(PRESETS[n]):.2f} / {r0_mutant(PRESETS[n]):.2f}" for n in PRESET_ORDER) + r" \\")
    lines.append(r"$I^* = d_E/q$ & activation threshold, set point & cells & " + " & ".join(fmt(immune_activation_threshold(PRESETS[n])) for n in PRESET_ORDER) + r" \\")
    lines.append(r"$\varepsilon^*$ & efficacy at which $R_0 = 1$ & & " + " & ".join(f"{suppression_efficacy(PRESETS[n]):.3f}" for n in PRESET_ORDER) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def macro(name, value):
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def pct(x, digits=1):
    return f"{100 * float(x):.{digits}f}"


def numbers():
    """Every number quoted in the manuscript, as a LaTeX macro, read from the code or a CSV."""
    out = []
    m = lambda name, value: out.append(macro(name, value))
    sel = lambda rows, **kw: [r for r in rows if all((r[k] == v) if isinstance(v, str) else abs(float(r[k]) - v) < 1e-9 for k, v in kw.items())]
    f = float

    # presets
    pr = PRESETS["resistance"]
    m("RzeroWt", f"{r0_wild_type(pr):.2f}")
    m("RzeroMut", f"{r0_mutant(pr):.2f}")
    m("RzeroWtWeak", f"{r0_wild_type(pr.replace(eps_b=0.5)):.1f}")
    m("RzeroWtKnife", f"{r0_wild_type(pr.replace(eps_b=0.8)):.2f}")
    m("EpsStar", f"{suppression_efficacy(pr):.3f}")
    m("RzeroEstablishment", f"{r0_wild_type(PRESETS['establishment']):.1f}")
    m("RzeroAcute", f"{r0_wild_type(PRESETS['acute']):.1f}")
    from viraldyn.theory import mutant_lineage_probabilities
    via = {r["preset"]: r for r in rows("preset_viability.csv")}
    qV, _ = mutant_lineage_probabilities(pr, 1000.0, f(via["resistance"]["ode_setpoint_E"]))
    m("MutVirionEstablish", f"{1 - float(qV):.3f}")

    # viability of the two therapy presets
    m("TherapyTrough", fmt(f(via["therapy"]["ode_trough_load"])))
    m("TherapyTroughDay", f"{f(via['therapy']['ode_trough_day']):.1f}")
    m("TherapySetIw", f"{f(via['therapy']['ode_setpoint_Iw']):.0f}")
    m("TherapyAlive", via["therapy"]["n_alive_day100"])
    m("ViabilityN", via["therapy"]["n"])
    m("ResTrough", f"{f(via['resistance']['ode_trough_load']):.0f}")
    m("ResSetIw", f"{f(via['resistance']['ode_setpoint_Iw']):.0f}")
    m("ResSetVw", f"{f(via['resistance']['ode_setpoint_Vw']):.0f}")
    m("ResSetE", f"{f(via['resistance']['ode_setpoint_E']):.0f}")
    m("ResistanceAlive", via["resistance"]["n_alive_day100"])

    # baseline validation and fluctuations
    base = {r["species"]: r for r in rows("baseline_validation.csv")}
    m("BaselineN", base["T"]["n"])
    m("BaselineMaxLogDev", f"{max(f(base[s]['max_abs_log10_dev_median']) for s in ('T', 'Iw', 'Vw', 'E')):.3f}")
    m("BaselineMutLogDev", f"{f(base['Im']['max_abs_log10_dev_median']):.2f}")
    m("BaselineMutMin", f"{f(base['Im']['ode_min_after_day1']):.1f}")
    m("BaselineMutMax", f"{f(base['Im']['ode_max_after_day1']):.0f}")
    m("BaselineExtinct", base["T"]["n_extinct_at_end"])
    fl = rows("fluctuations.csv")
    m("FlucN", fl[0]["n"])
    slopes, reldev = [], []
    for sp in ("T", "Iw", "Vw", "E"):
        s = sorted(sel(fl, species=sp), key=lambda r: f(r["omega"]))
        m(f"Cv{sp}One", f"{f(s[0]['cv_sim']):.2f}")
        m(f"Cv{sp}Hundred", f"{f(s[-1]['cv_sim']):.3f}")
        slopes.append(np.polyfit(np.log([f(r["omega"]) for r in s]), np.log([f(r["cv_sim"]) for r in s]), 1)[0])
        reldev.append(max(abs(f(r["cv_sim"]) / f(r["cv_lna"]) - 1) for r in s))
    m("CvSlopeMin", f"{min(slopes):.3f}")
    m("CvSlopeMax", f"{max(slopes):.3f}")
    m("CvMaxRelDev", pct(max(reldev), 0))

    # establishment
    est = rows("establishment.csv")
    m("EstN", est[0]["n"])
    m("EstConditions", str(len(est)))
    m("EstInsideInterval", str(sum(f(r["wilson_low"]) <= f(r["p_theory"]) <= f(r["wilson_high"]) for r in est)))
    k = np.array([f(r["k_established"]) for r in est]); n = np.array([f(r["n"]) for r in est]); pt = np.array([f(r["p_theory"]) for r in est])
    m("EstPooledZ", f"{(k - n * pt).sum() / np.sqrt((n * pt * (1 - pt)).sum()):+.2f}")
    m("EstMaxDev", f"{max(abs(f(r['p_hat']) - f(r['p_theory'])) for r in est):.3f}")
    one = sel(est, sweep="V0", E0=0, V0=1)[0]
    m("EstOneVirionTheory", f"{f(one['p_theory']):.3f}")
    m("EstOneVirionObserved", f"{f(one['p_hat']):.3f}")
    m("EstOneVirionLow", f"{f(one['wilson_low']):.3f}")
    m("EstOneVirionHigh", f"{f(one['wilson_high']):.3f}")
    fifty = sel(est, sweep="V0", E0=50, V0=1)[0]
    m("EstFiftyObserved", f"{f(fifty['p_hat']):.3f}")
    m("EstFiftyTheory", f"{f(fifty['p_theory']):.3f}")
    for V0, tag in ((1, "One"), (5, "Five")):
        s = sorted(sel(est, sweep="beta", V0=V0), key=lambda r: f(r["R0_wt"]))
        near3 = min(s, key=lambda r: abs(f(r["R0_wt"]) - 3))
        half = next(r for r in s if f(r["p_hat"]) >= 0.5)
        if V0 == 1:
            m("EstRzeroNearThree", f"{f(near3['R0_wt']):.1f}")
        m(f"Est{tag}AtThree", pct(near3["p_hat"], 1))
        m(f"Est{tag}HalfRzero", f"{f(half['R0_wt']):.0f}")
    thr = rows("establishment_threshold.csv")
    m("EstThresholdSpread", f"{max(max(f(r[c]) for c in r if c.startswith('p_reach_')) - min(f(r[c]) for c in r if c.startswith('p_reach_')) for r in thr):.4f}")
    m("EstThresholdRzeroMin", f"{min(f(r['R0_wt']) for r in thr):.1f}")
    m("EstThresholdRzeroMax", f"{max(f(r['R0_wt']) for r in thr):.0f}")

    # immune activation failure
    imm = rows("immune_activation.csv")
    m("ImmN", imm[0]["n"])
    head = sel(imm, sweep="E0", variant="full", E0=1, dE=0.1)[0]
    m("ImmFailObserved", pct(head["p_fail"]))
    m("ImmFailLow", pct(head["wilson_low"]))
    m("ImmFailHigh", pct(head["wilson_high"]))
    m("ImmFailTheory", pct(head["p_theory"]))
    m("ImmFailN", head["n_immune_failure"])
    m("ImmEstN", head["n_established"])
    m("ImmFailUnconditional", pct(f(head["n_immune_failure"]) / f(head["n"])))
    two = sel(imm, sweep="E0", variant="full", E0=2, dE=0.1)[0]
    m("ImmFailTwo", pct(two["p_fail"]))
    m("ImmFailTwoTheory", pct(two["p_theory"]))
    three = sel(imm, sweep="E0", variant="full", E0=3, dE=0.1)[0]
    m("ImmFailThreeN", three["n_immune_failure"])
    m("ImmEstThreeN", three["n_established"])
    de = sorted(sel(imm, sweep="dE", variant="full"), key=lambda r: f(r["dE"]))
    m("ImmDeMin", f"{f(de[0]['dE']):.2f}")
    m("ImmDeMax", f"{f(de[-1]['dE']):.0f}")
    m("ImmFailDeMin", pct(de[0]["p_fail"]))
    m("ImmFailDeMax", pct(de[-1]["p_fail"], 0))
    m("ImmFailDeMaxEarly", pct(de[-1]["p_fail_early"]))
    m("ImmFailDeMaxTheory", pct(de[-1]["p_theory"]))
    m("ImmCollapseDeMaxN", de[-1]["n_failure_collapse"])
    m("ImmCollapseDeMaxEst", de[-1]["n_established"])
    m("ImmCollapseDeMaxPct", pct(f(de[-1]["n_failure_collapse"]) / f(de[-1]["n_established"])))
    sub = [r for r in de if f(r["dE"]) < 1.0]
    m("ImmCollapseBelowOne", str(sum(int(f(r["n_failure_collapse"])) for r in sub)))
    mid = next(r for r in de if 1.5 < f(r["dE"]) < 2)
    m("ImmDeMid", f"{f(mid['dE']):.2f}")
    m("ImmCollapseMidN", mid["n_failure_collapse"])
    m("ImmCollapseMidEst", mid["n_established"])
    q = sorted(sel(imm, sweep="q", variant="full"), key=lambda r: f(r["q"]))
    m("ImmFailQLow", pct(q[0]["p_fail"]))
    m("ImmFailQHigh", pct(q[-1]["p_fail"]))
    def agreement(group, key):
        k = np.array([f(r["n_immune_failure"]) for r in group]); n = np.array([f(r["n_established"]) for r in group]); pt = np.array([f(r[key]) for r in group])
        outside = sum(not (f(r["wilson_low"]) <= f(r[key]) <= f(r["wilson_high"])) for r in group)
        return outside, (k - n * pt).sum() / np.sqrt((n * pt * (1 - pt)).sum())
    full = [r for r in imm if r["variant"] == "full"]; k0 = [r for r in imm if r["variant"] == "k0"]
    m("ImmConditions", str(len(full)))
    for tag, group, key in (("FullDet", full, "p_theory"), ("KzeroDet", k0, "p_theory"), ("KzeroPath", k0, "p_theory_per_path")):
        o, z = agreement(group, key)
        m(f"Imm{tag}Outside", str(o))
        m(f"Imm{tag}Z", f"{z:+.1f}")
    col = min(rows("immune_collapse.csv"), key=lambda r: f(r["dE"]))  # the d_E sweep point nearest 0.1
    m("ImmCollapseDe", f"{f(col['dE']):.3f}")
    m("ImmFailDeathDay", f"{f(col['early_median_t_effector_death']):.2f}")
    m("ImmNoImmuneSetPoint", f"{f(col['no_immune_set_point']):.0f}")
    m("ImmFailLoadSixty", f"{f(col['early_median_load_day60']):.0f}")
    m("ImmFailEarlyN", col["early_n"])
    m("ImmFailClearedByTwoHundred", col["early_n_extinct_by_day200"])
    fates = np.load(RESULTS / "immune_activation_trajectories.npz")["fate"]
    m("TrajBottleneck", str(int((fates == "bottleneck").sum())))
    m("TrajCleared", str(int((fates == "cleared").sum())))
    m("TrajFailure", str(int((fates == "immune_failure").sum())))

    # resistance
    res = rows("resistance.csv")
    m("ResN", str(max(int(r["n"]) for r in res)))
    m("ResCells", str(len(res)))
    m("OdeTakeoverCells", str(sum(int(f(r["ode_takeover"])) for r in res)))
    def band(group, key, digits=1, denom=None):
        vals = [f(r[key]) / (f(r[denom]) if denom else 1.0) for r in group]
        return pct(min(vals), digits), pct(max(vals), digits)
    for eps, tag in ((0.5, "Weak"), (0.7, "Mid")):
        g = sel(res, eps=eps)
        lo, hi = band(g, "p_emerge"); m(f"Emerge{tag}Min", lo); m(f"Emerge{tag}Max", hi)
        lo, hi = band(g, "p_theory_cond"); m(f"Theory{tag}CondMin", lo); m(f"Theory{tag}CondMax", hi)
        lo, hi = band(g, "p_theory"); m(f"Theory{tag}Min", lo); m(f"Theory{tag}Max", hi)
        lo, hi = band(g, "n_cure", 0, "n"); m(f"Cure{tag}Min", lo); m(f"Cure{tag}Max", hi)
        lo, hi = band(g, "n_wt_at_horizon", 0, "n"); m(f"WtHorizon{tag}Min", lo); m(f"WtHorizon{tag}Max", hi)
        lo, hi = band(g, "n_wt_survived_window", 0, "n"); m(f"WtWindow{tag}Min", lo); m(f"WtWindow{tag}Max", hi)
        m(f"Dominance{tag}Min", f"{min(f(r['median_t_dominance']) for r in g):.0f}")
        m(f"Dominance{tag}Max", f"{max(f(r['median_t_dominance']) for r in g):.0f}")
    hi_eps = [r for r in res if f(r["eps"]) >= 0.8]
    lo, hi = band(hi_eps, "p_emerge"); m("EmergeSuppressedMin", lo); m("EmergeSuppressedMax", hi)
    lo, hi = band(hi_eps, "p_theory_cond"); m("TheorySuppressedCondMin", lo); m("TheorySuppressedCondMax", hi)
    lo, hi = band(hi_eps, "n_cure", 1, "n"); m("CureSuppressedMin", lo); m("CureSuppressedMax", hi)
    pre = [f(r["n_pre_existing"]) / f(r["n"]) for r in res]
    m("PreMin", pct(min(pre), 0)); m("PreMax", pct(max(pre), 0))
    m("PreDayFive", pct(np.mean([f(r["n_pre_existing"]) / f(r["n"]) for r in sel(res, t_treat=5)]), 0))
    weak = sel(res, eps=0.5)
    lo, hi = band(weak, "p_emerge_pre"); m("EmergeWeakPreMin", lo); m("EmergeWeakPreMax", hi)
    lo, hi = band(weak, "p_emerge_denovo"); m("EmergeWeakDenovoMin", lo); m("EmergeWeakDenovoMax", hi)
    m("ThresholdSpreadWeak", f"{max(max(f(r[f'p_emerge_threshold_{t}']) for t in (20, 50, 100, 200, 500)) - min(f(r[f'p_emerge_threshold_{t}']) for t in (20, 50, 100, 200, 500)) for r in weak):.3f}")
    m("ThresholdSpreadRest", f"{max(max(f(r[f'p_emerge_threshold_{t}']) for t in (20, 50, 100, 200, 500)) - min(f(r[f'p_emerge_threshold_{t}']) for t in (20, 50, 100, 200, 500)) for r in res if f(r['eps']) >= 0.7):.3f}")
    short = [r for r in res if int(r["n"]) < 1000]
    m("ResShortCells", str(len(short)))
    m("ResShortN", short[0]["n"] if short else "1000")
    e30 = np.load(RESULTS / "competitive_release_ensemble.npz")
    i_treat = int(np.searchsorted(e30["t"], f(e30["t_treat"])))
    m("EnsembleN", str(int(e30["n"])))
    m("EnsemblePreAtTreat", pct(e30["frac_mut"][i_treat]))
    m("EnsembleMutAtEnd", pct(e30["frac_mut"][-1]))
    m("EnsembleWtAtEnd", pct(e30["frac_wt"][-1]))
    size = sorted(rows("resistance_system_size.csv"), key=lambda r: f(r["omega"]))
    m("OmegaList", ", ".join(f"{f(r['omega']):g}" for r in size))
    m("EmergeOmegaList", ", ".join(pct(r["p_emerge"], 1 if f(r["p_emerge"]) < 0.1 else 0) for r in size))
    m("TheoryOmegaList", ", ".join(pct(r["p_theory_cond"], 1 if f(r["p_theory_cond"]) < 0.1 else 0) for r in size))
    m("CureOmegaList", ", ".join(pct(f(r["n_cure"]) / f(r["n"]), 0) for r in size))
    m("OmegaMax", f"{f(size[-1]['omega']):g}")
    m("EmergeOmegaMax", pct(size[-1]["p_emerge"], 0))
    m("CureOmegaMax", pct(f(size[-1]["n_cure"]) / f(size[-1]["n"]), 0))
    m("ScaledThresholdMaxDiff", f"{max(abs(f(r['p_emerge_threshold_scaled']) - f(r['p_emerge'])) for r in size):.3f}")
    sens = rows("resistance_preset_sensitivity.csv")
    kvar = [r for r in sens if int(r["n"]) > 0 and abs(f(r["q"]) - pr.q) < 1e-12]
    m("SensKMin", f"{min(f(r['k']) for r in kvar):g}"); m("SensKMax", f"{max(f(r['k']) for r in kvar):g}")
    lo, hi = band(kvar, "p_emerge"); m("SensKEmergeMin", lo); m("SensKEmergeMax", hi)
    lo, hi = band(kvar, "n_cure", 1, "n"); m("SensKCureMin", lo)
    big = [r for r in sens if int(r["n"]) > 0 and f(r["q"]) < pr.q][0]
    m("SensIstarBig", f"{0.1 / f(big['q']):.0f}"); m("SensIstarBigEmerge", pct(big["p_emerge"]))
    dead = [r for r in sens if int(r["n"]) == 0][0]
    m("SensIstarDead", f"{0.1 / f(dead['q']):.0f}"); m("SensIstarDeadN", dead["n_extinct_before_treatment"])
    hz = rows("resistance_horizon.csv")
    m("HorizonLong", f"{f(hz[0]['horizon']):.0f}")
    lo, hi = band(hz, "p_emerge", 0); m("EmergeLongMin", lo); m("EmergeLongMax", hi)
    lo, hi = band(hz, "p_theory", 0); m("TheoryLongMin", lo); m("TheoryLongMax", hi)
    lo, hi = band(hz, "p_theory_cond", 0); m("TheoryLongCondMin", lo); m("TheoryLongCondMax", hi)
    lo, hi = band(hz, "n_wt_at_horizon", 0, "n"); m("WtLongMin", lo); m("WtLongMax", hi)

    # regime map
    rm = rows("regime_map.csv")
    m("MapN", rm[0]["n"]); m("MapCells", str(len(rm)))
    chronic = [f(r["p_persist"]) for r in rm if r["ode_class"] == "chronic"]
    m("MapChronicCells", str(len(chronic)))
    m("MapClearedCells", str(sum(r["ode_class"] == "cleared" for r in rm)))
    m("MapExtinctionCells", str(sum(r["ode_class"] == "extinction" for r in rm)))
    m("MapNeverPersist", str(sum(p == 0 for p in chronic)))
    m("MapIntermediate", str(sum(0 < p <= 0.95 for p in chronic)))
    m("MapAlwaysPersist", str(sum(p > 0.95 for p in chronic)))
    m("MapTheoryOutside", str(sum(not (f(r["p_est_low"]) <= f(r["p_est_theory"]) <= f(r["p_est_high"])) for r in rm)))
    m("MapTheoryFixedOutside", str(sum(not (f(r["p_est_low"]) <= f(r["p_est_theory_fixedE"]) <= f(r["p_est_high"])) for r in rm)))
    ks = sorted({f(r["k"]) for r in rm}); k_show = min(ks, key=lambda k: abs(np.log10(k) - np.log10(0.1)))
    tr = sorted([r for r in rm if abs(f(r["k"]) - k_show) < 1e-12], key=lambda r: f(r["R0_wt"]))
    m("MapTransectK", f"{k_show:.2g}")
    m("MapOdePersistFrom", f"{next(f(r['R0_wt']) for r in tr if f(r['ode_load_end']) >= 1):.1f}")
    m("MapEstHalfRzero", f"{next(f(r['R0_wt']) for r in tr if f(r['p_est']) >= 0.5):.0f}")
    m("MapPersistHalfRzero", f"{next(f(r['R0_wt']) for r in tr if f(r['p_persist']) >= 0.5):.0f}")
    m("MapPersistMax", f"{max(f(r['p_persist']) for r in tr):.2f}")
    trans = [r for r in tr if 0.05 < f(r["p_est"]) < 0.95]
    m("MapTransitionDecades", f"{np.log10(max(f(r['bw']) for r in trans) / min(f(r['bw']) for r in trans)):.1f}")
    m("MapTransitionRzeroMin", f"{min(f(r['R0_wt']) for r in trans):.1f}")
    m("MapTransitionRzeroMax", f"{max(f(r['R0_wt']) for r in trans):.0f}")
    hzm = rows("regime_map_horizon.csv")
    dd = np.array([f(r["p_persist_1000"]) - f(r["p_persist_200"]) for r in hzm])
    m("MapHorizonCells", str(len(hzm))); m("MapHorizonMeanDrop", f"{-dd.mean():.3f}"); m("MapHorizonWorstDrop", f"{-dd.min():.2f}")
    m("MapHorizonFlips", str(sum((f(r["p_persist_200"]) >= 0.5) > (f(r["p_persist_1000"]) >= 0.5) for r in hzm)))
    nb = rows("regime_map_notebook_q.csv")
    m("MapNotebookChronicCells", str(sum(r["ode_class"] == "chronic" for r in nb)))
    m("MapNotebookClearedCells", str(sum(r["ode_class"] == "cleared" for r in nb)))
    m("MapNotebookRuns", f"{sum(int(r['n']) for r in nb):,}")
    m("MapNotebookPersisting", str(sum(int(f(r["n_persist"])) for r in nb)))
    return "".join(out)


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "presets.tex").write_text(presets_table())
    (OUT / "numbers.tex").write_text("% generated by scripts/paper_tables.py; do not edit\n" + numbers())
    print("wrote", OUT / "presets.tex", "and", OUT / "numbers.tex")


if __name__ == "__main__":
    main()
