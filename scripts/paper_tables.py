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
    out = []
    p = PRESETS["resistance"]
    out.append(macro("RzeroWt", f"{r0_wild_type(p):.2f}"))
    out.append(macro("RzeroMut", f"{r0_mutant(p):.2f}"))
    out.append(macro("EpsStar", f"{suppression_efficacy(p):.3f}"))
    out.append(macro("RzeroEstablishment", f"{r0_wild_type(PRESETS['establishment']):.1f}"))

    est = rows("establishment.csv")
    inside = sum(float(r["wilson_low"]) <= float(r["p_theory"]) <= float(r["wilson_high"]) for r in est)
    out.append(macro("EstConditions", str(len(est))))
    out.append(macro("EstInsideInterval", str(inside)))
    one = next(r for r in est if r["sweep"] == "V0" and float(r["E0"]) == 0 and float(r["V0"]) == 1)
    out.append(macro("EstOneVirionTheory", f"{float(one['p_theory']):.3f}"))
    out.append(macro("EstOneVirionObserved", f"{float(one['p_hat']):.3f}"))

    imm = rows("immune_activation.csv")
    head = next(r for r in imm if r["sweep"] == "E0" and r["variant"] == "full" and float(r["E0"]) == 1 and abs(float(r["dE"]) - 0.1) < 1e-9)
    out.append(macro("ImmFailObserved", pct(head["p_fail"])))
    out.append(macro("ImmFailLow", pct(head["wilson_low"])))
    out.append(macro("ImmFailHigh", pct(head["wilson_high"])))
    out.append(macro("ImmFailTheory", pct(head["p_theory"])))
    out.append(macro("ImmFailUnconditional", pct(float(head["n_immune_failure"]) / float(head["n"]))))
    de = [r for r in imm if r["sweep"] == "dE" and r["variant"] == "full"]
    out.append(macro("ImmFailDeMin", pct(min(float(r["p_fail"]) for r in de))))
    out.append(macro("ImmFailDeMax", pct(max(float(r["p_fail"]) for r in de), 0)))

    via = {r["preset"]: r for r in rows("preset_viability.csv")}
    out.append(macro("TherapyTrough", fmt(float(via["therapy"]["ode_trough_load"]))))
    out.append(macro("TherapyTroughDay", f"{float(via['therapy']['ode_trough_day']):.1f}"))
    out.append(macro("TherapyAlive", via["therapy"]["n_alive_day100"]))
    out.append(macro("ResistanceAlive", via["resistance"]["n_alive_day100"]))

    res = rows("resistance.csv")
    hi = [float(r["p_emerge"]) for r in res if float(r["eps"]) >= 0.8]
    lo = [float(r["p_emerge"]) for r in res if float(r["eps"]) == 0.5]
    out.append(macro("EmergeSuppressedMin", pct(min(hi))))
    out.append(macro("EmergeSuppressedMax", pct(max(hi))))
    out.append(macro("EmergeWeakMin", pct(min(lo), 0)))
    out.append(macro("EmergeWeakMax", pct(max(lo), 0)))
    out.append(macro("CureSuppressedMin", pct(min(float(r["n_cure"]) / float(r["n"]) for r in res if float(r["eps"]) >= 0.8), 0)))
    out.append(macro("OdeTakeoverCells", str(sum(int(float(r["ode_takeover"])) for r in res))))
    size = rows("resistance_system_size.csv")
    big = max(size, key=lambda r: float(r["omega"]))
    out.append(macro("OmegaMax", f"{float(big['omega']):g}"))
    out.append(macro("EmergeOmegaMax", pct(big["p_emerge"], 0)))
    out.append(macro("CureOmegaMax", pct(float(big["n_cure"]) / float(big["n"]), 0)))

    rm = rows("regime_map.csv")
    chronic = [float(r["p_persist"]) for r in rm if r["ode_class"] == "chronic"]
    out.append(macro("MapChronicCells", str(len(chronic))))
    out.append(macro("MapNeverPersist", str(sum(p == 0 for p in chronic))))
    out.append(macro("MapIntermediate", str(sum(0 < p <= 0.95 for p in chronic))))
    out.append(macro("MapAlwaysPersist", str(sum(p > 0.95 for p in chronic))))
    nb = rows("regime_map_notebook_q.csv")
    out.append(macro("MapNotebookChronicCells", str(sum(r["ode_class"] == "chronic" for r in nb))))
    out.append(macro("MapNotebookPersisting", str(sum(float(r["n_persist"]) for r in nb) and "" or "0")))

    base = {r["species"]: r for r in rows("baseline_validation.csv")}
    out.append(macro("BaselineMaxLogDev", f"{max(float(base[s]['max_abs_log10_dev_median']) for s in ('T', 'Iw', 'Vw', 'E')):.3f}"))
    out.append(macro("BaselineN", base["T"]["n"]))
    return "".join(out)


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "presets.tex").write_text(presets_table())
    (OUT / "numbers.tex").write_text("% generated by scripts/paper_tables.py; do not edit\n" + numbers())
    print("wrote", OUT / "presets.tex", "and", OUT / "numbers.tex")


if __name__ == "__main__":
    main()
