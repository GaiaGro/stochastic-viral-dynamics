"""Figure 4: probability of establishment, exact simulation versus branching theory.
Reads results/establishment.csv; runs no simulation."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from exp_establishment import BASE, E0_LEVELS, T0, V0_FOR_BETA
from viraldyn.plotting import BLUE_RAMP, COLOR, DOUBLE, MUTED, panel_label, save, setup
from viraldyn.theory import establishment_probability

ROOT = Path(__file__).resolve().parents[1]


def load():
    with open(ROOT / "results" / "establishment.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for key, v in r.items():
            if key != "sweep":
                r[key] = float(v)
    return rows


def errbars(rows):
    p = np.array([r["p_hat"] for r in rows])
    lo = np.array([r["wilson_low"] for r in rows])
    hi = np.array([r["wilson_high"] for r in rows])
    return p, np.vstack([p - lo, hi - p])


def main():
    setup()
    rows = load()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(DOUBLE, 2.9))

    # (a) inoculum sweep at fixed effector counts
    v_dense = np.linspace(1, 21, 200)
    for colour, E0 in zip(BLUE_RAMP, E0_LEVELS):
        sel = sorted((r for r in rows if r["sweep"] == "V0" and r["E0"] == E0), key=lambda r: r["V0"])
        theory = [establishment_probability(BASE, [T0, 0, 0, v, 0, E0]) for v in v_dense]
        ax_a.plot(v_dense, theory, color=colour, lw=1.2)
        p, err = errbars(sel)
        ax_a.errorbar([r["V0"] for r in sel], p, yerr=err, fmt="o", color=colour, ecolor=colour, elinewidth=0.8, capsize=0, label=f"$E_0 = {E0:g}$")
    ax_a.set_xscale("log")
    ax_a.set_xticks([1, 2, 3, 5, 10, 20])
    ax_a.set_xticklabels(["1", "2", "3", "5", "10", "20"])
    ax_a.set_ylim(0, 1.02)
    ax_a.set_xlabel("inoculum $V_0$ (wild-type virions)")
    ax_a.set_ylabel("probability of establishment")
    ax_a.legend(title="fixed effectors", loc="lower right")
    panel_label(ax_a, "a")

    # (b) R0 sweep for two inocula, no effectors
    r0_dense = np.logspace(np.log10(0.2), np.log10(40), 300)
    bw_dense = BASE.cw / T0 / (BASE.pw * (1 - BASE.mu) / BASE.delta / r0_dense - 1)  # invert R0(beta)
    ax_b.step([0.2, 1, 1, 40], [0, 0, 1, 1], where="post", color=COLOR["ode"], ls="--", lw=1.2, label="deterministic (ODE)")
    for colour, V0 in zip((BLUE_RAMP[1], BLUE_RAMP[3]), V0_FOR_BETA):
        ok = bw_dense > 0
        theory = [establishment_probability(BASE.replace(bw=float(b), bm=float(b) * 0.8), [T0, 0, 0, V0, 0, 0]) for b in bw_dense[ok]]
        ax_b.plot(r0_dense[ok], theory, color=colour, lw=1.2)
        sel = sorted((r for r in rows if r["sweep"] == "beta" and r["V0"] == V0), key=lambda r: r["R0_wt"])
        p, err = errbars(sel)
        ax_b.errorbar([r["R0_wt"] for r in sel], p, yerr=err, fmt="o", color=colour, ecolor=colour, elinewidth=0.8, capsize=0, label=f"$V_0 = {V0}$")
    ax_b.axvline(1.0, color=MUTED, lw=0.8, ls=":")
    ax_b.set_xscale("log")
    ax_b.set_xlim(0.2, 40)
    ax_b.set_xticks([0.3, 1, 3, 10, 30])
    ax_b.set_xticklabels(["0.3", "1", "3", "10", "30"])
    ax_b.set_ylim(0, 1.02)
    ax_b.set_xlabel("basic reproduction number $R_0$")
    ax_b.legend(loc="upper left")
    panel_label(ax_b, "b")

    out = save(fig, "fig4_establishment")
    print("wrote", out)


if __name__ == "__main__":
    main()
