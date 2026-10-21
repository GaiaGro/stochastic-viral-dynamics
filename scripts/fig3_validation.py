"""Figure 3: the exact simulation against the deterministic solution at large
populations; the phase plane at two system sizes; fluctuation size against system
size with the linear-noise prediction.
Reads results/baseline_ensemble.npz, results/fluctuations_phase.npz and
results/fluctuations.csv; runs no simulation."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from viraldyn.model import E, T, VM, VW
from viraldyn.plotting import BLUE_RAMP, COLOR, DOUBLE, MUTED, panel_label, save, setup

ROOT = Path(__file__).resolve().parents[1]
FLOOR = 0.5
SPECIES_STYLE = {"Iw": (COLOR["wt"], "infected cells $I_w$"), "Vw": (BLUE_RAMP[3], "wild-type virions $V_w$"), "E": (COLOR["ctl"], "effector cells $E$"), "T": (COLOR["target"], "target cells $T$")}


def main():
    setup()
    d = np.load(ROOT / "results" / "baseline_ensemble.npz")
    t, q, ode = d["t"], d["q"].astype(float), d["ode"]
    ql = list(d["quantiles"])
    i05, i50, i95 = ql.index(5), ql.index(50), ql.index(95)
    ph = np.load(ROOT / "results" / "fluctuations_phase.npz")
    with open(ROOT / "results" / "fluctuations.csv") as f:
        fl = [{k: (v if k == "species" else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]

    fig = plt.figure(figsize=(DOUBLE, 6.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1])
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    ax_phase = fig.add_subplot(gs[:2, 1])
    ax_cv = fig.add_subplot(gs[2, 1])

    # (a) ensemble against the ODE at the large size
    panels = (
        (axes[0], ((T, "target"),), "cells", False),
        (axes[1], ((VW, "wt"), (VM, "mut")), "virions", True),
        (axes[2], ((E, "ctl"),), "effector cells", False),
    )
    for ax, species, ylabel, log in panels:
        for j, name in species:
            ax.fill_between(t, np.maximum(q[i05, :, j], FLOOR), np.maximum(q[i95, :, j], FLOOR), color=COLOR[name], alpha=0.22, lw=0)
            ax.plot(t, np.maximum(q[i50, :, j], FLOOR), color=COLOR[name], lw=1.5)
            ax.plot(t, np.maximum(ode[:, j], FLOOR), color=COLOR[name], lw=1.0, ls=":")
        ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale("log")
            ax.set_ylim(FLOOR, None)
    axes[-1].set_xlabel("time (days)")
    for ax in axes[:-1]:
        ax.set_xticklabels([])
    handles = [
        Line2D([], [], color=MUTED, lw=1.5, label=f"median of {int(d['n'])} realisations"),
        Patch(facecolor=MUTED, alpha=0.3, label="5th to 95th percentile"),
        Line2D([], [], color=MUTED, lw=1.0, ls=":", label="deterministic"),
        Line2D([], [], color=COLOR["wt"], lw=1.5, label="wild type"),
        Line2D([], [], color=COLOR["mut"], lw=1.5, label="mutant"),
    ]
    axes[1].legend(handles=handles, loc="center right", ncol=1, fontsize=6.5)
    panel_label(axes[0], "a")

    # (b) phase plane at two system sizes
    tp = ph["t"]
    late = tp >= 20.0  # skip the primary peak so the clouds around the fixed points are visible
    for omega, shade in zip(ph["omegas"], (BLUE_RAMP[3], BLUE_RAMP[1])):
        X, pode = ph[f"X_{omega}"].astype(float), ph[f"ode_{omega}"]
        ax_phase.plot(np.maximum(X[late, VW], FLOOR), np.maximum(X[late, E], FLOOR), color=shade, lw=0.5, alpha=0.8, label=f"exact realisation, $\\Omega = {omega}$")
        ax_phase.plot(np.maximum(pode[late, VW], FLOOR), np.maximum(pode[late, E], FLOOR), color=COLOR["ode"], lw=1.3)
        ax_phase.plot(pode[-1, VW], pode[-1, E], "o", color=COLOR["ode"], ms=5)
    ax_phase.plot([], [], color=COLOR["ode"], lw=1.3, label="deterministic orbit and fixed point")
    ax_phase.set_xscale("log")
    ax_phase.set_yscale("log")
    ax_phase.set_xlabel("wild-type virions $V_w$ (from day 20)")
    ax_phase.set_ylabel("effector cells $E$")
    ax_phase.legend(loc="center right", fontsize=7)
    panel_label(ax_phase, "b")

    # (c) coefficient of variation in the stationary window against system size
    for name, (colour, label) in SPECIES_STYLE.items():
        sel = sorted((r for r in fl if r["species"] == name), key=lambda r: r["omega"])
        om = [r["omega"] for r in sel]
        ax_cv.plot(om, [r["cv_lna"] for r in sel], color=colour, lw=1.0, ls="-")
        ax_cv.plot(om, [r["cv_sim"] for r in sel], "o", color=colour, ms=4, label=label)
    ax_cv.set_xscale("log")
    ax_cv.set_yscale("log")
    ax_cv.set_xticks([1, 3, 10, 30, 100])
    ax_cv.set_xticklabels(["1", "3", "10", "30", "100"])
    ax_cv.set_xlabel("system size $\\Omega$")
    ax_cv.set_ylabel("coefficient of variation")
    ax_cv.legend(loc="lower left", fontsize=6.5, title="points: simulation; lines: linear noise", title_fontsize=6.5)
    panel_label(ax_cv, "c")

    out = save(fig, "fig3_validation")
    print("wrote", out)


if __name__ == "__main__":
    main()
