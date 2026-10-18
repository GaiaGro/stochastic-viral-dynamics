"""Figure 3: the exact simulation against the deterministic solution at large
populations, and the phase plane of chronic infection.
Reads results/baseline_ensemble.npz; runs no simulation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from viraldyn.model import E, IM, IW, T, VM, VW
from viraldyn.plotting import COLOR, DOUBLE, MUTED, panel_label, save, setup

ROOT = Path(__file__).resolve().parents[1]
FLOOR = 0.5


def main():
    setup()
    d = np.load(ROOT / "results" / "baseline_ensemble.npz")
    t, q, ode = d["t"], d["q"].astype(float), d["ode"]
    ql = list(d["quantiles"])
    i05, i50, i95 = ql.index(5), ql.index(50), ql.index(95)

    fig = plt.figure(figsize=(DOUBLE, 5.6))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1])
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    ax_phase = fig.add_subplot(gs[:, 1])

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
    axes[2].legend(handles=handles, loc="lower right", ncol=2, fontsize=7)
    panel_label(axes[0], "a")

    X, pode = d["phase_X"].astype(float), d["phase_ode"]
    ax_phase.plot(np.maximum(X[:, VW], FLOOR), np.maximum(X[:, E], FLOOR), color=MUTED, lw=0.5, alpha=0.8, label="one exact realisation")
    ax_phase.plot(np.maximum(pode[:, VW], FLOOR), np.maximum(pode[:, E], FLOOR), color=COLOR["ode"], lw=1.4, label="deterministic orbit")
    ax_phase.plot(pode[-1, VW], pode[-1, E], "o", color=COLOR["ode"], ms=6, label=f"deterministic state at day {d['phase_t'][-1]:g}")
    ax_phase.set_xscale("log")
    ax_phase.set_yscale("log")
    ax_phase.set_xlabel("wild-type virions $V_w$")
    ax_phase.set_ylabel("effector cells $E$")
    ax_phase.legend(loc="center left", fontsize=7)
    panel_label(ax_phase, "b")

    out = save(fig, "fig3_validation")
    print("wrote", out)


if __name__ == "__main__":
    main()
