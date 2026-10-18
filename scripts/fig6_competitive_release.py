"""Figure 6: competitive release, deterministic and stochastic.
Reads results/competitive_release_ensemble.npz; the only computation is the ODE."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from exp_resistance import BASE, H, X0, grid
from viraldyn import solve_ode
from viraldyn.model import E, VM, VW
from viraldyn.plotting import COLOR, DOUBLE, MUTED, panel_label, save, setup

ROOT = Path(__file__).resolve().parents[1]
FLOOR = 0.5


def main():
    setup()
    data = np.load(ROOT / "results" / "competitive_release_ensemble.npz")
    t, t_treat, eps = data["t"], float(data["t_treat"]), float(data["eps"])
    q = list(data["quantiles"])
    i05, i50, i95 = q.index(5), q.index(50), q.index(95)

    g1, g2 = grid(0.0, t_treat), grid(t_treat, t_treat + H)
    ode1 = solve_ode(BASE, X0, g1)
    ode2 = solve_ode(BASE.replace(eps_b=eps), ode1[-1], g2)
    ode = np.vstack([ode1, ode2[1:]])

    fig, axes = plt.subplots(3, 1, figsize=(DOUBLE, 6.6), sharex=True)
    ax_a, ax_b, ax_c = axes
    for ax in axes:
        ax.axvline(t_treat, color=MUTED, lw=0.9, ls="--")

    ax_a.plot(t, np.maximum(ode[:, VW], FLOOR), color=COLOR["wt"], label="wild-type virions")
    ax_a.plot(t, np.maximum(ode[:, VM], FLOOR), color=COLOR["mut"], label="mutant virions")
    ax_a.plot(t, np.maximum(ode[:, E], FLOOR), color=COLOR["ctl"], label="effector cells")
    ax_a.set_yscale("log")
    ax_a.set_ylim(FLOOR, 3e4)
    ax_a.set_ylabel("count (deterministic)")
    ax_a.text(t_treat - 1.5, 1.5e4, f"therapy starts, $\\varepsilon_\\beta = {eps:g}$", color=MUTED, fontsize=8, ha="right")
    ax_a.legend(loc="upper right", ncol=3)
    panel_label(ax_a, "a")

    for key, name in (("Vw_q", "wt"), ("Vm_q", "mut")):
        Q = data[key]
        ax_b.fill_between(t, np.maximum(Q[i05], FLOOR), np.maximum(Q[i95], FLOOR), color=COLOR[name], alpha=0.22, lw=0)
        ax_b.plot(t, np.maximum(Q[i50], FLOOR), color=COLOR[name], lw=1.6)
    ax_b.plot(t, np.maximum(ode[:, VW], FLOOR), color=COLOR["wt"], lw=1.0, ls=":")
    ax_b.plot(t, np.maximum(ode[:, VM], FLOOR), color=COLOR["mut"], lw=1.0, ls=":")
    ax_b.set_yscale("log")
    ax_b.set_ylim(FLOOR, 3e4)
    ax_b.set_ylabel(f"virions ({int(data['n'])} realisations)")
    handles = [
        Line2D([], [], color=COLOR["wt"], lw=1.6, label="wild type, median"),
        Line2D([], [], color=COLOR["mut"], lw=1.6, label="mutant, median"),
        Patch(facecolor=MUTED, alpha=0.3, label="5th to 95th percentile"),
        Line2D([], [], color=MUTED, lw=1.0, ls=":", label="deterministic"),
    ]
    ax_b.legend(handles=handles, loc="upper right", ncol=2)
    panel_label(ax_b, "b")

    ax_c.plot(t, data["frac_wt"], color=COLOR["wt"], label="wild-type virions present")
    ax_c.plot(t, data["frac_mut"], color=COLOR["mut"], label="mutant virions present")
    ax_c.plot(t, data["frac_dominant"], color=COLOR["mut"], ls="--", label="mutant outnumbers wild type")
    ax_c.set_ylim(0, 1.02)
    ax_c.set_ylabel("fraction of realisations")
    ax_c.set_xlabel("time (days)")
    ax_c.set_xlim(0, t[-1])
    ax_c.legend(loc="center right")
    panel_label(ax_c, "c")

    out = save(fig, "fig6_competitive_release")
    print("wrote", out)


if __name__ == "__main__":
    main()
