"""Figure 2: the two deterministic regimes of the model, chronic and acute.
Only the ODE is computed; no simulation results are read."""

import matplotlib.pyplot as plt
import numpy as np

from viraldyn import PRESETS, immune_activation_threshold, r0_wild_type, solve_ode
from viraldyn.model import E, IM, IW, T, VM, VW
from viraldyn.plotting import COLOR, DOUBLE, MUTED, panel_label, save, setup

SCENARIOS = (
    ("chronic", [1000, 0, 0, 10, 0, 10], 100.0, "Chronic infection"),
    ("acute", [1000, 0, 0, 50, 0, 2], 60.0, "Acute infection: clearance"),
)
FLOOR = 0.5


def main():
    setup()
    fig, axes = plt.subplots(3, 2, figsize=(DOUBLE, 6.2), sharex="col")
    for col, (name, x0, t_end, title) in enumerate(SCENARIOS):
        p = PRESETS[name]
        t = np.linspace(0.0, t_end, 2001)
        y = solve_ode(p, x0, t)
        ax_cells, ax_vir, ax_eff = axes[:, col]
        ax_cells.set_title(f"{title}\n$R_0 = {r0_wild_type(p):.1f}$, activation threshold $d_E/q = {immune_activation_threshold(p):g}$ cells", fontsize=8, loc="left", color=MUTED)
        ax_cells.plot(t, y[:, T], color=COLOR["target"], label="target cells $T$")
        ax_cells.plot(t, y[:, IW], color=COLOR["wt"], label="infected, wild type $I_w$")
        ax_cells.plot(t, y[:, IM], color=COLOR["mut"], label="infected, mutant $I_m$")
        ax_cells.set_yscale("log")
        ax_cells.set_ylim(FLOOR, 3e6 if name == "chronic" else 3e3)
        ax_vir.plot(t, np.maximum(y[:, VW], FLOOR), color=COLOR["wt"], label="wild-type virions $V_w$")
        ax_vir.plot(t, np.maximum(y[:, VM], FLOOR), color=COLOR["mut"], label="mutant virions $V_m$")
        ax_vir.set_yscale("log")
        ax_vir.set_ylim(FLOOR, 3e6 if name == "chronic" else 3e4)
        ax_eff.plot(t, np.maximum(y[:, E], FLOOR), color=COLOR["ctl"], label="effector cells $E$")
        ax_eff.set_yscale("log")
        ax_eff.set_xlabel("time (days)")
        ax_cells.legend(loc="lower right" if name == "chronic" else "center right", fontsize=7)
        ax_vir.legend(loc="upper right", fontsize=7)
        ax_eff.legend(loc="lower right" if name == "chronic" else "upper right", fontsize=7)
        if col == 0:
            ax_cells.set_ylabel("cells")
            ax_vir.set_ylabel("virions")
            ax_eff.set_ylabel("effector cells")
        panel_label(ax_cells, "ab"[col])
    out = save(fig, "fig2_deterministic")
    print("wrote", out)


if __name__ == "__main__":
    main()
