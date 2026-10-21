"""Figure 8: stochastic regime map against the deterministic classification.
Reads results/regime_map.csv; runs no simulation."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from exp_regime_map import BASE, BETA, K, X0, k_threshold
from viraldyn import r0_wild_type
from viraldyn.plotting import BLUE_RAMP, COLOR, DOUBLE, MUTED, SEQ_RAMP, panel_label, save, setup

ROOT = Path(__file__).resolve().parents[1]


def load():
    with open(ROOT / "results" / "regime_map.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for key, v in r.items():
            if key != "ode_class":
                r[key] = float(v)
    return rows


def edges(x):
    lx = np.log10(x)
    step = lx[1] - lx[0]
    return 10 ** np.concatenate([[lx[0] - step / 2], lx[:-1] + step / 2, [lx[-1] + step / 2]])


def main():
    setup()
    rows = load()
    cell = {(r["bw"], r["k"]): r for r in rows}
    bw = np.array(sorted({r["bw"] for r in rows}))
    kk = np.array(sorted({r["k"] for r in rows}))
    field = lambda key: np.array([[cell[(b, k)][key] for b in bw] for k in kk])
    P_est, P_per, theory, ode_end = field("p_est"), field("p_persist"), field("p_est_theory"), field("ode_load_end")
    cmap = LinearSegmentedColormap.from_list("seq", SEQ_RAMP)

    fig = plt.figure(figsize=(DOUBLE, 5.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])
    ax_a, ax_b, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

    beta_dense = np.logspace(np.log10(bw[0]), np.log10(bw[-1]), 300)
    r0_one = BASE.cw / 1000.0 / (BASE.pw * (1 - BASE.mu) / BASE.delta - 1)  # beta at which R0 = 1 with no effectors
    for ax, P, title in ((ax_a, P_est, "P(establishment)"), (ax_b, P_per, "P(persistence at day 200)")):
        m = ax.pcolormesh(edges(bw), edges(kk), P, cmap=cmap, vmin=0, vmax=1, shading="flat", rasterized=True)
        ax.plot(beta_dense, k_threshold(beta_dense, X0[5]), color=COLOR["ode"], ls="--", lw=1.2)
        ax.axvline(r0_one, color=COLOR["ode"], ls=":", lw=1.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"wild-type infectivity $\beta_w$")
        ax.grid(False)
        cb = fig.colorbar(m, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(title)
        cb.outline.set_visible(False)
    ax_a.contour(bw, kk, theory, levels=[0.5], colors=[COLOR["mut"]], linewidths=1.2)
    ax_b.contour(bw, kk, ode_end, levels=[1.0], colors=[COLOR["mut"]], linewidths=1.2)
    ax_a.set_ylabel("effector killing rate $k$")
    ax_b.set_yticklabels([])
    panel_label(ax_a, "a")
    panel_label(ax_b, "b")
    handles = [
        Line2D([], [], color=COLOR["ode"], ls="--", lw=1.2, label=f"$R_\\mathrm{{eff}} = 1$ with $E_0 = {X0[5]}$ effectors"),
        Line2D([], [], color=COLOR["ode"], ls=":", lw=1.0, label="$R_0 = 1$ without effectors"),
        Line2D([], [], color=COLOR["mut"], lw=1.2, label="a: branching theory with decaying effectors, P = 0.5"),
        Line2D([], [], color=COLOR["mut"], lw=1.2, label="b: deterministic model persists at day 200"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False, fontsize=7.5)

    # (c) transect at the preset killing rate
    k_show = kk[np.argmin(np.abs(np.log10(kk) - np.log10(BASE.k)))]
    sel = [cell[(b, k_show)] for b in bw]
    x = np.array([r["R0_wt"] for r in sel])
    for key, colour, label in (("p_est", BLUE_RAMP[1], "establishment"), ("p_persist", BLUE_RAMP[3], "persistence at day 200")):
        y = np.array([r[key] for r in sel])
        yerr = np.vstack([y - [r[key + "_low"] for r in sel], [r[key + "_high"] for r in sel] - y])
        ax_c.errorbar(x, y, yerr=yerr, fmt="o-", color=colour, ecolor=colour, elinewidth=0.8, capsize=0, lw=1.0, label=label)
    ax_c.plot(x, [r["p_est_theory"] for r in sel], color=COLOR["ode"], lw=1.0, ls="-", label="branching theory, establishment")
    ode_persist = np.array([r["ode_load_end"] >= 1.0 for r in sel], dtype=float)
    ax_c.step(x, ode_persist, where="mid", color=COLOR["mut"], lw=1.2, label="deterministic: persists at day 200")
    ax_c.set_xscale("log")
    ax_c.set_xticks([0.1, 0.3, 1, 3, 10, 30])
    ax_c.set_xticklabels(["0.1", "0.3", "1", "3", "10", "30"])
    ax_c.set_ylim(-0.03, 1.05)
    ax_c.set_xlabel(f"basic reproduction number $R_0$ (no effectors), transect at $k = {k_show:.3g}$")
    ax_c.set_ylabel("probability")
    ax_c.legend(loc="center left", ncol=1)
    panel_label(ax_c, "c")

    out = save(fig, "fig8_regime_map")
    print("wrote", out)


if __name__ == "__main__":
    main()
