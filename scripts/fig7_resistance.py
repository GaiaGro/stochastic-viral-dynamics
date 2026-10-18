"""Figure 7: probability that resistance emerges under therapy.
Reads results/resistance.csv and results/resistance_runs.csv; runs no simulation."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from exp_resistance import BASE, EPS, SHOWCASE, T_TREAT
from viraldyn import r0_wild_type
from viraldyn.plotting import BLUE_RAMP5, COLOR, DOUBLE, MUTED, SEQ_RAMP, TEXT, panel_label, save, setup

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    with open(ROOT / "results" / name) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for key, v in r.items():
            r[key] = float(v)
    return rows


def main():
    setup()
    cells = load("resistance.csv")
    sizes = sorted(load("resistance_system_size.csv"), key=lambda r: r["omega"])
    cell = {(r["t_treat"], r["eps"]): r for r in cells}

    fig = plt.figure(figsize=(DOUBLE, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
    ax_a, ax_b, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

    # (a) heatmap of P(emergence), rows efficacy, columns treatment start
    P = np.array([[cell[(tt, e)]["p_emerge"] for tt in T_TREAT] for e in EPS])
    cmap = LinearSegmentedColormap.from_list("seq", SEQ_RAMP)
    im = ax_a.imshow(P, cmap=cmap, vmin=0, vmax=1, origin="lower", aspect="auto")
    for i, e in enumerate(EPS):
        for j, tt in enumerate(T_TREAT):
            c = cell[(tt, e)]
            ax_a.text(j, i, f"{100 * c['p_emerge']:.1f}%", ha="center", va="center", fontsize=7.5, color="white" if c["p_emerge"] > 0.55 else TEXT)
            if c["ode_takeover"]:
                ax_a.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=COLOR["ode"], lw=1.6))
    eps_star = 1.0 - 1.0 / r0_wild_type(BASE)
    k = sum(e < eps_star for e in EPS) - 0.5
    ax_a.axhline(k, color=COLOR["ode"], ls="--", lw=0.9)
    ax_a.set_xticks(range(len(T_TREAT)))
    ax_a.set_xticklabels([f"{tt:g}" for tt in T_TREAT])
    ax_a.set_yticks(range(len(EPS)))
    ax_a.set_yticklabels([f"{e:g}" for e in EPS])
    ax_a.set_xlabel("treatment start (day)")
    ax_a.set_ylabel("drug efficacy $\\varepsilon_\\beta$")
    ax_a.grid(False)
    cb = fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.03)
    cb.set_label("P(resistance emerges)")
    cb.outline.set_visible(False)
    ax_a.set_title(f"outlined: deterministic takeover; dashed: $\\varepsilon^* = {eps_star:.2f}$, wild type suppressed above", fontsize=7.5, color=MUTED, loc="left")
    panel_label(ax_a, "a")

    # (b) observed against the per-realisation theory band
    ax_b.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":")
    for colour, e in zip(BLUE_RAMP5, EPS):
        sel = [cell[(tt, e)] for tt in T_TREAT]
        lo = np.array([c["p_theory_low"] for c in sel])
        hi = np.array([c["p_theory_high"] for c in sel])
        y = np.array([c["p_emerge"] for c in sel])
        yerr = np.vstack([y - [c["wilson_low"] for c in sel], [c["wilson_high"] for c in sel] - y])
        ax_b.errorbar(hi, y, xerr=np.vstack([hi - lo, np.zeros_like(hi)]), yerr=yerr, fmt="o", color=colour, ecolor=colour, elinewidth=0.9, capsize=0, label=f"$\\varepsilon_\\beta = {e:g}$")
    ax_b.set_xlim(-0.02, 1.02)
    ax_b.set_ylim(-0.02, 1.02)
    ax_b.set_xlabel("predicted P(resistance emerges), theory band")
    ax_b.set_ylabel("observed P(resistance emerges)")
    ax_b.legend(loc="upper left")
    panel_label(ax_b, "b")

    # (c) system size: every population scaled by omega at the showcase cell
    om = np.array([r["omega"] for r in sizes])
    y = np.array([r["p_emerge"] for r in sizes])
    yerr = np.vstack([y - [r["wilson_low"] for r in sizes], [r["wilson_high"] for r in sizes] - y])
    lo = np.array([r["p_theory_low"] for r in sizes])
    hi = np.array([r["p_theory_high"] for r in sizes])
    ax_c.axhline(1.0, color=COLOR["ode"], ls="--", lw=1.2, label="deterministic (takeover in every case)")
    ax_c.fill_between(om, lo, hi, color=MUTED, alpha=0.18, lw=0, label="theory band")
    ax_c.errorbar(om, y, yerr=yerr, fmt="o", color=COLOR["mut"], ecolor=COLOR["mut"], elinewidth=0.9, capsize=0, label="simulation")
    ax_c.set_xscale("log")
    ax_c.set_xticks(om)
    ax_c.set_xticklabels([f"{o:g}" for o in om])
    ax_c.set_ylim(0, 1.05)
    ax_c.set_xlabel(f"system size $\\Omega$ (all populations scaled; treatment at day {SHOWCASE[0]:g}, $\\varepsilon_\\beta = {SHOWCASE[1]:g}$)")
    ax_c.set_ylabel("P(resistance emerges)")
    ax_c.legend(loc="center left")
    panel_label(ax_c, "c")

    out = save(fig, "fig7_resistance")
    print("wrote", out)


if __name__ == "__main__":
    main()
