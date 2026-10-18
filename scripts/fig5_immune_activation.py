"""Figure 5: stochastic failure of immune activation.
Reads results/immune_activation.csv and results/immune_activation_trajectories.npz; runs no simulation."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from exp_immune_activation import BASE, DE_LEVELS, FATES, T0, T_END, V0
from viraldyn import solve_ode
from viraldyn.plotting import BLUE_RAMP, COLOR, DOUBLE, MUTED, panel_label, save, setup
from viraldyn.theory import immune_failure_probability

ROOT = Path(__file__).resolve().parents[1]
FATE_COLOR = {"bottleneck": "#b0aea8", "cleared": COLOR["ctl"], "immune_failure": COLOR["wt"]}
FATE_LABEL = {"bottleneck": "early extinction of the virus", "cleared": "cleared by effectors", "immune_failure": "effector lineage dies, virus persists"}


def load():
    with open(ROOT / "results" / "immune_activation.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for key, v in r.items():
            if key not in ("sweep", "variant"):
                r[key] = float(v)
    return rows


def points(ax, sel, xkey, colour, fill=True, label=None, key="p_fail", marker="o"):
    suffix = key[len("p_fail"):]
    x = [r[xkey] for r in sel]
    p = np.array([r[key] for r in sel])
    err = np.vstack([p - [r["wilson_low" + suffix] for r in sel], [r["wilson_high" + suffix] for r in sel] - p])
    ax.errorbar(x, p, yerr=err, fmt=marker, color=colour, markerfacecolor=colour if fill else "white", markeredgewidth=1.2, ecolor=colour, elinewidth=0.8, capsize=0, label=label, zorder=3)


def draw_sweep(ax, rows, sweep, xkey, theory_x, theory_fn):
    ax.plot(theory_x, [theory_fn(x) for x in theory_x], color=COLOR["theory"], lw=1.2, label="Kendall, deterministic load", zorder=1)
    full = sorted((r for r in rows if r["sweep"] == sweep and r["variant"] == "full"), key=lambda r: r[xkey])
    k0 = sorted((r for r in rows if r["sweep"] == sweep and r["variant"] == "k0"), key=lambda r: r[xkey])
    points(ax, full, xkey, COLOR["ctl"], True, "full model, any route")
    points(ax, full, xkey, COLOR["ctl"], True, "full model, founding lineage never took off", key="p_fail_early", marker="_")
    points(ax, k0, xkey, COLOR["ctl"], False, "no killing ($k=0$)")
    ax.set_ylabel("P(activation fails | established)")


def main():
    setup()
    rows = load()
    fig = plt.figure(figsize=(DOUBLE, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    ax_a, ax_b, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

    e_dense = np.linspace(1, 10, 50)
    for colour, dE in zip(BLUE_RAMP[1:], DE_LEVELS):
        ax_a.plot(e_dense, [immune_failure_probability(BASE.replace(dE=dE), [T0, 0, 0, V0, 0, e], t_max=T_END) for e in e_dense], color=colour, lw=1.2, zorder=1)
        sel = sorted((r for r in rows if r["sweep"] == "E0" and r["variant"] == "full" and abs(r["dE"] - dE) < 1e-9), key=lambda r: r["E0"])
        points(ax_a, sel, "E0", colour, label=f"$d_E = {dE:g}$ /day")
    ax_a.set_xlabel("initial effector cells $E_0$")
    ax_a.set_xticks([1, 2, 3, 5, 10])
    ax_a.set_ylim(0, 0.6)
    ax_a.set_ylabel("P(activation fails | established)")
    ax_a.legend(loc="upper right", title="full model; lines: theory")
    panel_label(ax_a, "a")

    de_dense = np.logspace(np.log10(0.03), np.log10(3.0), 25)
    draw_sweep(ax_b, rows, "dE", "dE", de_dense, lambda d: immune_failure_probability(BASE.replace(dE=d), [T0, 0, 0, V0, 0, 1], t_max=T_END))
    ax_b.set_xscale("log")
    ax_b.set_xticks([0.03, 0.1, 0.3, 1, 3])
    ax_b.set_xticklabels(["0.03", "0.1", "0.3", "1", "3"])
    ax_b.set_xlabel("effector death rate $d_E$ (per day)")
    ax_b.set_ylim(0, 1.02)
    ax_b.legend(loc="upper left")
    panel_label(ax_b, "b")

    data = np.load(ROOT / "results" / "immune_activation_trajectories.npz")
    t, V, fates = data["t"], data["V"].astype(float), data["fate"]
    V[V < 0] = np.nan
    for fate in ("bottleneck", "cleared", "immune_failure"):
        idx = np.where(fates == fate)[0]
        for j, i in enumerate(idx):
            ax_c.plot(t, np.maximum(V[i], 0.8), color=FATE_COLOR[fate], lw=0.7, alpha=0.6, label=f"{FATE_LABEL[fate]} ({len(idx)})" if j == 0 else None)
    ode = solve_ode(BASE, [T0, 0, 0, V0, 0, 1], t)
    ax_c.plot(t, np.maximum(ode[:, 1:5].sum(axis=1), 0.8), color=COLOR["ode"], lw=1.6, label="deterministic (ODE)")
    ax_c.set_yscale("log")
    ax_c.set_ylim(0.8, 3e4)
    ax_c.set_xlim(0, t[-1])
    ax_c.set_xlabel("time (days)")
    ax_c.set_ylabel("infected cells + virions")
    ax_c.legend(loc="upper right", ncol=2)
    other = [f for f in FATES if f not in FATE_COLOR and (fates == f).any()]
    if other:
        ax_c.text(0.01, 0.02, "not shown: " + ", ".join(f"{f} ({(fates == f).sum()})" for f in other), transform=ax_c.transAxes, fontsize=7, color=MUTED)
    panel_label(ax_c, "c")

    out = save(fig, "fig5_immune_activation")
    print("wrote", out)


if __name__ == "__main__":
    main()
