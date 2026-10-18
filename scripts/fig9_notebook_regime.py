"""Appendix figure: the regime map with the notebook's own effector proliferation
rate (q = 0.1, chronic set point of one infected cell). The deterministic
classification the notebook drew, next to what exact realisations do.
Reads results/regime_map_notebook_q.csv; runs no simulation."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch

from exp_regime_map import BASE, X0, k_threshold
from fig8_regime_map import edges
from viraldyn.plotting import COLOR, DOUBLE, MUTED, SEQ_RAMP, panel_label, save, setup

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("extinction", "cleared", "chronic")
CLASS_COLOR = {"extinction": "#b0aea8", "cleared": COLOR["ctl"], "chronic": COLOR["wt"]}
CLASS_LABEL = {"extinction": "extinction ($R_0 < 1$)", "cleared": "cleared (load below 1 at day 200)", "chronic": "chronic (load above 1 at day 200)"}


def main():
    setup()
    with open(ROOT / "results" / "regime_map_notebook_q.csv") as f:
        rows = [{k: (v if k == "ode_class" else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]
    cell = {(r["bw"], r["k"]): r for r in rows}
    bw = np.array(sorted({r["bw"] for r in rows}))
    kk = np.array(sorted({r["k"] for r in rows}))
    field = lambda key: np.array([[cell[(b, k)][key] for b in bw] for k in kk])
    classes = np.array([[CLASSES.index(cell[(b, k)]["ode_class"]) for b in bw] for k in kk])
    n_runs = int(sum(r["n"] for r in rows))
    n_persist = int(sum(r["n_persist"] for r in rows))
    cmap = LinearSegmentedColormap.from_list("seq", SEQ_RAMP)
    beta_dense = np.logspace(np.log10(bw[0]), np.log10(bw[-1]), 300)
    r0_one = BASE.cw / 1000.0 / (BASE.pw * (1 - BASE.mu) / BASE.delta - 1)

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(DOUBLE, 2.9))
    ax_a.pcolormesh(edges(bw), edges(kk), classes, cmap=ListedColormap([CLASS_COLOR[c] for c in CLASSES]), vmin=-0.5, vmax=2.5, shading="flat", rasterized=True)
    ax_a.legend(handles=[Patch(facecolor=CLASS_COLOR[c], label=CLASS_LABEL[c]) for c in CLASSES], loc="upper left", fontsize=6)
    ax_a.set_title("deterministic classification", fontsize=8, color=MUTED, loc="left")
    for ax, P, title in ((ax_b, field("p_est"), "P(establishment)"), (ax_c, field("p_persist"), "P(persistence at day 200)")):
        m = ax.pcolormesh(edges(bw), edges(kk), P, cmap=cmap, vmin=0, vmax=1, shading="flat", rasterized=True)
        ax.set_title(title, fontsize=8, color=MUTED, loc="left")
        ax.set_yticklabels([])
    cb = fig.colorbar(m, ax=ax_c, fraction=0.06, pad=0.03)
    cb.set_label("probability")
    cb.outline.set_visible(False)
    ax_c.text(0.5, 0.5, f"{n_persist} of {n_runs:,} realisations\npersist", transform=ax_c.transAxes, ha="center", va="center", fontsize=8, color=MUTED)
    for ax in (ax_a, ax_b, ax_c):
        ax.plot(beta_dense, k_threshold(beta_dense, X0[5]), color=COLOR["ode"], ls="--", lw=1.0)
        ax.axvline(r0_one, color=COLOR["ode"], ls=":", lw=0.9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"wild-type infectivity $\beta_w$")
        ax.grid(False)
    ax_a.set_ylabel("effector killing rate $k$")
    for ax, lab in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c")):
        panel_label(ax, lab)
    out = save(fig, "fig9_notebook_regime")
    print("wrote", out)


if __name__ == "__main__":
    main()
