"""Shared figure style.

Entities keep one colour in every figure: wild type blue, mutant orange, effector
cells aqua (the three slots that pass the all-pairs colour-vision check), target
cells a neutral gray in their own panel, deterministic solutions and theory in
black. Ordered levels (inoculum, effector count) use the single-hue blue ramp.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

COLOR = {
    "wt": "#2a78d6",
    "mut": "#eb6834",
    "ctl": "#1baf7a",
    "target": "#52514e",
    "ode": "#0b0b0b",
    "theory": "#0b0b0b",
}
BLUE_RAMP = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")  # light to dark, validated ordinal
BLUE_RAMP5 = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b")  # five ordered levels, validated
SEQ_RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")  # continuous magnitude
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#dedcd7"
DOUBLE = 7.0  # figure width in inches

FIGURES = Path(__file__).resolve().parents[1] / "figures"


def setup():
    mpl.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "figure.constrained_layout.use": True,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
        }
    )


def panel_label(ax, text):
    ax.text(-0.12, 1.04, text, transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom")


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}")
    plt.close(fig)
    return FIGURES / f"{name}.png"
