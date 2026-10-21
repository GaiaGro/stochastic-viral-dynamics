"""Time the compiled SSA in the regimes that matter for the compute budget.
Run with `make bench`."""

import time

import numpy as np

from viraldyn import PRESETS, simulate

CASES = [
    ("establishment, 30 days", PRESETS["establishment"], [1000, 0, 0, 1, 0, 1], 30.0, 20),
    ("acute, 60 days", PRESETS["acute"], [1000, 0, 0, 50, 0, 2], 60.0, 5),
    ("therapy preset, 120 days", PRESETS["therapy"], [1000, 0, 0, 10, 0, 10], 120.0, 3),
    ("chronic preset, 120 days", PRESETS["chronic"], [1000, 5, 0, 30, 0, 10], 120.0, 1),
]

simulate(PRESETS["acute"], [10, 0, 0, 1, 0, 1], [0.0, 0.1], seed_value=0)  # compile
print(f"{'case':<28}{'s/run':>8}{'events/run':>14}{'events/s':>12}")
for name, p, x0, t_max, n in CASES:
    grid = np.linspace(0, t_max, int(t_max * 20) + 1)
    t0 = time.perf_counter()
    ev = [simulate(p, x0, grid, seed_value=s).n_events for s in range(n)]
    dt = (time.perf_counter() - t0) / n
    print(f"{name:<28}{dt:>8.2f}{np.mean(ev):>14.0f}{np.mean(ev) / dt:>12.2e}")
