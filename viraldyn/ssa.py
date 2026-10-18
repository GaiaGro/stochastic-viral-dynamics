"""Exact stochastic simulation (Gillespie direct method), compiled with numba.

The state is recorded on a fixed time grid rather than at every event: a 120-day
run in the chronic regime has about 2e7 events, which would be about 1 GB per
realisation if every event were stored.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit

from .model import N_REACTIONS, N_SPECIES, STOICH, Params, propensities

STOP_T_END = 0
STOP_EXTINCT = 1
STOP_CUM_INFECTIONS = 2
STOP_MAX_EVENTS = 3
STOP_NO_REACTIONS = 4
STOP_ABOVE = 5
STOP_REASONS = ("t_end", "extinct", "cum_infections", "max_events", "no_reactions", "above")

# lineages tracked for extinction times
WT, MUT, CTL = range(3)
LINEAGES = ("wt", "mut", "ctl")


@dataclass
class SSAResult:
    t_grid: np.ndarray  # (n,)
    X: np.ndarray  # (n, 6) state at each grid time; -1 after an early stop
    C: np.ndarray  # (n, 15) cumulative reaction counts at each grid time; -1 after stop
    t_end: float  # time at which the run stopped
    x_end: np.ndarray  # (6,) state at t_end
    c_end: np.ndarray  # (15,) cumulative reaction counts at t_end
    n_events: int
    t_ext: np.ndarray  # (3,) first extinction time of (wt, mut, ctl) lineage, nan if never
    x_max: np.ndarray  # (6,) running maximum of each species
    stop_reason: int  # index into STOP_REASONS

    @property
    def stopped(self) -> str:
        return STOP_REASONS[self.stop_reason]

    @property
    def extinct(self) -> bool:
        """No infected cells and no free virus at the end of the run."""
        return int(self.x_end[1:5].sum()) == 0

    @property
    def cumulative_infections(self) -> int:
        return int(self.c_end[0] + self.c_end[1])


@njit(cache=True)
def seed(s):
    """Seed numba's random generator (separate from numpy's global one)."""
    np.random.seed(s)


@njit(cache=True)
def _lineage_sizes(x, out):
    out[0] = x[1] + x[3]  # Iw + Vw
    out[1] = x[2] + x[4]  # Im + Vm
    out[2] = x[5]  # E


@njit(cache=True)
def _ssa(x0, p, S, t_grid, stop_on_extinction, stop_cum_infections, stop_above, max_events):
    n_grid = t_grid.shape[0]
    x = x0.copy()
    X = np.full((n_grid, N_SPECIES), -1, np.int64)
    C = np.full((n_grid, N_REACTIONS), -1, np.int64)
    counts = np.zeros(N_REACTIONS, np.int64)
    a = np.empty(N_REACTIONS)
    x_max = x.copy()
    t_ext = np.full(3, np.nan)
    size = np.zeros(3, np.int64)
    size_prev = np.zeros(3, np.int64)
    _lineage_sizes(x, size_prev)

    t = t_grid[0]
    gi = 0
    n_events = 0
    stop = STOP_T_END
    for i in range(N_SPECIES):  # a cap already met at t0 ends the run before any event
        if stop_above[i] > 0 and x[i] >= stop_above[i]:
            stop = STOP_ABOVE
    if stop == STOP_ABOVE:
        X[0, :] = x
        C[0, :] = counts
        return X, C, t, x, counts, n_events, t_ext, x_max, stop
    while True:
        propensities(x, p, a)
        a0 = 0.0
        for j in range(N_REACTIONS):
            a0 += a[j]
        if a0 <= 0.0:
            # nothing can happen again: state is frozen for the rest of the grid
            while gi < n_grid:
                X[gi, :] = x
                C[gi, :] = counts
                gi += 1
            t = t_grid[n_grid - 1]
            stop = STOP_NO_REACTIONS
            break
        t_new = t + np.random.exponential(1.0 / a0)
        while gi < n_grid and t_grid[gi] <= t_new:
            X[gi, :] = x  # state is constant on [t, t_new)
            C[gi, :] = counts
            gi += 1
        if gi >= n_grid:
            t = t_grid[n_grid - 1]
            break
        t = t_new
        r = np.random.random() * a0
        cum = 0.0
        j = 0
        for j in range(N_REACTIONS):
            cum += a[j]
            if r < cum:
                break
        for i in range(N_SPECIES):
            x[i] += S[i, j]
            if x[i] > x_max[i]:
                x_max[i] = x[i]
        counts[j] += 1
        n_events += 1

        _lineage_sizes(x, size)
        for l in range(3):
            if size_prev[l] > 0 and size[l] == 0 and np.isnan(t_ext[l]):
                t_ext[l] = t
            size_prev[l] = size[l]

        if stop_on_extinction and size[WT] == 0 and size[MUT] == 0:
            stop = STOP_EXTINCT
            break
        if stop_cum_infections > 0 and counts[0] + counts[1] >= stop_cum_infections:
            stop = STOP_CUM_INFECTIONS
            break
        for i in range(N_SPECIES):
            if stop_above[i] > 0 and x[i] >= stop_above[i]:
                stop = STOP_ABOVE
        if stop == STOP_ABOVE:
            break
        if n_events >= max_events:
            stop = STOP_MAX_EVENTS
            break
    return X, C, t, x, counts, n_events, t_ext, x_max, stop


def simulate(
    p: Params,
    x0,
    t_grid,
    seed_value=None,
    stop_on_extinction=False,
    stop_cum_infections=0,
    stop_above=None,
    max_events=np.iinfo(np.int64).max,
) -> SSAResult:
    """One exact realisation from state `x0` recorded at the times in `t_grid`.

    `stop_on_extinction` ends the run when no infected cell or virion of either
    strain is left. `stop_cum_infections` ends it when the cumulative number of
    infection events reaches that value (0 disables). `stop_above` is a per-species
    cap (6 integers, 0 disables) that ends the run as soon as a species reaches it,
    for example to stop an effector population that nothing limits. Grid rows after
    an early stop hold -1.
    """
    p = Params(*map(float, p))
    x0 = np.asarray(x0, dtype=np.int64)
    t_grid = np.asarray(t_grid, dtype=np.float64)
    cap = np.zeros(N_SPECIES, dtype=np.int64) if stop_above is None else np.asarray(stop_above, dtype=np.int64)
    if x0.shape != (N_SPECIES,) or (x0 < 0).any():
        raise ValueError("x0 must be 6 non-negative integers")
    if cap.shape != (N_SPECIES,) or (cap < 0).any():
        raise ValueError("stop_above must be 6 non-negative integers")
    if t_grid.ndim != 1 or t_grid.size < 1 or (np.diff(t_grid) <= 0).any():
        raise ValueError("t_grid must be strictly increasing")
    if seed_value is not None:
        if not 0 <= int(seed_value) < 2**32:
            raise ValueError("seed_value must fit in 32 bits; numba's generator wraps larger seeds silently")
        seed(int(seed_value))
    X, C, t, x, counts, n, t_ext, x_max, stop = _ssa(
        x0, p, STOICH, t_grid, bool(stop_on_extinction), int(stop_cum_infections), cap, int(max_events)
    )
    return SSAResult(t_grid, X, C, float(t), x, counts, int(n), t_ext, x_max, int(stop))
