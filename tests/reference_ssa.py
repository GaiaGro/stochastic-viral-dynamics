"""The pure-Python Gillespie implementation from the course notebook, kept as a
reference for distributional tests of the compiled simulator. Not used elsewhere."""

import numpy as np

from viraldyn.model import STOICH


def gillespie_reference(p, t_max, x0, rng):
    t = 0.0
    state = np.asarray(x0, dtype=np.int64).copy()
    S = STOICH
    while t < t_max:
        T, Iw, Im, Vw, Vm, E = state
        a = np.array(
            [
                (1 - p.eps_b) * p.bw * T * Vw,
                p.bm * T * Vm,
                (1 - p.eps_p) * p.pw * (1 - p.mu) * Iw,
                (1 - p.eps_p) * p.pw * p.mu * Iw,
                p.pm * Im,
                p.lam,
                p.dT * T,
                p.cw * Vw,
                p.cm * Vm,
                p.delta * Iw,
                p.delta * Im,
                p.k * Iw * E,
                p.k * Im * E,
                p.q * (Iw + Im) * E,
                p.dE * E,
            ]
        )
        a0 = a.sum()
        if a0 == 0:
            return t_max, state
        tau = rng.exponential(1 / a0)
        if t + tau > t_max:
            return t_max, state
        t += tau
        j = np.searchsorted(np.cumsum(a), rng.uniform(0, a0))
        state = state + S[:, j]
        if state[1] + state[2] + state[3] + state[4] == 0:
            return t, state
    return t, state
