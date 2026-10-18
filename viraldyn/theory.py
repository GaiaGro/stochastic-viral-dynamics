"""Branching-process approximations for the early phase of infection.

While target cells are abundant and the effector population is fixed, every virion
and every infected cell founds an independent lineage. A virion infects with
probability pi = beta T / (c + beta T) and is otherwise cleared. An infected cell
lives for an Exp(d) time with d = delta + k E and produces virions at rate p, so
the number it produces is geometric with generating function
g(s) = d / (d + p (1 - s)). Wild-type cells produce a mutant virion with
probability mu, which couples the two strains into a two-type process; mutant
lineages only produce mutants. Extinction probabilities are the smallest fixed
point of the generating-function equations, found by iterating from zero.

For mu = 0 the wild-type equations collapse to a quadratic with the closed form
P(establish from one virion) = pi (1 - 1/R0), with R0 = pi p / d.

When the environment changes in time (effectors decaying or expanding, target
cells recovering) a lineage of infected cells is approximated by a linear
birth-death process with per-capita birth rate p pi(T(t)) (virions produced times
the probability each one infects) and death rate delta + k E(t); in a constant
environment this gives the same per-cell extinction probability 1/R as the exact
two-stage process. Kendall (1948) gives its extinction probability in closed form.
"""

from typing import NamedTuple

import numpy as np
from scipy.integrate import cumulative_trapezoid

from .model import E, IM, IW, T, VM, VW, Params, solve_ode


class ExtinctionProbabilities(NamedTuple):
    q_Vw: float  # lineage founded by one wild-type virion
    q_Iw: float  # lineage founded by one wild-type-infected cell
    q_Vm: float  # one mutant virion
    q_Im: float  # one mutant-infected cell


def infection_probability(beta, c, T) -> float:
    """Probability that a free virion infects a cell before it is cleared."""
    return beta * T / (c + beta * T)


def extinction_probabilities(p: Params, T, E_fixed=0.0, tol=1e-14, max_iter=2_000_000) -> ExtinctionProbabilities:
    """Smallest fixed point of the two-type generating-function equations.

    `T` is the target-cell count seen by the founding lineage and `E_fixed` the
    (constant) effector count, which adds k E_fixed to the infected-cell death rate.
    Drug efficacies in `p` are applied to the wild type.
    """
    d = p.delta + p.k * E_fixed
    pi_w = infection_probability((1.0 - p.eps_b) * p.bw, p.cw, T)
    pi_m = infection_probability(p.bm, p.cm, T)
    pw = (1.0 - p.eps_p) * p.pw
    q = np.zeros(4)  # order: q_Vw, q_Iw, q_Vm, q_Im
    for _ in range(max_iter):
        s_w = (1.0 - p.mu) * q[0] + p.mu * q[2]  # a WT cell's virion is wild type or mutant
        new = np.array(
            [
                (1.0 - pi_w) + pi_w * q[1],
                d / (d + pw * (1.0 - s_w)),
                (1.0 - pi_m) + pi_m * q[3],
                d / (d + p.pm * (1.0 - q[2])),
            ]
        )
        done = np.abs(new - q).max() < tol
        q = new
        if done:
            break
    return ExtinctionProbabilities(*np.minimum(q, 1.0))


def establishment_probability(p: Params, x0, T=None, E_fixed=None) -> float:
    """1 - P(every founding lineage in x0 dies out), effectors held fixed.

    By default the target-cell count is x0[T] and the fixed effector count x0[E].
    """
    x0 = np.asarray(x0, dtype=float)
    q = extinction_probabilities(p, x0[0] if T is None else T, x0[E] if E_fixed is None else E_fixed)
    log_ext = x0[VW] * np.log(q.q_Vw) + x0[IW] * np.log(q.q_Iw) + x0[VM] * np.log(q.q_Vm) + x0[IM] * np.log(q.q_Im)
    return float(1.0 - np.exp(log_ext))


def mutant_lineage_probabilities(p: Params, T, E_fixed=0.0):
    """Closed-form extinction probabilities (q_Vm, q_Im) of lineages founded by one
    mutant virion or one mutant-infected cell, vectorised over `T` and `E_fixed`.

    The mutant only produces mutants, so it is a one-type process: a virion founds a
    surviving lineage with probability pi_m (1 - 1/R_m) when R_m = pi_m p_m / d > 1,
    with d = delta + k E, and never otherwise.
    """
    T = np.asarray(T, dtype=float)
    E_fixed = np.asarray(E_fixed, dtype=float)
    d = p.delta + p.k * E_fixed
    pi = infection_probability(p.bm, p.cm, T)
    R = pi * p.pm / d
    u_V = np.where(R > 1.0, pi * (1.0 - 1.0 / np.maximum(R, 1e-300)), 0.0)
    q_I = d / (d + p.pm * u_V)
    return 1.0 - u_V, q_I


def single_virion_closed_form(p: Params, T, E_fixed=0.0) -> float:
    """P(establish from one wild-type virion) for mu = 0: pi (1 - 1/R0), or 0 if R0 <= 1."""
    d = p.delta + p.k * E_fixed
    pi = infection_probability((1.0 - p.eps_b) * p.bw, p.cw, T)
    r0 = pi * (1.0 - p.eps_p) * p.pw / d
    return max(0.0, pi * (1.0 - 1.0 / r0)) if r0 > 0 else 0.0


# --- linear birth-death lineages in a changing environment (Kendall 1948) ---

_EXP_CAP = 700.0  # exp overflows above this; beyond it W is effectively infinite and p0 = 1


def kendall_extinction(t, birth, death):
    """Probability that a linear birth-death lineage founded by one individual at
    t[0] is extinct by each time in `t`, for per-capita birth and death rates given
    on the grid (`death` may be a scalar):
    rho(t) = int_0^t (death - birth), W(t) = 1 + int_0^t death e^{rho}, p0 = 1 - 1/W.
    """
    t = np.asarray(t, dtype=float)
    birth = np.broadcast_to(np.asarray(birth, dtype=float), t.shape)
    death = np.broadcast_to(np.asarray(death, dtype=float), t.shape)
    rho = cumulative_trapezoid(death - birth, t, initial=0.0)
    W = 1.0 + cumulative_trapezoid(death * np.exp(np.minimum(rho, _EXP_CAP)), t, initial=0.0)
    return 1.0 - 1.0 / W


def lineage_extinction_from(t, birth, death):
    """Extinction probability by t[-1] of a lineage founded by one individual at
    each grid time t[i], for per-capita rates `birth(t)`, `death(t)` on the grid.
    Same formula as `kendall_extinction` with the integrals starting at t[i].
    O(n^2); the grid step must be small against 1 / |birth - death| (trapezoid)."""
    t = np.asarray(t, dtype=float)
    birth = np.broadcast_to(np.asarray(birth, dtype=float), t.shape)
    death = np.broadcast_to(np.asarray(death, dtype=float), t.shape)
    G = cumulative_trapezoid(death - birth, t, initial=0.0)
    p0 = np.empty(t.shape)
    for i in range(t.size):
        if i == t.size - 1:
            p0[i] = 0.0  # no time left to die
            continue
        integrand = death[i:] * np.exp(np.minimum(G[i:] - G[i], _EXP_CAP))
        p0[i] = 1.0 - 1.0 / (1.0 + np.trapezoid(integrand, t[i:]))
    return p0


def ctl_extinction_curve(p: Params, x0, t):
    """Extinction probability by time t of the lineage of one effector cell.

    Its birth rate q I(t) follows the deterministic infected-cell load in the
    absence of immune killing (k = 0, E = 0): while the lineage is small it does not
    affect the infection, and by the time it is large its fate is decided.
    """
    x_free = np.asarray(x0, dtype=float).copy()
    x_free[E] = 0.0
    sol = solve_ode(p.replace(k=0.0), x_free, t)
    return kendall_extinction(t, p.q * (sol[:, IW] + sol[:, IM]), p.dE)


def immune_failure_probability(p: Params, x0, t_max=60.0, n=60001) -> float:
    """P(all x0[E] founding effector lineages die out) = p0(t_max)^E0."""
    t = np.linspace(0.0, t_max, n)
    return float(ctl_extinction_curve(p, x0, t)[-1] ** float(x0[E]))


def mutant_lineage_rates(p: Params, T, E_t):
    """Lumped birth and death rates of a mutant-infected-cell lineage in the
    environment (T, E): birth p_m pi_m(T), death delta + k E."""
    return p.pm * infection_probability(p.bm, p.cm, np.asarray(T, dtype=float)), p.delta + p.k * np.asarray(E_t, dtype=float)


def establishment_probability_decaying_effectors(p: Params, x0, t_max=400.0, n=8001) -> float:
    """P(establish) for x0 when the effectors are not yet activated and decay from
    x0[E] at rate d_E while target cells stay at x0[T]: a wild-type cell lineage has
    birth rate p_w (1 - mu) pi_w and death rate delta + k E0 e^{-d_E t} (Kendall).
    The time grid is log-spaced so that the fast early decay of e^{rho} is resolved."""
    x0 = np.asarray(x0, dtype=float)
    t = np.concatenate([[0.0], np.logspace(-5, np.log10(t_max), n - 1)])
    pi = infection_probability((1.0 - p.eps_b) * p.bw, p.cw, x0[T])
    birth = (1.0 - p.eps_p) * p.pw * (1.0 - p.mu) * pi
    death = p.delta + p.k * x0[E] * np.exp(-p.dE * t)
    p0_cell = kendall_extinction(t, np.full(t.shape, birth), death)[-1]
    q_virion = 1.0 - pi * (1.0 - p0_cell)
    return float(1.0 - q_virion ** x0[VW] * p0_cell ** x0[IW])
