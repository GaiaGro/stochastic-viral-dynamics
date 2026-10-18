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
"""

from typing import NamedTuple

import numpy as np
from scipy.integrate import cumulative_trapezoid

from .model import E, IM, IW, VM, VW, Params, solve_ode


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
    q = np.zeros(4)
    for _ in range(max_iter):
        s_w = (1.0 - p.mu) * q[VW - 3] + p.mu * q[VM - 3]  # offspring of a WT-infected cell
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
    """1 - P(every founding lineage in x0 dies out).

    By default the target-cell count is x0[T] and the fixed effector count x0[E].
    """
    x0 = np.asarray(x0, dtype=float)
    q = extinction_probabilities(p, x0[0] if T is None else T, x0[E] if E_fixed is None else E_fixed)
    log_ext = x0[VW] * np.log(q.q_Vw) + x0[IW] * np.log(q.q_Iw) + x0[VM] * np.log(q.q_Vm) + x0[IM] * np.log(q.q_Im)
    return float(1.0 - np.exp(log_ext))


def single_virion_closed_form(p: Params, T, E_fixed=0.0) -> float:
    """P(establish from one wild-type virion) for mu = 0: pi (1 - 1/R0), or 0 if R0 <= 1."""
    d = p.delta + p.k * E_fixed
    pi = infection_probability((1.0 - p.eps_b) * p.bw, p.cw, T)
    r0 = pi * (1.0 - p.eps_p) * p.pw / d
    return max(0.0, pi * (1.0 - 1.0 / r0)) if r0 > 0 else 0.0


# --- the effector lineage: a linear birth-death process driven by the infected-cell load ---


def kendall_extinction(t, birth, death):
    """Probability that a linear birth-death lineage founded by one individual at
    t[0] is extinct by each time in `t`, for a per-capita birth rate given on the
    grid by `birth` and a constant per-capita death rate `death` (Kendall 1948):
    rho(t) = death t - int_0^t birth, W(t) = 1 + int_0^t death e^{rho}, p0 = 1 - 1/W.
    """
    t = np.asarray(t, dtype=float)
    rho = death * (t - t[0]) - cumulative_trapezoid(np.asarray(birth, dtype=float), t, initial=0.0)
    W = 1.0 + cumulative_trapezoid(death * np.exp(rho), t, initial=0.0)
    return 1.0 - 1.0 / W


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
