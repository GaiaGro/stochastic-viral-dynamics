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

from .model import E, IM, IW, VM, VW, Params


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
