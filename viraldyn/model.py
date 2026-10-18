"""Two-strain within-host viral model with a CTL response.

State x = (T, Iw, Im, Vw, Vm, E): uninfected target cells, cells infected by the
wild type (w) or the mutant (m), free virions of each strain, and effector CTLs.
Fifteen reaction channels; see the table in PLAN.md. The deterministic model is
dx/dt = S a(x), so the ODE and the stochastic simulation share one definition.
"""

from typing import NamedTuple

import numpy as np
from numba import njit
from scipy.integrate import solve_ivp

T, IW, IM, VW, VM, E = range(6)
SPECIES = ("T", "Iw", "Im", "Vw", "Vm", "E")
N_SPECIES = 6
N_REACTIONS = 15


class Params(NamedTuple):
    lam: float  # target-cell production (cells / day)
    dT: float  # target-cell death rate
    bw: float  # wild-type infection rate
    bm: float  # mutant infection rate
    pw: float  # wild-type virion production rate per infected cell
    pm: float  # mutant virion production rate per infected cell
    mu: float  # probability that a wild-type-produced virion is a mutant
    cw: float  # wild-type virion clearance rate
    cm: float  # mutant virion clearance rate
    delta: float  # infected-cell death rate
    k: float  # CTL killing rate
    q: float  # CTL proliferation rate
    dE: float  # CTL death rate
    eps_b: float = 0.0  # drug efficacy blocking wild-type infection
    eps_p: float = 0.0  # drug efficacy blocking wild-type virion production

    @staticmethod
    def make(**kw) -> "Params":
        """Build from keyword arguments, casting to float (numba needs one type)."""
        return Params(**{key: float(v) for key, v in kw.items()})

    def replace(self, **kw) -> "Params":
        return Params.make(**{**self._asdict(), **kw})


def stoichiometry() -> np.ndarray:
    S = np.zeros((N_SPECIES, N_REACTIONS), dtype=np.int64)
    S[[T, IW, VW], 0] = (-1, +1, -1)  # 0  wild-type infection
    S[[T, IM, VM], 1] = (-1, +1, -1)  # 1  mutant infection
    S[VW, 2] = +1  # 2  wild-type production
    S[VM, 3] = +1  # 3  wild-type production with mutation
    S[VM, 4] = +1  # 4  mutant production
    S[T, 5] = +1  # 5  target-cell production
    S[T, 6] = -1  # 6  target-cell death
    S[VW, 7] = -1  # 7  wild-type clearance
    S[VM, 8] = -1  # 8  mutant clearance
    S[IW, 9] = -1  # 9  Iw lysis
    S[IM, 10] = -1  # 10 Im lysis
    S[IW, 11] = -1  # 11 CTL kills Iw
    S[IM, 12] = -1  # 12 CTL kills Im
    S[E, 13] = +1  # 13 CTL proliferation
    S[E, 14] = -1  # 14 CTL death
    return S


STOICH = stoichiometry()
STOICH.setflags(write=False)

REACTIONS = (
    "WT infection",
    "mutant infection",
    "WT production",
    "WT production with mutation",
    "mutant production",
    "target production",
    "target death",
    "WT clearance",
    "mutant clearance",
    "Iw lysis",
    "Im lysis",
    "CTL kills Iw",
    "CTL kills Im",
    "CTL proliferation",
    "CTL death",
)


@njit(cache=True)
def propensities(x, p, out):
    """Reaction propensities a_j(x) written into `out` (length 15)."""
    Tc, Iw, Im, Vw, Vm, Ec = x[0], x[1], x[2], x[3], x[4], x[5]
    out[0] = (1.0 - p.eps_b) * p.bw * Tc * Vw
    out[1] = p.bm * Tc * Vm
    out[2] = (1.0 - p.eps_p) * p.pw * (1.0 - p.mu) * Iw
    out[3] = (1.0 - p.eps_p) * p.pw * p.mu * Iw
    out[4] = p.pm * Im
    out[5] = p.lam
    out[6] = p.dT * Tc
    out[7] = p.cw * Vw
    out[8] = p.cm * Vm
    out[9] = p.delta * Iw
    out[10] = p.delta * Im
    out[11] = p.k * Iw * Ec
    out[12] = p.k * Im * Ec
    out[13] = p.q * (Iw + Im) * Ec
    out[14] = p.dE * Ec
    return out


def ode_rhs(t, x, p):
    """dx/dt = S a(x). Negative components (solver undershoot) are clipped to 0."""
    a = propensities(np.maximum(x, 0.0), p, np.empty(N_REACTIONS))
    return STOICH @ a


def solve_ode(p, x0, t_eval, rtol=1e-8, atol=1e-8):
    """Deterministic trajectory on `t_eval`; returns an array of shape (len(t_eval), 6)."""
    t_eval = np.asarray(t_eval, dtype=float)
    sol = solve_ivp(
        ode_rhs,
        (t_eval[0], t_eval[-1]),
        np.asarray(x0, dtype=float),
        t_eval=t_eval,
        args=(p,),
        method="LSODA",
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T


def target_cell_equilibrium(p) -> float:
    """Uninfected steady state T* = lambda / d_T."""
    return p.lam / p.dT


def r0_wild_type(p) -> float:
    """Basic reproduction number of the wild type at the disease-free state (E = 0).

    One infected cell lives 1/delta and produces (1-eps_p) p_w (1-mu) wild-type
    virions, each of which infects with probability beta T* / (c + beta T*) because
    a virion is consumed on infection.
    """
    Ts = target_cell_equilibrium(p)
    beta = (1.0 - p.eps_b) * p.bw
    infect = beta * Ts / (p.cw + beta * Ts)
    return (1.0 - p.eps_p) * p.pw * (1.0 - p.mu) / p.delta * infect


def r0_mutant(p) -> float:
    """Basic reproduction number of the mutant (not affected by the drug)."""
    Ts = target_cell_equilibrium(p)
    infect = p.bm * Ts / (p.cm + p.bm * Ts)
    return p.pm / p.delta * infect


def immune_activation_threshold(p) -> float:
    """Infected-cell load above which CTLs grow: q (Iw + Im) > d_E."""
    return p.dE / p.q


# Parameter sets carried over from the course notebook (illustrative, not fitted).
PRESETS = {
    "chronic": Params.make(
        lam=1e4, dT=0.01, bw=2e-5, bm=1e-5, pw=20.0, pm=18.0, mu=1e-4,
        cw=4.0, cm=4.0, delta=0.5, k=1e-4, q=1e-4, dE=0.1,
    ),
    "acute": Params.make(
        lam=100.0, dT=0.1, bw=2e-3, bm=2e-3, pw=100.0, pm=100.0, mu=1e-4,
        cw=25.0, cm=25.0, delta=1.0, k=0.2, q=0.05, dE=0.1,
    ),
    # notebook section "Probability of Infection Establishment": R0 = 33, the ODE clears
    "establishment": Params.make(
        lam=10.0, dT=0.01, bw=0.02, bm=0.016, pw=20.0, pm=20.0, mu=1e-4,
        cw=4.0, cm=4.0, delta=0.5, k=0.1, q=0.1, dE=0.1,
    ),
    # notebook section 3.3, deterministic competitive release. Its deterministic chronic
    # infection passes through 3e-8 virions at the first trough; stochastically the
    # infection dies there in every realisation, so it is not used for the therapy grid.
    "therapy": Params.make(
        lam=100.0, dT=0.1, bw=2e-4, bm=1.6e-4, pw=100.0, pm=100.0, mu=1e-4,
        cw=4.0, cm=4.0, delta=1.0, k=0.01, q=0.05, dE=0.1,
    ),
    # "therapy" with a slower, weaker effector response (q, k) so that the chronic set
    # point (I* = d_E / q = 50 infected cells, about 1200 virions) is stochastically
    # viable while a mutant lineage is present only part of the time. Same R0 values.
    "resistance": Params.make(
        lam=100.0, dT=0.1, bw=2e-4, bm=1.6e-4, pw=100.0, pm=100.0, mu=1e-4,
        cw=4.0, cm=4.0, delta=1.0, k=0.003, q=0.002, dE=0.1,
    ),
}
