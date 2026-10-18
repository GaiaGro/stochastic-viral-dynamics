import numpy as np
import pytest

from viraldyn.model import (
    E,
    IM,
    IW,
    N_REACTIONS,
    PRESETS,
    STOICH,
    T,
    VM,
    VW,
    Params,
    ode_rhs,
    propensities,
    r0_mutant,
    r0_wild_type,
    target_cell_equilibrium,
)

RNG = np.random.default_rng(0)


def test_stoichiometry_conservation():
    S = STOICH
    # infections: one target cell becomes one infected cell and one virion is consumed
    for j, (I, V) in ((0, (IW, VW)), (1, (IM, VM))):
        assert S[T, j] + S[I, j] == 0 and S[V, j] == -1
        assert np.abs(S[:, j]).sum() == 3
    # every other channel changes exactly one species by exactly one
    for j in range(2, N_REACTIONS):
        assert np.abs(S[:, j]).sum() == 1
    # production channels create virions from infected cells without consuming them
    assert S[VW, 2] == 1 and S[VM, 3] == 1 and S[VM, 4] == 1
    assert S[IW, 2] == 0 and S[IW, 3] == 0 and S[IM, 4] == 0
    # CTL channels: proliferation +1, death -1, killing removes an infected cell
    assert S[E, 13] == 1 and S[E, 14] == -1 and S[IW, 11] == -1 and S[IM, 12] == -1


@pytest.mark.parametrize("name", list(PRESETS))
def test_propensities_nonnegative(name):
    p = PRESETS[name]
    for _ in range(50):
        x = RNG.integers(0, 10_000, size=6)
        a = propensities(x, p, np.empty(N_REACTIONS))
        assert (a >= 0).all()


def written_equations(x, p):
    """The six ODEs as written in the paper's Model section, with drug efficacies."""
    Tc, Iw, Im, Vw, Vm, Ec = x
    bw = (1 - p.eps_b) * p.bw
    pw = (1 - p.eps_p) * p.pw
    return np.array(
        [
            p.lam - p.dT * Tc - bw * Tc * Vw - p.bm * Tc * Vm,
            bw * Tc * Vw - p.delta * Iw - p.k * Iw * Ec,
            p.bm * Tc * Vm - p.delta * Im - p.k * Im * Ec,
            pw * (1 - p.mu) * Iw - p.cw * Vw - bw * Tc * Vw,
            pw * p.mu * Iw + p.pm * Im - p.cm * Vm - p.bm * Tc * Vm,
            p.q * (Iw + Im) * Ec - p.dE * Ec,
        ]
    )


@pytest.mark.parametrize("name", list(PRESETS))
@pytest.mark.parametrize("drug", [(0.0, 0.0), (0.9, 0.0), (0.0, 0.7), (0.5, 0.5)])
def test_ode_matches_reaction_network(name, drug):
    p = PRESETS[name].replace(eps_b=drug[0], eps_p=drug[1])
    for _ in range(20):
        x = RNG.uniform(0, 1e4, size=6)
        np.testing.assert_allclose(ode_rhs(0.0, x, p), written_equations(x, p), rtol=1e-12, atol=1e-9)


def next_generation_r0(p, strain):
    """Spectral radius of F V^-1 for the (I, V) infection subsystem at the disease-free state."""
    Ts = target_cell_equilibrium(p)
    if strain == "wt":
        beta, prod, c = (1 - p.eps_b) * p.bw, (1 - p.eps_p) * p.pw * (1 - p.mu), p.cw
    else:
        beta, prod, c = p.bm, p.pm, p.cm
    F = np.array([[0.0, beta * Ts], [0.0, 0.0]])  # new infections
    V = np.array([[p.delta, 0.0], [-prod, c + beta * Ts]])  # transitions and removal
    return max(abs(np.linalg.eigvals(F @ np.linalg.inv(V))))


@pytest.mark.parametrize("name", list(PRESETS))
@pytest.mark.parametrize("eps_b", [0.0, 0.9])
def test_r0_formulas_match_next_generation_matrix(name, eps_b):
    p = PRESETS[name].replace(eps_b=eps_b)
    assert r0_wild_type(p) == pytest.approx(next_generation_r0(p, "wt"), rel=1e-10)
    assert r0_mutant(p) == pytest.approx(next_generation_r0(p, "mut"), rel=1e-10)


def test_r0_threshold_matches_linear_stability():
    """R0 > 1 iff the disease-free state is unstable to the wild type (E = 0)."""
    base = PRESETS["therapy"]
    for bw in np.logspace(-7, -3, 15):
        p = base.replace(bw=bw)
        Ts = target_cell_equilibrium(p)
        J = np.array([[-p.delta, p.bw * Ts], [p.pw * (1 - p.mu), -p.cw - p.bw * Ts]])
        growth = max(np.linalg.eigvals(J).real)
        assert (r0_wild_type(p) > 1) == (growth > 0)


def test_params_make_casts_to_float():
    p = Params.make(lam=10, dT=1, bw=1, bm=1, pw=1, pm=1, mu=0, cw=1, cm=1, delta=1, k=0, q=0, dE=0)
    assert all(isinstance(v, float) for v in p)
    assert p.eps_b == 0.0 and p.replace(eps_b=0.5).eps_b == 0.5
