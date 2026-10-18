import numpy as np
import pytest

from viraldyn.model import PRESETS, r0_wild_type
from viraldyn.ssa import STOP_CUM_INFECTIONS, STOP_EXTINCT, simulate
from viraldyn.stats import wilson
from viraldyn.theory import (
    establishment_probability,
    extinction_probabilities,
    infection_probability,
    single_virion_closed_form,
)

P = PRESETS["establishment"]


def test_single_virion_matches_closed_form():
    for E0 in (0.0, 5.0, 20.0, 50.0):
        p = P.replace(mu=0.0)
        q = extinction_probabilities(p, T=1000.0, E_fixed=E0)
        assert 1.0 - q.q_Vw == pytest.approx(single_virion_closed_form(p, 1000.0, E0), abs=1e-9)
        # the mutant is a one-type process in its own right
        pm = p.replace(bw=p.bm, pw=p.pm, cw=p.cm)
        assert 1.0 - q.q_Vm == pytest.approx(single_virion_closed_form(pm, 1000.0, E0), abs=1e-9)


def test_mutant_closed_form_matches_fixed_point():
    from viraldyn.theory import mutant_lineage_probabilities

    p = PRESETS["therapy"]
    for T, E0 in ((1000.0, 0.0), (1000.0, 50.0), (300.0, 10.0), (1000.0, 400.0)):
        q = extinction_probabilities(p, T=T, E_fixed=E0)
        qV, qI = mutant_lineage_probabilities(p, T, E0)
        assert float(qV) == pytest.approx(q.q_Vm, abs=1e-9) and float(qI) == pytest.approx(q.q_Im, abs=1e-9)
    qV, qI = mutant_lineage_probabilities(p, [1000.0, 100.0], [0.0, 500.0])  # vectorised; second is subcritical
    assert qV.shape == (2,) and qV[1] == 1.0 and qI[1] == 1.0 and qV[0] < 1.0


def test_subcritical_lineages_die_out():
    p = P.replace(bw=1e-5, bm=1e-5, mu=0.0)  # R0 far below 1
    assert r0_wild_type(p) < 1
    q = extinction_probabilities(p, T=1000.0)
    assert q == pytest.approx((1.0, 1.0, 1.0, 1.0), abs=1e-9)
    assert establishment_probability(p, [1000, 0, 0, 5, 0, 0]) == pytest.approx(0.0, abs=1e-9)


def test_two_type_coupling_uses_the_mutant_virion():
    """With mu = 0.5 and mutant virions that cannot infect (bm = 0), a wild-type
    cell's effective production is p (1 - mu) and the wild type is one-type again."""
    p = P.replace(mu=0.5, bm=0.0, k=0.0)
    q = extinction_probabilities(p, T=1000.0)
    assert 1.0 - q.q_Vw == pytest.approx(single_virion_closed_form(p.replace(pw=p.pw * (1 - p.mu), mu=0.0), 1000.0), abs=1e-9)
    assert q.q_Vm == 1.0 and q.q_Im == 1.0


def test_two_type_fixed_point_matches_hand_iteration():
    p = P.replace(mu=0.2)
    d, T0 = p.delta + p.k * 0.0, 1000.0
    pi_w = infection_probability(p.bw, p.cw, T0)
    pi_m = infection_probability(p.bm, p.cm, T0)
    qVw = qIw = qVm = qIm = 0.0
    for _ in range(100000):
        qVw, qIw, qVm, qIm = (
            (1 - pi_w) + pi_w * qIw,
            d / (d + p.pw * (1 - ((1 - p.mu) * qVw + p.mu * qVm))),
            (1 - pi_m) + pi_m * qIm,
            d / (d + p.pm * (1 - qVm)),
        )
    q = extinction_probabilities(p, T=T0)
    assert q == pytest.approx((qVw, qIw, qVm, qIm), abs=1e-10)


def test_establishment_is_monotone():
    pest = [establishment_probability(P, [1000, 0, 0, v, 0, 0]) for v in (1, 2, 5, 10)]
    assert all(np.diff(pest) > 0)
    pE = [establishment_probability(P, [1000, 0, 0, 1, 0, 0], E_fixed=e) for e in (0, 5, 20, 50)]
    assert all(np.diff(pE) < 0)
    assert establishment_probability(P, [1000, 0, 0, 1, 0, 0]) == pytest.approx(0.808, abs=2e-3)


def test_infection_probability_limits():
    assert infection_probability(1.0, 0.0, 10.0) == 1.0
    assert infection_probability(0.0, 1.0, 10.0) == 0.0


def test_ssa_establishment_frequency_matches_theory():
    """Immune response off (k = 0): a single virion establishes with probability
    pi (1 - 1/R0). Establishment means 100 cumulative infections before extinction."""
    p = P.replace(k=0.0, q=0.0, dE=0.0)
    x0 = [1000, 0, 0, 1, 0, 0]
    n = 2000
    outcomes = [simulate(p, x0, [0.0, 500.0], seed_value=5000 + s, stop_on_extinction=True, stop_cum_infections=100).stop_reason for s in range(n)]
    assert set(outcomes) <= {STOP_EXTINCT, STOP_CUM_INFECTIONS}
    k = outcomes.count(STOP_CUM_INFECTIONS)
    lo, hi = wilson(k, n, z=3.29)  # 99.9%: a fixed seed set must not fail by chance
    assert lo < establishment_probability(p, x0) < hi


def test_wilson_interval():
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and 0.25 < hi < 0.35
    lo, hi = wilson(50, 100)
    assert lo == pytest.approx(0.404, abs=2e-3) and hi == pytest.approx(0.596, abs=2e-3)
