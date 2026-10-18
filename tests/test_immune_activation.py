import numpy as np
import pytest

from exp_immune_activation import classify
from viraldyn.model import PRESETS, Params
from viraldyn.ssa import STOP_EXTINCT, STOP_T_END, simulate
from viraldyn.stats import wilson
from viraldyn.theory import (
    establishment_probability,
    establishment_probability_decaying_effectors,
    immune_failure_probability,
    kendall_extinction,
    lineage_extinction_from,
)

P = PRESETS["establishment"]
ZERO = Params.make(lam=0, dT=0, bw=0, bm=0, pw=0, pm=0, mu=0, cw=0, cm=0, delta=0, k=0, q=0, dE=0)


def test_kendall_constant_rates_closed_form():
    lam, mu = 1.0, 2.0
    t = np.linspace(0, 3, 30001)
    exact = mu * (np.exp((lam - mu) * t) - 1) / (lam * np.exp((lam - mu) * t) - mu)
    assert np.allclose(kendall_extinction(t, np.full_like(t, lam), mu), exact, atol=1e-6)
    lam, mu = 2.0, 1.0  # supercritical: ultimate extinction mu / lam
    t = np.linspace(0, 40, 40001)
    assert kendall_extinction(t, np.full_like(t, lam), mu)[-1] == pytest.approx(0.5, abs=1e-6)


def test_kendall_with_time_varying_death():
    t = np.linspace(0, 20, 20001)
    lam, mu = 1.0, 2.0
    constant = kendall_extinction(t, np.full_like(t, lam), mu)
    as_array = kendall_extinction(t, np.full_like(t, lam), np.full_like(t, mu))
    assert np.allclose(constant, as_array)
    # death rate decaying to zero: the lineage eventually becomes supercritical, so p0 < 1
    decaying = kendall_extinction(t, np.full_like(t, lam), 3.0 * np.exp(-t))
    assert 0.5 < decaying[-1] < 1.0
    # founded later in a constant environment: same ultimate extinction probability
    from_each = lineage_extinction_from(t, np.full_like(t, 2.0), np.full_like(t, 1.0))
    assert from_each[0] == pytest.approx(0.5, abs=1e-5) and from_each[5000] == pytest.approx(0.5, abs=1e-5)
    assert from_each[-1] == 0.0


def test_decaying_effector_establishment_limits():
    p = PRESETS["establishment"].replace(q=0.002)
    x0 = [1000, 0, 0, 5, 0, 10]
    with_decay = establishment_probability_decaying_effectors(p, x0)
    fixed = establishment_probability(p, x0)
    none = establishment_probability(p, x0, E_fixed=0.0)
    assert fixed <= with_decay <= none  # decaying effectors lie between fixed effectors and no effectors
    assert establishment_probability_decaying_effectors(p, [1000, 0, 0, 5, 0, 0]) == pytest.approx(none, abs=1e-6)


def test_kendall_matches_ssa_constant_rates():
    """Only effector proliferation and death active, Iw frozen at 20: birth q*20 = 1, death 2."""
    p = ZERO.replace(q=0.05, dE=2.0)
    n = 1000
    dead = sum(int(simulate(p, [0, 20, 0, 0, 0, 1], [0.0, 1.0], seed_value=9000 + s).x_end[5] == 0) for s in range(n))
    lo, hi = wilson(dead, n, z=3.29)
    assert lo < kendall_extinction(np.linspace(0, 1, 1001), np.full(1001, 1.0), 2.0)[-1] < hi


def test_immune_failure_probability_is_monotone_and_converged():
    x0 = lambda E0: [1000, 0, 0, 1, 0, E0]
    assert all(np.diff([immune_failure_probability(P, x0(e)) for e in (1, 2, 5)]) < 0)
    assert all(np.diff([immune_failure_probability(P.replace(q=q), x0(1)) for q in (0.01, 0.1, 1.0)]) < 0)
    assert immune_failure_probability(P, x0(1), t_max=40) == pytest.approx(immune_failure_probability(P, x0(1), t_max=60), abs=1e-6)
    assert 0.02 < immune_failure_probability(P, x0(1)) < 0.5


def test_ssa_immune_failure_matches_theory_without_feedback():
    """k = 0: effectors do not act on the infection, so the lineage is a linear
    birth-death process driven by I(t). Compare with the deterministic-forcing formula."""
    p = P.replace(k=0.0)
    x0 = [1000, 0, 0, 1, 0, 1]
    fails = established = 0
    for s in range(400):
        r = simulate(p, x0, [0.0, 20.0], seed_value=7000 + s, stop_on_extinction=True, stop_above=[0, 0, 0, 0, 0, 1000])
        if r.cumulative_infections >= 100:
            established += 1
            fails += int(not np.isnan(r.t_ext[2]))
    lo, hi = wilson(fails, established, z=3.29)
    assert lo < immune_failure_probability(p, x0, t_max=20) < hi


def test_classify():
    nan = float("nan")
    assert classify(STOP_EXTINCT, 1.0, nan, 5) == "bottleneck"
    assert classify(STOP_EXTINCT, 2.0, nan, 500) == "cleared"
    assert classify(STOP_EXTINCT, 2.0, 0.5, 500) == "other"
    assert classify(STOP_T_END, 60.0, 0.3, 500) == "immune_failure"
    assert classify(STOP_T_END, 60.0, nan, 500) == "persistent_immune"
