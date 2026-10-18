import numpy as np
import pytest

from exp_resistance import BASE, dominance_time, one, theory


def test_dominance_time():
    t = np.arange(5.0)
    assert dominance_time(t, np.array([10, 5, 2, 1, 0]), np.array([0, 0, 3, 4, 5])) == 2.0
    assert np.isnan(dominance_time(t, np.array([10, 5, 2, 1, 0]), np.array([0, 0, 0, 0, 0])))
    assert np.isnan(dominance_time(t, np.zeros(5), np.zeros(5)))


def test_theory_band_is_ordered_and_bounded():
    p = BASE.replace(eps_b=0.9)
    state = np.array([800.0, 20.0, 2.0, 500.0, 30.0, 100.0])
    pre_lo, pre_hi, dn, lo, hi = theory(p, state, 30.0)
    assert 0.0 <= pre_lo <= pre_hi <= 1.0 and 0.0 <= dn <= 1.0 and 0.0 <= lo <= hi <= 1.0
    # no mutant present: only the de novo term remains
    state[2] = state[4] = 0.0
    pre_lo, pre_hi, dn, lo, hi = theory(p, state, 30.0)
    assert pre_lo == 0.0 and pre_hi == 0.0 and lo == pytest.approx(dn) and hi == pytest.approx(dn)


def test_theory_de_novo_vanishes_without_mutation():
    p = BASE.replace(eps_b=0.9, mu=0.0)
    assert theory(p, np.array([1000.0, 20.0, 0.0, 500.0, 0.0, 10.0]), 30.0)[2] == 0.0


def test_two_phase_run_is_reproducible():
    first = one(42, 30, 0.9)
    assert first is not None, "the pre-treatment infection died; the resistance preset should be viable"
    a, _ = first
    b, _ = one(42, 30, 0.9)
    clean = lambda d: {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}
    assert clean(a) == clean(b)
    assert set(a) >= {"pre", "emerged", "mut_h", "wt_h", "dominant_h", "t_dom", "p_lo", "p_hi"}
