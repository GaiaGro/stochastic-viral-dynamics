import numpy as np
import pytest

from exp_regime_map import BASE, X0, classify_ode, k_threshold, params, r_eff
from viraldyn import r0_wild_type


def test_r_eff_reduces_to_r0_without_effectors():
    p = params(2e-3, 0.5)
    assert r_eff(p, 0.0) == pytest.approx(r0_wild_type(p), rel=1e-12)
    assert r_eff(p, 10.0) < r_eff(p, 0.0)


def test_k_threshold_puts_r_eff_at_one():
    for beta in (3e-4, 1e-3, 1e-2):
        k = float(k_threshold(beta, X0[5]))
        assert r_eff(params(beta, k), X0[5]) == pytest.approx(1.0, rel=1e-9)
    assert np.isnan(k_threshold(1e-5, X0[5]))  # R0 < 1 even without effectors


def test_classify_ode():
    assert classify_ode(peak=3.0, end=0.0) == "extinction"
    assert classify_ode(peak=500.0, end=0.01) == "cleared"
    assert classify_ode(peak=500.0, end=40.0) == "chronic"
