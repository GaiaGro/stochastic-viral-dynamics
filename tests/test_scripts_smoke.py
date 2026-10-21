"""One cheap call of each experiment script's per-realisation function, so that a
broken signature, index or record is caught in seconds rather than at the end of
a long run. The resistance and immune-activation scripts have their own tests."""

import numpy as np

import exp_baseline
import exp_establishment
import exp_fluctuations
import exp_immune_activation
import exp_immune_collapse
import exp_regime_map
from viraldyn.ssa import STOP_REASONS


def test_establishment_one():
    x0 = [1000, 0, 0, 1, 0, 0]
    assert exp_establishment.one(1, exp_establishment.BASE, x0) in range(len(STOP_REASONS))
    assert exp_establishment.one_cumulative(1, exp_establishment.BASE, x0) >= 0


def test_regime_map_one():
    p = exp_regime_map.params(1e-3, 0.1)
    established, persists = exp_regime_map.one(1, p)
    assert established in (True, False) and persists in (True, False)
    present_200, present_1000 = exp_regime_map.one_long(1, p)
    assert present_200 or not present_1000  # alive at day 1000 implies alive at day 200
    peak, end = exp_regime_map.deterministic(p)
    assert peak >= end > -1e-6
    assert exp_regime_map.classify_ode(peak, end) in ("extinction", "cleared", "chronic")


def test_baseline_one():
    X = exp_baseline.one(1)
    assert X.shape == (exp_baseline.GRID.size, 6) and (X >= 0).all()
    assert exp_baseline._alive(1, exp_baseline.P, exp_baseline.X0, 5.0) in (True, False)


def test_fluctuations_one():
    X, extinct = exp_fluctuations.one(1, *exp_fluctuations.scaled(1))
    assert X.shape == (exp_fluctuations.GRID.size, len(exp_fluctuations.SPECIES)) and not extinct


def test_immune_collapse_one_and_summary():
    o = exp_immune_collapse.one(1, exp_immune_collapse.BASE)
    assert exp_immune_collapse.fate_at_60(o) in exp_immune_activation.FATES
    s = exp_immune_collapse.summarise([o])
    assert s["n"] == 1 and {"median_t_effector_death", "median_peak_E", "n_extinct_by_day200"} <= set(s)
    assert exp_immune_collapse.summarise([]) == {"n": 0}
    assert np.isfinite(o[5][0])  # the load is recorded from day 0
