import numpy as np
import pytest
from scipy.stats import ks_2samp

from viraldyn.model import PRESETS, Params, solve_ode
from viraldyn.ssa import STOP_EXTINCT, STOP_NO_REACTIONS, STOP_T_END, simulate

from reference_ssa import gillespie_reference

ZERO = Params.make(lam=0, dT=0, bw=0, bm=0, pw=0, pm=0, mu=0, cw=0, cm=0, delta=0, k=0, q=0, dE=0)


def test_zero_rates_keep_state_constant():
    x0 = [5, 4, 3, 2, 1, 6]
    r = simulate(ZERO, x0, np.linspace(0, 10, 11), seed_value=1)
    assert r.stop_reason == STOP_NO_REACTIONS and r.n_events == 0
    assert (r.X == x0).all() and (r.C == 0).all() and (r.x_end == x0).all()


def test_seed_reproduces_and_changes():
    p = PRESETS["establishment"]
    x0 = [1000, 0, 0, 1, 0, 1]
    t = np.linspace(0, 5, 51)
    a = simulate(p, x0, t, seed_value=7)
    b = simulate(p, x0, t, seed_value=7)
    c = simulate(p, x0, t, seed_value=8)
    assert (a.X == b.X).all() and a.n_events == b.n_events
    assert not (a.X == c.X).all()


def test_immigration_death_is_poisson():
    """With only target production and death, T is an M/M/inf queue: Poisson(lam/dT)."""
    p = ZERO.replace(lam=50.0, dT=1.0)
    finals = np.array([simulate(p, [50, 0, 0, 0, 0, 0], [0.0, 10.0], seed_value=s).x_end[0] for s in range(400)])
    assert finals.mean() == pytest.approx(50, abs=1.2)  # 3.4 standard errors
    assert 36 < finals.var() < 66  # about 4 standard errors of the sample variance


def test_grid_recording_is_piecewise_constant_state():
    """The recorded state at a grid time equals the state after all events before it."""
    p = ZERO.replace(lam=1000.0)  # pure immigration, so T(t) == x0 + number of events before t
    r = simulate(p, [0, 0, 0, 0, 0, 0], np.linspace(0, 2, 21), seed_value=3)
    assert r.stop_reason == STOP_T_END
    assert (np.diff(r.X[:, 0]) >= 0).all() and (r.X[:, 0] == r.C[:, 5]).all()
    assert r.X[0, 0] == 0 and r.x_end[0] >= r.X[-1, 0]


def test_stop_on_extinction_and_lineage_times():
    p = PRESETS["acute"]
    r = simulate(p, [1000, 0, 0, 50, 0, 2], np.linspace(0, 60, 601), seed_value=11, stop_on_extinction=True)
    assert r.stop_reason == STOP_EXTINCT and r.extinct
    assert r.t_ext[0] == pytest.approx(r.t_end)  # wild type died at the stopping time
    assert np.isnan(r.t_ext[1])  # no mutant ever existed in this short run, or it died earlier
    assert (r.X[r.t_grid > r.t_end] == -1).all() and (r.X[r.t_grid <= r.t_end] >= 0).all()


def test_stop_on_cumulative_infections():
    r = simulate(PRESETS["chronic"], [1000, 5, 0, 30, 0, 10], np.linspace(0, 60, 61), seed_value=2, stop_cum_infections=100)
    assert r.stopped == "cum_infections" and r.C.max() >= 0
    assert r.n_events >= 100 and r.t_end < 60


def test_mean_tracks_ode_at_large_populations():
    """Chronic regime, first three days: populations are in the thousands and the
    ensemble should follow the deterministic solution closely."""
    p = PRESETS["chronic"]
    x0 = [1000, 5, 0, 30, 0, 10]
    t = np.array([0.0, 1.0, 2.0, 3.0])
    ode = solve_ode(p, x0, t)
    runs = np.array([simulate(p, x0, t, seed_value=100 + s).X for s in range(40)], dtype=float)
    T_rel = np.abs(runs[:, 1:, 0].mean(0) / ode[1:, 0] - 1)
    assert T_rel.max() < 0.02
    logV = np.log10(np.median(runs[:, 1:, 3], axis=0)) - np.log10(ode[1:, 3])
    assert np.abs(logV).max() < 0.2


def test_matches_reference_implementation():
    """Extinction-time distribution agrees with the notebook's pure-Python SSA."""
    p = PRESETS["establishment"]
    x0 = [1000, 0, 0, 1, 0, 1]
    n = 150
    fast = np.array([simulate(p, x0, [0.0, 30.0], seed_value=1000 + s, stop_on_extinction=True).t_end for s in range(n)])
    rng = np.random.default_rng(2024)
    ref = np.array([gillespie_reference(p, 30.0, x0, rng)[0] for _ in range(n)])
    assert ks_2samp(fast, ref).pvalue > 0.01
