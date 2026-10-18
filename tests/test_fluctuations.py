import numpy as np
import pytest

from exp_fluctuations import coefficient_of_variation
from viraldyn import PRESETS, scale_system_size
from viraldyn.model import Params
from viraldyn.theory import fixed_point, jacobian, linear_noise_covariance

ZERO = Params.make(lam=0, dT=0, bw=0, bm=0, pw=0, pm=0, mu=0, cw=0, cm=0, delta=0, k=0, q=0, dE=0)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # the five inert species make J singular; scipy perturbs it
def test_linear_noise_recovers_poisson_for_immigration_death():
    p = ZERO.replace(lam=50.0, dT=1.0)
    C = linear_noise_covariance(p, [50, 0, 0, 0, 0, 0])
    assert C[0, 0] == pytest.approx(50.0, rel=1e-6)  # stationary variance equals the mean
    assert np.allclose(C[1:, :], 0.0) and np.allclose(C[:, 1:], 0.0)


def test_jacobian_matches_analytic_entries():
    p = PRESETS["resistance"]
    x = np.array([300.0, 50.0, 1.0, 1200.0, 10.0, 140.0])
    J = jacobian(p, x)
    assert J[0, 0] == pytest.approx(-p.dT - p.bw * x[3] - p.bm * x[4], rel=1e-6)  # dT'/dT
    assert J[5, 5] == pytest.approx(p.q * (x[1] + x[2]) - p.dE, rel=1e-6)  # dE'/dE
    assert J[1, 5] == pytest.approx(-p.k * x[1], rel=1e-6)  # dIw'/dE


def test_coefficients_of_variation_scale_as_inverse_root_of_system_size():
    p1 = PRESETS["resistance"]
    x1 = fixed_point(p1, [1000, 10, 0, 0, 0, 10])
    assert x1[1] == pytest.approx(p1.dE / p1.q, rel=1e-3)  # infected cells at I* = d_E / q
    p10 = scale_system_size(p1, 10)
    x10 = fixed_point(p10, [10000, 100, 0, 0, 0, 100])
    assert np.allclose(x10[[0, 1, 3, 5]], 10 * x1[[0, 1, 3, 5]], rtol=1e-3)
    cv1 = np.sqrt(np.diag(linear_noise_covariance(p1, x1)))[[0, 1, 3, 5]] / x1[[0, 1, 3, 5]]
    cv10 = np.sqrt(np.diag(linear_noise_covariance(p10, x10)))[[0, 1, 3, 5]] / x10[[0, 1, 3, 5]]
    assert np.allclose(cv10, cv1 / np.sqrt(10), rtol=2e-3)


def test_coefficient_of_variation_helper():
    cv, mean, std = coefficient_of_variation(np.array([[2.0, 4.0], [4.0, 2.0]]))
    assert mean == 3.0 and std == 1.0 and cv == pytest.approx(1 / 3)
