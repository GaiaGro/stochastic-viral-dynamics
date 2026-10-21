import csv

import numpy as np
import pytest

from exp_resistance import BASE, dominance_time, one, summarise, theory, write
from viraldyn import suppression_efficacy, r0_wild_type


def test_dominance_time():
    t = np.arange(5.0)
    assert dominance_time(t, np.array([10, 5, 2, 1, 0]), np.array([0, 0, 3, 4, 5])) == 2.0
    assert np.isnan(dominance_time(t, np.array([10, 5, 2, 1, 0]), np.array([0, 0, 0, 0, 0])))
    assert np.isnan(dominance_time(t, np.zeros(5), np.zeros(5)))


def test_theory_terms_are_probabilities_and_ordered():
    p = BASE.replace(eps_b=0.9)
    state = np.array([800.0, 20.0, 2.0, 500.0, 30.0, 100.0])
    th = theory(p, state, 30.0)
    for v in th.values():
        assert 0.0 <= v <= 1.0
    assert th["p_denovo_cond"] <= th["p_denovo"] and th["p_theory_cond"] <= th["p_theory"]
    assert th["p_theory"] >= th["p_pre"] and th["p_theory"] >= th["p_denovo"]
    # conditioning on an early wild-type death removes the de novo term
    early = theory(p, state, 30.0, t_wt_end=30.05)
    assert early["p_denovo_cond"] < 0.1 * early["p_denovo"] and early["p_pre"] == pytest.approx(th["p_pre"])
    # no mutant present: only the de novo term remains
    state[2] = state[4] = 0.0
    th0 = theory(p, state, 30.0)
    assert th0["p_pre"] == 0.0 and th0["p_theory"] == pytest.approx(th0["p_denovo"])


def test_theory_de_novo_vanishes_without_mutation():
    p = BASE.replace(eps_b=0.9, mu=0.0)
    assert theory(p, np.array([1000.0, 20.0, 0.0, 500.0, 0.0, 10.0]), 30.0)["p_denovo"] == 0.0


def test_suppression_efficacy_is_exact():
    eps = suppression_efficacy(BASE)
    assert 0.79 < eps < 0.81
    assert r0_wild_type(BASE.replace(eps_b=eps)) == pytest.approx(1.0, rel=1e-9)
    assert r0_wild_type(BASE.replace(eps_b=0.8)) < 1.0 < r0_wild_type(BASE.replace(eps_b=0.79))


def test_two_phase_run_is_reproducible():
    first = one(42, 30, 0.9)
    assert first is not None, "the pre-treatment infection died; the resistance preset should be viable"
    a, _ = first
    b, _ = one(42, 30, 0.9)
    clean = lambda d: {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}
    assert clean(a) == clean(b)
    assert set(a) >= {"pre", "emerged", "mut_h", "wt_h", "dominant_h", "t_dom", "t_wt_ext", "p_theory", "p_theory_cond"}


def test_non_viable_condition_is_written_with_nan_columns(tmp_path):
    """A condition where every realisation dies before treatment gives a short row; the
    writer must still produce the full header and fill the missing values with nan."""
    rec, _ = one(42, 30, 0.9)
    full = summarise(30, 0.9, [rec], 0, 42)
    short = summarise(30, 0.9, [], 1000, 0)
    assert short["n"] == 0 and set(short) < set(full)
    write(tmp_path / "r.csv", [short, full])  # the short row first: the header must come from the fullest row
    rows = list(csv.DictReader(open(tmp_path / "r.csv")))
    assert list(rows[0]) == list(full) and rows[0]["n"] == "0" and rows[0]["p_emerge"] == "nan"
    assert rows[1]["n"] == "1" and rows[1]["p_emerge"] == str(full["p_emerge"])
