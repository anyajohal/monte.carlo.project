"""Checks that each estimator agrees with known answers within its own error bars."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcoptions import (  # noqa: E402
    asian_call_mc,
    bs_delta,
    bs_price,
    european_call_delta_mc,
    european_mc,
    geometric_asian_call,
)

PARAMS = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)


def test_black_scholes_reference_value():
    # Standard textbook value for these parameters.
    assert bs_price(**PARAMS, kind="call") == pytest.approx(10.4506, abs=1e-4)


def test_put_call_parity():
    S0, K, T, r = PARAMS["S0"], PARAMS["K"], PARAMS["T"], PARAMS["r"]
    lhs = bs_price(**PARAMS, kind="call") - bs_price(**PARAMS, kind="put")
    assert lhs == pytest.approx(S0 - K * math.exp(-r * T), abs=1e-10)


@pytest.mark.parametrize("method", ["plain", "antithetic", "control"])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_european_mc_matches_black_scholes(method, kind):
    res = european_mc(**PARAMS, n_paths=200_000, kind=kind, method=method, seed=1)
    exact = bs_price(**PARAMS, kind=kind)
    # Within 4 standard errors: fails by chance less than 1 time in 10,000.
    assert abs(res.price - exact) < 4 * res.std_error


def test_variance_reduction_lowers_standard_error():
    plain = european_mc(**PARAMS, n_paths=100_000, method="plain", seed=2)
    anti = european_mc(**PARAMS, n_paths=100_000, method="antithetic", seed=2)
    ctrl = european_mc(**PARAMS, n_paths=100_000, method="control", seed=2)
    assert anti.std_error < plain.std_error
    assert ctrl.std_error < plain.std_error


def test_pathwise_delta_matches_black_scholes():
    res = european_call_delta_mc(**PARAMS, n_paths=200_000, seed=3)
    assert abs(res.price - bs_delta(**PARAMS)) < 4 * res.std_error


def test_geometric_asian_closed_form_matches_simulation():
    # Simulate the geometric Asian directly and compare with the formula.
    from mcoptions.simulate import price_paths

    rng = np.random.default_rng(4)
    n_steps = 12
    paths = price_paths(PARAMS["S0"], PARAMS["T"], PARAMS["r"], PARAMS["sigma"],
                        rng.standard_normal((200_000, n_steps)))
    G = np.exp(np.log(paths).mean(axis=1))
    payoffs = math.exp(-PARAMS["r"] * PARAMS["T"]) * np.maximum(G - PARAMS["K"], 0.0)
    se = payoffs.std(ddof=1) / math.sqrt(len(payoffs))
    exact = geometric_asian_call(**PARAMS, n_steps=n_steps)
    assert abs(payoffs.mean() - exact) < 4 * se


def test_asian_control_variate_agrees_and_is_more_precise():
    plain = asian_call_mc(**PARAMS, n_steps=12, n_paths=100_000, control_variate=False, seed=5)
    ctrl = asian_call_mc(**PARAMS, n_steps=12, n_paths=100_000, control_variate=True, seed=6)
    combined_se = math.hypot(plain.std_error, ctrl.std_error)
    assert abs(plain.price - ctrl.price) < 4 * combined_se
    assert ctrl.std_error < plain.std_error / 5
