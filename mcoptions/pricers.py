"""Monte Carlo option pricers, each returning a price with its standard error.

A Monte Carlo price is the sample mean of discounted payoffs, so by the Central
Limit Theorem its error shrinks like sigma_payoff / sqrt(N). Variance reduction
lowers sigma_payoff instead of raising N.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .black_scholes import geometric_asian_call
from .simulate import price_paths, standard_normals, terminal_prices


@dataclass
class MCResult:
    price: float
    std_error: float
    n_paths: int

    @property
    def ci95(self) -> tuple[float, float]:
        """95% confidence interval from the normal approximation."""
        return self.price - 1.96 * self.std_error, self.price + 1.96 * self.std_error


def _mean_and_se(samples: np.ndarray, n_paths: int) -> MCResult:
    """Mean and standard error of i.i.d. samples."""
    m = len(samples)
    return MCResult(float(samples.mean()), float(samples.std(ddof=1) / np.sqrt(m)), n_paths)


def _payoff(S: np.ndarray, K: float, kind: str) -> np.ndarray:
    if kind == "call":
        return np.maximum(S - K, 0.0)
    if kind == "put":
        return np.maximum(K - S, 0.0)
    raise ValueError("kind must be 'call' or 'put'")


def european_mc(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int,
    kind: str = "call",
    method: str = "plain",
    seed: int | None = None,
) -> MCResult:
    """Price a European option by Monte Carlo.

    method:
      "plain"      - ordinary Monte Carlo.
      "antithetic" - pairs Z with -Z. The two payoffs in a pair are NOT
                     independent, so the standard error is computed from the
                     pair averages, which are.
      "control"    - control variate using the discounted stock price
                     X = e^{-rT} S_T, whose mean is known exactly (S0).
                     Estimator: Y - b (X - S0), with b = Cov(Y, X) / Var(X)
                     chosen to minimise variance.
    """
    rng = np.random.default_rng(seed)
    disc = np.exp(-r * T)

    if method == "plain":
        Z = standard_normals(rng, n_paths)
        Y = disc * _payoff(terminal_prices(S0, T, r, sigma, Z), K, kind)
        return _mean_and_se(Y, n_paths)

    if method == "antithetic":
        Z = standard_normals(rng, n_paths, antithetic=True)
        Y = disc * _payoff(terminal_prices(S0, T, r, sigma, Z), K, kind)
        half = n_paths // 2
        pair_means = 0.5 * (Y[:half] + Y[half:])
        return _mean_and_se(pair_means, n_paths)

    if method == "control":
        Z = standard_normals(rng, n_paths)
        S_T = terminal_prices(S0, T, r, sigma, Z)
        Y = disc * _payoff(S_T, K, kind)
        X = disc * S_T
        b = np.cov(Y, X, ddof=1)[0, 1] / X.var(ddof=1)
        adjusted = Y - b * (X - S0)
        return _mean_and_se(adjusted, n_paths)

    raise ValueError("method must be 'plain', 'antithetic' or 'control'")


def european_call_delta_mc(
    S0: float, K: float, T: float, r: float, sigma: float, n_paths: int, seed: int | None = None
) -> MCResult:
    """Delta of a European call with the pathwise estimator.

    Differentiating the discounted payoff e^{-rT} max(S_T - K, 0) with respect
    to S0, and using dS_T/dS0 = S_T / S0, gives e^{-rT} 1{S_T > K} S_T / S0.
    Its average is an unbiased estimate of delta.
    """
    rng = np.random.default_rng(seed)
    S_T = terminal_prices(S0, T, r, sigma, standard_normals(rng, n_paths))
    samples = np.exp(-r * T) * (S_T > K) * S_T / S0
    return _mean_and_se(samples, n_paths)


def asian_call_mc(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
    n_paths: int,
    control_variate: bool = True,
    seed: int | None = None,
) -> MCResult:
    """Arithmetic-average Asian call, payoff max(mean(S_t1..S_tn) - K, 0).

    There is no closed form for this option. With control_variate=True, the
    geometric-average Asian call (which has an exact price) is used as the
    control. The two averages are almost perfectly correlated, so this removes
    most of the noise.
    """
    rng = np.random.default_rng(seed)
    paths = price_paths(S0, T, r, sigma, standard_normals(rng, (n_paths, n_steps)))
    disc = np.exp(-r * T)

    Y = disc * np.maximum(paths.mean(axis=1) - K, 0.0)
    if not control_variate:
        return _mean_and_se(Y, n_paths)

    geometric_mean = np.exp(np.log(paths).mean(axis=1))
    X = disc * np.maximum(geometric_mean - K, 0.0)
    X_exact = geometric_asian_call(S0, K, T, r, sigma, n_steps)
    b = np.cov(Y, X, ddof=1)[0, 1] / X.var(ddof=1)
    return _mean_and_se(Y - b * (X - X_exact), n_paths)
