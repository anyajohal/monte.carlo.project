"""Closed-form prices used as benchmarks for the Monte Carlo estimates.

Everything here assumes the Black-Scholes model: the stock follows geometric
Brownian motion with constant risk-free rate r and volatility sigma.
"""
from __future__ import annotations

import math


def norm_cdf(x: float) -> float:
    """Standard normal CDF, written with math.erf so we don't need scipy."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _d1_d2(S0: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_price(S0: float, K: float, T: float, r: float, sigma: float, kind: str = "call") -> float:
    """Black-Scholes price of a European call or put."""
    d1, d2 = _d1_d2(S0, K, T, r, sigma)
    if kind == "call":
        return S0 * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    if kind == "put":
        return K * math.exp(-r * T) * norm_cdf(-d2) - S0 * norm_cdf(-d1)
    raise ValueError("kind must be 'call' or 'put'")


def bs_delta(S0: float, K: float, T: float, r: float, sigma: float, kind: str = "call") -> float:
    """Black-Scholes delta (dPrice/dS0) of a European call or put."""
    d1, _ = _d1_d2(S0, K, T, r, sigma)
    return norm_cdf(d1) if kind == "call" else norm_cdf(d1) - 1.0


def geometric_asian_call(S0: float, K: float, T: float, r: float, sigma: float, n_steps: int) -> float:
    """Exact price of a call on the GEOMETRIC average of n equally spaced prices.

    The geometric average of lognormal prices is itself lognormal, so this has a
    closed form. With monitoring dates t_i = i*T/n (i = 1..n):

        ln G ~ Normal(mu, v),
        mu = ln S0 + (r - sigma^2/2) * T * (n+1) / (2n)
        v  = sigma^2 * T * (n+1)(2n+1) / (6 n^2)

    It is used as a control variate for the arithmetic Asian option, which has
    no closed form.
    """
    n = n_steps
    mu = math.log(S0) + (r - 0.5 * sigma**2) * T * (n + 1) / (2 * n)
    v = sigma**2 * T * (n + 1) * (2 * n + 1) / (6 * n**2)
    d1 = (mu - math.log(K) + v) / math.sqrt(v)
    d2 = d1 - math.sqrt(v)
    return math.exp(-r * T) * (math.exp(mu + 0.5 * v) * norm_cdf(d1) - K * norm_cdf(d2))
