"""Simulating stock prices under geometric Brownian motion (GBM).

Under the risk-neutral measure, S_T = S0 * exp((r - sigma^2/2) T + sigma sqrt(T) Z)
with Z ~ Normal(0, 1). We use this exact solution, so there is no
discretisation error: the only error in the prices is Monte Carlo noise.
"""
from __future__ import annotations

import numpy as np


def standard_normals(rng: np.random.Generator, shape, antithetic: bool = False) -> np.ndarray:
    """Draw standard normals. With antithetic=True the second half is -1 x the first.

    Pairing Z with -Z makes the two payoffs negatively correlated for monotone
    payoffs, which lowers the variance of their average.
    """
    shape = tuple(np.atleast_1d(shape))
    n = shape[0]
    if not antithetic:
        return rng.standard_normal(shape)
    if n % 2:
        raise ValueError("antithetic sampling needs an even number of paths")
    half = rng.standard_normal((n // 2,) + shape[1:])
    return np.concatenate([half, -half], axis=0)


def terminal_prices(S0: float, T: float, r: float, sigma: float, Z: np.ndarray) -> np.ndarray:
    """Stock price at time T for each normal draw in Z."""
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)


def price_paths(S0: float, T: float, r: float, sigma: float, Z: np.ndarray) -> np.ndarray:
    """Full price paths. Z has shape (n_paths, n_steps); returns prices at t_1..t_n.

    Each step uses the exact GBM increment over dt = T / n_steps.
    """
    n_steps = Z.shape[1]
    dt = T / n_steps
    log_increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    return S0 * np.exp(np.cumsum(log_increments, axis=1))
