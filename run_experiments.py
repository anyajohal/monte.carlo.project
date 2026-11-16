"""Runs every experiment in the README and writes results/results.md and the plots.

Usage:  python run_experiments.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mcoptions import (  # noqa: E402
    asian_call_mc,
    bs_delta,
    bs_price,
    european_call_delta_mc,
    european_mc,
)

PARAMS = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
OUT = Path(__file__).parent / "results"
METHODS = ["plain", "antithetic", "control"]


def accuracy_table(n_paths: int = 100_000) -> tuple[str, dict]:
    """Price with each method at the same N and compare with Black-Scholes."""
    exact = bs_price(**PARAMS)
    rows, se = [], {}
    for m in METHODS:
        res = european_mc(**PARAMS, n_paths=n_paths, method=m, seed=42)
        se[m] = res.std_error
        lo, hi = res.ci95
        rows.append(
            f"| {m} | {res.price:.4f} | {res.std_error:.4f} | [{lo:.4f}, {hi:.4f}] "
            f"| {abs(res.price - exact):.4f} | {(se['plain'] / res.std_error) ** 2:.1f}x |"
        )
    header = (
        f"Black-Scholes price: **{exact:.4f}**. Each row uses {n_paths:,} paths.\n\n"
        "| Method | MC price | Std error | 95% CI | Abs error | Variance reduction |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
    )
    return header + "\n".join(rows), se


def convergence(n_list=(100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000), reps=40):
    """Root-mean-square error against Black-Scholes, averaged over many seeds."""
    exact = bs_price(**PARAMS)
    rmse = {m: [] for m in METHODS}
    for m in METHODS:
        for n in n_list:
            errs = [european_mc(**PARAMS, n_paths=n, method=m, seed=1000 + s).price - exact for s in range(reps)]
            rmse[m].append(float(np.sqrt(np.mean(np.square(errs)))))
    slopes = {m: float(np.polyfit(np.log10(n_list), np.log10(rmse[m]), 1)[0]) for m in METHODS}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in METHODS:
        ax.loglog(n_list, rmse[m], marker="o", label=f"{m} (slope {slopes[m]:.2f})")
    ref = rmse["plain"][0] * np.sqrt(n_list[0] / np.array(n_list))
    ax.loglog(n_list, ref, "k--", linewidth=1, label=r"$1/\sqrt{N}$ reference")
    ax.set_xlabel("Number of simulated paths N")
    ax.set_ylabel(f"RMS pricing error over {reps} runs")
    ax.set_title("European call: Monte Carlo error vs number of paths")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "convergence.png", dpi=150)
    plt.close(fig)
    return slopes


def ci_coverage(n_paths: int = 10_000, reps: int = 1_000) -> float:
    """How often the 95% confidence interval actually contains the true price."""
    exact = bs_price(**PARAMS)
    hits = 0
    for s in range(reps):
        lo, hi = european_mc(**PARAMS, n_paths=n_paths, method="plain", seed=5000 + s).ci95
        hits += lo <= exact <= hi
    return hits / reps


def estimate_spread(n_paths: int = 10_000, reps: int = 1_000) -> None:
    """Histogram of repeated estimates: plain vs control variate."""
    exact = bs_price(**PARAMS)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in ["plain", "control"]:
        est = [european_mc(**PARAMS, n_paths=n_paths, method=m, seed=9000 + s).price for s in range(reps)]
        ax.hist(est, bins=40, alpha=0.6, label=f"{m} (std {np.std(est):.3f})")
    ax.axvline(exact, color="k", linestyle="--", label=f"Black-Scholes {exact:.4f}")
    ax.set_xlabel(f"Estimated call price ({n_paths:,} paths per estimate)")
    ax.set_ylabel(f"Count out of {reps} runs")
    ax.set_title("Spread of repeated estimates")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "estimate_spread.png", dpi=150)
    plt.close(fig)


def delta_check(n_paths: int = 200_000) -> str:
    res = european_call_delta_mc(**PARAMS, n_paths=n_paths, seed=7)
    exact = bs_delta(**PARAMS)
    return (
        f"Pathwise delta ({n_paths:,} paths): **{res.price:.4f}** +/- {1.96 * res.std_error:.4f} (95%), "
        f"Black-Scholes delta: **{exact:.4f}**."
    )


def asian_table(n_steps: int = 52, n_paths: int = 100_000) -> tuple[str, float]:
    plain = asian_call_mc(**PARAMS, n_steps=n_steps, n_paths=n_paths, control_variate=False, seed=11)
    ctrl = asian_call_mc(**PARAMS, n_steps=n_steps, n_paths=n_paths, control_variate=True, seed=11)
    vr = (plain.std_error / ctrl.std_error) ** 2
    table = (
        f"Arithmetic Asian call, weekly averaging ({n_steps} dates), {n_paths:,} paths:\n\n"
        "| Method | MC price | Std error | Variance reduction |\n| --- | --- | --- | --- |\n"
        f"| plain | {plain.price:.4f} | {plain.std_error:.4f} | 1.0x |\n"
        f"| geometric control variate | {ctrl.price:.4f} | {ctrl.std_error:.5f} | {vr:,.0f}x |"
    )
    return table, vr


def main() -> None:
    OUT.mkdir(exist_ok=True)
    print("Accuracy table...")
    acc, _ = accuracy_table()
    print("Convergence study...")
    slopes = convergence()
    print("Confidence interval coverage...")
    coverage = ci_coverage()
    print("Spread histogram...")
    estimate_spread()
    print("Delta...")
    delta = delta_check()
    print("Asian option...")
    asian, _ = asian_table()

    p = PARAMS
    report = f"""# Results

Parameters: S0 = {p['S0']}, K = {p['K']}, T = {p['T']} year, r = {p['r']}, sigma = {p['sigma']}.
Generated by `python run_experiments.py`.

## 1. European call: Monte Carlo vs Black-Scholes

{acc}

Variance reduction = (plain std error / method std error)^2, i.e. how many times
more plain paths you would need for the same precision.

## 2. Convergence

Fitted log-log slopes of RMS error against N: {", ".join(f"{m} {s:.3f}" for m, s in slopes.items())}.
Theory says -0.5 (error proportional to 1/sqrt(N)).

![convergence](convergence.png)

## 3. Do the confidence intervals mean what they say?

Over 1,000 independent runs of 10,000 paths, the 95% confidence interval
contained the true price **{coverage:.1%}** of the time.

![spread](estimate_spread.png)

## 4. Delta

{delta}

## 5. Asian option (no closed form)

{asian}
"""
    (OUT / "results.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
