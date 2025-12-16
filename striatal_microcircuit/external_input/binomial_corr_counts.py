"""
binomial_corr_counts.py

Compare direct Beta-Binomial count generation (draw K ~ Binomial(N, p_bin))
versus simulating individual Bernoulli spikes per neuron and summing.
Also compute empirical pairwise correlations from the individual spike streams.

Usage:
    python binomial_corr_counts.py
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from typing import Tuple

# --------- Simulation utilities ----------


def beta_params_from_p_rho(p: float, rho: float) -> Tuple[float, float]:
    """Return (alpha, beta) for Beta so that E[P]=p and Var[P]=rho*p*(1-p)."""
    if rho <= 0.0:
        raise ValueError(
            "rho must be > 0 to compute Beta parameters (handle rho==0 separately)."
        )
    if rho >= 1.0:
        raise ValueError("rho must be < 1 for Beta params (handle rho==1 separately).")
    s = 1.0 / rho - 1.0
    alpha = p * s
    beta = (1.0 - p) * s
    return alpha, beta


def simulate_counts_direct(
    N: int, p: float, rho: float, num_bins: int, rng: np.random.Generator
) -> np.ndarray:
    """Direct Beta-Binomial: for each bin draw p_bin ~ Beta(alpha,beta) and then K ~ Binomial(N, p_bin)."""
    if rho <= 0.0:
        return rng.binomial(N, p, size=num_bins)
    if rho >= 1.0:
        all_fire = rng.binomial(1, p, size=num_bins)
        return N * all_fire
    alpha, beta = beta_params_from_p_rho(p, rho)
    p_bins = rng.beta(alpha, beta, size=num_bins)
    counts = rng.binomial(N, p_bins)
    return counts


def simulate_individual_spikes(
    N: int,
    p: float,
    rho: float,
    num_bins: int,
    rng: np.random.Generator,
    max_full_entries: int = 50_000_000,
):
    """
    Simulate per-neuron binary spikes.
    If num_bins * N is too large (> max_full_entries), only simulate a random subset of neurons
    for correlation estimation but still generate the full counts by drawing Binomial(N,p_bin).
    Returns a tuple (counts_from_individuals, spikes_subset, subset_indices, full_simulated_flag, p_bins)
      - counts_from_individuals: array shape (num_bins,)
      - spikes_subset: None or array shape (num_bins, m) (m = number of subset neurons)
      - subset_indices: None or array of neuron indices represented in spikes_subset
      - full_simulated_flag: True if spikes for all N simulated, False if only subset simulated
      - p_bins: latent p per bin (size num_bins)
    """
    if rho <= 0.0:
        # independent Bernoulli: easy
        # but check memory
        total_entries = num_bins * N
        if total_entries <= max_full_entries:
            spikes = rng.binomial(1, p, size=(num_bins, N)).astype(np.uint8)
            counts = spikes.sum(axis=1)
            return counts, spikes, np.arange(N), True, None
        else:
            # too big -> sample subset for correlations, but counts from Binomial
            alpha_beta = None
            counts = rng.binomial(N, p, size=num_bins)
            m = min(200, N)
            subset_idx = np.sort(rng.choice(N, size=m, replace=False))
            spikes_subset = rng.binomial(1, p, size=(num_bins, m)).astype(np.uint8)
            return counts, spikes_subset, subset_idx, False, None

    if rho >= 1.0:
        # perfect correlation: all neurons equal Bernoulli(p)
        all_fire = rng.binomial(1, p, size=num_bins)
        counts = N * all_fire
        spikes = (all_fire[:, None].repeat(N, axis=1)).astype(np.uint8)
        # memory check similar as above
        if num_bins * N <= max_full_entries:
            return counts, spikes, np.arange(N), True, all_fire
        else:
            # return subset
            m = min(200, N)
            subset_idx = np.sort(rng.choice(N, size=m, replace=False))
            spikes_subset = (all_fire[:, None].repeat(m, axis=1)).astype(np.uint8)
            return counts, spikes_subset, subset_idx, False, all_fire

    # 0 < rho < 1:
    alpha, beta = beta_params_from_p_rho(p, rho)
    p_bins = rng.beta(alpha, beta, size=num_bins)

    total_entries = num_bins * N
    if total_entries <= max_full_entries:
        # simulate full spikes matrix
        spikes = rng.binomial(1, p_bins[:, None], size=(num_bins, N)).astype(np.uint8)
        counts = spikes.sum(axis=1)
        return counts, spikes, np.arange(N), True, p_bins
    else:
        # do not allocate full matrix; generate counts directly (Binomial(N, p_bin))
        counts = rng.binomial(N, p_bins)
        # simulate small subset of neurons for correlation estimation
        m = min(200, N)
        subset_idx = np.sort(rng.choice(N, size=m, replace=False))
        spikes_subset = rng.binomial(1, p_bins[:, None], size=(num_bins, m)).astype(
            np.uint8
        )
        return counts, spikes_subset, subset_idx, False, p_bins


# ---------- Analysis utilities ----------


def empirical_pairwise_corr_stats(spikes: np.ndarray) -> Tuple[float, float, int]:
    """
    Compute mean and std of pairwise Pearson correlations for spike matrix shape (num_bins, num_neurons).
    Returns (mean_corr, std_corr, n_pairs).
    Replaces NaN correlations (due to zero variance neurons) by 0.
    """
    if spikes is None:
        return float("nan"), float("nan"), 0
    X = spikes.astype(float)
    # if only one neuron, no pairs
    num_neurons = X.shape[1]
    if num_neurons < 2:
        return float("nan"), float("nan"), 0

    # compute correlation matrix across columns
    with np.errstate(invalid="ignore"):
        corrmat = np.corrcoef(X.T)
    iu = np.triu_indices(num_neurons, k=1)
    pair_corrs = corrmat[iu]
    # replace nan (due to constant columns) with 0
    pair_corrs = np.nan_to_num(pair_corrs, nan=0.0)
    return float(pair_corrs.mean()), float(pair_corrs.std()), int(pair_corrs.size)


def empirical_rate(spikes: np.ndarray) -> float:
    """Mean firing probability across all simulated spikes (spikes shape num_bins x neurons)."""
    if spikes is None:
        return float("nan")
    return float(spikes.mean())


def empirical_pmf(counts: np.ndarray):
    kmin = int(counts.min())
    kmax = int(counts.max())
    ks = np.arange(kmin, kmax + 1)
    pmf = np.bincount(counts - kmin, minlength=len(ks)) / counts.size
    return ks, pmf


def compare_counts(counts_a: np.ndarray, counts_b: np.ndarray):
    """Return comparison dict (pmf distances, mean/var differences, plus pmfs)."""
    assert counts_a.shape == counts_b.shape
    n = counts_a.size
    mean_a = float(counts_a.mean())
    mean_b = float(counts_b.mean())
    var_a = float(counts_a.var(ddof=1))
    var_b = float(counts_b.var(ddof=1))
    ks_a, pa = empirical_pmf(counts_a)
    ks_b, pb = empirical_pmf(counts_b)
    kmin = min(ks_a[0], ks_b[0])
    kmax = max(ks_a[-1], ks_b[-1])
    ks = np.arange(kmin, kmax + 1)
    pa_full = np.zeros_like(ks, dtype=float)
    pb_full = np.zeros_like(ks, dtype=float)
    pa_full[ks_a - kmin] = pa
    pb_full[ks_b - kmin] = pb
    tv = 0.5 * np.abs(pa_full - pb_full).sum()
    l2 = math.sqrt(((pa_full - pb_full) ** 2).sum())
    maxdiff = float(np.abs(pa_full - pb_full).max())
    argmax = int(ks[np.argmax(np.abs(pa_full - pb_full))])
    eps = 1e-12
    chisq = float(((pa_full - pb_full) ** 2 / (pb_full + eps)).sum())
    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_diff": mean_a - mean_b,
        "var_a": var_a,
        "var_b": var_b,
        "var_diff": var_a - var_b,
        "tv_distance": float(tv),
        "l2_distance": float(l2),
        "max_pmf_diff": maxdiff,
        "argmax_k": argmax,
        "ks": ks,
        "pa": pa_full,
        "pb": pb_full,
    }


# ---------- Plotting helpers ----------


def plot_pmf(ks, pa, pb, title, filename):
    plt.figure(figsize=(8, 4))
    plt.plot(ks, pa, label="direct pmf", linewidth=1.5)
    plt.plot(ks, pb, label="from individuals pmf", linestyle=":", linewidth=1.2)
    plt.xlabel("Spike count K per bin")
    plt.ylabel("Empirical probability")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_corr_hist(pair_corrs, title, filename, bins=50):
    plt.figure(figsize=(6, 4))
    plt.hist(pair_corrs, bins=bins, density=False)
    plt.xlabel("Pairwise Pearson correlation")
    plt.ylabel("Count of pairs")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


# ---------- Main experiment loop ----------


def main():
    rng_master = np.random.default_rng(20250401)

    experiments = [
        {"N": 100, "p": 0.01, "rho": 0.05, "bins": 10000},
        {"N": 100, "p": 0.05, "rho": 0.10, "bins": 10000},
        {"N": 100, "p": 0.10, "rho": 0.50, "bins": 10000},
        {"N": 100, "p": 0.50, "rho": 0.90, "bins": 10000},
        # Large N example: will limit full-individual simulation if memory heavy
        # {"N": 10000, "p": 0.01, "rho": 0.01, "bins": 2000},
    ]

    for i, exp in enumerate(experiments):
        N = int(exp["N"])
        p = float(exp["p"])
        rho = float(exp["rho"])
        num_bins = int(exp["bins"])
        print("=" * 72)
        print(
            f"Experiment {i+1}/{len(experiments)}: N={N}, p={p}, rho={rho}, bins={num_bins}"
        )

        # RNGs
        seed_a = int(rng_master.integers(1, 2**30))
        seed_b = int(rng_master.integers(1, 2**30))
        rng_a = np.random.default_rng(seed_a)
        rng_b = np.random.default_rng(seed_b)

        # Direct counts (no per-neuron spikes)
        counts_direct = simulate_counts_direct(N, p, rho, num_bins, rng_a)

        # Individual spikes simulation (may simulate only subset for correlation)
        counts_ind, spikes_subset, subset_idx, full_flag, p_bins = (
            simulate_individual_spikes(
                N, p, rho, num_bins, rng_b, max_full_entries=50_000_000
            )
        )

        # If full_flag is True, we have full spikes in spikes_subset (and subset_idx == np.arange(N))
        # If full_flag is False, spikes_subset contains data for a subset of neurons

        # Compute pairwise correlations and rates from the individual spike streams (or subset)
        mean_corr, std_corr, n_pairs = empirical_pairwise_corr_stats(spikes_subset)
        sample_rate = empirical_rate(spikes_subset)

        # If we have the full simulation, recompute sample_rate across all neurons instead of subset:
        if full_flag:
            # spikes_subset is full matrix
            sample_rate = empirical_rate(spikes_subset)

        # Print results: (1) pairwise stats and rate from individuals, (2) compare counts
        print("Individual spikes (from simulation):")
        if full_flag:
            print(f"  simulated full spike matrix for all N neurons")
        else:
            print(
                f"  simulated spike matrix for subset of {spikes_subset.shape[1]} neurons (subset indices available)"
            )
        print(
            f"  empirical marginal rate (from simulated spikes / subset) = {sample_rate:.6f} (target p = {p:.6f})"
        )
        if n_pairs > 0:
            print(
                f"  empirical mean pairwise corr = {mean_corr:.6f}, std = {std_corr:.6f}, n_pairs = {n_pairs}"
            )
        else:
            print("  not enough neurons to compute pairwise correlations.")
        print("")

        # Compare counts (direct vs from individuals)
        cmp = compare_counts(counts_direct, counts_ind)
        print("Counts comparison (direct Beta-Binomial vs sum-of-individuals):")
        print(
            f"  mean direct = {cmp['mean_a']:.6f}, mean individuals = {cmp['mean_b']:.6f}, diff = {cmp['mean_diff']:.6f}"
        )
        print(
            f"  var  direct = {cmp['var_a']:.6f}, var  individuals = {cmp['var_b']:.6f}, diff = {cmp['var_diff']:.6f}"
        )
        print(
            f"  PMF distances: TV = {cmp['tv_distance']:.6f}, L2 = {cmp['l2_distance']:.6f}, max-pmf-diff = {cmp['max_pmf_diff']:.6f} (at k={cmp['argmax_k']})"
        )
        print("")

        # Save PMF plot
        pmf_title = f"PMF comparison (N={N}, p={p}, rho={rho})"
        pmf_fname = f"pmf_compare_exp{i+1}.png"
        plot_pmf(cmp["ks"], cmp["pa"], cmp["pb"], pmf_title, pmf_fname)
        print(f"  PMF plot saved to '{pmf_fname}'")

        # If we have pairwise correlations (from subset or full), save histogram
        if spikes_subset is not None:
            corrmat = np.corrcoef(spikes_subset.astype(float).T)
            iu = np.triu_indices(corrmat.shape[0], k=1)
            pair_corrs = corrmat[iu]
            # replace nan with 0 (due to zero-variance neurons)
            pair_corrs = np.nan_to_num(pair_corrs, nan=0.0)
            hist_title = f"Pairwise corr (N={N}, p={p}, rho={rho})"
            hist_fname = f"corr_hist_exp{i+1}.png"
            plot_corr_hist(pair_corrs, hist_title, hist_fname, bins=50)
            print(
                f"  Pairwise-corr histogram saved to '{hist_fname}' (computed on simulated neurons/subset)"
            )
        print("")

    print("All experiments done.")


if __name__ == "__main__":
    main()
