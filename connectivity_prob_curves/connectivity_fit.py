import numpy as np
from scipy.integrate import quad
from skopt import gp_minimize
from skopt.space import Real
import matplotlib.pyplot as plt
from scipy.stats import norm
import json
import os


class Dataset:
    def __init__(
        self,
        pairs_recorded,
        pairs_connected_uni,
        pairs_connected_bi=None,
        d_min=None,
        d_max=None,
        mean=None,
        std=None,
        label=None,
    ):
        """
        Initializes a Dataset instance.

        Args:
            pairs_recorded (int): Total number of recorded pairs.
            pairs_connected_uni (int): Number of unidirectionally connected pairs.
            pairs_connected_bi (Optional[int]): Number of bidirectionally connected pairs (if given assume bidirectional recordings).
            d_min (Optional[float]): Minimum distance for the uniform distribution.
            d_max (Optional[float]): Maximum distance for the uniform distribution.
            mean (Optional[float]): Mean of the normal distance distribution.
            std (Optional[float]): Standard deviation of the normal distance distribution.
            label (Optional[str]): Label for the dataset.
        """
        trials = pairs_recorded if pairs_connected_bi is None else pairs_recorded * 2
        successes = (
            pairs_connected_uni
            if pairs_connected_bi is None
            else pairs_connected_uni + 2 * pairs_connected_bi
        )
        self.x = successes
        self.n = trials
        self.d_min = 0 if d_min is None else d_min
        self.d_max = d_max
        self.mean = mean
        self.std = std
        self.label = label

    def avg_pred(self, sigma):
        """
        Computes the expected connectivity probability using a half-Gaussian model.

        For a specified sigma, the expectation (average) of the half-Gaussian function
        is calculated via an integral. The half-Gaussian function,
        p(d) = exp(-d^2/(sigma^2)), describes how connectivity probability decays
        with distance. It is weighted by the distance distribution P(d) and integrated.

        General form:
        E[p] = ∫[0,maxdistance] p(d) * P(d) dd

        Uniform interval [d_min, d_max]:
            E[p] = (1 / (d_max - d_min)) * ∫[d_min, d_max] p(d) dd

        Truncated normal distances (mean, std):
            Z = norm.cdf(∞) - norm.cdf(0)
            E[p] = ∫[0, mean+5*std] p(d) * norm.pdf(d, mean, std) / Z dd

        Args:
            sigma (float): Scale parameter of the half-Gaussian.

        Returns:
            float: Expected connectivity probability.
        """
        if self.d_max is not None:
            # Calculate expectation over a uniform distance interval [d_min, d_max]
            integrand = lambda d: np.exp(-(d**2) / (sigma**2))
            integral, _ = quad(integrand, self.d_min, self.d_max)
            return integral / (self.d_max - self.d_min)
        elif self.mean is not None and self.std is not None:
            # Calculate expectation over a truncated normal distribution for distances >= 0
            a, b = 0, np.inf
            Z = norm.cdf((b - self.mean) / self.std) - norm.cdf(
                (a - self.mean) / self.std
            )
            integrand = (
                lambda d: np.exp(-(d**2) / (sigma**2))
                * norm.pdf(d, loc=self.mean, scale=self.std)
                / Z
            )
            integral, _ = quad(integrand, 0, self.mean + 5 * self.std)
            return integral
        else:
            raise ValueError("Dataset must have either d_max or [mean,std]")


def neg_log_likelihood(params, datasets):
    """
    Computes the negative log likelihood for a list of datasets using a scaled half-Gaussian model.

    Args:
        params (list): A list containing [sigma, A] where sigma is the scale parameter and A is a scaling factor.
        datasets (list): A list of Dataset instances.

    Returns:
        float: The negative log likelihood value.
    """
    sigma, A = params
    if sigma <= 0 or not (0 <= A <= 1):
        return np.inf
    total = 0
    for ds in datasets:
        p_avg = ds.avg_pred(sigma)
        p = np.clip(A * p_avg, 1e-6, 1 - 1e-6)
        # weight by confidence from Wilson interval
        _, margin = wilson_interval(ds.x, ds.n)
        weight = 1.0 / margin
        total -= weight * (ds.x * np.log(p) + (ds.n - ds.x) * np.log(1 - p))
    return total


def wilson_interval(x, n, z=1.96):
    """
    Computes the Wilson score interval for a binomial proportion.

    Args:
        x (int): Number of successes.
        n (int): Total number of trials.
        z (float): Z-score corresponding to the desired confidence level (default is 1.96 for 95% CI).

    Returns:
        tuple: A tuple containing the center and margin of the confidence interval.
    """
    p = x / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center, margin


def plot_connectivity(datasets, sigma, A, group_label, output_dir):
    """
    Plots the observed and fitted connectivity probabilities for a list of datasets.

    Args:
        datasets (list): A list of Dataset instances.
        sigma (float): Fitted scale parameter of the half-Gaussian.
        A (float): Fitted scaling factor.
        group_label (str): Label for the group of datasets.
        output_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(8, 6))

    # Plot observed data points with error bars
    for ds in datasets:
        if ds.d_max is not None:
            x_pos = (ds.d_max + ds.d_min) / 2
        else:
            x_pos = ds.mean
        p_obs = ds.x / ds.n
        _, margin = wilson_interval(ds.x, ds.n)
        # x errorbar is 95% CI for the position distribution (+- 1.96*std or half the uniform interval)
        if ds.d_max is not None:
            x_err = 0.95 * (ds.d_max - ds.d_min) / 2
        else:
            x_err = 1.96 * ds.std
        plt.errorbar(
            x_pos,
            p_obs,
            yerr=margin,
            xerr=x_err,
            fmt="o",
            label=ds.label,
            capsize=5,
            markersize=8,
        )

    # Plot fitted curve
    d_values = np.linspace(
        0, max(ds.d_max or (ds.mean + 5 * ds.std) for ds in datasets), 500
    )
    p_values = [A * np.exp(-(d**2) / (sigma**2)) for d in d_values]
    plt.plot(d_values, p_values, label="Fitted curve", color="black")

    plt.xlabel("Distance (μm)")
    plt.ylabel("Connectivity Probability")
    plt.title(f"Connectivity Fit: {group_label}")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir, f"connectivity_fit_{group_label.replace(' ', '_')}.png"
        )
    )
    plt.close()


if __name__ == "__main__":
    # Datasets from Connectivity_intrinsic_striatum/connectivity_probabilities.ods
    """
    Some datasets were for SPN-SPN --> I use them for all combinations
    (dSPN-dSPN, dSPN-iSPN, iSPN-dSPN, iSPN-iSPN) but divide the counts by 4
    Some datasets were for FS-SPN --> I use them for both combinations
    (FS-dSPN, FS-iSPN) but divide the counts by 2

    """
    datasets = {
        "dSPN-dSPN": [
            Dataset(7, 0, 0, d_max=50, label="[1] 6-OHDA"),
            Dataset(8, 0, 0, d_max=50, label="[1] reserpine"),
            Dataset(19, 5, 0, d_max=50, label="[1] baseline"),
            Dataset(7, 3, 0, d_max=50, label="[1] baseline"),
            Dataset(43, 3, d_max=100, label="[2] baseline"),
            Dataset(202 // 4, 40 // 4, 0 // 4, d_max=100, label="[2] baseline"),
            Dataset(
                45 // 4, 9 // 4, 0 // 4, d_min=153, d_max=509, label="[7] baseline"
            ),
            Dataset(69 // 4, 18 // 4, 8 // 4, d_max=5, label="[8] baseline"),
            Dataset(38 // 4, 5 // 4, d_max=10, label="[8] baseline"),
            Dataset(38 // 4, 12 // 4, 1 // 4, d_min=2, d_max=50, label="[9] baseline"),
            Dataset(325 // 4, 39 // 4, d_max=100, label="[10] baseline"),
        ],
        "dSPN-iSPN": [
            Dataset(24, 3, d_max=50, label="[1] baseline"),
            Dataset(66, 3, d_max=100, label="[2] baseline"),
            Dataset(202 // 4, 40 // 4, 0 // 4, d_max=100, label="[2] baseline"),
            Dataset(
                45 // 4, 9 // 4, 0 // 4, d_min=153, d_max=509, label="[7] baseline"
            ),
            Dataset(69 // 4, 18 // 4, 8 // 4, d_max=5, label="[8] baseline"),
            Dataset(38 // 4, 5 // 4, d_max=10, label="[8] baseline"),
            Dataset(38 // 4, 12 // 4, 1 // 4, d_min=2, d_max=50, label="[9] baseline"),
            Dataset(325 // 4, 39 // 4, d_max=100, label="[10] baseline"),
        ],
        "FS-dSPN": [
            Dataset(9, 8, d_max=100, label="[2] baseline"),
            Dataset(90, 48, d_max=250, label="[3] baseline"),
            Dataset(80, 43, mean=105, std=50.1, label="[6] 6-OHDA"),
            Dataset(96, 58, mean=113, std=49, label="[6] baseline"),
            Dataset(39 // 2, 29 // 2, d_max=100, label="[2] baseline"),
            Dataset(167 // 2, 75 // 2, mean=106, std=25, label="[3] baseline"),
        ],
        "FS-FS": [
            Dataset(6, 1, 3, d_max=250, label="[3] baseline"),
            # Dataset(6, 0, 2, d_max=250, label="[4] baseline gap"),
            # Dataset(721650, 0, 167, d_max=1000, label="[11] baseline gap"),
            # Dataset(721650, 0, 4000, d_max=1000, label="[11] baseline gap"),
            # Dataset(78, 0, 6, d_max=200, label="[12] baseline gap"),
            Dataset(78, 0, 0, d_max=200, label="[12] baseline"),
            Dataset(85, 50, 22, d_max=100, label="[14] baseline"),
            Dataset(66, 20, 9, d_min=100, d_max=200, label="[14] baseline"),
            Dataset(19, 0, 0, d_min=200, d_max=800, label="[14] baseline"),
        ],
        "FS-iSPN": [
            Dataset(9, 6, d_max=100, label="[2] baseline"),
            Dataset(86, 66, mean=101, std=48, label="[6] 6-OHDA"),
            Dataset(108, 42, mean=116, std=46, label="[6] baseline"),
            Dataset(77, 27, d_max=250, label="[3] baseline"),
            Dataset(39 // 2, 29 // 2, d_max=100, label="[2] baseline"),
            Dataset(167 // 2, 75 // 2, mean=106, std=25, label="[3] baseline"),
        ],
        "iSPN-dSPN": [
            Dataset(12, 3, d_max=50, label="[1] 6-OHDA"),
            Dataset(10, 1, d_max=50, label="[1] reserpine"),
            Dataset(24, 13, d_max=50, label="[1] baseline"),
            Dataset(80, 10, d_max=100, label="[2] baseline"),
            Dataset(202 // 4, 40 // 4, 0 // 4, d_max=100, label="[2] baseline"),
            Dataset(
                45 // 4, 9 // 4, 0 // 4, d_min=153, d_max=509, label="[7] baseline"
            ),
            Dataset(69 // 4, 18 // 4, 8 // 4, d_max=5, label="[8] baseline"),
            Dataset(38 // 4, 5 // 4, d_max=10, label="[8] baseline"),
            Dataset(38 // 4, 12 // 4, 1 // 4, d_min=2, d_max=50, label="[9] baseline"),
            Dataset(325 // 4, 39 // 4, d_max=100, label="[10] baseline"),
        ],
        "iSPN-iSPN": [
            Dataset(17, 3, 0, d_max=50, label="[1] 6-OHDA"),
            Dataset(18, 5, 0, d_max=50, label="[1] reserpine"),
            Dataset(39, 14, 0, d_max=50, label="[1] baseline"),
            Dataset(9, 4, 0, d_max=50, label="[1] baseline"),
            Dataset(31, 7, d_max=100, label="[2] baseline"),
            Dataset(202 // 4, 40 // 4, 0 // 4, d_max=100, label="[2] baseline"),
            Dataset(
                45 // 4, 9 // 4, 0 // 4, d_min=153, d_max=509, label="[7] baseline"
            ),
            Dataset(69 // 4, 18 // 4, 8 // 4, d_max=5, label="[8] baseline"),
            Dataset(38 // 4, 5 // 4, d_max=10, label="[8] baseline"),
            Dataset(38 // 4, 12 // 4, 1 // 4, d_min=2, d_max=50, label="[9] baseline"),
            Dataset(325 // 4, 39 // 4, d_max=100, label="[10] baseline"),
        ],
    }

    output_dir = "connectivity_fits"
    os.makedirs(output_dir, exist_ok=True)

    params = {}  # initialize dict to collect fitted parameters

    for group, ds_list in datasets.items():
        # Bayesian optimization of negative log-likelihood
        search_space = [Real(1e-3, 500, name="sigma"), Real(0.0, 1.0, name="A")]
        res = gp_minimize(
            lambda params: neg_log_likelihood(params, ds_list),
            search_space,
            n_calls=50,
            n_initial_points=10,
            random_state=42,
        )
        sigma_hat, A_hat = res.x
        print(f"[{group}] Fitted sigma: {sigma_hat:.2f} μm, A: {A_hat:.2f}")

        # plotting
        plot_connectivity(ds_list, sigma_hat, A_hat, group, output_dir)

        # store in params dict under tuple key
        params[group] = {
            "amplitude": A_hat,
            "sigma_um": sigma_hat,
        }

    # after all fits, write out JSON
    with open(os.path.join(output_dir, "fitted_params.json"), "w") as f:
        json.dump(params, f, indent=4)
