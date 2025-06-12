import numpy as np
from scipy.integrate import quad
from skopt import gp_minimize
from skopt.space import Real
import matplotlib.pyplot as plt
from scipy.stats import norm
import json  # added for JSON output


class Dataset:
    def __init__(self, successes, trials, d_max=None, mean=None, std=None, label=None):
        """
        Initializes a Dataset instance.

        Args:
            successes (int): Number of successful observations.
            trials (int): Total number of trials.
            d_max (Optional[float]): Maximum distance for the uniform distribution.
            mean (Optional[float]): Mean of the normal distance distribution.
            std (Optional[float]): Standard deviation of the normal distance distribution.
            label (Optional[str]): Label for the dataset.
        """
        self.x = successes
        self.n = trials
        self.d_min = 0
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
            # Calculate expectation over a uniform distance interval [0, d_max]
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


if __name__ == "__main__":
    # Datasets from experimental_data/Connectivity_intrinsic_striatum/connectivity_probabilities.ods
    datasets = {
        "dSPN-dSPN": [
            Dataset(0, 7, d_max=50, label="0/7 6-OHDA lesion"),
            Dataset(5, 19, d_max=50, label="5/19 baseline"),
            Dataset(3, 43, d_max=100, label="3/43 baseline"),
        ],
        "dSPN-iSPN": [
            Dataset(3, 47, d_max=50, label="3/47 baseline"),
            Dataset(3, 66, d_max=100, label="3/66 baseline"),
        ],
        # "FS-Chol": [
        #     Dataset(0, 3, d_max=250, label="0/3 baseline"),
        # ],
        "FS-dSPN": [
            Dataset(22, 40, mean=105, std=50.1, label="22/40 6-OHDA lesion"),
            Dataset(8, 9, d_max=100, label="8/9 baseline"),
            Dataset(48, 90, d_max=250, label="48/90 baseline"),
            Dataset(29, 48, mean=113, std=49, label="29/48 baseline"),
        ],
        "FS-FS": [
            Dataset(7, 12, mean=106, std=25, label="7/12 baseline"),
            Dataset(2, 6, d_max=250, label="2/6 baseline"),
            Dataset(3, 7, d_max=250, label="3/7 baseline"),
        ],
        "FS-iSPN": [
            Dataset(33, 43, mean=101, std=48, label="33/43 6-OHDA lesion"),
            Dataset(6, 9, d_max=100, label="6/9 baseline"),
            Dataset(21, 54, mean=116, std=46, label="21/54 baseline"),
            Dataset(27, 77, d_max=250, label="27/77 baseline"),
        ],
        # "FS-PLTS": [
        #     Dataset(0, 9, d_max=250, label="0/9 baseline"),
        # ],
        "iSPN-dSPN": [
            Dataset(3, 12, d_max=50, label="3/12 6-OHDA lesion"),
            Dataset(13, 47, d_max=50, label="13/47 baseline"),
            Dataset(10, 80, d_max=100, label="10/80 baseline"),
        ],
        "iSPN-iSPN": [
            Dataset(3, 17, d_max=50, label="3/17 6-OHDA lesion"),
            Dataset(14, 39, d_max=50, label="14/39 baseline"),
            Dataset(7, 31, d_max=100, label="7/31 baseline"),
        ],
        # "NPYNGF-SPN": [
        #     Dataset(25, 29, d_max=100, label="25/29 baseline"),
        # ],
        # "NPYPLTS-SPN": [
        #     Dataset(0, 9, d_max=100, label="0/9 baseline"),
        #     Dataset(3, 21, d_max=100, label="3/21 baseline"),
        # ],
        # "PLTS-Chol": [
        #     Dataset(0, 8, d_max=250, label="0/8 baseline"),
        # ],
        # "PLTS-FS": [
        #     Dataset(0, 9, d_max=250, label="0/9 baseline"),
        # ],
        # "PLTS-PLTS": [
        #     Dataset(0, 26, d_max=250, label="0/26 baseline"),
        # ],
        # "PLTS-SPN": [
        #     Dataset(2, 60, mean=153, std=80, label="2/60 baseline"),
        # ],
    }

    import os

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

        d_vals = np.linspace(0, 200, 500)
        p_curve = A_hat * np.exp(-(d_vals**2) / (sigma_hat**2))

        plt.figure()
        plt.plot(d_vals, p_curve, label="Fitted curve")

        for ds in ds_list:
            if ds.d_max is not None:
                x_pos = ds.d_max / 2
            else:
                x_pos = ds.mean
            p_obs = ds.x / ds.n
            center, margin = wilson_interval(ds.x, ds.n)
            plt.errorbar(x_pos, p_obs, yerr=margin, fmt="o", label=ds.label)

        plt.xlabel("Distance (μm)")
        plt.ylabel("Connectivity probability")
        plt.title(f"Scaled half-Gaussian fit: {group}")
        plt.xlim(0, 200)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(output_dir, f"{group.replace(' ', '_')}.png"))
        plt.close()

        # store in params dict under tuple key
        params[group] = {
            "amplitude": A_hat,
            "sigma_um": sigma_hat,
        }

    # after all fits, write out JSON
    with open(os.path.join(output_dir, "fitted_params.json"), "w") as f:
        json.dump(params, f, indent=4)
