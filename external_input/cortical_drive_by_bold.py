from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.fft import fft, ifft

CORTICAL_LABELS = (
    "M1",
    "PMd",
    "PMv",
    "preSMA",
    "SMA",
    "S1",
    "dlPFC",
)


def spm_hrf(tr, oversampling=1):
    """
    Generates a canonical Hemodynamic Response Function (Double Gamma)
    similar to the default SPM HRF.

    Parameters:
    -----------
    tr : float
        Repetition time in seconds.
    oversampling : int
        Sampling resolution (1 = sampled at TR).

    Returns:
    --------
    hrf : array
        The hemodynamic response function.
    """
    dt = tr / oversampling
    duration = 32.0  # Duration of HRF in seconds
    t = np.arange(0, duration, dt)

    # SPM default parameters for Double Gamma
    # Peak at 6s, Undershoot at 16s, Ratio of peak to undershoot 1/6
    a1 = 6.0
    a2 = 16.0
    b1 = 1.0
    b2 = 1.0
    c = 1.0 / 6.0

    # Probability Density Functions (using Gamma distribution)
    # We use scipy's gamma pdf, but we must adjust scale/loc to match SPM definition
    # SPM gamma: t^(a-1) * exp(-b*t) / Gamma(a) * b^a
    # Scipy gamma: x^(a-1) * exp(-x/scale) / Gamma(a) * scale^a
    # So scale = 1/b

    hrf_peak = stats.gamma.pdf(t, a1, scale=1 / b1)
    hrf_undershoot = stats.gamma.pdf(t, a2, scale=1 / b2)

    hrf = hrf_peak - c * hrf_undershoot

    # Normalize HRF so sum is 1 (preserves magnitude of signal)
    hrf = hrf / np.sum(hrf)

    return hrf


def deconvolve_neuronal_signal(bold_signal, tr, regularization=1.0):
    """
    Derives underlying neuronal drive from BOLD signal using
    Regularized Fourier Deconvolution (Wiener Deconvolution).

    Parameters:
    -----------
    bold_signal : array-like
        1D array of the preprocessed BOLD time series.
    tr : float
        Repetition Time in seconds.
    regularization : float
        Noise-to-signal ratio for regularization.
        Higher values smooth the output more; lower values allow more high-freq noise.
        Default 1.0 is conservative and standard for this simple implementation.

    Returns:
    --------
    neuronal_drive : array
        Estimated neuronal time series.
    """
    n_timepoints = len(bold_signal)

    # 1. Generate HRF sampled at TR
    hrf = spm_hrf(tr)

    # 2. Pad HRF to match the length of the BOLD signal for FFT
    # We must ensure length consistency for element-wise division in freq domain
    hrf_padded = np.zeros(n_timepoints)
    hrf_len = min(len(hrf), n_timepoints)
    hrf_padded[:hrf_len] = hrf[:hrf_len]

    # 3. FFT (Fast Fourier Transform)
    B_f = fft(bold_signal)
    H_f = fft(hrf_padded)

    # 4. Regularized Deconvolution (Wiener Filter)
    # Formula: N(f) = [B(f) * H*(f)] / [|H(f)|^2 + lambda]
    # This avoids division by zero where H(f) is small
    H_conj = np.conj(H_f)
    H_power = np.abs(H_f) ** 2

    # Note: If noise variance is unknown, lambda is often chosen empirically.
    # We assume standard normalized data, lambda=0.5 to 1.0 is robust.
    # To strictly match Gitelman, one might use noise estimation,
    # but a constant constant is the "simple" standard.

    damped_denominator = H_power + regularization

    neuronal_f = (B_f * H_conj) / damped_denominator

    # 5. Inverse FFT to get back to time domain
    neuronal_drive = np.real(ifft(neuronal_f))

    # Optional: z-score the output to match input scaling
    neuronal_drive = stats.zscore(neuronal_drive)

    return neuronal_drive


def load_cortical_bold_timeseries(file_path, condition="on"):
    """Load cortical BOLD time series from the Berlin dataset file."""

    file_path = Path(file_path)
    with h5py.File(file_path, "r") as f:
        labels = np.array([label.decode("UTF-8") for label in f["labels"][()]])
        if condition not in f["time_series"]:
            raise KeyError(
                f"Condition '{condition}' not found; available: {list(f['time_series'].keys())}"
            )
        time_series = f["time_series"][condition][()]

    n_cols = time_series.shape[1]
    trimmed_labels = labels[:n_cols]
    cortical_indices = [
        i for i, lbl in enumerate(trimmed_labels) if lbl in CORTICAL_LABELS
    ]
    if not cortical_indices:
        raise ValueError("No cortical labels found in dataset.")

    return {trimmed_labels[idx]: time_series[:, idx] for idx in cortical_indices}


def plot_bold_neuronal_and_rate(
    bold_data, neuronal_results, rate_results, tr, condition
):
    """Plot z-scored BOLD, neuronal drive (z) and firing rate for each cortical region."""

    regions = list(bold_data.keys())
    n_regions = len(regions)
    time = np.arange(len(next(iter(bold_data.values())))) * tr

    fig, axes = plt.subplots(n_regions, 3, figsize=(16, 3 * n_regions), sharex="col")
    axes = np.atleast_2d(axes)

    for row, region in enumerate(regions):
        bold_ts = bold_data[region]
        neuronal_ts = neuronal_results[region]
        firing_rate_ts = rate_results[region]

        axes[row, 0].plot(time, stats.zscore(bold_ts), color="tab:blue")
        axes[row, 0].set_ylabel(f"{region} (z)")
        if row == 0:
            axes[row, 0].set_title("BOLD")

        axes[row, 1].plot(time, neuronal_ts, color="tab:green")
        if row == 0:
            axes[row, 1].set_title("Neuronal drive (z)")

        axes[row, 2].plot(time, firing_rate_ts, color="tab:red")
        if row == 0:
            axes[row, 2].set_title("Firing rate (Hz)")

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    axes[-1, 2].set_xlabel("Time (s)")
    fig.suptitle(
        f"Cortical BOLD, neuronal drive, and firing rate (condition='{condition}')",
        y=0.995,
    )
    fig.tight_layout()
    plt.show()


def convert_to_firing_rate(neuronal_drive, target_mean_hz):
    """
    Converts a Z-scored neuronal drive into a positive firing rate (Hz)
    using an Exponential Linking Function (GLM approach).

    Parameters:
    -----------
    neuronal_drive : array
        The dimensionless Z-scored drive (centered at 0).
    target_mean_hz : float
        The desired average firing rate over the entire time series (e.g., 5.0).

    Returns:
    --------
    firing_rate : array
        Time series in Hz, strictly positive, with mean = target_mean_hz.
    """
    # 1. Apply Exponential Transfer Function (ensures positivity)
    # This assumes the drive modulates the rate log-linearly
    raw_rate = np.exp(neuronal_drive)

    # 2. Calculate current mean of the raw exponential data
    current_mean = np.mean(raw_rate)

    # 3. Determine scaling factor to hit the target mean
    scaling_factor = target_mean_hz / current_mean

    # 4. Apply scaling
    firing_rate = raw_rate * scaling_factor

    return firing_rate


# --- Main Execution Example ---

if __name__ == "__main__":
    # 1. Setup Parameters
    TR = 2.31  # TR in seconds
    TARGET_MEAN_HZ = 5.0  # Target mean firing rate in Hz for scaling
    condition = "on"  # use "on" or "off" block from the dataset
    data_file = (
        Path(__file__).resolve().parent.parent
        / "experimental_data/berlin_data/bold_data_roi/sub-01/sub-01_subdiv_results.h5"
    )

    # 2. Load actual cortical BOLD data
    bold_data = load_cortical_bold_timeseries(data_file, condition=condition)
    regions = list(bold_data.keys())

    print(
        f"Processing {len(regions)} cortical regions from {data_file.name} (condition='{condition}') with TR={TR}s..."
    )
    print("-" * 30)

    # 3. Run Deconvolution
    neuronal_drive_results = {}
    rate_results = {}

    for region_name, bold_timeseries in bold_data.items():
        # Perform deconvolution
        neuronal_drive = deconvolve_neuronal_signal(bold_timeseries, TR)
        neuronal_drive_results[region_name] = neuronal_drive

        # Convert to Firing Rate (Hz)
        rate = convert_to_firing_rate(neuronal_drive, TARGET_MEAN_HZ)
        rate_results[region_name] = rate

        # Report summary statistics for the three signals
        bold_z = stats.zscore(bold_timeseries)
        print(
            f"{region_name}: BOLD z-mean={np.mean(bold_z):.3f}, z-std={np.std(bold_z):.3f}; "
            f"Neuronal drive z-mean={np.mean(neuronal_drive):.3f}, z-std={np.std(neuronal_drive):.3f}; "
            f"Firing rate mean={np.mean(rate):.3f} Hz, std={np.std(rate):.3f} Hz"
        )

    print("-" * 30)
    print("Done.")

    # 4. Plot BOLD and neuronal drive per region
    plot_bold_neuronal_and_rate(
        bold_data, neuronal_drive_results, rate_results, TR, condition
    )
