"""
Deconvolution idea from: Glover, G. H. (1999). Deconvolution of Impulse Response in Event-Related BOLD fMRI1. NeuroImage, 9(4), 416–429. https://doi.org/10.1006/nimg.1998.0419
HRF model from SPM implementation.
"""

from pathlib import Path
import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.fft import fft, ifft
import matlab.engine
import os
from scipy.interpolate import CubicSpline

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


def matlab_hrf(tr):
    """
    Generates a canonical Hemodynamic Response Function (Double Gamma)
    using the default SPM HRF from MATLAB.

    Parameters:
    -----------
    tr : float
        Repetition time in seconds.

    Returns:
    --------
    hrf : array
        The hemodynamic response function.
    """
    # Start the engine
    eng = matlab.engine.start_matlab()

    # Get the current directory where the python script and .m files are located
    current_path = os.getcwd()

    # Add this path to the MATLAB environment so it can find the files
    eng.addpath(current_path)

    # Call the function.
    # nargout=2 tells MATLAB we expect 2 return values ([hrf, p])
    # Ensure inputs are floats, not integers
    hrf_matlab, p_matlab = eng.spm_hrf(float(tr), nargout=2)

    # Stop the engine
    eng.quit()

    # Convert the result (MATLAB array) to a Python/Numpy list for easier use
    hrf_data = np.array(hrf_matlab).flatten()

    return hrf_data


def deconvolve_neuronal_signal(
    bold_signal, tr, regularization=1.0, use_matlab_hrf=False
):
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
    use_matlab_hrf : bool
        If True, use the MATLAB/SPM HRF; otherwise use the local scipy-based HRF.

    Returns:
    --------
    neuronal_drive : array
        Estimated neuronal time series.
    """
    n_timepoints = len(bold_signal)

    # 1. Generate HRF sampled at TR
    hrf = matlab_hrf(tr) if use_matlab_hrf else spm_hrf(tr)

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
    diff = len(labels) - n_cols
    if diff < 0 or diff > 1:
        raise ValueError(
            f"Mismatch between labels ({len(labels)}) and time_series columns ({n_cols})."
        )

    if diff == 1:
        trimmed_labels = labels[-n_cols:]
        print(
            f"Warning: labels ({len(labels)}) exceed time_series columns ({n_cols}); using last {n_cols} labels."
        )
    else:  # diff == 0
        trimmed_labels = labels
    cortical_indices = [
        i for i, lbl in enumerate(trimmed_labels) if lbl in CORTICAL_LABELS
    ]
    if not cortical_indices:
        raise ValueError("No cortical labels found in dataset.")

    return {trimmed_labels[idx]: time_series[:, idx] for idx in cortical_indices}


def plot_bold_neuronal_and_rate(
    bold_data, neuronal_results, rate_results, tr, condition, output_path
):
    """Save plots of BOLD, neuronal drive (z) and upsampled firing rate for each cortical region."""

    regions = list(bold_data.keys())
    n_regions = len(regions)
    coarse_time = np.arange(len(next(iter(bold_data.values())))) * tr

    fig, axes = plt.subplots(n_regions, 3, figsize=(16, 3 * n_regions))
    axes = np.atleast_2d(axes)

    for row, region in enumerate(regions):
        bold_ts = bold_data[region]
        neuronal_ts = neuronal_results[region]
        firing_data = rate_results[region]
        fine_time = firing_data["time"]
        fine_rate = firing_data["rate"]

        axes[row, 0].plot(coarse_time, stats.zscore(bold_ts), color="tab:blue")
        axes[row, 0].set_ylabel(f"{region} (z)")
        if row == 0:
            axes[row, 0].set_title("BOLD")

        axes[row, 1].plot(coarse_time, neuronal_ts, color="tab:green")
        if row == 0:
            axes[row, 1].set_title("Neuronal drive (z)")

        axes[row, 2].plot(fine_time, fine_rate, color="tab:red")
        if row == 0:
            axes[row, 2].set_title("Firing rate (Hz, upsampled)")

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    axes[-1, 2].set_xlabel("Time (s)")
    fig.suptitle(
        f"Cortical BOLD, neuronal drive, and firing rate (condition='{condition}')",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


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


def upsample_time_series(
    coarse_rate, tr, dt_target=0.0001, add_noise=False, noise_level=0.05
):
    """
    Upsamples a coarse firing rate time series (e.g. TR=2s) to a fine resolution (e.g. 0.1ms)
    using Cubic Spline Interpolation.

    Parameters:
    -----------
    coarse_rate : array
        The firing rate time series at TR resolution.
    tr : float
        The original Repetition Time in seconds (e.g., 2.0).
    dt_target : float
        The desired time step in seconds (e.g., 0.0001 for 0.1ms).
    add_noise : bool
        If True, adds Ornstein-Uhlenbeck (colored) noise to mimic synaptic fluctuations.
    noise_level : float
        Amplitude of the noise (relative to signal standard deviation).

    Returns:
    --------
    fine_time : array
        The new time vector.
    fine_rate : array
        The upsampled firing rate.
    """
    n_coarse = len(coarse_rate)
    duration = n_coarse * tr

    # Create the original time grid (0, 2, 4...)
    t_coarse = np.arange(0, duration, tr)

    # Create the target fine time grid (0, 0.0001, 0.0002...)
    t_fine = np.arange(0, duration, dt_target)

    # --- A. Cubic Spline Interpolation ---
    # This fits a smooth curve through the coarse points
    # bc_type='natural' ensures 2nd derivative is zero at endpoints (prevents wild oscillations at edges)
    cs = CubicSpline(t_coarse, coarse_rate, bc_type="natural")

    fine_rate = cs(t_fine)

    # --- B. Optional: Add Synaptic Noise (Ornstein-Uhlenbeck) ---
    if add_noise:
        # Simple implementation of colored noise
        # This adds "texture" so the signal isn't perfectly smooth at 0.1ms
        num_steps = len(t_fine)
        noise = np.zeros(num_steps)
        sigma = np.std(fine_rate) * noise_level  # Scale noise to signal amplitude
        tau = 0.01  # Time constant of synaptic noise (10ms)

        # Euler-Maruyama integration for OU process
        dt_sqrt = np.sqrt(dt_target)
        for i in range(1, num_steps):
            noise[i] = (
                noise[i - 1]
                - (noise[i - 1] / tau) * dt_target
                + sigma * np.random.normal() * dt_sqrt
            )

        fine_rate += noise

    # Ensure rate doesn't go below zero due to spline undershoot or noise
    fine_rate = np.maximum(fine_rate, 0.0)

    return t_fine, fine_rate


# --- Main Execution Example ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute cortical neuronal drive and upsampled firing rates from BOLD."
    )
    parser.add_argument(
        "--condition",
        default="on",
        help="Condition block to process (e.g., 'on' or 'off'). Default: on",
    )
    args = parser.parse_args()

    # 1. Setup Parameters
    TR = 2.31  # TR in seconds
    TARGET_MEAN_HZ = 5.0  # Target mean firing rate in Hz for scaling
    condition = args.condition
    base_dir = Path(__file__).resolve().parents[2]
    results_dir = Path(__file__).resolve().parent / "cortical_firing_rates_data"
    results_dir.mkdir(parents=True, exist_ok=True)
    data_file = (
        base_dir
        / "experimental_data/berlin_data/bold_data_roi/sub-01/sub-01_subdiv_results.h5"
    )

    # 2. Load actual cortical BOLD data
    bold_data = load_cortical_bold_timeseries(data_file, condition=condition)
    regions = list(bold_data.keys())

    # 3. Run Deconvolution
    # with scipy HRF
    print(
        f"Processing {len(regions)} cortical regions from {data_file.name} (condition='{condition}') with TR={TR}s using scipy HRF..."
    )
    print("-" * 30)
    neuronal_drive_results = {}
    rate_results = {}

    for region_name, bold_timeseries in bold_data.items():
        # Perform deconvolution
        neuronal_drive = deconvolve_neuronal_signal(bold_timeseries, TR)
        neuronal_drive_results[region_name] = neuronal_drive

        # Convert to Firing Rate (Hz)
        rate = convert_to_firing_rate(neuronal_drive, TARGET_MEAN_HZ)
        fine_time, fine_rate = upsample_time_series(rate, TR, dt_target=0.0001)
        rate_results[region_name] = {"time": fine_time, "rate": fine_rate}

        # Report summary statistics for the three signals
        bold_z = stats.zscore(bold_timeseries)
        print(
            f"{region_name} (scipy HRF): BOLD z-mean={np.mean(bold_z):.3f}, z-std={np.std(bold_z):.3f}; "
            f"Neuronal drive z-mean={np.mean(neuronal_drive):.3f}, z-std={np.std(neuronal_drive):.3f}; "
            f"Firing rate mean={np.mean(fine_rate):.3f} Hz, std={np.std(fine_rate):.3f} Hz"
            f"Shape of BOLD: {bold_timeseries.shape}, Neuronal drive: {neuronal_drive.shape}, Firing rate: {fine_rate.shape}\n"
        )

    print("-" * 30)
    print("Done.\n\n")

    # Save fine firing rate time series for scipy HRF
    scipy_payload = {}
    for region, data in rate_results.items():
        scipy_payload[f"{region}_time"] = data["time"]
        scipy_payload[f"{region}_rate"] = data["rate"]

    np.savez_compressed(
        results_dir / f"firing_rates_scipy_condition-{condition}.npz", **scipy_payload
    )

    # with MATLAB HRF
    print(
        f"Processing {len(regions)} cortical regions from {data_file.name} (condition='{condition}') with TR={TR}s using MATLAB HRF..."
    )
    print("-" * 30)

    neuronal_drive_results_matlab = {}
    rate_results_matlab = {}

    for region_name, bold_timeseries in bold_data.items():
        neuronal_drive_matlab = deconvolve_neuronal_signal(
            bold_timeseries, TR, use_matlab_hrf=True
        )
        neuronal_drive_results_matlab[region_name] = neuronal_drive_matlab

        rate_matlab = convert_to_firing_rate(neuronal_drive_matlab, TARGET_MEAN_HZ)
        fine_time_matlab, fine_rate_matlab = upsample_time_series(
            rate_matlab, TR, dt_target=0.0001
        )
        rate_results_matlab[region_name] = {
            "time": fine_time_matlab,
            "rate": fine_rate_matlab,
        }

        bold_z = stats.zscore(bold_timeseries)
        print(
            f"{region_name} (MATLAB HRF): BOLD z-mean={np.mean(bold_z):.3f}, z-std={np.std(bold_z):.3f}; "
            f"Neuronal drive z-mean={np.mean(neuronal_drive_matlab):.3f}, z-std={np.std(neuronal_drive_matlab):.3f}; "
            f"Firing rate mean={np.mean(fine_rate_matlab):.3f} Hz, std={np.std(fine_rate_matlab):.3f} Hz"
            f"Shape of BOLD: {bold_timeseries.shape}, Neuronal drive: {neuronal_drive_matlab.shape}, Firing rate: {fine_rate_matlab.shape}\n"
        )

    print("-" * 30)
    print("Done.")

    # Save fine firing rate time series for MATLAB HRF
    matlab_payload = {}
    for region, data in rate_results_matlab.items():
        matlab_payload[f"{region}_time"] = data["time"]
        matlab_payload[f"{region}_rate"] = data["rate"]

    np.savez_compressed(
        results_dir / f"firing_rates_matlab_condition-{condition}.npz",
        **matlab_payload,
    )

    # 4. Plot BOLD and neuronal drive per region
    # for scipy HRF
    plot_bold_neuronal_and_rate(
        bold_data,
        neuronal_drive_results,
        rate_results,
        TR,
        condition,
        results_dir / f"plot_scipy_condition-{condition}.png",
    )

    # for MATLAB HRF
    plot_bold_neuronal_and_rate(
        bold_data,
        neuronal_drive_results_matlab,
        rate_results_matlab,
        TR,
        condition,
        results_dir / f"plot_matlab_condition-{condition}.png",
    )
