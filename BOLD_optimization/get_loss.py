from ANNarchy import setup, get_population, set_seed, reset, simulate, get_projection
from ANNarchy.extensions.bold import BoldMonitor
from CompNeuroPy.full_models import BGM
from CompNeuroPy import CompNeuroMonitors, CompNeuroExp, DBSstimulator
import argparse
import json
import numpy as np
from pathlib import Path
import h5py
import sys

### local
from parameters import parameters_test_microcircuit as paramsS


def set_opt_params(param_list, model_dict):
    """
    Function to set the optimized parameters of the model.

    The ranges of the parameters ar given as follows:
    - minimum is always 0.0
    - maximum is given by the numerical stabilization of the conductance values
        g_eff = g / (1 + g * dt / C), C=50 for striatum d1 and d2 and C=80 for fsi and C=1 for other populations
        with dt = 0.1 ms
        --> for striatal populations d1, d2 max g ~ 500
        --> for striatal populations fsi max g ~ 800
        --> for other populations max g ~ 10
    - the base currents ranges are obtain by multiplying the conductance extreme values
    with the fixed driving force of 50 mV (current based excitation in the model)
        --> 10 * 50 = 500 max base current for snr and gpe_proto
    """

    ### FIRING RATE PARAMETERS ###
    # parameters which influence the firing rates of the populations:
    exp_input_weight_dict = {
        "str_d1": param_list[0],  # [0, 500]
        "str_d2": param_list[1],  # [0, 500]
        "str_fsi": param_list[2],  # [0, 800]
        "thal": param_list[3],  # [0, 10]
        "gpe_arky": param_list[4],  # [0, 10]
        "gpe_cp": param_list[5],  # [0, 10]
        "stn": param_list[6],  # [0, 10]
    }
    base_current_dict = {"snr": param_list[7], "gpe_proto": param_list[8]}  # [0, 500]

    for loop in ["caudate", "putamen"]:
        # get bgm model of loop
        bgm_model: BGM = model_dict[loop]
        # set the cortical input weight parameters
        # use the names of the populations (see model creation function) and add :loop (which is done internally in BGM)
        for compartment_name, param_val in exp_input_weight_dict.items():
            bgm_model.set_param(
                compartment=f"{compartment_name}:{loop}",
                parameter_name="exp_input_weight",
                parameter_value=param_val,
            )
        # snr, gpe_proto: without cortical inputs --> regulate baseline current
        for compartment_name, param_val in base_current_dict.items():
            bgm_model.set_param(
                compartment=f"{compartment_name}:{loop}",
                parameter_name="base_mean",
                parameter_value=param_val,
            )

    ### WEIGHT PARAMETERS ###
    # projections are:
    # str_d1__snr
    # str_d1__gpe_cp
    # str_d1__str_d1
    # str_d1__str_d2
    # str_d2__gpe_proto
    # str_d2__gpe_arky
    # str_d2__gpe_cp
    # str_d2__str_d1
    # str_d2__str_d2
    # str_fsi__str_d1
    # str_fsi__str_d2
    # str_fsi__str_fsi
    # stn__snr
    # stn__gpe_proto
    # stn__gpe_arky
    # stn__gpe_cp
    # gpe_proto__stn
    # gpe_proto__snr
    # gpe_proto__gpe_arky
    # gpe_proto__gpe_cp
    # gpe_proto__str_fsi
    # gpe_arky__str_d1
    # gpe_arky__str_d2
    # gpe_arky__str_fsi
    # gpe_arky__gpe_proto
    # gpe_arky__gpe_cp
    # gpe_cp__str_d1
    # gpe_cp__str_d2
    # gpe_cp__str_fsi
    # gpe_cp__gpe_proto
    # gpe_cp__gpe_arky
    # snr__thal
    # thal__str_d1
    # thal__str_d2
    # thal__str_fsi

    # instad of setting all the weights indvidually, use the already defined weights of BGM and scale groups of them, we scale the origianl values by a factor between 0 and 5
    proj_clusters = {
        "str_d1__bg": ["str_d1__snr", "str_d1__gpe_cp"],
        "str_d2__bg": ["str_d2__gpe_proto", "str_d2__gpe_arky", "str_d2__gpe_cp"],
        "str_fsi__striatum": ["str_fsi__str_d1", "str_fsi__str_d2", "str_fsi__str_fsi"],
        "str_laterals": [
            "str_d1__str_d1",
            "str_d1__str_d2",
            "str_d2__str_d1",
            "str_d2__str_d2",
        ],
        "stn__gpe": ["stn__gpe_proto", "stn__gpe_arky", "stn__gpe_cp"],
        "stn__snr": ["stn__snr"],
        "snr__thal": ["snr__thal"],
        "thal__striatum": ["thal__str_d1", "thal__str_d2", "thal__str_fsi"],
        "gpe_laterals": [
            "gpe_proto__gpe_arky",
            "gpe_proto__gpe_cp",
            "gpe_arky__gpe_proto",
            "gpe_arky__gpe_cp",
            "gpe_cp__gpe_proto",
            "gpe_cp__gpe_arky",
        ],
        "gpe_striatum": [
            "gpe_proto__str_fsi",
            "gpe_arky__str_d1",
            "gpe_arky__str_d2",
            "gpe_arky__str_fsi",
            "gpe_cp__str_d1",
            "gpe_cp__str_d2",
            "gpe_cp__str_fsi",
        ],
        "gpe_proto__stn": ["gpe_proto__stn"],
        "gpe_proto__snr": ["gpe_proto__snr"],
    }
    all_projections = [
        proj_name for proj_list in proj_clusters.values() for proj_name in proj_list
    ]
    # create the reverse mapping for each projection to its cluster
    proj_to_cluster = {
        proj_name: cluster_name
        for cluster_name, proj_list in proj_clusters.items()
        for proj_name in proj_list
    }
    proj_cluster_scaling = {
        "str_d1__bg": param_list[9],  # [0, 5]
        "str_d2__bg": param_list[10],  # [0, 5]
        "str_fsi__striatum": param_list[11],  # [0, 5]
        "str_laterals": param_list[12],  # [0, 5]
        "stn__gpe": param_list[13],  # [0, 5]
        "stn__snr": param_list[14],  # [0, 5]
        "snr__thal": param_list[15],  # [0, 5]
        "thal__striatum": param_list[16],  # [0, 5]
        "gpe_laterals": param_list[17],  # [0, 5]
        "gpe_striatum": param_list[18],  # [0, 5]
        "gpe_proto__stn": param_list[19],  # [0, 5]
        "gpe_proto__snr": param_list[20],  # [0, 5]
    }

    # set the weights for both loops
    for loop in ["caudate", "putamen"]:
        bgm_model: BGM = model_dict[loop]
        for proj_name in all_projections:
            # get the scaling factor for the projection
            cluster_name = proj_to_cluster[proj_name]
            scaling_factor = proj_cluster_scaling[cluster_name]
            # get the original value of the weight defined in BGM.params
            weight_val_orig = bgm_model.params[f"{proj_name}:{loop}.weights"]
            # set the weight value
            bgm_model.set_param(
                compartment=f"{proj_name}:{loop}",
                parameter_name="w",
                parameter_value=weight_val_orig * scaling_factor,
            )


def infer_max_sim_time_ms(
    dbs_condition: str,
    cortical_rate_path=paramsS["mc.cortical_rate_path"],
    dt_ms: float = 0.1,
):
    """Load MATLAB cortical firing rates for a DBS condition and infer max simulation time (ms).

    The firing-rate arrays are sampled at the BOLD TR (2.31 s) without upsampling.
    Duration rule: (len(rates) * 2.31 / 1000) / dt_ms.

    Returns
    -------
    duration_ms : float
        Maximum simulation time derived from rate length.
    mixed_rates : dict
        Dict with caudate/putamen mixed firing rates and time vectors:
        {loop: {"time": array, "rate": array}}.
    """

    if not isinstance(cortical_rate_path, dict):
        raise ValueError(
            "cortical_rate_path must map dbs conditions to MATLAB rate files."
        )
    if dbs_condition not in cortical_rate_path:
        raise KeyError(
            f"DBS condition '{dbs_condition}' not found in cortical_rate_path mapping."
        )

    rate_file = cortical_rate_path[dbs_condition]

    with np.load(rate_file) as data:
        # Expect mixed signals saved as *_rate and *_time
        missing_keys = [k for k in ["caudate_rate", "putamen_rate"] if k not in data]
        if missing_keys:
            raise ValueError(
                f"Missing mixed rate entries {missing_keys} in cortical rate file '{rate_file}'."
            )

        caudate_rate = data["caudate_rate"]
        putamen_rate = data["putamen_rate"]

        # Time vectors are optional but preferred
        caudate_time = data.get("caudate_time")
        putamen_time = data.get("putamen_time")

    # New duration rule using coarse TR samples (no upsampling)
    duration_ms = (len(caudate_rate) * 2.31 * 1000.0) / dt_ms

    mixed_rates = {
        "caudate": {"time": caudate_time, "rate": caudate_rate},
        "putamen": {"time": putamen_time, "rate": putamen_rate},
    }

    return duration_ms, mixed_rates


def update_TimedInput(bgm_model: BGM):
    inputs = bgm_model.model_creation_kwargs["input.rates"]
    inputs = np.repeat(
        inputs[:, np.newaxis],
        repeats=bgm_model.params[f"str_d1:{bgm_model.name_appendix}.size"],
        axis=1,
    )
    schedule = bgm_model.model_creation_kwargs["input.schedule"]
    inp = get_population(f"TimedInput_cortex:{bgm_model.name_appendix}")
    # set schedule and period in c by my own (prevent ANNarchy bug)
    value = [float(schedule * i) for i in range(inputs.shape[0])]
    val_int = np.rint(
        np.atleast_1d(value) / bgm_model.model_creation_kwargs["timestep"]
    ).astype(np.int64)
    inp.cyInstance.set_schedule(val_int)
    value = -1
    period_steps = int(np.rint(value / bgm_model.model_creation_kwargs["timestep"]))
    inp.cyInstance.set_period(period_steps)


class Spikes10s(CompNeuroExp):
    def __init__(
        self,
        monitors: CompNeuroMonitors = None,
        model_dict: dict = None,
        seed: int = 42,
    ):
        super().__init__(monitors)
        self.model_dict = model_dict
        self.seed = seed

    def run(self, param_list: list):
        # at the begining always reset (also annarchy random!) and start monitors
        self.reset()
        self.monitors.start()
        set_seed(self.seed)

        # update the TimedInput to reset it
        for loop in ["caudate", "putamen"]:
            bgm_model: BGM = self.model_dict[loop]
            update_TimedInput(bgm_model)

        # set optimized parameters
        set_opt_params(param_list, self.model_dict)

        # calculate the update steps for 10 s
        duration_ms = 10000
        simulate(duration_ms)

        self.data["duration"] = duration_ms

        return self.results()


def get_firing_rate_10s(param_list: list, experiment: Spikes10s):
    # run the experiment and get the results
    results = experiment.run(param_list)
    duration_ms = results.data["duration"]

    # spike recordings are stored in the first entry of results.recordings
    recordings = results.recordings[0]
    max_steps = int(duration_ms / paramsS["timestep"])
    eval_proportion = 0.8
    eval_time_s = eval_proportion * (duration_ms / 1000)  # last 80% in seconds

    firing_rates = {}
    for key, spikes in recordings.items():
        # only process spike traces (keys look like "pop_name;spike")
        if not key.endswith(";spike"):
            continue

        pop_name = key.split(";")[0]
        n_neurons = len(spikes)
        if n_neurons == 0:
            firing_rates[pop_name] = 0.0
            continue

        # flatten all spike times from all neurons
        all_spike_times = (
            np.concatenate(list(spikes.values())) if len(spikes) > 0 else np.array([])
        )
        eval_spike_times = all_spike_times[
            all_spike_times >= (1 - eval_proportion) * max_steps
        ]
        firing_rate_hz = len(eval_spike_times) / (n_neurons * eval_time_s)
        firing_rates[pop_name] = firing_rate_hz

    return firing_rates


def get_BOLD_full(model_dict, seed, bold_monitor_dict, param_list: list):
    ### RESETS AND PREPARE ### TODO: how to reset BOLD monitor - skipped this just run scripts separately
    # reset ANNarchy and seed
    reset()
    set_seed(seed)

    # update the TimedInput to reset it
    for loop in ["caudate", "putamen"]:
        bgm_model: BGM = model_dict[loop]
        update_TimedInput(bgm_model)

    ### OPTIMIZED PARAMETERS ###
    # set optimized parameters
    set_opt_params(param_list, model_dict)

    ### RAMP UP ###
    # initial ramp up
    ramp_up_duration_ms = paramsS["t.rampup"]
    simulate(ramp_up_duration_ms)

    # start each bold monitor
    for bold_monitor in bold_monitor_dict.values():
        bold_monitor.start()

    ### REST OF SIMULATION ###
    simulate(paramsS["t.duration"] - ramp_up_duration_ms)

    ### RESULTS ###
    # get the bold time signals of each bold monitor
    bold_signals = {}
    for bold_region, bold_monitor in bold_monitor_dict.items():
        bold_signals[bold_region] = bold_monitor.get("BOLD")

    return bold_signals


def load_experimental_bold_timeseries(
    condition: str = "on",
    data_file: str | Path | None = None,
    target_labels: list[str] | None = None,
):
    """Load experimental BOLD time series.

    The helper mirrors the loader in
    striatal_microcircuit_requirements/cortical_firing_rates/cortical_drive_by_bold.py
    but keeps all labels unless ``target_labels`` filters them.
    """

    base_dir = Path(__file__).resolve().parents[1]
    if data_file is None:
        data_file = (
            base_dir
            / "experimental_data/berlin_data/bold_data_roi/sub-01/sub-01_subdiv_results.h5"
        )
    data_file = Path(data_file)

    with h5py.File(data_file, "r") as f:
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
    trimmed_labels = labels[-n_cols:] if diff == 1 else labels

    bold_dict = {lbl: time_series[:, idx] for idx, lbl in enumerate(trimmed_labels)}

    if target_labels:
        missing = [lbl for lbl in target_labels if lbl not in bold_dict]
        if missing:
            print(
                f"Warning: missing experimental BOLD labels {missing} in file {data_file.name}."
            )
        bold_dict = {lbl: bold_dict[lbl] for lbl in target_labels if lbl in bold_dict}

    return bold_dict


def downsample_bold_to_tr(
    sim_signal: np.ndarray, dt_ms: float, tr_s: float
) -> np.ndarray:
    """Downsample a simulated BOLD trace (dt in ms) to the TR grid."""

    step = int(np.rint((tr_s * 1000.0) / dt_ms))
    step = max(step, 1)
    return sim_signal[::step]


def compute_bold_correlation_loss(
    sim_bold: dict[str, np.ndarray],
    condition: str = "on",
    tr_s: float = 2.31,
    ramp_up_ms: float = paramsS["t.rampup"],
    data_file: str | Path | None = None,
    region_map: dict[str, str] | None = None,
    dt_ms: float = paramsS["timestep"],
):
    """Compare simulated BOLD to experimental data via mean correlation.

    Parameters
    ----------
    sim_bold : dict
        Simulated BOLD signals keyed by region name.
    condition : str
        Experimental condition to load from the HDF5 file.
    tr_s : float
        Repetition time (seconds) of the experimental data.
    ramp_up_ms : float
        Ramp-up duration in the simulation to remove from experimental data.
    data_file : str or Path, optional
        Optional custom path to the experimental HDF5 file.
    region_map : dict, optional
        Mapping from simulated region names to experimental labels.
    dt_ms : float
        Simulation timestep in milliseconds.

    Returns
    -------
    loss : float
        1 - mean correlation across matched regions (higher is worse).
    per_region_corr : dict
        Correlation per region for inspection.
    """

    region_map = region_map or {}
    target_labels = [region_map.get(region, region) for region in sim_bold.keys()]
    exp_bold = load_experimental_bold_timeseries(
        condition=condition, data_file=data_file, target_labels=target_labels
    )

    ramp_up_steps = int(np.ceil(ramp_up_ms / (tr_s * 1000.0)))
    correlations = {}

    def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
        # Flatten and trim to the shared length to avoid shape mismatches (e.g., column vectors)
        x_flat = np.asarray(x).ravel()
        y_flat = np.asarray(y).ravel()
        n_shared = min(len(x_flat), len(y_flat))

        if n_shared < 2:
            return float("nan")

        x_use = x_flat[:n_shared]
        y_use = y_flat[:n_shared]

        x_std = np.std(x_use)
        y_std = np.std(y_use)
        if x_std == 0.0 or y_std == 0.0:
            return float("nan")

        return float(np.corrcoef(x_use, y_use)[0, 1])

    for sim_region, exp_label in zip(sim_bold.keys(), target_labels):
        if exp_label not in exp_bold:
            continue

        sim_series = np.asarray(sim_bold[sim_region])
        exp_series = np.asarray(exp_bold[exp_label])

        sim_coarse = downsample_bold_to_tr(sim_series, dt_ms=dt_ms, tr_s=tr_s)
        exp_trimmed = exp_series[ramp_up_steps:] if ramp_up_steps > 0 else exp_series

        if sim_region == "GPi":
            print(
                f"size of sim_series: {len(sim_series)}, size of exp_series: {len(exp_series)}"
            )
            print(
                f"size of sim_coarse: {len(sim_coarse)}, size of exp_trimmed: {len(exp_trimmed)}"
            )

        n = min(len(sim_coarse), len(exp_trimmed))
        if n < 2:
            correlations[sim_region] = float("nan")
            continue

        correlations[sim_region] = _safe_corr(sim_coarse[:n], exp_trimmed[:n])

    valid_corrs = [c for c in correlations.values() if not np.isnan(c)]
    if not valid_corrs:
        return 1.0, correlations

    # Map mean correlation in [-1, 1] to goodness in [0, 1], then convert to loss.
    mean_corr = float(np.mean(valid_corrs))
    goodness = float(np.clip((mean_corr + 1.0) / 2.0, 0.0, 1.0))
    loss = 1.0 - goodness  # loss=0 at perfect (1.0), loss=1 at worst (-1.0)
    return loss, correlations


def get_firing_rate_loss(firing_rate_dict: dict[str, float]) -> float:
    """
    Calculate a loss based on how far the firing rates are from plausible ranges.
    Goodness is smooth and bounded in [0, 1] using a logistic on relative deviation
    from the center of the plausible band; loss is 1 - mean_goodness.

    Parameters
    ----------
    firing_rate_dict : dict
        Dictionary mapping population names to their firing rates in Hz.

    Returns
    -------
    float
        The calculated loss value.
    """
    # D1 and D2 extracted from: (Liang et al., 2008) using with levodopa treatment, see experimental_data/activity_striatum/extract_from_liang_etal_2008.py
    # FS: 10 Hz based on: (Yamada et al., 2016; Marche und Apicella, 2021; Adler et al., 2013; Hernandez et al., 2013; He et al., 2024)
    # stn and snr (gpi): from [Li et al., 2015]
    plausible_ranges = {
        "str_d1": (20.45, 53.69),
        "str_d2": (12.99, 45.15),
        "str_fsi": (5.0, 15.0),
        "gpe_proto": (75.0, 85.0),
        "gpe_arky": (15.0, 20.0),
        "gpe_cp": (75.0, 85.0),
        "stn": (28.0, 80.0),
        "snr": (21.0, 93.0),
        "thal": (15.0, 30.0),
    }
    loop_plausible_ranges = {}
    for loop in ["caudate", "putamen"]:
        for pop_name, bounds in plausible_ranges.items():
            loop_plausible_ranges[f"{pop_name}:{loop}"] = bounds

    goodness_scores = []
    k_sharpness = 4.0  # larger = steeper penalty once outside the band
    for pop_name, (lower, upper) in loop_plausible_ranges.items():
        fr = firing_rate_dict[pop_name]  # crashes if pop_name not found

        center = 0.5 * (lower + upper)
        half_width = max(0.5 * (upper - lower), 1e-6)
        rel_dev = abs(fr - center) / half_width  # =1 at band edge

        x = k_sharpness * (rel_dev - 1.0)
        # Logistic goodness: stable evaluation avoids overflow for large x
        goodness = float(np.exp(-np.logaddexp(0.0, x)))
        goodness_scores.append(goodness)

    mean_goodness = float(np.mean(goodness_scores))
    # loss in [0, 1]; 0 when all within range, increasing as rates drift away
    return 1.0 - mean_goodness


def plot_firing_rate_loss(
    lower: float,
    upper: float,
    k_sharpness: float = 4.0,
    fr_min: float | None = None,
    fr_max: float | None = None,
    n_points: int = 400,
):
    """Visualize loss vs. firing rate for a plausible range.

    Uses the same logistic goodness as ``get_firing_rate_loss`` (per-pop),
    plotting loss = 1 - goodness across a rate grid.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        print("matplotlib is required for plotting this demo:", exc)
        return

    center = 0.5 * (lower + upper)
    half_width = max(0.5 * (upper - lower), 1e-6)

    if fr_min is None:
        fr_min = max(0.0, lower - 1.5 * half_width)
    if fr_max is None:
        fr_max = upper + 1.5 * half_width

    x = np.linspace(fr_min, fr_max, n_points)
    rel_dev = np.abs(x - center) / half_width
    goodness = 1.0 / (1.0 + np.exp(k_sharpness * (rel_dev - 1.0)))
    loss = 1.0 - goodness

    plt.figure(figsize=(6, 4))
    plt.plot(x, loss, label="loss (1 - goodness)")
    plt.axvspan(lower, upper, color="green", alpha=0.15, label="plausible range")
    plt.axvline(lower, color="green", linestyle="--", linewidth=1)
    plt.axvline(upper, color="green", linestyle="--", linewidth=1)
    plt.xlabel("Firing rate (Hz)")
    plt.ylabel("Loss")
    plt.ylim(-0.05, 1.05)
    plt.title("Firing-rate loss vs. rate")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # example usage:
    # first only do compilation with appendix:
    #  python get_loss.py --dbs on --compile --compile-appendix test
    # then run with 21 parameters using the same appendix:
    #  python get_loss.py --dbs on --compile-appendix test 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.1
    parser = argparse.ArgumentParser(
        description=(
            "Run BOLD optimization with 21 optimization parameters supplied on the command line "
            "(values mapped in order to set_opt_params)."
        )
    )
    parser.add_argument(
        "--dbs",
        type=str,
        default=paramsS.get("dbs", "on"),
        help="DBS condition string (e.g., 'on' or 'off').",
    )
    parser.add_argument(
        "params",
        metavar="P",
        nargs="*",
        type=float,
        help="21 parameter values in the exact order expected by set_opt_params.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the model and exit without running simulations.",
    )
    parser.add_argument(
        "--compile-appendix",
        type=str,
        default="",
        help="Optional suffix for the ANNarchy compile folder (e.g., run tag).",
    )
    args = parser.parse_args()
    dbs_condition = args.dbs
    param_list = args.params

    # print(f"DBS condition: {dbs_condition}")
    # if param_list:
    #     print(f"Parameter list: {param_list}")
    # print(f"compile only: {args.compile}")
    # if args.compile_appendix:
    #     print(f"compile folder appendix: {args.compile_appendix}")

    ### SETUP TIMESTEP + SEED ###
    if paramsS["seed"] == None:
        setup(dt=paramsS["timestep"])
    else:
        setup(dt=paramsS["timestep"], seed=paramsS["seed"])

    ### Obtain the maximum simulation time from the cortical rate files if needed
    if "t.duration" not in paramsS or paramsS["t.duration"] is None:
        paramsS["t.duration"], mixed_rates = infer_max_sim_time_ms(
            dbs_condition=dbs_condition,
            cortical_rate_path=paramsS["mc.cortical_rate_path"],
            dt_ms=paramsS["timestep"],
        )
    else:
        _, mixed_rates = infer_max_sim_time_ms(
            dbs_condition=dbs_condition,
            cortical_rate_path=paramsS["mc.cortical_rate_path"],
            dt_ms=paramsS["timestep"],
        )

    ### CREATE THE TWO LOOP MODEL ###
    # loop for BG loops
    compilation_appendix = args.compile_appendix
    if compilation_appendix:
        compile_folder = f"bgm_v08_{dbs_condition}_{compilation_appendix}"
    else:
        compile_folder = f"bgm_v08_{dbs_condition}"
    model_dict = {}
    for loop in ["caudate", "putamen"]:
        ### Prepare the model_creation kwargs for the current dbs condition
        # - "input.rates": array with (steps, post_size) shape containing the input rates for each time step
        # - "input.schedule": a single scalar with the schedule time in ms for updating the input rates
        # - "timestep": simulation timestep in ms
        model_creation_kwargs = {
            "input.rates": mixed_rates[loop]["rate"],
            "input.schedule": 2.31 * 1000,
            "timestep": paramsS["timestep"],
        }

        ### Create model for the current loop
        model_dict[loop] = BGM(
            name="BGM_v08_p01",
            model_creation_kwargs=model_creation_kwargs,
            seed=paramsS["seed"],
            compile_folder_name=compile_folder,
            name_appendix=loop,
            do_create=True,
            do_compile=False,
        )

    ### DBS SIMULATOR ###
    # parameters 21, 22, 23 are used for DBS if dbs_condition is "on"
    if dbs_condition == "on":
        dbs_stimulator = DBSstimulator(
            stimulated_population=get_population("stn:putamen"),
            # VTA from berlin data subject 1:
            population_proportion=(35 + 23) / (70 + 75),
            # exclude all populations containing "TimedInput" in their name:
            excluded_populations_list=[
                "TimedInput_cortex:caudate",
                "TimedInput_cortex:putamen",
            ],
            # the dbs_depolarization parameter actually reduces the membrane potential
            # so its actually a hyperpolarization
            dbs_depolarization=param_list[21],  # [0,10] like weight/conductance in stn
            orthodromic=True,
            antidromic=True,
            efferents=True,
            afferents=True,
            passing_fibres=True,
            # snr__thal is actually gpi__thal, pasing fibre based on Miocinovic et al. 2006
            passing_fibres_list=[get_projection("snr__thal:putamen")],
            passing_fibres_strength=param_list[22],  # [0,1] scale between 0 and 1
            dbs_pulse_frequency_Hz=125,  # from berlin data subject 1
            # pulse width needs to be multiple of timestep (0.1 ms --> min 100 us)
            # pulse width in berlin data is 60 us but we use 100 us here
            dbs_pulse_width_us=100,
            axon_spikes_per_pulse=param_list[
                23
            ],  # [0,1] max 1 spike per pulse(=timestep)
            seed=paramsS["seed"],
            auto_implement=True,
            model=model_dict["putamen"],
        )

    ### BOLD MONITORING ###
    # create BOLD monitors for the following regions:
    # BOLD : model
    # GPi : snr [both loops]
    # GPe : gpe_proto, gpe_arky, gpe_cp [both loops]
    # STN : stn [both loops]
    # Cau : str_d1, str_d2, str_fsi [caudate loop]
    # Put : str_d1, str_d2, str_fsi [putamen loop]
    # MD : thal [caudate loop]
    # VAp : thal [putamen loop]
    bold_region_dict: dict[str, list[str]] = {
        "GPi": ["snr:caudate", "snr:putamen"],
        "GPe": [
            "gpe_proto:caudate",
            "gpe_arky:caudate",
            "gpe_cp:caudate",
            "gpe_proto:putamen",
            "gpe_arky:putamen",
            "gpe_cp:putamen",
        ],
        "STN": ["stn:caudate", "stn:putamen"],
        "Cau": ["str_d1:caudate", "str_d2:caudate", "str_fsi:caudate"],
        "Put": ["str_d1:putamen", "str_d2:putamen", "str_fsi:putamen"],
        "MD": ["thal:caudate"],
        "VAp": ["thal:putamen"],
    }
    # calculate scaling factors for the different GPe populations
    gpe_proportions = {"gpe_proto": 0.5, "gpe_arky": 0.17, "gpe_cp": 0.10}
    gpe_scaling_factors = [
        gpe_proportions[key.split(":")[0]] for key in bold_region_dict["GPe"]
    ]
    gpe_scaling_factors = np.array(gpe_scaling_factors) / np.sum(gpe_scaling_factors)
    # scaling factors for striatum proportions (del Rey et al. 2022)
    # Cau and Put have the same proportions
    props_delRey = {"str_d1": 0.86 / 2, "str_d2": 0.86 / 2, "str_fsi": 0.026}
    str_scaling_factors = [
        props_delRey[key.split(":")[0]] for key in bold_region_dict["Cau"]
    ]
    str_scaling_factors = np.array(str_scaling_factors) / np.sum(str_scaling_factors)

    # create the BoldMonitor objects
    bold_monitor_dict: dict[str, BoldMonitor] = {}
    for bold_region, population_names in bold_region_dict.items():
        if bold_region == "GPe":
            scale_factors = gpe_scaling_factors.tolist()
        elif bold_region in ["Cau", "Put"]:
            scale_factors = str_scaling_factors.tolist()
        else:
            scale_factors = None

        # for BGM pops input variable = I
        # for Microcircuit pops input variable = I_v
        if bold_region in ["Cau", "Put"]:
            input_var = "I_v"
        else:
            input_var = "I"

        # get populations from population names
        populations = [get_population(pop_name) for pop_name in population_names]

        bold_monitor_dict[bold_region] = BoldMonitor(
            populations=populations,
            mapping={"I_CBF": input_var},
            normalize_input=2000,
            scale_factor=scale_factors,
            start=False,
        )

    ### COMPILE ###
    ### Compile model (i.e. both loops in a single model) afterwards we are ready to simulate
    model_dict["caudate"].compile()

    if args.compile:
        print("Compilation completed; skipping simulations (--compile).")
        sys.exit(0)

    ### ACTIVATE DBS ###
    if dbs_condition == "on":
        dbs_stimulator.on()

    ### MONITORS ###
    ### create monitors to record the spikes from all populations
    # first collect the populations to monitor, simply self.populations of both loops without pops starting with "TimedInput"
    pops_to_monitor = []
    for loop in ["caudate", "putamen"]:
        bgm_model: BGM = model_dict[loop]
        for pop_name in bgm_model.populations:
            if not pop_name.startswith("TimedInput"):
                pops_to_monitor.append(pop_name)
    # create the monitor dictionary with variables to record
    monitor_dictionary = {pop_name: ["spike"] for pop_name in pops_to_monitor}
    monitors = CompNeuroMonitors(monitor_dictionary)

    ### EXPERIMENT ###
    ### Define a CompNeuroPy experiment to run the model 10 s and obtain the spike recordings
    experiment = Spikes10s(
        monitors=monitors, model_dict=model_dict, seed=paramsS["seed"]
    )

    ### SIMULATIONS ###

    ### SIMULATION FOR RATES: ###
    firing_rate_dict = get_firing_rate_10s(param_list=param_list, experiment=experiment)
    # obtain loss based on firing rates, between 0 and 1
    firing_rate_loss = get_firing_rate_loss(firing_rate_dict)

    ### SIMULATION FOR BOLD: ###
    bold_data = get_BOLD_full(
        model_dict=model_dict,
        seed=paramsS["seed"],
        bold_monitor_dict=bold_monitor_dict,
        param_list=param_list,
    )
    # obtain loss based on BOLD correlation, between 0 and 1
    bold_loss, per_region_corr = compute_bold_correlation_loss(
        sim_bold=bold_data,
        condition=dbs_condition,
        tr_s=2.31,
        ramp_up_ms=paramsS["t.rampup"],
        data_file=None,
        region_map=None,
        dt_ms=paramsS["timestep"],
    )

    ### TOTAL LOSS ###
    total_loss = float(firing_rate_loss + bold_loss)
    loss_payload = {
        "total_loss": total_loss,
    }

    loss_folder = Path(__file__).resolve().parent / paramsS["data_folder"]
    loss_folder.mkdir(parents=True, exist_ok=True)
    appendix_tag = compilation_appendix if compilation_appendix else "default"
    loss_file = loss_folder / f"loss_{appendix_tag}.json"
    with open(loss_file, "w", encoding="ascii") as f:
        json.dump(loss_payload, f, ensure_ascii=True, indent=2)
