from ANNarchy import (
    setup,
    get_population,
    set_seed,
    reset,
    simulate,
    get_projection,
    populations,
    TimedArray,
    CurrentInjection,
    report,
)
from ANNarchy.extensions.bold import BoldMonitor
from CompNeuroPy.full_models import BGM
from CompNeuroPy import (
    CompNeuroMonitors,
    CompNeuroExp,
    DBSstimulator,
    add_dbs_mechanisms,
)
import argparse
import json
import numpy as np
from pathlib import Path
import h5py
import sys

### local
from parameters import parameters_test_microcircuit as paramsS

### Repetition time of the experimental BOLD data (s). The cortical drive is
### sampled at this rate, one value per TR, and the simulated BOLD is recorded
### on the same grid.
TR_S = 2.31

### The two BG loops simulated side by side, each driven by its own cortical mix
LOOPS = ("caudate", "putamen")

### The DBS footprint: the populations and projections DBS can actually reach, and
### the only ones the DBS equation terms are added to. See DBS.md.
###   stn        - the stimulated population
###   snr, gpe_* - the postsynaptic targets of its efferents
###   gpe_proto  - also its only spiking afferent
###   thal       - the target of the snr__thal passing fibre
### Everything else is left alone: the caudate loop is excluded from every DBS
### effect by design, the striatal microcircuit is neither afferent nor efferent to
### STN (so its terms would always be zero while costing two RNG draws per neuron
### per step on the largest population), and the TimedArray/CurrentInjection input
### machinery cannot carry them at all.
DBS_POPULATION_NAMES = ("stn", "snr", "gpe_proto", "gpe_arky", "gpe_cp", "thal")
DBS_PROJECTION_NAMES = (
    "stn__snr",
    "stn__gpe_proto",
    "stn__gpe_arky",
    "stn__gpe_cp",
    "gpe_proto__stn",
    "snr__thal",
)
### DBS is applied to the putamen loop only
DBS_LOOP = "putamen"

### Model versions this script can drive.
### - v07 is the full model: the striatum comes from the Microcircuit class
###   (distance-dependent connectivity, correlated cortical input, missing-GABA
###   compensation) and thal/gpe_arky/gpe_cp/stn get CorticalInputs. Its striatal
###   populations are created by Microcircuit and keep names like "caudate_dSPN".
### - v08 is the reduced model: plain BGM populations whose cortical drive is a
###   single TimedArray scaled by exp_input_weight. Fast, no input caches, and
###   useful as an end-to-end smoke test of this pipeline.
MODEL_VERSIONS = ("v07", "v08")


def striatal_pop_name(model_version: str, loop: str, compartment: str) -> str:
    """Population name of a striatal compartment for the given model version.

    compartment is one of "str_d1", "str_d2", "str_fsi".
    """
    if model_version == "v07":
        mc_name = {"str_d1": "dSPN", "str_d2": "iSPN", "str_fsi": "FS"}[compartment]
        return f"{loop}_{mc_name}"
    return f"{compartment}:{loop}"


def bg_pop_name(loop: str, compartment: str) -> str:
    """Population name of a non-striatal BG compartment (same in both versions)."""
    return f"{compartment}:{loop}"


def population_name(model_version: str, loop: str, compartment: str) -> str:
    """Population name of any compartment, dispatching on where it was created."""
    if compartment in ("str_d1", "str_d2", "str_fsi"):
        return striatal_pop_name(model_version, loop, compartment)
    return bg_pop_name(loop, compartment)


### Projection weight clusters. Instead of freeing every weight, the literature
### values already in BGM.params are scaled by one factor per functional cluster,
### which preserves their relative balance and keeps the search well conditioned.
### v08 adds the intra-striatal projections; in v07 those live inside the
### Microcircuit with weights sampled from the fitted connectivity, so they are
### not model parameters here.
PROJ_CLUSTERS_COMMON = {
    "str_d1__bg": ["str_d1__snr", "str_d1__gpe_cp"],
    "str_d2__bg": ["str_d2__gpe_proto", "str_d2__gpe_arky", "str_d2__gpe_cp"],
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

PROJ_CLUSTERS_V08_ONLY = {
    "str_fsi__striatum": ["str_fsi__str_d1", "str_fsi__str_d2", "str_fsi__str_fsi"],
    "str_laterals": [
        "str_d1__str_d1",
        "str_d1__str_d2",
        "str_d2__str_d1",
        "str_d2__str_d2",
    ],
}


def proj_clusters(model_version: str) -> dict:
    """Ordered weight clusters for a model version; index in this order = parameter order."""
    if model_version == "v08":
        return {**PROJ_CLUSTERS_COMMON, **PROJ_CLUSTERS_V08_ONLY}
    return dict(PROJ_CLUSTERS_COMMON)


def n_opt_params(model_version: str) -> int:
    """Number of optimized parameters: 9 drive parameters + one per weight cluster."""
    return 9 + len(proj_clusters(model_version))


def _set_cluster_weights(param_list, model_dict, model_version, offset=9, loops=LOOPS):
    """Scale the literature weights of every projection by its cluster's factor."""
    clusters = proj_clusters(model_version)
    scaling = {name: param_list[offset + i] for i, name in enumerate(clusters)}

    for loop in loops:
        bgm_model: BGM = model_dict[loop]
        for cluster_name, proj_name_list in clusters.items():
            for proj_name in proj_name_list:
                weight_val_orig = bgm_model.params[f"{proj_name}:{loop}.weights"]
                bgm_model.set_param(
                    compartment=f"{proj_name}:{loop}",
                    parameter_name="w",
                    parameter_value=weight_val_orig * scaling[cluster_name],
                )


def set_opt_params_v07(param_list, model_dict):
    """
    Set the optimized parameters of the v07 model (19 parameters).

    0-2  cortical input weights for dSPN, iSPN, FS (set on the Microcircuit)
    3-6  cortical input weights for thal, gpe_arky, gpe_cp, stn (set on CorticalInputs)
    7-8  baseline currents for snr and gpe_proto, which receive no cortical input
    9+   one scaling factor per projection weight cluster
    """
    str_input_weights = {
        "dSPN": param_list[0],
        "iSPN": param_list[1],
        "FS": param_list[2],
    }
    ci_input_weights = {
        "thal": param_list[3],
        "gpe_arky": param_list[4],
        "gpe_cp": param_list[5],
        "stn": param_list[6],
    }
    base_current_dict = {"snr": param_list[7], "gpe_proto": param_list[8]}

    for loop in LOOPS:
        bgm_model: BGM = model_dict[loop]

        # striatal cortical drive lives in the Microcircuit; only the cortical
        # keys are touched, the local (pre_type, post_type) entries hold the
        # sampled intrinsic weights and must stay as they are
        mc = bgm_model.mc
        for post_type, weight in str_input_weights.items():
            for cortical_region in mc.cortical_proportions_dict:
                mc.mean_weights_by_type[(cortical_region, post_type)] = weight

        # cortical drive of the remaining BG populations
        ci = bgm_model.ci
        for post_type, weight in ci_input_weights.items():
            for cortical_region in ci.cortical_proportions_dict:
                ci.mean_weights_by_type[(cortical_region, post_type)] = weight

        # snr and gpe_proto have no cortical input, so their drive is a baseline current
        for compartment_name, param_val in base_current_dict.items():
            bgm_model.set_param(
                compartment=f"{compartment_name}:{loop}",
                parameter_name="base_mean",
                parameter_value=param_val,
            )

    _set_cluster_weights(param_list, model_dict, "v07")


def set_opt_params_v08(param_list, model_dict):
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
    _set_cluster_weights(param_list, model_dict, "v08")


def set_opt_params(param_list, model_dict, model_version: str):
    """Dispatch to the parameter mapping of the requested model version.

    param_list holds the base parameters only; split_param_list separates those
    from the DBS parameters and the optional putamen-only weight overrides.
    """
    if model_version == "v07":
        set_opt_params_v07(param_list, model_dict)
    else:
        set_opt_params_v08(param_list, model_dict)


def apply_opt_params(
    param_list, model_dict, model_version: str, putamen_cluster_scalings=None
):
    """Set the base parameters on both loops, then any putamen-only overrides."""
    set_opt_params(param_list, model_dict, model_version)
    if putamen_cluster_scalings is not None:
        _set_cluster_weights(
            putamen_cluster_scalings,
            model_dict,
            model_version,
            offset=0,
            loops=("putamen",),
        )


def split_param_list(param_list, model_version: str, dbs_condition: str):
    """Split the command-line parameter vector into its three parts.

    Returns (base_params, putamen_cluster_scalings, dbs_params). The layouts are

        off : base                                    [+ 3 ignored DBS slots]
        on  : base                                    + 3 DBS
        on  : base + one scaling per weight cluster   + 3 DBS   (staged)

    where base is 19 values for v07 and 21 for v08. The staged layout is the one
    the inference calls for: caudate is excluded from every DBS effect and shares
    no projection with putamen, so refitting its weights under DBS would assert a
    mechanism the model does not contain. The base vector, carried over from the
    DBS-off fit, therefore sets both loops and the extra block then re-scales the
    putamen weights only. The layouts differ in length, so they cannot be
    confused for one another.
    """
    n_base = n_opt_params(model_version)
    n_clusters = len(proj_clusters(model_version))
    n_given = len(param_list)

    if dbs_condition == "off":
        # the 3 DBS slots are accepted so one caller can pad every vector alike
        if n_given not in (n_base, n_base + 3):
            raise ValueError(
                f"DBS off with model {model_version} takes {n_base} parameters "
                f"(optionally followed by 3 ignored DBS slots), got {n_given}."
            )
        return list(param_list[:n_base]), None, None

    if n_given == n_base + 3:
        return list(param_list[:n_base]), None, list(param_list[-3:])
    if n_given == n_base + n_clusters + 3:
        return (
            list(param_list[:n_base]),
            list(param_list[n_base : n_base + n_clusters]),
            list(param_list[-3:]),
        )
    raise ValueError(
        f"DBS on with model {model_version} takes {n_base + 3} parameters "
        f"(base + 3 DBS) or {n_base + n_clusters + 3} (base + {n_clusters} "
        f"putamen weight scalings + 3 DBS), got {n_given}."
    )


def infer_max_sim_time_ms(
    dbs_condition: str,
    cortical_rate_path=paramsS["mc.cortical_rate_path"],
    dt_ms: float = 0.1,
    n_trs: int | None = None,
):
    """Load MATLAB cortical firing rates for a DBS condition and infer max simulation time (ms).

    The firing-rate arrays are sampled at the BOLD TR (2.31 s) without upsampling,
    so one sample covers one TR and the total duration is len(rates) * TR.

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

    # Optionally shorten the run (smoke tests); the experimental comparison
    # trims to the shared length, so a shorter simulation stays aligned.
    if n_trs is not None:
        caudate_rate = caudate_rate[:n_trs]
        putamen_rate = putamen_rate[:n_trs]
        if caudate_time is not None:
            caudate_time = caudate_time[:n_trs]
        if putamen_time is not None:
            putamen_time = putamen_time[:n_trs]

    # One rate sample covers one TR, so the run lasts len(rates) * TR.
    duration_ms = len(caudate_rate) * TR_S * 1000.0

    mixed_rates = {
        "caudate": {"time": caudate_time, "rate": caudate_rate},
        "putamen": {"time": putamen_time, "rate": putamen_rate},
    }

    return duration_ms, mixed_rates


def v07_model_creation_kwargs(
    loop: str,
    dbs_condition: str,
    duration_ms: float,
    cache_dir: str | None = None,
    build_caches: bool = False,
):
    """model_creation_kwargs for BGM_v07: the Microcircuit and CorticalInputs setup.

    build_caches=True regenerates the precomputed spike counts instead of loading
    them; that is what build_input_caches.py does. Every evaluation loads them,
    so both paths have to agree on all of these values -- in particular the
    caches are only accepted when their n_steps equals int(duration_ms / dt),
    and the stored cortical_rate_path string must match exactly.
    """
    cache_dir = cache_dir if cache_dir is not None else paramsS["mc_ci_cache_dir"]
    return {
        "build_mc": build_caches,
        "build_ci": build_caches,
        "mc.name": loop,
        "mc.nx": paramsS["mc.nx"],
        "mc.b": paramsS["mc.b"],
        "mc.firing_rate_dict": paramsS["mc.firing_rate_dict"],
        "dbs": dbs_condition,
        "timestep": paramsS["timestep"],
        "t.duration": duration_ms,
        "update_time": paramsS["update_time"],
        "mc.storage_dir": f"{cache_dir}/mc_{loop}_cache_{dbs_condition}",
        "mc.seed": paramsS["seed"],
        "mc.fitted_params_path": paramsS["mc.fitted_params_path"],
        "mc.cortical_rate_path": paramsS["mc.cortical_rate_path"][dbs_condition],
        "ci.storage_dir": f"{cache_dir}/ci_{loop}_cache_{dbs_condition}",
        "ci.seed": paramsS["seed"],
        "ci.n_thal": paramsS["ci.n_thal"],
        "ci.n_gpe_arky": paramsS["ci.n_gpe_arky"],
        "ci.n_gpe_cp": paramsS["ci.n_gpe_cp"],
        "ci.n_stn": paramsS["ci.n_stn"],
    }


def update_TimedInput(bgm_model: BGM):
    """Rewind the cortical TimedArray so the next simulation starts at its first block.

    reset() restores rates, schedule and period to their construction values and
    zeroes the internal timers.
    """
    inp = get_population(f"TimedInput_cortex:{bgm_model.name_appendix}")
    inp.reset()


def rewind_inputs(model_dict, model_version: str):
    """Rewind the cortical input streams so the next simulation starts at t=0.

    v07 streams precomputed spike counts through Microcircuit/CorticalInputs
    iterators, v08 replays a single TimedArray per loop.
    """
    for loop in LOOPS:
        bgm_model: BGM = model_dict[loop]
        if model_version == "v07":
            bgm_model.mc.reset()
            bgm_model.ci.reset()
        else:
            update_TimedInput(bgm_model)


def simulate_model(model_dict, model_version: str, duration_ms: float):
    """Simulate for duration_ms, refreshing the v07 input streams as needed.

    v07 can only be simulated in steps of update_time, because the inputs are
    handed to ANNarchy one chunk at a time. duration_ms must be a multiple of it.
    """
    if model_version != "v07":
        simulate(duration_ms)
        return

    update_functions = []
    for loop in LOOPS:
        update_functions.append(model_dict[loop].mc.update)
        update_functions.append(model_dict[loop].ci.update)

    update_time = model_dict[LOOPS[0]].mc.update_time
    n_updates = int(round(duration_ms / update_time))
    if not np.isclose(n_updates * update_time, duration_ms):
        raise ValueError(
            f"v07 simulation time {duration_ms} ms is not a multiple of the input "
            f"update_time {update_time} ms."
        )

    for _ in range(n_updates):
        # refresh every input stream, then let the last call run the simulation
        for update_function in update_functions[:-1]:
            update_function(run_simulation=False)
        update_functions[-1](run_simulation=True)


def assert_dbs_state(dbs_stimulator, dbs_condition: str, where: str):
    """Check that the model is really in the DBS state the run claims.

    The failure this guards against is silent: on() writes plain parameters, and a
    reset() that restores pop.init would put them back to zero without raising.
    The evaluation would then run without DBS and still record "dbs": "on" in the
    loss file. Call this immediately before anything is simulated.
    """
    stim_pop = get_population(f"stn:{DBS_LOOP}")
    expected_on = (
        dbs_stimulator.dbs_on_array.flatten()
        if dbs_condition == "on"
        else np.zeros(stim_pop.size)
    )
    actual_on = np.asarray(stim_pop.dbs_on).flatten()
    if not np.array_equal(actual_on, expected_on):
        raise AssertionError(
            f"{where}: dbs_on on stn:{DBS_LOOP} is not what DBS '{dbs_condition}' "
            f"requires (sum {actual_on.sum()} vs expected {expected_on.sum()})"
        )

    expected_depol = dbs_stimulator.dbs_depolarization if dbs_condition == "on" else 0.0
    if not np.isclose(stim_pop.dbs_depolarization, expected_depol):
        raise AssertionError(
            f"{where}: dbs_depolarization is {stim_pop.dbs_depolarization}, "
            f"expected {expected_depol}"
        )

    ### the caudate loop is the free control: it must never carry a DBS effect
    for pop_name in DBS_POPULATION_NAMES:
        caudate_pop = get_population(f"{pop_name}:caudate")
        if "dbs_on" in caudate_pop.attributes:
            raise AssertionError(
                f"{where}: {pop_name}:caudate carries DBS mechanisms, but the "
                "caudate loop is supposed to be free of them"
            )


class Spikes10s(CompNeuroExp):
    def __init__(
        self,
        monitors: CompNeuroMonitors = None,
        model_dict: dict = None,
        seed: int = 42,
        model_version: str = "v08",
        putamen_cluster_scalings=None,
    ):
        super().__init__(monitors)
        self.model_dict = model_dict
        self.seed = seed
        self.model_version = model_version
        self.putamen_cluster_scalings = putamen_cluster_scalings

    def run(self, param_list: list):
        # at the begining always reset (also annarchy random!) and start monitors
        self.reset()
        self.monitors.start()
        set_seed(self.seed)

        rewind_inputs(self.model_dict, self.model_version)

        # set optimized parameters
        apply_opt_params(
            param_list,
            self.model_dict,
            self.model_version,
            self.putamen_cluster_scalings,
        )

        duration_ms = paramsS["t.firing_rate_sim"]
        simulate_model(self.model_dict, self.model_version, duration_ms)

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

        recorded_pop = key.split(";")[0]
        n_neurons = len(spikes)
        if n_neurons == 0:
            firing_rates[recorded_pop] = 0.0
            continue

        # flatten all spike times from all neurons
        all_spike_times = (
            np.concatenate(list(spikes.values())) if len(spikes) > 0 else np.array([])
        )
        eval_spike_times = all_spike_times[
            all_spike_times >= (1 - eval_proportion) * max_steps
        ]
        firing_rate_hz = len(eval_spike_times) / (n_neurons * eval_time_s)
        firing_rates[recorded_pop] = firing_rate_hz

    return firing_rates


def get_BOLD_full(
    model_dict,
    seed,
    bold_monitor_dict,
    param_list: list,
    model_version: str,
    putamen_cluster_scalings=None,
):
    ### RESETS AND PREPARE ### TODO: how to reset BOLD monitor - skipped this just run scripts separately
    # reset ANNarchy and seed
    reset()
    set_seed(seed)

    rewind_inputs(model_dict, model_version)

    ### OPTIMIZED PARAMETERS ###
    # set optimized parameters
    apply_opt_params(param_list, model_dict, model_version, putamen_cluster_scalings)

    ### RAMP UP ###
    # initial ramp up
    ramp_up_duration_ms = paramsS["t.rampup"]
    simulate_model(model_dict, model_version, ramp_up_duration_ms)

    # start each bold monitor
    for bold_monitor in bold_monitor_dict.values():
        bold_monitor.start()

    ### REST OF SIMULATION ###
    simulate_model(
        model_dict, model_version, paramsS["t.duration"] - ramp_up_duration_ms
    )

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


def compute_bold_correlation_loss(
    sim_bold: dict[str, np.ndarray],
    condition: str = "on",
    tr_s: float = TR_S,
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

        # The BOLD monitors record on the TR grid, so no downsampling is needed.
        sim_series = np.asarray(sim_bold[sim_region]).ravel()
        exp_series = np.asarray(exp_bold[exp_label])

        exp_trimmed = exp_series[ramp_up_steps:] if ramp_up_steps > 0 else exp_series

        n = min(len(sim_series), len(exp_trimmed))
        if n < 2:
            correlations[sim_region] = float("nan")
            continue

        correlations[sim_region] = _safe_corr(sim_series[:n], exp_trimmed[:n])

    valid_corrs = [c for c in correlations.values() if not np.isnan(c)]
    if not valid_corrs:
        return 1.0, correlations

    # Map mean correlation in [-1, 1] to goodness in [0, 1], then convert to loss.
    mean_corr = float(np.mean(valid_corrs))
    goodness = float(np.clip((mean_corr + 1.0) / 2.0, 0.0, 1.0))
    loss = 1.0 - goodness  # loss=0 at perfect (1.0), loss=1 at worst (-1.0)
    return loss, correlations


def get_firing_rate_loss(
    firing_rate_dict: dict[str, float], model_version: str = "v08"
) -> float:
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
    # str_d1/str_d2: mean +- 1 SD of the parkinsonian Off state (levodopa withdrawn)
    # in (Liang et al., 2008), Table 1. The centres are exactly the rates the
    # missing-GABA caches are drawn at (parameters.py, "mc.firing_rate_dict"), so
    # change the two together. Derivation and caveats:
    # ../experimental_data/activity_striatum/README.md
    # FS: 10 Hz based on: (Yamada et al., 2016; Marche und Apicella, 2021; Adler et al., 2013; Hernandez et al., 2013; He et al., 2024)
    # stn and snr (gpi): from [Li et al., 2015]
    plausible_ranges = {
        "str_d1": (12.67, 37.33),
        "str_d2": (21.22, 44.78),
        "str_fsi": (5.0, 15.0),
        "gpe_proto": (75.0, 85.0),
        "gpe_arky": (15.0, 20.0),
        "gpe_cp": (75.0, 85.0),
        "stn": (28.0, 80.0),
        "snr": (21.0, 93.0),
        "thal": (15.0, 30.0),
    }
    # resolve to the actual population names, which differ between model versions
    # for the striatum (v07 gets those populations from the Microcircuit)
    loop_plausible_ranges = {}
    for loop in LOOPS:
        for compartment, bounds in plausible_ranges.items():
            loop_plausible_ranges[
                population_name(model_version, loop, compartment)
            ] = bounds

    goodness_scores = []
    k_sharpness = 4.0  # larger = steeper penalty once outside the band
    for pop_name_, (lower, upper) in loop_plausible_ranges.items():
        fr = firing_rate_dict[pop_name_]  # crashes if pop_name_ not found

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


def add_TimedInputs(model_creation_kwargs, params, loop_name):

    ### TimedInputs + CurrentÜrojections ###
    inputs = model_creation_kwargs["input.rates"]
    # inputs is currently shaped (steps,) and needs to be reshaped into (steps, post_pop_size) so post_pop_size times the same input
    inputs = np.repeat(
        inputs[:, np.newaxis], repeats=params[f"str_d1:{loop_name}.size"], axis=1
    )
    schedule = model_creation_kwargs["input.schedule"]
    inp = TimedArray(
        rates=inputs,
        schedule=schedule,
        name=f"TimedInput_cortex:{loop_name}",
    )
    # create current projections
    for pop_name in [
        f"str_d1:{loop_name}",
        f"str_d2:{loop_name}",
        f"str_fsi:{loop_name}",
        f"thal:{loop_name}",
        f"gpe_arky:{loop_name}",
        f"gpe_cp:{loop_name}",
        f"stn:{loop_name}",
    ]:
        proj = CurrentInjection(
            pre=inp,
            post=get_population(pop_name),
            target="cor",
            name=f"{inp.name}__{pop_name}",
        )
        proj.connect_current()


if __name__ == "__main__":
    # example usage:
    # first only do compilation with appendix:
    #  python get_loss.py --dbs on --compile --compile-appendix test
    # then run with 21 parameters using the same appendix:
    #  python get_loss.py --dbs on --compile-appendix test 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.1
    # python get_loss.py --compile --dbs on --compile-appendix test 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one parameter vector: simulate the firing rates, then (unless "
            "gated) the BOLD, and write the loss and its components to a JSON file. "
            "See split_param_list for the accepted vector layouts."
        )
    )
    parser.add_argument(
        "--dbs",
        type=str,
        required=True,
        choices=["on", "off"],
        help="DBS condition. Required: silently defaulting here means a whole "
        "optimization can run against the wrong condition.",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="v07",
        choices=list(MODEL_VERSIONS),
        help="Which BGM model to simulate. v07 is the full model with the "
        "striatal microcircuit; v08 is the reduced model used as a fast "
        "end-to-end test of this pipeline.",
    )
    parser.add_argument(
        "params",
        metavar="P",
        nargs="*",
        type=float,
        help="Optimized parameter values, in the order expected by set_opt_params "
        "for the chosen model version (19 for v07, 21 for v08). With --dbs on, "
        "followed by the 3 DBS parameters, and optionally by one putamen-only "
        "weight scaling per cluster in between -- see split_param_list.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the model and exit without running simulations.",
    )
    parser.add_argument(
        "--n-trs",
        type=int,
        default=None,
        help="Use only the first N TRs of the cortical drive. For smoke tests; "
        "the full run uses all 310.",
    )
    parser.add_argument(
        "--gate-threshold",
        type=float,
        default=paramsS["firing_rate_gate"],
        help="Skip the expensive BOLD run when the firing-rate loss exceeds this "
        "and charge the worst BOLD loss (1.0) instead. Both losses are in [0, 1], "
        "so 1.0 disables the gate.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Where the v07 input caches live (see build_input_caches.py). "
        "Defaults to mc_ci_cache_dir from parameters.py. Ignored for v08.",
    )
    parser.add_argument(
        "--compile-appendix",
        type=str,
        default="",
        help="Optional suffix for the ANNarchy compile folder (e.g., run tag).",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Write an ANNarchy report of the compiled network to this path and "
        "exit. The filename must end in .md or .tex. Documents what this script "
        "actually builds -- populations, projections, neuron and synapse models "
        "including the DBS retrofit -- rather than what the source suggests. "
        "Implies --compile.",
    )
    args = parser.parse_args()
    dbs_condition = args.dbs
    model_version = args.model_version

    # --compile is run with a placeholder vector whose length nobody guarantees,
    # and nothing is simulated, so only a real evaluation validates the layout.
    # --report simulates nothing either, and is meant to be callable without a
    # parameter vector at all.
    if args.compile or args.report:
        param_list, putamen_cluster_scalings, dbs_params = list(args.params), None, None
    else:
        param_list, putamen_cluster_scalings, dbs_params = split_param_list(
            args.params, model_version, dbs_condition
        )

    ### SETUP TIMESTEP + SEED ###
    if paramsS["seed"] == None:
        setup(dt=paramsS["timestep"])
    else:
        setup(dt=paramsS["timestep"], seed=paramsS["seed"])

    ### Obtain the simulation time from the cortical rate files
    inferred_duration_ms, mixed_rates = infer_max_sim_time_ms(
        dbs_condition=dbs_condition,
        cortical_rate_path=paramsS["mc.cortical_rate_path"],
        dt_ms=paramsS["timestep"],
        n_trs=args.n_trs,
    )
    # An explicit t.duration in parameters.py wins, unless --n-trs shortens the run
    if paramsS.get("t.duration") is None or args.n_trs is not None:
        paramsS["t.duration"] = inferred_duration_ms
    print(
        f"cortical drive: {len(mixed_rates['caudate']['rate'])} TRs -> "
        f"simulating {paramsS['t.duration'] / 1000.0:.1f} s "
        f"(ramp-up {paramsS['t.rampup'] / 1000.0:.2f} s)"
    )

    ### CREATE THE TWO LOOP MODEL ###
    # loop for BG loops
    compilation_appendix = args.compile_appendix
    if compilation_appendix:
        compile_folder = f"bgm_{model_version}_{dbs_condition}_{compilation_appendix}"
    else:
        compile_folder = f"bgm_{model_version}_{dbs_condition}"
    model_dict = {}
    # keep the kwargs per loop: each loop is driven by its own cortical mix, and
    # add_TimedInputs below needs the matching one
    model_creation_kwargs_dict = {}
    for loop in LOOPS:
        if model_version == "v07":
            ### v07 builds the striatum with Microcircuit and drives thal/gpe_arky/
            ### gpe_cp/stn with CorticalInputs, both streaming precomputed spike
            ### counts from the caches under mc_ci_cache_dir (see
            ### build_input_caches.py).
            model_creation_kwargs_dict[loop] = v07_model_creation_kwargs(
                loop=loop,
                dbs_condition=dbs_condition,
                duration_ms=paramsS["t.duration"],
                cache_dir=args.cache_dir,
            )
        else:
            ### v08 replaces the whole cortical drive with one TimedArray per loop
            # - "input.rates": array with (steps, post_size) shape containing the input rates for each time step
            # - "input.schedule": a single scalar with the schedule time in ms for updating the input rates
            # - "timestep": simulation timestep in ms
            model_creation_kwargs_dict[loop] = {
                "input.rates": mixed_rates[loop]["rate"],
                "input.schedule": TR_S * 1000.0,
                "timestep": paramsS["timestep"],
            }

        ### Create model for the current loop
        model_dict[loop] = BGM(
            name=f"BGM_{model_version}_p01",
            model_creation_kwargs=model_creation_kwargs_dict[loop],
            seed=paramsS["seed"],
            compile_folder_name=compile_folder,
            name_appendix=loop,
            do_create=True,
            do_compile=False,
        )

    ### ADD TIMED INPUTS TO BOTH LOOPS ###
    # only v08 drives the model this way; v07 gets its inputs from the
    # Microcircuit/CorticalInputs streams created during model creation.
    # This used to run *after* the DBS block, because DBSstimulator(auto_implement=True)
    # cleared and recreated the network and could not reconstruct a TimedArray or a
    # CurrentInjection. The DBS mechanisms are now retrofitted onto the existing
    # objects instead, so the inputs can be built here where they belong, and every
    # population exists before the DBS footprint is resolved.
    if model_version == "v08":
        for loop in LOOPS:
            add_TimedInputs(
                model_creation_kwargs=model_creation_kwargs_dict[loop],
                params=model_dict[loop].params,
                loop_name=loop,
            )

    ### DBS MECHANISMS ###
    # Added in BOTH conditions, so off and on compile the same network and differ
    # only in parameter values. Two consequences worth knowing:
    #  - DBSstimulator has to be constructed in the off condition too, because its
    #    __init__ is what creates the `pulse` function and the dbs_pulse_* constants
    #    that the rewritten equations reference.
    #  - the off-condition numerics changed when this landed: ANNarchy's RNG is one
    #    global stream, so the two extra Uniform draws shift every population,
    #    including the caudate loop, which carries no DBS terms at all.
    # See DBS.md.
    dbs_population_list = [
        get_population(f"{name}:{DBS_LOOP}") for name in DBS_POPULATION_NAMES
    ]
    dbs_projection_list = [
        get_projection(f"{name}:{DBS_LOOP}") for name in DBS_PROJECTION_NAMES
    ]
    add_dbs_mechanisms(
        populations=dbs_population_list, projections=dbs_projection_list
    )

    ### DBS SIMULATOR ###
    # The last three parameters are the DBS ones. They used to be read at fixed
    # indices 21-23, which only lined up with v08's 21 base parameters; v07 has 19.
    # --compile is called with a placeholder vector and simulates nothing, so the
    # strengths do not matter there.
    dbs_depolarization, passing_fibres_strength, axon_spikes_per_pulse = (
        dbs_params if dbs_params is not None else (0.0, 0.0, 0.0)
    )
    dbs_stimulator = DBSstimulator(
        stimulated_population=get_population(f"stn:{DBS_LOOP}"),
        # VTA from berlin data subject 1:
        population_proportion=(35 + 23) / (70 + 75),
        # everything outside the DBS footprint. This is load-bearing, not just
        # documentation: the efferent and afferent branches of _set_orthodromic and
        # _set_antidromic write axon_transmission and prob_axon_spike without a
        # hasattr guard and skip only via this list, so without it the afferent pass
        # would set axon_transmission on the CurrentInjection projections that carry
        # the cortical drive into STN.
        excluded_populations_list=[
            pop for pop in populations() if pop not in dbs_population_list
        ],
        # the dbs_depolarization parameter actually reduces the membrane potential
        # so its actually a hyperpolarization
        dbs_depolarization=dbs_depolarization,  # [0,10] like weight/conductance in stn
        orthodromic=True,
        antidromic=True,
        efferents=True,
        afferents=True,
        passing_fibres=True,
        # snr__thal is actually gpi__thal, pasing fibre based on Miocinovic et al. 2006
        passing_fibres_list=[get_projection(f"snr__thal:{DBS_LOOP}")],
        passing_fibres_strength=passing_fibres_strength,  # [0,1] scale between 0 and 1
        dbs_pulse_frequency_Hz=125,  # from berlin data subject 1
        # pulse width needs to be multiple of timestep (0.1 ms --> min 100 us)
        # pulse width in berlin data is 60 us but we use 100 us here
        dbs_pulse_width_us=100,
        axon_spikes_per_pulse=axon_spikes_per_pulse,  # [0,1] max 1 spike per pulse(=timestep)
        seed=paramsS["seed"],
        auto_implement=False,
    )

    ### ACTIVATE DBS ###
    # Before compile() on purpose. Pre-compile, Population.__setattr__ writes to
    # pop.init and Projection._set_flag writes to proj.init, so the on-state becomes
    # the compile-time state and every reset() restores it. Called after compile it
    # went to the C++ instance instead, and the first reset() in Spikes10s.run put
    # dbs_on back to 0 - which is how DBS-on evaluations ran with DBS switched off.
    if dbs_condition == "on":
        dbs_stimulator.on()

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
    # each entry maps an experimental ROI to the (compartment, loop) pairs pooled
    # into it; the population names differ between model versions for the striatum
    bold_region_compartments: dict[str, list[tuple[str, str]]] = {
        "GPi": [("snr", "caudate"), ("snr", "putamen")],
        "GPe": [
            ("gpe_proto", "caudate"),
            ("gpe_arky", "caudate"),
            ("gpe_cp", "caudate"),
            ("gpe_proto", "putamen"),
            ("gpe_arky", "putamen"),
            ("gpe_cp", "putamen"),
        ],
        "STN": [("stn", "caudate"), ("stn", "putamen")],
        "Cau": [
            ("str_d1", "caudate"),
            ("str_d2", "caudate"),
            ("str_fsi", "caudate"),
        ],
        "Put": [
            ("str_d1", "putamen"),
            ("str_d2", "putamen"),
            ("str_fsi", "putamen"),
        ],
        "MD": [("thal", "caudate")],
        "VAp": [("thal", "putamen")],
    }
    bold_region_dict: dict[str, list[str]] = {
        region: [population_name(model_version, loop, comp) for comp, loop in comps]
        for region, comps in bold_region_compartments.items()
    }
    # calculate scaling factors for the different GPe populations
    gpe_proportions = {"gpe_proto": 0.5, "gpe_arky": 0.17, "gpe_cp": 0.10}
    gpe_scaling_factors = [
        gpe_proportions[comp] for comp, _ in bold_region_compartments["GPe"]
    ]
    gpe_scaling_factors = np.array(gpe_scaling_factors) / np.sum(gpe_scaling_factors)
    # scaling factors for striatum proportions (del Rey et al. 2022)
    # Cau and Put have the same proportions
    # v08 only: its three striatal populations all have 100 neurons, so they have
    # to be weighted explicitly. In v07 the Microcircuit already sizes dSPN/iSPN/FS
    # by these very proportions, and BoldMonitor's default is to weight by
    # population size, so passing them again would only restate that default (up to
    # the integer rounding of the cell counts). Leave it to the default there.
    if model_version == "v08":
        props_delRey = {"str_d1": 0.86 / 2, "str_d2": 0.86 / 2, "str_fsi": 0.026}
        str_scaling_factors = [
            props_delRey[comp] for comp, _ in bold_region_compartments["Cau"]
        ]
        str_scaling_factors = (
            np.array(str_scaling_factors) / np.sum(str_scaling_factors)
        ).tolist()
    else:
        str_scaling_factors = None

    # create the BoldMonitor objects
    bold_monitor_dict: dict[str, BoldMonitor] = {}
    for bold_region, population_names in bold_region_dict.items():
        if bold_region == "GPe":
            scale_factors = gpe_scaling_factors.tolist()
        elif bold_region in ["Cau", "Put"]:
            scale_factors = str_scaling_factors
        else:
            scale_factors = None

        # for BGM pops input variable = I
        # for Microcircuit pops input variable = I_v
        if bold_region in ["Cau", "Put"]:
            input_var = "I_v"
        else:
            input_var = "I"

        # get populations from population names
        # named bold_populations, not populations: a local assignment here would
        # shadow ANNarchy's populations() for the whole function, including the
        # DBS block above
        bold_populations = [get_population(name) for name in population_names]

        bold_monitor_dict[bold_region] = BoldMonitor(
            populations=bold_populations,
            mapping={"I_CBF": input_var},
            normalize_input=2000,
            scale_factor=scale_factors,
            start=False,
        )

    ### COMPILE ###
    ### Compile model (i.e. both loops in a single model) afterwards we are ready to simulate
    model_dict["caudate"].compile()

    ### NETWORK REPORT ###
    # After compile, so the report describes the network ANNarchy really built,
    # including the equations add_dbs_mechanisms rewrote. Before the BoldMonitor
    # period is changed below, so the monitors appear as they were declared.
    if args.report:
        report(filename=args.report)
        print(f"Wrote network report to {args.report}; skipping simulations.")
        sys.exit(0)

    ### BOLD SAMPLING RATE ###
    # Record BOLD on the TR grid of the experimental data instead of every dt.
    # At dt = 0.1 ms a full run would otherwise store 7.16 million samples per
    # region, several GB per process, of which only every 23100th is ever used.
    for bold_monitor in bold_monitor_dict.values():
        bold_monitor._monitor.period = TR_S * 1000.0

    if args.compile:
        print("Compilation completed; skipping simulations (--compile).")
        sys.exit(0)

    ### MONITORS ###
    ### create monitors to record the spikes from all populations
    # first collect the populations to monitor, simply self.populations of both loops without pops starting with "TimedInput"
    pops_to_monitor = []
    for loop in ["caudate", "putamen"]:
        bgm_model: BGM = model_dict[loop]
        for name in bgm_model.populations:
            if not name.startswith("TimedInput"):
                pops_to_monitor.append(name)
    # create the monitor dictionary with variables to record
    monitor_dictionary = {pop_name: ["spike"] for pop_name in pops_to_monitor}
    monitors = CompNeuroMonitors(monitor_dictionary)

    ### EXPERIMENT ###
    ### Define a CompNeuroPy experiment to run the model 10 s and obtain the spike recordings
    experiment = Spikes10s(
        monitors=monitors,
        model_dict=model_dict,
        seed=paramsS["seed"],
        model_version=model_version,
        putamen_cluster_scalings=putamen_cluster_scalings,
    )

    ### SIMULATIONS ###

    ### SIMULATION FOR RATES: ###
    assert_dbs_state(dbs_stimulator, dbs_condition, "before the firing-rate probe")
    firing_rate_dict = get_firing_rate_10s(param_list=param_list, experiment=experiment)
    # obtain loss based on firing rates, between 0 and 1
    firing_rate_loss = get_firing_rate_loss(firing_rate_dict, model_version)

    ### FIRING-RATE GATE ###
    # The rate probe costs seconds, the BOLD run costs half an hour. An individual
    # whose populations are far outside their plausible bands cannot produce
    # meaningful BOLD, so skip it and charge the worst possible BOLD loss, 1.0.
    # Ordering stays consistent for CMA-ES, which is rank-based: an evaluated
    # individual scores firing_rate_loss + bold_loss with bold_loss <= 1, so a
    # gated one never displaces an evaluated one that had the same rate loss.
    gate_threshold = args.gate_threshold
    bold_skipped = firing_rate_loss > gate_threshold

    if bold_skipped:
        print(
            f"firing-rate loss {firing_rate_loss:.4f} exceeds the gate "
            f"{gate_threshold:.4f}; skipping the BOLD run"
        )
        bold_data = {}
        bold_loss = 1.0
        per_region_corr = {}
    else:
        ### SIMULATION FOR BOLD: ###
        assert_dbs_state(dbs_stimulator, dbs_condition, "before the BOLD run")
        bold_data = get_BOLD_full(
            model_dict=model_dict,
            seed=paramsS["seed"],
            bold_monitor_dict=bold_monitor_dict,
            param_list=param_list,
            model_version=model_version,
            putamen_cluster_scalings=putamen_cluster_scalings,
        )
        # obtain loss based on BOLD correlation, between 0 and 1
        bold_loss, per_region_corr = compute_bold_correlation_loss(
            sim_bold=bold_data,
            condition=dbs_condition,
            tr_s=TR_S,
            ramp_up_ms=paramsS["t.rampup"],
            data_file=None,
            region_map=None,
            dt_ms=paramsS["timestep"],
        )

    ### TOTAL LOSS ###
    total_loss = float(firing_rate_loss + bold_loss)
    # Record the components too: a bare total cannot tell a good BOLD fit with
    # implausible rates from the reverse, and the diagnosis matters more than the
    # number when a multi-day run produces something unexpected.
    loss_payload = {
        "total_loss": total_loss,
        "firing_rate_loss": float(firing_rate_loss),
        "bold_loss": float(bold_loss),
        "bold_correlations": {k: float(v) for k, v in per_region_corr.items()},
        "firing_rates_hz": {k: float(v) for k, v in firing_rate_dict.items()},
        "n_bold_samples": {k: int(np.asarray(v).size) for k, v in bold_data.items()},
        "bold_skipped": bool(bold_skipped),
        "gate_threshold": float(gate_threshold),
        "params": [float(p) for p in param_list],
        "putamen_cluster_scalings": (
            None
            if putamen_cluster_scalings is None
            else [float(p) for p in putamen_cluster_scalings]
        ),
        "dbs_params": None if dbs_params is None else [float(p) for p in dbs_params],
        "dbs": dbs_condition,
        "model_version": model_version,
        "n_trs": len(mixed_rates["caudate"]["rate"]),
        "duration_ms": float(paramsS["t.duration"]),
    }
    print(
        f"loss {total_loss:.4f} = firing_rate {firing_rate_loss:.4f} + bold "
        f"{bold_loss:.4f}{' (gated, not simulated)' if bold_skipped else ''}"
    )

    loss_folder = Path(__file__).resolve().parent / paramsS["data_folder"]
    loss_folder.mkdir(parents=True, exist_ok=True)
    appendix_tag = compilation_appendix if compilation_appendix else "default"
    loss_file = loss_folder / f"loss_{appendix_tag}.json"
    with open(loss_file, "w", encoding="ascii") as f:
        json.dump(loss_payload, f, ensure_ascii=True, indent=2)
