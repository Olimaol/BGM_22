from ANNarchy import setup, get_population, Uniform, set_seed, reset, get_time
from ANNarchy.extensions.bold import BoldMonitor
from CompNeuroPy.full_models import BGM
from CompNeuroPy import (
    CompNeuroMonitors,
    PlotRecordings,
    Microcircuit,
    CorticalInputs,
    print_df,
    CompNeuroExp,
)
import numpy as np
from time import time

### local
from parameters import parameters_test_microcircuit as paramsS


def set_opt_params(param_list, model_dict):
    """
    Function to set the optimized parameters of the model.

    The ranges of the parameters ar given as follows:
    - minimum is always 0.0
    - maximum is given by the numerical stabilization of the conductance values
        g_eff = g / (1 + g * dt / C), C=50 for striatum and C=1 for other populations
        with dt = 0.1 ms
        --> for striatal populations max g ~ 500
        --> for other populations max g ~ 10
    - the base currents ranges are obtain by multiplying the conductance extreme values
    with the fixed driving force of 50 mV (current based excitation in the model)
        --> 10 * 50 = 500 max base current for snr and gpe_proto
    """
    # catch wrong number of parameters
    if len(param_list) != 37:
        raise ValueError(
            f"Expected 37 parameters, got {len(param_list)} parameters: {param_list}"
        )

    ### FIRING RATE PARAMETERS ###
    # parameters which influence the firing rates of the populations:
    cor_input_weights_dict_str = {
        "dSPN": param_list[0],  # [0, 500]
        "iSPN": param_list[1],  # [0, 500]
        "FS": param_list[2],  # [0, 500]
    }
    cor_input_weights_dict_others = {
        "thal": param_list[3],  # [0, 10]
        "gpe_arky": param_list[4],  # [0, 10]
        "gpe_cp": param_list[5],  # [0, 10]
        "stn": param_list[6],  # [0, 10]
    }
    base_current_dict = {"snr": param_list[7], "gpe_proto": param_list[8]}  # [0, 500]

    for loop in ["caudate", "putamen"]:
        # dSPN, iSPN, FS: excitatory inputs defined in the microcircuit
        mc: Microcircuit = model_dict[loop].mc
        # set the weights in the striatum
        for post_type in mc.cell_types:
            for cortical_region in mc.cortical_proportions_dict.keys():
                key = (cortical_region, post_type)
                # set the weight value for the excitatory cortical input
                mc.mean_weights_by_type[key] = cor_input_weights_dict_str[post_type]
        # thal, gpe_arky, gpe_cp, stn: excitatory inputs defined in the cortical inputs
        ci: CorticalInputs = model_dict[loop].ci
        # the ci class uses the unchanged names of the populations (see model creation function for names)
        for post_type in ci.cell_types:
            for cortical_region in ci.cortical_proportions_dict.keys():
                key = (cortical_region, post_type)
                # set the weight value for the excitatory cortical input
                ci.mean_weights_by_type[key] = cor_input_weights_dict_others[post_type]
        # snr, gpe_proto: without cortical inputs --> regulate baseline current
        bgm_model: BGM = model_dict[loop]
        # use the names of the populations (see model creation function) and add :loop (which is done internally in BGM)
        for compartment_name, param_val in base_current_dict.items():
            bgm_model.set_param(
                compartment=f"{compartment_name}:{loop}",
                parameter_name="base_mean",
                parameter_value=param_val,
            )

    ### WEIGHT PARAMETERS ###
    # projections are:
    # ['str_d1__snr', 'str_d1__gpe_cp', 'str_d2__gpe_proto', 'str_d2__gpe_arky',
    # 'str_d2__gpe_cp', 'stn__snr', 'stn__gpe_proto', 'stn__gpe_arky', 'stn__gpe_cp',
    # 'gpe_proto__stn', 'gpe_proto__snr', 'gpe_proto__gpe_arky', 'gpe_proto__gpe_cp',
    # 'gpe_proto__str_fsi', 'gpe_arky__str_d1', 'gpe_arky__str_d2', 'gpe_arky__str_fsi',
    # 'gpe_arky__gpe_proto', 'gpe_arky__gpe_cp', 'gpe_cp__str_d1', 'gpe_cp__str_d2',
    # 'gpe_cp__str_fsi', 'gpe_cp__gpe_proto', 'gpe_cp__gpe_arky', 'snr__thal',
    # 'thal__str_d1', 'thal__str_d2', 'thal__str_fsi']
    # get the projections whose weights should be optimized
    proj_weights_dict = {
        "str_d1__snr": param_list[9],  # [0, 10]
        "str_d1__gpe_cp": param_list[10],  # [0, 10]
        "str_d2__gpe_proto": param_list[11],  # [0, 10]
        "str_d2__gpe_arky": param_list[12],  # [0, 10]
        "str_d2__gpe_cp": param_list[13],  # [0, 10]
        "stn__snr": param_list[14],  # [0, 10]
        "stn__gpe_proto": param_list[15],  # [0, 10]
        "stn__gpe_arky": param_list[16],  # [0, 10]
        "stn__gpe_cp": param_list[17],  # [0, 10]
        "gpe_proto__stn": param_list[18],  # [0, 10]
        "gpe_proto__snr": param_list[19],  # [0, 10]
        "gpe_proto__gpe_arky": param_list[20],  # [0, 10]
        "gpe_proto__gpe_cp": param_list[21],  # [0, 10]
        "gpe_proto__str_fsi": param_list[22],  # [0, 500]
        "gpe_arky__str_d1": param_list[23],  # [0, 500]
        "gpe_arky__str_d2": param_list[24],  # [0, 500]
        "gpe_arky__str_fsi": param_list[25],  # [0, 500]
        "gpe_arky__gpe_proto": param_list[26],  # [0, 10]
        "gpe_arky__gpe_cp": param_list[27],  # [0, 10]
        "gpe_cp__str_d1": param_list[28],  # [0, 500]
        "gpe_cp__str_d2": param_list[29],  # [0, 500]
        "gpe_cp__str_fsi": param_list[30],  # [0, 500]
        "gpe_cp__gpe_proto": param_list[31],  # [0, 10]
        "gpe_cp__gpe_arky": param_list[32],  # [0, 10]
        "snr__thal": param_list[33],  # [0, 10]
        "thal__str_d1": param_list[34],  # [0, 500]
        "thal__str_d2": param_list[35],  # [0, 500]
        "thal__str_fsi": param_list[36],  # [0, 500]
    }

    # set the weights for both loops
    for loop in ["caudate", "putamen"]:
        bgm_model: BGM = model_dict[loop]
        for proj_name, weight_val in proj_weights_dict.items():
            # set the weight value
            bgm_model.set_param(
                compartment=f"{proj_name}:{loop}",
                parameter_name="w",
                parameter_value=weight_val,
            )


def infer_max_sim_time_ms(
    cortical_rate_path=paramsS["mc.cortical_rate_path"], dt_ms=0.1
):
    """Load any cortical firing-rate trace and infer the maximum simulation time in ms."""

    # pick one file path from the provided mapping or string
    if isinstance(cortical_rate_path, dict):
        if len(cortical_rate_path) == 0:
            raise ValueError("cortical_rate_path is an empty mapping.")
        rate_file = next(iter(cortical_rate_path.values()))
    else:
        rate_file = cortical_rate_path

    # load one rate array; all arrays share the same length
    with np.load(rate_file) as data:
        rate_key = next((k for k in data.files if k.endswith("_rate")), data.files[0])
        rate_trace = data[rate_key]

    # saved arrays use a 0.1 ms step; infer total duration
    return rate_trace.shape[0] * dt_ms


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

        # collect all update functions of the mc and ci and reset all mc/ci
        # reset = inputs start again at the beginning
        update_functions = []
        for loop in ["caudate", "putamen"]:
            mc: Microcircuit = self.model_dict[loop].mc
            ci: CorticalInputs = self.model_dict[loop].ci
            mc.reset()
            ci.reset()
            update_functions.append(mc.update)
            update_functions.append(ci.update)

        # set optimized parameters
        set_opt_params(param_list, self.model_dict)

        # do the simulation in update steps, obtain the update_time in ms of any mc/ci (they are all the same)
        update_time = self.model_dict["caudate"].mc.update_time
        # calculate the update steps for 10 s
        duration_ms = 200  # TODO currently for testing set to 200 ms
        n_updates = int(round(duration_ms / update_time))
        for _ in range(n_updates):
            # call all update functions and set run_simulation=True for the last one
            for uf in update_functions[:-1]:
                uf(run_simulation=False)
            # the following simulates the model for update_time ms
            update_functions[-1](run_simulation=True)

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

    ### RESETS AND PREPARE ###
    # reset ANNarchy and seed
    reset()
    set_seed(seed)

    # collect all update functions of the mc and ci and reset all mc/ci
    # reset = inputs start again at the beginning
    update_functions = []
    for loop in ["caudate", "putamen"]:
        mc: Microcircuit = model_dict[loop].mc
        ci: CorticalInputs = model_dict[loop].ci
        mc.reset()
        ci.reset()
        update_functions.append(mc.update)
        update_functions.append(ci.update)

    # do the simulation in update steps, obtain the update_time in ms of any mc/ci (they are all the same)
    update_time = model_dict["caudate"].mc.update_time

    ### OPTIMIZED PARAMETERS ###
    # set optimized parameters
    set_opt_params(param_list, model_dict)

    ### RAMP UP ###
    # initial 1s ramp up
    ramp_up_duration_ms = 1000
    n_updates = int(round(ramp_up_duration_ms / update_time))
    for _ in range(n_updates):
        # call all update functions and set run_simulation=True for the last one
        for uf in update_functions[:-1]:
            uf(run_simulation=False)
        # the following simulates the model for update_time ms
        update_functions[-1](run_simulation=True)

    # start each bold monitor
    for bold_monitor in bold_monitor_dict.values():
        bold_monitor.start()

    ### REST OF SIMULATION ###
    # just repeat the update functions until they crash which means the end of the input rates
    while True:
        print(
            f"Simulating next {update_time} ms... total time: {get_time():.1f} ms of {paramsS['t.duration']} ms",
            end="\r",
        )
        try:
            # call all update functions and set run_simulation=True for the last one
            for uf in update_functions[:-1]:
                uf(run_simulation=False)
            # the following simulates the model for update_time ms
            update_functions[-1](run_simulation=True)
        except StopIteration:
            # cortical inputs have reached the end of their input rates
            break

    ### RESULTS ###
    # get the bold time signals of each bold monitor
    bold_signals = {}
    for bold_region, bold_monitor in bold_monitor_dict.items():
        bold_signals[bold_region] = bold_monitor.get("BOLD")

    return bold_signals


if __name__ == "__main__":

    ### SETUP TIMESTEP + SEED ###
    if paramsS["seed"] == None:
        setup(dt=paramsS["timestep"])
    else:
        setup(dt=paramsS["timestep"], seed=paramsS["seed"])

    ### Obtain the maximum simulation time from the cortical rate files if needed
    if "t.duration" not in paramsS or paramsS["t.duration"] is None:
        paramsS["t.duration"] = infer_max_sim_time_ms(
            cortical_rate_path=paramsS["mc.cortical_rate_path"],
            dt_ms=paramsS["timestep"],
        )

    ### CREATE THE TWO LOOP MODEL ###
    # loop for BG loops
    model_dict = {}
    for loop in ["caudate", "putamen"]:
        ### Prepare the model_creation kwargs for the current dbs condition
        model_creation_kwargs = {
            "build_mc": False,  # The first time this needs to be True to build the microcircuits
            "build_ci": False,  # The first time this needs to be True to build the cortical inputs
            "mc.name": loop,
            "mc.nx": paramsS["mc.nx"],
            "mc.b": paramsS["mc.b"],
            "dbs": paramsS["dbs"],
            "timestep": paramsS["timestep"],
            "t.duration": paramsS["t.duration"],
            "update_time": paramsS["update_time"],
            "mc.storage_dir": f"mc_ci_cache/mc_{loop}_cache_{paramsS['dbs']}",
            "mc.seed": paramsS.get("seed", 42),
            "mc.fitted_params_path": paramsS["mc.fitted_params_path"],
            "mc.cortical_rate_path": paramsS["mc.cortical_rate_path"][paramsS["dbs"]],
            "ci.storage_dir": f"mc_ci_cache/ci_{loop}_cache_{paramsS['dbs']}",
            "ci.seed": paramsS.get("seed", 42),
            "ci.n_thal": paramsS["ci.n_thal"],
            "ci.n_gpe_arky": paramsS["ci.n_gpe_arky"],
            "ci.n_gpe_cp": paramsS["ci.n_gpe_cp"],
            "ci.n_stn": paramsS["ci.n_stn"],
        }

        ### Create model for the current loop
        model_dict[loop] = BGM(
            name="BGM_v07_p01",
            model_creation_kwargs=model_creation_kwargs,
            seed=paramsS["seed"],
            compile_folder_name=f"bgm_v07_{paramsS['dbs']}",
            name_appendix=loop,
            do_create=True,
            do_compile=False,
        )

    ### BOLD MONITORING ###
    # create BOLD monitors for the following regions:
    # BOLD : model
    # GPi : snr [both loops]
    # GPe : gpe_proto, gpe_arky, gpe_cp [both loops]
    # STN : stn [both loops]
    # Cau : dSPN, iSPN, FS  [caudate loop]
    # Put : dSPN, iSPN, FS  [putamen loop]
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
        "Cau": ["caudate_FS", "caudate_dSPN", "caudate_iSPN"],
        "Put": ["putamen_FS", "putamen_dSPN", "putamen_iSPN"],
        "MD": ["thal:caudate"],
        "VAp": ["thal:putamen"],
    }
    # calculate scaling factors for the different GPe populations
    gpe_proportions = {"gpe_proto": 0.5, "gpe_arky": 0.17, "gpe_cp": 0.10}
    gpe_scaling_factors = [
        gpe_proportions[key.split(":")[0]] for key in bold_region_dict["GPe"]
    ]
    gpe_scaling_factors = np.array(gpe_scaling_factors) / np.sum(gpe_scaling_factors)

    # create the BoldMonitor objects
    bold_monitor_dict: dict[str, BoldMonitor] = {}
    for bold_region, population_names in bold_region_dict.items():
        if bold_region == "GPe":
            scale_factors = gpe_scaling_factors.tolist()
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
            normalize_input=100,  # 2000, TODO change back to 2000
            scale_factor=scale_factors,
            start=False,
        )

    ### COMPILE ###
    ### Compile model (i.e. both loops in a single model) afterwards we are ready to simulate
    model_dict["caudate"].compile()

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

    ### TEST SIMULATIONS ###
    # print("First run:")
    # param_list = [0.0] * 37
    # param_list[3] = 0.0  # increase cor-thal weight to 10.0
    # # param_list[33] = 0.0  # increase snr-thal weight to 0.0
    # firing_rate_dict = get_firing_rate_10s(param_list=param_list, experiment=experiment)
    # print("Firing rates:")
    # for pop_name, fr in firing_rate_dict.items():
    #     print(f"{pop_name}: {fr:.2f} Hz")

    # print("\nSecond run:")
    # param_list = [0.0] * 37
    # param_list[3] = 1.0  # increase cor-thal weight to 10.0
    # # param_list[33] = 10.0  # increase snr-thal weight to 10.0
    # firing_rate_dict = get_firing_rate_10s(param_list=param_list, experiment=experiment)
    # print("Firing rates:")
    # for pop_name, fr in firing_rate_dict.items():
    #     print(f"{pop_name}: {fr:.2f} Hz")

    # print("\nThird run:")
    # param_list = [0.0] * 37
    # param_list[3] = 10.0  # increase cor-thal weight to 10.0
    # # param_list[33] = 15.0  # increase snr-thal weight to 15.0
    # firing_rate_dict = get_firing_rate_10s(param_list=param_list, experiment=experiment)
    # print("Firing rates:")
    # for pop_name, fr in firing_rate_dict.items():
    #     print(f"{pop_name}: {fr:.2f} Hz")

    # print("\nFourth run:")
    # param_list = [0.0] * 37
    # param_list[3] = 20.0  # increase cor-thal weight to 10.0
    # # param_list[33] = 20.0  # increase snr-thal weight to 20.0
    # firing_rate_dict = get_firing_rate_10s(param_list=param_list, experiment=experiment)
    # print("Firing rates:")
    # for pop_name, fr in firing_rate_dict.items():
    #     print(f"{pop_name}: {fr:.2f} Hz")

    ### TEST BOLD SIMULATION ###
    print("\nFirst BOLD run:")
    start_time = time()
    param_list = [0.0] * 37
    bold_data_1 = get_BOLD_full(
        model_dict=model_dict,
        seed=paramsS["seed"],
        bold_monitor_dict=bold_monitor_dict,
        param_list=param_list,
    )
    print(f"BOLD simulation took {time() - start_time:.1f} seconds.")

    print("\nSecond BOLD run:")
    start_time = time()
    param_list = [0.0] * 37
    bold_data_2 = get_BOLD_full(
        model_dict=model_dict,
        seed=paramsS["seed"],
        bold_monitor_dict=bold_monitor_dict,
        param_list=param_list,
    )
    print(f"BOLD simulation took {time() - start_time:.1f} seconds.")

    # compare if the two bold data are identical
    for bold_region in bold_region_dict.keys():
        bold_signal_1 = bold_data_1[bold_region]
        bold_signal_2 = bold_data_2[bold_region]
        if not np.array_equal(bold_signal_1, bold_signal_2):
            print(f"BOLD signals for region {bold_region} are different between runs!")
        else:
            print(f"BOLD signals for region {bold_region} are identical between runs.")
        # plot the bold signals
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        plt.plot(bold_signal_1, label="Run 1")
        plt.plot(bold_signal_2, label="Run 2", linestyle="--")
        plt.title(f"BOLD signal in region {bold_region}")
        plt.xlabel("Time (a.u.)")
        plt.ylabel("BOLD signal (a.u.)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ### PLOT RECORDINGS ###
    # # use plot recordings to generate a figure for each loop and experiment run showing spikes of all recorded populations
    # for i, results in enumerate([results_1, results_2], start=1):
    #     for loop in ["caudate", "putamen"]:
    #         # get recordings and recording times
    #         recordings = results.recordings
    #         recording_times = results.recording_times
    #         # create plan for plotting all populations in the loop
    #         pops_to_monitor_loop = [
    #             pop_name for pop_name in pops_to_monitor if loop in pop_name
    #         ]
    #         plan = {
    #             "position": list(range(1, len(pops_to_monitor_loop) + 1)),
    #             "compartment": pops_to_monitor_loop,
    #             "variable": ["spike"] * len(pops_to_monitor_loop),
    #             "format": ["hybrid"] * len(pops_to_monitor_loop),
    #         }
    #         # plot recordings, make the shape more square-like
    #         n_pops = len(pops_to_monitor_loop)
    #         n_rows = int(n_pops**0.5)
    #         n_cols = (n_pops + n_rows - 1) // n_rows
    #         PlotRecordings(
    #             figname=f"results/test_microcircuit_bgm/{loop}_experiment_run_{i}.png",
    #             recordings=recordings,
    #             recording_times=recording_times,
    #             shape=(n_rows, n_cols),
    #             plan=plan,
    #         )

    # use plot recordings to generate a figure for each loop and experiment run showing offset_base of all recorded populations
    # for i, results in enumerate([results_1, results_2], start=1):
    #     for loop in ["caudate", "putamen"]:
    #         # get recordings and recording times
    #         recordings = results.recordings
    #         recording_times = results.recording_times
    #         # create plan for plotting all populations in the loop
    #         pops_to_monitor_loop = [
    #             pop_name for pop_name in pops_to_monitor_offset if loop in pop_name
    #         ]
    #         plan = {
    #             "position": list(range(1, len(pops_to_monitor_loop) + 1)),
    #             "compartment": pops_to_monitor_loop,
    #             "variable": ["offset_base"] * len(pops_to_monitor_loop),
    #             "format": ["line"] * len(pops_to_monitor_loop),
    #         }
    #         # plot recordings, make the shape more square-like
    #         n_pops = len(pops_to_monitor_loop)
    #         n_rows = int(n_pops**0.5)
    #         n_cols = (n_pops + n_rows - 1) // n_rows
    #         PlotRecordings(
    #             figname=f"results/test_microcircuit_bgm/{loop}_experiment_run_{i}_offset_base.png",
    #             recordings=recordings,
    #             recording_times=recording_times,
    #             shape=(n_rows, n_cols),
    #             plan=plan,
    #         )

    # ### Print parameters
    # print("\nmodel attributes caudate:")
    # print_df(model_dict["caudate"].attribute_df)

    # collect the parameters to change to change firing rates
    # for both loops caudate/putamen:

    # ### INIT CompNeuroMonitors ###
    # mon = CompNeuroMonitors(
    #     {
    #         # "cor_go": ["spike"],
    #         # "cor_stop": ["spike"],
    #         # "cor_pause": ["spike"],
    #         # "str_d1": ["spike"],
    #         "str_d2": ["spike"],
    #         "str_fsi": ["spike"],
    #         "gpe_proto": ["spike", "I_base"],
    #         # "gpe_arky": ["spike", "u"],
    #         # "gpe_cp": ["spike"],
    #     }
    # )

    # ### SIMULATION ###
    # mon.start()

    # ### simulate some time
    # simulate(paramsS["t.duration"])

    # ### GET RECORDINGS ###
    # recordings = mon.get_recordings()
    # recording_times = mon.get_recording_times()

    # ### QUICK PLOTS ###

    # ### some populations activity
    # plan = {
    #     "position": [1, 2, 3, 4],
    #     "compartment": ["str_d2", "str_fsi", "gpe_proto", "gpe_proto"],
    #     "variable": ["spike", "spike", "spike", "I_base"],
    #     "format": ["hybrid", "hybrid", "hybrid", "line"],
    # }
    # chunk = 0
    # PlotRecordings(
    #     figname=f"results/test_resting/{model.name}/overview1.png",
    #     recordings=recordings,
    #     recording_times=recording_times,
    #     chunk=chunk,
    #     shape=(1, 4),
    #     plan=plan,
    # )
