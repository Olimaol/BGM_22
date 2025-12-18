from ANNarchy import setup, get_population, Uniform, set_seed
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


if __name__ == "__main__":

    ### SETUP TIMESTEP + SEED ###
    if paramsS["seed"] == None:
        setup(dt=paramsS["timestep"])
    else:
        setup(dt=paramsS["timestep"], seed=paramsS["seed"])

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
    print("First run:")
    param_list = [0.0] * 37
    param_list[3] = 0.0  # increase cor-thal weight to 10.0
    # param_list[33] = 0.0  # increase snr-thal weight to 0.0
    firing_rate_dict = get_firing_rate_10s(param_list=param_list, experiment=experiment)
    print("Firing rates:")
    for pop_name, fr in firing_rate_dict.items():
        print(f"{pop_name}: {fr:.2f} Hz")

    print("\nSecond run:")
    param_list = [0.0] * 37
    param_list[3] = 1.0  # increase cor-thal weight to 10.0
    # param_list[33] = 10.0  # increase snr-thal weight to 10.0
    firing_rate_dict = get_firing_rate_10s(param_list=param_list, experiment=experiment)
    print("Firing rates:")
    for pop_name, fr in firing_rate_dict.items():
        print(f"{pop_name}: {fr:.2f} Hz")

    print("\nThird run:")
    param_list = [0.0] * 37
    param_list[3] = 10.0  # increase cor-thal weight to 10.0
    # param_list[33] = 15.0  # increase snr-thal weight to 15.0
    firing_rate_dict = get_firing_rate_10s(param_list=param_list, experiment=experiment)
    print("Firing rates:")
    for pop_name, fr in firing_rate_dict.items():
        print(f"{pop_name}: {fr:.2f} Hz")

    print("\nFourth run:")
    param_list = [0.0] * 37
    param_list[3] = 20.0  # increase cor-thal weight to 10.0
    # param_list[33] = 20.0  # increase snr-thal weight to 20.0
    firing_rate_dict = get_firing_rate_10s(param_list=param_list, experiment=experiment)
    print("Firing rates:")
    for pop_name, fr in firing_rate_dict.items():
        print(f"{pop_name}: {fr:.2f} Hz")

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
