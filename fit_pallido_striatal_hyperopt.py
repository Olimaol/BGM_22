from ANNarchy import (
    setup,
    simulate,
    raster_plot,
    get_projection,
    set_seed,
    get_population,
)
from CompNeuroPy.models import BGM
from CompNeuroPy import Monitors, create_dir
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK
import json

### local
from parameters import parameters_fit_pallido_striatal as paramsS


def set_parameters(parameter_dict, model_dd_list):
    for param_key, param_val in parameter_dict.items():
        compartment, param_name = param_key.split(".")
        if compartment == "general":
            continue
        population_list = compartment.split("__")
        if len(population_list) == 1:
            ### compartment only population
            for model_idx in range(paramsS["nbr_models"]):
                model_dd_list[model_idx].set_param(
                    compartment=population_list[0]
                    + model_dd_list[model_idx].name_appendix,
                    parameter_name=param_name,
                    parameter_value=param_val,
                )
        else:
            ### compartment is a projection
            for model_idx in range(paramsS["nbr_models"]):
                model_dd_list[model_idx].set_param(
                    compartment=f"{population_list[0]}__{population_list[1]}"
                    + model_dd_list[model_idx].name_appendix,
                    parameter_name=param_name,
                    parameter_value=param_val,
                )


def create_monitors(model_dd_list, analyze=False):
    dd_name_appendix_list = []
    for model_idx in range(paramsS["nbr_models"]):
        dd_name_appendix_list.append(model_dd_list[model_idx].name_appendix)

    if analyze:
        mon_dict = {
            f"pop;{pop_name}{dd_name_appendix}": [
                "spike",
                "v",
                "u",
                "g_ampa",
                "g_gaba",
                "I_base",
            ]
            for pop_name in ["str_d2", "str_fsi", "gpe_proto"]
            for dd_name_appendix in dd_name_appendix_list
        }
    else:
        mon_dict = {
            f"pop;{pop_name}{dd_name_appendix}": ["spike"]
            for pop_name in ["str_d2", "str_fsi", "gpe_proto"]
            for dd_name_appendix in dd_name_appendix_list
        }
    mon = Monitors(mon_dict)

    return mon


def get_results(mon, model_dd_list, only_simulate, analyze):
    recordings, recording_times = mon.get_recordings_and_clear()

    ### mean firing rates
    firing_rate_dict = {}
    for pop_name in ["str_d2", "str_fsi", "gpe_proto"]:
        if isinstance(only_simulate, str):
            model_version_list = [only_simulate]
        else:
            model_version_list = ["dd", "control"]
        for model_version_idx, model_version in enumerate(model_version_list):

            firing_rate_dict[f"{pop_name}__{model_version}"] = np.zeros(
                (
                    paramsS["nbr_models"],
                    recording_times.nr_periods(chunk=model_version_idx),
                )
            )
            for model_idx in range(paramsS["nbr_models"]):
                # get spike dict
                spike_dict = recordings[model_version_idx][
                    f"{pop_name}{model_dd_list[model_idx].name_appendix};spike"
                ]
                t, _ = raster_plot(spike_dict)
                nbr_of_neurons = len(spike_dict)
                ### calculate for each preiod in chunk the firing rate
                for rec_period in range(
                    recording_times.nr_periods(chunk=model_version_idx)
                ):
                    start_time, end_time = recording_times.time_lims(
                        chunk=model_version_idx, period=rec_period
                    )
                    start_time = start_time + paramsS["t.init"]
                    nbr_of_spikes = np.sum(
                        (t > start_time).astype(int) * (t < end_time).astype(int)
                    )
                    firing_rate_dict[f"{pop_name}__{model_version}"][
                        model_idx, rec_period
                    ] = nbr_of_spikes / (
                        nbr_of_neurons * ((end_time - start_time) / 1000)
                    )  # in Hz
    ### average over the models
    ### what remains is an array with firing rates for each recording preiod
    mean_firing_rate_dict = {
        key: np.mean(firing_rate_dict[key], axis=0).tolist()
        for key in firing_rate_dict.keys()
    }

    if analyze:
        return {
            "mean_firing_rate_dict": mean_firing_rate_dict,
            "recordings": recordings,
            "recording_times": recording_times,
        }
    else:
        return {"mean_firing_rate_dict": mean_firing_rate_dict}


def get_loss(results_dict, only_simulate):
    if isinstance(only_simulate, str) or paramsS["simulation_protocol"] == "increase":
        return 0
    ### target:
    ### firing rates
    mean_firing_rate_dict_target = {
        "str_d2__control": 2,  # [2, 0.24],
        "str_d2__dd": 5,  # [5, 0.63],
        "gpe_proto__control": 24.5,  # [24.5, 1.14],
        "gpe_proto__dd": 18.9,  # [18.9, 0.87],
        "str_fsi__control": 21.4,  # [21.4, 0.75],
        "str_fsi__dd": 23.7,  # [23.7, 0.69],
    }
    ### differences
    diff_dict_target = {
        pop: mean_firing_rate_dict_target[f"{pop}__dd"]
        - mean_firing_rate_dict_target[f"{pop}__control"]
        for pop in ["str_d2", "gpe_proto", "str_fsi"]
    }
    ### is:
    ### firing rates
    mean_firing_rate_dict = results_dict["mean_firing_rate_dict"]
    ### differences
    diff_dict = {
        pop: mean_firing_rate_dict[f"{pop}__dd"][0]
        - mean_firing_rate_dict[f"{pop}__control"][0]
        for pop in ["str_d2", "gpe_proto", "str_fsi"]
    }
    ### loss calculation
    ### firing rates control loss
    loss_firing_rates_control = np.mean(
        [
            exp_loss(
                mean_firing_rate_dict[key][0],
                mean_firing_rate_dict_target[key],
                mean_firing_rate_dict_target[key],
            )
            for key in ["str_d2__control", "gpe_proto__control", "str_fsi__control"]
        ]
    )
    ### differences loss
    loss_firing_rate_diffs = np.mean(
        [
            exp_loss(
                diff_dict[key],
                diff_dict_target[key],
                np.abs(diff_dict_target[key]),
            )
            for key in diff_dict.keys()
        ]
    )

    return loss_firing_rates_control + loss_firing_rate_diffs


def exp_loss(x, mu, sig):
    return 1 - np.exp(-((x - mu) ** 2) / sig**2)


def dd_to_control(a, b, model_dd_list):
    """
    cut synapses in str_fsi__str_d2 and decrease str_d2 input
    a: modulation of base_mean in str_d2
    b: prune probability (zero means no synapses are pruned, 1 means all synapses are pruned)
    """
    for model_idx in range(paramsS["nbr_models"]):
        name_appendix = model_dd_list[model_idx].name_appendix
        ### decrease str_d2 input
        get_population(f"str_d2{name_appendix}").base_mean = (
            a * get_population(f"str_d2{name_appendix}").base_mean
        )
        if isinstance(paramsS["base_noise"], type(None)):
            ### update str_d2 input noise
            get_population(f"str_d2{name_appendix}").base_noise = (
                0.1 * get_population(f"str_d2{name_appendix}").base_mean
            )
        ### prune synapses
        rng = np.random.default_rng(paramsS["seed"])
        proj = get_projection(f"str_fsi__str_d2{name_appendix}")

        weights = proj.w
        ### get ranks of all synapses
        synapses_ranks_list = []
        for post_idx, post in enumerate(weights):
            for pre_idx, pre in enumerate(post):
                synapses_ranks_list.append([post_idx, pre_idx])
        ### create a mask which only contains a subset of the synapses which should be pruned
        nbr_synapses_to_prune = np.round(len(synapses_ranks_list) * b).astype(int)
        mask_to_prune = np.zeros(len(synapses_ranks_list))
        mask_to_prune[:nbr_synapses_to_prune] = 1
        rng.shuffle(mask_to_prune)
        ### prune the subset of synapses
        for post, pre in np.array(synapses_ranks_list)[mask_to_prune.astype(bool)]:
            weights[post][pre] = 0.0
        proj.w = weights


def which_simulation(model_dd_list, mon):
    if paramsS["simulation_protocol"] == "resting":
        simulate(paramsS["t.init"] + paramsS["t.duration"])

    if paramsS["simulation_protocol"] == "increase":
        ### increase baseline of gpe_proto
        for n_it in range(paramsS["increase_iterations"]):
            mon.start()
            for model_idx in range(len(model_dd_list)):
                for pop_name in ["gpe_proto"]:
                    name_appendix = model_dd_list[model_idx].name_appendix
                    get_population(f"{pop_name}{name_appendix}").base_mean = (
                        paramsS["increase_step"] * n_it
                    )
                    if isinstance(paramsS["base_noise"], type(None)):
                        get_population(f"{pop_name}{name_appendix}").base_noise = (
                            0.1 * get_population(f"{pop_name}{name_appendix}").base_mean
                        )
            simulate(paramsS["t.increase_duration"])
            mon.pause()


def do_simulation(mon, parameter_dict, model_dd_list, only_simulate):
    if not (isinstance(only_simulate, str)):
        mon.start()
        ### simulate dd models
        ### reset model/paramters
        set_seed(paramsS["seed"])
        mon.reset(synapses=True, projections=True)
        ### set parameters
        set_parameters(parameter_dict, model_dd_list)
        ### simulate
        # print_dendrites()
        which_simulation(model_dd_list, mon)

        ### simulate control models
        ### reset model/paramters
        set_seed(paramsS["seed"])
        mon.reset(synapses=True, projections=True)
        ### set parameters
        set_parameters(parameter_dict, model_dd_list)
        ### transform dd models into control models
        dd_to_control(
            a=parameter_dict["general.str_d2_factor"],
            b=0.5,
            model_dd_list=model_dd_list,
        )
        ### simulate
        # print_dendrites()
        which_simulation(model_dd_list, mon)
    elif only_simulate == "control":
        mon.start()
        ### simulate control models
        ### reset model/paramters
        set_seed(paramsS["seed"])
        mon.reset(synapses=True, projections=True)
        ### set parameters
        set_parameters(parameter_dict, model_dd_list)
        ### transform dd models into control models
        dd_to_control(
            a=parameter_dict["general.str_d2_factor"],
            b=0.5,
            model_dd_list=model_dd_list,
        )
        ### simulate
        # print_dendrites()
        which_simulation(model_dd_list, mon)
    elif only_simulate == "dd":
        mon.start()
        ### simulate dd models
        ### reset model/paramters
        set_seed(paramsS["seed"])
        mon.reset(synapses=True, projections=True)
        ### set parameters
        set_parameters(parameter_dict, model_dd_list)
        ### simulate
        # print_dendrites()
        which_simulation(model_dd_list, mon)


def get_parameter_dict(parameter_list):
    if isinstance(parameter_list, list):
        parameter_list = np.array(parameter_list).astype(float)
    elif isinstance(parameter_list, tuple):
        pass
    else:
        parameter_list = parameter_list.astype(float)
    parameter_dict = {}
    parameter_list_idx = 0
    for key in paramsS["parameter_bound_dict"].keys():
        if isinstance(paramsS["parameter_bound_dict"][key], list):
            parameter_dict[key] = parameter_list[parameter_list_idx]
            parameter_list_idx += 1
        else:
            parameter_dict[key] = paramsS["parameter_bound_dict"][key]

    key_list = list(parameter_dict.keys())
    for key in key_list:
        ### if base_mean is set
        if key.split(".")[1] == "base_mean" and isinstance(
            paramsS["base_noise"], type(None)
        ):
            ### also set base_noise
            parameter_dict[f"{key.split('.')[0]}.base_noise"] = (
                0.1 * parameter_dict[key]
            )
    return parameter_dict


def simulate_and_return_loss(
    parameter_list,
    return_results=False,
    mon=None,
    model_dd_list=None,
    only_simulate=None,
    dump=True,
    analyze=False,
):
    """
    called multiple times during fitting
    """
    ### get parameter dict
    parameter_dict = get_parameter_dict(parameter_list)
    ### do simulateion
    do_simulation(mon, parameter_dict, model_dd_list, only_simulate)
    ### get results
    results_dict = get_results(mon, model_dd_list, only_simulate, analyze)
    ### calculate and return loss
    loss = get_loss(results_dict, only_simulate)
    ### store params and loss in txt file
    if dump:
        with open("results/fit_pallido_striatal/fit_results.json", "a") as f:
            json.dump(
                {
                    "parameter_dict": parameter_dict,
                    "loss": loss,
                    "mean_firing_rate_dict": results_dict["mean_firing_rate_dict"],
                },
                f,
            )
        f.close()
    if return_results:
        return {
            "status": STATUS_OK,
            "loss": loss,
            "parameter_dict": parameter_dict,
            "results_dict": results_dict,
        }
    else:
        return {
            "status": STATUS_OK,
            "loss": loss,
        }


def print_dendrites(model_dd_list):
    ### TEST: PRINT DENDRITE SIZES
    for model_idx in range(paramsS["nbr_models"]):
        n1 = 0
        n2 = 0
        n3 = 0
        for n_post in get_projection(
            f"str_fsi__str_d2{model_dd_list[model_idx].name_appendix}"
        ):
            n1 += len(n_post.pre_ranks)
        for n_post in get_projection(
            f"str_d2__gpe_proto{model_dd_list[model_idx].name_appendix}"
        ):
            n2 += len(n_post.pre_ranks)
        for n_post in get_projection(
            f"gpe_proto__str_fsi{model_dd_list[model_idx].name_appendix}"
        ):
            n3 += len(n_post.pre_ranks)

        print(
            f"model {model_idx}:\t str_fsi__str_d2 = {n1},\t str_d2__gpe_proto = {n2},\t gpe_proto__str_fsi = {n3}"
        )


def get_fit_space():
    fit_space = []
    for key in paramsS["parameter_bound_dict"].keys():
        if isinstance(paramsS["parameter_bound_dict"][key], list):
            fit_space.append(
                hp.uniform(
                    key,
                    paramsS["parameter_bound_dict"][key][0],
                    paramsS["parameter_bound_dict"][key][1],
                )
            )
    return fit_space


def setup_ANNarchy():
    if paramsS["seed"] == None:
        setup(
            dt=paramsS["timestep"],
            num_threads=paramsS["num_threads"],
            structural_plasticity=True,
        )
    else:
        setup(
            dt=paramsS["timestep"],
            seed=paramsS["seed"],
            num_threads=paramsS["num_threads"],
            structural_plasticity=True,
        )


def compile_models():
    ### get model list
    model_dd_list = []
    for model_idx in range(paramsS["nbr_models"]):
        model_dd_list.append(
            BGM(
                name="BGM_v04newgpe_p01",
                seed=paramsS["seed"],
                do_compile=False,
                name_appendix=f"dd_{model_idx}",
            )
        )
    ### set noise values
    if not (isinstance(paramsS["base_noise"], type(None))):
        for key, val in paramsS["base_noise"].items():
            for model in model_dd_list:
                name_appendix = model.name_appendix
                for pop_name in model.populations:
                    if pop_name == f"{key}{name_appendix}":
                        get_population(pop_name).base_noise = val

    ### compile
    model_dd_list[-1].compile()

    return model_dd_list


if __name__ == "__main__":
    create_dir("results/fit_pallido_striatal/", clear=True)

    ### create file to store results
    with open("results/fit_pallido_striatal/fit_results.json", "w") as f:
        pass
    f.close()

    ### SETUP TIMESTEP + SEED
    setup_ANNarchy()

    ### COMPILE MODELS
    model_dd_list = compile_models()

    ### create monitors
    mon = create_monitors(model_dd_list)

    ## OPTIMIZE ###
    fit_space = get_fit_space()

    best = fmin(
        fn=lambda x: simulate_and_return_loss(x, mon=mon, model_dd_list=model_dd_list),
        space=fit_space,
        algo=tpe.suggest,
        max_evals=paramsS["nbr_fit_runs"],
    )
