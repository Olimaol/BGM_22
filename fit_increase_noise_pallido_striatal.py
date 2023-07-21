from ANNarchy import (
    setup,
    simulate,
    raster_plot,
    get_projection,
    set_seed,
    get_population,
)
from ANNarchy.core.Global import _network
from CompNeuroPy.models import BGM
from CompNeuroPy import Monitors, create_dir
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK
import json

### local
from parameters import parameters_fit_pallido_striatal as paramsS


def set_parameters(parameter_dict):
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


def create_and_start_monitors():
    ### first remove all previous monitors
    clear_monitors()
    dd_name_appendix_list = []
    for model_idx in range(paramsS["nbr_models"]):
        dd_name_appendix_list.append(model_dd_list[model_idx].name_appendix)

    mon_dict = {
        f"pop;{pop_name}{dd_name_appendix}": ["spike"]
        for pop_name in ["str_d2", "str_fsi", "gpe_arky"]
        for dd_name_appendix in dd_name_appendix_list
    }
    mon = Monitors(mon_dict)
    mon.start()
    return mon


def get_results(mon):
    recordings = mon.get_recordings()

    ### mean firing rates
    firing_rate_dict = {}
    for pop_name in ["str_d2", "str_fsi", "gpe_arky"]:
        for model_version_idx, model_version in enumerate(["dd", "control"]):
            firing_rate_dict[f"{pop_name}__{model_version}"] = []
            for model_idx in range(paramsS["nbr_models"]):
                # get spike dict
                spike_dict = recordings[model_version_idx][
                    f"{pop_name}{model_dd_list[model_idx].name_appendix};spike"
                ]
                t, _ = raster_plot(spike_dict)
                # calculate firing rate
                nbr_of_spikes = len(t)
                nbr_of_neurons = len(spike_dict)
                firing_rate_dict[f"{pop_name}__{model_version}"].append(
                    nbr_of_spikes / (nbr_of_neurons * (paramsS["t.duration"] / 1000))
                )  # in Hz
    mean_firing_rate_dict = {
        key: np.mean(firing_rate_dict[key]) for key in firing_rate_dict.keys()
    }

    return {"mean_firing_rate_dict": mean_firing_rate_dict}


def get_loss(results_dict):
    ### target:
    ### firing rates
    mean_firing_rate_dict_target = {
        "str_d2__control": 2,  # [2, 0.24],
        "str_d2__dd": 5,  # [5, 0.63],
        "gpe_arky__control": 24.5,  # [24.5, 1.14],
        "gpe_arky__dd": 18.9,  # [18.9, 0.87],
        "str_fsi__control": 21.4,  # [21.4, 0.75],
        "str_fsi__dd": 23.7,  # [23.7, 0.69],
    }
    ### differences
    diff_dict_target = {
        pop: mean_firing_rate_dict_target[f"{pop}__dd"]
        - mean_firing_rate_dict_target[f"{pop}__control"]
        for pop in ["str_d2", "gpe_arky", "str_fsi"]
    }
    ### is:
    ### firing rates
    mean_firing_rate_dict = results_dict["mean_firing_rate_dict"]
    ### differences
    diff_dict = {
        pop: mean_firing_rate_dict[f"{pop}__dd"]
        - mean_firing_rate_dict[f"{pop}__control"]
        for pop in ["str_d2", "gpe_arky", "str_fsi"]
    }
    ### loss calculation
    ### firing rates control loss
    loss_firing_rates_control = np.mean(
        [
            exp_loss(
                mean_firing_rate_dict[key],
                mean_firing_rate_dict_target[key],
                mean_firing_rate_dict_target[key],
            )
            for key in ["str_d2__control", "gpe_arky__control", "str_fsi__control"]
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


def dd_to_control(a, b):
    """
    cut synapses in str_fsi__str_d2 and decrease str_d2 input
    a: modulation of increase_noise in str_d2
    b: prune probability (zero means no synapses are pruned, 1 means all synapses are pruned)
    """
    for model_idx in range(paramsS["nbr_models"]):
        name_appendix = model_dd_list[model_idx].name_appendix
        ### decrease str_d2 input
        get_population(f"str_d2{name_appendix}").increase_noise = (
            a * get_population(f"str_d2{name_appendix}").increase_noise
        )
        ### prune synapses
        rng = np.random.default_rng(paramsS["seed"])
        proj = get_projection(f"str_fsi__str_d2{name_appendix}")
        synapses_ranks_list = []
        ### get ranks of all synapses
        for post in proj.post_ranks:
            pre_rank_list = proj[post].pre_ranks
            for pre in pre_rank_list:
                synapses_ranks_list.append([post, pre])
        ### create a mask which only contains a subset of the synapses which should be pruned
        nbr_synapses_to_prune = np.round(len(synapses_ranks_list) * b).astype(int)
        mask_to_prune = np.zeros(len(synapses_ranks_list))
        mask_to_prune[:nbr_synapses_to_prune] = 1
        rng.shuffle(mask_to_prune)
        ### prune the subset of synapses
        for post, pre in np.array(synapses_ranks_list)[mask_to_prune.astype(bool)]:
            proj[post.astype(tuple)].prune_synapse(pre.astype(tuple))


def do_simulation(mon, parameter_dict):
    ### simulate dd models
    ### reset model/paramters
    set_seed(paramsS["seed"])
    mon.reset(synapses=True, projections=True)
    ### set parameters
    set_parameters(parameter_dict)
    ### simulate
    # print_dendrites()
    simulate(paramsS["t.duration"])

    ### simulate control models
    ### reset model/paramters
    set_seed(paramsS["seed"])
    mon.reset(synapses=True, projections=True)
    ### set parameters
    set_parameters(parameter_dict)
    ### transform dd models into control models
    dd_to_control(a=parameter_dict["general.str_d2_factor"], b=1)
    ### simulate
    # print_dendrites()
    simulate(paramsS["t.duration"])


def get_parameter_dict(parameter_list):
    parameter_dict = {}
    parameter_list_idx = 0
    for key in paramsS["parameter_bound_dict"].keys():
        if isinstance(paramsS["parameter_bound_dict"][key], list):
            parameter_dict[key] = parameter_list[parameter_list_idx]
            parameter_list_idx += 1
        else:
            parameter_dict[key] = paramsS["parameter_bound_dict"][key]
    return parameter_dict


def simulate_and_return_loss(parameter_list, return_results=False):
    """
    called multiple times during fitting
    """
    ### get parameter dict
    parameter_dict = get_parameter_dict(parameter_list)
    ### create and start monitors
    mon = create_and_start_monitors()
    ### do simulateion
    do_simulation(mon, parameter_dict)
    ### get results
    results_dict = get_results(mon)
    ### calculate and return loss
    loss = get_loss(results_dict)
    ### store params and loss in txt file
    with open("results/fit_pallido_striatal/fit_results.json", "a") as f:
        json.dump(
            {
                "parameter_dict": parameter_dict,
                "loss": loss,
                "results_dict": results_dict,
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


def clear_monitors():
    _network[0]["monitors"] = []


def print_dendrites():
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
            f"str_d2__gpe_arky{model_dd_list[model_idx].name_appendix}"
        ):
            n2 += len(n_post.pre_ranks)
        for n_post in get_projection(
            f"gpe_arky__str_fsi{model_dd_list[model_idx].name_appendix}"
        ):
            n3 += len(n_post.pre_ranks)

        print(
            f"model {model_idx}:\t str_fsi__str_d2 = {n1},\t str_d2__gpe_arky = {n2},\t gpe_arky__str_fsi = {n3}"
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


if __name__ == "__main__":
    create_dir("results/fit_pallido_striatal/", clear=True)

    ### create file to store results
    with open("results/fit_pallido_striatal/fit_results.json", "w") as f:
        pass
    f.close()

    ### SETUP TIMESTEP + SEED
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

    ### COMPILE MODELS
    model_control_list = []
    model_dd_list = []
    for model_idx in range(paramsS["nbr_models"]):
        model_dd_list.append(
            BGM(
                name="BGM_v04oliver_p03",
                seed=paramsS["seed"],
                do_compile=False,
                name_appendix=f"dd_{model_idx}",
            )
        )
    model_dd_list[-1].compile()

    ### OPTIMIZE ###
    # fit_space = get_fit_space()

    # best = fmin(
    #     fn=simulate_and_return_loss,
    #     space=fit_space,
    #     algo=tpe.suggest,
    #     max_evals=paramsS["nbr_fit_runs"],
    # )

    param_list = [13, 0, 0]
    result = simulate_and_return_loss(param_list, return_results=True)
    print("only str_d2 active")
    print(result["parameter_dict"])
    print(result["results_dict"], "\n")
    param_list = [13, 1.5, 0]
    result = simulate_and_return_loss(param_list, return_results=True)
    print("also str_fsi active, weight is zero")
    print(result["parameter_dict"])
    print(result["results_dict"], "\n")
    param_list = [13, 1.5, 1]
    result = simulate_and_return_loss(param_list, return_results=True)
    print("now also with weight >0 --> str_fsi inhibits str_d2")
    print("in control all synapses are pruned --> there should be no input to str_d2")
    print(result["parameter_dict"])
    print(result["results_dict"], "\n")
