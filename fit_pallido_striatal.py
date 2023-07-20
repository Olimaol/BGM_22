from ANNarchy import setup, simulate, reset, raster_plot
from CompNeuroPy.models import BGM
from CompNeuroPy import Monitors, create_dir
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK

### local
from parameters import parameters_fit_pallido_striatal as paramsS


def set_parameters(parameter_list):
    parameter_key_list = [
        "str_d2.increase_noise",
        "str_fsi.increase_noise",
        "gpe_arky.increase_noise",
        "str_d2__gpe_arky.mod_factor",
        "str_d2__str_d2.mod_factor",
        "gpe_arky__str_fsi.mod_factor",
        "gpe_arky__gpe_arky.mod_factor",
        "str_fsi__str_d2.mod_factor",
        "str_fsi__str_fsi.mod_factor",
    ]
    paramter_dict = {
        parameter_key_list[idx]: parameter_list[idx]
        for idx in range(len(parameter_list))
    }

    for param_key, param_val in paramter_dict.items():
        compartment, param_name = param_key.split(".")
        population_list = compartment.split("__")
        if len(population_list) == 1:
            ### compartment only population
            for model_idx in range(paramsS["nbr_models"]):
                model_control_list[model_idx].set_param(
                    compartment=population_list[0]
                    + model_control_list[model_idx].name_appendix,
                    parameter_name=param_name,
                    parameter_value=param_val,
                )
                model_dd_list[model_idx].set_param(
                    compartment=population_list[0]
                    + model_dd_list[model_idx].name_appendix,
                    parameter_name=param_name,
                    parameter_value=param_val,
                )
        else:
            ### compartment is a projection
            for model_idx in range(paramsS["nbr_models"]):
                model_control_list[model_idx].set_param(
                    compartment=f"{population_list[0]}__{population_list[1]}"
                    + model_control_list[model_idx].name_appendix,
                    parameter_name=param_name,
                    parameter_value=param_val,
                )
                model_dd_list[model_idx].set_param(
                    compartment=f"{population_list[0]}__{population_list[1]}"
                    + model_dd_list[model_idx].name_appendix,
                    parameter_name=param_name,
                    parameter_value=param_val,
                )


def create_and_start_monitors():
    mon_control_list = []
    mon_dd_list = []
    for model_idx in range(paramsS["nbr_models"]):
        control_name_appendix = model_control_list[model_idx].name_appendix
        dd_name_appendix = model_dd_list[model_idx].name_appendix
        mon_control_list.append(
            Monitors(
                {
                    f"pop;str_d2{control_name_appendix}": ["spike"],
                    f"pop;str_fsi{control_name_appendix}": ["spike"],
                    f"pop;gpe_arky{control_name_appendix}": ["spike"],
                }
            )
        )
        mon_dd_list.append(
            Monitors(
                {
                    f"pop;str_d2{dd_name_appendix}": ["spike"],
                    f"pop;str_fsi{dd_name_appendix}": ["spike"],
                    f"pop;gpe_arky{dd_name_appendix}": ["spike"],
                }
            )
        )
        mon_control_list[-1].start()
        mon_dd_list[-1].start()
    return [mon_control_list, mon_dd_list]


def get_results(mon_control_list, mon_dd_list):
    recordings_control_list = []
    recordings_dd_list = []
    for model_idx in range(paramsS["nbr_models"]):
        recordings_control_list.append(mon_control_list[model_idx].get_recordings())
        # recording_times_control = mon_control.get_recording_times()
        recordings_dd_list.append(mon_dd_list[model_idx].get_recordings())
        # recording_times_dd = mon_dd.get_recording_times()

    ### mean firing rates
    firing_rate_dict = {}
    for pop_name in ["str_d2", "str_fsi", "gpe_arky"]:
        for model_version in ["control", "dd"]:
            firing_rate_dict[f"{pop_name}__{model_version}"] = []
            for model_idx in range(paramsS["nbr_models"]):
                if model_version == "dd":
                    spike_dict = recordings_dd_list[model_idx][0][
                        f"{pop_name}{model_dd_list[model_idx].name_appendix};spike"
                    ]
                    t, _ = raster_plot(spike_dict)
                else:
                    spike_dict = recordings_control_list[model_idx][0][
                        f"{pop_name}{model_control_list[model_idx].name_appendix};spike"
                    ]
                    t, _ = raster_plot(spike_dict)
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


def simulate_and_return_loss(parameter_list, return_results=False):
    """
    called multiple times during fitting
    """
    ### reset model/paramters
    reset()
    ### set parameters
    set_parameters(parameter_list)
    ### create and start monitors
    mon_control_list, mon_dd_list = create_and_start_monitors()
    ### simulate resting
    simulate(paramsS["t.duration"])
    ### get results
    results_dict = get_results(mon_control_list, mon_dd_list)
    ### calculate and return loss
    loss = get_loss(results_dict)
    ### store params and loss in txt file
    with open("results/fit_pallido_striatal/loss_results.txt", "a") as f:
        print("\t".join(np.array(parameter_list).astype(str)) + f"\t{loss}", file=f)

    if return_results:
        return {"status": STATUS_OK, "loss": loss, "results": results_dict}
    else:
        return {"status": STATUS_OK, "loss": loss}


if __name__ == "__main__":
    create_dir("results/fit_pallido_striatal/", clear=True)

    ### SETUP TIMESTEP + SEED
    if paramsS["seed"] == None:
        setup(dt=paramsS["timestep"], num_threads=paramsS["num_threads"])
    else:
        setup(
            dt=paramsS["timestep"],
            seed=paramsS["seed"],
            num_threads=paramsS["num_threads"],
        )

    ### COMPILE MODELS
    model_control_list = []
    model_dd_list = []
    for model_idx in range(paramsS["nbr_models"]):
        model_control_list.append(
            BGM(
                name="BGM_v04oliver_p02",
                seed=paramsS["seed"],
                do_compile=False,
                name_appendix=f"control_{model_idx}",
            )
        )
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
    parameter_bound_dict = {
        "str_d2.increase_noise": [12, 120],
        "str_fsi.increase_noise": [0.5, 5],
        "gpe_arky.increase_noise": [0.13, 1.3],
        "str_d2__gpe_arky.mod_factor": [0, 1],
        "str_d2__str_d2.mod_factor": [0, 1],
        "gpe_arky__str_fsi.mod_factor": [0, 1],
        "gpe_arky__gpe_arky.mod_factor": [0, 1],
        "str_fsi__str_d2.mod_factor": [0, 1],
        "str_fsi__str_fsi.mod_factor": [0, 1],
    }

    fit_space = [
        hp.uniform(key, parameter_bound_dict[key][0], parameter_bound_dict[key][1])
        for key in parameter_bound_dict.keys()
    ]

    with open("results/fit_pallido_striatal/loss_results.txt", "w") as f:
        print("\t".join(list(parameter_bound_dict.keys())) + "\tloss", file=f)

    best = fmin(
        fn=simulate_and_return_loss,
        space=fit_space,
        algo=tpe.suggest,
        max_evals=paramsS["nbr_fit_runs"],
    )
