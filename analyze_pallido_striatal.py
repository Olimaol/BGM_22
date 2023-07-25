from CompNeuroPy import create_dir, plot_recordings
import numpy as np

# from tqdm import tqdm
import json

### local
from fit_hyperopt_pallido_striatal import (
    create_monitors,
    simulate_and_return_loss,
    setup_ANNarchy,
    compile_models,
    paramsS,
)


def get_I_0(mode):
    ### parameters
    pop_list = ["str_d2", "str_fsi", "gpe_arky"]
    alpha_0 = 0.1
    tolerance = 0.02
    momentum = 0.3
    n_iter_max = 1000
    mean_firing_rate_dict_target = {
        "str_d2__control": 2,  # [2, 0.24],
        "str_d2__dd": 5,  # [5, 0.63],
        "gpe_arky__control": 24.5,  # [24.5, 1.14],
        "gpe_arky__dd": 18.9,  # [18.9, 0.87],
        "str_fsi__control": 21.4,  # [21.4, 0.75],
        "str_fsi__dd": 23.7,  # [23.7, 0.69],
    }

    pbar = range(n_iter_max)  # tqdm(range(n_iter_max))
    param_arr = np.array([12.053147640924328, 0.5288320347987706, 0.13616425194592505])
    alpha = np.ones(len(param_arr)) * alpha_0

    target = np.array(
        [mean_firing_rate_dict_target[f"{pop}__{mode}"] for pop in pop_list]
    )
    ### iteration loop to find paramters with target firing rates
    diff = 0
    error = 0
    for _ in pbar:
        result = simulate_and_return_loss(
            [param_arr[0], param_arr[1], param_arr[2], 0, 0, 0, 0, 0, 0, 1],
            return_results=True,
            mon=mon,
            model_dd_list=model_dd_list,
            only_simulate=mode,
        )

        ist = np.array(
            [
                result["results_dict"]["mean_firing_rate_dict"][f"{pop}__{mode}"]
                for pop in pop_list
            ]
        )
        ### if sign of error changed --> went over the target --> reduce alpha
        error_changed = (error * (target - ist) < 0).astype(bool)
        alpha[error_changed] = alpha[error_changed] / 2
        alpha[np.logical_not(error_changed)] = (
            alpha[np.logical_not(error_changed)]
            + (alpha_0 - alpha[np.logical_not(error_changed)]) * 0.1
        )
        ### update params depending on error
        error = target - ist
        diff = momentum * diff + alpha * error
        param_arr = np.clip(param_arr + diff, 0, None)
        ### if error is very small --> break
        if (np.absolute(error) / target < tolerance).all():
            break
        ### progress bar info
        # pbar.set_description(f"{ist}")

    return {pop_list[idx]: param_arr[idx] for idx in range(len(param_arr))}


def get_I_lat_inp(mode, I_0):
    ### parameters
    alpha_0 = 0.1
    tolerance = 0.02
    momentum = 0.3
    n_iter_max = 1000
    mean_firing_rate_dict_target = {
        "str_d2__control": 2,  # [2, 0.24],
        "str_d2__dd": 5,  # [5, 0.63],
        "gpe_arky__control": 24.5,  # [24.5, 1.14],
        "gpe_arky__dd": 18.9,  # [18.9, 0.87],
        "str_fsi__control": 21.4,  # [21.4, 0.75],
        "str_fsi__dd": 23.7,  # [23.7, 0.69],
    }

    pop_list = ["str_d2", "str_fsi", "gpe_arky"]
    ### list with lateral mod_factors
    lat_list = np.array([(idx_lat_list + 1) * 0.1 for idx_lat_list in range(31)])
    ### list with input mod_factors
    inp_list = np.array([(idx_inp_list + 1) * 0.1 for idx_inp_list in range(31)])
    I = {}
    global_idx = 0
    for pop in pop_list:
        I[pop] = []
        I[pop].append([0, 0, I_0[mode][pop]])
        for lat in lat_list:
            for inp in inp_list:
                print(f"{global_idx}/{len(pop_list)*len(lat_list)*len(inp_list)}")
                global_idx += 1
                pbar = range(n_iter_max)  # tqdm(range(n_iter_max))
                ### get start value
                I_entry_loss = np.zeros(len(I[pop]))
                I_entry_arr = np.zeros(len(I[pop]))
                for I_entry_idx, I_entry in enumerate(I[pop]):
                    I_entry_loss[I_entry_idx] = abs(I_entry[0] - lat) + abs(
                        I_entry[1] - inp
                    )
                    I_entry_arr[I_entry_idx] = I_entry[2]
                param_start_value = I_entry_arr[np.argmin(I_entry_loss)]
                param_arr = np.array([param_start_value])
                alpha = np.ones(len(param_arr)) * alpha_0

                target = np.array([mean_firing_rate_dict_target[f"{pop}__{mode}"]])
                ### iteration loop to find paramter with target firing rate
                diff = 0
                error = 0
                for _ in pbar:
                    parameter_dict = {}
                    for key in paramsS["parameter_bound_dict"]:
                        compartment, param_name = key.split(".")
                        if pop == compartment:
                            ### parameter of population varied
                            parameter_dict[key] = param_arr[0]
                        elif (
                            "str_d2" == compartment
                            or "str_fsi" == compartment
                            or "gpe_arky" == compartment
                        ):
                            ### parameter of other populations
                            parameter_dict[key] = I_0[mode][compartment]
                        elif compartment == "general":
                            parameter_dict[key] = 1
                        elif len(compartment.split("__")) > 0:
                            if compartment.split("__")[1] == pop:
                                ### input projections of population varied
                                if compartment.split("__")[0] == pop:
                                    ### lateral projection
                                    parameter_dict[key] = lat
                                else:
                                    ### input projection
                                    parameter_dict[key] = inp
                            else:
                                ### other projections
                                parameter_dict[key] = 0
                        else:
                            parameter_dict[key] = 0

                    result = simulate_and_return_loss(
                        list(parameter_dict.values()),
                        return_results=True,
                        mon=mon,
                        model_dd_list=model_dd_list,
                        only_simulate=mode,
                    )

                    ist = np.array(
                        [
                            result["results_dict"]["mean_firing_rate_dict"][
                                f"{pop}__{mode}"
                            ]
                        ]
                    )
                    ### if sign of error changed --> went over the target --> reduce alpha
                    error_changed = (error * (target - ist) < 0).astype(bool)
                    if error_changed[0]:
                        alpha = alpha / 2
                    else:
                        alpha = alpha + (alpha_0 - alpha) * 0.1
                    ### update params depending on error
                    error = target - ist
                    diff = momentum * diff + alpha * error
                    param_arr = np.clip(param_arr + diff, 0, None)
                    ### if error is very small --> break
                    # print(f"target:{target}, ist:{ist}, diff:{diff}")
                    if (np.absolute(error) / target < tolerance).all():
                        # print("BREAK")
                        break
                    ### progress bar info
                    # pbar.set_description(f"{ist}")
                I[pop].append([lat, inp, param_arr[0]])
    return I


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
    mon = create_monitors(model_dd_list, analyze=True)

    ## ANALYZE ###
    results = simulate_and_return_loss(
        [
            2.7141971537532985,
            0.03844303223162199,
            0.36502362739781125,
            2,
            0,
            0,
            2,
            0,
            0,
            1,
        ],
        return_results=True,
        mon=mon,
        model_dd_list=model_dd_list,
        only_simulate="control",
        dump=True,
        analyze=True,
    )

    recordings = results["results_dict"]["recordings"]
    recording_times = results["results_dict"]["recording_times"]

    ### plot spikes of all 9 models / 3 populations
    plot_plan = []
    plot_plan_idx = 1
    for n_model in range(paramsS["nbr_models"]):
        for pop_name in ["str_d2", "str_fsi", "gpe_arky"]:
            plot_plan.append(f"{plot_plan_idx};{pop_name}:dd_{n_model};spike;hybrid")
            plot_plan_idx += 1
    shape = (paramsS["nbr_models"], 3)
    plot_recordings(
        "results/fit_pallido_striatal/analyze_spikes.png",
        recordings=recordings,
        recording_times=recording_times,
        chunk=0,
        shape=shape,
        plan=plot_plan,
    )

    ### plot v of gpe arky
    for variable in ["v", "u", "g_ampa", "g_gaba"]:
        plot_plan = []
        plot_plan_idx = 1
        for n_model in range(paramsS["nbr_models"]):
            for pop_name in ["gpe_arky"]:
                plot_plan.append(
                    f"{plot_plan_idx};{pop_name}:dd_{n_model};{variable};line"
                )
                plot_plan_idx += 1
        shape = (3, 3)
        plot_recordings(
            f"results/fit_pallido_striatal/analyze_{variable}.png",
            recordings=recordings,
            recording_times=recording_times,
            chunk=0,
            shape=shape,
            plan=plot_plan,
        )
