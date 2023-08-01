from CompNeuroPy import create_dir
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

### local
from fit_pallido_striatal_hyperopt import (
    create_monitors,
    simulate_and_return_loss,
    setup_ANNarchy,
    compile_models,
    paramsS,
)


def set_parameters(mode, pop_list, mean_firing_rate_dict_target):
    alpha_0 = 0.1
    tolerance = 0.02
    momentum = 0
    n_iter_max = 60
    target = np.array(
        [mean_firing_rate_dict_target[f"{pop}__{mode}"] for pop in pop_list]
    )

    return [alpha_0, tolerance, momentum, n_iter_max, target]


def init_parameter_search(n_iter_max, alpha_0, param_arr_start, use_tqdm=True):
    if use_tqdm:
        pbar = tqdm(range(n_iter_max))
    else:
        pbar = range(n_iter_max)
    param_arr = param_arr_start
    alpha = np.ones(len(param_arr)) * alpha_0
    alpha_0 = alpha.copy()
    diff = np.zeros(len(param_arr))
    error = np.zeros(len(param_arr))
    optimization_values_param_list = []
    optimization_values_ist_list = []
    return [
        pbar,
        param_arr,
        alpha,
        alpha_0,
        diff,
        error,
        optimization_values_param_list,
        optimization_values_ist_list,
        use_tqdm,
    ]


def improve_params_increment(
    param_arr,
    target,
    ist,
    alpha,
    alpha_0,
    tolerance,
    momentum,
    error,
    diff,
    optimization_values_param_list,
    optimization_values_ist_list,
):
    optimization_values_param_list.append(param_arr)
    optimization_values_ist_list.append(ist)
    ### if sign of error changed --> went over the target --> reduce alpha
    new_error = target - ist
    decrease_alpha_mask_0 = (error * new_error < 0).astype(bool)
    if len(decrease_alpha_mask_0) > 1:
        alpha[decrease_alpha_mask_0] = alpha[decrease_alpha_mask_0] / 2
    else:
        if decrease_alpha_mask_0[0]:
            alpha = alpha / 2
    ### also decrease alpha if relative change of error ist too large or even positive
    sign_error = np.sign(error)
    sign_error[sign_error == 0] = 1
    sign_error = sign_error * 1e-20
    error = error + sign_error
    d_error_rel = (new_error - error) / error
    decrease_alpha_mask_1 = (
        (d_error_rel < -0.1).astype(int) + (d_error_rel > 0).astype(int)
    ) > 0
    if len(decrease_alpha_mask_1) > 1:
        alpha[decrease_alpha_mask_1] = alpha[decrease_alpha_mask_1] / 2
        alpha_0[decrease_alpha_mask_1] = alpha_0[decrease_alpha_mask_1] / 2
        alpha[np.logical_not(decrease_alpha_mask_1)] = (
            alpha[np.logical_not(decrease_alpha_mask_1)] * 2
        )
        alpha_0[np.logical_not(decrease_alpha_mask_1)] = (
            alpha_0[np.logical_not(decrease_alpha_mask_1)] * 2
        )
    else:
        if decrease_alpha_mask_1[0]:
            alpha = alpha / 2
            alpha_0 = alpha_0 / 2
        else:
            alpha = alpha * 2
            alpha_0 = alpha_0 * 2

    ### increase learning speed if correct direction but very small
    too_small_change = tolerance * target / 2
    mask_faster = (
        (np.abs(new_error - error) < too_small_change).astype(int)  # too small
        * (d_error_rel < 0).astype(int)  # correct direction
        * (np.absolute(error) / target > tolerance).astype(
            int
        )  # error is larger than tollerance
    ).astype(bool)
    if len(mask_faster) > 1:
        alpha_0[mask_faster] = alpha_0[mask_faster] * 1.5
        alpha[mask_faster] = alpha[mask_faster] * 1.5
    else:
        if mask_faster[0]:
            alpha_0 = alpha_0 * 1.5
            alpha = alpha * 1.5

    ### all alphas not decerased --> go to alpha_0
    all_decrease_alpha_mask = (
        decrease_alpha_mask_0.astype(int) + decrease_alpha_mask_1.astype(int)
    ) > 0
    if len(all_decrease_alpha_mask) > 1:
        alpha[np.logical_not(all_decrease_alpha_mask)] = (
            alpha[np.logical_not(all_decrease_alpha_mask)]
            + (
                alpha_0[np.logical_not(all_decrease_alpha_mask)]
                - alpha[np.logical_not(all_decrease_alpha_mask)]
            )
            * 0.1
        )
    else:
        if not (all_decrease_alpha_mask):
            alpha = alpha + (alpha_0 - alpha) * 0.1

    ### update params depending on error
    error = target - ist
    diff_pre = diff
    diff = momentum * diff + alpha * error
    ### if diff has opposite sign --> clip it to not jump over the previous errors in the opposite direction
    ### i.e. new diff should not be larger than previous diff in opposite direction
    mask_changed_sign_diff_pos = (
        decrease_alpha_mask_0.astype(int) * (diff >= 0).astype(int)
    ).astype(bool)
    mask_changed_sign_diff_neg = (
        decrease_alpha_mask_0.astype(int) * (diff < 0).astype(int)
    ).astype(bool)
    diff[mask_changed_sign_diff_pos] = np.clip(
        diff[mask_changed_sign_diff_pos], None, -diff_pre[mask_changed_sign_diff_pos]
    )
    diff[mask_changed_sign_diff_neg] = np.clip(
        diff[mask_changed_sign_diff_neg], -diff_pre[mask_changed_sign_diff_neg], None
    )
    param_arr = param_arr + diff

    return [
        param_arr,
        alpha,
        error,
        diff,
        optimization_values_param_list,
        optimization_values_ist_list,
    ]


def get_I_0(mode, mean_firing_rate_dict_target, mon, model_dd_list):
    ### parameters
    pop_list = ["str_d2", "str_fsi", "gpe_proto"]
    alpha_0, tolerance, momentum, n_iter_max, target = set_parameters(
        mode, pop_list, mean_firing_rate_dict_target
    )
    ### initialize parameters search
    (
        pbar,
        param_arr,
        alpha,
        alpha_0,
        diff,
        error,
        optimization_values_param_list,
        optimization_values_ist_list,
        use_tqdm,
    ) = init_parameter_search(
        n_iter_max, alpha_0, param_arr_start=np.ones(len(pop_list))
    )
    ### iteration loop to find paramters with target firing rates
    print(target, end="\n\n")
    for _ in pbar:
        ### TODO seting base_noise = base_mean*0.1 does not work for gpe, because it does not need base_mean to reach firing rate... but what should be noise?
        result = simulate_and_return_loss(
            [param_arr[0], param_arr[1], param_arr[2], 0, 0, 0, 0, 0, 0, 1],
            return_results=True,
            mon=mon,
            model_dd_list=model_dd_list,
            only_simulate=mode,
            dump=True,
        )

        ist = np.array(
            [
                result["results_dict"]["mean_firing_rate_dict"][f"{pop}__{mode}"][0]
                for pop in pop_list
            ]
        )

        (
            param_arr,
            alpha,
            error,
            diff,
            optimization_values_param_list,
            optimization_values_ist_list,
        ) = improve_params_increment(
            param_arr,
            target,
            ist,
            alpha,
            alpha_0,
            tolerance,
            momentum,
            error,
            diff,
            optimization_values_param_list,
            optimization_values_ist_list,
        )

        ### if error is very small --> break
        if (np.absolute(error) / target < tolerance).all():
            break
        ### progress bar info
        if use_tqdm:
            pbar.set_description(f"{ist}")

    ### plot of fitting rung
    plt.figure(dpi=300)
    plt.subplot(211)
    plt.plot(np.array(optimization_values_ist_list))
    plt.subplot(212)
    plt.plot(np.array(optimization_values_param_list))
    plt.tight_layout()
    plt.savefig(f"results/fit_pallido_striatal/opt_run_figs/{mode}.png")

    I_0 = {pop_list[idx]: param_arr[idx] for idx in range(len(param_arr))}

    with open(f"results/fit_pallido_striatal/I_0_{mode}.json", "w") as f:
        json.dump(I_0, f)
    f.close()

    return I_0


def get_param_start_arr(I, pop, lat, inp):
    I_entry_loss = np.zeros(len(I[pop]))
    I_entry_arr = np.zeros(len(I[pop]))
    for I_entry_idx, I_entry in enumerate(I[pop]):
        I_entry_loss[I_entry_idx] = abs(I_entry[0] - lat) + abs(I_entry[1] - inp)
        I_entry_arr[I_entry_idx] = I_entry[2]
    param_start_value = I_entry_arr[np.argmin(I_entry_loss)]
    return np.array([param_start_value])


def get_parameter_dict_lat_inp(pop, param_arr, lat, inp, mode):
    parameter_dict = {}
    for key in paramsS["parameter_bound_dict"]:
        compartment, param_name = key.split(".")
        if pop == compartment:
            ### parameter of population varied
            parameter_dict[key] = param_arr[0]
        elif (
            "str_d2" == compartment
            or "str_fsi" == compartment
            or "gpe_proto" == compartment
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
    return parameter_dict


def get_I_lat_inp(mode, I_0, mean_firing_rate_dict_target, mon, model_dd_list):

    pop_list = ["str_d2", "str_fsi", "gpe_proto"]
    ### list with lateral mod_factors
    lat_list = [0.1]
    ### list with input mod_factors
    inp_list = [0.1, 0.2]
    I = {}
    with tqdm(total=len(pop_list) * len(lat_list) * len(inp_list)) as global_pbar:
        for pop in pop_list:
            I[pop] = []
            I[pop].append([0, 0, I_0[mode][pop], True])
            for lat in lat_list:
                for inp in inp_list:
                    ### parameters for paramter search
                    (
                        alpha_0,
                        tolerance,
                        momentum,
                        n_iter_max,
                        target,
                    ) = set_parameters(
                        mode,
                        pop_list=[pop],
                        mean_firing_rate_dict_target=mean_firing_rate_dict_target,
                    )
                    ### initialize parameters search
                    param_arr_start = get_param_start_arr(I, pop, lat, inp)
                    (
                        pbar,
                        param_arr,
                        alpha,
                        alpha_0,
                        diff,
                        error,
                        optimization_values_param_list,
                        optimization_values_ist_list,
                        use_tqdm,
                    ) = init_parameter_search(
                        n_iter_max,
                        alpha_0,
                        param_arr_start=param_arr_start,
                        use_tqdm=False,
                    )
                    reached_target = False
                    ### iteration loop to find paramter with target firing rate
                    print(f"\n{target}", end="\n\n")
                    for _ in pbar:
                        parameter_dict = get_parameter_dict_lat_inp(
                            pop, param_arr, lat, inp, mode
                        )

                        result = simulate_and_return_loss(
                            list(parameter_dict.values()),
                            return_results=True,
                            mon=mon,
                            model_dd_list=model_dd_list,
                            only_simulate=mode,
                            dump=False,
                        )

                        ist = np.array(
                            [
                                result["results_dict"]["mean_firing_rate_dict"][
                                    f"{pop}__{mode}"
                                ][0]
                            ]
                        )

                        (
                            param_arr,
                            alpha,
                            error,
                            diff,
                            optimization_values_param_list,
                            optimization_values_ist_list,
                        ) = improve_params_increment(
                            param_arr,
                            target,
                            ist,
                            alpha,
                            alpha_0,
                            tolerance,
                            momentum,
                            error,
                            diff,
                            optimization_values_param_list,
                            optimization_values_ist_list,
                        )

                        ### if error is very small --> break
                        if (np.absolute(error) / target < tolerance).all():
                            reached_target = True
                            break
                        ### progress bar info
                        if use_tqdm:
                            pbar.set_description(f"{ist}")

                    ### plot of fitting rung
                    plt.figure(dpi=300)
                    plt.subplot(211)
                    plt.plot(np.array(optimization_values_ist_list))
                    plt.subplot(212)
                    plt.plot(np.array(optimization_values_param_list))
                    plt.tight_layout()
                    plt.savefig(
                        f"results/fit_pallido_striatal/opt_run_figs/{mode}_{pop}_{lat}_{inp}.png"
                    )

                    I[pop].append([lat, inp, param_arr[0], reached_target])

                    with open(
                        f"results/fit_pallido_striatal/I_lat_inp_{mode}.json", "w"
                    ) as f:
                        json.dump(I, f)
                    f.close()

                    global_pbar.update()
    return I


if __name__ == "__main__":
    if paramsS["simulation_protocol"] != "resting":
        raise ValueError("simulation_protocol has to be resting!")
        quit()

    create_dir("results/fit_pallido_striatal/", clear=True)
    create_dir("results/fit_pallido_striatal/opt_run_figs", clear=True)

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
    mean_firing_rate_dict_target = {
        "str_d2__control": 2,  # [2, 0.24],
        "str_d2__dd": 5,  # [5, 0.63],
        "gpe_proto__control": 24.5,  # [24.5, 1.14],
        "gpe_proto__dd": 18.9,  # [18.9, 0.87],
        "str_fsi__control": 21.4,  # [21.4, 0.75],
        "str_fsi__dd": 23.7,  # [23.7, 0.69],
    }
    ### get increase_noise values for lateral and input mod_factors = 0
    I_0 = {
        "control": get_I_0("control", mean_firing_rate_dict_target, mon, model_dd_list),
        "dd": get_I_0("dd", mean_firing_rate_dict_target, mon, model_dd_list),
    }
    with open("results/fit_pallido_striatal/I_0.json", "a") as f:
        json.dump(I_0, f)
    f.close()
    ### get increase_noise values for different lateral and input mod_factors
    I_lat_inp = {
        "control": get_I_lat_inp(
            mode="control",
            I_0=I_0,
            mean_firing_rate_dict_target=mean_firing_rate_dict_target,
            mon=mon,
            model_dd_list=model_dd_list,
        ),
        "dd": get_I_lat_inp(
            mode="dd",
            I_0=I_0,
            mean_firing_rate_dict_target=mean_firing_rate_dict_target,
            mon=mon,
            model_dd_list=model_dd_list,
        ),
    }

    with open("results/fit_pallido_striatal/I_lat_inp.json", "a") as f:
        json.dump(I_lat_inp, f)
    f.close()
