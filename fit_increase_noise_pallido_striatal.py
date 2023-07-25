from CompNeuroPy import create_dir
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

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
    momentum = 0
    n_iter_max = 60
    mean_firing_rate_dict_target = {
        "str_d2__control": 2,  # [2, 0.24],
        "str_d2__dd": 5,  # [5, 0.63],
        "gpe_arky__control": 24.5,  # [24.5, 1.14],
        "gpe_arky__dd": 18.9,  # [18.9, 0.87],
        "str_fsi__control": 21.4,  # [21.4, 0.75],
        "str_fsi__dd": 23.7,  # [23.7, 0.69],
    }

    pbar = tqdm(range(n_iter_max))
    param_arr = np.array([1, 1, 1])
    alpha = np.ones(len(param_arr)) * alpha_0
    alpha_0 = alpha.copy()

    target = np.array(
        [mean_firing_rate_dict_target[f"{pop}__{mode}"] for pop in pop_list]
    )
    ### iteration loop to find paramters with target firing rates
    diff = 0
    error = np.zeros(len(param_arr))
    print(target)
    optimization_values_param_list = []
    optimization_values_ist_list = []
    for _ in pbar:
        result = simulate_and_return_loss(
            [param_arr[0], param_arr[1], param_arr[2], 0, 0, 0, 0, 0, 0, 1],
            return_results=True,
            mon=mon,
            model_dd_list=model_dd_list,
            only_simulate=mode,
            dump=False,
        )

        ist = np.array(
            [
                result["results_dict"]["mean_firing_rate_dict"][f"{pop}__{mode}"][0]
                for pop in pop_list
            ]
        )

        optimization_values_param_list.append(param_arr)
        optimization_values_ist_list.append(ist)
        ### if sign of error changed --> went over the target --> reduce alpha
        new_error = target - ist
        decrease_alpha_mask_0 = (error * new_error < 0).astype(bool)
        alpha[decrease_alpha_mask_0] = alpha[decrease_alpha_mask_0] / 2
        ### also decrease alpha if relative change of error ist too large or even positive
        sign_error = np.sign(error)
        sign_error[sign_error == 0] = 1
        sign_error = sign_error * 1e-20
        error = error + sign_error
        d_error_rel = (new_error - error) / error
        decrease_alpha_mask_1 = (
            (d_error_rel < -0.1).astype(int) + (d_error_rel > 0).astype(int)
        ) > 0
        alpha[decrease_alpha_mask_1] = alpha[decrease_alpha_mask_1] / 2
        alpha_0[decrease_alpha_mask_1] = alpha_0[decrease_alpha_mask_1] / 2
        alpha[np.logical_not(decrease_alpha_mask_1)] = (
            alpha[np.logical_not(decrease_alpha_mask_1)] * 2
        )
        alpha_0[np.logical_not(decrease_alpha_mask_1)] = (
            alpha_0[np.logical_not(decrease_alpha_mask_1)] * 2
        )
        ### increase learning speed if correct direction but very small
        too_small_change = tolerance * target / 2
        mask_faster = (
            (np.abs(new_error - error) < too_small_change).astype(int)
            * (d_error_rel < 0).astype(int)
        ).astype(bool)
        alpha_0[mask_faster] = alpha_0[mask_faster] * 1.5
        alpha[mask_faster] = alpha[mask_faster] * 1.5
        ### all alphas not decerased --> go to alpha_0
        all_decrease_alpha_mask = (
            decrease_alpha_mask_0.astype(int) + decrease_alpha_mask_1.astype(int)
        ) > 0
        alpha[np.logical_not(all_decrease_alpha_mask)] = (
            alpha[np.logical_not(all_decrease_alpha_mask)]
            + (
                alpha_0[np.logical_not(all_decrease_alpha_mask)]
                - alpha[np.logical_not(all_decrease_alpha_mask)]
            )
            * 0.1
        )

        # print("DO RUN")
        # print(f"{'param_arr':<15}: {np.round(param_arr,2)}")
        # print(f"{'ist':<15}: {np.round(ist,2)}")
        # print(f"{'target':<15}: {np.round(target,2)}")
        # print(f"{'alpha:':<15}: {np.round(alpha,2)}")
        ### update params depending on error
        error = target - ist
        # print(f"{'error':15}: {np.round(error,2)}")
        # print(f"{'d_error_rel':<15}: {np.round(d_error_rel,2)}")
        diff = momentum * diff + alpha * error
        # print(f"{'diff':<15}: {np.round(diff,2)}")
        param_arr_prev = param_arr.copy()
        # print(f"{'param_arr_prev':<15}: {np.round(param_arr_prev,2)}")
        param_arr = np.clip(param_arr + diff, 0, None)
        diff = param_arr - param_arr_prev
        # print(f"{'param_arr':<15}: {np.round(param_arr,2)}")
        # print(f"{'diff':<15}: {np.round(param_arr,2)}")
        # print("FIN")
        ### if error is very small --> break
        if (np.absolute(error) / target < tolerance).all():
            break
        ### progress bar info
        pbar.set_description(f"{ist}")
    ### plot of fitting rung
    plt.figure(dpi=300)
    plt.subplot(211)
    plt.plot(np.array(optimization_values_ist_list))
    plt.subplot(212)
    plt.plot(np.array(optimization_values_param_list))
    plt.tight_layout()
    plt.savefig(f"results/fit_pallido_striatal/opt_run_{mode}.png")

    I_0 = {pop_list[idx]: param_arr[idx] for idx in range(len(param_arr))}

    with open(f"results/fit_pallido_striatal/I_0_{mode}.json", "w") as f:
        json.dump(I_0, f)
    f.close()

    return I_0


def get_I_lat_inp(mode, I_0):
    ### parameters
    alpha_0 = 0.1
    tolerance = 0.02
    momentum = 0
    n_iter_max = 60
    mean_firing_rate_dict_target = {
        "str_d2__control": 2,  # [2, 0.24],
        "str_d2__dd": 5,  # [5, 0.63],
        "gpe_arky__control": 24.5,  # [24.5, 1.14],
        "gpe_arky__dd": 18.9,  # [18.9, 0.87],
        "str_fsi__control": 21.4,  # [21.4, 0.75],
        "str_fsi__dd": 23.7,  # [23.7, 0.69],
    }

    pop_list = ["gpe_arky"]  # ["str_d2", "str_fsi", "gpe_arky"]
    ### list with lateral mod_factors
    lat_list = [
        0.1,
        2,
    ]  # np.array([(idx_lat_list + 1) * 0.1 for idx_lat_list in range(31)])
    ### list with input mod_factors
    inp_list = [
        0.1,
        2,
    ]  # np.array([(idx_inp_list + 1) * 0.1 for idx_inp_list in range(31)])
    I = {}
    global_idx = 0
    for pop in pop_list:
        I[pop] = []
        I[pop].append([0, 0, I_0[mode][pop]])
        for lat in lat_list:
            for inp in inp_list:
                pbar = tqdm(range(n_iter_max))
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
                error = np.zeros(len(param_arr))
                print(target)
                optimization_values_param_list = []
                optimization_values_ist_list = []
                for opt_run in pbar:
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
                        dump=False,
                    )

                    ist = np.array(
                        [
                            result["results_dict"]["mean_firing_rate_dict"][
                                f"{pop}__{mode}"
                            ][0]
                        ]
                    )

                    optimization_values_param_list.append(param_arr)
                    optimization_values_ist_list.append(ist)
                    ### if sign of error changed --> went over the target --> reduce alpha
                    new_error = target - ist
                    decrease_alpha_mask_0 = (error * new_error < 0).astype(bool)
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
                    if decrease_alpha_mask_1[0]:
                        alpha = alpha / 2
                        alpha_0 = alpha_0 / 2
                    else:
                        alpha = alpha * 2
                        alpha_0 = alpha_0 * 2
                    ### increase learning speed if correct direction but very small
                    too_small_change = tolerance * target / 2
                    mask_faster = (
                        (np.abs(new_error - error) < too_small_change).astype(int)
                        * (d_error_rel < 0).astype(int)
                    ).astype(bool)
                    if mask_faster[0]:
                        alpha_0 = alpha_0 * 1.5
                        alpha = alpha * 1.5
                    ### all alphas not decerased --> go to alpha_0
                    all_decrease_alpha_mask = (
                        decrease_alpha_mask_0.astype(int)
                        + decrease_alpha_mask_1.astype(int)
                    ) > 0
                    if not (all_decrease_alpha_mask):
                        alpha = alpha + (alpha_0 - alpha) * 0.1

                    # print("DO RUN")
                    # print(f"{'param_arr':<15}: {np.round(param_arr,2)}")
                    # print(f"{'ist':<15}: {np.round(ist,2)}")
                    # print(f"{'target':<15}: {np.round(target,2)}")
                    # print(f"{'alpha:':<15}: {np.round(alpha,2)}")
                    ### update params depending on error
                    error = target - ist
                    # print(f"{'error':15}: {np.round(error,2)}")
                    # print(f"{'d_error_rel':<15}: {np.round(d_error_rel,2)}")
                    diff = momentum * diff + alpha * error
                    # print(f"{'diff':<15}: {np.round(diff,2)}")
                    param_arr_prev = param_arr.copy()
                    # print(f"{'param_arr_prev':<15}: {np.round(param_arr_prev,2)}")
                    param_arr = np.clip(param_arr + diff, 0, None)
                    diff = param_arr - param_arr_prev
                    # print(f"{'param_arr':<15}: {np.round(param_arr,2)}")
                    # print(f"{'diff':<15}: {np.round(param_arr,2)}")
                    # print("FIN")
                    ### if error is very small --> break
                    if (np.absolute(error) / target < tolerance).all():
                        break
                    ### progress bar info
                    pbar.set_description(f"{ist}")
                ### plot of fitting rung
                plt.figure(dpi=300)
                plt.subplot(211)
                plt.plot(np.array(optimization_values_ist_list))
                plt.subplot(212)
                plt.plot(np.array(optimization_values_param_list))
                plt.tight_layout()
                plt.savefig(
                    f"results/fit_pallido_striatal/opt_run_{mode}_{pop}_{lat}_{inp}.png"
                )

                I[pop].append([lat, inp, param_arr[0]])

                with open(
                    f"results/fit_pallido_striatal/I_lat_inp_{mode}.json", "w"
                ) as f:
                    json.dump(I, f)
                f.close()

                print(
                    f"{global_idx}/{len(pop_list)*len(lat_list)*len(inp_list)} with {opt_run} runs"
                )
                global_idx += 1
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
    mon = create_monitors(model_dd_list)

    ## OPTIMIZE ###
    ### get increase_noise values for lateral and input mod_factors = 0
    I_0 = {
        "control": get_I_0("control"),
        "dd": get_I_0("dd"),
    }
    quit()
    ### get increase_noise values for different lateral and input mod_factors
    I_lat_inp = {
        "control": get_I_lat_inp(mode="control", I_0=I_0),
        "dd": get_I_lat_inp(mode="dd", I_0=I_0),
    }

    with open("results/fit_pallido_striatal/I_lat_inp.json", "a") as f:
        json.dump(I_lat_inp, f)
    f.close()
