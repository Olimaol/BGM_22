# In this script the relative impact onto the activity of the target population
# for each projection of the pallido striatal network is investigated. This is
# done by obtaining the caused gaba current of each projection if the mod_f
# of the projection is 1. To obtian the relation of "changing mod_f this much
# changes activity this much" (activity/mod_f), the gaba currents are divided by the base_noise
# values, which represent the change of input current neccessary to obtain a
# specific change in firing rate i.e. activity. Finally, since the base_noise
# values consider absolute firing rate changes and the target firing rates of
# the populations are different, the obtained relations (activity/mod_f) are
# further divided by the target firing rates of the target populations.
# Final result = values for each projection representing relation activity/mod_f

from CompNeuroPy import create_dir
import numpy as np
import json

### local
from fit_pallido_striatal_hyperopt import (
    create_monitors,
    simulate_and_return_loss,
    setup_ANNarchy,
    compile_models,
    paramsS,
)


def get_I_gaba_values(I_0, I_lat_inp, proj):
    """
    Sets the mod_f of one projection to 1 and obtains the caused gaba current in the target population.
    Meanwhile all firing rates are set to the target values using I_0 and I_lat_inp.

    Args:

        I_0: dict
            Contains the base_mean values for the network with all mod_f==0
            keys are the population names.
            Used to set the correct firing rate for all populations without input.

        I_lat_inpt: dict
            Contains the base_mean values for different mod_f of the lateral and input
            projections (here lat and inp mod_f have to be either 0 or 1) of the
            populations, keys are the population names.
            Used to set the correct firing rate for the population with input from
            the given projection.

        proj: str
            The name of the projection whose mod_f is set to 1 and whose caused gaba
            current in its target population is obtained.
    """

    ### set the standard baselines to the baselines obtained from all mod_f==0
    baseline_dict = I_0.copy()

    ### get the dict entry of I_lat_inp for the given projection
    ### to obtain the baseline of the corresponding post-population
    post_pop = proj.split("__")[1]
    proj_is_lat = proj.split("__")[0] == proj.split("__")[1]

    I_lat_inp = I_lat_inp[post_pop]
    for lat, inp, base_mean, reached_target in I_lat_inp:
        if proj_is_lat:
            ### lat has to be 1 and inp has to be 0
            if lat == 1 and inp == 0:
                baseline_dict[post_pop] = base_mean
        else:
            ### lat has to be 0 and inp has to be 1
            if lat == 0 and inp == 1:
                baseline_dict[post_pop] = base_mean

    ### get the projection dict
    ### set all mod_f to zeros except the mod_f of the projection
    projection_list = []
    for key in paramsS["parameter_bound_dict"].keys():
        if not ("__" in key.split(".")[0]):
            continue

        if key.split(".")[0] == proj:
            projection_list.append(1)
        else:
            projection_list.append(0)

    ### simulate
    results = simulate_and_return_loss(
        [
            baseline_dict["str_d2"],
            baseline_dict["str_fsi"],
            baseline_dict["gpe_proto"],
            projection_list[0],
            projection_list[1],
            projection_list[2],
            projection_list[3],
            projection_list[4],
            projection_list[5],
            1,
        ],
        return_results=True,
        mon=mon,
        model_dd_list=model_dd_list,
        only_simulate="dd",
        dump=True,
        analyze=True,
    )
    recordings = results["results_dict"]["recordings"]
    mean_firing_rates = results["results_dict"]["mean_firing_rate_dict"]
    print(proj, mean_firing_rates)
    ### calculate mean g_ampa for 3 pops
    mean_I_gaba = {}
    for pop_name in ["str_d2", "str_fsi", "gpe_proto"]:
        mean_I_gaba[pop_name] = 0
        for n_model in range(paramsS["nbr_models"]):
            model_params = model_dd_list[n_model].params
            model_name_appendix = model_dd_list[n_model].name_appendix
            ### get values
            E_gaba = model_params[f"{pop_name}{model_name_appendix}.E_gaba"]
            g_gaba = recordings[0][f"{pop_name}:dd_{n_model};g_gaba"]
            v = recordings[0][f"{pop_name}:dd_{n_model};v"]
            ### calc I_gaba, do not use negative sign to obtain positive values at the end
            I_gaba = g_gaba * (v - E_gaba)
            mean_I_gaba[pop_name] += np.mean(I_gaba) / paramsS["nbr_models"]
    return mean_I_gaba[post_pop]


if __name__ == "__main__":
    if paramsS["simulation_protocol"] != "resting" or isinstance(
        paramsS["base_noise"], type(None)
    ):
        raise ValueError(
            "simulation_protocol has to be resting! and there have to be noise values!"
        )
        quit()

    create_dir("results/fit_pallido_striatal/")

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

    ### load fitted baselines for mod_f==0 for each projection
    with open("fit_pallido_striatal_archive/I_0_dd_mod_f_0_1.json") as f:
        I_0 = json.load(f)
    ### load fitted baselines for mod_f==1 for each projection
    with open("fit_pallido_striatal_archive/I_lat_inp_dd_mod_f_0_1.json") as f:
        I_lat_inp = json.load(f)
    ### for each projection get I_gaba of target pop when mod_f==1
    I_gaba_values_dict = {}
    for proj in [
        "str_d2__gpe_proto",
        "str_d2__str_d2",
        "gpe_proto__str_fsi",
        "gpe_proto__gpe_proto",
        "str_fsi__str_d2",
        "str_fsi__str_fsi",
    ]:
        I_gaba_values_dict[proj] = get_I_gaba_values(I_0, I_lat_inp, proj)

    ### use base noise values to normalize the I_gaba_values
    ### I_gaba_values = current / mod_f
    ### base_noise_values = current / activity
    ### relation activity / mod_f = I_gaba_values / base_noise_values
    activity_by_mod_f_dict = {}
    for key, val in I_gaba_values_dict.items():
        post_pop = key.split("__")[1]
        base_noise_value = paramsS["base_noise"][post_pop]
        activity_by_mod_f_dict[key] = I_gaba_values_dict[key] / base_noise_value

    ### activity here referes to the difference between 20Hz and 10Hz --> absolute
    ### make activity relative (divide by target firing rate in dd)
    mean_firing_rate_dict_target = {
        "str_d2__control": 2,  # [2, 0.24],
        "str_d2__dd": 5,  # [5, 0.63],
        "gpe_proto__control": 24.5,  # [24.5, 1.14],
        "gpe_proto__dd": 18.9,  # [18.9, 0.87],
        "str_fsi__control": 21.4,  # [21.4, 0.75],
        "str_fsi__dd": 23.7,  # [23.7, 0.69],
    }
    for key, val in activity_by_mod_f_dict.items():
        post_pop = key.split("__")[1]
        activity_by_mod_f_dict[key] = (
            activity_by_mod_f_dict[key]
            / mean_firing_rate_dict_target[f"{post_pop}__dd"]
        )
    with open("results/fit_pallido_striatal/activity_by_mod_f.json", "w") as f:
        json.dump(activity_by_mod_f_dict, f)
    f.close()
