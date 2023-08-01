from CompNeuroPy import create_dir, plot_recordings
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


def calc_rate():
    pass


def silence_pop(pop):
    while rate > 0:
        increase_mod_factor()

        results = simulate_and_return_loss(
            [
                2.7141971537532985,
                0.03844303223162199,
                0.22132777314884586,
                3,
                0,
                3,
                0,
                3,
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

        rate = calc_rate(recordings, recording_times)


def get_proj_name_list():
    proj_name_list = []
    for key in paramsS.keys():
        compartment = key.split(".")[0]
        if len(compartment.split("__")):
            proj_name_list.append(compartment)
    return proj_name_list


def get_proj_mod_factors():
    proj_name_list = get_proj_name_list()
    for proj in proj_name_list:
        corresponding_pop = proj.split("__")[1]
        silence_pop(corresponding_pop)


def get_ampa_values(I_0):
    ### simulate once with I_0 and record ampa values
    results = simulate_and_return_loss(
        [
            I_0["str_d2"],
            I_0["str_fsi"],
            I_0["gpe_proto"],
            0,
            0,
            0,
            0,
            0,
            0,
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
    ### calculate mean g_ampa for 3 pops
    mean_g_ampa = {}
    for pop_name in ["str_d2", "str_fsi", "gpe_proto"]:
        mean_g_ampa[pop_name] = 0
        for n_model in range(paramsS["nbr_models"]):
            mean_g_ampa[pop_name] += (
                np.mean(recordings[0][f"{pop_name}:dd_{n_model};g_ampa"])
                / paramsS["nbr_models"]
            )
    return mean_g_ampa


if __name__ == "__main__":
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

    ### get ampa values from I_0 model
    with open("results/fit_pallido_striatal/I_0_dd.json") as f:
        I_0 = json.load(f)
    ampa_values = get_ampa_values(I_0)
    print(ampa_values)
    quit()
    get_proj_mod_factors()
