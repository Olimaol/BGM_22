# In this script, a base_noise value is obtained for each population of the striato_pallidal network.
# This represents the baseline current difference necessary to obtain an increase in firing rate from
# 10 Hz to 20 Hz.

from CompNeuroPy import create_dir
import json

### local
from fit_pallido_striatal_hyperopt import (
    create_monitors,
    setup_ANNarchy,
    compile_models,
    paramsS,
)
from fit_pallido_striatal_base_mean import get_I_0


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
        "str_d2__control": 10,
        "str_d2__dd": 20,
        "gpe_proto__control": 10,
        "gpe_proto__dd": 20,
        "str_fsi__control": 10,
        "str_fsi__dd": 20,
    }
    ### get increase_noise values for lateral and input mod_factors = 0
    ### for rates 10Hz and 20 Hz
    I_0 = {
        "10Hz": get_I_0(
            "control",
            mean_firing_rate_dict_target,
            mon=mon,
            model_dd_list=model_dd_list,
        ),
        "20Hz": get_I_0(
            "dd",
            mean_firing_rate_dict_target,
            mon=mon,
            model_dd_list=model_dd_list,
        ),
    }
    ### use difference between 10Hz and 20Hz for noise
    I_0["base_noise"] = {
        pop_name: I_0["20Hz"][pop_name] - I_0["10Hz"][pop_name]
        for pop_name in I_0["20Hz"].keys()
    }

    with open("results/fit_pallido_striatal/I_0_10Hz_20Hz.json", "a") as f:
        json.dump(I_0, f)
    f.close()
