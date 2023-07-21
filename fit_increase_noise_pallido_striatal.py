# from ANNarchy import (
#     setup,
# )
# from CompNeuroPy.models import BGM
from CompNeuroPy import create_dir
import numpy as np
from tqdm import tqdm

### local
# from parameters import parameters_fit_pallido_striatal as paramsS
from fit_hyperopt_pallido_striatal import (
    create_monitors,
    simulate_and_return_loss,
    setup_ANNarchy,
    compile_models,
)


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
    pop_list = ["str_d2", "str_fsi", "gpe_arky"]
    mode = "control"
    alpha = 0.01
    momentum = 0.3
    n_iter_max = 100
    pbar = tqdm(range(n_iter_max))
    param_arr = np.array([10, 0.5293052295359941, 0.1361904761905049])
    mean_firing_rate_dict_target = {
        "str_d2__control": 2,  # [2, 0.24],
        "str_d2__dd": 5,  # [5, 0.63],
        "gpe_arky__control": 24.5,  # [24.5, 1.14],
        "gpe_arky__dd": 18.9,  # [18.9, 0.87],
        "str_fsi__control": 21.4,  # [21.4, 0.75],
        "str_fsi__dd": 23.7,  # [23.7, 0.69],
    }
    target = np.array(
        [mean_firing_rate_dict_target[f"{pop}__{mode}"] for pop in pop_list]
    )
    print(target)
    diff = 0
    for n_iter in pbar:
        result = simulate_and_return_loss(
            param_arr, return_results=True, mon=mon, model_dd_list=model_dd_list
        )

        ist = np.array(
            [
                result["results_dict"]["mean_firing_rate_dict"][f"{pop}__{mode}"]
                for pop in pop_list
            ]
        )

        error = target - ist
        diff = momentum * diff + alpha * error
        param_arr = param_arr + diff
        pbar.set_description(f"{ist}")

    print(param_arr)
