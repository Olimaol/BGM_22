import argparse
import json
from pathlib import Path

from CompNeuroPy import DeapCma, run_script_parallel, save_variables, load_variables
import numpy as np
from parameters import parameters_test_microcircuit as paramsS


parser = argparse.ArgumentParser(description="Run DEAP CMA-ES optimization for BOLD.")
parser.add_argument(
    "--dbs",
    type=str,
    required=True,
    choices=["on", "off"],
    help="DBS condition passed to get_loss.py. Required: silently defaulting "
    "here means a whole optimization can run against the wrong condition.",
)
parser.add_argument(
    "--optimization-run",
    type=int,
    default=0,
    help="Run identifier used in compile appendix (e.g., 0, 1, 2).",
)
args = parser.parse_args()
dbs_condition = args.dbs
optimization_run = args.optimization_run

if dbs_condition == "on":
    # load the fitted parameters of the on DBS fits
    # first load all deap cma results of the off DBS optimizations
    result_folder = (
        Path(__file__).resolve().parent / paramsS["data_folder"] / "deap_cma_result"
    )
    available_deap_cma_result_files = sorted(
        [p.name for p in result_folder.glob("deap_cma_result_off_run_*")]
    )
    # without .pkl extension
    available_deap_cma_result_files = [f[:-4] for f in available_deap_cma_result_files]

    # load these files using load_variables
    loaded_dict = load_variables(
        name_list=available_deap_cma_result_files,
        path=paramsS["data_folder"] + "/deap_cma_result",
    )

    # now go over each of these deap_cma_result varaibles (which are dicts) and find the one with the lowest value for the key 'best_fitness'
    best_loss = float("inf")
    best_param_dict = None
    for key, deap_cma_result in loaded_dict.items():
        if deap_cma_result["best_fitness"] < best_loss:
            best_loss = deap_cma_result["best_fitness"]
            best_param_dict = {
                param_name: deap_cma_result[param_name]
                for param_name in [f"param{i}" for i in range(21)]
            }
    best_params_from_off_dbs = [best_param_dict[f"param{i}"] for i in range(21)]


### for DeapCma we need to define the evaluate_function
def evaluate_function(population):

    loss_list = []
    loss_folder = Path(__file__).resolve().parent / paramsS["data_folder"]

    # convert the parameter lists to list of strings
    population_str = [list(map(str, individual)) for individual in population]

    # if dbs=off add the last three parameters as fixed strings = parameters affecting dbs
    if dbs_condition == "off":
        population_str = [individual + ["0"] * 3 for individual in population_str]
    # if dbs=on add the first 9 parameters as fixed strings = parameters affecting firing rates
    # here use the fits of dbs off
    else:  # dbs_condition == "on"
        fixed_params = [best_params_from_off_dbs[i] for i in range(9)]
        fixed_params_str = list(map(str, fixed_params))
        population_str = [
            fixed_params_str + individual for individual in population_str
        ]

    run_script_parallel(
        script_path="get_loss.py",
        n_jobs=len(population_str),
        args_list=[
            [
                "--compile-appendix",
                f"run_{optimization_run}_ind{i}",
                "--dbs",
                dbs_condition,
            ]
            + individual
            for i, individual in enumerate(population_str)
        ],
    )

    # load the losses of the individuals that were stored by get_loss.py
    for i, _ in enumerate(population):
        loss_file = loss_folder / f"loss_run_{optimization_run}_ind{i}.json"
        if not loss_file.exists():
            raise FileNotFoundError(
                f"Expected loss file for individual {i} at {loss_file}, but it was not found."
            )

        with open(loss_file, "r", encoding="ascii") as f:
            loss_payload = json.load(f)

        if "total_loss" not in loss_payload:
            raise KeyError(
                f"Loss file {loss_file} does not contain a 'total_loss' entry."
            )

        loss_list.append((float(loss_payload["total_loss"]),))

    return loss_list


if dbs_condition == "off":
    ### define lower bounds of paramters to optimize
    lb = np.array([0] * 21)

    ### define upper bounds of paramters to optimize
    ub = np.array([500, 500, 800, 10, 10, 10, 10, 500, 500] + [5] * 12)

    ### initial values = without excitatory input and original weights
    p0 = np.array([0] * 9 + [1] * 12)
else:  # dbs_condition == "on"
    ### define lower bounds of paramters to optimize
    lb = np.array([0] * 15)

    ### define upper bounds of paramters to optimize
    ub = np.array([5] * 12 + [10, 1, 1])

    ### initial values = use fits from dbs off and dbs parameters set to zero
    initial_params_from_off = [best_params_from_off_dbs[i] for i in range(9, 21)]
    p0 = np.array(initial_params_from_off + [0, 0, 0])

### create an "minimal" instance of the DeapCma class
deap_cma = DeapCma(
    lower=lb,
    upper=ub,
    evaluate_function=evaluate_function,
    max_evals=paramsS["deap_cma.run.max_evals"],
    p0=p0,
    hard_bounds=True,
    plot_file=(
        f"{paramsS['data_folder']}/deap_cma_plot_{dbs_condition}_run_{optimization_run}.png"
    ),
    cma_params_dict={
        "lambda_": paramsS["deap_cma.lambda"]
    },  # TODO set this depending on how many parallel jobs we can run
)

# run the get_loss script the number of individuals times to compile the models
number_of_individuals = deap_cma.deap_dict["strategy"].lambda_

run_script_parallel(
    script_path="get_loss.py",
    n_jobs=number_of_individuals,
    args_list=[
        [
            "--compile",
            "--compile-appendix",
            f"run_{optimization_run}_ind{i}",
            "--dbs",
            dbs_condition,
        ]+["0"]*24
        for i in range(number_of_individuals)
    ],
)

### run the optimization
deap_cma_result = deap_cma.run()

save_variables(
    variable_list=[deap_cma_result],
    name_list=[f"deap_cma_result_{dbs_condition}_run_{optimization_run}"],
    path=f"{paramsS['data_folder']}/deap_cma_result",
)
