### parameters for test_microcircuit
parameters_test_microcircuit = {}
### general
parameters_test_microcircuit["timestep"] = 0.1
parameters_test_microcircuit["seed"] = 42
### simulation time
parameters_test_microcircuit["t.duration"] = None  # whole data time
parameters_test_microcircuit["t.rampup"] = 2310  # I match this to TR of BOLD
# paths
parameters_test_microcircuit["data_folder"] = "data_BOLD_optimization"
parameters_test_microcircuit["mc.fitted_params_path"] = (
    "../striatal_microcircuit_requirements/connectivity_parameters/connectivity_fit_data/fitted_params.json"
)
parameters_test_microcircuit["mc.cortical_rate_path"] = {
    "on": "../striatal_microcircuit_requirements/cortical_firing_rates/cortical_firing_rates_data/firing_rates_matlab_condition-on.npz",
    "off": "../striatal_microcircuit_requirements/cortical_firing_rates/cortical_firing_rates_data/firing_rates_matlab_condition-off.npz",
}
# deap cma parameters
parameters_test_microcircuit["deap_cma.run.max_evals"] = (
    2000  # should be approx 14 days
)
