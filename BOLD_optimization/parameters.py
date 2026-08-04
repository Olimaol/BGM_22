### parameters for test_microcircuit
parameters_test_microcircuit = {}
### general
parameters_test_microcircuit["timestep"] = 0.1
parameters_test_microcircuit["seed"] = 42
### simulation time
parameters_test_microcircuit["t.duration"] = None  # whole data time
parameters_test_microcircuit["t.rampup"] = 2310  # I match this to TR of BOLD
# short run used only to score the firing rates; must be a multiple of
# update_time because v07 hands its inputs to ANNarchy one chunk at a time
parameters_test_microcircuit["t.firing_rate_sim"] = 10000

### v07 model (Microcircuit + CorticalInputs)
parameters_test_microcircuit["mc.nx"] = 10
parameters_test_microcircuit["mc.b"] = 10
parameters_test_microcircuit["update_time"] = 100.0
# where the precomputed input spike counts live; on the workstations point this
# at /scratch/olmai/... , the caches are ~138 GiB per DBS condition
parameters_test_microcircuit["mc_ci_cache_dir"] = "../mc_ci_cache"
# number of cortical input neurons per receiver TODO use lit motivated values
parameters_test_microcircuit["ci.n_thal"] = 1000
parameters_test_microcircuit["ci.n_gpe_arky"] = 500
parameters_test_microcircuit["ci.n_gpe_cp"] = 500
parameters_test_microcircuit["ci.n_stn"] = 500
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
parameters_test_microcircuit["deap_cma.lambda"] = 12
