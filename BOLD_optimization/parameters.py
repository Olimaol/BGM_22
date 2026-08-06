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
parameters_test_microcircuit["t.firing_rate_sim"] = 9900
# The rate probe gates the BOLD run: above this firing-rate loss the BOLD run is
# skipped and charged 1.0. Both losses are in [0, 1]; the rate loss is
# 1 - mean(logistic goodness) over 18 populations, which is ~0.02 when every
# population sits at its band centre and 0.5 when every population sits exactly
# on a band edge. 0.5 therefore means "roughly band-edge plausible or better".
# NOT yet calibrated against real v07 rate losses -- see TODO.md.
parameters_test_microcircuit["firing_rate_gate"] = 0.5

### v07 model (Microcircuit + CorticalInputs)
parameters_test_microcircuit["mc.nx"] = 10
parameters_test_microcircuit["mc.b"] = 10
# v07 hands its cached inputs to ANNarchy one update_time chunk at a time, so
# every simulated stretch has to be a whole number of chunks. The stretches are
# the ramp-up (2310 ms = 1 TR), the rest of the run ((n_trs - 1) * 2310 ms) and
# the firing-rate probe. 110 ms divides 2310 (21 chunks per TR) and 9900;
# the previous 100 ms divided none of them.
parameters_test_microcircuit["update_time"] = 110.0
# Rates assumed for the striatal neurons that surround the simulated lattice but
# are not themselves simulated; the missing-GABA spike counts are drawn at these,
# so they are baked into the input caches. dSPN/iSPN are the parkinsonian Off
# state (levodopa withdrawn) of Liang et al. 2008 -- see
# ../experimental_data/activity_striatum/README.md for the derivation, and keep
# these in sync with the plausible bands in get_loss.get_firing_rate_loss.
parameters_test_microcircuit["mc.firing_rate_dict"] = {
    "FS": 10.0,
    "dSPN": 25.0,
    "iSPN": 33.0,
}
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
