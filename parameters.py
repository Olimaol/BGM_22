### define global simulation paramters
parameters_default = {}
### general
parameters_default["timestep"] = 0.1
parameters_default["seed"] = 1
parameters_default["trials"] = 1
### simulation times
parameters_default["t.init"] = 600
parameters_default["t.ssd"] = 250
parameters_default["t.decay"] = 300
parameters_default["t.cor_pause__dur"] = 5
parameters_default["t.cor_go__delay"] = 75
parameters_default["t.cor_go__delay_sd"] = 0
parameters_default["t.cor_stop__delay_cue"] = 50
parameters_default["t.cor_stop__delay_response"] = 50
parameters_default["t.cor_stop__dur_cue"] = 5
parameters_default["t.cor_stop__dur_response"] = 200
### cor_go
parameters_default["cor_go.rates"] = 400
parameters_default["cor_go.rates_sd"] = 0
parameters_default["cor_go.tau_up"] = 200
parameters_default["cor_go.tau_down"] = 10
### cor_pause
parameters_default["cor_pause.rates_go"] = 500
parameters_default["cor_pause.rates_stop"] = 600
parameters_default["cor_pause.rates_sd"] = 0
parameters_default["cor_pause.tau_up"] = 1
parameters_default["cor_pause.tau_down"] = 150
### cor_stop
parameters_default["cor_stop.rates_cue"] = 400
parameters_default["cor_stop.rates_response"] = 400
parameters_default["cor_stop.rates_sd"] = 0
parameters_default["cor_stop.tau_up"] = 1
parameters_default["cor_stop.tau_down"] = 70


### parameters for test_power
parameters_test_power = {}
### general
parameters_test_power["timestep"] = 0.1
parameters_test_power["seed"] = 1
### simulation time
parameters_test_power["t.duration"] = 1000
### cor_go
parameters_test_power["cor_go.amplitude"] = 50
parameters_test_power["cor_go.frequency"] = 50
parameters_test_power["cor_go.phase"] = 0
parameters_test_power["cor_go.base"] = 50


### parameters for test_resting
parameters_test_resting = {}
### general
parameters_test_resting["timestep"] = 0.1
parameters_test_resting["seed"] = 1
### simulation time
parameters_test_resting["t.duration"] = 3000


### parameters for test_microcircuit
parameters_test_microcircuit = {}
### general
parameters_test_microcircuit["timestep"] = 0.1
parameters_test_microcircuit["seed"] = 42
### simulation time
parameters_test_microcircuit["t.duration"] = 200
## dbs parameter
parameters_test_microcircuit["dbs"] = "off"
# microcircuit parameters
parameters_test_microcircuit["mc.nx"] = 10
parameters_test_microcircuit["mc.b"] = 10
parameters_test_microcircuit["update_time"] = 100.0

parameters_test_microcircuit["mc.fitted_params_path"] = (
    "striatal_microcircuit_requirements/connectivity_parameters/connectivity_fit_data/fitted_params.json"
)
parameters_test_microcircuit["mc.cortical_rate_path"] = {
    "on": "striatal_microcircuit_requirements/cortical_firing_rates/cortical_firing_rates_data/firing_rates_matlab_condition-on.npz",
    "off": "striatal_microcircuit_requirements/cortical_firing_rates/cortical_firing_rates_data/firing_rates_matlab_condition-off.npz",
}
# cortical inputs parameters TODO use lit motivated values
parameters_test_microcircuit["ci.n_thal"] = 1000
parameters_test_microcircuit["ci.n_gpe_arky"] = 500
parameters_test_microcircuit["ci.n_gpe_cp"] = 500
parameters_test_microcircuit["ci.n_stn"] = 500
