import json
import warnings

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
parameters_default["cor_go.rates"] = 0  # 400
parameters_default["cor_go.rates_sd"] = 0
parameters_default["cor_go.tau_up"] = 200
parameters_default["cor_go.tau_down"] = 10
### cor_pause
parameters_default["cor_pause.rates_go"] = 0  # 500
parameters_default["cor_pause.rates_stop"] = 0  # 600
parameters_default["cor_pause.rates_sd"] = 0
parameters_default["cor_pause.tau_up"] = 1
parameters_default["cor_pause.tau_down"] = 150
### cor_stop
parameters_default["cor_stop.rates_cue"] = 0  # 400
parameters_default["cor_stop.rates_response"] = 0  # 400
parameters_default["cor_stop.rates_sd"] = 0
parameters_default["cor_stop.tau_up"] = 1
parameters_default["cor_stop.tau_down"] = 70


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

### parameters for test_iliana
parameters_test_iliana = parameters_default
parameters_test_iliana["trials"] = 1

### parameters for test_resting
parameters_test_resting = {}
### general
parameters_test_resting["timestep"] = 0.1
parameters_test_resting["seed"] = 1
### simulation time
parameters_test_resting["t.duration"] = 3000


### parameters for fit_pallido_striatal
parameters_fit_pallido_striatal = {}
### general
parameters_fit_pallido_striatal["timestep"] = 0.1
parameters_fit_pallido_striatal["seed"] = 10
parameters_fit_pallido_striatal["num_threads"] = 1
try:
    with open("fit_pallido_striatal_archive/I_0_10Hz_20Hz.json") as f:
        parameters_fit_pallido_striatal["base_noise"] = json.load(f)["base_noise"]
except:
    warnings.warn(
        "WARNING: parameters: no 'fit_pallido_striatal_archive/I_0_10Hz_20Hz.json' for base_noise"
    )
    parameters_fit_pallido_striatal["base_noise"] = None
try:
    with open("fit_pallido_striatal_archive/activity_by_mod_f.json") as f:
        parameters_fit_pallido_striatal["activity_by_mod"] = json.load(f)
except:
    warnings.warn(
        "WARNING: parameters: no 'fit_pallido_striatal_archive/activity_by_mod_f.json' for fit increase"
    )
    parameters_fit_pallido_striatal["activity_by_mod"] = None
### simulation, can be resting or increase (see function "which_simulation" in fit_hyperopt...)
parameters_fit_pallido_striatal["simulation_protocol"] = "resting"
parameters_fit_pallido_striatal["t.duration"] = 3000
parameters_fit_pallido_striatal["t.init"] = 2000
parameters_fit_pallido_striatal["increase_iterations"] = 15
parameters_fit_pallido_striatal["increase_step"] = 3
parameters_fit_pallido_striatal["t.increase_duration"] = 1000
### optimization
parameters_fit_pallido_striatal["nbr_models"] = 9
parameters_fit_pallido_striatal["nbr_fit_runs"] = 10
parameters_fit_pallido_striatal["parameter_bound_dict"] = {
    "str_d2.base_mean": [8, 14],
    "str_fsi.base_mean": [0.5, 5],
    "gpe_proto.base_mean": [0.13, 1.3],
    "str_d2__gpe_proto.mod_factor": [0, 1],
    "str_d2__str_d2.mod_factor": [0, 1],
    "gpe_proto__str_fsi.mod_factor": [0, 1],
    "gpe_proto__gpe_proto.mod_factor": [0, 1],
    "str_fsi__str_d2.mod_factor": [0, 1],
    "str_fsi__str_fsi.mod_factor": [0, 1],
    "general.str_d2_factor": [0, 1],
}
