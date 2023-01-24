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
parameters_test_iliana["trials"]=1
