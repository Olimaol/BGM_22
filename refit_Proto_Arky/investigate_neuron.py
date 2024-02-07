from CompNeuroPy.neuron_models.experimental_models.fit_Bogacz_nm import (
    _fit_Bogacz as neuron_model,
)
from CompNeuroPy import compile_in_folder, CompNeuroMonitors, PlotRecordings
from CompNeuroPy.simulation_functions import attr_ramp
from ANNarchy import Population, setup

setup(dt=0.1)

variables_valuess = {
    "a": 0.039191890241715294,
    "b": 0.000548238111291427,
    "c": -49.88014418530518,
    "d": 108.0208225074675,
    "n0": 24.2219699019072,
    "n1": 1.1929776239208976,
    "n2": 0.08899515481507077,
    "R_input_megOhm": 1,
    "x": 1,
    "tau_ampa": 10,
    "tau_gaba": 10,
    "E_ampa": 0,
    "E_gaba": -90,
    "increase_noise": 0,
    "rates_noise": 0,
    "r": 0,
}

pop = Population(1, neuron=neuron_model, name="single_neuron")
for var_name, var_val in variables_valuess.items():
    if var_name in pop.attributes:
        setattr(pop, var_name, var_val)

compile_in_folder(folder_name="investigate_neuron")

mon = CompNeuroMonitors(mon_dict={"single_neuron": ["spike", "v", "u", "g_ampa", "I"]})

mon.start()

attr_ramp(pop.name, "I_app", v0=0, v1=100, dur=1000, n=100)
mon.reset()
pop.tau_ampa = 1e20
attr_ramp(pop.name, "g_ampa", v0=0, v1=1.2 * 100 / 70, dur=1000, n=100)

recordings = mon.get_recordings()
recording_times = mon.get_recording_times()

### plot chunk 0
PlotRecordings(
    figname="investigate_neuron_0.png",
    recordings=recordings,
    recording_times=recording_times,
    chunk=0,
    shape=(5, 1),
    plan={
        "position": [1, 2, 3, 4, 5],
        "compartment": [
            "single_neuron",
            "single_neuron",
            "single_neuron",
            "single_neuron",
            "single_neuron",
        ],
        "variable": ["spike", "v", "u", "g_ampa", "I"],
        "format": ["raster", "line", "line", "line", "line"],
    },
)

### plot chunk 1
PlotRecordings(
    figname="investigate_neuron_1.png",
    recordings=recordings,
    recording_times=recording_times,
    chunk=1,
    shape=(5, 1),
    plan={
        "position": [1, 2, 3, 4, 5],
        "compartment": [
            "single_neuron",
            "single_neuron",
            "single_neuron",
            "single_neuron",
            "single_neuron",
        ],
        "variable": ["spike", "v", "u", "g_ampa", "I"],
        "format": ["raster", "line", "line", "line", "line"],
    },
)
