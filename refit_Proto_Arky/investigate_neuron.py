from CompNeuroPy.neuron_models import _fit_Bogacz as neuron_model
from CompNeuroPy import compile_in_folder, Monitors, plot_recordings
from ANNarchy import Population, simulate, setup
import numpy as np


def replace_substrings_except_within_braces(input_string, replacement_mapping):
    result = []
    inside_braces = False
    i = 0

    while i < len(input_string):
        if input_string[i] == "{":
            inside_braces = True
            result.append(input_string[i])
            i += 1
        elif input_string[i] == "}":
            inside_braces = False
            result.append(input_string[i])
            i += 1
        else:
            if not inside_braces:
                found_match = False
                for old_substr, new_substr in replacement_mapping.items():
                    if input_string[i : i + len(old_substr)] == old_substr:
                        result.append(new_substr)
                        i += len(old_substr)
                        found_match = True
                        break
                if not found_match:
                    result.append(input_string[i])
                    i += 1
            else:
                result.append(input_string[i])
                i += 1

    return "".join(result)


def evl_str_formula(equation_str, avail_variables_dict):
    """
    Args:
        equation_str: str
            str of the equation using the available variables

        avail_variables_dict: dict
            keys=the names of the available variables
            values=the available variables

    """
    new_equation_str = equation_str
    sorted_names_list = sorted(list(avail_variables_dict.keys()), key=len, reverse=True)
    ### first replace largest variable names
    ### --> if smaller variable names are within larger variable names this should not cause a problem
    for name in sorted_names_list:
        if name in new_equation_str:
            ### replace the name in the new_equation_str
            ### only replace things which are not between {}
            new_equation_str = replace_substrings_except_within_braces(
                new_equation_str, {name: "{" + name + "}"}
            )
    ### evaluate the value with the values of the dictionary
    return eval(new_equation_str.format(**avail_variables_dict))


eq = "a+b-2"
a = 10
b = np.ones(10)
evl_str_formula(equation_str, avail_variables_dict)

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
    "E_gaba": 0,
    "increase_noise": 0,
    "rates_noise": 0,
    "r": 0,
}

pop = Population(1, neuron=neuron_model, name="single_neuron")
for var_name, var_val in variables_valuess.items():
    if var_name in pop.attributes:
        setattr(pop, var_name, var_val)

compile_in_folder(folder_name="investigate_neuron")

mon = Monitors(monDict={"pop;single_neuron": ["spike", "v", "u", "g_ampa"]})

mon.start()
pop.tau_ampa = 1e20

simulate(100)
pop.g_ampa = 1
simulate(100)
pop.g_ampa = 2
simulate(100)
pop.g_ampa = 3
simulate(100)
pop.g_ampa = 10
simulate(100)

recordings = mon.get_recordings()
recording_times = mon.get_recording_times()

plot_recordings(
    figname="investigate_neuron.png",
    recordings=recordings,
    recording_times=recording_times,
    chunk=0,
    shape=(4, 1),
    plan=[
        "1;single_neuron;spike;single",
        "2;single_neuron;v;line",
        "3;single_neuron;u;line",
        "4;single_neuron;g_ampa;line",
    ],
)
