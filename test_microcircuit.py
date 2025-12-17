from ANNarchy import setup, simulate
from CompNeuroPy.full_models import BGM
from CompNeuroPy import (
    CompNeuroMonitors,
    PlotRecordings,
    print_df,
)
from striatal_microcircuit.connectivity_construct import Microcircuit
from striatal_microcircuit.cortical_inputs import CorticalInputs

### local
from parameters import parameters_test_microcircuit as paramsS


if __name__ == "__main__":

    ### SETUP TIMESTEP + SEED
    if paramsS["seed"] == None:
        setup(dt=paramsS["timestep"])
    else:
        setup(dt=paramsS["timestep"], seed=paramsS["seed"])

    ### COMPILE MODEL & GET MODEL PARAMTERS
    model = BGM(name="BGM_v04newgpe_p01", seed=paramsS["seed"])
    params = model.params

    print("model paramters:")
    print_df(model.attribute_df)

    ### INIT CompNeuroMonitors ###
    mon = CompNeuroMonitors(
        {
            # "cor_go": ["spike"],
            # "cor_stop": ["spike"],
            # "cor_pause": ["spike"],
            # "str_d1": ["spike"],
            "str_d2": ["spike"],
            "str_fsi": ["spike"],
            "gpe_proto": ["spike", "I_base"],
            # "gpe_arky": ["spike", "u"],
            # "gpe_cp": ["spike"],
        }
    )

    ### SIMULATION ###
    mon.start()

    ### simulate some time
    simulate(paramsS["t.duration"])

    ### GET RECORDINGS ###
    recordings = mon.get_recordings()
    recording_times = mon.get_recording_times()

    ### QUICK PLOTS ###

    ### some populations activity
    plan = {
        "position": [1, 2, 3, 4],
        "compartment": ["str_d2", "str_fsi", "gpe_proto", "gpe_proto"],
        "variable": ["spike", "spike", "spike", "I_base"],
        "format": ["hybrid", "hybrid", "hybrid", "line"],
    }
    chunk = 0
    PlotRecordings(
        figname=f"results/test_resting/{model.name}/overview1.png",
        recordings=recordings,
        recording_times=recording_times,
        chunk=chunk,
        shape=(1, 4),
        plan=plan,
    )
