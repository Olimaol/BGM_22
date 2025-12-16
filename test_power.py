from ANNarchy import setup, dt, simulate
from CompNeuroPy.full_models import BGM
from CompNeuroPy import (
    CompNeuroMonitors,
    PlotRecordings,
    get_population_power_spectrum,
)
import matplotlib.pyplot as plt

### local
from parameters import parameters_test_power as paramsS


if __name__ == "__main__":

    ### SETUP TIMESTEP + SEED
    if paramsS["seed"] == None:
        setup(dt=paramsS["timestep"])
    else:
        setup(dt=paramsS["timestep"], seed=paramsS["seed"])

    ### COMPILE MODEL & GET MODEL PARAMTERS
    model_name = "BGM_v02_p02"
    model = BGM(name="BGM_v02_p02", seed=paramsS["seed"])
    params = model.params

    ### INIT CompNeuroMonitors ###
    mon = CompNeuroMonitors(
        {
            "cor_go": ["spike"],
            "cor_stop": ["spike"],
            "cor_pause": ["spike"],
            "str_d1": ["spike"],
            "str_d2": ["spike"],
            "str_fsi": ["spike"],
            "gpe_proto": ["spike"],
            "gpe_arky": ["spike"],
            "gpe_cp": ["spike"],
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
        "position": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "compartment": [
            "cor_go",
            "cor_stop",
            "cor_pause",
            "str_d1",
            "str_d2",
            "str_fsi",
            "gpe_proto",
            "gpe_arky",
            "gpe_cp",
        ],
        "variable": [
            "spike",
            "spike",
            "spike",
            "spike",
            "spike",
            "spike",
            "spike",
            "spike",
            "spike",
        ],
        "format": [
            "hybrid",
            "hybrid",
            "hybrid",
            "hybrid",
            "hybrid",
            "hybrid",
            "hybrid",
            "hybrid",
            "hybrid",
        ],
    }
    chunk = 0
    PlotRecordings(
        figname=f"results/test_power/overview2_{model_name}.png",
        recordings=recordings,
        recording_times=recording_times,
        chunk=chunk,
        shape=(3, 3),
        plan=plan,
    )

    ### CREATE FIGURES WITH POWER SPECTRUM
    chunk = 0
    plt.figure(figsize=([6.4 * 3, 4.8 * 3]))
    for nr, pop_name in zip(plan["position"], plan["compartment"]):
        freq, pow = get_population_power_spectrum(
            spikes=recordings[chunk][f"{pop_name};spike"],
            time_step=dt(),
            t_start=recording_times.time_lims(chunk=chunk)[0],
            t_end=recording_times.time_lims(chunk=chunk)[1],
            fft_size=4096,
        )
        plt.subplot(3, 3, int(nr))
        plt.title(pop_name)
        plt.plot(freq, pow, color="k")
        plt.xlim(0, paramsS["cor_go.frequency"] * 4)
        # plt.yscale("log")
        plt.xlabel("frequency [Hz]")
        plt.ylabel("power")
    plt.tight_layout()
    plt.savefig(f"results/test_power/freq_{model_name}.png")
