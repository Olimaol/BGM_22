from ANNarchy import setup, dt, simulate
from CompNeuroPy.models import BGM
from CompNeuroPy import (
    Monitors,
    plot_recordings,
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

    ### INIT MONITORS ###
    mon = Monitors(
        {
            "pop;cor_go": ["spike"],
            "pop;cor_stop": ["spike"],
            "pop;cor_pause": ["spike"],
            "pop;str_d1": ["spike"],
            "pop;str_d2": ["spike"],
            "pop;str_fsi": ["spike"],
            "pop;gpe_proto": ["spike"],
            "pop;gpe_arky": ["spike"],
            "pop;gpe_cp": ["spike"],
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
    plot_list = [
        "1;cor_go;spike;hybrid",
        "2;cor_stop;spike;hybrid",
        "3;cor_pause;spike;hybrid",
        "4;str_d1;spike;hybrid",
        "5;str_d2;spike;hybrid",
        "6;str_fsi;spike;hybrid",
        "7;gpe_proto;spike;hybrid",
        "8;gpe_arky;spike;hybrid",
        "9;gpe_cp;spike;hybrid",
    ]
    chunk = 0
    plot_recordings(
        figname=f"results/test_power/overview2_{model_name}.png",
        recordings=recordings,
        recording_times=recording_times,
        chunk=chunk,
        shape=(3, 3),
        plan=plot_list,
    )

    ### CREATE FIGURES WITH POWER SPECTRUM
    chunk = 0
    plt.figure(figsize=([6.4 * 3, 4.8 * 3]))
    for sub_plot_str in plot_list:
        nr, pop_name, _, _ = sub_plot_str.split(";")
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
