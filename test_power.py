from ANNarchy import setup, dt, simulate, get_population
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
    model = BGM(name="BGM_vTEST_pTEST", seed=paramsS["seed"])
    params = model.params

    ### INIT MONITORS ###
    mon = Monitors(
        {
            "pop;cor_go": ["spike", "rates"],
            "pop;cor_stop": ["spike"],
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

    ### define the sinus oscillation of cor_go
    get_population("cor_go").amplitude = paramsS["cor_go.amplitude"]
    get_population("cor_go").frequency = paramsS["cor_go.frequency"]
    get_population("cor_go").phase = paramsS["cor_go.phase"]
    get_population("cor_go").base = paramsS["cor_go.base"]
    ### simulate some time
    simulate(paramsS["t.duration"])

    ### GET RECORDINGS ###
    recordings = mon.get_recordings()
    recording_times = mon.get_recording_times()

    ### QUICK PLOTS ###
    ### cor-go period time
    plot_list = [
        "1;cor_go;spike;hybrid",
        "2;cor_go;rates;line",
    ]
    chunk = 0
    plot_recordings(
        figname="results/test_power/overview1.png",
        recordings=recordings,
        recording_times=recording_times,
        chunk=chunk,
        shape=(2, 1),
        plan=plot_list,
        time_lim=[0, (1 / paramsS["cor_go.frequency"]) * 1000],
    )
    ### some populations activity
    plot_list = [
        "1;cor_go;spike;hybrid",
        "2;cor_stop;spike;hybrid",
        "4;str_d1;spike;hybrid",
        "5;str_d2;spike;hybrid",
        "6;str_fsi;spike;hybrid",
        "7;gpe_proto;spike;hybrid",
        "8;gpe_arky;spike;hybrid",
        "9;gpe_cp;spike;hybrid",
    ]
    chunk = 0
    plot_recordings(
        figname="results/test_power/overview2.png",
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
    plt.savefig("results/test_power/freq.png")
