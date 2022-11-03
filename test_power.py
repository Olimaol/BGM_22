from ANNarchy import setup, dt, simulate, get_population
from CompNeuroPy.models import BGM
from CompNeuroPy import (
    Monitors,
    plot_recordings,
    get_population_power_spectrum,
)
import matplotlib.pyplot as plt


if __name__ == "__main__":

    ### SETUP TIMESTEP + SEED
    seed_val = 1
    setup(dt=0.1, seed=seed_val)

    ### COMPILE MODEL & GET PARAMTERS
    ### in BGM_vTEST_pTEST cor_go consists of poisson_sin neurons
    model = BGM(name="BGM_vTEST_pTEST", seed=seed_val)
    params = model.params
    paramsS = {}
    paramsS["trials"] = 1

    ### INIT MONITORS ###
    mon = Monitors(
        {
            "pop;cor_go": ["spike", "rates"],
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

    frequency = 50
    phase = 0  # (1 / frequency) / 4
    get_population("cor_go").amplitude = 50
    get_population("cor_go").frequency = frequency
    get_population("cor_go").phase = phase
    get_population("cor_go").base = 50

    simulate(1000)

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
        time_lim=[0, (1 / frequency) * 1000],
    )
    ### some populations activity
    plot_list = [
        "1;cor_go;spike;hybrid",
        "2;cor_go;rates;line",
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
            fft_size=None,
        )
        plt.subplot(3, 3, int(nr))
        plt.title(pop_name)
        plt.plot(freq, pow, color="k")
        plt.xlim(0, frequency * 4)
        # plt.yscale("log")
        plt.xlabel("frequency [Hz]")
        plt.ylabel("power")
    plt.tight_layout()
    plt.savefig("results/test_power/freq.png")
