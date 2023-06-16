from ANNarchy import setup, dt, simulate
from CompNeuroPy.models import BGM
from CompNeuroPy import (
    Monitors,
    plot_recordings,
    get_population_power_spectrum,
)
import matplotlib.pyplot as plt
import numpy as np


### local
from parameters import parameters_test_power as paramsS

if __name__ == "__main__":

    ### SETUP TIMESTEP + SEED
    if paramsS["seed"] == None:
        setup(dt=paramsS["timestep"])
    else:
        setup(dt=paramsS["timestep"], seed=paramsS["seed"])

    

    ### COMPILE MODEL & GET MODEL PARAMTERS
    model_name = "BGM_v05_p01"                      
# BGM v02_p03 : BGM version with fitted FSI Neuron based on Corbit et al (2016) and synaptic delays based on Kumarave et al(2015)
    model = BGM(model_name, seed=paramsS["seed"],do_compile=False)
    params = model.params

 
    model.compile()

    ### INIT MONITORS ###
    mon = Monitors(
        {   
             "pop;str_d2": ["spike", "g_ampa", "g_gaba"],#"osc"],
             "pop;gpe_proto": ["spike", "g_ampa", "g_gaba"],#"osc"],
             "pop;str_fsi": ["spike", "g_ampa", "g_gaba"],      
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
        "1;str_fsi;spike;hybrid",
        "2;str_d2;spike;hybrid",
        "3;gpe_proto;spike;hybrid",

  
    ]
    chunk = 0
    plot_recordings(
        figname=f"results/test_pallidostriatalnet/overview_freq_{model_name}_short_simulation_check_gpe_fsi_dynamic_increasemodfactor_1_5.png",
        recordings=recordings,
        recording_times=recording_times,
        chunk=chunk,
        shape=(3, 3),
        plan=plot_list,
    )

    ### CREATE FIGURES WITH POWER SPECTRUM
    chunk = 0
    plt.figure(figsize=([6.4 * 3, 4.8 * 3]))
    # ueber alle augezeichneten Populationen : 
    for sub_plot_str in plot_list:
        nr, pop_name, _, _ = sub_plot_str.split(";")
        freq, pow = get_population_power_spectrum(          # -> siehe CompNeuroPy analysis_functions 
                                                             # get_population_power_spectrum(spikes,time_step,t_start,t_end,fft_size)
            spikes=recordings[chunk][f"{pop_name};spike"],
            time_step=dt(),
            t_start=recording_times.time_lims(chunk=chunk)[0],
            t_end=recording_times.time_lims(chunk=chunk)[1],
            fft_size= 4096, #8192,     # 4096,
        )
        plt.subplot(3, 3, int(nr))
        plt.title(pop_name)
        plt.plot(freq, pow, color="k")
        plt.xlim(0,100) 
        # plt.yscale("log")
        plt.xlabel("frequency [Hz]")
        plt.ylabel("power")
    plt.tight_layout()
    plt.savefig(f"results/test_pallidostriatalnet/freq_{model_name}_short_simulation_check_gpe_fsi_dynamic_increasemodfactor_1_5.png")

