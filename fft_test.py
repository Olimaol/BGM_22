from ANNarchy import setup, dt, simulate, raster_plot
from CompNeuroPy.models import BGM
from CompNeuroPy import (
    Monitors,
    plot_recordings,
    get_nanmean,
    hanning_split_overlap,
    get_number_of_zero_decimals,
    get_number_of_decimals #anzahl an stellen nach komma
)
import matplotlib.pyplot as plt
import numpy as np

### local
from parameters import parameters_test_power as paramsS


time_step=dt()

def ms_to_s(x):
        return x / 1000


def my_raster_plot(spikes):
    """
    Returns two vectors representing for each recorded spike 1) the spike times and 2) the ranks of the neurons.

    The spike times are always in simulation steps (in contrast to default ANNarchy raster_plot)
    """
    t, n = raster_plot(spikes)
    t = t / dt()
    return t, n

###########################################################################################################

if __name__ == "__main__":

    ### SETUP TIMESTEP + SEED
    if paramsS["seed"] == None:
        setup(dt=paramsS["timestep"])
    else:
        setup(dt=paramsS["timestep"], seed=paramsS["seed"])

    

    ### COMPILE MODEL & GET MODEL PARAMTERS
    model_name = "BGM_v02_p02"                      
# BGM v02_p03 : BGM version with fitted FSI Neuron based on Corbit et al (2016) and synaptic delays based on Kumarave et al(2015)
    model = BGM(model_name, seed=paramsS["seed"],do_compile=False)
    params = model.params

 
    model.compile()

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

    np.save("./numpy_data.npy",recordings)
   

     ### QUICK PLOTS ###

    ### some populations activity
    plot_list = [
        "1;cor_go;spike;hybrid",
        "2;str_d1;spike;hybrid",
        "3;str_d2;spike;hybrid",
        "4;str_fsi;spike;hybrid",
        "5;gpe_proto;spike;hybrid",
        "6;gpe_arky;spike;hybrid",
        "7;gpe_cp;spike;hybrid",
    ]

    ### CALCULATE FFT POWER
    chunk = 0
    fft_size= 4096
 
   
    plt.figure(figsize=([6.4 * 3, 4.8 * 3]))
    ### ueber alle aufgezeichneten populations : 
    for sub_plot_str in plot_list:
        nr, pop_name, _, _ = sub_plot_str.split(";")
        sampling_frequency = 1 / ms_to_s(time_step)  # in Hz
        spikes = recordings[chunk][f"{pop_name};spike"]             # longer simulation -> more data : spikes -> 
        populations_size = len(list(spikes.keys()))                 # print -> populations_size = 100
        t_start=recording_times.time_lims(chunk=chunk)[0]
        t_end=recording_times.time_lims(chunk=chunk)[1]

        ### calculate time
        t, _ = my_raster_plot(spikes)

        if t_start == None:                                                          # time of the first spike
            t_start = round(t.min() * time_step, get_number_of_decimals(time_step))
        if t_end == None:                                                            # time of the last spike
            t_end = round(t.max() * time_step, get_number_of_decimals(time_step))

        simulation_time = round(t_end - t_start, get_number_of_decimals(time_step))  # in ms   

         ### generate power spectrum
        spectrum = np.zeros((populations_size, fft_size))           # define variable for power spectrum  (100,4096)

        for neuron in range(populations_size):
            ### sampling steps array
            spiketrain = np.zeros(
                int(np.round(ms_to_s(simulation_time) * sampling_frequency))
            )
            ### spike times as sampling steps
            idx = (
                np.round(
                    ms_to_s((np.array(spikes[neuron]) * time_step)) * sampling_frequency
                )
            ).astype(np.int32)
            ### cut the spikes before t_start and after t_end
            idx_start = ms_to_s(t_start) * sampling_frequency
            idx_end = ms_to_s(t_end) * sampling_frequency
            mask = ((idx > idx_start).astype(int) * (idx < idx_end).astype(int)).astype(
                bool
            )
            idx = (idx[mask] - idx_start).astype(np.int32)

            ### set spiketrain array to one if there was a spike at sampling step
            spiketrain[idx] = 1

            ### generate multiple overlapping sequences out of the spike trains
            spiketrain_sequences = hanning_split_overlap(
                spiketrain, fft_size, int(fft_size / 2)                 # versuch anstatt spiketrain [1]*fft_size ergibt nur Nulllinie
            )


            spectrum[neuron] = get_nanmean(
                np.abs(np.fft.fft([spiketrain_sequences])) ** 2, 0
            )


        ### mean spectrum over all neurons
        spectrum = get_nanmean(spectrum, 0)

        freq = np.fft.fftfreq(fft_size, 1.0 / sampling_frequency) 

        idx = np.argsort(freq)

        #print(len(freq))  #4096
          
        plt.subplot(3, 3, int(nr))
        plt.title(pop_name)
        plt.plot(freq[idx],spectrum[idx])
    plt.tight_layout() # alles in einen plot
    plt.show()
