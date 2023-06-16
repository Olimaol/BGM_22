from ANNarchy import setup, get_population, get_time,raster_plot,get_projection,simulate
from CompNeuroPy import Monitors, generate_simulation, plot_recordings
from tqdm import tqdm
import pylab as plt
import numpy as np


from parameters import parameters_test_iliana as paramsS

global saveresults
saveresults = []

### load data
data = np.load('numpy_results/recordings_pallidostriatal_network_BGM_v04_p01_updated_simulation9500ms_9trials_gpe_proto_parameter_fixed_cor_off.npy', allow_pickle=True)
data_DD = np.load('numpy_results/recordings_pallidostriatal_network_BGM_v05_p01_updated_simulation9500ms_9trials_gpe_proto_parameter_fixed_cor_off.npy', allow_pickle=True)


dt = paramsS["timestep"]    # = 0.1
sim_time = int(9500/dt)     # simulation time from test_pallidostriatalnetwork.py
spikes = {} # empty dict
spikes_DD = {}
trials = paramsS["trials"]               # note : if more trails -> data[0][0][pop] becomes data[0][trails 0-8][pop] !
sumspikes = {}
sumspikes_DD = {}
freq_spike_counter = {}
freq_spike_counter_DD = {}
freq = {}
freq_DD = {}


# generate (again) matrix containing all time points and all spikes timings :
for run in range(trials):
    spikes[run]={}
    sumspikes[run]={}
    spikes_DD[run]={}
    sumspikes_DD[run]={}
    for pop in ["str_d2","str_fsi","gpe_proto"]:    

        spikes[run][pop] = np.zeros((100,sim_time))
        spikes_DD[run][pop] = np.zeros((100,sim_time))
        sumspikes[run][pop] = 0
        sumspikes_DD[run][pop] = 0

        for i in range(100):                                        # for all neurons "i" in one population

            for n in data[0][run][f"{pop};spike"][i]:                     
                spikes[run][pop][i,n]= 1
            sumspikes[run][pop] += len(data[0][run][f"{pop};spike"][i])

            for n in data_DD[0][run][f"{pop};spike"][i]: 
                spikes_DD[run][pop][i,n]= 1
            sumspikes_DD[run][pop] += len(data_DD[0][run][f"{pop};spike"][i])
 

# discard the first 500ms (Methods,Corbit et al., 2016)
for run in range(trials):
    for pop in ["str_d2","str_fsi","gpe_proto"]: 
        spikes[run][pop] = spikes[run][pop][:,int(500/dt):sim_time]
        spikes_DD[run][pop] = spikes_DD[run][pop][:,int(500/dt):sim_time]


# overwrite sim_time without first 500ms
sim_time = sim_time - int(500/dt)


### calculation of average firing rate : 
for  pop in ["str_d2","str_fsi","gpe_proto"]:  
    freq_spike_counter[pop] = 0
    freq[pop] = 0

    freq_spike_counter_DD[pop] = 0
    freq_DD[pop] = 0

    for run in range(trials):
        freq_spike_counter[pop]  = np.sum(spikes[run][pop])
        freq_spike_counter_DD[pop]  = np.sum(spikes_DD[run][pop])

    #print("freq_spike_counter ", pop, " : ", freq_spike_counter[pop])

    freq_spike_counter[pop] = freq_spike_counter[pop] / 100 # 100 = number neurons
    freq_spike_counter_DD[pop] = freq_spike_counter_DD[pop] / 100

    freq_spike_counter[pop] = freq_spike_counter[pop]/trials    
    freq_spike_counter_DD[pop] = freq_spike_counter_DD[pop]/trials 

    freq[pop] = freq_spike_counter[pop]/sim_time    # ms   
    freq_DD[pop] = freq_spike_counter_DD[pop]/sim_time 
                    
    freq[pop] = freq[pop] * 10000                       # ms to s / remember dt = 0.1ms
    freq_DD[pop] = freq_DD[pop] * 10000 


# spike histogram for each pop
pop = ["str_d2","str_fsi","gpe_proto"]
plt.bar(pop,[sumspikes[0][p] for p in pop], width=0.5)
plt.show()

quit()

# average firing rate as histogramm 

pop = ["str_d2","str_fsi","gpe_proto"]
plt.bar(pop, [freq[p] for p in pop], width=0.5)
plt.bar(pop, [freq_DD[p] for p in pop], width=0.5, label = 'DD')
plt.xlabel("pop")
plt.ylabel("average firing rate in Hz")
plt.show()


