from ANNarchy import setup, get_population, get_time,raster_plot,get_projection,simulate, dt
from CompNeuroPy import Monitors, generate_simulation, plot_recordings,get_population_power_spectrum
from tqdm import tqdm
import pylab as plt
import numpy as np


from parameters import parameters_test_iliana as paramsS

global saveresults
saveresults = []

### load data 
#data = np.load('numpy_results/recordings_pallidostriatal_network_BGM_v04_p01_updated.npy', allow_pickle=True)
#data = np.load('numpy_results/recordings_pallidostriatal_network_BGM_v04_p01_updated_simulation9500ms_9trials.npy', allow_pickle=True)
#data = np.load('numpy_results/recordings_pallidostriatal_network_BGM_v05_p01_updated_simulation9500ms_9trials_inputs_off.npy', allow_pickle=True)
#data = np.load('numpy_results/recordings_pallidostriatal_network_BGM_v05_p01_updated_simulation9500ms_9trials_inputs_off_osc_factor_gpe_5_str_d2_250.npy', allow_pickle=True)
data = np.load('numpy_results/recordings_pallidostriatal_network_BGM_v04_p01_updated_simulation9500ms_9trials_gpe_proto_parameter_fixed_cor_off.npy', allow_pickle=True)
#data = np.load('numpy_results/recordings_pallidostriatal_network_BGM_v05_p01_updated_simulations9500ms_9trials_gpe_proto_parameter_fixed_cor_off_gpe-fsi-modfactor-1.npy', allow_pickle=True)


# print(rec[0][0].keys())
# output : dict_keys(['cor_go;period', 'cor_go;spike', 'cor_stop;period', 'cor_stop;spike', 'str_d2;period', 'str_d2;spike', 'str_d2;g_ampa', 'str_d2;g_gaba', 'gpe_proto;period', 'gpe_proto;spike', 'gpe_proto;g_ampa', 'gpe_proto;g_gaba', 'str_fsi;period', 'str_fsi;spike', 'str_fsi;g_ampa', 'str_fsi;g_gaba', 'dt'])


# example spike timings for first str-d2 neuron :
# print(rec[0][0]['str_d2;spike'][0])
# output : [2710, 3359, 3404, 3585, 3639, 8172, 9626, 11065, 11095, 11155, 12960, 13017, 13039, 13067, 13109, 14295, 14364, 15638, 15665, 15717, 15929, 16073, 18104, 18132, 18186, 19155, 19184, 19428, 19489]



### spike synchrony computation by using counts of spikes binned into 15ms bins (Methods,Corbit et al., 2016)
# run simulated for 9500ms,  first 500ms discarded, 9 trials (Corbit et al., 2016)
# problem / question : how much and how long should be the inputs for str_d2 and gpe_proto ?
# first approach : first half stimulate str_d2, then gpe_proto, SSD 4500ms 

timestep = paramsS["timestep"]    # = 0.1
sim_time = int(9500/timestep)
bin_size = int(15.0/timestep)
spikes = {} # empty dict
counter = {} # empty dict 
var_cell = {}
counter_pop = {}
var_pop = {}
trials = paramsS["trials"]               # note : if more trails -> data[0][0][pop] becomes data[0][trails 0-8][pop] !
sumspikes = {}
thres = 0
freq_spike_counter = {}
freq = {}



### calculate the within cell variance :
# generate (again) matrix containing all time points and all spikes timings :
for run in range(trials):
    spikes[run]={}
    sumspikes[run]={}
    for pop in ["str_d2","str_fsi","gpe_proto"]:    
        spikes[run][pop] = np.zeros((100,sim_time))
        sumspikes[run][pop] = 0
        for i in range(100):                                        # for all neurons "i" in one population
            for n in data[0][run][f"{pop};spike"][i]:                     
                spikes[run][pop][i,n]= 1
            sumspikes[run][pop] += len(data[0][run][f"{pop};spike"][i])

"""
# check if number of spikes is correct by calculating the difference between the "zaehler" and "sumspikes" for each run and pop -> difference should be 0
zaehler = {}
for run in range(trials):
    zaehler[run]= {}
    for pop in ["str_d2","str_fsi","gpe_proto"]:
        zaehler[run][pop] = 0
        for i in range(100):
            for n in range(sim_time):
                if spikes[run][pop][i,n] == 1:
                    zaehler[run][pop]+=1

for run in range(trials):
    print(zaehler[run]["str_d2"]-sumspikes[run]["str_d2"])              # 53382
quit()
"""


# discard the first 500ms (Methods,Corbit et al., 2016)
for run in range(trials):
    for pop in ["str_d2","str_fsi","gpe_proto"]: 
        spikes[run][pop] = spikes[run][pop][:,int(500/timestep):sim_time]


# overwrite sim_time without first 500ms
sim_time = sim_time - int(500/timestep)

# TODO : FSI Pausen finden, unterscheidung bei sychronen und asynchronen GPe Spikes ? 
### FSI firing pause detection defined as epochs of at least 20ms  (Methods,Corbit et al., 2016)   & power spectrum              
pause_bin = int(20./timestep)
pause_counter = {}          # which FSI cell pauses ? 
pause_neuron = {}
pause_matrix = {}           # matrix neuron i x sim_time n, value 1 if pause begins, else 0
power_spectrum = {} 
sampling_rate = 1.0 /(timestep/1000) #  Attention : ms to s ! 
frequency = {}
combined_power_spectrum={}
pause_timings = {}

for run in range(trials):
    pause_matrix[run]={}
    for pop in ["str_d2","str_fsi","gpe_proto"]:    
        pause_matrix[run][pop] = np.zeros((100,sim_time))
        power_spectrum[pop] =[[] for _ in range(100)]
        frequency[pop] = 0
        combined_power_spectrum[pop]= 0
        pause_timings[pop] = [] 

        for i in range(100):                                                                                                         # for all neurons "i" in one population

            for n in data[0][run][f"{pop};spike"][i]:   
                if n < sim_time and spikes[run][pop][i, n] == 1 and np.sum(spikes[run][pop][i, n:n + pause_bin] == 0):                 
                    pause_matrix[run][pop][i,n]= 1

            power_spectrum[pop][i] = np.abs(np.fft.fft(pause_matrix[run][pop][i])) ** 2
            
        pause_timings[pop] = np.where(pause_matrix[run][pop] == 1) # find indices where matrix is 1
        #print("pause timings von ", pop, " : ", pause_timings[pop])


# pause_timings[pop] has the type tuple, for the function get_population_power_spectrum() it has to be a dict (-> problem with command .keys())
pause_timings[pop] = dict(zip(pause_timings[pop][0], pause_timings[pop][1]))            # zip() function in order to combine the multiple tuples into a single iterator 


plot_list = [
    "1;str_fsi;spike;hybrid",
    "2;str_d2;spike;hybrid",
    "3;gpe_proto;spike;hybrid",]

chunk = 0
plt.figure(figsize=([6.4 * 3, 4.8 * 3])) 
for sub_plot_str in plot_list:
    nr, pop_name, _, _ = sub_plot_str.split(";")
    freq, pow = get_population_power_spectrum(          
    spikes=pause_timings[pop],
    time_step=dt(),
    t_start=0, #recording_times.time_lims(chunk=chunk)[0],
    t_end=sim_time, #recording_times.time_lims(chunk=chunk)[1],
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
    plt.show()

"""
        combined_power_spectrum[pop] = np.mean(power_spectrum[pop], axis=0)  # average over all neurons, axis = 0 = column = i 
        frequency[pop] =  np.fft.fftfreq(sim_time, sampling_rate)       #  retrieve the size of the matrix along axis -> 100 neurons

        


# Plot the power spectrum 

        plt.plot(frequency[pop], combined_power_spectrum[pop])
        plt.xlabel('Frequency')
        plt.ylabel('Power')
        plt.show()
"""
quit()


for run in range(trials):
    pause_counter[run] = {}
    for pop in ["str_d2","str_fsi","gpe_proto"]:
        pause_counter[run][pop] = 0
        pause_neuron[pop] = []
        for i in range(100): 
            for n in data[0][run][f"{pop};spike"][i]:                       # wenn zelle i gespikt hat und danach 20ms nicht spikt, dann pause 

                if n < sim_time and spikes[run][pop][i, n] == 1 and np.sum(spikes[run][pop][i, n:n + pause_bin] == 0): 
    
                    if i not in pause_neuron[pop]:
                         pause_neuron[pop] = np.append(pause_neuron[pop], i)

  
                    pause_counter[run][pop] += 1


        print("pausen in pop ", pop, " : ", pause_counter[run][pop])
        print("Welche neurone in ", pop, " pausieren ? ", pause_neuron[pop], ", insgesamt sind es : ", len(pause_neuron[pop]))
   


    
"""

# divide generated matrix in the bins and count spikes for each cell in each pop in each run
for run in range(trials):
    counter[run] = {}
    for pop in ["str_d2","str_fsi","gpe_proto"]:
        counter[run][pop] = np.zeros((100,int(sim_time/bin_size)))
        for i in range(100): 
            bin_counter = 0
            for bins in range(0,(sim_time-bin_size),bin_size):       
                bins = int(bins)
                #print("bins: ",bins)
                #print("sim_time:", sim_time)
                #print("bin_size:", bin_size)
                #print("update: ",bins+bin_size)
                counter[run][pop][i,bin_counter] = np.sum(spikes[run][pop][i,bins:bins+bin_size]) # calculate sum for each bin -> boundary from bins to bins + bin_size 
                bin_counter+=1   



# calculate the within cell variance 
for run in range(trials):
    var_cell[run] = {}
    for pop in ["str_d2","str_fsi","gpe_proto"]:
        var_cell[run][pop] =  np.zeros(100)
        for i in range(100):
            var_cell[run][pop][i] = np.var(counter[run][pop][i])
        # variance calculated like Corbit et al., 2016 :
        # var_cell[pop][i]= abs((counter[pop][i])**2)- (np.sum(counter[pop][i])/len(counter[pop][i])))**2


######################################################################
### calculate the population variance :
for run in range(trials):
    var_pop[run] = {}
    counter_pop[run] = {}
    for pop in ["str_d2","str_fsi","gpe_proto"]:
        var_pop[run][pop]= np.zeros((1,bin_counter+1)) 
        counter_pop[run][pop] = np.zeros(bin_counter+1) 
        for bins in range(bin_counter+1):            
            for i in range(100):      
                counter_pop[run][pop][bins]+= counter[run][pop][i,bins]                 # or : counter_pop[pop] = np.sum(counter[pop][i,bins], axis=0)            
                                                                           # axis 0, sum along the columns/neurons ; axis 1, sum along the lines/bins
        #print("run :", run, " pop :", pop, "counter: ",counter_pop[run][pop])

    #print(len(counter_pop[pop])) # 600 werte
        var_pop[run][pop] = np.var(counter_pop[run][pop])  # or : var_pop[run][pop]= abs((counter_pop[pop][:]**2)- (np.sum(counter[pop][:])/(bin_counter+1)))**2
        #print("run :", run, " pop :", pop, "variance of pop: ", var_pop[run][pop])
  
    
    #print(var_pop[run]["str_d2"])                    # 97.8, 1 run             
    #print(var_pop["str_fsi"])                   # 6204.07, 1 run
    #print(var_pop["gpe_proto"])                 # 249.81, 1run
    #print("variance in d2: ", var_pop[run]["str_d2"])

    #print("within cell variance in d2 :",var_cell[run]["str_d2"])


### measure synchrony Xi (Methods, Corbit et al.,2016)
Xi = {}
sum_var_cell = {}
sum_var_pop = {}

for pop in ["str_d2","str_fsi","gpe_proto"]:
    Xi[pop] = 0
    sum_var_cell[pop]= 0
    sum_var_pop[pop]=0
    for run in range(trials):
        for i in range(100):
            #sum_var_pop[pop]+= var_pop[run][pop]             # sum of population variance 
            sum_var_pop[pop] = np.sum(var_pop[run][pop])    
            sum_var_cell[pop] += var_cell[run][pop][i]          # sum over all neurons, all runs for each pop = sum of within cell variance 
    
    sum_var_pop[pop] = sum_var_pop[pop]/trials
    sum_var_cell[pop] = sum_var_cell[pop]/trials
    #print(" pop :", pop,"sum_var_cell[pop]: ", sum_var_cell[pop])
    #print(" pop :", pop,"sum_var_pop[pop]: ", sum_var_pop[pop])            
    #print("sum_var_cell[pop]: ", sum_var_cell[pop])
    Xi[pop] = sum_var_pop[pop] / ((1/100) * sum_var_cell[pop])          # 100 = neuron number 
    print(" pop :", pop, "Synchronizität: ",Xi[pop])

# Xi corrected based on number of neurons (strd2 : 100 vs 40 -> 2.5,  str_fsi & gpe : 100 vs 8 -> 12.5) in model and model of Corbit et al.(2016)
print("Str_d2 Synchronizität korrigiert um Neuronenzahlfaktor 2.5: ",Xi["str_d2"]/2.5)
print("Str_fsi Synchronizität korrigiert um Neuronenzahlfaktor 12.5: ",Xi["str_fsi"]/12.5)
print("GPe proto Synchronizität korrigiert um Neuronenzahlfaktor 12.5: ",Xi["gpe_proto"]/12.5)"""
