from ANNarchy import setup, get_population, get_time,raster_plot,get_projection,simulate, dt
from CompNeuroPy import Monitors, generate_simulation, plot_recordings,get_population_power_spectrum
from tqdm import tqdm
import pylab as plt
import numpy as np
import pickle
from CompNeuroPy.models import BGM

from parameters import parameters_test_iliana as paramsS

global dt 
global saveresults
saveresults = []


"""
NOTE : all generated data with simulation time 9500ms should be 9000ms long (- first 500ms) for all following calculations !!!!

"""

### example for structure
#data = np.load('numpy_results/recordings_pallidostriatal_network_BGM_v04_p01_updated_simulation9500ms_9trials_gpe_arky_parameter_fixed_cor_off.npy', allow_pickle=True)
# print(rec[0][0].keys())             --> output : dict_keys(['cor_go;period', 'cor_go;spike', 'cor_stop;period', 'cor_stop;spike'... 'str_fsi;g_gaba', 'dt'])
# print(rec[0][0]['str_d2;spike'][0]) --> output : [2710, 3359,..., 19155, 19184, 19428, 19489]



def bool_spike_matrix(data,pop_name, sim_time,del_time):
    """
    generate matrix containing all time points, all spikes time points get values 1, else 0 

    args :      data = recordings
                pop_name = population names as string
                analysis_time = simulation time in dt steps - first 500ms 
                del_time = discarded first 500ms for each analysis
  
    returns :   bool_spikes = boolean spike matrix, value = 1 for a spike, else 0, discarded first 500ms -> needed for other calculations, was my first approch 
                spikes[chunk][pop] = spike matrix (neurons x analysis_time/dt) containing all spiketimings, else 0 ,discarded first 500ms -> necessary for power spectrum !
                sumspikes[chunk][pop] = amount of spikes in each pop for each trial, discarded first 500ms 

    """

    
    spikes = {} # empty dict
    sumspikes = {}
    bool_spikes = {}

    for chunk in range(paramsS["trials"]):

        spikes[chunk]={}
        for pop in pop_name:                                           # ["str_d2","str_fsi","gpe_arky"] 
            spikes[chunk][pop] = {}
            for i in range(paramsS[f"{pop}.size"]):
                spikes[chunk][pop][i] = []                                                 
                for n in range(len(recordings[chunk][f"{pop};spike"][i])):          # tip : iterating over a list, does not need a "range"  
                    if recordings[chunk][f"{pop};spike"][i][n] >= del_time:    
                        spikes[chunk][pop][i].append(recordings[chunk][f"{pop};spike"][i][n])


    for chunk in range(paramsS["trials"]):
        bool_spikes[chunk]={}
        sumspikes[chunk]={}
        for pop in pop_name:                                           # ["str_d2","str_fsi","gpe_arky"] 
            bool_spikes[chunk][pop]  = np.zeros((paramsS[f"{pop}.size"],sim_time)) 
            sumspikes[chunk][pop] = 0
            temp = 0
            for i in range(paramsS[f"{pop}.size"]):                                                
                for n in recordings[chunk][f"{pop};spike"][i]:          # tip : iterating over a list, does not need a "range"  
                    bool_spikes[chunk][pop][i,n] = 1
               
                temp += len(recordings[chunk][f"{pop};spike"][i]) 
            sumspikes[chunk][pop] = temp

    
    for chunk in range(paramsS["trials"]):
        for pop in pop_name:  
                bool_spikes[chunk][pop]= bool_spikes[chunk][pop][:,del_time:sim_time]
    
    return bool_spikes, spikes, sumspikes 



def check_spike_matrix(data,pop_name, analysis_time, spikes,sumspikes) : 
    """
    check if number of spikes is correct by calculating the difference between the "zaehler" and "sumspikes" for each chunk and pop -> difference should be 0
    
    args :      data = recordings
                pop_name = population names as string
                analysis_time = simulation time in dt steps - first 500ms 
                spikes = boolean spike matrix, generated in function boolean_spike_matrix
                sumspikes = = amount of spikes in each pop for each trial

    return :    difference of the spike counters ; should be 0, if the boolean spike array is correct 

    """
    zaehler = {}
    for chunk in range(paramsS["trials"]):
        zaehler[chunk]= {}
        for pop in pop_name:               #["str_d2","str_fsi","gpe_arky"]:
            zaehler[chunk][pop] = 0
            for i in range(paramsS[f"{pop}.size"]):
                for n in range(analysis_time):
                    if spikes[chunk][pop][i,n] != 0:               
                        zaehler[chunk][pop]+=1
    
    differences = []
    for chunk in range(paramsS["trials"]):
        for pop in pop_name:
            differences.append(zaehler[chunk][pop] - sumspikes[chunk][pop])

    return differences
       

def mean_firingrate(data,pop_name,analysis_time,del_time):
    """
    calculates the average firingrate for each pop

    args :      data = recordings  
                pop_name = population names as string
                pop_size = amount of neruons in the populations
                analysis_time = simulation time in dt steps - first 500ms 
                del_time = discarded first 500ms for each analysis

    return :    freq = dict with mean firingrate for each pop for each chunk
    """ 

    spike_counter = {}
    freq = {}

    for chunk in range(paramsS["trials"]):
        spike_counter[chunk] =  {}
        freq[chunk] = {}
        for pop in pop_name: 
            spike_counter[chunk][pop] = 0
            freq[chunk][pop] = 0 
            for i in range(paramsS[f"{pop}.size"]):
                temp = del_time
                temp = len([t for t in recordings[chunk][f"{pop};spike"][i] if t >= temp])
                spike_counter[chunk][pop] += temp
            print("fertig mit feuerrate")
            spike_counter[chunk][pop] = spike_counter[chunk][pop]/paramsS[f"{pop}.size"]              #TODO mean firing rate calculation divide through pop_size
            freq[chunk][pop] = spike_counter[chunk][pop]/analysis_time  # ms                  
            freq[chunk][pop] = freq[chunk][pop] * 10000                       # ms to s, with dt = 0.1ms
        

    with open("/scratch/ilko/BGM_22/analysis_data/mean_firingrate_" + model_name + ".pkl", "wb") as file:           
        pickle.dump(freq, file)

    return freq


def spike_synchrony(bool_spikes, pop_name, bin_size, analysis_time):           
    """
    calculates spike synchrony Xi based on the within cell and population variance (equations, Corbit et al., 2016)
    using counts of spikes binned into 15ms bins (Methods,Corbit et al., 2016)
    chunk simulated for 9500ms,  first 500ms discarded, 9 trials (Corbit et al., 2016)
        -> problem / question : how much and how long should be the inputs for str_d2 and gpe_arky ?
        -> first approach : first half stimulate str_d2, then gpe_arky, SSD 4500ms 

    args :  bool_spikes = boolean spike matrix calculated in function before, dict
            pop_name = population names as string, list
            bin_size = time range in which spikes will be counted, should be in dt steps, int
            analysis_time = simulation time in dt steps - first 500ms, int
           

    return : variances for each cell and each population, 
             spike synchrony values for each population over all trials, 
             spike synchrony values depending on the population size (Corbit et al.,2016 use different population sizes)
    """

    counter = {} # empty dict 
    var_cell = {}
    counter_pop = {}
    var_pop = {}


    # divide generated matrix in bins and count spikes for each cell in each pop in each chunk
    for chunk in range(paramsS["trials"]):
        counter[chunk] = {}
        for pop in pop_name:
            counter[chunk][pop] = np.zeros((paramsS[f"{pop}.size"],int(analysis_time/bin_size)))      # count all spikes per bin for each pop for each chunk
            for i in range(paramsS[f"{pop}.size"]): 
                bin_counter = 0
                for bins in range(0,(analysis_time-bin_size),bin_size):            
                    bins = int(bins)
                    
                    # calculate sum for each bin -> boundary from bins to bins + bin_size :
                    beg = bins
                    end = bins+bin_size
                    temp = np.sum(bool_spikes[chunk][pop][i,beg:end])
                    counter[chunk][pop][i,bin_counter] = temp
                    bin_counter+=1   
                   


    # calculate within cell variance (equation,Corbit et al.,2016)
    for chunk in range(paramsS["trials"]):
        var_cell[chunk] = {}
        for pop in pop_name:
            var_cell[chunk][pop] =  np.zeros(paramsS[f"{pop}.size"])
            for i in range(paramsS[f"{pop}.size"]):
                # equation from Corbit et al., 2016 :
                var_cell[chunk][pop][i] = (np.mean(counter[chunk][pop][i]**2)) - (np.mean(counter[chunk][pop][i])**2)


    # calculate population variance (equation,Corbit et al.,2016)
    for chunk in range(paramsS["trials"]):
        var_pop[chunk] = {}
        counter_pop[chunk] = {}
        for pop in pop_name:
            var_pop[chunk][pop]= np.zeros((1,bin_counter+1)) 
            counter_pop[chunk][pop] = np.zeros(bin_counter+1) 
            for bins in range(bin_counter+1):          # 9000 / 15 = 600 -> 599 +1        
                counter_pop[chunk][pop][bins]= np.mean(counter[chunk][pop][:,bins])       
                # or : counter_pop[pop] = np.sum(counter[pop][i,bins], axis=0)    # mean over all cells i for each pop        
                # axis 0, sum along the columns/neurons ; axis 1, sum along the lines/bins
            # equation from Corbit et al., 2016 :
            var_pop[chunk][pop] =  (np.mean(counter_pop[chunk][pop]**2)) - (np.mean(counter_pop[chunk][pop])**2)
    print("fertig mit varianz")      
    # calculate spike synchrony
    Xi = {}
    mean_var_cell = {}
    mean_var_pop = {}
    Xi_corrected = {}
    Xi_corrected[pop] = 0
    
    for pop in pop_name:
        Xi[pop] = 0
        mean_var_cell[pop] = 0
        mean_var_pop[pop] = 0
   
    for chunk in range(paramsS["trials"]):
        for pop in pop_name:
            mean_var_pop[pop] += var_pop[chunk][pop]
            for i in range(paramsS[f"{pop}.size"]):
                mean_var_cell[pop] += var_cell[chunk][pop][i]
            
    for pop in pop_name:
        mean_var_cell[pop] = mean_var_cell[pop] / (paramsS["trials"] * paramsS[f"{pop}.size"]) # equals 1/N * sum of cell variances
        mean_var_pop[pop] = mean_var_pop[pop] / paramsS["trials"] 
        Xi[pop] = mean_var_pop[pop] / mean_var_cell[pop]
    

    # Xi corrected based on number of neurons (strd2 : 100 vs 40 -> 2.5,  str_fsi & gpe : 100 vs 8 -> 12.5) in model and model of Corbit et al.(2016)
    #Xi_corrected["str_d2"] = Xi["str_d2"]*2.5
    #Xi_corrected["str_fsi"] = Xi["str_fsi"]*12.5
    #Xi_corrected["gpe_arky"] = Xi["gpe_arky"]*12.5

    with open("/scratch/ilko/BGM_22/analysis_data/spike_synchrony_" + model_name + ".pkl", "wb") as file:           
        pickle.dump(Xi, file)

    #with open("/scratch/ilko/BGM_22/analysis_data/spike_synchrony_corrected_" + model_name + ".pkl", "wb") as file:           
        #pickle.dump(Xi_corrected, file)

    return var_cell, var_pop, Xi#, Xi_corrected


def spike_synchrony_matrix(data,pop_name,analysis_time,del_time,bool_spikes):
    """
    classiying spikes in each pop in synchronous and asynchronous spikes;
    spike synchronous, if spike is followed by at least three other spikes from the same pop within 10ms before or after this spike (Corbit et al., 2016)
    
    args :  data = recordings, dict
            pop_name = population names as string, list
            analysis_time = simulation time in dt steps - first 500ms, int
            del_time = discarded first 500ms for each analysis
            bool_spikes[chunk][pop] = boolean spike matrix,discarded first 500ms

    return : sync_spikes[chunk][pop] = matrix (neurons x simulation time in dt steps) contains spiketimings of synchronous spike, else 0 ; discarded first 500ms
             async_spikes[chunk][pop] = matrix (neurons x simulation time in dt steps), contains spiketimings of asynchronous spike, else 0; discarded first 500ms
             sync_prop[chunk][pop] = proportion of synchronous spikes in each pop per chunk
    """

    sync_spikes = {}
    async_spikes = {}
    total_spike_amount = {}
    sync_spike_amount = {}
    sync_range = int(10/dt)
       
    # generate boolean matrix with all synchronous spikes per pop per chunk with values 1, else 0 = asynchronous    -> update : sync_spike and async_spike matrix necessary for power spectra !
    for chunk in range(paramsS["trials"]):
        sync_spikes[chunk]={}
        async_spikes[chunk] = {}
        total_spike_amount[chunk] = {}
        sync_spike_amount[chunk] = {}

        for pop in pop_name:                                          
            sync_spikes[chunk][pop] = {} #np.zeros((paramsS[f"{pop}.size"],analysis_time))
            async_spikes[chunk][pop] = {} # np.zeros((paramsS[f"{pop}.size"],analysis_time))
            total_spike_amount[chunk][pop] = 0
            sync_spike_amount[chunk][pop] = 0
       
            for i in range(paramsS[f"{pop}.size"]):    
                sync_spikes[chunk][pop][i] = []
                async_spikes[chunk][pop][i] = []
                   
                for n in range(len(recordings[chunk][f"{pop};spike"][i])):  
                    if recordings[chunk][f"{pop};spike"][i][n] >= del_time:   
                        total_spike_amount[chunk][pop] += 1    
                            
                        if n-sync_range > del_time and n+sync_range <= len(bool_spikes[chunk][pop][i]):
                            # find all synchronous spikes, value 1
                            if np.sum(bool_spikes[chunk][pop][i,n-sync_range:n+sync_range]) > 3:             # sync_spikes and async_spikes should be 9000/dt steps long, because spikes has already discardet first 500ms 
                                sync_spikes[chunk][pop][i].append(n)        
                                sync_spike_amount[chunk][pop]+= 1
                            # find all asynchronous spike, value 2
                            else:
                                #sync_spikes[chunk][pop][i,n] = 2
                                async_spikes[chunk][pop][i].append(n)
            print("fertig mit sync und async matrizen")
            
    # calculate proportion of synchronous spikes
    sync_prop= {}
    for pop in pop_name:
        sync_prop[pop] = 0
        for chunk in range(paramsS["trials"]):
            sync_prop[pop] += sync_spike_amount[chunk][pop] / total_spike_amount[chunk][pop]
        sync_prop[pop] /= paramsS["trials"]
        

    with open("/scratch/ilko/BGM_22/analysis_data/spike_synchrony_matrix_" + model_name + ".pkl", "wb") as file:           
        pickle.dump(sync_spikes, file)

    with open("/scratch/ilko/BGM_22/analysis_data/spike_asynchrony_matrix_" + model_name + ".pkl", "wb") as file:           
        pickle.dump(async_spikes, file)

    with open("/scratch/ilko/BGM_22/analysis_data/spike_synchrony_proportion_" + model_name + ".pkl", "wb") as file:           
        pickle.dump(sync_prop, file)

    return sync_spikes, async_spikes, sync_prop



def pause(data, pop_name, analysis_time): 

    """    
    calculates the amount of firing pauses in fsi and their power spectrum 
    pause is defined as epochs of at least 20ms of no spike after a spike (Corbit et al., 2016)

    args :  data = recordings
            pop_name = population names as string
            analysis_time = simulation time in dt steps - first 500ms, int         

    return  pause_matrix =  matrix with contains the index of fsi pause, else 0     # dict with all entries as fsi pause beginnings
    """            
    

    pause_matrix = {}           
    pause_time = int(20/dt)

    # generate matrix containing all time points; all pause beginning time points get values 1, else 
    for chunk in range(paramsS["trials"]):
        pause_matrix[chunk]={}

        for pop in pop_name:    
            pause_matrix[chunk][pop] = {} #np.zeros((paramsS[f"{pop}.size"],analysis_time)) 
           
            for i in range(paramsS[f"{pop}.size"]):
                pause_matrix[chunk][pop][i] = []
                tmp = recordings[chunk][f"{pop};spike"][i]    
                                                                                      
                for n_idx in range(len(tmp)-1):
                    n = tmp[n_idx]
                    if n >= del_time:    
                        if n == recordings[chunk][f"{pop};spike"][i][-1]:          # -1 in order to reference the last entry
                            continue
                                                                                                
                        else:
                            if recordings[chunk][f"{pop};spike"][i][n_idx+1] - recordings[chunk][f"{pop};spike"][i][n_idx] >= pause_time:     
                                pause_matrix[chunk][pop][i].append(n_idx) 

        print("fertig mit fsi pausen")
    with open("/scratch/ilko/BGM_22/analysis_data/spike_pause_matrix" + model_name + ".pkl", "wb") as file:           
        pickle.dump(pause_matrix, file)
                
    return pause_matrix



def latency_after_synchrony(data,spikes,sync_spikes,async_spikes):
    """
    calculates the latency of the first spike of an FSI / MSN neuron (averaged over all neurons of the population) after each synchronous and asychronous GPe Spike

    args :      data = recordings 
                spikes =  matrix containing spiketimings of each neuron, else 0 -> FSI and MSN needed      
                sync_spikes = matrix containing all spike values for each neuron in each pop, with spiketiming for a synchronous spike, else 0 -> GPe needed
                async_spikes = matrix containing all spike values for each neuron in each pop, with spiketiming for a asynchronous spike, else 0 -> GPe needed
                pop_name = population names

    return:     sync_diff_gpe_fsi_spikes = list of latencies of a first FSI spike after each synchronous Gpe Spike, list
                async_diff_gpe_fsi_spikes = list of latencies of a first FSI spike after each asynchronous Gpe Spike, list 
    """

    sync_diff_gpe_fsi_spike = []
    async_diff_gpe_fsi_spike = []
    

    # calculate the time difference between a synchronous GPe spike and the first following FSI spike 
    for chunk in range(paramsS["trials"]):
            for i in range(paramsS["str_fsi.size"]):
                
                for n in range(len(spikes[chunk]["str_fsi"][i])):  

                    sync_gpe_array =np.concatenate(list(sync_spikes[chunk]["gpe_arky"].values())) # spikes werden zu 1D Array
                    fsi_time = spikes[chunk]["str_fsi"][i][n]
                    if np.sum(sync_gpe_array < fsi_time) > 0:
                        gpe_sync_time = sync_gpe_array[sync_gpe_array < fsi_time][-1]  # zeit des letzten sync. gpe spike vor fsi_time
                        latency = fsi_time - gpe_sync_time    
                        sync_diff_gpe_fsi_spike.append(latency)

                    async_gpe_array =np.concatenate(list(async_spikes[chunk]["gpe_arky"].values())) # spikes werden zu 1D Array
                    fsi_time = spikes[chunk]["str_fsi"][i][n]
                    if np.sum(async_gpe_array < fsi_time) > 0:
                        gpe_async_time = async_gpe_array[async_gpe_array < fsi_time][-1]  # zeit des letzten sync. gpe spike vor fsi_time
                        latency_async = fsi_time - gpe_async_time    
                        async_diff_gpe_fsi_spike.append(latency_async)
                    """       
                    # case for each synchronous gpe spike
                    if sync_spikes[chunk]["gpe_arky"][i] != 0 :
                        for latency in range(1,len(bool_spikes[chunk]["gpe_arky"][i])):
                            tmp = len(bool_spikes[chunk]["str_fsi"][i])
                            if n + latency < (tmp-1):                           
                                if bool_spikes[chunk]["str_fsi"][i,n+latency] == 1:
                                    sync_diff_gpe_fsi_spike.append(latency)
                                    break

                    # case for each asynchronous gpe spike 
                    if async_spikes[chunk]["gpe_arky"][i] != 0 :
                        for latency in range(1,len(bool_spikes[chunk]["gpe_arky"][i])):
                            if n + latency < len(bool_spikes[chunk]["str_fsi"][i]):      # latency index starts with 1,ensure that latency in nor out of bounds
                                if bool_spikes[chunk]["str_fsi"][i,n+latency] == 1:
                                    async_diff_gpe_fsi_spike.append(latency)
                                    break
                    """  
                   

    print("fertig mit latenzen")
    # Note : connectivity pattern : fixed_number_pre = Each fsi neuron receives connections from a fixed number of gpe neurons (10) chosen randomlY -> alternative method ?
    

    with open("/scratch/ilko/BGM_22/analysis_data/fsi_latency_after_synchronous_gpe_spike_" + model_name + ".pkl", "wb") as file:           
        pickle.dump(sync_diff_gpe_fsi_spike, file)

    with open("/scratch/ilko/BGM_22/analysis_data/fsi_latency_after_asynchronous_gpe_spike_" + model_name + ".pkl", "wb") as file:           
        pickle.dump(async_diff_gpe_fsi_spike, file)

    return sync_diff_gpe_fsi_spike, async_diff_gpe_fsi_spike 


def get_total_beta_power(spikes,rec_times,pop_name):
    """
    calculates the total beta power of each population,
    concrete : area integral in the range of the beta frequency band (13-30 Hz)

    args:       spikes = dict containing spike timings of each cell in each pop for each chunk
                rec_times = recordings_times from the monitor 
                pop_name = population names as string


    return :    beta_power = dict containing the power of all frequencies within the beta frequency band(13-30Hz)
                total_beta_power = dict containing the total beta power of each pop over all chunks, area integral           # check how many chunks !
    """

    f_min = 13 # in Hz
    f_max = 30 # in Hz
    beta_power = {}    
    total_beta_power = {}
    freq = {}
    power = {}

    # get the frequency and spectrum/power of each pop and each chunk, dict with list of frequencies
    for chunk in range(paramsS["trials"]):
        freq[chunk] = {}
        power[chunk] = {}

        for pop in pop_name:
            freq[chunk][pop] = []
            power[chunk][pop] = []

            freq[chunk][pop],power[chunk][pop] = get_population_power_spectrum(
            spikes= spikes[chunk][pop],                         
                time_step= 0.1,                             #recordings_BGM_v04_p01[chunk][f"{pop_name};dt")      #TODO
                t_start= rec_times.time_lims(chunk=chunk)[0],
                t_end= rec_times.time_lims(chunk=chunk)[1],
                fft_size=4096,
            )

    # check if frequencies are in the beta band -> add their power to a list to calculate the integral
    for chunk in range(paramsS["trials"]):
        beta_power[chunk] = {}
        for pop in pop_name:
            beta_power[chunk][pop] = []
            for i in range(paramsS[f"{pop}.size"]):
                if freq[chunk][pop][i] > f_min and freq[chunk][pop][i] <= f_max:
                    beta_power[chunk][pop].append(power[chunk][pop][i])
    # integral equal to sum of samples :
            beta_power[chunk][pop] = np.sum(beta_power[chunk][pop])

    for pop in pop_name:
        total_beta_power[pop] = 0
        for chunk in range(paramsS["trials"]):      
            total_beta_power[pop] += np.sum(beta_power[chunk][pop])        
            total_beta_power[pop] = total_beta_power[pop] / paramsS["trials"]

    print("fertig mit beta power")
    with open("/scratch/ilko/BGM_22/analysis_data/beta_power_" + model_name + ".pkl", "wb") as file:           
        pickle.dump(beta_power, file)

    with open("/scratch/ilko/BGM_22/analysis_data/total_beta_power_" + model_name + ".pkl", "wb") as file:           
        pickle.dump(total_beta_power, file)

    return beta_power, total_beta_power
                




if __name__ == "__main__":

    ### SETUP TIMESTEP + SEED
    for seed_value in range(1, 10):
        setup(dt=paramsS["timestep"], seed=seed_value)                  # important to change the seed for significance tests ! 


    ### COMPILE MODEL & GET MODEL PARAMTERS
    global model_name 
    model_name= "BGM_v05_p02"                      
    model = BGM(model_name, seed=seed_value,do_compile=False)
    params = model.params

    # SET AFTER MODEL IS SET 
    for pop in ["str_d2","str_fsi","gpe_arky"]:
        for var in ["increase_noise"]:
            setattr(get_population(pop), var, paramsS[f"{pop}.{var}"])

    model.compile()

    ### INIT MONITORS ###
    mon = Monitors(
    {   
    "pop;str_d2": ["v","spike", "g_ampa", "g_gaba"],
    "pop;gpe_arky": ["v","spike", "g_ampa", "g_gaba"],
    "pop;str_fsi": ["v","spike", "g_ampa", "g_gaba"],      
    }
    )

    ### SIMULATION ###
    mon.start()
      

    ### LOOP OVER TRIALS
    for _ in tqdm(range(paramsS["trials"])):
        ### TRIAL RUN
        #paramsS["str_d2.I_app"] = 100
        #simulate(500)
        #paramsS["str_d2.I_app"] = 0
        #simulate(500)
        #paramsS["str_d2.I_app"] = 100
        #simulate(500)
        #paramsS["str_d2.I_app"] = 0
        #simulate(500)
        simulate(9500)
        ### RESET model/monitors before next trial starts
        mon.reset(populations=True, projections=True, synapses=False, monitors = False, net_id=0)


    ### GET RECORDINGS ###
    recordings = mon.get_recordings()
    recording_times = mon.get_recording_times()

    ### save recordings()
    with open("/scratch/ilko/BGM_22/analysis_data/recordings_" + model_name + ".pkl", "wb") as file:           # wb = write in binary mode
        pickle.dump(recordings, file)

    with open("/scratch/ilko/BGM_22/analysis_data/recording_times_" + model_name + ".pkl", "wb") as file:           # wb = write in binary mode
        pickle.dump(recording_times, file)


    """
    first execute function "bool_spike_matrix" that returns a boolean spike matrix and a spike matrix containing the specific spiketimings , which are arguments for other functions 

    """

    # settings for pallido-striatal network
    pops = ["str_d2","str_fsi","gpe_arky"]
    dt = paramsS["timestep"]
    sim_time = int(9500/dt)
    del_time = int(500/dt)
    analysis_time = sim_time - del_time
    bin_size = int(15/dt)

    #saveresults.append(recordings)
    #np.save('./numpy_results/recordings_pallidostriatal_network_'+model_name+'_spiking_pattern_str_d2_fsi_gpe_arky',saveresults)   
   
 
    # 1. generate boolean spike matrix for all other calculations !
    bool_spike_matrix,spike_matrix, sumspikes = bool_spike_matrix(recordings,pops,sim_time,del_time)
    
    # Mean Firingrates
    freq = mean_firingrate(recordings,pops, analysis_time,del_time)
    print(freq)
    
    # Spike Synchrony and Variances (within cell and population)
    var_cell, var_pop, Xi = spike_synchrony(bool_spike_matrix, pops,bin_size, analysis_time)
    print(Xi)
    """
    # Synchronous, asynchronous matrix and the proportion of synchronous spikes
    sync_matrix,async_matrix,sync_prop = spike_synchrony_matrix(recordings,pops,analysis_time,del_time,bool_spike_matrix)
    
    # Pause matrix 
    pause_matrix = pause(recordings, pops, analysis_time)

    # Latencies of fsi spikes after a synchronous and asynchronous gpe spike
    latency, async_latency = latency_after_synchrony(recordings,spike_matrix,sync_matrix,async_matrix)
    """
    # list of all power values in the beta band range and total beta power
    beta_pow, total_beta_pow = get_total_beta_power(spike_matrix,recording_times,pops)
    print(total_beta_pow)
    

