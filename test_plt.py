from ANNarchy import setup, get_population, get_time,raster_plot,get_projection, Population, reset
from CompNeuroPy.models import BGM, H_and_H_model_Bischop,H_and_H_model_Corbit
from CompNeuroPy import generate_model as gm
from CompNeuroPy import system_functions as sf
from CompNeuroPy import Monitors, generate_simulation, plot_recordings
from tqdm import tqdm
import pylab as plt
import CompNeuroPy as cnp
import matplotlib.pyplot as plt
from CompNeuroPy.neuron_models import Izhikevich2007_fsi_noisy_AMPA,Izhikevich2007_Corbit_FSI_noisy_AMPA 
from CompNeuroPy import create_dir
import numpy as np

### local
from trial_procedure import trial_procedure_cl
from trial_events import add_events
from parameters import parameters_test_iliana as paramsS
from test_trial import SST_trial_function

import pylab as plt
import numpy as np
from CompNeuroPy import create_dir


### v- I plot / comparison of membrane potentials of different FSI neuron implementations ###

### load data
#results_target = np.load('dataRaw/generate_H_and_H_data/results.npy', allow_pickle=True).item()
results_target = np.load('/scratch/ilko/compneuro_test/dataRaw/generate_H_and_H_Corbit_data_update/results.npy', allow_pickle=True).item()

sp_target = results_target.data['sim'][0]
recordings_target = results_target.recordings
recording_times_target = results_target.data['recording_times']

### from optimization
sim_id=1
#best = np.load('dataRaw/parameter_fit/best_'+str(sim_id)+'.npy', allow_pickle=True).item()
best = np.load('/scratch/ilko/compneuro_test/dataRaw/parameter_fit/best_64.npy', allow_pickle=True).item()


results_opt = best['results']
sp_opt = results_opt['sim'][0]
recordings_opt = results_opt['recordings']
recording_times_opt = results_opt['data']['recording_times']


#print(recordings_target[0])
### combine data
times, data_arr_target = recording_times_target.combine_chunks(recordings_target, sp_target.kwargs[0]['pop']+';v', mode='consecutive')
_    , data_arr_opt    = recording_times_opt.combine_chunks(recordings_opt, sp_opt.kwargs[0]['pop']+';v', mode='consecutive')


### obtain simulation times and stimulation current from simulation protocol
time_lims_0 = recording_times_target.time_lims(chunk=0)
time_lims_1 = recording_times_target.time_lims(chunk=1)
simulation_times = np.arange(time_lims_0[0], np.diff(time_lims_0)+np.diff(time_lims_1), recordings_target[0]['dt'])# both chunks combined
stimulation_current = simulation_times.copy()
for i in range(3):
    #time_before=sum([sp_target[j]['t1']+sp_target[j]['t2'] for j in range(i)])
    #stimulation_current[simulation_times>time_before]=sp_target[i]['a1']
    #time_before+=sp_target[i]['t1']
    #stimulation_current[simulation_times>time_before]=sp_target[i]['a2']
    

    time_before=sum([sp_target.kwargs[j]['t1']+sp_target.kwargs[j]['t2'] for j in range(i)])
    stimulation_current[simulation_times>time_before]=sp_target.kwargs[i]['a1']
    time_before+=sp_target.kwargs[i]['t1']
    stimulation_current[simulation_times>time_before]=sp_target.kwargs[i]['a2']

x_max = 20499
x_min = x_max - 30000

#print(np.max(data_arr_target))
#print(np.min(data_arr_target))
#print(type(data_arr_target))
#print(data_arr_target.shape)

### plot results
plt.figure(dpi=300)
plt.subplot(311)
y_target = np.clip(data_arr_target, None, 0)
plt.plot(times, y_target, color='k', label='target', lw=0.3)
plt.ylim(min([data_arr_target.min(),data_arr_opt.min()]),0)
plt.ylabel('v [mV]')
plt.xlim(x_min,x_max)
plt.legend()
plt.subplot(312)
y_opt = np.clip(data_arr_opt, None, 0)
plt.plot(times, y_opt, color='grey', label='ist', lw=0.3)
plt.ylim(min([data_arr_target.min(),data_arr_opt.min()]),0)
plt.ylabel('v [mV]')
plt.xlim(x_min,x_max)
plt.legend()
plt.subplot(313)
plt.plot(simulation_times, stimulation_current)
plt.ylabel('$I_{app}$ [pA]')
plt.xlabel('time [ms]')
plt.xlim(x_min,x_max)
### save
#create_dir('results/parameter_fit')
plt.savefig('./plot_v_I_best67_print_.svg')
quit()


### firing rate plot / f- I plots ###
### setup ANNarchy
setup(dt=0.01)

def frequency(my_sim_f,run,chunk,recordings,population):
    neuron= 0   
    freq = {}  
    counter = 0
    counter_freq = 0
    bound = 0
    spiketimings ={}

    spiketimings = np.array(recordings[chunk][population+';spike'][neuron])
        
    freq = np.ones(my_sim_f.kwargs[run]['nr_steps']*2)
    for i in range(len(spiketimings)):
        if spiketimings[i]  <= bound   :     
            counter += 1
                
        else :  
            freq[counter_freq] = counter * (1000/my_sim_f.kwargs[run]['durationI2'])                
            counter_freq += 1
            bound += my_sim_f.kwargs[run]['durationI2']/recordings[0]['dt']  
            counter = 0

    return freq

def parameter_BGM(pop):
    pop.a = 0.2; pop.a_n = 0; pop.a_s = 0; pop.b = 0.025; pop.b_n = 0; pop.c = -60.0;pop.k = 1.0;pop.v_t = -50.0;
    pop.x = 1;pop.v_r = -70.0; pop.C = 80; pop.I_app = 0; pop.v_peak = 25 ;pop.v = -70.0; pop.u = pop.s = pop.n = 0

def parameter_Corbit(pop):
    pop.a = 0.3920799898889739; pop.a_n = 0.00839718024601107; pop.a_s = 0.008634047286600421; pop.b = 0.1924761718877654; pop.b_n = 0.8582265026999014; pop.c = -76.85523437197861; pop.d = 1.9341071763119118;
    pop.k = 0.06583963677366121; pop.v_t = -62.978664531308496; pop.x = 2.057807159546473;pop.v_r = -70.04; pop.C = 1; pop.I_app = 0; pop.v_peak = 30 ;pop.v = -70.04; pop.u = pop.s = pop.n = 0


def create_model(model, neuron,neuronname, parameter_function):
     pop = Population(neuron, neuron=model , name= neuronname)
     parameter_function(pop)



fit_C_model = cnp.generate_model(model_creation_function=create_model, 
                          model_kwargs={'model':Izhikevich2007_Corbit_FSI_noisy_AMPA, 'neuron':1, 'neuronname':'Izhikevich2007_Corbit_FSI_noisy_AMPA','parameter_function':parameter_Corbit},          
                          name='Izhikevich2007_Corbit_final',                       
                          do_create=True,                      
                          do_compile=False)                      
                          #compile_folder_name='Izhikevich2007_Corbit_final')      
  

BGM01_model = cnp.generate_model(model_creation_function=create_model, 
                          model_kwargs={'model':Izhikevich2007_fsi_noisy_AMPA, 'neuron':1, 'neuronname':'Izhikevich2007_fsi_noisy_AMPA','parameter_function':parameter_BGM },            
                          name='Izhikevich2007_fsi_noisy_AMPA',                       
                          do_create=True,                      
                          do_compile=False)                   
                          #compile_folder_name='Izhikevich2007_fsi_noisy_AMPA')   
 
### create and compile model and population
model_C = H_and_H_model_Corbit()
population = model_C.populations[0]
fit_population = fit_C_model.populations[0]
BGM01_population = BGM01_model.populations[0]

mon = cnp.Monitors({'pop;'+population:['v','spike'],'pop;'+fit_population:['v','spike'],'pop;'+BGM01_population:['v','spike']})
      
### define the simulation for f - I data
my_sim_f = cnp.generate_simulation(cnp.simulation_functions.increasing_current,
    simulation_kwargs={'pop':population, 'I1': 0, 'step':0.5, 'nr_steps':400, 'durationI2':500,},
    name='increasing_current') #,            
    # requirements=[{'req':cnp.simulation_requirements.req_pop_attr, 'model':'simulation_kwargs.pop', 'attr':'I_app'}],
    # kwargs_warning=False)
        
mon.start()
my_sim_f.run({'pop':population})
mon.reset()
my_sim_f.run({'pop':fit_population})
mon.reset()
my_sim_f.run({'pop':BGM01_population})
### SIMULATION END

recordings= mon.get_recordings()
recording_times=mon.get_recording_times()   
              
plot_recordings(
    figname='f-I_Plot_H&H.png',
    recordings=recordings,
    recording_times=recording_times,
    chunk=0,
    shape=(1,1),
    plan= [f"1;{population};v;line"]
)

plot_recordings(
    figname='f-I_PlotCorbit_Fit.png',
    recordings=recordings,
    recording_times=recording_times,
    chunk=1,
    shape=(1,1),
    plan= [f"1;{fit_population};v;line"]
)
plot_recordings(
    figname='f-I_Plot_BGMv01.png',
    recordings=recordings,
    recording_times=recording_times,
    chunk=2,
    shape=(1,1),
    plan= [f"1;{BGM01_population};v;line"]
)

    
### calculate firingrate for each current step 
run=0; chunk=0; neuron= 0   
freq = frequency(my_sim_f,run,chunk,recordings,population)
run=1; chunk=1; neuron= 0   
freq_fit = frequency(my_sim_f,run,chunk,recordings,fit_population)
run=2; chunk=2; neuron= 0   
freq_BGM01 = frequency(my_sim_f,run,chunk,recordings,BGM01_population)
        
### f-I plot
plt.figure(dpi=300)
plt.subplot(111)
x = my_sim_f.info[run]['current_list']
plt.plot(x,freq_fit,label='FSI neuron fit based on Corbit et al.(2016)')
plt.plot(x,freq, label='H&H FSI Model')
plt.plot(x,freq_BGM01, label='BGM01 FSI model')
plt.title("Comparison firing rates of different FSI models ")
plt.xlabel('$I_{app}$ [pA]')
plt.ylabel('frequency [Hz]')
plt.legend()

plt.show()
plt.savefig('plot_f_I_comparison_id64.svg')


### rasterplots ###

### Histograms of Go related reaction time distributions ###

go_rt_distribution = np.load('././numpy_results/results_decisiontime_go_array_BGM_v01_p01.npy')

# get reaction time in ms
for i in range(len(go_rt_distribution)):
    go_rt_distribution[i]= go_rt_distribution[i]*paramsS["timestep"]


for i in range(len(go_rt_distribution)):
    go_rt_distribution[i]= go_rt_distribution[i]/10
    go_rt_distribution[i]= np.around(go_rt_distribution[i])
    go_rt_distribution[i]= go_rt_distribution[i] * 10


list = [3.2,4.5,3,56.4,74.467,458,45,85,69,45.3,45.2,45,45.7,45,6,56,45.6]
#print(list)   
#list = np.around(list)   # np.around() zum auf bzw. abrunden auf integer werte

plt.hist(go_rt_distribution[0], bins = np.arange(800,1200,10), edgecolor = 'black')
plt.title("Distribution of response times of successful Go trials")
plt.xlabel("Reaction times in ms")
plt.ylabel("Number of trials")
#plt.show()


print((go_rt_distribution[0]))
