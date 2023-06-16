from ANNarchy import setup, get_population, get_time,raster_plot,get_projection
from CompNeuroPy.models import BGM
from CompNeuroPy import Monitors, generate_simulation, plot_recordings
from tqdm import tqdm
import pylab as plt
from CompNeuroPy import create_dir
import numpy as np

### local
from trial_procedure import trial_procedure_cl
from trial_events import add_events
from parameters import parameters_test_iliana as paramsS
from test_trial import SST_trial_function

 

### load data
results_target = np.load('numpy_results/recordings_BGM_v01_p01_.npy', allow_pickle=True)
results_target = results_target[0]
print(results_target)
quit()


sp_target = results_target['sim'][0]
recordings_target = results_target['recordings']
recording_times_target = results_target['data']['recording_times']


### from optimization
sim_id=1
#best = np.load('dataRaw/parameter_fit/best_'+str(sim_id)+'.npy', allow_pickle=True).item()
best = np.load('dataRaw/parameter_fit/best_64.npy', allow_pickle=True).item()

results_opt = best['results']
sp_opt = results_opt['sim'][0]
recordings_opt = results_opt['recordings']
recording_times_opt = results_opt['data']['recording_times']


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
#quit()
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
create_dir('results/parameter_fit')
plt.savefig('results/parameter_fit/plot_v_I_best67_print_.svg')



