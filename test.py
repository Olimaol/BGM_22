from ANNarchy import *
from CompNeuroPy.models import BGM
from CompNeuroPy import Monitors, generate_simulation
import CompNeuroPy.analysis_functions as af
from CompNeuroPy.extra_functions import flatten_list
from CompNeuroPy.analysis_functions import get_number_of_zero_decimals
import pylab as plt
import time
from tqdm import tqdm
from trial_procedure import trial_procedure_cl
from trial_events import add_events


### SETUP TIMESTEP + SEED
seed_val = 1
setup(dt=0.1, seed=seed_val)


### COMPILE MODEL & GET PARAMTERS
model = BGM(name='BGM_v01_p01', seed=seed_val)
params = model.params
paramsS = {}
paramsS['trials'] = 1


### INIT MONITORS ###
mon = Monitors({'pop;gpe_arky':['spike','g_ampa','g_gaba'],
                'pop;str_d1':['spike','g_ampa','g_gaba'],
                'pop;str_d2':['spike','g_ampa','g_gaba'],
                'pop;stn':['spike','g_ampa','g_gaba'],
                'pop;cor_go':['spike'],
                'pop;gpe_cp':['spike','g_ampa','g_gaba'],
                'pop;gpe_proto':['spike','g_ampa','g_gaba'],
                'pop;snr':['spike','g_ampa','g_gaba'],
                'pop;thal':['spike','g_ampa','g_gaba'],
                'pop;cor_stop':['spike'],
                'pop;str_fsi':['spike','g_ampa','g_gaba'],
                'pop;integrator_go':['g_ampa', 'decision'],
                'pop;integrator_stop':['g_ampa', 'decision']})


### DEFINE TRIAL FUNCTION ###
def SST_trial_function(params, paramsS, mode='go'):
    start=time.time()    

    ### TRIAL START
    ### trial INITIALIZATION simulation to get stable state
    get_population('cor_go').rates = 0
    get_population('cor_stop').rates = 0
    get_population('cor_pause').rates = 0

    ### simulate t_init resting period
    simulate(params['sim.t_init'])

    ### Integrator Reset
    get_population('integrator_go').decision = 0
    get_population('integrator_go').g_ampa = 0  
    get_population('integrator_stop').decision = 0 
    
    
    ### define trial procedure
    trial_procedure = trial_procedure_cl(params, paramsS, mode=mode)
    
    ### add events
    add_events(trial_procedure, params)
    
    ### run trial procedure
    trial_procedure.run()
    
    ### return if go decision was made
    if get_population('integrator_go').decision[0] == -1 :
        return 1
    else:
        return 0


### GENERATE TRIAL SIMULATION ###
SST_trial = generate_simulation(simulation_function=SST_trial_function,
                                simulation_kwargs={'params':params, 'paramsS':paramsS},
                                name='SST_trial',
                                description='One trial of SST with cor_go, cor_stop, cor_pause, integrator_go and integrator_stop',
                                monitor_object=mon)


### TRIALS ###
mon.start()
for mode in ['go','stop']:
    print('\n\nSTART '+mode+' TRIALS')
    ### LOOP OVER TRIALS
    for _ in tqdm(range(paramsS['trials'])):

        ### TRIAL RUN
        SST_trial.run({'mode':mode})

        ### RESET model/monitors before next trial starts
        mon.reset(populations=True, projections=True, synapses=False, net_id=0)

### END OF ALL TRIALS ###
counter_go = sum(SST_trial.info)
print('TRIALS FINISHED\ncounter_go:',counter_go,'\n')


### GET RECORDINGS ###
recordings=mon.get_recordings()
recording_times=mon.get_recording_times()


### QUICK PLOT ###
plot_list = ['1;gpe_arky;spike;hybrid',
             '2;str_d1;spike;hybrid',
             '3;str_d2;spike;hybrid',
             '4;stn;spike;hybrid',
             '5;cor_go;spike;hybrid',
             '6;gpe_cp;spike;hybrid',
             '7;gpe_proto;spike;hybrid',
             '8;snr;spike;hybrid',
             '9;thal;spike;hybrid',
             '10;cor_stop;spike;hybrid',
             '11;str_fsi;spike;hybrid',
             '12;integrator_stop;g_ampa;line']

chunk=0
time_lims = recording_times.time_lims(chunk=chunk)
idx_lims  = recording_times.idx_lims(chunk=chunk)
af.plot_recordings('overview1.png', recordings[chunk], time_lims, idx_lims, (2,6), plot_list)

chunk=1
time_lims = recording_times.time_lims(chunk=chunk)
idx_lims  = recording_times.idx_lims(chunk=chunk)
af.plot_recordings('overview2.png', recordings[chunk], time_lims, idx_lims, (2,6), plot_list)

