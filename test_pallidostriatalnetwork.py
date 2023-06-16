from ANNarchy import setup, get_population, get_time,raster_plot,get_projection,simulate
from CompNeuroPy.models import BGM
from CompNeuroPy import Monitors, generate_simulation, plot_recordings
from tqdm import tqdm
import pylab as plt
from CompNeuroPy import create_dir
import numpy as np

from parameters import parameters_test_iliana as paramsS

global saveresults
saveresults = []

### SETUP TIMESTEP + SEED
if paramsS["seed"] == None:
    setup(dt=paramsS["timestep"])
else:
    setup(dt=paramsS["timestep"], seed=paramsS["seed"])


### COMPILE MODEL & GET PARAMTERS   
# BGM_v04_p01 = Corbit FSI FIT with just Proto-FSI-STRD2 network 
# BGM_v05_p01 = Corbit DD Version with just Proto-FSI-STRD2 network

modelname = "BGM_v05_p01" 
model = BGM(name=modelname, seed=paramsS["seed"],do_compile=False)
params = model.params
 

# set / transfer new parameter values for specific population
for pop in ["cor_go", "cor_pause", "cor_stop"]:
    for var in ["tau_up", "tau_down"]:
        model.set_param(
            compartment=pop,
            parameter_name=var,
            parameter_value=paramsS[f"{pop}.{var}"])

#for pop in ["str_d2"]:
#    for var in ["increase_noise"]:
 #       setattr(get_population(pop), var, paramsS[f"{pop}.{var}"])

model.compile()


### INIT MONITORS ###
mon = Monitors(
    {
        "pop;cor_go": ["spike"],
        "pop;cor_stop": ["spike"],
        "pop;str_d2": ["spike", "g_ampa", "g_gaba"],#"osc"],
        "pop;gpe_proto": ["u","v","spike", "g_ampa", "g_gaba"],#"osc"],
        "pop;str_fsi": ["spike", "g_ampa", "g_gaba"],
    }
)


"""
### simulate network
mon.start()
simulate(2000)
#get_population("cor_go").rates = 400
#simulate(1000)
#get_population("cor_stop").rates = 400
#get_population("cor_go").rates = 0
#simulate(1000)
"""
### LOOP OVER TRIALS
for _ in tqdm(range(paramsS["trials"])):

    ### TRIAL RUN
    ### simulate network for snychrony analysis based on Corbit et al.,2016
    mon.start()
   # simulate(500)
   # get_population("cor_go").rates = 0
  #  simulate(4500)
    get_population("cor_stop").rates = 0
    get_population("cor_go").rates = 0
    simulate(9500)

    ### RESET model/monitors before next trial starts
    mon.reset(populations=True, projections=True, synapses=False, net_id=0)


### GET RECORDINGS ###
recordings = mon.get_recordings()
recording_times = mon.get_recording_times()


### QUICK PLOT ###
plot_list = [
    "1;cor_go;spike;hybrid",
    "2;cor_stop;spike;hybrid",
    "3;str_d2;spike;hybrid",
    "4;gpe_proto;spike;hybrid",
    "5;str_fsi;spike;hybrid",
   # "6;gpe_proto;g_ampa;line",
   # "7;gpe_proto;g_gaba;line",  
  #  "6;str_d2;osc;line",
   # "7;gpe_proto;osc;line",

]

create_dir('results_parameter_fit')


chunk = 0
plot_recordings(
    #figname='results/test_pallidostriatalnet/overview_'+str(modelname)+'_short_simulation_check_gpe_fsi_dynamic_increasemodfactor_1.png',
    figname='results/test_pallidostriatalnet/overview_'+str(modelname)+'_updated_simulations9500ms_9trials_gpe_proto_parameter_fixed_cor_off_gpe-fsi-modfactor-1.png',
    recordings=recordings,
    recording_times=recording_times,
    chunk=chunk,
    shape=(3, 4),
    plan=plot_list,
)

saveresults.append(recordings)

np.save('./numpy_results/recordings_pallidostriatal_network_'+modelname+'_updated_simulations9500ms_9trials_gpe_proto_parameter_fixed_cor_off_gpe-fsi-modfactor-1',saveresults)

