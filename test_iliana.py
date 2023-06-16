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


global saveresults
saveresults = []

### SETUP TIMESTEP + SEED
if paramsS["seed"] == None:
    setup(dt=paramsS["timestep"])
else:
    setup(dt=paramsS["timestep"], seed=paramsS["seed"])

### COMPILE MODEL & GET PARAMTERS   
# BGM_v02_p01 = v01_p01 + Corbit FSI Fit, 
# BGM_v02_p02 = Corbit FSI Fit with Baseline,
# BGM_v02_p03 = Corbit FSI FIt with synaptic delays based on Kumavarelu et al (2015)
# BGM_v02_p04 = Corit FSI Fit model with noise off 
# BGM_v03_p01 = Corbit DD Version
# BGM_v03_p02 = BGM_01_p01 DD Version
modelname = "BGM_v02_p02" 
model = BGM(name=modelname, seed=paramsS["seed"],do_compile=False)
params = model.params


### Set the time constants of the cortex populations                                #### Achtung : hier ueberschriebene Parameterwerte uebergeben um mit neuen Werten zu simulieren !
    ### and compile model
for pop in ["cor_go", "cor_pause", "cor_stop"]:
    for var in ["tau_up", "tau_down"]:
        model.set_param(
            compartment=pop,
            parameter_name=var,
            parameter_value=paramsS[f"{pop}.{var}"])

#for pop in ["str_fsi"]:
#    for var in ["increase_noise"]:
#        setattr(get_population(pop), var, paramsS[f"{pop}.{var}"])

       
model.compile()


### INIT MONITORS ###
mon = Monitors(
    {
        "pop;gpe_arky": ["spike", "g_ampa", "g_gaba"],
        "pop;str_d1": ["spike", "g_ampa", "g_gaba"],
        "pop;str_d2": ["spike", "g_ampa", "g_gaba"],
        "pop;stn": ["spike", "g_ampa", "g_gaba"],
        "pop;cor_go": ["spike"],
        "pop;gpe_cp": ["spike", "g_ampa", "g_gaba"],
        "pop;gpe_proto": ["spike", "g_ampa", "g_gaba"],
        "pop;snr": ["spike", "g_ampa", "g_gaba"],
        "pop;thal": ["spike", "g_ampa", "g_gaba"],
        "pop;cor_stop": ["spike"],
        "pop;str_fsi": ["spike", "g_ampa", "g_gaba"],
        "pop;integrator_go": ["g_ampa", "decision"],
        "pop;integrator_stop": ["g_ampa", "decision"],
    }
)



### GENERATE TRIAL SIMULATION ###
SST_trial = generate_simulation(
    simulation_function=SST_trial_function,
    simulation_kwargs={"params": params, "paramsS": paramsS},
    name="SST_trial",
    description="One trial of SST with cor_go, cor_stop, cor_pause, integrator_go and integrator_stop",
    monitor_object=mon,
)


### TRIALS ###
mon.start()
for mode in ["go","stop"]:
    print("\n\nSTART " + mode + " TRIALS")
    ### LOOP OVER TRIALS
    for _ in tqdm(range(paramsS["trials"])):

        ### TRIAL RUN
        SST_trial.run({"mode": mode})

        ### RESET model/monitors before next trial starts
        mon.reset(populations=True, projections=True, synapses=False, net_id=0)

### END OF ALL TRIALS ###
counter_go = sum(SST_trial.info)
print("TRIALS FINISHED\ncounter_go:", counter_go, "\n")

    


### GET RECORDINGS ###
recordings = mon.get_recordings()
recording_times = mon.get_recording_times()


### Calculate Go reaction times ###
decisiontime_go_array=[]       # array with timesteps of first decision value of -1 = go decision 
decisiontime_stop_array=[] 
decisiontime_stop_array=[]


index_go=[] 
bool_go_array = np.zeros(paramsS["trials"]*2)



for chunk in range(paramsS["trials"]*2):
  #  print(chunk)
  #  print(int(paramsS['t.init']/paramsS['timestep']))
  #  print(len(recordings[chunk]['integrator_go;g_ampa']))

    
    for i in range(int(paramsS['t.init']/paramsS['timestep']),len(recordings[chunk]['integrator_go;g_ampa'])):
       # print(i,end=" ")

       # print('zweite for schleife wird ausgeuehrt')
        if recordings[chunk]['integrator_go;g_ampa'][i]>= params['integrator_go.threshold']:
        #    print("oke")
            index_go.append(chunk)                        # speichert indices der erfolgreichen go versuche 
       
            if chunk < paramsS['trials']:
                decisiontime_go_array.append(i)
            break    
            #else: 
            #    decisiontime_stop_array.append(i)           ### Achtung : das sind gerade failed stop trails !!!!
    #    print(" ")      
         
           
           # break
            
    
for i in index_go:
    bool_go_array[i] = 1

  

#print('go trails of all trails:', bool_go_array)
#print(decisiontime_go_array)


saveresults.append(recordings)
saveresults.append(recording_times)

go_rt= sum(decisiontime_go_array)/ (paramsS["trials"])
stop_rt= sum(decisiontime_stop_array)/ (paramsS["trials"])

go_rt= go_rt*paramsS["timestep"]
stop_rt= stop_rt*paramsS["timestep"]
       

#for chunk in range(paramsS["trials"]*2):
#    spikedict = recordings[chunk]['cor_stop;spike']
#    timearray,_ = raster_plot(spikedict)

#    for i in range(int(timearray.min()/paramsS['timestep']),len(recordings[chunk]['integrator_go;g_ampa'])):
     
#        if recordings[chunk]['integrator_stop;g_ampa'][i]>= params['integrator_stop.threshold']:
#            decisiontime_stop_array.append(i)
#            break


print('zaehler_go:', decisiontime_go_array)
print('average RT:', go_rt)                                         

#with open("output_file.txt", "a") as f:
#    print(f"average RT_go for modulation "+str(modulation)+":" +str(go_rt), file=f)


    


### QUICK PLOT ###
plot_list = [
    "1;gpe_arky;spike;hybrid", # von hybrid auf raster wechseln wenn inputs vom cortex ausgeschaltet sind 
    "2;str_d1;spike;hybrid",
    "3;str_d2;spike;hybrid",
    "4;stn;spike;hybrid",
    "5;cor_go;spike;hybrid",
    "6;gpe_cp;spike;hybrid",
    "7;gpe_proto;spike;hybrid",
    "8;snr;spike;hybrid",
    "9;thal;spike;hybrid",
    "10;cor_stop;spike;hybrid",
    "11;str_fsi;spike;hybrid",
    "12;integrator_stop;g_ampa;line",
    "13;str_fsi;g_ampa;line",
    "14;str_fsi;g_gaba;line",
]

create_dir('results_parameter_fit')

######  fsi connectivity paramter changes

##                                  100%    - 30%       -50%    -70%    -80%    -90%

# fsi_increase_noise =               64      44.8        32     19.2    12.8    6.4for i in range(1,paramsS["trials"]*2):
# cortex_go -> str_fsi.mod_factor =  2.8     1.96       1.4     0.84    0.56    0.28
# str_fsi   -> str_fsi.mod_factor =  0.8     0.56       0.4     0.24    0.16    0.08
# gpe_proto -> str_fsi.mod_factor =  1.6     1.12       0.8     0.48    0.32    0.16
# gpe_arky  -> str_fsi.mod_factor =  6.4     4.48       3.2     1.92    1.28    0.64
# gpe_cp    -> str_fsi.mod_factor =  0.8     0.56       0.4     0.24    0.16    0.08
# thal      -> str_fsi.mod_factor =  9.6     6.72       4.8     2.88    1.92    0.96



chunk = 0
plot_recordings(
    figname='results/test_iliana/overview1_'+str(modelname)+'_inputs_off.png',
    #figname='results/test_iliana/overview1'+str(modelname)+'_ssd320_1000trials.png',
    #figname='results/test_iliana/overview1_str_fsi.increase_noise'+str(modelname)+'_'+str(paramsS["str_fsi.increase_noise"])+'_str_fsi.mean_rate_noise_'+str(paramsS["str_fsi.mean_rate_noise"])+'_fsi_inh_mod_factor_modulation_0_6.png',
    recordings=recordings,
    recording_times=recording_times,
    chunk=chunk,
    shape=(2, 7),
    plan=plot_list,
)

chunk = 1
plot_recordings(
    figname='results/test_iliana/overview2_'+str(modelname)+'_inputs_off.png',
    #figname='results/test_iliana/overview2'+str(modelname)+'_ssd320_1000trials.png'
    #figname='results/test_iliana/overview2_str_fsi.increase_noise'+str(modelname)+'_'+str(paramsS["str_fsi.increase_noise"])+'_str_fsi.mean_rate_noise_'+str(paramsS["str_fsi.mean_rate_noise"])+'_fsi_inh_mod_factor_modulation_0_6.png',
    recordings=recordings,
    recording_times=recording_times,
    chunk=chunk,
    shape=(2, 7),
    plan=plot_list,
)


#np.save('./numpy_results/recordings_'+modelname+'_',saveresults)

