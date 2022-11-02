from ANNarchy import setup, get_population, get_time
from CompNeuroPy.models import BGM
from CompNeuroPy import Monitors, generate_simulation, plot_recordings
from tqdm import tqdm
import pylab as plt
from CompNeuroPy import create_dir

### local
from trial_procedure import trial_procedure_cl
from trial_events import add_events


### SETUP TIMESTEP + SEED
seed_val = 1
setup(dt=0.1, seed=seed_val)


### COMPILE MODEL & GET PARAMTERS
model = BGM(name="BGM_v02_p01", seed=seed_val)
params = model.params
paramsS = {}
paramsS["trials"] = 100


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


### DEFINE TRIAL FUNCTION ###
def SST_trial_function(params, paramsS, mode="go"):
    ### TRIAL START

    ### define trial procedure
    trial_procedure = trial_procedure_cl(params, paramsS, mode=mode)

    ### add events
    add_events(trial_procedure, params)

    ### run trial procedure
    trial_procedure.run()

    ### return if go decision was made
    if get_population("integrator_go").decision[0] == -1:
        return 1
    else:
        return 0


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


### QUICK PLOT ###
plot_list = [
    "1;gpe_arky;spike;hybrid",
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
]

#create_dir('results_parameter_fit')

######  fsi connectivity paramter changes

##                                  100%    - 30%       -50%    -70%    -80%    -90%

# fsi_increase_noise =               64      44.8        32     19.2    12.8    6.4
# cortex_go -> str_fsi.mod_factor =  2.8     1.96       1.4     0.84    0.56    0.28
# gpe_proto -> str_fsi.mod_factor =  1.6     1.12       0.8     0.48    0.32    0.16
# gpe_arky  -> str_fsi.mod_factor =  6.4     4.48       3.2     1.92    1.28    0.64
# gpe_cp    -> str_fsi.mod_factor =  0.8     0.56       0.4     0.24    0.16    0.08
# thal      -> str_fsi.mod_factor =  9.6     6.72       4.8     2.88    1.92    0.96

chunk = 0
plot_recordings(
    figname='overview1_go_stop_BGM_v02_p01_mod_factor_increase_noise_-90per.png',
    recordings=recordings,
    recording_times=recording_times,
    chunk=chunk,
    shape=(2, 6),
    plan=plot_list,
)

chunk = 1
plot_recordings(
    figname='overview2_go_stop_BGM_v02_p01_mod_factor_increase_noise_-90per.png',
    recordings=recordings,
    recording_times=recording_times,
    chunk=chunk,
    shape=(2, 6),
    plan=plot_list,
)

#plt.savefig('results_parameter_fit/overview1_v02_go_stop_BGM_v02_p01.svg')
#plt.savefig('results_parameter_fit/overview2_v02_go_stop_BGM_v02_p01.svg')

