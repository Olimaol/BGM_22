from ANNarchy import setup, get_population
from CompNeuroPy.models import BGM
from CompNeuroPy import Monitors, generate_simulation, plot_recordings, print_df
from tqdm import tqdm

### local
from trial_procedure import trial_procedure_cl
from trial_events import add_events
from parameters import parameters_default as paramsS


### DEFINE TRIAL FUNCTION ###
def SST_trial_function(params, paramsS, mode="go"):
    ### TRIAL START

    ### define trial procedure
    trial_procedure = trial_procedure_cl(params, paramsS, mode=mode)

    ### add events
    add_events(trial_procedure)

    ### run trial procedure
    trial_procedure.run()

    ### return if go decision was made
    if get_population("integrator_go").decision[0] == -1:
        return 1
    else:
        return 0


if __name__ == "__main__":

    ### SETUP TIMESTEP + SEED
    if paramsS["seed"] == None:
        setup(dt=paramsS["timestep"])
    else:
        setup(dt=paramsS["timestep"], seed=paramsS["seed"])

    ### CREATE MODEL & GET MODEL PARAMTERS
    model = BGM(name="BGM_v01_p01", seed=paramsS["seed"], do_compile=False)
    params = model.params

    ### Set the time constants of the cortex populations
    ### and compile model
    for pop in ["cor_go", "cor_pause", "cor_stop"]:
        for var in ["tau_up", "tau_down"]:
            model.set_param(
                compartment=pop,
                parameter_name=var,
                parameter_value=paramsS[f"{pop}.{var}"],
            )
    model.compile()

    print("model paramters:")
    print_df(model.attribute_df)

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
    for mode in ["go", "stop"]:
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

    ### 1st trial
    chunk = 0
    plot_recordings(
        figname=f"results/test_trial/{model.name}/overview1.png",
        recordings=recordings,
        recording_times=recording_times,
        chunk=chunk,
        shape=(2, 6),
        plan=plot_list,
    )

    ### 2nd trial
    chunk = 1
    plot_recordings(
        figname=f"results/test_trial/{model.name}/overview2.png",
        recordings=recordings,
        recording_times=recording_times,
        chunk=chunk,
        shape=(2, 6),
        plan=plot_list,
    )
