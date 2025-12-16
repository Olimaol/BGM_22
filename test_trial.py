from ANNarchy import setup, get_population
from CompNeuroPy.full_models import BGM
from CompNeuroPy import CompNeuroMonitors, CompNeuroSim, PlotRecordings, print_df
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
    if get_population("integrator_go").decision[0] >= 0:
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

    ### INIT CompNeuroMonitors ###
    mon = CompNeuroMonitors(
        {
            "gpe_arky": ["spike", "g_ampa", "g_gaba"],
            "str_d1": ["spike", "g_ampa", "g_gaba"],
            "str_d2": ["spike", "g_ampa", "g_gaba"],
            "stn": ["spike", "g_ampa", "g_gaba"],
            "cor_go": ["spike"],
            "gpe_cp": ["spike", "g_ampa", "g_gaba"],
            "gpe_proto": ["spike", "g_ampa", "g_gaba"],
            "snr": ["spike", "g_ampa", "g_gaba"],
            "thal": ["spike", "g_ampa", "g_gaba"],
            "cor_stop": ["spike"],
            "str_fsi": ["spike", "g_ampa", "g_gaba"],
            "integrator_go": ["g_ampa", "decision"],
            "integrator_stop": ["g_ampa", "decision"],
        }
    )

    ### GENERATE TRIAL SIMULATION ###
    SST_trial = CompNeuroSim(
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
    plan = {
        "position": list(range(1, 13)),
        "compartment": [
            "gpe_arky",
            "str_d1",
            "str_d2",
            "stn",
            "cor_go",
            "gpe_cp",
            "gpe_proto",
            "snr",
            "thal",
            "cor_stop",
            "str_fsi",
            "integrator_stop",
        ],
        "variable": [
            "spike",
            "spike",
            "spike",
            "spike",
            "spike",
            "spike",
            "spike",
            "spike",
            "spike",
            "spike",
            "spike",
            "g_ampa",
        ],
        "format": [
            "hybrid",
            "hybrid",
            "hybrid",
            "hybrid",
            "hybrid",
            "hybrid",
            "hybrid",
            "hybrid",
            "hybrid",
            "hybrid",
            "hybrid",
            "line",
        ],
    }

    ### 1st trial
    chunk = 0
    PlotRecordings(
        figname=f"results/test_trial/{model.name}/overview1.png",
        recordings=recordings,
        recording_times=recording_times,
        chunk=chunk,
        shape=(2, 6),
        plan=plan,
    )

    ### 2nd trial
    chunk = 1
    PlotRecordings(
        figname=f"results/test_trial/{model.name}/overview2.png",
        recordings=recordings,
        recording_times=recording_times,
        chunk=chunk,
        shape=(2, 6),
        plan=plan,
    )
