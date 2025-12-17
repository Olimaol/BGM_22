from ANNarchy import setup
from CompNeuroPy.full_models import BGM
from CompNeuroPy import (
    # CompNeuroMonitors,
    # PlotRecordings,
    # Microcircuit,
    # CorticalInputs,
    print_df,
)

### local
from parameters import parameters_test_microcircuit as paramsS


if __name__ == "__main__":

    ### SETUP TIMESTEP + SEED
    if paramsS["seed"] == None:
        setup(dt=paramsS["timestep"])
    else:
        setup(dt=paramsS["timestep"], seed=paramsS["seed"])

    # loop for BG loops
    model_dict = {}
    for loop in ["caudate", "putamen"]:
        ### Prepare the model_creation kwargs for the current dbs condition
        model_creation_kwargs = {
            "build_mc": True,  # The first time this needs to be True to build the microcircuits
            "build_ci": True,  # The first time this needs to be True to build the cortical inputs
            "mc.name": loop,
            "mc.nx": paramsS["mc.nx"],
            "mc.b": paramsS["mc.b"],
            "dbs": paramsS["dbs"],
            "timestep": paramsS["timestep"],
            "t.duration": paramsS["t.duration"],
            "update_time": paramsS["update_time"],
            "mc.storage_dir": f"mc_ci_cache/mc_{loop}_cache_{paramsS['dbs']}",
            "mc.seed": paramsS.get("seed", 42),
            "mc.fitted_params_path": paramsS["mc.fitted_params_path"],
            "mc.cortical_rate_path": paramsS["mc.cortical_rate_path"][paramsS["dbs"]],
            "ci.storage_dir": f"mc_ci_cache/ci_{loop}_cache_{paramsS['dbs']}",
            "ci.seed": paramsS.get("seed", 42),
        }

        ### Create model for the current loop
        model_dict[loop] = BGM(
            name="BGM_v07_p01",
            model_creation_kwargs=model_creation_kwargs,
            seed=paramsS["seed"],
            compile_folder_name=f"bgm_v07_{paramsS['dbs']}",
            name_appendix=loop,
            do_create=True,
            do_compile=False,
        )

    ### Compile model (i.e. both loops in a single model)
    model_dict["caudate"].compile()

    ### Print parameters
    print("model parameters caudate:")
    params = model_dict["caudate"].params
    print(params)
    print("\nmodel attributes caudate:")
    print_df(model_dict["caudate"].attribute_df)

    # ### INIT CompNeuroMonitors ###
    # mon = CompNeuroMonitors(
    #     {
    #         # "cor_go": ["spike"],
    #         # "cor_stop": ["spike"],
    #         # "cor_pause": ["spike"],
    #         # "str_d1": ["spike"],
    #         "str_d2": ["spike"],
    #         "str_fsi": ["spike"],
    #         "gpe_proto": ["spike", "I_base"],
    #         # "gpe_arky": ["spike", "u"],
    #         # "gpe_cp": ["spike"],
    #     }
    # )

    # ### SIMULATION ###
    # mon.start()

    # ### simulate some time
    # simulate(paramsS["t.duration"])

    # ### GET RECORDINGS ###
    # recordings = mon.get_recordings()
    # recording_times = mon.get_recording_times()

    # ### QUICK PLOTS ###

    # ### some populations activity
    # plan = {
    #     "position": [1, 2, 3, 4],
    #     "compartment": ["str_d2", "str_fsi", "gpe_proto", "gpe_proto"],
    #     "variable": ["spike", "spike", "spike", "I_base"],
    #     "format": ["hybrid", "hybrid", "hybrid", "line"],
    # }
    # chunk = 0
    # PlotRecordings(
    #     figname=f"results/test_resting/{model.name}/overview1.png",
    #     recordings=recordings,
    #     recording_times=recording_times,
    #     chunk=chunk,
    #     shape=(1, 4),
    #     plan=plan,
    # )
