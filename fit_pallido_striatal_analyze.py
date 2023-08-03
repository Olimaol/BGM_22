from CompNeuroPy import create_dir, plot_recordings

### local
from fit_pallido_striatal_hyperopt import (
    create_monitors,
    simulate_and_return_loss,
    setup_ANNarchy,
    compile_models,
    paramsS,
)


if __name__ == "__main__":
    create_dir("results/fit_pallido_striatal/")

    ### create file to store results
    with open("results/fit_pallido_striatal/fit_results.json", "w") as f:
        pass
    f.close()

    ### SETUP TIMESTEP + SEED
    setup_ANNarchy()

    ### COMPILE MODELS
    model_dd_list = compile_models()

    ### create monitors
    mon = create_monitors(model_dd_list, analyze=True)

    ## ANALYZE ###
    results = simulate_and_return_loss(
        [
            245.36431788951705,
            1.6751924690779205,
            36,
            2.3,
            0,
            0,
            7.8,
            0,
            0,
            1,
        ],
        return_results=True,
        mon=mon,
        model_dd_list=model_dd_list,
        only_simulate="dd",
        dump=True,
        analyze=True,
    )

    recordings = results["results_dict"]["recordings"]
    recording_times = results["results_dict"]["recording_times"]

    ### plot spikes of all 9 models / 3 populations
    plot_plan = []
    plot_plan_idx = 1
    for n_model in range(paramsS["nbr_models"]):
        for pop_name in ["str_d2", "str_fsi", "gpe_proto"]:
            plot_plan.append(f"{plot_plan_idx};{pop_name}:dd_{n_model};spike;hybrid")
            plot_plan_idx += 1
    shape = (paramsS["nbr_models"], 3)
    plot_recordings(
        "results/fit_pallido_striatal/analyze_spikes.png",
        recordings=recordings,
        recording_times=recording_times,
        chunk=0,
        shape=shape,
        plan=plot_plan,
    )

    ### plot variables of gpe proto
    for variable in ["I_base"]:
        plot_plan = []
        plot_plan_idx = 1
        for n_model in range(paramsS["nbr_models"]):
            for pop_name in ["str_d2", "str_fsi", "gpe_proto"]:
                plot_plan.append(
                    f"{plot_plan_idx};{pop_name}:dd_{n_model};{variable};line"
                )
                plot_plan_idx += 1
        shape = (paramsS["nbr_models"], 3)
        plot_recordings(
            f"results/fit_pallido_striatal/analyze_{variable}.png",
            recordings=recordings,
            recording_times=recording_times,
            chunk=0,
            shape=shape,
            plan=plot_plan,
        )

    ### gpe exploded for too high lateral inhibition --> question was why
    ### answer: I_gaba fluctuated because v fluctuated around E_gaba (-90)
    ### this section helped finding this error
    # ### analyze in detail gpe neurons of model 5
    # plot_plan = [
    #     "1;gpe_proto:dd_5;g_gaba;line",
    #     "2;gpe_proto:dd_5;g_ampa;line",
    #     "3;gpe_proto:dd_5;v;line",
    #     "4;gpe_proto:dd_5;u;line",
    # ]
    # plot_recordings(
    #     "results/fit_pallido_striatal/analyze_model_5_gpe.png",
    #     recordings=recordings,
    #     recording_times=recording_times,
    #     chunk=0,
    #     shape=(2, 2),
    #     plan=plot_plan,
    #     time_lim=[0, 50],
    # )
    # ### even more detailed
    # for neuron_idx in range(8):
    #     plt.figure(figsize=(6.4 * 2, 4.8 * 3), dpi=300)
    #     plt.subplot(311)
    #     plt.plot(
    #         recordings[0]["gpe_proto:dd_5;g_gaba"][
    #             : int(40 / paramsS["timestep"]), neuron_idx
    #         ]
    #     )

    #     plt.subplot(312)
    #     plt.plot(
    #         recordings[0]["gpe_proto:dd_5;v"][
    #             : int(40 / paramsS["timestep"]), neuron_idx
    #         ]
    #     )

    #     plt.subplot(313)
    #     plt.plot(
    #         recordings[0]["gpe_proto:dd_5;u"][
    #             : int(40 / paramsS["timestep"]), neuron_idx
    #         ]
    #     )

    #     plt.tight_layout()
    #     plt.savefig(
    #         f"results/fit_pallido_striatal/analyze_model_5_gpe_neuron_{neuron_idx}.png"
    #     )
    #     plt.close()
