from CompNeuroPy import Microcircuit
from CompNeuroPy import CompNeuroModel, print_df, CompNeuroMonitors, PlotRecordings
from ANNarchy import setup


if __name__ == "__main__":
    TIMESTEP = 0.1  # ms
    TMAX = 1000.0  # ms
    DBS_CONDITION = "off"  # "off" or "on"
    NAME = "caudate"
    STORAGE_DIR = f"mc_{NAME}_{DBS_CONDITION}_cache"
    setup(dt=TIMESTEP)
    mc = Microcircuit(
        name=NAME,
        dbs_condition=DBS_CONDITION,
        build_connectivity=False,
        build_missing_gaba_input=False,
        build_cortical_input=False,
        dt=TIMESTEP,
        T=TMAX,
        update_time=100.0,
        cortical_rate_path=f"striatal_microcircuit_requirements/cortical_firing_rates/cortical_firing_rates_data/firing_rates_matlab_condition-{DBS_CONDITION}.npz",
        fitted_params_path="striatal_microcircuit_requirements/connectivity_parameters/connectivity_fit_data/fitted_params.json",
        storage_dir=STORAGE_DIR,
    )
    # use all the plot functions of microcircuit
    mc.plot_ext_kdtree_points(show=False)
    mc.plot_neighborhood(show=False)
    mc.plot_neighbor_candidate_counts(show=False)
    mc.plot_connection_probability_boxplots(show=False)
    mc.print_connections_created()
    mc.analyze_degree_distributions(show=False)
    mc.plot_3d_neurons(show=False)
    mc.plot_half_gaussian_curves(show=False)
    mc.plot_all_weight_matrices(show=False)
    mc.plot_weight_matrix("dSPN", "dSPN", show=False)
    mc.plot_weight_matrix("FS", "iSPN", show=False)
    mc.plot_connection_counts_vs_distance(
        per_pair=False, micrometers=True, density=True
    )
    mc.plot_connection_counts_vs_distance(
        per_pair=True, micrometers=True, cumulative=False
    )
    # create a model with the microcircuit
    model = CompNeuroModel(
        model_creation_function=mc.create_model,
        name="Testing Microcircuit Model",
        do_create=True,
        do_compile=True,
        compile_folder_name="microcircuit_test_compile",
    )
    print_df(model.attribute_df)
    print(mc.mean_weights_by_type)

    monitor_dictionary = {
        "caudate_FS": ["spike", "v", "g_ampa", "g_gaba"],
        "caudate_dSPN": ["spike", "v", "g_ampa", "g_gaba"],
        "caudate_iSPN": ["spike", "v", "g_ampa", "g_gaba"],
        "TimedInput_FS_FS_caudate": ["r"],
    }
    monitors = CompNeuroMonitors(monitor_dictionary)
    monitors.start()
    mc.update(run_simulation=True)
    recordings = monitors.get_recordings()
    recording_times = monitors.get_recording_times()

    PlotRecordings(
        figname=f"{STORAGE_DIR}/microcircuit_test_recordings.png",
        recordings=recordings,
        recording_times=recording_times,
        shape=(3, 4),
        plan={
            "position": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "compartment": [
                "caudate_FS",
                "caudate_FS",
                "caudate_FS",
                "caudate_FS",
                "caudate_dSPN",
                "caudate_dSPN",
                "caudate_dSPN",
                "caudate_dSPN",
                "caudate_iSPN",
                "caudate_iSPN",
                "caudate_iSPN",
                "caudate_iSPN",
            ],
            "variable": [
                "spike",
                "v",
                "g_ampa",
                "g_gaba",
                "spike",
                "v",
                "g_ampa",
                "g_gaba",
                "spike",
                "v",
                "g_ampa",
                "g_gaba",
            ],
            "format": [
                "raster",
                "line",
                "line",
                "line",
                "raster",
                "line",
                "line",
                "line",
                "raster",
                "line",
                "line",
                "line",
            ],
        },
    )
