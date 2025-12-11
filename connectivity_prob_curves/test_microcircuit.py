from connectivity_construct import Microcircuit
from CompNeuroPy import CompNeuroModel, print_df, CompNeuroMonitors, PlotRecordings
from ANNarchy import setup


if __name__ == "__main__":
    TIMESTEP = 0.1  # ms
    TMAX = 1000.0  # ms
    setup(dt=TIMESTEP)
    rebuild = False
    mc = Microcircuit(
        name="caudate",
        dbs_condition="off",
        build_connectivity=rebuild,
        build_missing_gaba_input=rebuild,
        build_cortical_input=rebuild,
        dt=TIMESTEP,
        T=TMAX,
        update_time=100.0,
    )
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
    mc.update(run_simulation=True)
    recordings = monitors.get_recordings()
    recording_times = monitors.get_recording_times()

    PlotRecordings(
        figname="microcircuit_test_recordings.png",
        recordings=recordings,
        recording_times=recording_times,
        shape=(5, 1),
        plan={
            "position": [1, 2, 3, 4, 5],
            "compartment": [
                "caudate_FS",
                "caudate_FS",
                "caudate_FS",
                "caudate_FS",
                "TimedInput_FS_FS_caudate",
            ],
            "variable": [
                "spike",
                "v",
                "g_ampa",
                "g_gaba",
                "r",
            ],
            "format": [
                "raster",
                "line",
                "line",
                "line",
                "matrix",
            ],
        },
    )
