from ANNarchy import setup, simulate
from CompNeuroPy.models import BGM
from CompNeuroPy import (
    Monitors,
    plot_recordings,
    print_df,
)

### local
from parameters import parameters_test_resting as paramsS


if __name__ == "__main__":

    ### SETUP TIMESTEP + SEED
    if paramsS["seed"] == None:
        setup(dt=paramsS["timestep"])
    else:
        setup(dt=paramsS["timestep"], seed=paramsS["seed"])

    ### COMPILE MODEL & GET MODEL PARAMTERS
    model = BGM(name="BGM_v04newgpe_p01", seed=paramsS["seed"])
    params = model.params

    print("model paramters:")
    print_df(model.attribute_df)

    ### INIT MONITORS ###
    mon = Monitors(
        {
            # "pop;cor_go": ["spike"],
            # "pop;cor_stop": ["spike"],
            # "pop;cor_pause": ["spike"],
            # "pop;str_d1": ["spike"],
            "pop;str_d2": ["spike"],
            "pop;str_fsi": ["spike"],
            "pop;gpe_proto": ["spike", "I_base"],
            # "pop;gpe_arky": ["spike", "u"],
            # "pop;gpe_cp": ["spike"],
        }
    )

    ### SIMULATION ###
    mon.start()

    ### simulate some time
    simulate(paramsS["t.duration"])

    ### GET RECORDINGS ###
    recordings = mon.get_recordings()
    recording_times = mon.get_recording_times()

    ### QUICK PLOTS ###

    ### some populations activity
    plot_list = [
        # "1;cor_go;spike;hybrid",
        # "2;cor_stop;spike;hybrid",
        # "3;cor_pause;spike;hybrid",
        # "4;str_d1;spike;hybrid",
        "1;str_d2;spike;hybrid",
        "2;str_fsi;spike;hybrid",
        "3;gpe_proto;spike;hybrid",
        # "3;gpe_arky;spike;hybrid",
        # "9;gpe_cp;spike;hybrid",
        "4;gpe_proto;I_base;line",
    ]
    chunk = 0
    plot_recordings(
        figname=f"results/test_resting/{model.name}/overview1.png",
        recordings=recordings,
        recording_times=recording_times,
        chunk=chunk,
        shape=(1, 4),
        plan=plot_list,
    )
