import sys
from pathlib import Path

from CompNeuroPy import run_script_parallel, create_data_raw_folder

### the cortical proportions that mix caudate_rate/putamen_rate are not defined
### here, they are the same ones the striatal input streams are built from
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "BOLD_optimization"))
from parameters import parameters_test_microcircuit as paramsS


if __name__ == "__main__":
    ### create the data folder, recording the mixing weights in __data_raw_meta__
    ### so a rate file can always be traced back to the proportions it used
    create_data_raw_folder(
        folder_name="cortical_firing_rates_data",
        parameter_dict={
            "cortical_proportions_dict": paramsS["cortical_proportions_dict"],
        },
    )
    ### run the cortical_drive_by_bold.py script
    run_script_parallel(
        script_path="cortical_drive_by_bold.py",
        n_jobs=1,
        args_list=[["--condition", "off"], ["--condition", "on"]],
    )
