from CompNeuroPy import run_script_parallel, create_data_raw_folder


if __name__ == "__main__":
    ### create the data folder
    create_data_raw_folder(
        folder_name="cortical_firing_rates_data",
    )
    ### run the cortical_drive_by_bold.py script
    run_script_parallel(
        script_path="cortical_drive_by_bold.py",
        n_jobs=1,
        args_list=[["--condition", "off"], ["--condition", "on"]],
    )
