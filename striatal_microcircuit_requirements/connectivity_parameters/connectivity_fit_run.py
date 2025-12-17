from CompNeuroPy import run_script_parallel, create_data_raw_folder


if __name__ == "__main__":
    ### create the data folder
    create_data_raw_folder(
        folder_name="connectivity_fit_data",
    )
    ### run the connectivity_fit.py script
    run_script_parallel(
        script_path="connectivity_fit.py",
        n_jobs=1,
    )
