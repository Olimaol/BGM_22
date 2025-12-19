from CompNeuroPy import create_data_raw_folder
from parameters import parameters_test_microcircuit as paramsS


if __name__ == "__main__":
    ### create the data folder
    create_data_raw_folder(
        folder_name=paramsS["data_folder"],
        parameter_dict=paramsS,
    )
