from connectivity_construct import Microcircuit
from CompNeuroPy import CompNeuroModel, CompNeuroMonitors, print_df
from ANNarchy import setup


if __name__ == "__main__":
    setup(dt=0.1)
    mc = Microcircuit(build_connectivity=False, build_missing_gaba_input=False)
    model = CompNeuroModel(
        model_creation_function=mc.create_model,
        name="Testing Microcircuit Model",
        do_create=True,
        do_compile=True,
        compile_folder_name="microcircuit_test_compile",
    )
    print_df(model.attribute_df)
    print(mc.mean_weights_by_type)
