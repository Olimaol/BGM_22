### This script uses the fitted I_lat_inp values to obtain a model_config of the
### pallido-striatal network.
### This is done in two major steps.
### 1.:
###     Get the inp_mod_f of a population given a lat_mod_f factor. Get a matrix
###     with differences of base_mean values for different lat/inp mod_f values
###     between control and dd. Then train a regression using inp_mod_f and
###     difference==0 as predictors.
### 2.:
###     Get base_mean values using lat_mod_f and inp_mod_f using the I_lat_inp
###     matrices.

import json
import numpy as np
import matplotlib.pyplot as plt
from CompNeuroPy import create_dir
from sklearn.svm import SVR
from sklearn import preprocessing


def get_I_matrix(I_lat_inp, mode, pop_name):
    I_lat_inp_arr = np.array(I_lat_inp[mode][pop_name])[:, :3]
    lat_unique = np.unique(I_lat_inp_arr[:, 0])
    inp_unique = np.unique(I_lat_inp_arr[:, 1])
    I_lat_inp_mat = np.reshape(I_lat_inp_arr[:, 2], (len(lat_unique), len(inp_unique)))
    return [lat_unique, inp_unique, I_lat_inp_mat]


def get_long_mat(lat_arr, inp_arr, mat):
    """
    do the opposite as get_I_matrix(), output is (n,3) shaped array
    n is the number of lat/inp pairs i.e. the number of elements of the matrix
    """
    long_mat = np.zeros((np.prod(mat.shape), 3))
    counter = 0
    for dim0 in range(mat.shape[0]):
        lat = lat_arr[dim0]
        for dim1 in range(mat.shape[1]):
            inp = inp_arr[dim1]
            long_mat[counter] = np.array([lat, inp, mat[dim0, dim1]])
            counter += 1
    return long_mat


def plot_matrix(lat, inp, I_mat, diff):
    plt.imshow(
        I_mat,
        interpolation="none",
        extent=(
            inp[0] - np.diff(inp)[0] / 2,
            inp[-1] + np.diff(inp)[0] / 2,
            lat[-1] + np.diff(lat)[0] / 2,
            lat[0] - np.diff(lat)[0] / 2,
        ),
        cmap=["viridis", "bwr"][int(diff)],
        vmin=[I_mat.min(), -np.absolute(I_mat).max()][int(diff)],
        vmax=[I_mat.max(), +np.absolute(I_mat).max()][int(diff)],
    )
    plt.colorbar()
    # set the labels and ticks
    plt.xlabel("inp")
    plt.ylabel("lat")
    plt.xticks(inp, rotation=90)
    plt.yticks(lat)
    # Calculate the aspect ratio of the figure
    aspect_ratio = (inp[-1] - inp[0]) / (lat[-1] - lat[0])
    # Set the aspect of the image to match the aspect of the figure
    plt.gca().set_aspect(aspect_ratio)


def plot_column(column, row, mode, pop_name, I_lat_inp):

    plt.subplot(3, 3, 3 * row + column + 1)
    plt.title(f"{pop_name}, {mode}")
    if mode != "diff":
        lat, inp, I_mat = get_I_matrix(I_lat_inp, mode=mode, pop_name=pop_name)
        plot_matrix(lat, inp, I_mat, diff=False)
        return get_long_mat(lat, inp, I_mat)
    else:
        lat_control, inp_control, I_mat_control = get_I_matrix(
            I_lat_inp,
            mode="control",
            pop_name=pop_name,
        )
        lat_dd, inp_dd, I_mat_dd = get_I_matrix(
            I_lat_inp,
            mode="dd",
            pop_name=pop_name,
        )
        if (lat_control == lat_dd).all() and (inp_control == inp_dd).all():
            plot_matrix(lat_dd, inp_dd, I_mat_dd - I_mat_control, diff=True)
            return get_long_mat(lat_dd, inp_dd, I_mat_dd - I_mat_control)
        else:
            raise ValueError("lat and inp not the same in dd and control!")


def plot_row(row, pop_name, I_lat_inp):
    long_mat = {}
    for column, mode in enumerate(["control", "dd", "diff"]):
        long_mat[mode] = plot_column(column, row, mode, pop_name, I_lat_inp)

    return long_mat


def get_lat_given_inp(long_diff_mat):
    X_raw = long_diff_mat[:, 1:]
    y_raw = long_diff_mat[:, 0][:, None]

    svr_model, scaler_X, scaler_y = train_SVR(X_raw, y_raw)

    return lambda X: predict_lat(X, svr_model, scaler_X, scaler_y)


def predict_lat(X, svr_model, scaler_X, scaler_y):
    """
    Args:

        X: number, list, or array
    """

    if not (isinstance(X, type(np.array([])))):
        if not (isinstance(X, list)):
            X = [X]
        X = np.array(X)

    X = np.concatenate([X[:, None], np.zeros(len(X))[:, None]], axis=1)

    X_scaled = scaler_X.transform(X)
    pred_scaled = svr_model.predict(X_scaled)
    pred = scaler_y.inverse_transform(pred_scaled)

    return pred


def get_base_given_lat_inp(long_mat_dict):

    ### train regression with I_lat_inp control
    long_mat = long_mat_dict["control"]
    X_raw = long_mat[:, :2]
    y_raw = long_mat[:, 2][:, None]
    svr_model_control, scaler_X_control, scaler_y_control = train_SVR(X_raw, y_raw)

    ### train regression with I_lat_inp dd
    long_mat = long_mat_dict["dd"]
    X_raw = long_mat[:, :2]
    y_raw = long_mat[:, 2][:, None]
    svr_model_dd, scaler_X_dd, scaler_y_dd = train_SVR(X_raw, y_raw)

    return lambda X: predict_base_mean(
        X,
        svr_model_control,
        svr_model_dd,
        scaler_X_control,
        scaler_y_control,
        scaler_X_dd,
        scaler_y_dd,
    )


def train_SVR(X_raw, y_raw):
    scaler_X = preprocessing.StandardScaler().fit(X_raw)
    scaler_y = preprocessing.StandardScaler().fit(y_raw)

    X_scaled = scaler_X.transform(X_raw)
    y_scaled = scaler_y.transform(y_raw)[:, 0]

    # SV Regression
    svr_model = SVR(kernel="rbf")
    svr_model.fit(X_scaled, y_scaled)

    return [svr_model, scaler_X, scaler_y]


def predict_base_mean(
    X,
    svr_model_control,
    svr_model_dd,
    scaler_X_control,
    scaler_y_control,
    scaler_X_dd,
    scaler_y_dd,
):
    """
    Args:

        X: list with two numbers, list of lists with two numbers, or array shape (n,2)
            two numbers are lat_mod_f and inp_mod_f
    """
    if not (isinstance(X, type(np.array([])))):
        X = np.array(X)
    if len(X.shape) == 1:
        X = X[None, :]

    ### get base_mean from control
    X_scaled = scaler_X_control.transform(X)
    pred_scaled = svr_model_control.predict(X_scaled)
    base_mean_control = scaler_y_control.inverse_transform(pred_scaled)

    ### get base_mean from dd
    X_scaled = scaler_X_dd.transform(X)
    pred_scaled = svr_model_dd.predict(X_scaled)
    base_mean_dd = scaler_y_dd.inverse_transform(pred_scaled)

    return {
        "base_mean_control": base_mean_control,
        "base_mean_dd": base_mean_dd,
        "dd_base_factor": base_mean_dd / base_mean_control,
    }


def get_model_config(inp_dict, lat_str_d2):
    lat_given_inp, base_given_lat_inp = prepare_model_config()
    lat_dict = {}
    base_dict = {}
    for pop_name in inp_dict.keys():
        ### get lat given inp
        inp = inp_dict[pop_name]
        if pop_name != "str_d2":
            lat_dict[pop_name] = lat_given_inp[pop_name](inp)[0]
        else:
            lat_dict[pop_name] = lat_str_d2
        ### get base given lat and inp
        lat = lat_dict[pop_name]
        base = base_given_lat_inp[pop_name]([lat, inp])
        base_dict[pop_name] = base["base_mean_control"][0]
        ### get str_d2 base factor
        if pop_name == "str_d2":
            str_d2_base_factor = base["dd_base_factor"][0]
        elif abs(base["dd_base_factor"][0] - 1) > 0.1:
            print(
                f"WARNING get_model_config: Relative difference between base_mean of control and dd ({abs(base['dd_base_factor'][0]-1)}) is larger than 10% for {pop_name}!"
            )

    return {
        "lat_dict": lat_dict,
        "base_dict": base_dict,
        "str_d2_base_factor": str_d2_base_factor,
    }


def prepare_model_config():
    create_dir("results/fit_pallido_striatal/", clear=False)
    ### load I_lat_inp
    with open("fit_pallido_striatal_archive/I_lat_inp.json") as f:
        I_lat_inp = json.load(f)
    ### plot figure with matrices
    pop_name_list = ["str_d2", "str_fsi", "gpe_proto"]
    plt.figure(figsize=(6.4 * 3, 6.4 * 3))
    long_mat_dict = {}
    for row, pop_name in enumerate(pop_name_list):
        long_diff_mat = plot_row(row, pop_name, I_lat_inp)
        long_mat_dict[pop_name] = long_diff_mat
    plt.tight_layout()
    plt.savefig("results/fit_pallido_striatal/I_base_matrix.png")

    ### use matrices which contain differences between dd and control
    ### to predict lat mod_f using inp mod_f
    lat_given_inp = {}
    for pop_idx, pop_name in enumerate(pop_name_list):
        ### for str_d2 we do not need to predict the mod_f for lat, one can use any combination
        if pop_name == "str_d2":
            continue
        long_diff_mat = long_mat_dict[pop_name]["diff"]
        lat_given_inp[pop_name] = get_lat_given_inp(long_diff_mat)

    ### use I_lat_inp matrices to predict base_mean using inp_mod_f and lat_mod_f
    ### I_lat_inp matrices for controll and dd should result in same base_mean values
    base_given_lat_inp = {}
    for pop_idx, pop_name in enumerate(pop_name_list):
        base_given_lat_inp[pop_name] = get_base_given_lat_inp(long_mat_dict[pop_name])

    return [lat_given_inp, base_given_lat_inp]


if __name__ == "__main__":
    ### inp_mod_f for 3 projections and lat_mod_f from str_d2
    ### --> lat_mod_f values, base_mean values, str_d2 dd base factor
    inp_dict = {"str_d2": 1, "str_fsi": 0.03, "gpe_proto": 0.76}
    lat_str_d2 = 2
    model_config = get_model_config(inp_dict, lat_str_d2)
    print(model_config)
