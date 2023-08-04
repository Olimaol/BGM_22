import json
import numpy as np
import matplotlib.pyplot as plt
from CompNeuroPy import create_dir
from sklearn.svm import SVR
from sklearn import preprocessing


def get_I_matrix(I_lat_inp, mode, pop_name):
    I_lat_inp_arr = np.array(I_lat_inp[mode][pop_name])[:, :3]
    # I_lat_inp_arr = np.array(
    #     [
    #         [1, 1, 2],
    #         [1, 2, 2],
    #         [1, 3, 2],
    #         [2, 1, 3],
    #         [2, 2, 3],
    #         [2, 3, 3],
    #         [3, 1, 4],
    #         [3, 2, 4],
    #         [3, 3, 4],
    #         [4, 1, 5],
    #         [4, 2, 5],
    #         [4, 3, 5],
    #     ]
    # )
    lat_unique = np.unique(I_lat_inp_arr[:, 0])
    inp_unique = np.unique(I_lat_inp_arr[:, 1])
    I_lat_inp_mat = np.reshape(I_lat_inp_arr[:, 2], (len(lat_unique), len(inp_unique)))
    return [lat_unique, inp_unique, I_lat_inp_mat]


def get_long_diff_mat(lat_arr, inp_arr, mat):
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
        return None
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
            return get_long_diff_mat(lat_dd, inp_dd, I_mat_dd - I_mat_control)
        else:
            raise ValueError("lat and inp not the same in dd and control!")


def plot_row(row, pop_name, I_lat_inp):

    for column, mode in enumerate(["control", "dd", "diff"]):
        if mode == "diff":
            long_diff_mat = plot_column(column, row, mode, pop_name, I_lat_inp)
        else:
            plot_column(column, row, mode, pop_name, I_lat_inp)

    return long_diff_mat


def get_lat_given_inp(long_diff_mat):
    X_raw = long_diff_mat[:, 1:]
    y_raw = long_diff_mat[:, 0][:, None]

    scaler_X = preprocessing.StandardScaler().fit(X_raw)
    scaler_y = preprocessing.StandardScaler().fit(y_raw)

    X_scaled = scaler_X.transform(X_raw)
    y_scaled = scaler_y.transform(y_raw)[:, 0]

    # SV Regression
    svr_model = SVR(kernel="rbf")
    svr_model.fit(X_scaled, y_scaled)

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


if __name__ == "__main__":
    create_dir("results/fit_pallido_striatal/", clear=False)
    ### load I_lat_inp
    with open("archive/I_lat_inp_10_steps.json") as f:
        I_lat_inp = json.load(f)
    ### plot figure with matrices
    pop_name_list = ["str_d2", "str_fsi", "gpe_proto"]
    plt.figure(figsize=(6.4 * 3, 6.4 * 3))
    long_diff_mat_list = []
    for row, pop_name in enumerate(pop_name_list):
        long_diff_mat = plot_row(row, pop_name, I_lat_inp)
        long_diff_mat_list.append(long_diff_mat)
    plt.tight_layout()
    plt.savefig("results/fit_pallido_striatal/I_base_matrix.png")

    ### use matrices which contain differences between dd and control
    ### to predict lat mod_f based on inp mod_f
    for pop_idx, pop_name in enumerate(pop_name_list):
        ### for str_d2 we do not need to predict the mod_f for lat, one can use any combination
        if pop_name == "str_d2":
            continue
        long_diff_mat = long_diff_mat_list[pop_idx]
        lat_given_inp = get_lat_given_inp(long_diff_mat)

        print(lat_given_inp(0.06))
        print(lat_given_inp([0.05, 0.06, 0.07]))
        quit()  # TODO first rergession done, now second regression TODO
