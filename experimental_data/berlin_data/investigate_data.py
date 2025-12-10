import h5py
import matplotlib.pyplot as plt
import numpy as np


CORTICAL_LABELS = (
    "M1",
    "PMd",
    "PMv",
    "preSMA",
    "SMA",
    "S1",
    "dlPFC",
)


def print_file_contents(filename):
    print(f">>>>>>>>>>>>> Reading {filename}\n")
    with h5py.File(filename, "r") as f:
        ### print keys
        print(f"Keys: {list(f.keys())}\n")

        ### datasets
        con_mats = f[list(f.keys())[0]]
        labels = f[list(f.keys())[1]]
        time_series = f[list(f.keys())[2]]

        print("Type of datasets:")
        print(list(f.keys())[0], type(con_mats))
        print(list(f.keys())[1], type(labels))
        print(list(f.keys())[2], type(time_series), "\n")

        ### labels = dataset --> could obtain a numpy array
        labels_arr = labels[()]
        labels_arr = np.array([label.decode("UTF-8") for label in labels_arr])
        print("Labels shape:", labels_arr.shape)
        print("Labels:", labels_arr, "\n")

        ### mats and time seris are groups with labels "on" and "off"
        ### with labels get datasets --> again can get array
        time_series_arr_on = time_series["on"][()]
        time_series_arr_off = time_series["off"][()]
        mats_arr_on = con_mats["on"][()]
        mats_arr_off = con_mats["off"][()]
        print(f"{list(f.keys())[2]} 'on' shape:", time_series_arr_on.shape)
        print(f"{list(f.keys())[2]} 'off' shape:", time_series_arr_off.shape)
        print(f"{list(f.keys())[0]} 'on' shape:", mats_arr_on.shape)
        print(f"{list(f.keys())[0]} 'off' shape:", mats_arr_off.shape)
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n\n")


def load_time_series(filename):
    with h5py.File(filename, "r") as f:
        labels = np.array([label.decode("UTF-8") for label in f["labels"][()]])
        ts_on = f["time_series"]["on"][()]
        ts_off = f["time_series"]["off"][()]
    return labels, ts_on, ts_off


def get_cortical_indices(labels, n_cols):
    if len(labels) != n_cols:
        print(
            f"Warning: labels ({len(labels)}) and columns ({n_cols}) differ; "
            f"using first {n_cols} labels."
        )
    trimmed_labels = labels[:n_cols]
    indices = [i for i, lbl in enumerate(trimmed_labels) if lbl in CORTICAL_LABELS]
    if not indices:
        raise ValueError("No cortical labels found in dataset.")
    return trimmed_labels, indices


def plot_cortical_time_series(filename):
    labels, ts_on, ts_off = load_time_series(filename)
    labels, cortical_indices = get_cortical_indices(labels, ts_on.shape[1])

    print("Cortical BOLD statistics (mean +/- std):")
    for idx in cortical_indices:
        on_mean = float(np.mean(ts_on[:, idx]))
        on_std = float(np.std(ts_on[:, idx]))
        off_mean = float(np.mean(ts_off[:, idx]))
        off_std = float(np.std(ts_off[:, idx]))
        print(
            f"  {labels[idx]}: on={on_mean:.4f} +/- {on_std:.4f}, "
            f"off={off_mean:.4f} +/- {off_std:.4f}"
        )

    time = np.arange(ts_on.shape[0])
    fig, axes = plt.subplots(
        len(cortical_indices), 1, figsize=(10, 2 * len(cortical_indices)), sharex=True
    )
    axes = np.atleast_1d(axes)

    for ax, idx in zip(axes, cortical_indices):
        ax.plot(time, ts_on[:, idx], label="on", color="tab:blue")
        ax.plot(time, ts_off[:, idx], label="off", color="tab:orange", linestyle="--")
        ax.set_ylabel(labels[idx])
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Time (samples)")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":

    filename = "bold_data_roi/sub-01/sub-01_subdiv_results.h5"
    print_file_contents(filename)
    plot_cortical_time_series(filename)
