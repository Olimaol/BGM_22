import numpy as np
from fit_Proto_Arky import get_loss
import matplotlib.pyplot as plt
from ANNarchy import setup

setup(dt=0.1)

id_list = list(range(1, 2))
loss_arr = np.zeros((len(id_list), 2))
for id_idx, id in enumerate(id_list):
    best_proto = np.load(
        f"dataRaw/parameter_fit/best_proto_{id}.npy", allow_pickle=True
    ).item()
    best_arky = np.load(
        f"dataRaw/parameter_fit/best_arky_{id}.npy", allow_pickle=True
    ).item()
    loss_arr[id_idx, 0] = best_proto["loss"]
    loss_arr[id_idx, 1] = best_arky["loss"]

print(loss_arr)

best_id_proto = np.argmin(loss_arr[:, 0])
best_proto = np.load(
    f"dataRaw/parameter_fit/best_proto_{id_list[best_id_proto]}.npy", allow_pickle=True
).item()
best_id_arky = np.argmin(loss_arr[:, 1])
best_arky = np.load(
    f"dataRaw/parameter_fit/best_arky_{id_list[best_id_arky]}.npy", allow_pickle=True
).item()

f_I_proto = get_loss(
    best_proto["results"],
    best_proto["results_soll"],
    return_ist=True,
    which_neuron="proto",
)
print_fitting_dict = {}
for key, val in best_proto.items():
    if not (key in ["loss", "all_loss", "std", "results", "results_soll"]):
        print_fitting_dict[key] = val
print(
    np.sqrt(np.mean((f_I_proto["ist"] - f_I_proto["target"]) ** 2)),
    print_fitting_dict,
)
f_I_arky = get_loss(
    best_arky["results"],
    best_arky["results_soll"],
    return_ist=True,
    which_neuron="arky",
)
print_fitting_dict = {}
for key, val in best_arky.items():
    if not (key in ["loss", "all_loss", "std", "results", "results_soll"]):
        print_fitting_dict[key] = val
print(
    np.sqrt(np.mean((f_I_arky["ist"] - f_I_arky["target"]) ** 2)),
    print_fitting_dict,
)

plt.figure(dpi=300)
plt.plot(f_I_proto["inj_current_pA"], f_I_proto["ist"], label="PROTO fit")
plt.plot(f_I_proto["inj_current_pA"], f_I_proto["target"], label="PROTO target")
plt.plot(f_I_arky["inj_current_pA"], f_I_arky["ist"], label="ARKY fit")
plt.plot(f_I_arky["inj_current_pA"], f_I_arky["target"], label="ARKY target")
plt.legend()
plt.tight_layout()
plt.savefig("best_fit.png")
