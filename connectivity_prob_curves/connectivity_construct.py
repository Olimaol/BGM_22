# %%
import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt
import os
import json

# =====================================
# Parameters
# =====================================
# Dimensions (number of cells)
nx = 10  # along caudate-putamen axis (x)
b = 10  # along y and z axes
density = 84900  # neurons per mm^3

# Proportions, del Rey et al. 2022
props_delRey = np.array([0.026, 0.86 / 2, 0.86 / 2])
props = props_delRey / np.sum(props_delRey)
props = {
    "FS": props[0],
    "dSPN": props[1],
    "iSPN": props[2],
}
cell_types = list(props.keys())

# Connection parameters: load from JSON created by connectivity_fit.py
script_dir = os.path.dirname(__file__)
output_dir = os.path.join(script_dir, "connectivity_construct")
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(script_dir, "connectivity_fits", "fitted_params.json")) as f:
    fitted = json.load(f)
conn_params = {
    tuple(key.split("-")): (val["amplitude"], val["sigma_um"])
    for key, val in fitted.items()
}

# compute max sigma (um) per post type to set neighbor‐query radius
max_sigma_um = {
    post_type: max(
        sigma_um
        for (_, post_type2), (_, sigma_um) in conn_params.items()
        if post_type2 == post_type
    )
    for post_type in cell_types
}

# =====================================
# Generate neuron positions and types
# =====================================
n_total = nx * b * b
# physical spacing (mm) per index step assuming each cell occupies equal volume cube
d = (1.0 / density) ** (1 / 3)  # mm per cell index
# Margin for analysis (in index units), based on 3× the global max σ
# This is not needed because the outer neurons get the same number of neighbors due to
# periodic boundaries
global_max_sigma_um = max(sigma_um for (_, _), (_, sigma_um) in conn_params.items())
margin = int(np.ceil(3 * global_max_sigma_um * 1e-3 / d))  # margin in index units
margin = 0
# positions in mm
xs = np.arange(nx) * d
ys = np.arange(b) * d
zs = np.arange(b) * d

# meshgrid to get coordinates
X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
positions = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T  # shape (n_total, 3)

# initialize RNG with a fixed seed
seed = 42
rng = np.random.default_rng(seed)

# assign types by calculating exact counts per type and randomizing their positions
# types array (n_total,)
type_counts = {ct: int(props[ct] * n_total) for ct in cell_types}
rem = n_total - sum(type_counts.values())
for ct in sorted(cell_types, key=lambda x: props[x], reverse=True)[:rem]:
    type_counts[ct] += 1
types = np.array([ct for ct, count in type_counts.items() for _ in range(count)])
rng.shuffle(types)


# compute and print cuboid dimensions, volume, and neuron counts
d_um = d * 1e3
dim_x_um = nx * d_um
dim_y_um = b * d_um
dim_z_um = b * d_um
volume_mm3 = n_total / density
counts = {ct: np.sum(types == ct) for ct in cell_types}
print(f"Cuboid dimensions (μm): X={dim_x_um:.2f}, Y={dim_y_um:.2f}, Z={dim_z_um:.2f}")
print(f"Volume: {volume_mm3:.3f} mm³")
for ct, cnt in counts.items():
    print(f"Neurons ({ct}): {cnt}")
print(f"Grid spacing: {d_um:.2f} μm")

# =====================================
# Build KDTree for neighbor queries
# =====================================
# Build periodic KDTree by tiling the cube in all 27 shifts (26 neighbors + original)
L = np.array([nx * d, b * d, b * d])  # cube dimensions in mm
shifts = np.array(
    [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
)
ext_positions = np.vstack([positions + shift * L for shift in shifts])
tree = sp.cKDTree(ext_positions)

# visualize the full tree points in 3D (µm)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection="3d")
ax.scatter(
    ext_positions[:, 0] * 1e3,
    ext_positions[:, 1] * 1e3,
    ext_positions[:, 2] * 1e3,
    c="gray",
    s=5,
    alpha=0.6,
)
ax.set_xlabel("X (µm)")
ax.set_ylabel("Y (µm)")
ax.set_zlabel("Z (µm)")
plt.tight_layout()
plt.show()

# Pre-allocate adjacency lists
adj = {ct: {ct2: [] for ct2 in cell_types} for ct in cell_types}

# compute index of neuron at center-right side
mid_y = b // 2
mid_z = b // 2
j_center = (nx - 1) * b * b + mid_y * b + mid_z

# =====================================
# Create connections (with periodic distances)
# =====================================
# iterate over target (postsynaptic) neurons
con_probs = {
    (pre_type, post_type): [] for pre_type in cell_types for post_type in cell_types
}
neighbor_idxs_list = []
for j in range(n_total):
    post_type = types[j]
    # query neighbors within 3 * max sigma (converted to mm)
    dmax = max_sigma_um[post_type] * 3 * 1e-3
    dmax = min(dmax, L.max() / 2)  # cap at half the cube size
    idxs = tree.query_ball_point(positions[j], r=dmax)
    neighbor_idxs = set(idx % n_total for idx in idxs)
    neighbor_idxs_list.append(len(neighbor_idxs))

    # visualize the neighborhood within the original cube
    # show all neurons as dots, highlight the target neuron and its neighbors
    if j == j_center:  # only visualize for the center-right neuron
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(projection="3d")
        ax.scatter(
            positions[:, 0] * 1e3,
            positions[:, 1] * 1e3,
            positions[:, 2] * 1e3,
            c="gray",
            s=5,
            alpha=0.6,
        )
        ax.scatter(
            positions[j, 0] * 1e3,
            positions[j, 1] * 1e3,
            positions[j, 2] * 1e3,
            c="orange",
            s=50,
        )
        for i in neighbor_idxs:
            if i == j:
                continue
            pre_type = types[i]
            ax.scatter(
                positions[i, 0] * 1e3,
                positions[i, 1] * 1e3,
                positions[i, 2] * 1e3,
                c=(
                    "blue"
                    if pre_type == "FS"
                    else "green" if pre_type == "dSPN" else "purple"
                ),
                s=20,
            )
        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")
        ax.set_zlabel("Z (µm)")
        plt.tight_layout()
        plt.show()

    # iterate over pre-synaptic neurons (neighbors)
    for i in neighbor_idxs:
        if i == j:
            continue
        # get pre-synaptic type
        pre_type = types[i]
        # skip if no parameters for this pre->post pair
        if (pre_type, post_type) not in conn_params:
            continue
        # otherwise get connection parameters
        P0, sigma_um = conn_params[(pre_type, post_type)]
        # convert sigma to mm
        sigma = sigma_um * 1e-3
        # periodic minimum-image distance in mm
        delta = positions[i] - positions[j]
        delta = delta - L * np.round(delta / L)
        dist = np.linalg.norm(delta)
        # half-gaussian connection probability based on distance in mm
        p = P0 * np.exp(-(dist**2) / (sigma**2))
        # store how likely this connection was, and whether it was made
        con_probs[(pre_type, post_type)].append((p, 0))
        if rng.random() < p:
            adj[pre_type][post_type].append((i, j))
            con_probs[(pre_type, post_type)][-1] = (p, 1)  # mark as connected


# %% =====================================
# Analyze connection probabilities
# plot the sorted neighbor_idxs_list
plt.figure()
plt.plot(sorted(neighbor_idxs_list))
plt.title("Number of neighbor candidates per neuron")
plt.xlabel("Neuron index (sorted)")
plt.ylabel("Number of neighbors within dmax")
plt.grid(True)
plt.tight_layout()
plt.show()

# plot con_probs, for each pre->post pair create two boxplots for connected and unconnected
plt.figure(figsize=(12, 6))
for (pre_type, post_type), probs in con_probs.items():
    if not probs:
        continue
    connected = [p for p, conn in probs if conn == 1]
    unconnected = [p for p, conn in probs if conn == 0]
    plt.boxplot(
        [connected, unconnected],
        positions=[0, 1],
        widths=0.4,
        labels=[
            f"{pre_type} -> {post_type} (connected - {len(connected)})",
            f"{pre_type} -> {post_type} (unconnected - {len(unconnected)})",
        ],
    )
    plt.title(f"Connection probabilities: {pre_type} -> {post_type}")
    plt.ylabel("Connection probability")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    fig_name = f"boxplot_{pre_type}_{post_type}.png"
    plt.savefig(os.path.join(output_dir, fig_name))
    plt.close()

print("\nConnections created:")
for pre_type in cell_types:
    for post_type in cell_types:
        if (pre_type, post_type) in conn_params:
            count = len(adj[pre_type][post_type])
            print(f"{pre_type} -> {post_type}: {count} connections")

# %%
# =====================================
# Analyze degree distributions
# =====================================
# For each pre->post, count inputs per post cell (in central margin)
hist_data = {}
for pre_type in cell_types:
    hist_data[pre_type] = {}
    for post_type in cell_types:
        # skip if no parameters for this pre->post pair
        if (pre_type, post_type) not in conn_params:
            continue
        # get pairs of indices (pre, post) from adjacency list
        pairs = adj[pre_type][post_type]
        # count inputs per post
        counts = np.zeros(n_total, dtype=int)
        for pre_idx, post_idx in pairs:
            counts[post_idx] += 1
        print(f"{pre_type} -> {post_type}: {np.sum(counts)} total inputs")
        # restrict to central neurons and only those of type 'post_type'
        margin_mm = margin * d
        dim_x_mm = nx * d
        dim_y_mm = b * d
        dim_z_mm = b * d
        mask_x = (X.ravel() >= margin_mm) & (X.ravel() < (dim_x_mm - margin_mm))
        mask_y = (Y.ravel() >= margin_mm) & (Y.ravel() < (dim_y_mm - margin_mm))
        mask_z = (Z.ravel() >= margin_mm) & (Z.ravel() < (dim_z_mm - margin_mm))
        central = mask_x & mask_y & mask_z & (types == post_type)
        print(
            f"{pre_type} -> {post_type}: {np.sum(central)} central neurons of type {post_type}"
        )
        hist_data[pre_type][post_type] = counts[central]
        # plot histogram
        plt.figure()
        plt.hist(hist_data[pre_type][post_type], bins=30)
        plt.title(f"{pre_type} -> {post_type} input counts")
        plt.xlabel("Number of inputs")
        plt.ylabel("Cell count")
        fig_name = f"hist_{pre_type}_{post_type}.png"
        plt.savefig(os.path.join(output_dir, fig_name))
        plt.close()


# %%
# =====================================
# 3D Visualization
# =====================================

# 1) 3D scatter of neurons colored by type (µm)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection="3d")
colors = {"FS": "red", "dSPN": "green", "iSPN": "blue"}
ax.scatter(
    positions[:, 0] * 1e3,
    positions[:, 1] * 1e3,
    positions[:, 2] * 1e3,
    c=[colors[t] for t in types],
    s=5,
    alpha=0.6,
)
ax.set_xlabel("X (µm)")
ax.set_ylabel("Y (µm)")
ax.set_zlabel("Z (µm)")
# legend
for ct, col in colors.items():
    ax.scatter([], [], [], c=col, label=ct)
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "3D_neurons.png"))
plt.close()

# 2) Half‐gaussian connection‐probability curves along x (µm)
plt.figure(figsize=(8, 4))
max_x = dim_x_um / 2
x_vals = np.linspace(0, max_x, 500)
for (pre, post), (P0, sigma_um) in conn_params.items():
    p_vals = P0 * np.exp(-(x_vals**2) / (sigma_um**2))
    plt.plot(x_vals, p_vals, label=f"{pre}→{post} (d={sigma_um:.0f}µm)")
# mark maximum periodic distance
plt.axvline(max_x, color="black", linestyle="--", label="Max periodic dist")
plt.xlabel("Distance (µm)")
plt.ylabel("Connection probability")
plt.title("Half-gaussian connection-probability curves")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "conn_prob_curves.png"))
plt.close()

# End of script

# %%
