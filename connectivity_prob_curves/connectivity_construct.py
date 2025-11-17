# %%
import numpy as np
import scipy.spatial as sp
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
import os
import json


class Microcircuit:
    """
    Microcircuit class that constructs a 3D periodic microcircuit with distance-dependent
    connection probabilities and provides analysis/visualization methods.

    Key attributes after initialization:
    - weights_by_type: dict[(pre_type, post_type)] -> sparse lil_matrix with shape
        (n_pre_type_cells, n_post_type_cells); indices local to each population.
    - adj: dict[pre_type][post_type] -> list of (i_pre, j_post) index pairs
    - con_probs: dict[(pre_type, post_type)] -> list of tuples (p, connected_flag)
    - positions: (n_total, 3) array of neuron positions in mm
    - types: (n_total,) array of neuron type strings
    - cell_types: list of type labels
    - output_dir: path where plots are saved
    """

    def __init__(
        self,
        nx: int = 10,
        b: int = 10,
        density: float = 84900.0,
        seed: int = 42,
        props_delRey: np.ndarray | None = None,
        fitted_params_path: str | None = None,
        output_dir: str | None = None,
        verbose: bool = True,
    ) -> None:
        # --- Parameters ---
        self.nx = nx
        self.b = b
        self.density = density
        self.seed = seed
        self.verbose = verbose

        # proportions (del Rey et al. 2022)
        if props_delRey is None:
            props_delRey = np.array([0.026, 0.86 / 2, 0.86 / 2])
        props = props_delRey / np.sum(props_delRey)
        self.props = {"FS": props[0], "dSPN": props[1], "iSPN": props[2]}
        self.cell_types = list(self.props.keys())

        # paths
        script_dir = os.path.dirname(__file__)
        if fitted_params_path is None:
            fitted_params_path = os.path.join(
                script_dir, "connectivity_fits", "fitted_params.json"
            )
        if output_dir is None:
            output_dir = os.path.join(script_dir, "connectivity_construct")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # load connectivity parameters
        with open(fitted_params_path) as f:
            fitted = json.load(f)
        self.conn_params: dict[tuple[str, str], tuple[float, float]] = {
            tuple(key.split("-")): (val["amplitude"], val["sigma_um"])
            for key, val in fitted.items()
        }

        # compute max sigma (um) per post type (for neighbor query radius)
        self.max_sigma_um = {
            post_type: max(
                sigma_um
                for (_, post_type2), (_, sigma_um) in self.conn_params.items()
                if post_type2 == post_type
            )
            for post_type in self.cell_types
        }

        # --- Lattice and neuron types ---
        self.n_total = self.nx * self.b * self.b
        # spacing per cell (mm)
        self.d = (1.0 / self.density) ** (1 / 3)  # mm
        # margin for central analysis (index units) – unused due to periodic boundaries
        global_max_sigma_um = max(
            sigma_um for (_, _), (_, sigma_um) in self.conn_params.items()
        )
        self.margin = int(np.ceil(3 * global_max_sigma_um * 1e-3 / self.d))
        self.margin = 0

        # positions (mm)
        xs = np.arange(self.nx) * self.d
        ys = np.arange(self.b) * self.d
        zs = np.arange(self.b) * self.d
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        self.X, self.Y, self.Z = X, Y, Z
        self.positions = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T

        # RNG
        self.rng = np.random.default_rng(self.seed)

        # assign neuron types
        type_counts = {ct: int(self.props[ct] * self.n_total) for ct in self.cell_types}
        rem = self.n_total - sum(type_counts.values())
        for ct in sorted(self.cell_types, key=lambda x: self.props[x], reverse=True)[
            :rem
        ]:
            type_counts[ct] += 1
        types = np.array(
            [ct for ct, count in type_counts.items() for _ in range(count)]
        )
        self.rng.shuffle(types)
        self.types = types

        # derived geometry
        self.d_um = self.d * 1e3
        self.dim_x_um = self.nx * self.d_um
        self.dim_y_um = self.b * self.d_um
        self.dim_z_um = self.b * self.d_um
        self.volume_mm3 = self.n_total / self.density

        # periodic KDTree (original cube + 26 neighbors)
        self.L = np.array([self.nx * self.d, self.b * self.d, self.b * self.d])
        shifts = np.array(
            [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
        )
        self.ext_positions = np.vstack(
            [self.positions + shift * self.L for shift in shifts]
        )
        self.tree = sp.cKDTree(self.ext_positions)

        # adjacency storage and default center index (center-right as before)
        self.adj = {ct: {ct2: [] for ct2 in self.cell_types} for ct in self.cell_types}
        mid_y = self.b // 2
        mid_z = self.b // 2
        self.j_center = (self.nx - 1) * self.b * self.b + mid_y * self.b + mid_z

        # containers for analysis
        self.con_probs = {
            (pre_type, post_type): []
            for pre_type in self.cell_types
            for post_type in self.cell_types
        }
        self.neighbor_sizes: list[int] = []

        # Prepare per-type index mappings (global -> local) and initialize weight matrices
        self.indices_by_type: dict[str, np.ndarray] = {
            ct: np.flatnonzero(self.types == ct) for ct in self.cell_types
        }
        self.local_index_map: dict[str, dict[int, int]] = {
            ct: {g_idx: l_idx for l_idx, g_idx in enumerate(self.indices_by_type[ct])}
            for ct in self.cell_types
        }
        # Create weight matrices only for pairs present in connectivity params
        self.weights_by_type: dict[tuple[str, str], lil_matrix] = {}
        for pre_type, post_type in self.conn_params.keys():
            n_pre = len(self.indices_by_type[pre_type])
            n_post = len(self.indices_by_type[post_type])
            self.weights_by_type[(pre_type, post_type)] = lil_matrix((n_pre, n_post))

        # container to store distances (mm) for actual formed connections per pair
        self.connection_distances_by_pair: dict[tuple[str, str], list[float]] = {
            key: [] for key in self.conn_params.keys()
        }

        # build connectivity and fill per-type weight matrices
        self._build_connectivity()

        if self.verbose:
            self.summary()

    # ----------------------
    # Connectivity creation
    # ----------------------
    def _neighbors_within(self, j: int, r_mm: float) -> set:
        idxs = self.tree.query_ball_point(self.positions[j], r=r_mm)
        return set(idx % self.n_total for idx in idxs)

    def _periodic_distance(self, i: int, j: int) -> float:
        delta = self.positions[i] - self.positions[j]
        delta = delta - self.L * np.round(delta / self.L)
        return float(np.linalg.norm(delta))

    def _build_connectivity(self) -> None:
        # loop over postsynaptic neurons (global indices)
        for post_global in range(self.n_total):
            post_type = self.types[post_global]
            dmax = self.max_sigma_um[post_type] * 3 * 1e-3
            dmax = min(dmax, self.L.max() / 2)
            neighbor_idxs = self._neighbors_within(post_global, dmax)
            self.neighbor_sizes.append(len(neighbor_idxs))

            # loop over presynaptic candidates (global indices)
            for pre_global in neighbor_idxs:
                if pre_global == post_global:
                    continue
                pre_type = self.types[pre_global]
                key = (pre_type, post_type)
                if key not in self.conn_params:
                    continue
                P0, sigma_um = self.conn_params[key]
                sigma = sigma_um * 1e-3
                dist = self._periodic_distance(pre_global, post_global)
                p = P0 * np.exp(-(dist**2) / (sigma**2))
                self.con_probs[key].append((p, 0))
                if self.rng.random() < p:
                    # store global adjacency
                    self.adj[pre_type][post_type].append((pre_global, post_global))
                    self.con_probs[key][-1] = (p, 1)
                    # translate to local indices per type and set weight
                    pre_local = self.local_index_map[pre_type][pre_global]
                    post_local = self.local_index_map[post_type][post_global]
                    self.weights_by_type[key][pre_local, post_local] = 1.0
                    # record distance (mm)
                    self.connection_distances_by_pair[key].append(dist)

    # ----------------------
    # Reporting & summaries
    # ----------------------
    def summary(self) -> None:
        counts = {ct: int(np.sum(self.types == ct)) for ct in self.cell_types}
        print(
            f"Cuboid dimensions (μm): X={self.dim_x_um:.2f}, Y={self.dim_y_um:.2f}, Z={self.dim_z_um:.2f}"
        )
        print(f"Volume: {self.volume_mm3:.3f} mm³")
        for ct, cnt in counts.items():
            print(f"Neurons ({ct}): {cnt}")
        print(f"Grid spacing: {self.d_um:.2f} μm")

    def print_connections_created(self) -> None:
        print("\nConnections created:")
        for pre_type in self.cell_types:
            for post_type in self.cell_types:
                if (pre_type, post_type) in self.conn_params:
                    count = len(self.adj[pre_type][post_type])
                    print(f"{pre_type} -> {post_type}: {count} connections")
        # Also report matrix shapes
        print("\nPer-type weight matrix shapes:")
        for (pre_type, post_type), W in self.weights_by_type.items():
            print(f"W[{pre_type}->{post_type}] shape = {W.shape}")

    def get_weight_matrix(self, pre_type: str, post_type: str):
        """Return the sparse weight matrix for a pre->post type pair."""
        key = (pre_type, post_type)
        if key not in self.weights_by_type:
            raise KeyError(f"No weight matrix for pair {pre_type}->{post_type}")
        return self.weights_by_type[key]

    # ----------------------
    # Plots & visualizations
    # ----------------------
    def plot_ext_kdtree_points(self, show: bool = False) -> None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(projection="3d")
        ax.scatter(
            self.ext_positions[:, 0] * 1e3,
            self.ext_positions[:, 1] * 1e3,
            self.ext_positions[:, 2] * 1e3,
            c="gray",
            s=5,
            alpha=0.6,
        )
        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")
        ax.set_zlabel("Z (µm)")
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(os.path.join(self.output_dir, "kdtree_tiled_points.png"))
            plt.close()

    def plot_neighborhood(self, j: int | None = None, show: bool = False) -> None:
        if j is None:
            j = self.j_center
        post_type = self.types[j]
        dmax = self.max_sigma_um[post_type] * 3 * 1e-3
        dmax = min(dmax, self.L.max() / 2)
        neighbor_idxs = self._neighbors_within(j, dmax)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(projection="3d")
        ax.scatter(
            self.positions[:, 0] * 1e3,
            self.positions[:, 1] * 1e3,
            self.positions[:, 2] * 1e3,
            c="gray",
            s=5,
            alpha=0.6,
        )
        ax.scatter(
            self.positions[j, 0] * 1e3,
            self.positions[j, 1] * 1e3,
            self.positions[j, 2] * 1e3,
            c="orange",
            s=50,
        )
        for i in neighbor_idxs:
            if i == j:
                continue
            pre_type = self.types[i]
            color = (
                "blue"
                if pre_type == "FS"
                else ("green" if pre_type == "dSPN" else "purple")
            )
            ax.scatter(
                self.positions[i, 0] * 1e3,
                self.positions[i, 1] * 1e3,
                self.positions[i, 2] * 1e3,
                c=color,
                s=20,
            )
        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")
        ax.set_zlabel("Z (µm)")
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(os.path.join(self.output_dir, f"neighborhood_j{j}.png"))
            plt.close()

    def plot_neighbor_candidate_counts(self, show: bool = False) -> None:
        plt.figure()
        plt.plot(sorted(self.neighbor_sizes))
        plt.title("Number of neighbor candidates per neuron")
        plt.xlabel("Neuron index (sorted)")
        plt.ylabel("Number of neighbors within dmax")
        plt.grid(True)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(os.path.join(self.output_dir, "neighbor_candidate_counts.png"))
            plt.close()

    def plot_connection_probability_boxplots(self, show: bool = False) -> None:
        for (pre_type, post_type), probs in self.con_probs.items():
            if not probs:
                continue
            connected = [p for p, conn in probs if conn == 1]
            unconnected = [p for p, conn in probs if conn == 0]
            plt.figure(figsize=(8, 4))
            plt.boxplot(
                [connected, unconnected],
                positions=[0, 1],
                widths=0.4,
                tick_labels=[
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
            if show:
                plt.show()
            else:
                plt.savefig(os.path.join(self.output_dir, fig_name))
                plt.close()

    def analyze_degree_distributions(self, show: bool = False) -> dict:
        hist_data: dict[str, dict[str, np.ndarray]] = {}
        for pre_type in self.cell_types:
            hist_data[pre_type] = {}
            for post_type in self.cell_types:
                if (pre_type, post_type) not in self.conn_params:
                    continue
                pairs = self.adj[pre_type][post_type]
                counts = np.zeros(self.n_total, dtype=int)
                for pre_idx, post_idx in pairs:
                    counts[post_idx] += 1
                print(f"{pre_type} -> {post_type}: {int(np.sum(counts))} total inputs")
                margin_mm = self.margin * self.d
                dim_x_mm = self.nx * self.d
                dim_y_mm = self.b * self.d
                dim_z_mm = self.b * self.d
                mask_x = (self.X.ravel() >= margin_mm) & (
                    self.X.ravel() < (dim_x_mm - margin_mm)
                )
                mask_y = (self.Y.ravel() >= margin_mm) & (
                    self.Y.ravel() < (dim_y_mm - margin_mm)
                )
                mask_z = (self.Z.ravel() >= margin_mm) & (
                    self.Z.ravel() < (dim_z_mm - margin_mm)
                )
                central = mask_x & mask_y & mask_z & (self.types == post_type)
                print(
                    f"{pre_type} -> {post_type}: {int(np.sum(central))} central neurons of type {post_type}"
                )
                hist_data[pre_type][post_type] = counts[central]
                plt.figure()
                plt.hist(hist_data[pre_type][post_type], bins=30)
                plt.title(f"{pre_type} -> {post_type} input counts")
                plt.xlabel("Number of inputs")
                plt.ylabel("Cell count")
                fig_name = f"hist_{pre_type}_{post_type}.png"
                if show:
                    plt.show()
                else:
                    plt.savefig(os.path.join(self.output_dir, fig_name))
                    plt.close()
        return hist_data

    def plot_3d_neurons(self, show: bool = False) -> None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(projection="3d")
        colors = {"FS": "red", "dSPN": "green", "iSPN": "blue"}
        ax.scatter(
            self.positions[:, 0] * 1e3,
            self.positions[:, 1] * 1e3,
            self.positions[:, 2] * 1e3,
            c=[colors[t] for t in self.types],
            s=5,
            alpha=0.6,
        )
        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")
        ax.set_zlabel("Z (µm)")
        for ct, col in colors.items():
            ax.scatter([], [], [], c=col, label=ct)
        ax.legend(loc="upper right")
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(os.path.join(self.output_dir, "3D_neurons.png"))
            plt.close()

    def plot_half_gaussian_curves(self, show: bool = False) -> None:
        plt.figure(figsize=(8, 4))
        max_x = self.dim_x_um / 2
        x_vals = np.linspace(0, max_x, 500)
        for (pre, post), (P0, sigma_um) in self.conn_params.items():
            p_vals = P0 * np.exp(-(x_vals**2) / (sigma_um**2))
            plt.plot(x_vals, p_vals, label=f"{pre}→{post} (d={sigma_um:.0f}µm)")
        plt.axvline(max_x, color="black", linestyle="--", label="Max periodic dist")
        plt.xlabel("Distance (µm)")
        plt.ylabel("Connection probability")
        plt.title("Half-gaussian connection-probability curves")
        plt.legend()
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(os.path.join(self.output_dir, "conn_prob_curves.png"))
            plt.close()

    # ----------------------
    # Connectivity matrix visualizations
    # ----------------------
    def plot_weight_matrix(
        self,
        pre_type: str,
        post_type: str,
        show: bool = False,
        method: str = "spy",
        figsize: tuple[int, int] = (5, 5),
        markersize: float = 1.0,
    ) -> None:
        """Visualize the sparse connectivity matrix for a given pre->post pair.

        Parameters
        ----------
        pre_type, post_type : str
            Neuron type labels for pre and post populations.
        show : bool
            If True display interactively, else save to file.
        method : str
            "spy" (default) uses plt.spy; "density" renders a low-res density image.
        figsize : (int, int)
            Figure size in inches.
        markersize : float
            Marker size for plt.spy.
        """
        key = (pre_type, post_type)
        if key not in self.weights_by_type:
            raise KeyError(f"No weight matrix for {pre_type}->{post_type}")
        W = self.weights_by_type[key]
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        if method == "spy":
            ax.spy(W, markersize=markersize)
        elif method == "density":
            # downsample for visualization to avoid huge dense conversion
            W_csr = W.tocsr()
            # Create a coarse grid
            n_pre, n_post = W.shape
            scale = max(1, int(max(n_pre, n_post) / 500))
            # aggregate blocks
            dense = np.zeros((n_pre // scale + 1, n_post // scale + 1), dtype=float)
            rows, cols = W_csr.nonzero()
            for r, c in zip(rows, cols):
                dense[r // scale, c // scale] += 1
            ax.imshow(dense, origin="lower", aspect="auto", cmap="viridis")
        else:
            raise ValueError("method must be 'spy' or 'density'")
        nnz = W.nnz
        density_val = (
            nnz / (W.shape[0] * W.shape[1]) if W.shape[0] and W.shape[1] else 0
        )
        ax.set_title(
            f"Connectivity {pre_type}→{post_type}\nshape={W.shape} nnz={nnz} dens={density_val:.3e}"
        )
        ax.set_xlabel(f"post ({post_type}) index")
        ax.set_ylabel(f"pre ({pre_type}) index")
        # make the axes square if different pre and post sizes
        pre_size, post_size = W.shape
        aspect = post_size / pre_size
        ax.set_aspect(aspect=aspect)
        plt.tight_layout()
        fname = f"weights_{pre_type}_{post_type}.png"
        if show:
            plt.show()
        else:
            plt.savefig(os.path.join(self.output_dir, fname))
            plt.close()

    def plot_all_weight_matrices(
        self,
        show: bool = False,
        method: str = "spy",
        max_cols: int = 3,
        markersize: float = 1.0,
        figsize_per: float = 3.5,
    ) -> None:
        """Plot all existing weight matrices in a grid.

        Parameters
        ----------
        show : bool
            Display interactively instead of saving.
        method : str
            'spy' or 'density' (see plot_weight_matrix).
        max_cols : int
            Maximum number of subplot columns.
        markersize : float
            Marker size passed to spy.
        figsize_per : float
            Base size per subplot (width & height scale).
        """
        keys = list(self.weights_by_type.keys())
        if not keys:
            print("No weight matrices to plot.")
            return
        n = len(keys)
        cols = min(max_cols, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(
            rows, cols, figsize=(cols * figsize_per, rows * figsize_per)
        )
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.reshape(rows, cols)
        for idx, key in enumerate(keys):
            r = idx // cols
            c = idx % cols
            ax = axes[r, c]
            pre_type, post_type = key
            W = self.weights_by_type[key]
            if method == "spy":
                ax.spy(W, markersize=markersize)
            elif method == "density":
                W_csr = W.tocsr()
                n_pre, n_post = W.shape
                scale = max(1, int(max(n_pre, n_post) / 300))
                dense = np.zeros((n_pre // scale + 1, n_post // scale + 1), dtype=float)
                rows_n, cols_n = W_csr.nonzero()
                for rr, cc in zip(rows_n, cols_n):
                    dense[rr // scale, cc // scale] += 1
                ax.imshow(dense, origin="lower", aspect="auto", cmap="viridis")
            else:
                raise ValueError("method must be 'spy' or 'density'")
            nnz = W.nnz
            dens = nnz / (W.shape[0] * W.shape[1]) if W.shape[0] and W.shape[1] else 0
            ax.set_title(
                f"{pre_type}→{post_type}\n{W.shape} nnz={nnz} d={dens:.2e}", fontsize=8
            )
            ax.set_xlabel("post")
            ax.set_ylabel("pre")
        # hide unused axes
        for extra in range(n, rows * cols):
            r = extra // cols
            c = extra % cols
            axes[r, c].axis("off")
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(os.path.join(self.output_dir, "weights_all.png"), dpi=150)
            plt.close()

    # ----------------------
    # Distance distribution visualization
    # ----------------------
    def plot_connection_counts_vs_distance(
        self,
        bins: int | np.ndarray = 50,
        per_pair: bool = True,
        micrometers: bool = True,
        show: bool = False,
        density: bool = False,
        cumulative: bool = False,
    ) -> dict[tuple[str, str], dict[str, np.ndarray]]:
        """Plot connection counts versus true periodic Euclidean distance.

        Parameters
        ----------
        bins : int or array-like
            Number of bins or explicit bin edges (in mm if micrometers=False, else µm).
        per_pair : bool
            If True, create one subplot per (pre_type, post_type) pair; else aggregate all distances.
        micrometers : bool
            Convert distances to µm for plotting.
        show : bool
            Display instead of saving.
        density : bool
            If True, normalize histogram to form a probability density.
        cumulative : bool
            If True, plot cumulative counts/density.

        Returns
        -------
        dict mapping (pre_type, post_type) to {'bin_edges','counts','centers'} arrays (aggregated key 'ALL' if per_pair=False).
        """
        # Prepare data
        scale = 1e3 if micrometers else 1.0
        label_unit = "µm" if micrometers else "mm"

        def _hist(dist_list):
            arr = np.asarray(dist_list) * scale
            if isinstance(bins, int):
                counts, edges = np.histogram(arr, bins=bins, density=density)
            else:
                counts, edges = np.histogram(arr, bins=bins, density=density)
            if cumulative:
                counts = np.cumsum(counts)
            centers = 0.5 * (edges[:-1] + edges[1:])
            return counts, edges, centers

        results: dict[tuple[str, str], dict[str, np.ndarray]] = {}

        if per_pair:
            keys = [
                k
                for k in self.conn_params.keys()
                if self.connection_distances_by_pair[k]
            ]
            if not keys:
                print("No connections to plot distance distribution.")
                return {}
            n = len(keys)
            cols = min(3, n)
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])
            axes = axes.reshape(rows, cols)
            for idx, key in enumerate(keys):
                r = idx // cols
                c = idx % cols
                ax = axes[r, c]
                dist_list = self.connection_distances_by_pair[key]
                counts, edges, centers = _hist(dist_list)
                ax.plot(centers, counts, drawstyle="steps-mid")
                pre_type, post_type = key
                ax.set_title(f"{pre_type}→{post_type} (n={len(dist_list)})", fontsize=9)
                ax.set_xlabel(f"Distance ({label_unit})")
                ax.set_ylabel(
                    "Cumulative" if cumulative else ("Density" if density else "Count")
                )
                results[key] = {
                    "bin_edges": edges,
                    "counts": counts,
                    "centers": centers,
                }
            # hide unused axes
            for extra in range(n, rows * cols):
                axes[extra // cols, extra % cols].axis("off")
            plt.tight_layout()
            fname = "connection_distance_per_pair.png"
        else:
            # aggregate all distances
            all_dists = [
                d for lst in self.connection_distances_by_pair.values() for d in lst
            ]
            if not all_dists:
                print("No connections to plot distance distribution.")
                return {}
            counts, edges, centers = _hist(all_dists)
            plt.figure(figsize=(6, 4))
            plt.plot(centers, counts, drawstyle="steps-mid")
            plt.xlabel(f"Distance ({label_unit})")
            plt.ylabel(
                "Cumulative" if cumulative else ("Density" if density else "Count")
            )
            plt.title("Connection counts vs distance (ALL pairs)")
            plt.tight_layout()
            results[("ALL", "ALL")] = {
                "bin_edges": edges,
                "counts": counts,
                "centers": centers,
            }
            fname = "connection_distance_all.png"

        if show:
            plt.show()
        else:
            plt.savefig(os.path.join(self.output_dir, fname), dpi=150)
            plt.close()
        return results


if __name__ == "__main__":
    # Example usage: build microcircuit and reproduce main analyses, saving to output_dir
    mc = Microcircuit(verbose=True)
    mc.plot_ext_kdtree_points(show=False)
    mc.plot_neighborhood(show=False)
    mc.plot_neighbor_candidate_counts(show=False)
    mc.plot_connection_probability_boxplots(show=False)
    mc.print_connections_created()
    mc.analyze_degree_distributions(show=False)
    mc.plot_3d_neurons(show=False)
    mc.plot_half_gaussian_curves(show=False)
    mc.plot_all_weight_matrices(show=False)
    mc.plot_weight_matrix("dSPN", "dSPN", show=False)
    mc.plot_weight_matrix("FS", "iSPN", show=False)
    mc.plot_connection_counts_vs_distance(
        per_pair=False, micrometers=True, density=True
    )
    mc.plot_connection_counts_vs_distance(
        per_pair=True, micrometers=True, cumulative=False
    )
