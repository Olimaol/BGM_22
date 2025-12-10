# %%
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as sp
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.sparse import lil_matrix, load_npz, save_npz

# ANNarchy imports
from ANNarchy import (
    CurrentInjection,
    Population,
    Projection,
    TimedArray,
    simulate,
)

# Local imports
from external_input.spike_input_cortex import (
    build_distance_groups_state,
    _periodic_distance_float,
    plot_empirical_and_target_distance_dependent_shared_fraction,
    simulate_receiver_counts_distance_dependent_on_drive,
    iter_memmap_spike_counts,
)
from striatal_weights.get_weights import (
    components_spn_spn,
    components_fsi_spn,
    components_fsi_fsi,
)

# CompNeuroPy imports
from CompNeuroPy import CombinedSampler
from CompNeuroPy.neuron_models import (
    Izhikevich2007Humphries2009SPND1,
    Izhikevich2007Humphries2009SPND2,
    Izhikevich2007Humphries2009FSI,
)


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

    # ----------------------
    # Storage helpers
    # ----------------------
    def _connectivity_state_path(self) -> str:
        return os.path.join(self.connectivity_dir, "connectivity_state.pkl")

    def _weight_matrix_path(self, pre_type: str, post_type: str) -> str:
        return os.path.join(
            self.connectivity_dir, f"weights_{pre_type}_{post_type}.npz"
        )

    def _missing_input_state_path(self) -> str:
        return os.path.join(self.inputs_dir, "missing_input_state.pkl")

    def _dist_state_path(self, pre_type: str, post_type: str) -> str:
        return os.path.join(self.inputs_dir, f"dist_state_{pre_type}_{post_type}.pkl")

    def _spike_counts_path(self, pre_type: str, post_type: str) -> str:
        return os.path.join(
            self.inputs_dir,
            f"receiver_counts_distance_dependent_{pre_type}_{post_type}.dat",
        )

    def __init__(
        self,
        name: str = "caudate",
        nx: int = 10,
        b: int = 10,
        density: float = 84900.0,
        firing_rate_dict: dict | None = None,
        correlation_dict: dict | None = None,
        dt: float = 0.1,
        T: float = 1000.0,
        update_time: float = 100.0,
        seed: int = 42,
        props_delRey: np.ndarray | None = None,
        fitted_params_path: str | None = None,
        storage_dir: str | None = None,
        output_dir: str | None = None,
        build_connectivity: bool = True,
        build_missing_gaba_input: bool = True,
        verbose: bool = True,
    ) -> None:
        # --- Parameters ---
        self.update_time = update_time  # ms for how long inputs are defined
        self.name = name
        self.nx = nx
        self.b = b
        self.density = density
        self.seed = seed
        self.verbose = verbose

        # firing rates per cell type (Hz)
        # default for D1 and D2 extracted from: (Liang et al., 2008) using with levodopa treatment, see experimental_data/activity_striatum/extract_from_liang_etal_2008.py
        # default for FS: 10 Hz based on: (Yamada et al., 2016; Marche und Apicella, 2021; Adler et al., 2013; Hernandez et al., 2013; He et al., 2024)
        if firing_rate_dict is None:
            firing_rate_dict = {"FS": 10.0, "dSPN": 37.07, "iSPN": 29.07}
        self.firing_rate_dict = firing_rate_dict

        # average correlations between pairs of cell types
        if correlation_dict is None:
            # default based on (Adler et al., 2013):
            correlation_dict = {"FS": 0.06, "dSPN": 0.004, "iSPN": 0.004}
        self.correlation_dict = correlation_dict

        # timestep for simulation in ms
        self.dt = dt

        # total simulation time in ms and simulation steps
        self.T = T
        self.n_steps = int(T / dt)

        # proportions (del Rey et al. 2022)
        if props_delRey is None:
            props_delRey = np.array([0.026, 0.86 / 2, 0.86 / 2])
        props = props_delRey / np.sum(props_delRey)
        self.props = {"FS": props[0], "dSPN": props[1], "iSPN": props[2]}
        self.cell_types = list(self.props.keys())

        # paths
        script_dir = os.path.dirname(__file__)
        self.storage_dir = storage_dir or os.path.join(script_dir, ".microcircuit")
        os.makedirs(self.storage_dir, exist_ok=True)
        self.connectivity_dir = os.path.join(self.storage_dir, "connectivity")
        self.inputs_dir = os.path.join(self.storage_dir, "inputs")
        os.makedirs(self.connectivity_dir, exist_ok=True)
        os.makedirs(self.inputs_dir, exist_ok=True)
        if fitted_params_path is None:
            fitted_params_path = os.path.join(
                script_dir, "connectivity_fits", "fitted_params.json"
            )
        figures_subdir = output_dir if output_dir is not None else "figures"
        figures_subdir = os.path.basename(figures_subdir)
        self.output_dir = os.path.join(self.storage_dir, figures_subdir)
        os.makedirs(self.output_dir, exist_ok=True)

        # load connectivity parameters
        with open(fitted_params_path) as f:
            fitted = json.load(f)
        self.conn_params: dict[tuple[str, str], tuple[float, float]] = {
            tuple(key.split("-")): (val["amplitude"], val["sigma_um"])
            for key, val in fitted.items()
        }

        # --- Lattice and neuron types ---
        self.n_total = self.nx * self.b * self.b
        # spacing per cell (mm)
        self.d = (1.0 / self.density) ** (1 / 3)  # mm

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

        # store counts for delayed ANNarchy population creation
        self.type_counts = type_counts

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
        # Create weight matrices only for pairs present in connectivity params for later use in ANNarchy connection
        self.weights_by_type: dict[tuple[str, str], lil_matrix] = {}
        for pre_type, post_type in self.conn_params.keys():
            n_pre = len(self.indices_by_type[pre_type])
            n_post = len(self.indices_by_type[post_type])
            self.weights_by_type[(pre_type, post_type)] = lil_matrix((n_pre, n_post))

        # prepare the samplers from which the values of the weights are sampled
        self.weight_samplers: dict[tuple[str, str], CombinedSampler] = {}
        for pre_type, post_type in self.conn_params.keys():
            if pre_type in ("dSPN", "iSPN") and post_type in ("dSPN", "iSPN"):
                components = components_spn_spn
            elif pre_type == "FS" and post_type in ("dSPN", "iSPN"):
                components = components_fsi_spn
            elif pre_type == "FS" and post_type == "FS":
                components = components_fsi_fsi
            else:
                raise ValueError(
                    f"No weight components defined for pair {pre_type}->{post_type}"
                )
            sampler = CombinedSampler(components=components, rng=self.rng)
            self.weight_samplers[(pre_type, post_type)] = sampler

        # prepare the dictionaries to hold the input iterators and corresponding TimedArray populations for each pre-post type pair and the cortical inputs (from dlPFC, PM/SMA, M1) for later use in the update function
        self.inp_iterator_dict: dict[tuple[str, str], iter] = (
            {}
        )  # key: (pre_type, post_type)
        self.annarchy_inp_populations: dict[tuple[str, str], TimedArray] = (
            {}
        )  # key: (pre_type, post_type)

        # container for ANNarchy populations; created lazily in create_model
        self.annarchy_populations: dict[str, Population] = {}
        self.model_built: bool = False
        self.dist_state_dict = None

        # container to store distances (mm) for actual formed connections per pair
        self.connection_distances_by_pair: dict[tuple[str, str], list[float]] = {
            key: [] for key in self.conn_params.keys()
        }

        # get the radius for the simulated neighborhood per post type (mm) and
        # theoretical max sigma, required for _missing_local_input() and _build_connectivity()
        self.neighborhood_radii_mm, self.max_sigma_mm = (
            self._get_neighborhood_radii_mm()
        )

        # build or load connectivity and fill per-type weight matrices
        if build_connectivity:
            self._build_connectivity()
            self._save_connectivity_state()
        else:
            self._load_connectivity_state()

        if self.verbose:
            self.summary()

        # prepare or load missing local gaba inputs (spike counts) for all neurons
        if build_missing_gaba_input:
            self._missing_local_input()
            self._save_missing_input_state()
        else:
            self._load_missing_input_state()

        # TODO: define excitatory inputs (spike counts) for all neurons

    def update(self, run_simulation: bool = False) -> None:
        """Update function to be called during simulation to update the input populations."""
        if not self.model_built:
            raise RuntimeError(
                "create_model() must be called before update to build ANNarchy objects."
            )
        # Loop over all input iterators and update the corresponding TimedArray populations
        for key, inp_iterator in self.inp_iterator_dict.items():
            inp_population = self.annarchy_inp_populations[key]
            # get next chunk of inputs (incoming spike counts)
            inputs = next(inp_iterator)
            # reshape inputs from (n_neurons, n_steps) into (n_steps, n_neurons)
            inputs = inputs.T
            # if the key is for missing gaba input, scale the inputs by the mean weight for the pre-post type pair
            if key in self.mean_weights_by_type:
                inputs *= self.mean_weights_by_type[key]

            # update the TimedArray population with weighted inputs
            inp_population.reset()
            inp_population.update(rates=inputs)

        # Optional simulation the network for the update_time
        if run_simulation:
            simulate(self.update_time)

    def create_model(self) -> None:
        """Instantiate ANNarchy objects (populations, inputs/projections).

        Call this after constructing the Microcircuit to keep heavy ANNarchy
        objects separate from data preparation.
        """
        if self.model_built:
            if self.verbose:
                print("ANNarchy model already built; skipping create_model().")
            return

        # Create neuron populations
        self.create_populations_annarchy(type_counts=self.type_counts)

        # create projections between striatal populations
        self.create_local_projections_annarchy()

        # Ensure distance-dependent input state exists
        if self.dist_state_dict is None:
            self._missing_local_input()

        # Build ANNarchy TimedArray inputs and projections for local gaba inputs
        self._create_missing_gaba_inputs_annarchy(dist_state_dict=self.dist_state_dict)

        self.model_built = True

    def create_local_projections_annarchy(self) -> None:
        """Create ANNarchy Projections between the striatal populations based on the
        sampled connectivity and weights.
        """
        self.annarchy_projections: dict[tuple[str, str], Projection] = {}
        for (pre_type, post_type), weight_matrix in self.weights_by_type.items():
            pre_pop = self.annarchy_populations[pre_type]
            post_pop = self.annarchy_populations[post_type]

            # Create projection
            proj = Projection(
                pre=pre_pop,
                post=post_pop,
                target="gaba",
                name=f"Proj_{pre_type}_{post_type}_{self.name}",
            )
            proj.connect_from_sparse(weight_matrix)
            self.annarchy_projections[(pre_type, post_type)] = proj

    def create_populations_annarchy(self, type_counts: dict[str, int]) -> None:
        """
        Create ANNarchy populations for each cell type with the specified counts.

        Args:
            type_counts: Dictionary mapping cell type labels to their respective counts.
        """

        self.annarchy_populations = {}
        for cell_type, count in type_counts.items():

            if cell_type == "dSPN":
                neuron_model = Izhikevich2007Humphries2009SPND1
            elif cell_type == "iSPN":
                neuron_model = Izhikevich2007Humphries2009SPND2
            elif cell_type == "FS":
                neuron_model = Izhikevich2007Humphries2009FSI
            else:
                raise ValueError(f"No neuron model for cell type: {cell_type}")

            population = Population(
                geometry=count, neuron=neuron_model, name=f"{self.name}_{cell_type}"
            )
            self.annarchy_populations[cell_type] = population

    # ----------------------
    # Connectivity creation
    # ----------------------
    def _get_neighborhood_radii_mm(self):
        """
        Get the neighborhood radius (mm) for each post type based on max sigma. It can
        not exceed half the max dimension of the periodic cube.
        """

        # compute max sigma (um) per post type (for neighbor query radius)
        max_sigma_um = {
            post_type: max(
                sigma_um
                for (_, post_type2), (_, sigma_um) in self.conn_params.items()
                if post_type2 == post_type
            )
            for post_type in self.cell_types
        }
        # 3 times max sigma
        max_sigma_mm = {pt: max_sigma_um[pt] * 3 * 1e-3 for pt in self.cell_types}

        # compute neighborhood radii (mm) which cannot exceed half the box size
        radii_mm = {
            post_type: min(max_sigma_mm[post_type], self.L.max() / 2)
            for post_type in self.cell_types
        }
        return radii_mm, max_sigma_mm

    def _neighbors_within(self, j: int, r_mm: float) -> set:
        """Return indices of neurons within radius ``r_mm`` from neuron ``j`` under periodic boundaries."""
        idxs = self.tree.query_ball_point(self.positions[j], r=r_mm)
        return set(idx % self.n_total for idx in idxs)

    def _periodic_distance(self, i: int, j: int) -> float:
        """Compute true periodic Euclidean distance (mm) between neurons ``i`` and ``j``."""
        delta = self.positions[i] - self.positions[j]
        delta = delta - self.L * np.round(delta / self.L)
        return float(np.linalg.norm(delta))

    def _build_connectivity(self) -> None:
        """Instantiate probabilistic connections and sample weights for all permitted pre/post type pairs."""
        # loop over postsynaptic neurons (global indices)
        for post_global in range(self.n_total):
            post_type = self.types[post_global]
            neighbor_idxs = self._neighbors_within(
                post_global, r_mm=self.neighborhood_radii_mm[post_type]
            )
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
                p = self._p_exp(dist, P0, sigma)
                self.con_probs[key].append((p, 0))
                if self.rng.random() < p:
                    # store global adjacency
                    self.adj[pre_type][post_type].append((pre_global, post_global))
                    self.con_probs[key][-1] = (p, 1)
                    # translate to local indices per type and set weight
                    pre_local = self.local_index_map[pre_type][pre_global]
                    post_local = self.local_index_map[post_type][post_global]
                    self.weights_by_type[key][pre_local, post_local] = (
                        self.weight_samplers[key].sample()
                    )
                    # record distance (mm)
                    self.connection_distances_by_pair[key].append(dist)

    def _save_connectivity_state(self) -> None:
        """Persist connectivity-related data for reuse without rebuilding."""
        meta = {
            "neighbor_sizes": self.neighbor_sizes,
            "con_probs": {f"{k[0]}-{k[1]}": v for k, v in self.con_probs.items()},
            "connection_distances_by_pair": {
                f"{k[0]}-{k[1]}": v
                for k, v in self.connection_distances_by_pair.items()
            },
            "adj": {
                pre: {post: pairs for post, pairs in inner.items()}
                for pre, inner in self.adj.items()
            },
            "rng_state": self.rng.bit_generator.state,
            "types": self.types,
            "type_counts": self.type_counts,
            "cell_types": self.cell_types,
            "n_total": self.n_total,
            "nx": self.nx,
            "b": self.b,
            "density": self.density,
            "conn_params": {
                f"{pre}-{post}": vals for (pre, post), vals in self.conn_params.items()
            },
        }
        with open(self._connectivity_state_path(), "wb") as f:
            pickle.dump(meta, f)
        for (pre_type, post_type), W in self.weights_by_type.items():
            save_npz(self._weight_matrix_path(pre_type, post_type), W.tocsr())

    def _load_connectivity_state(self) -> None:
        """Load connectivity data from disk if available; raise if missing."""
        state_path = self._connectivity_state_path()
        if not os.path.exists(state_path):
            raise FileNotFoundError(
                f"Connectivity cache not found at {state_path}. Rebuild by setting build_connectivity=True."
            )
        with open(state_path, "rb") as f:
            meta = pickle.load(f)

        # basic compatibility checks
        if meta.get("n_total") != self.n_total or meta.get("nx") != self.nx:
            raise ValueError(
                "Cached connectivity was generated for a different lattice; rebuild connectivity."
            )
        if not np.array_equal(meta.get("types"), self.types):
            raise ValueError(
                "Cached connectivity uses a different type assignment; rebuild connectivity or reuse the same seed/params."
            )

        self.neighbor_sizes = meta.get("neighbor_sizes", [])
        self.con_probs = {
            tuple(k.split("-")): v for k, v in meta.get("con_probs", {}).items()
        }
        self.connection_distances_by_pair = {
            tuple(k.split("-")): v
            for k, v in meta.get("connection_distances_by_pair", {}).items()
        }
        self.adj = meta.get("adj", self.adj)
        # ensure keys exist for all defined pairs
        for key in self.conn_params.keys():
            self.con_probs.setdefault(key, [])
            self.connection_distances_by_pair.setdefault(key, [])
            pre_type, post_type = key
            self.adj.setdefault(pre_type, {})
            self.adj[pre_type].setdefault(post_type, [])
        # load weights
        self.weights_by_type = {}
        for key in self.conn_params.keys():
            pre_type, post_type = key
            weight_path = self._weight_matrix_path(pre_type, post_type)
            if not os.path.exists(weight_path):
                raise FileNotFoundError(
                    f"Missing cached weight matrix at {weight_path}; rebuild connectivity."
                )
            self.weights_by_type[key] = load_npz(weight_path).tolil()

        rng_state = meta.get("rng_state")
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state

    def _expected_outer(self, rho, Rin, Rout, p_func):
        """
        Expected number of inputs from the outer shell [Rin, Rout] for a single receiver.

        Math:
            E[N_outer] = rho * ∫_{shell} p(r) dV
                    = 4*pi * rho * ∫_{Rin}^{Rout} p(r) r^2 dr

        Parameters
        ----------
        rho : float
            Presynaptic neuron density (neurons per unit volume).
        Rin : float
            Inner radius of the outer shell (units consistent with distance).
        Rout : float
            Outer cutoff radius.
        p_func : callable
            p_func(r) returns connection probability at distance r.

        Returns
        -------
        float
            Expected number of connections originating in the outer shell.

        Practical notes
        ---------------
        - Evaluate the integral with scipy.integrate.quad (adaptive).
        - Ensure units are consistent.
        - If p_func returns 0 beyond some radius, that is fine; integration will account for it.

        Example
        -------
        >>> E_outer = expected_outer(1e5, 0.05, 0.8, lambda r: p_exp(r, 0.5, 0.2))
        """
        integrand = lambda r: p_func(r) * r**2
        val, err = integrate.quad(
            integrand, Rin, Rout, epsabs=1e-8, epsrel=1e-6, limit=200
        )
        return 4 * np.pi * rho * val

    def _expected_shared_for_d(self, rho, Rin, Rout, p_func, d):
        """
        Expected number of shared presynaptic inputs from the outer shells of two receivers separated by distance d.

        Math derivation (summary):
            E[N_shared(d)] = rho * ∫ p(r_A) p(r_B) dV
        Place receiver A at origin, receiver B on polar axis at distance d.
            E[N_shared(d)] = 2*pi * rho * ∫_{r=Rin}^{Rout} r^2 ∫_{theta=0}^{pi}
                            p(r) p(r_B) sin(theta) dtheta dr
        where r_B = sqrt(r^2 + d^2 - 2*r*d*cos(theta)).

        Parameters
        ----------
        rho : float
            Presynaptic density (neurons per unit volume).
        Rin, Rout : float
            Inner and outer radii defining the shell of interest.
        p_func : callable
            Connection kernel p(r).
        d : float
            Distance between the two receiving neurons (units consistent with radii).

        Returns
        -------
        float
            Expected number of shared presynaptic neurons that are in both outer shells and connect to both receivers.

        Edge cases & checks
        -------------------
        - If d >= 2*Rout there is no overlap of the outer shells -> returns 0.
        - If d == 0, this reduces to E[N_shared(0)] = 4*pi*rho * ∫_{Rin}^{Rout} p(r)^2 r^2 dr.

        Numerical considerations
        ------------------------
        - This is a nested integral (r then theta). Use quad for the inner theta integral and then quad for r.
        - For many d values, consider caching/interpolating results.
        """
        if d >= 2 * Rout:
            return 0.0

        def inner_theta(theta, r):
            # distance to receiver B
            rB = np.sqrt(max(0.0, r * r + d * d - 2 * r * d * np.cos(theta)))
            if (rB < Rin) or (rB > Rout):
                return 0.0
            return p_func(r) * p_func(rB) * (r**2) * np.sin(theta)

        def integrand_r(r):
            val_theta, _ = integrate.quad(
                lambda th: inner_theta(th, r),
                0.0,
                np.pi,
                epsabs=1e-6,
                epsrel=1e-5,
                limit=200,
            )
            return val_theta

        val_r, _ = integrate.quad(
            integrand_r, Rin, Rout, epsabs=1e-6, epsrel=1e-5, limit=200
        )
        return 2 * np.pi * rho * val_r

    def _p_exp(
        self, d: float | np.ndarray, P0: float, sigma: float
    ) -> float | np.ndarray:
        """
        Exponential connection probability function.
        """
        return P0 * np.exp(-(d**2) / (sigma**2))

    def _missing_local_input(self):
        """Construct distance-dependent shared input groups and simulate local inhibitory spike counts."""
        # Get distance dependent shared input curves f(d)
        (
            f_d_interp_dict,
            f_d_raw_dict,
            expected_outer_dict,
            expected_shared_dict,
        ) = self._define_distance_dependent_shared_input_curves()

        # Build input groups based on f(d)
        dist_state_dict = self._define_distance_dependent_shared_input_groups(
            expected_outer_dict=expected_outer_dict,
            f_d_interp_dict=f_d_interp_dict,
            expected_shared_dict=expected_shared_dict,
        )

        # Simulate spike counts for these groups and assign to receivers and store them
        self._simulate_distance_dependent_spike_counts(dist_state_dict=dist_state_dict)
        # Store state for later ANNarchy creation in create_model()
        self.dist_state_dict = dist_state_dict
        # create a dictionary which stores the mean of the weights per pre-post type pair
        self.mean_weights_by_type: dict[tuple[str, str], float] = {}
        for key in self.conn_params.keys():
            weight_samples = self.weight_samplers[key].sample(n=10000)
            self.mean_weights_by_type[key] = float(np.mean(weight_samples))

    def _create_missing_gaba_inputs_annarchy(self, dist_state_dict):
        """Create ANNarchy TimedArray input populations for distance-dependent spike
        counts and the corresponding input iterators for setting the inputs during simulation using stored data.
        """
        # Loop over postsynaptic neuron type
        for post_type in self.cell_types:
            # get the receiver population
            post_pop: Population = self.annarchy_populations[post_type]
            # loop over presynaptic neuron type
            for pre_type in self.cell_types:
                key = (pre_type, post_type)
                if key not in self.conn_params:
                    continue
                # create the input population and connect it to the receiver population
                # the input is initialized with placeholder zeros, this needs to be updated before simulation
                n_steps_input = int(self.update_time / self.dt)
                inp = TimedArray(
                    rates=np.zeros((n_steps_input, post_pop.size)),
                    name=f"TimedInput_{pre_type}_{post_type}_{self.name}",
                )
                proj = CurrentInjection(
                    inp,
                    post_pop,
                    "gaba",
                    name=f"CurrentInjection_{pre_type}_{post_type}_{self.name}",
                )
                proj.connect_current()

                # create the input iterator for the update function
                spike_file = self._spike_counts_path(pre_type, post_type)
                inp_iterator = iter_memmap_spike_counts(
                    state=dist_state_dict[key],
                    filename=spike_file,
                    num_bins=self.n_steps,
                    chunk_size=n_steps_input,
                    copy=False,
                    verbose=self.verbose,
                )
                self.annarchy_inp_populations[key] = inp
                self.inp_iterator_dict[key] = inp_iterator

    def _simulate_distance_dependent_spike_counts(self, dist_state_dict):
        """Generate spike-count time series for each distance-dependent group configuration."""
        # Loop over postsynaptic neuron type
        for post_type in self.cell_types:
            # loop over presynaptic neuron type
            for pre_type in self.cell_types:
                key = (pre_type, post_type)
                if key not in self.conn_params:
                    continue

                spike_file = self._spike_counts_path(pre_type, post_type)
                simulate_receiver_counts_distance_dependent_on_drive(
                    filename=spike_file,
                    state=dist_state_dict[key],
                    rate=self.firing_rate_dict[pre_type],
                    dt=self.dt,
                    rho=self.correlation_dict[pre_type],
                    num_bins=self.n_steps,
                    rng=self.rng,
                    verbose=self.verbose,
                )

    def _save_missing_input_state(self) -> None:
        """Persist distance-dependent input state to allow reloading without recomputation."""
        if self.dist_state_dict is None:
            return
        payload = {
            "dist_state_dict": self.dist_state_dict,
            "rng_state": self.rng.bit_generator.state,
            "cell_types": self.cell_types,
            "conn_keys": [f"{pre}-{post}" for (pre, post) in self.conn_params.keys()],
            "mean_weights_by_type": getattr(self, "mean_weights_by_type", None),
        }
        with open(self._missing_input_state_path(), "wb") as f:
            pickle.dump(payload, f)

    def _load_missing_input_state(self) -> None:
        """Load distance-dependent input state; expect spike-count files to exist."""
        state_path = self._missing_input_state_path()
        if not os.path.exists(state_path):
            raise FileNotFoundError(
                f"Missing cached missing-input state at {state_path}. Rebuild by setting build_missing_gaba_input=True."
            )
        with open(state_path, "rb") as f:
            payload = pickle.load(f)

        expected_keys = set(tuple(k.split("-")) for k in payload.get("conn_keys", []))
        if expected_keys and expected_keys != set(self.conn_params.keys()):
            raise ValueError(
                "Cached missing-input state does not match current connectivity parameters; rebuild missing inputs."
            )

        self.dist_state_dict = payload.get("dist_state_dict")
        if self.dist_state_dict is None:
            raise ValueError(
                "Cached missing-input state is empty; rebuild missing inputs."
            )

        self.mean_weights_by_type = payload.get("mean_weights_by_type")
        if self.mean_weights_by_type is None:
            raise ValueError(
                "Cached missing-input state is missing mean weights; rebuild missing inputs."
            )

        # ensure spike-count files exist for all required pairs
        for pre_type, post_type in self.conn_params.keys():
            path = self._spike_counts_path(pre_type, post_type)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Spike-count file for {pre_type}->{post_type} not found at {path}; rebuild missing inputs."
                )

        rng_state = payload.get("rng_state")
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state

    def _define_distance_dependent_shared_input_groups(
        self, expected_outer_dict, f_d_interp_dict, expected_shared_dict
    ):
        """Create shared-input groups that match target input counts and shared-fraction curves."""
        bounding_box_width = self.L[0]  # assuming cubic box
        # Loop over postsynaptic neuron type
        dist_state_dict = {}
        for post_type in self.cell_types:
            # loop over presynaptic neuron type
            for pre_type in self.cell_types:
                key = (pre_type, post_type)
                if key not in self.conn_params:
                    continue
                # get receiver positions of this post type
                receiver_positions = self.positions[self.types == post_type]
                # N_target: expeted inputs from outer shell per receiver neuron
                N_target = expected_outer_dict[key]
                # f_target: distance-dependent shared input fraction f(d)
                f_target = f_d_interp_dict[key]
                # the given f_d expects distances in mm, but creating the groups uses grid coordinates
                f_target_grid = lambda d_grid: f_target(d_grid * self.d)
                # s_group: group size, I use max(expected shared) / 2 and min 1
                s_group = max(int(expected_shared_dict[key][1].max() / 2), 1)
                # create groups and distribute them over receiver neurons
                if self.verbose:
                    print(
                        f"{pre_type}->{post_type} - Defining distance-dependent shared input groups for..."
                    )
                    print(f"Number of receivers: {len(receiver_positions)}")
                    print(f"minimum distance between receivers: {self.d:.3f}")
                    print(
                        f"maximum possible periodic distance within bounding box: {np.sqrt(3) * (bounding_box_width / 2):.3f}"
                    )
                    print(f"receiver positions (first 5): {receiver_positions[:5]}")
                    print(f"N_target: {N_target}")
                    print(f"s_group: {s_group}")
                    print("\n")

                dist_state = build_distance_groups_state(
                    receiver_positions=receiver_positions,
                    bounding_box_width=bounding_box_width,
                    N_target=N_target,
                    s=s_group,
                    f_target=f_target_grid,
                    rng=self.rng,
                    fine_grid_resolution=10,
                )
                dist_state_dict[key] = dist_state

                if self.verbose:
                    # loop over all receiver positions pairs and calculate their periodic distances
                    distance_matrix = np.zeros((dist_state.R, dist_state.R))
                    for i in range(dist_state.R):
                        for j in range(i + 1, dist_state.R):
                            if i == j:
                                continue
                            pos_i = dist_state.receiver_positions[i]
                            pos_j = dist_state.receiver_positions[j]
                            d = _periodic_distance_float(pos_i, pos_j, dist_state.L)
                            distance_matrix[i, j] = d
                            distance_matrix[j, i] = d
                    print(
                        f"dist_state.L: {dist_state.L} and bounding_box_width: {bounding_box_width}"
                    )
                    print(
                        f"minimum distance in distance matrix: {distance_matrix.min()}"
                    )
                    print(
                        f"maximum distance in distance matrix: {distance_matrix.max()}"
                    )

                    print(
                        f"Optimized p(d)=p0*exp(-d^2/(2*sigma^2)) parameters: p0={dist_state.p0:.4f}, sigma={dist_state.sigma:.3f}"
                    )
                    mean_inputs_empirical = np.mean(
                        [len(g) * s_group for g in dist_state.groups_by_receiver]
                    )
                    print(
                        f"Empirical mean distinct inputs per receiver (groups * s): {mean_inputs_empirical:.2f} (target {N_target})"
                    )
                    print(f"Total groups G: {dist_state.G}")

                    # Empirical shared fraction estimation (sampled pairs for efficiency).
                    print(
                        f"{pre_type}->{post_type} - Sampling receiver pairs to estimate empirical shared fraction curve..."
                    )
                    plot_empirical_and_target_distance_dependent_shared_fraction(
                        dist_state=dist_state,
                        rng=self.rng,
                        title=f"Empirical vs Target shared input fraction: {pre_type}->{post_type}",
                        save_dir=self.output_dir,
                        filename=f"SharedInputFraction_{pre_type}_{post_type}.png",
                    )

        return dist_state_dict

    def _define_distance_dependent_shared_input_curves(self):
        """Compute distance-dependent shared-input fraction curves f(d) for all valid type pairs."""

        # Get shared input fraction depending on distance f(d) considering the size of
        # the simulated volume and the distance-dependent connection probability
        # Number of inputs from outer shell:
        Rin_mm = (
            self.neighborhood_radii_mm
        )  # inner radius of outer shell (mm), i.e. simulated volume around receiver neuron
        Rout_mm = (
            self.max_sigma_mm
        )  # outer cutoff radius (mm), i.e. theoretical max sigma
        rho_pre = {
            pre_type: self.props[pre_type] * self.density
            for pre_type in self.cell_types
        }  # presynaptic density (neurons/mm^3)

        # variables to store the returns
        f_d_interp_dict = {}
        f_d_raw_dict = {}
        expected_outer_dict = {}
        expected_shared_dict = {}

        # Loop over postsynaptic type
        for post_type in self.cell_types:
            # loop over presynaptic type
            for pre_type in self.cell_types:
                key = (pre_type, post_type)
                if key not in self.conn_params:
                    continue
                P0, sigma_um = self.conn_params[key]
                sigma_mm = sigma_um * 1e-3  # mm

                # define p_func
                p_func = lambda r_mm: self._p_exp(r_mm, P0, sigma_mm)

                # compute expected number of inputs from outer shell per receiver neuron
                expected_outer = self._expected_outer(
                    rho=rho_pre[pre_type],
                    Rin=Rin_mm[post_type],
                    Rout=Rout_mm[post_type],
                    p_func=p_func,
                )
                expected_inner = self._expected_outer(
                    rho=rho_pre[pre_type],
                    Rin=0.0,
                    Rout=Rin_mm[post_type],
                    p_func=p_func,
                )
                if self.verbose:
                    print(f"Computed E_outer for {pre_type}->{post_type}...")
                    print(
                        f"  Rin={Rin_mm[post_type]:.3f} mm, Rout={Rout_mm[post_type]:.3f} mm"
                    )
                    print(f"  rho_pre={rho_pre[pre_type]:.2f} neurons/mm^3")
                    print(
                        f"  p_func at 0 mm = {p_func(0):.4f}, p_func at Rin = {p_func(Rin_mm[post_type]):.4f}"
                    )
                    print(
                        f"  E_outer = {expected_outer:.4f} inputs per receiver neuron"
                    )
                    print(
                        f"  E_inner = {expected_inner:.4f} inputs per receiver neuron (has to match with local inputs in simulated volume)"
                    )
                    print("\n")

                # compute expected shared inputs for distance d between two neurons
                # precalculate the expected shared inputs for some distances to later interpolate
                dmax = (
                    np.sqrt(3) * self.L.max() / 2
                )  # maximum possible distance between pair of neurons in periodic cube, i.e. half the space diagonal
                d_vals = np.linspace(0, dmax, 50)
                expected_shared_vals = np.array(
                    [
                        self._expected_shared_for_d(
                            rho_pre[pre_type],
                            Rin_mm[post_type],
                            Rout_mm[post_type],
                            p_func,
                            d,
                        )
                        for d in d_vals
                    ]
                )
                if self.verbose:
                    print(f"Computed E_shared_outer for {pre_type}->{post_type}...")
                    print(f"  For distances between pairs d in [0, {dmax:.3f}] mm")
                    print(
                        f"  Expected shared inputs at d=0 mm: {expected_shared_vals[0]:.4f}"
                    )
                    print(
                        f"  Expected shared inputs at d={dmax:.3f} mm: {expected_shared_vals[-1]:.4f}"
                    )
                    print("\n")

                # distance-dependent shared input fraction f(d)
                f_d = expected_shared_vals / max(expected_outer, 1e-12)

                # store f(d) as an interpolating function
                f_d_interp_dict[key] = interp1d(
                    d_vals, f_d, kind="cubic", fill_value="extrapolate"
                )

                # store the raw f_d values and expected outer and shared numbers for later use
                f_d_raw_dict[key] = (d_vals, f_d)
                expected_outer_dict[key] = expected_outer
                expected_shared_dict[key] = (d_vals, expected_shared_vals)

                # visualization of f(d) (optional)
                # plt.figure(figsize=(8, 6))
                # plt.subplot(211)
                # d_vals_plot = np.linspace(0, dmax, 200)
                # plt.plot(d_vals_plot, f_d_dict[key](d_vals_plot))
                # plt.plot(d_vals, f_d, "o")
                # plt.title(
                #     f"Shared input fraction f(d) for {pre_type}->{post_type} \n E_outer={expected_outer:.2f}"
                # )
                # plt.xlabel("Distance d (mm)")
                # plt.ylabel("Shared input fraction f(d)")
                # plt.subplot(212)
                # plt.plot(d_vals, expected_shared_vals)
                # plt.title(f"Expected shared inputs for {pre_type}->{post_type}")
                # plt.xlabel("Distance d (mm)")
                # plt.ylabel("Expected shared inputs")
                # plt.show()

        return (
            f_d_interp_dict,
            f_d_raw_dict,
            expected_outer_dict,
            expected_shared_dict,
        )

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
        neighbor_idxs = self._neighbors_within(
            j, r_mm=self.neighborhood_radii_mm[post_type]
        )

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
                mask = self.types == post_type
                hist_data[pre_type][post_type] = counts[mask]
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
            p_vals = self._p_exp(x_vals, P0, sigma_um)
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
            # print the connection type and the expected number of inputs for a single post neuron:
            print(f"{pre_type} -> {post_type}: expected inputs per post neuron:")
            n_inputs_per_post = W.sum(axis=0)
            print(f"  Mean: {n_inputs_per_post.mean():.2f}")
            print(f"  Std: {n_inputs_per_post.std():.2f}")
            print(f"  Min: {n_inputs_per_post.min():.2f}")
            print(f"  Max: {n_inputs_per_post.max():.2f}")
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
    mc = Microcircuit(nx=10, b=10, verbose=True)
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
