"""Spike-count simulation with controllable input correlations and sharing.

This module provides utilities to simulate spike-count inputs across multiple
groups or receivers with a Beta-Binomial construction. For each time bin, a
probability is drawn from a Beta distribution (when ``0 < rho < 1``), and each
group's count is drawn from a Binomial distribution with that probability. This
yields marginal Beta-Binomial behavior and induces positive correlation across
groups within a time bin, controlled by ``rho``. Furthermore, groups are distributed
across receivers, allowing for flexible configurations of input sharing.

TODO: Use this to simulate input spikes for each striatal neuron and store the generated
spike trains as files to be able to load them later.
FSIs should get input from same pool (with same within correlations/caused fluctuations)
based on ([Damodaran et al., 2014](zotero://select/library/items/NDTWW3G7)).
Current idea for FSIS: check which MSNs are connected with the FSI, then obtain input
groups of these MSNs, then randomly select from these groups input groups for the FSI.

The script also contains a demonstration section (under ``if __name__ ==
"__main__":``) that showcases:

- Direct group count simulation for a chosen ``rho``.
- Building groups and inspecting their assignment to receivers.
- Receiver-level counts with overlap (``k > 1``) and the no-overlap fast path
    (``k == 1``).
New (distance-dependent) functionality
-------------------------------------
The module now also supports constructing input sharing using a distance-
dependent shared fraction target ``f(d)`` for a 3D periodic (toroidal) grid
of receivers. The user provides:

* ``receiver_positions``: integer grid coordinates of receivers with periodic
    boundary conditions (wrap-around along each axis).
* ``f_target(d)``: a callable giving the desired expected shared fraction of
    inputs between two receivers separated by distance ``d``.
* ``N``: desired approximate number of distinct inputs per receiver.
* ``s``: group size (inputs per group).

We model the probability that a group positioned at location ``x`` connects
to a receiver at location ``r`` via a Gaussian kernel

``p(d) = p0 * exp(-d^2 / (2 * sigma^2))``

with parameters ``(p0, sigma)`` chosen so that the resulting theoretical
expected shared fraction curve matches ``f_target(d)`` in a least-squares
sense. Under spatial homogeneity and uniform random placement of groups on
grid points, the expected shared fraction for receivers separated by distance
``d`` is:

``E[f(d)] = E[p(dist(r_i, g)) * p(dist(r_j, g))] / E[p(dist(r_i, g))]``.

Because ``p(d)`` is linear in ``p0`` and the numerator quadratic, this reduces
to ``E[f(d)] = p0 * h_sigma(d)`` for a precomputable kernel-dependent function
``h_sigma(d)`` when ``sigma`` is fixed. We therefore: (1) scan candidate
``sigma`` values, (2) obtain the optimal ``p0`` in closed form, and (3) select
``(p0, sigma)`` minimizing the squared error to the supplied ``f_target``.

Given ``(p0, sigma)``, we determine required total distinct inputs ``M = G * s``
by equating the target per-receiver inputs ``N`` with the theoretical
expectation ``E[I] = M * E[p(dist(r, g))]``. We then build ``G`` groups at
random grid locations and sample each receiver-group connection independently
with probability ``p(distance)``. A distance-based simulation state is stored
in ``DistanceGroupsState``.

The ``__main__`` demonstration constructs a 10x10x10 grid, fits ``p(d)``,
verifies the empirical shared fraction against the target, and visualizes
input count distributions across receivers.
"""

import numpy as np
import math
from typing import Tuple, Optional, Callable, Dict, List
from dataclasses import dataclass

# Optional plotting; demonstration will guard imports.
try:  # pragma: no cover - demo convenience
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None


def beta_params_from_p_rho(p: float, rho: float) -> Tuple[float, float]:
    """Compute Beta parameters for a target mean and overdispersion.

    Given a desired mean probability ``p`` and overdispersion/co-variability
    coefficient ``rho`` in the open interval (0, 1), returns parameters
    ``(alpha, beta)`` such that:

    - E[P] = p
    - Var[P] = rho * p * (1 - p)

    Args:
        p (float): Target mean success probability in [0, 1]. Typically
            ``p = rate * dt``.
        rho (float): Overdispersion coefficient in (0, 1). Controls the
            variance of the per-bin probability P. Use the calling code to
            handle the boundary cases ``rho == 0`` and ``rho == 1``.

    Returns:
        Tuple[float, float]: ``(alpha, beta)`` for the Beta(alpha, beta)
        distribution.

    Raises:
        ValueError: If ``rho <= 0`` or ``rho >= 1``. These edge cases should be
            handled by the caller (e.g., treat ``rho == 0`` as independent
            Binomial and ``rho == 1`` as fully shared all-or-none firing).
    """
    if rho <= 0.0:
        raise ValueError(
            "rho must be > 0 to compute Beta parameters (handle rho==0 separately)."
        )
    if rho >= 1.0:
        raise ValueError("rho must be < 1 for Beta params (handle rho==1 separately).")
    s = 1.0 / rho - 1.0
    alpha = p * s
    beta = (1.0 - p) * s
    return alpha, beta


def simulate_counts_direct(
    G: int,
    N: int,
    rate: float,
    dt: float,
    rho: float,
    num_bins: int,
    rng: np.random.Generator,
    dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """Simulate group spike counts via a Beta-Binomial construction.

    For each time bin b, draw ``p_b ~ Beta(alpha, beta)`` with mean
    ``p = rate * dt`` and variance ``Var[P] = rho * p * (1 - p)`` when
    ``0 < rho < 1``. Then, for each group g, draw the count
    ``K[g, b] ~ Binomial(N, p_b)``. This induces positive correlation across
    groups within a time bin that is controlled by ``rho``.

    Special cases handled explicitly:
    - ``rho <= 0``: independent Binomial draws with fixed ``p`` (no shared variability).
    - ``rho >= 1``: fully shared Bernoulli per bin (all-or-none across all neurons),
      broadcast to all groups and summed to ``N``.

    Args:
        G (int): Number of groups (e.g., cortical populations).
        N (int): Number of neurons per group.
        rate (float): Firing rate in Hz.
        dt (float): Bin width in seconds.
        rho (float): Overdispersion/correlation coefficient. Values in
            [0, 1] are meaningful; values outside this range are treated by
            the nearest special-case branch described above.
        num_bins (int): Number of time bins to simulate.
        rng (np.random.Generator): Random number generator to use.
        dtype (Optional[np.dtype]): Desired dtype of returned array; if
            provided, counts are cast without copying when possible.

    Returns:
        np.ndarray: Array of shape ``(G, num_bins)`` containing integer
        spike counts.

    Examples:
        >>> rng = np.random.default_rng(0)
        >>> simulate_counts_direct(2, 10, 50, 0.001, 0.3, 5, rng).shape
        (2, 5)
    """
    p = rate * dt
    # No overdispersion: draw Binomial counts with a fixed probability p.
    if rho <= 0.0:
        arr = rng.binomial(N, p, size=(G, num_bins))
        if dtype is not None and arr.dtype != dtype:
            arr = arr.astype(dtype, copy=False)
        return arr
    # Fully shared case: a single Bernoulli per bin, broadcast to groups and
    # multiplied by N (all-or-none across neurons).
    if rho >= 1.0:
        all_fire = rng.binomial(1, p, size=num_bins)
        arr = N * np.broadcast_to(all_fire, (G, num_bins))
        if dtype is not None and arr.dtype != dtype:
            arr = arr.astype(dtype, copy=False)
        return arr
    # Intermediate regime: Beta-Binomial via random per-bin probabilities.
    alpha, beta = beta_params_from_p_rho(p, rho)
    p_bins = rng.beta(alpha, beta, size=num_bins)
    counts = rng.binomial(N, p_bins, size=(G, num_bins))
    if dtype is not None and counts.dtype != dtype:
        counts = counts.astype(dtype, copy=False)
    return counts


def compute_k_from_f(R, f):
    """Compute receivers-per-group from a shared fraction.

    Maps a desired shared-input fraction to the number of receivers each
    input group should project to.

    Args:
        R (int): Total number of receivers.
        f (float): Shared-input fraction in [0, 1]. ``f=0`` yields ``k≈1``
            and ``f=1`` yields ``k≈R``.

    Returns:
        tuple[int, float]: ``(k, k_float)`` where ``k`` is the discretized
        receivers-per-group (at least 1) and ``k_float`` is the unrounded
        value before discretization.
    """
    k_float = 1.0 + f * (R - 1)
    # Use at least 1 receiver per group (no forced overlap when f≈0).
    k = max(1, int(round(k_float)))
    return k, k_float


def build_groups_homogeneous(R, N, f, s, rng, build_reverse: bool = True):
    """Build input groups and assign each group to ``k`` receivers.

    Constructs ``G`` input groups of size ``s`` and assigns every group to the
    same number of receivers ``k``, where ``k`` is computed from ``R`` and
    the desired shared-input fraction ``f``.

    Args:
        R (int): Number of receivers.
        N (int): Approximate number of inputs per receiver.
        f (float): Target shared-input fraction in [0, 1].
        s (int): Group size (number of inputs per group).
        rng (np.random.Generator): Random number generator.
        build_reverse (bool): If True, also build the reverse mapping from
            receiver to groups that connect to it.

    Returns:
        tuple[np.ndarray, int, float, int, Optional[list[np.ndarray]]]:
        ``(groups, k, k_float, G, groups_by_receiver)`` where ``groups`` has
        shape ``(G, k)`` and ``groups_by_receiver`` is a list of length ``R``
        (or ``None`` if ``build_reverse`` is False).
    """
    k, k_float = compute_k_from_f(R, f)
    # total inputs = N * R, each group outputs to k receivers -> required inputs = M = N*R/k
    # each group holds s inputs -> total groups = G = M/s
    M = N * R / k  # Total inputs.
    G = max(1, int(round(M / s)))
    # Build groups: G rows, each row has k random distinct receiver indices.
    receivers = np.arange(R, dtype=np.int32)
    groups = np.empty((G, k), dtype=np.int32)
    for g in range(G):
        groups[g] = rng.choice(receivers, size=k, replace=False)
    if build_reverse:
        # Build reverse mapping: for each receiver, which groups connect to it.
        lists = [[] for _ in range(R)]
        for g in range(G):
            for r in groups[g]:
                lists[r].append(g)
        groups_by_receiver = [np.array(ixs, dtype=np.int32) for ixs in lists]
    else:
        groups_by_receiver = None
    return groups, k, k_float, G, groups_by_receiver


def _smallest_unsigned_dtype(max_value: int) -> np.dtype:
    """Return the smallest unsigned integer dtype able to hold ``max_value``.

    Chooses among ``uint8``, ``uint16``, ``uint32``, and ``uint64``.

    Args:
        max_value (int): Maximum non-negative value to represent.

    Returns:
        np.dtype: Unsigned integer dtype that can represent ``max_value``.
    """
    if max_value <= np.iinfo(np.uint8).max:
        return np.uint8
    if max_value <= np.iinfo(np.uint16).max:
        return np.uint16
    if max_value <= np.iinfo(np.uint32).max:
        return np.uint32
    return np.uint64


@dataclass(frozen=True)
class GroupsState:
    """Reusable state describing wiring and dtypes for simulation.

    Attributes:
        fast_path (bool): If True, there is no overlap (``k == 1``); ``groups``
            is ``None`` and simulation uses direct per-receiver binomial draws.
        R (int): Number of receivers.
        N (int): Approximate number of inputs per receiver.
        s (int): Group size (inputs per group).
        k (int): Receivers per group.
        G (int): Number of groups (``R`` in fast path).
        groups (Optional[np.ndarray]): Array of shape ``(G, k)`` with receiver
            indices, or ``None`` if ``fast_path``.
        group_dtype (np.dtype): Unsigned dtype for per-group counts (based on ``s``).
        receiver_dtype (np.dtype): Unsigned dtype for per-receiver counts
            (based on ``s * max_degree`` in general case, or ``N`` in fast path).
    """

    fast_path: bool
    R: int
    N: int
    s: int
    k: int
    G: int
    groups: Optional[np.ndarray]
    group_dtype: np.dtype
    receiver_dtype: np.dtype


def build_groups_state(
    R: int, N: int, f: float, s: int, rng: np.random.Generator
) -> GroupsState:
    """Build a reusable groups state for repeated simulations.

    Constructs the groups (or marks the fast path if ``k == 1``) and chooses
    minimal unsigned integer dtypes for group and receiver counts. Reuse the
    returned state across multiple calls to simulate time bins in chunks.

    Args:
        R (int): Number of receivers.
        N (int): Approximate number of inputs per receiver.
        f (float): Target shared-input fraction in [0, 1].
        s (int): Group size (number of inputs per group).
        rng (np.random.Generator): Random number generator.

    Returns:
        GroupsState: Encapsulates topology, groups, and dtypes.
    """
    group_dtype = _smallest_unsigned_dtype(int(s))
    k, _k_float = compute_k_from_f(R, f)
    if k == 1:
        # Fast path: per-receiver direct simulation, no groups needed.
        receiver_dtype = _smallest_unsigned_dtype(int(N))
        return GroupsState(
            fast_path=True,
            R=R,
            N=N,
            s=s,
            k=k,
            G=R,
            groups=None,
            group_dtype=group_dtype,
            receiver_dtype=receiver_dtype,
        )

    # General case: build overlapping groups once and compute receiver dtype.
    groups, k, _, G, _ = build_groups_homogeneous(R, N, f, s, rng, build_reverse=False)
    degrees = np.bincount(groups.ravel(), minlength=R)
    max_degree = int(degrees.max()) if degrees.size else 0
    receiver_cap = int(s) * max_degree
    receiver_dtype = _smallest_unsigned_dtype(receiver_cap)
    return GroupsState(
        fast_path=False,
        R=R,
        N=N,
        s=s,
        k=k,
        G=G,
        groups=groups,
        group_dtype=group_dtype,
        receiver_dtype=receiver_dtype,
    )


# ---------------------------------------------------------------------------
# Distance-dependent shared fraction construction (periodic 3D grid)
# ---------------------------------------------------------------------------


def _periodic_component(delta: int, L: int) -> int:
    """Minimal wrapped distance along one axis for periodic boundaries."""
    delta = abs(delta)
    return min(delta, L - delta)


def periodic_distance(a: np.ndarray, b: np.ndarray, Lx: int, Ly: int, Lz: int) -> float:
    """Compute Euclidean distance on a 3D torus between integer grid points.

    Args:
        a, b: Arrays of length 3 with integer coordinates.
        Lx, Ly, Lz: Domain lengths along x, y, z.

    Returns:
        float: Wrapped Euclidean distance.
    """
    dx = _periodic_component(int(a[0]) - int(b[0]), Lx)
    dy = _periodic_component(int(a[1]) - int(b[1]), Ly)
    dz = _periodic_component(int(a[2]) - int(b[2]), Lz)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


@dataclass(frozen=True)
class DistanceGroupsState:
    """State for distance-dependent group sharing on a periodic 3D grid.

    Attributes:
        receiver_positions (np.ndarray): Shape (R, 3) integer coordinates.
        L (Tuple[int,int,int]): Domain lengths along each axis.
        R (int): Number of receivers.
        N_target (int): Desired expected number of distinct inputs per receiver.
        s (int): Group size (inputs per group).
        G (int): Number of groups.
        group_positions (np.ndarray): Shape (G, 3) chosen grid positions.
        p0 (float): Amplitude of Gaussian connection probability.
        sigma (float): Width of Gaussian connection probability.
        groups_by_receiver (List[np.ndarray]): For each receiver, array of group indices.
        group_dtype (np.dtype): Unsigned dtype for per-group counts.
        receiver_dtype (np.dtype): Unsigned dtype for per-receiver counts.
        f_target_samples (Dict[float, float]): Target shared fraction samples vs distance.
        f_model_samples (Dict[float, float]): Model shared fraction samples vs distance (theoretical).
    """

    receiver_positions: np.ndarray
    L: Tuple[int, int, int]
    R: int
    N_target: int
    s: int
    G: int
    group_positions: np.ndarray
    p0: float
    sigma: float
    groups_by_receiver: List[np.ndarray]
    group_dtype: np.dtype
    receiver_dtype: np.dtype
    f_target_samples: Dict[float, float]
    f_model_samples: Dict[float, float]


def _compute_distance_matrix_receivers_to_grid(
    receiver_positions: np.ndarray,
) -> Tuple[np.ndarray, Tuple[int, int, int], np.ndarray]:
    """Precompute distances from each receiver to every grid point (periodic).

    Returns:
        distances (np.ndarray): Shape (R, Ggrid) of wrapped distances.
        L (tuple): Domain lengths.
        grid_points (np.ndarray): Shape (Ggrid, 3) array of grid coordinates.
    """
    R = receiver_positions.shape[0]
    mins = receiver_positions.min(axis=0)
    shifted = receiver_positions - mins  # Ensure domain starts at 0.
    Lx = int(shifted[:, 0].max() + 1)
    Ly = int(shifted[:, 1].max() + 1)
    Lz = int(shifted[:, 2].max() + 1)
    grid_points = np.array(
        [(x, y, z) for x in range(Lx) for y in range(Ly) for z in range(Lz)],
        dtype=np.int16,
    )
    Ggrid = grid_points.shape[0]
    distances = np.empty((R, Ggrid), dtype=np.float32)
    for i in range(R):
        a = shifted[i]
        for j in range(Ggrid):
            b = grid_points[j]
            dx = _periodic_component(int(a[0]) - int(b[0]), Lx)
            dy = _periodic_component(int(a[1]) - int(b[1]), Ly)
            dz = _periodic_component(int(a[2]) - int(b[2]), Lz)
            distances[i, j] = math.sqrt(dx * dx + dy * dy + dz * dz)
    return distances, (Lx, Ly, Lz), grid_points


def _fit_gaussian_p(
    receiver_positions: np.ndarray,
    f_target: Callable[[float], float],
    sigma_candidates: np.ndarray,
    center_index: int = 0,
) -> Tuple[
    float, float, Dict[float, float], Dict[float, float], float, Dict[float, float]
]:
    """Fit Gaussian p(d)=p0*exp(-d^2/(2*sigma^2)) to target shared fraction curve.

    Exploits spatial homogeneity by fixing one receiver (``center_index``) and
    comparing it to all others. For a given ``sigma`` the theoretical curve is
    ``f_model(d) = p0 * h_sigma(d)`` where the optimal ``p0`` (least squares) is:

    ``p0 = sum_d f_target(d) * h_sigma(d) / sum_d h_sigma(d)^2``.

    Args:
        receiver_positions: (R,3) grid coordinates.
        f_target: Callable returning target shared fraction for distance ``d``.
        sigma_candidates: 1D array of sigma values to scan.
        center_index: Reference receiver index.

    Returns:
        p0, sigma, f_target_samples, f_model_samples, mean_g (for chosen sigma), h_sigma_map.
    """
    distances_matrix, L, grid_points = _compute_distance_matrix_receivers_to_grid(
        receiver_positions
    )
    R = receiver_positions.shape[0]
    # Distances between center receiver and all receivers.
    mins = receiver_positions.min(axis=0)
    shifted = receiver_positions - mins
    Lx, Ly, Lz = L
    center_pos = shifted[center_index]
    dist_center_to_receivers = np.array(
        [periodic_distance(center_pos, shifted[i], Lx, Ly, Lz) for i in range(R)],
        dtype=np.float32,
    )
    # Unique distance bins.
    unique_dists, unique_dists_inv = np.unique(
        dist_center_to_receivers, return_inverse=True
    )
    # Precompute distance arrays from center and each receiver to all grid points.
    dist_center_to_grid = distances_matrix[center_index]
    results = []
    for sigma in sigma_candidates:
        if sigma <= 0:
            continue
        g_center = np.exp(
            -(dist_center_to_grid**2) / (2.0 * sigma * sigma)
        )  # shape (Ggrid,)
        mean_g = float(g_center.mean())
        # For each receiver compute mean of g_center * g_receiver over grid points.
        g_all = np.exp(
            -(distances_matrix**2) / (2.0 * sigma * sigma)
        )  # shape (R, Ggrid)
        mean_g_gshift = (g_center * g_all).mean(axis=1)  # shape (R,)
        # Aggregate h_sigma(d) = mean_g_gshift / mean_g.
        h_vals = (
            np.array(
                [
                    mean_g_gshift[unique_dists_inv == i].mean()
                    for i in range(len(unique_dists))
                ]
            )
            / mean_g
        )
        h_sigma_map = {float(d): float(v) for d, v in zip(unique_dists, h_vals)}
        # Build arrays for least squares over actual occurring distances (weight by multiplicity).
        f_vals = []
        weights = []
        for d in unique_dists:
            f_t = float(f_target(float(d)))
            count = int((dist_center_to_receivers == d).sum())
            f_vals.append(f_t)
            weights.append(count)
        f_vals = np.array(f_vals)
        weights = np.array(weights)
        # This numerator/denominator gives optimal p0 in weighted least squares.
        # Using this p0 we minimize sum_d weights[d] * (f_vals[d] - p0 * h_vals[d])^2. with the current h_vals
        numerator = float(np.sum(f_vals * h_vals * weights))
        denominator = float(np.sum((h_vals**2) * weights))
        if denominator <= 0:
            continue
        p0 = numerator / denominator
        # Enforce p0 so that p(d)=p0*exp(-d^2/(2 sigma^2)) <= 1 for all d (d>=0 gives max at d=0).
        p0 = min(p0, 1.0)
        # Model curve samples.
        f_model_samples = {d: p0 * h_sigma_map[d] for d in h_sigma_map}
        # Loss (weighted MSE).
        mse = float(np.sum(((f_vals - p0 * h_vals) ** 2) * weights) / weights.sum())
        f_target_samples = {float(d): float(f_target(float(d))) for d in h_sigma_map}
        results.append(
            (mse, p0, sigma, f_target_samples, f_model_samples, mean_g, h_sigma_map)
        )
    if not results:
        raise RuntimeError("No valid sigma candidates for Gaussian fit.")
    results.sort(key=lambda x: x[0])
    mse, p0, sigma, f_target_samples, f_model_samples, mean_g, h_sigma_map = results[0]
    return p0, sigma, f_target_samples, f_model_samples, mean_g, h_sigma_map


def _periodic_distance_float(a: np.ndarray, b: np.ndarray, L: float) -> float:
    """Periodic wrapped Euclidean distance between two 3D points inside a cube of side L.

    Works for float coordinates. Assumes 0 <= coord < L along every axis.
    """
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    dz = abs(a[2] - b[2])
    dx = min(dx, L - dx)
    dy = min(dy, L - dy)
    dz = min(dz, L - dz)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _perdiodic_distance_float_parallel(a: np.ndarray, b: np.ndarray, L: float) -> float:
    """Periodic wrapped Euclidean distance between 3D points inside a cube of side L.

    Args:
        a (np.ndarray): Array of shape (n, 3) with float coordinates.
        b (np.ndarray): Array of shape (n, 3) with float coordinates.
        L (float): Side length of the periodic cube.

    Returns:
        np.ndarray: Array of shape (n,) with wrapped Euclidean distances.
    """
    deltas = np.abs(a - b)
    deltas = np.minimum(deltas, L - deltas)
    distances = np.sqrt(np.sum(deltas**2, axis=1))
    return distances


def _fit_gaussian_p_with_fine_grid(
    receiver_positions: np.ndarray,
    bounding_box_width: float,
    f_target: Callable[[float], float],
    sigma_candidates: np.ndarray,
    center_index: int = 0,
    fine_res: int = 100,
) -> Tuple[float, float, Dict[float, float], Dict[float, float], float]:
    """Fit p(d)=p0*exp(-d^2/(2*sigma^2)) using a fine uniform grid (fine_res^3).

    Memory-efficient chunked implementation to avoid storing full (R x fine_res^3) distance matrix.
    """
    print("Fitting Gaussian p(d) with fine grid...")
    R = receiver_positions.shape[0]
    # Ensure receiver positions lie inside [0, L).
    L = float(bounding_box_width)
    # Build fine grid (cell centers) as float coordinates.
    step = L / fine_res
    half = step / 2.0
    # Generate all grid points.
    grid_coords_1d = np.linspace(half, L - half, fine_res, dtype=np.float32)
    grid_points = np.array(
        [
            (x, y, z)
            for x in grid_coords_1d
            for y in grid_coords_1d
            for z in grid_coords_1d
        ],
        dtype=np.float32,
    )
    Ggrid = grid_points.shape[0]
    center = receiver_positions[center_index].astype(np.float32)
    print(f"Total fine grid points: {Ggrid}")
    # Distances from center receiver to all other receivers (for binning).
    dist_center_to_receivers = _perdiodic_distance_float_parallel(
        np.repeat(center[np.newaxis, :], R, axis=0),
        receiver_positions.astype(np.float32),
        L,
    )
    unique_dists, unique_dists_inv = np.unique(
        dist_center_to_receivers, return_inverse=True
    )

    # Precompute target f(d) for unique distances and multiplicity weights.
    f_vals = np.array(
        [float(f_target(float(d))) for d in unique_dists], dtype=np.float32
    )
    weights = np.bincount(unique_dists_inv).astype(np.float32)
    f_target_samples = {float(d): float(f_target(float(d))) for d in unique_dists}

    # Precompute distances matrix (receivers -> fine grid points) and center distances to grid.
    distances_matrix = np.empty((R, Ggrid), dtype=np.float32)
    rec_pos = receiver_positions.astype(np.float32, copy=False)
    for i in range(R):
        a = np.broadcast_to(rec_pos[i], (grid_points.shape[0], 3))
        d = _perdiodic_distance_float_parallel(a, grid_points, L)
        distances_matrix[i] = d.astype(np.float32, copy=False)
    dist_center_to_grid = distances_matrix[center_index]

    print(f"receiver_positions shape: {receiver_positions.shape}")
    print(f"distances_matrix shape: {distances_matrix.shape}")

    # Objective function: given sigma, compute weighted MSE and optimal p0.
    def evaluate_sigma(sigma: float):
        if sigma <= 0:
            return float("inf"), 0.0, 0.0, {}
        g_center = np.exp(
            -(dist_center_to_grid**2) / (2.0 * sigma * sigma)
        )  # shape (Ggrid,)
        mean_g = float(g_center.mean())  # scalar
        if mean_g <= 0:
            return float("inf"), 0.0, 0.0, {}
        g_all = np.exp(
            -(distances_matrix**2) / (2.0 * sigma * sigma)
        )  # shape (R, Ggrid)
        mean_g_gshift = (g_center * g_all).mean(axis=1)  # shape (R,)
        # Aggregate h_sigma(d) by distance bins.
        # Compute mean per unique distance using unique_dists_inv.
        # For efficiency, accumulate sums by bins then divide by counts.
        sums = np.bincount(
            unique_dists_inv, weights=mean_g_gshift, minlength=len(unique_dists)
        )
        h_vals = (sums / weights) / mean_g  # shape (num_unique_dists,)
        # Weighted least squares for p0.
        numerator = float(np.sum(f_vals * h_vals * weights))
        denominator = float(np.sum((h_vals**2) * weights))
        if denominator <= 0:
            return float("inf"), 0.0, mean_g, {}
        p0 = min(numerator / denominator, 1.0)
        # Model samples at the unique distances.
        f_model = p0 * h_vals
        mse = float(np.sum(((f_vals - f_model) ** 2) * weights) / np.sum(weights))
        f_model_samples = {float(d): float(m) for d, m in zip(unique_dists, f_model)}
        return mse, p0, mean_g, f_model_samples

    # Build initial search bracket for sigma.
    max_dist = math.sqrt(3.0) * L / 2.0
    sigma_lo = float(max_dist * 0.02)
    sigma_hi = float(max_dist)
    # Start from geometric mean as a reasonable initial point.
    sigma_mid = math.sqrt(sigma_lo * sigma_hi)
    expand_factor = 1.6

    # Evaluate at three points and ensure we have a proper bracket: f(mid) <= f(lo), f(mid) <= f(hi).
    mse_mid, p0_mid, mean_g_mid, fmodel_mid = evaluate_sigma(sigma_mid)
    mse_lo, p0_lo, mean_g_lo, fmodel_lo = evaluate_sigma(sigma_lo)
    mse_hi, p0_hi, mean_g_hi, fmodel_hi = evaluate_sigma(sigma_hi)

    # Expand lower side if needed.
    iter_guard = 0
    while mse_lo < mse_mid and iter_guard < 20:
        sigma_hi, mse_hi, p0_hi, mean_g_hi, fmodel_hi = (
            sigma_mid,
            mse_mid,
            p0_mid,
            mean_g_mid,
            fmodel_mid,
        )
        sigma_mid, mse_mid, p0_mid, mean_g_mid, fmodel_mid = (
            sigma_lo,
            mse_lo,
            p0_lo,
            mean_g_lo,
            fmodel_lo,
        )
        sigma_lo = max(sigma_mid / expand_factor, 1e-8)
        mse_lo, p0_lo, mean_g_lo, fmodel_lo = evaluate_sigma(sigma_lo)
        iter_guard += 1

    # Expand upper side if needed.
    iter_guard = 0
    while mse_hi < mse_mid and iter_guard < 20:
        sigma_lo, mse_lo, p0_lo, mean_g_lo, fmodel_lo = (
            sigma_mid,
            mse_mid,
            p0_mid,
            mean_g_mid,
            fmodel_mid,
        )
        sigma_mid, mse_mid, p0_mid, mean_g_mid, fmodel_mid = (
            sigma_hi,
            mse_hi,
            p0_hi,
            mean_g_hi,
            fmodel_hi,
        )
        sigma_hi = min(sigma_mid * expand_factor, max_dist * 5.0)
        mse_hi, p0_hi, mean_g_hi, fmodel_hi = evaluate_sigma(sigma_hi)
        iter_guard += 1

    # At this point we expect a bracket [sigma_lo, sigma_hi] with a minimum near sigma_mid.
    a, b = sigma_lo, sigma_hi
    gr = (math.sqrt(5) + 1.0) / 2.0
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    cache: Dict[float, Tuple[float, float, float, Dict[float, float]]] = {}

    def cached_eval(s: float):
        if s not in cache:
            cache[s] = evaluate_sigma(s)
        return cache[s]

    mse_c, p0_c, mean_g_c, fmodel_c = cached_eval(c)
    mse_d, p0_d, mean_g_d, fmodel_d = cached_eval(d)

    tol_rel = 1e-3
    max_iter = 32
    it = 0
    while (b - a) > tol_rel * (abs(a) + abs(b)) * 0.5 and it < max_iter:
        if mse_c < mse_d:
            b, mse_d = d, mse_c
            d, p0_d, mean_g_d, fmodel_d = c, p0_c, mean_g_c, fmodel_c
            c = b - (b - a) / gr
            mse_c, p0_c, mean_g_c, fmodel_c = cached_eval(c)
        else:
            a, mse_c = c, mse_d
            c, p0_c, mean_g_c, fmodel_c = d, p0_d, mean_g_d, fmodel_d
            d = a + (b - a) / gr
            mse_d, p0_d, mean_g_d, fmodel_d = cached_eval(d)
        it += 1

    # Choose best between c and d.
    if mse_c < mse_d:
        sigma_best, p0_best, mean_g_best, fmodel_best = c, p0_c, mean_g_c, fmodel_c
    else:
        sigma_best, p0_best, mean_g_best, fmodel_best = d, p0_d, mean_g_d, fmodel_d

    print(f"Optimized sigma: {sigma_best:.5f}, p0: {p0_best:.5f}")
    return p0_best, float(sigma_best), f_target_samples, fmodel_best, float(mean_g_best)


def build_distance_groups_state(
    receiver_positions: np.ndarray,
    bounding_box_width: float,
    N_target: int,
    s: int,
    f_target: Callable[[float], float],
    rng: np.random.Generator,
    sigma_candidates: Optional[np.ndarray] = None,
    center_index: int = 0,
    fine_grid_resolution: int = 100,
) -> DistanceGroupsState:
    """Construct distance-dependent group sharing state with explicit bounding box.

    The cube side length (``bounding_box_width``) defines periodic boundaries.
    Receiver coordinates must lie inside ``[0, bounding_box_width)`` along each axis.
    A fine grid (``fine_grid_resolution``^3, default 100^3) is used to fit the
    Gaussian connection probability ``p(d)=p0*exp(-d^2/(2*sigma^2))``.

    After fitting, the required number of groups ``G`` is computed. Group positions
    are then placed on a uniform coarse grid with side ``n_side = ceil(G^(1/3))``.
    The coarse grid cell centers (``n_side^3`` positions) are sampled *without*
    replacement to assign exactly ``G`` distinct group locations. This guarantees
    ``G <= n_side^3``.
    """
    if sigma_candidates is None:
        # Sigma heuristic based on box diagonal.
        diag = math.sqrt(3.0) * bounding_box_width
        sigma_candidates = np.linspace(diag * 0.02, diag * 0.5, 30, dtype=np.float32)
    p0, sigma, f_target_samples, f_model_samples, mean_g = (
        _fit_gaussian_p_with_fine_grid(
            receiver_positions=receiver_positions,
            bounding_box_width=bounding_box_width,
            f_target=f_target,
            sigma_candidates=sigma_candidates,
            center_index=center_index,
            fine_res=fine_grid_resolution,
        )
    )
    mean_p = p0 * mean_g
    if mean_p <= 0:
        raise RuntimeError(
            "Mean connection probability is zero; adjust sigma candidates or f_target."
        )
    else:
        print(f"Mean connection probability: {mean_p:.4f}")

    M = N_target / mean_p  # total distinct inputs required
    G_float = M / s
    G = max(1, int(round(G_float)))
    # Coarse grid side length.
    n_side = int(math.ceil(G ** (1.0 / 3.0)))
    coarse_step = bounding_box_width / n_side
    half = coarse_step / 2.0
    coarse_coords_1d = np.linspace(
        half, bounding_box_width - half, n_side, dtype=np.float32
    )
    coarse_grid_points = np.array(
        [
            (x, y, z)
            for x in coarse_coords_1d
            for y in coarse_coords_1d
            for z in coarse_coords_1d
        ],
        dtype=np.float32,
    )
    dists = _perdiodic_distance_float_parallel(
        np.repeat(
            coarse_grid_points[0][np.newaxis, :],
            receiver_positions.shape[0],
            axis=0,
        ),
        receiver_positions.astype(np.float32),
        bounding_box_width,
    )
    mean_p_calc = np.mean(p0 * np.exp(-(dists**2) / (2.0 * sigma * sigma)))
    print(f"Mean connection probability of first group: {mean_p_calc:.4f}")
    Ggrid = coarse_grid_points.shape[0]
    # Sample G distinct coarse grid positions (guaranteed G <= Ggrid).
    group_indices = rng.choice(Ggrid, size=G, replace=False)
    group_positions = coarse_grid_points[group_indices]
    # Build adjacency lists.
    R = receiver_positions.shape[0]
    groups_by_receiver: List[List[int]] = [[] for _ in range(R)]
    for g_idx, g_pos in enumerate(group_positions):
        # Distances to all receivers.
        dists = _perdiodic_distance_float_parallel(
            np.repeat(g_pos[np.newaxis, :], R, axis=0),
            receiver_positions.astype(np.float32),
            bounding_box_width,
        )
        probs = p0 * np.exp(-(dists**2) / (2.0 * sigma * sigma))
        probs = np.clip(probs, 0.0, 1.0)
        rand = rng.random(R)
        connected_mask = rand < probs
        connected_receivers = np.nonzero(connected_mask)[0]
        if connected_receivers.size == 0:
            print(
                f"Warning: Group {g_idx} at position {g_pos} connected to no receivers."
            )
            # nearest = int(np.argmin(dists))
            # connected_receivers = np.array([nearest], dtype=np.int32)
            # TODO There can be groups without any receivers --> I don't need to simulate their counts later
        for r in connected_receivers:
            groups_by_receiver[r].append(g_idx)
    groups_by_receiver_arr = [
        np.array(lst, dtype=np.int32) if lst else np.empty(0, dtype=np.int32)
        for lst in groups_by_receiver
    ]
    group_dtype = _smallest_unsigned_dtype(int(s))
    degrees = np.array([len(lst) for lst in groups_by_receiver_arr])
    receiver_cap = int(s) * int(degrees.max() if degrees.size else 0)
    receiver_dtype = _smallest_unsigned_dtype(max(1, receiver_cap))

    # Compute per-receiver degree and min degree
    degrees = np.array([len(lst) for lst in groups_by_receiver_arr])
    print("Min groups per receiver (degree):", degrees.min(), "Mean:", degrees.mean())

    # Compute pairwise shared groups and store all receiver pairs with no shared groups
    no_shared_pairs = []
    print(f"loop over {R} receivers to find pairs with no shared groups...")
    print(f"this makes {R*(R-1)//2} pairs to check...")
    for r1 in range(R):
        groups_r1 = set(groups_by_receiver_arr[r1])
        for r2 in range(r1 + 1, R):
            groups_r2 = set(groups_by_receiver_arr[r2])
            shared = groups_r1.intersection(groups_r2)
            if len(shared) == 0:
                no_shared_pairs.append((r1, r2))
    print("Number of receiver pairs with no shared groups:", len(no_shared_pairs))

    return DistanceGroupsState(
        receiver_positions=receiver_positions.astype(np.float32),
        L=bounding_box_width,
        R=R,
        N_target=N_target,
        s=s,
        G=G,
        group_positions=group_positions.astype(np.float32),
        p0=p0,
        sigma=sigma,
        groups_by_receiver=groups_by_receiver_arr,
        group_dtype=group_dtype,
        receiver_dtype=receiver_dtype,
        f_target_samples=f_target_samples,
        f_model_samples=f_model_samples,
    )


def simulate_receiver_counts_distance_dependent(
    state: DistanceGroupsState,
    rate: float,
    dt: float,
    rho: float,
    num_bins: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate receiver counts using a distance-dependent state.

    Uses the precomputed group positions and adjacency lists based on the fitted
    Gaussian connection probability. Group spike counts are generated and then
    aggregated to receivers.
    """
    group_counts = simulate_counts_direct(
        G=state.G,
        N=state.s,
        rate=rate,
        dt=dt,
        rho=rho,
        num_bins=num_bins,
        rng=rng,
        dtype=state.group_dtype,
    )
    receiver_counts = np.zeros((state.R, num_bins), dtype=state.receiver_dtype)
    for r, groups in enumerate(state.groups_by_receiver):
        if groups.size == 0:
            continue
        for g in groups:
            receiver_counts[r] += group_counts[g]
    return receiver_counts


def simulate_receiver_counts_with_groups(
    state: GroupsState,
    rate: float,
    dt: float,
    rho: float,
    num_bins: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate receiver counts for a chunk of time using prebuilt groups.

    Reuse the same ``state`` across multiple calls to accumulate total time
    bins without rebuilding groups.

    Args:
        state (GroupsState): Prebuilt groups and dtype configuration.
        rate (float): Firing rate of each input in Hz.
        dt (float): Bin width in seconds.
        rho (float): Overdispersion/correlation coefficient for input correlations.
        num_bins (int): Number of time bins to simulate in this chunk.
        rng (np.random.Generator): Random number generator.

    Returns:
        np.ndarray: Shape ``(R, num_bins)`` with integer spike counts per receiver.
    """
    if state.fast_path:
        # Fast-path: no overlap (k == 1), simulate directly for each receiver count from
        # N inputs.
        return simulate_counts_direct(
            G=state.R,
            N=state.N,
            rate=rate,
            dt=dt,
            rho=rho,
            num_bins=num_bins,
            rng=rng,
            dtype=state.receiver_dtype,
        )

    # With overlap --> Per-group counts.
    group_counts = simulate_counts_direct(
        G=state.G,
        N=state.s,
        rate=rate,
        dt=dt,
        rho=rho,
        num_bins=num_bins,
        rng=rng,
        dtype=state.group_dtype,
    )
    # Aggregate per-group counts into per-receiver counts.
    receiver_counts = np.zeros((state.R, num_bins), dtype=state.receiver_dtype)
    for g in range(state.G):
        gc = group_counts[g]
        for r in state.groups[g]:
            receiver_counts[r] += gc
    return receiver_counts


def simulate_receiver_counts_homogeneous(
    R: int,
    N: int,
    f: float,
    s: int,
    rate: float,
    dt: float,
    rho: float,
    num_bins: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate incoming spike counts per receiver with controlled overlap.

    This combines group construction with Beta-Binomial spike generation:

    1) Build ``G`` groups of ``s`` inputs each. Every group projects to exactly
       ``k`` receivers, where ``k`` is derived from the desired shared fraction
       ``f`` (see ``build_groups_homogeneous``).
    2) For every group, draw spike counts across ``num_bins`` time bins using
       ``simulate_counts_direct`` with ``N=s`` neurons per group.
    3) For each receiver, sum the spike counts of all groups that connect to it.

    Fast-path optimization:
        When ``k == 1`` (i.e., no overlap across receivers), building many small
        groups and summing is wasteful. In that case, we simulate one group per
        receiver of size ``N`` directly and return the result (shape ``R x num_bins``),
        avoiding intermediate allocations and indirection. This reduces both
        memory usage and runtime.

    Args:
        R (int): Number of receivers.
        N (int): Approximate number of inputs per receiver.
        f (float): Target shared-input fraction in [0, 1]. ``f=0`` means mostly
            private inputs, ``f=1`` means fully shared across receivers.
        s (int): Group size (number of inputs per group).
        rate (float): Firing rate of each input in Hz.
        dt (float): Bin width in seconds.
        rho (float): Overdispersion/correlation coefficient for shared variability
            across groups within a time bin (see ``simulate_counts_direct``).
        num_bins (int): Number of time bins to simulate.
        rng (np.random.Generator): Random number generator.

    Returns:
        np.ndarray: Array of shape ``(R, num_bins)`` with the summed spike counts
        per receiver over time.

        Notes:
                - Efficiency: avoids forming dense incidence matrices and sums per
                    receiver using precomputed group indices.
                - The total number of receiver-side inputs is approximately ``R * N``
                    by construction (subject to rounding in group counts).
    """
    # Backward-compatible wrapper: build groups once, then simulate for one chunk.
    state = build_groups_state(R=R, N=N, f=f, s=s, rng=rng)
    return simulate_receiver_counts_with_groups(
        state=state, rate=rate, dt=dt, rho=rho, num_bins=num_bins, rng=rng
    )


def plot_empirical_and_target_distance_dependent_shared_fraction(
    dist_state: DistanceGroupsState,
    rng: np.random.Generator,
    title: Optional[str] = None,
) -> None:
    print("Sampling receiver pairs to estimate empirical shared fraction curve...")
    max_pairs_sample = 20000
    group_sets = [set(arr.tolist()) for arr in dist_state.groups_by_receiver]
    pair_indices_i = rng.integers(0, dist_state.R, size=max_pairs_sample)
    pair_indices_j = rng.integers(0, dist_state.R, size=max_pairs_sample)
    # replace the sampled pairs with all unique pairs
    pair_indices_i = []
    pair_indices_j = []
    for i in range(dist_state.R):
        for j in range(i + 1, dist_state.R):
            pair_indices_i.append(i)
            pair_indices_j.append(j)
    bins_empirical_shared_groups: Dict[float, List[float]] = {}
    bins_empirical_shared_frac: Dict[float, List[float]] = {}
    for i, j in zip(pair_indices_i, pair_indices_j):
        if i == j:
            continue
        pos_i = dist_state.receiver_positions[i]
        pos_j = dist_state.receiver_positions[j]
        d = _periodic_distance_float(pos_i, pos_j, dist_state.L)
        shared_groups = group_sets[i].intersection(group_sets[j])
        # if not shared_groups:
        #     continue
        shared_inputs = len(shared_groups) * dist_state.s
        inputs_i = len(group_sets[i]) * dist_state.s
        if inputs_i == 0:
            continue
        frac = shared_inputs / inputs_i
        bins_empirical_shared_groups.setdefault(d, []).append(len(shared_groups))
        bins_empirical_shared_frac.setdefault(d, []).append(frac)
    # Average empirical shared fraction per distance bin.
    empirical_curve_shared_groups = {
        d: float(np.mean(vals)) for d, vals in bins_empirical_shared_groups.items()
    }
    empirical_curve = {
        d: float(np.mean(vals)) for d, vals in bins_empirical_shared_frac.items()
    }
    print("Plotting fitted and empirical shared fraction curves...")
    d_model = sorted(dist_state.f_model_samples.keys())
    f_model = [dist_state.f_model_samples[d] for d in d_model]
    f_target_vals = [dist_state.f_target_samples[d] for d in d_model]
    empirical_d_sorted = sorted(empirical_curve.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax0, ax1, ax2, _ = axes.flatten()
    ax0.plot(d_model, f_target_vals, label="target f(d)", color="black")
    ax0.plot(d_model, f_model, label="model f(d)", color="tab:blue")
    ax0.scatter(
        empirical_d_sorted,
        [empirical_curve[d] for d in empirical_d_sorted],
        s=12,
        color="tab:orange",
        alpha=0.7,
        label="empirical",
    )
    ax0.set_xlabel("distance d")
    ax0.set_ylabel("average shared fraction")
    ax0.set_title("Shared fraction vs distance")
    ax0.legend()
    distinct_inputs_counts = [
        len(g) * dist_state.s for g in dist_state.groups_by_receiver
    ]
    ax1.hist(distinct_inputs_counts, bins=30, color="tab:green", alpha=0.8)
    ax1.set_xlabel("distinct inputs per receiver")
    ax1.set_ylabel("count")
    ax1.set_title("Distribution of inputs")

    ax2.scatter(
        empirical_d_sorted,
        [empirical_curve_shared_groups[d] for d in empirical_d_sorted],
        s=12,
        color="tab:orange",
        alpha=0.7,
        label="empirical",
    )
    ax2.set_xlabel("distance d")
    ax2.set_ylabel("average shared groups")
    ax2.set_title("Shared groups vs distance")
    ax2.legend()

    if title:
        fig.suptitle(title, fontsize=16)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":

    rng = np.random.default_rng()

    # Demonstrate spike count simulation
    print("Demonstrating spike count simulation:")
    # Demonstration: print two example count matrices for different rho values.
    (G, N, rate, dt, rho, num_bins) = (5, 10, 200, 0.001, 0.8, 15)
    example_spike_counts = simulate_counts_direct(
        G=G, N=N, rate=rate, dt=dt, rho=rho, num_bins=num_bins, rng=rng
    )
    print("G (total groups):", G)
    print("N (neurons per group):", N)
    print("Rate (Hz):", rate)
    print("dt (s):", dt)
    print("rho (correlation):", rho)
    print("num_bins (time bins):", num_bins)
    print("-->")
    print("resulting spike counts:")
    print(example_spike_counts)

    # Demonstrate building groups
    print("\nDemonstrating group building:")
    (R, N, f, s) = (5, 10, 0.5, 2)
    groups, k, k_float, G, groups_by_receiver = build_groups_homogeneous(
        R=R, N=N, f=f, s=s, rng=rng
    )
    print("R (receivers):", R)
    print("N (inputs per receiver):", N)
    print("f (shared fraction of inputs):", f)
    print("s (input group size):", s)
    print("-->")
    print("resulting receivers per group:", k)
    print("resulting groups (with corresponding receiver indices):")
    print(groups)
    print("groups by receiver:")
    for r in range(R):
        print(f"  receiver {r}: {groups_by_receiver[r]}")

    # Demonstrate per-receiver incoming spike streams with overlap
    print("\nDemonstrating per-receiver spike streams with overlap:")
    R_demo, N_demo, f_demo, s_demo = 6, 100, 0.3, 5
    rate_demo, dt_demo, rho_demo, num_bins_demo = 50.0, 0.001, 0.2, 20
    receiver_counts = simulate_receiver_counts_homogeneous(
        R=R_demo,
        N=N_demo,
        f=f_demo,
        s=s_demo,
        rate=rate_demo,
        dt=dt_demo,
        rho=rho_demo,
        num_bins=num_bins_demo,
        rng=rng,
    )
    print("R (receivers):", R_demo)
    print("Approx N (inputs/receiver):", N_demo)
    print("f (shared fraction):", f_demo)
    print("s (group size):", s_demo)
    print("rate (Hz):", rate_demo, "dt (s):", dt_demo, "rho:", rho_demo)
    print("num_bins:", num_bins_demo)
    print("-->")
    print("receiver spike counts shape:", receiver_counts.shape)
    # Show a small snippet for quick inspection
    print("receiver 0, first 10 bins:", receiver_counts[0, :10])
    print("receiver 1, first 10 bins:", receiver_counts[1, :10])

    # Demonstrate the optimized no-overlap path (k == 1)
    print("\nDemonstrating no-overlap fast path (k == 1):")
    R_no, N_no, f_no, s_no = 5, 120, 0.0, 4  # f=0 -> k=1 regardless of s
    rate_no, dt_no, rho_no, num_bins_no = 30.0, 0.001, 0.1, 12
    receiver_counts_no = simulate_receiver_counts_homogeneous(
        R=R_no,
        N=N_no,
        f=f_no,
        s=s_no,
        rate=rate_no,
        dt=dt_no,
        rho=rho_no,
        num_bins=num_bins_no,
        rng=rng,
    )
    print("R (receivers):", R_no)
    print("N (inputs/receiver):", N_no)
    print("f (shared fraction):", f_no)
    print("s (group size, unused in fast path):", s_no)
    print("rate (Hz):", rate_no, "dt (s):", dt_no, "rho:", rho_no)
    print("num_bins:", num_bins_no)
    print("-->")
    print("receiver spike counts shape:", receiver_counts_no.shape)
    print("receiver 0, first 10 bins:", receiver_counts_no[0, :10])

    # ------------------------------------------------------------------
    # Distance-dependent shared fraction demonstration (updated)
    # ------------------------------------------------------------------
    print("\nDistance-dependent shared fraction demonstration (bounding box version):")
    bounding_box_width = 10.0  # cube side length
    # Receiver positions: same 10x10x10 lattice inside the bounding box.
    lattice_side = 10
    step = bounding_box_width / lattice_side
    half = step / 2.0
    receiver_positions = np.array(
        [
            (x * step + half, y * step + half, z * step + half)
            for x in range(lattice_side)
            for y in range(lattice_side)
            for z in range(lattice_side)
        ],
        dtype=np.float32,
    )
    R_dist = receiver_positions.shape[0]
    # print how many receivers we have
    print(f"Number of receivers: {R_dist}")
    print(f"minimum distance between receivers: {step:.3f}")
    print(
        f"maximum possible periodic distance within bounding box: {math.sqrt(3) * (bounding_box_width / 2):.3f}"
    )
    # Get the center receiver index
    center_index = R_dist // 2

    def f_target(d: float) -> float:
        return 0.2 * math.exp(-((d / (bounding_box_width / 2.5)) ** 2))

    N_target = 400
    s_group = 20
    print("Fitting Gaussian connection probability using fine 100^3 grid...")
    dist_state = build_distance_groups_state(
        receiver_positions=receiver_positions,
        bounding_box_width=bounding_box_width,
        N_target=N_target,
        s=s_group,
        f_target=f_target,
        rng=rng,
        center_index=0,  # center_index,
        fine_grid_resolution=10,
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
    rate_dd, dt_dd, rho_dd, num_bins_dd = 15.0, 0.001, 0.25, 30
    receiver_counts_dd = simulate_receiver_counts_distance_dependent(
        state=dist_state,
        rate=rate_dd,
        dt=dt_dd,
        rho=rho_dd,
        num_bins=num_bins_dd,
        rng=rng,
    )
    print("Receiver counts (distance-dependent) shape:", receiver_counts_dd.shape)
    plot_empirical_and_target_distance_dependent_shared_fraction(
        dist_state=dist_state,
        rng=rng,
    )


""" CHECK IF THIS WORKED

Change the script so that the user provides a bounding box width (width, heigth, depth are all the same, it's a cube) together with the receiver positions to the build_distance_groups_state() function. The bounding box defines the periodic borders. The receiver positions are the coordinates within the bounding box.

Therefore the periodic borders conditions do not need to be calculated based on the receiver positions but simply based on the bounding box dimensions everywhere.

Also the way how the group positions (on an uniform grid) are obtained should be changed. The group postions should be distributed on a uniform grid within the bounding box.

To fit the function p(d) a very fine grid (100x100x100) should be used for the group positions.

After fitting p(d) and calculating the actual number of groups G, a new grid over the bounding box should be defined for the actual number of groups G. If it is not possible to distribute the number of groups G uniformly over a 3D grid in the bounding box, the next larger possible grid size should be used (so there are equal or more grid elements than groups). Then the groups should be randomly assigned to these grid positions, as currently after the line "if G <= Ggrid", i.e., without replacement. In the new version it should be guaranteed that "G <= Ggrid" because the grid is defined to be just as large as necessary to contain all the groups.

Also update the section for the empirical shared fraction estimation according to the updated methods usig teh bounding box.


"""
