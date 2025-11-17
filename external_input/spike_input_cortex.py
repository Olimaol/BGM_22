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
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


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
