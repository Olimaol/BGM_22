"""Benchmark the real stream generators at true per-stream scale.

Each phase runs in its own forked subprocess so its peak RSS (VmHWM) is
attributable to that phase alone. Everything is sized for a laptop: one TR of
bins (23 100 at dt = 0.1 ms) per stream, memmaps written to scratch and
deleted afterwards. Extrapolations to the full cache are linear in n_steps and
cross-checked in the demo page against the known aggregates.

    /home/oliver/miniforge3/envs/compneuro/bin/python bench_real_scale.py
"""

import json
import multiprocessing as mp
import os
import platform
import shutil
import time

import numpy as np

from demo_common import DEMO_DIR, REAL, SCRATCH, lattice_spacing_mm, p_exp

BENCH_PATH = os.path.join(DEMO_DIR, "bench.json")
BINS_1TR = REAL["bins_per_tr"]
DT = REAL["dt_ms"]


def read_vm_hwm_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM"):
                return float(line.split()[1]) / 1024.0
    return float("nan")


def rate_1tr(key="dlPFC_rate"):
    from demo_common import load_rate_series

    return np.repeat(load_rate_series(key, n_trs=1), BINS_1TR)


# ---------------------------------------------------------------------------
# Phases (each runs in a fresh subprocess)
# ---------------------------------------------------------------------------


def phase_1a_pool_streams():
    """Case 1a, caudate dlPFC, all three receiver types, 1 TR."""
    from CompNeuroPy.striatal_microcircuit.spike_input_cortex import (
        axon_pool_size, simulate_cortical_axon_pool_streams_to_memmap,
    )

    prop = REAL["caudate_proportions"]["dlPFC"]
    n_ref = int(round(prop * REAL["N_cortical_inputs"]["dSPN"]))
    pool = axon_pool_size(n_ref, REAL["shared_fraction"])
    streams = {}
    for ctype in ("dSPN", "iSPN", "FS"):
        n_eff = int(round(prop * REAL["N_cortical_inputs"][ctype]))
        streams[ctype] = {
            "filename": os.path.join(SCRATCH, f"bench_1a_{ctype}.dat"),
            "R": REAL["type_counts"][ctype],
            "N": n_eff,
        }
    simulate_cortical_axon_pool_streams_to_memmap(
        streams=streams,
        pool_size=pool,
        rate=rate_1tr(),
        dt=DT,
        num_bins=BINS_1TR,
        receiver_dtype=np.float64,
        rng=np.random.default_rng(1),
    )
    written = sum(
        os.path.getsize(s["filename"]) for s in streams.values()
    )
    return {
        "pool_size": pool,
        "streams": {
            ct: {"R": s["R"], "N": s["N"]} for ct, s in streams.items()
        },
        "bytes_written": written,
    }


def phase_1a_primitives():
    """The three primitives of one 1a chunk, timed separately."""
    rng = np.random.default_rng(2)
    pool = 275000
    chunk = 5592  # what _chunk_size_for gives at r_total = 1000, 128 MB
    p_t = np.full(chunk, 5e-4)
    out = {}
    t0 = time.perf_counter()
    k = rng.binomial(pool, p_t)
    out["binomial_k_ms_per_chunk"] = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    rng.hypergeometric(3850, pool - 3850, np.broadcast_to(k, (486, chunk)))
    out["hypergeometric_ms_per_chunk_dSPN"] = (
        time.perf_counter() - t0
    ) * 1e3
    counts = np.zeros((486, chunk))
    mm = np.memmap(
        os.path.join(SCRATCH, "bench_prim.dat"), dtype=np.float64,
        mode="w+", shape=(486, BINS_1TR),
    )
    t0 = time.perf_counter()
    mm[:, :chunk] = counts
    mm.flush()
    out["memmap_write_ms_per_chunk"] = (time.perf_counter() - t0) * 1e3
    out["chunk_bins"] = chunk
    return out


def phase_1b_pool_build():
    """Case 1b, iSPN->dSPN at full real scale: build the source cloud."""
    from CompNeuroPy.striatal_microcircuit.spike_input_cortex import (
        build_geometric_source_pools,
    )

    rng = np.random.default_rng(3)
    d_mm = lattice_spacing_mm()
    nx, b = REAL["lattice_nx"], REAL["lattice_b"]
    xs = np.arange(nx) * d_mm
    ys = np.arange(b) * d_mm
    X, Y, Z = np.meshgrid(xs, ys, ys, indexing="ij")
    lattice = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    positions = lattice[
        rng.choice(lattice.shape[0], REAL["type_counts"]["dSPN"],
                   replace=False)
    ]
    P0, sigma_um = REAL["kernels"]["iSPN-dSPN"]
    rho = 0.43 / 0.886 * REAL["density_per_mm3"]
    pools = build_geometric_source_pools(
        receiver_positions=positions,
        p_func=lambda d: p_exp(d, P0, sigma_um * 1e-3),
        r_in=0.114,
        r_out=3 * 0.4003,  # 3 sigma of the widest kernel onto dSPN
        density_pre=rho,
        rng=rng,
        multiplicity=REAL["source_multiplicity"],
    )
    np.savez(
        os.path.join(SCRATCH, "bench_1b_pools.npz"),
        src_indptr=pools.src_indptr,
        src_receivers=pools.src_receivers,
        n_sources=pools.n_sources,
        multiplicity=pools.multiplicity,
        degrees=pools.degrees,
        n_receivers=pools.n_receivers,
    )
    return {
        "n_sources": int(pools.n_sources),
        "mean_n_eff": float(pools.mean_n_eff),
        "n_receivers": int(pools.n_receivers),
    }


def phase_1b_stream():
    """Case 1b, iSPN->dSPN, 1 TR of counts from the saved pool."""
    from CompNeuroPy.striatal_microcircuit.spike_input_cortex import (
        GeometricSourcePools, simulate_receiver_counts_geometric_to_memmap,
    )

    z = np.load(os.path.join(SCRATCH, "bench_1b_pools.npz"))
    pools = GeometricSourcePools(
        src_indptr=z["src_indptr"],
        src_receivers=z["src_receivers"],
        n_sources=int(z["n_sources"]),
        multiplicity=int(z["multiplicity"]),
        degrees=z["degrees"],
        n_receivers=int(z["n_receivers"]),
    )
    fname = os.path.join(SCRATCH, "bench_1b_stream.dat")
    simulate_receiver_counts_geometric_to_memmap(
        filename=fname,
        pools=pools,
        rate=REAL["firing_rate_dict"]["iSPN"],
        dt=DT,
        num_bins=BINS_1TR,
        receiver_dtype=np.float64,
        rng=np.random.default_rng(4),
    )
    return {"bytes_written": os.path.getsize(fname)}


def phase_1c_stream():
    """Case 1c, thal <- dlPFC (caudate), 1 TR."""
    from CompNeuroPy.striatal_microcircuit.spike_input_cortex import (
        simulate_receiver_counts_homogeneous_to_memmap,
    )

    n_eff = int(
        round(REAL["caudate_proportions"]["dlPFC"] * REAL["ci_N_total"]["thal"])
    )
    fname = os.path.join(SCRATCH, "bench_1c_stream.dat")
    simulate_receiver_counts_homogeneous_to_memmap(
        filename=fname,
        R=REAL["ci_pop_size"],
        N=n_eff,
        shared_input=REAL["ci_shared_fraction"],
        rate=rate_1tr(),
        dt=DT,
        num_bins=BINS_1TR,
        receiver_dtype=np.float64,
        rng=np.random.default_rng(5),
    )
    return {"N_eff": n_eff, "bytes_written": os.path.getsize(fname)}


PHASES = [
    ("1a_pool_streams_1tr", phase_1a_pool_streams),
    ("1a_primitives", phase_1a_primitives),
    ("1b_pool_build", phase_1b_pool_build),
    ("1b_stream_1tr", phase_1b_stream),
    ("1c_stream_1tr", phase_1c_stream),
]


def _child(fn, queue):
    baseline_mb = read_vm_hwm_mb()
    t0 = time.perf_counter()
    detail = fn()
    wall = time.perf_counter() - t0
    queue.put(
        {
            "wall_s": wall,
            "peak_rss_mb": read_vm_hwm_mb(),
            "baseline_rss_mb": baseline_mb,
            "detail": detail,
        }
    )


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    ctx = mp.get_context("fork")
    results = {
        "machine": platform.node(),
        "cpu": platform.processor() or platform.machine(),
        "bins_per_tr": BINS_1TR,
        "dt_ms": DT,
        "phases": {},
    }
    for name, fn in PHASES:
        print(f"phase {name} ...", flush=True)
        q = ctx.Queue()
        proc = ctx.Process(target=_child, args=(fn, q))
        proc.start()
        payload = q.get()
        proc.join()
        results["phases"][name] = payload
        print(
            f"  {payload['wall_s']:.2f} s, peak RSS "
            f"{payload['peak_rss_mb']:.0f} MB "
            f"(baseline {payload['baseline_rss_mb']:.0f} MB)"
        )

    with open(BENCH_PATH, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {BENCH_PATH}")
    shutil.rmtree(SCRATCH, ignore_errors=True)


if __name__ == "__main__":
    main()
