"""Build every figure of the stream-generation demo.

Run from anywhere with the compneuro interpreter:

    /home/oliver/miniforge3/envs/compneuro/bin/python make_figures.py

Toy structures are explicit reimplementations sized to be drawable
(exaggerated shared fraction); every case is then cross-checked by running the
real CompNeuroPy generator on the same parameters and comparing the measured
statistics against the closed-form targets. Snapshot 2026-08-11.
"""

import os
import shutil

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from demo_common import (
    BASELINE, C_DSPN, C_FS, C_ISPN, C_SHARED, C_TYPE, FIG_DIR, GRID, INK,
    INK2, MUTED, REAL, SCRATCH, SURFACE, dump_results, load_rate_series,
    lattice_spacing_mm, new_fig, p_exp, save_fig, style_axes,
)

from CompNeuroPy.striatal_microcircuit.spike_input_cortex import (
    build_geometric_source_pools,
    ou_sum_variance,
    simulate_cortical_axon_pool_streams_to_memmap,
    simulate_receiver_counts_geometric_to_memmap,
    simulate_receiver_counts_homogeneous_to_memmap,
    solve_modulation_amplitude,
    stream_target_statistics,
)

C_OTHER = "#e87ba4"  # magenta: the "other receiver" (B) in 1b/1c figures

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(SCRATCH, exist_ok=True)

RNG = np.random.default_rng(7)

# toy scale used across case 1a / 1c time courses
TOY_BINS_PER_TR = 40
TOY_P_MEAN = 0.25  # mean p per bin (real: 5 Hz * 0.1 ms -> 5e-4)


def toy_p_trace(n_trs=6):
    """p(t) staircase from the real dlPFC series, rescaled to be visible."""
    rate = load_rate_series("dlPFC_rate", n_trs=n_trs)
    p_per_tr = rate / rate.mean() * TOY_P_MEAN
    return np.repeat(p_per_tr, TOY_BINS_PER_TR), p_per_tr, rate


# ===========================================================================
# Section 0 -- what a stream is (count matrix heatmap)
# ===========================================================================


def fig_stream_matrix():
    p_t, _, _ = toy_p_trace(n_trs=3)
    n_show = 3 * TOY_BINS_PER_TR
    R = 7
    counts = RNG.binomial(8, p_t[None, :n_show] * np.ones((R, 1)))
    fig, ax = new_fig(7.2, 2.2)
    im = ax.imshow(
        counts, aspect="auto", cmap="Blues", interpolation="nearest",
        vmin=0, vmax=counts.max(),
    )
    ax.set_facecolor(SURFACE)
    ax.set_xlabel("time bin t  (one column per dt)")
    ax.set_ylabel("receiver i")
    ax.set_yticks(range(R))
    ax.set_yticklabels([f"r{i}" for i in range(R)])
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title("A stream is one (R, n_steps) matrix of counts")
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.01)
    cbar.set_label("spikes into receiver i in bin t", color=INK2, fontsize=8)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.outline.set_edgecolor(BASELINE)
    for tr in (1, 2):
        ax.axvline(tr * TOY_BINS_PER_TR - 0.5, color=SURFACE, lw=2)
    save_fig(fig, "stream_matrix.png")


# ===========================================================================
# Case 1a -- cortical axon pool
# ===========================================================================

M_TOY = 40
RECEIVERS_1A = [  # (type, label, N)
    ("dSPN", "A", 8), ("dSPN", "B", 8), ("dSPN", "C", 8),
    ("iSPN", "D", 8), ("iSPN", "E", 8),
    ("FS", "F", 4), ("FS", "G", 4),
]


def build_membership_1a():
    return [RNG.choice(M_TOY, size=n, replace=False) for _, _, n in RECEIVERS_1A]


def fig_1a_structure(membership):
    fig, (ax1, ax2) = new_fig(8.4, 4.6, nrows=2, height_ratios=[1.35, 1.0])

    # --- panel 1: bipartite pool -> receivers, pair (A, B) highlighted
    ax1.set_facecolor(SURFACE)
    ax1.set_xlim(-1.5, M_TOY + 0.5)
    ax1.set_ylim(-0.55, 1.35)
    ax1.axis("off")
    shared_ab = np.intersect1d(membership[0], membership[1])
    rx = np.linspace(3, M_TOY - 4, len(RECEIVERS_1A))
    for i, ((ctype, label, n), mem) in enumerate(zip(RECEIVERS_1A, membership)):
        col = C_TYPE[ctype]
        for a in mem:
            is_sh = i < 2 and a in shared_ab
            ax1.plot(
                [rx[i], a], [0.06, 0.94],
                color=C_SHARED if is_sh else col,
                lw=1.6 if is_sh else 0.7,
                alpha=0.95 if is_sh else 0.28,
                zorder=3 if is_sh else 1,
            )
        ax1.scatter(
            [rx[i]], [0.0], s=170, marker="s", color=col, zorder=4,
            edgecolors=SURFACE, linewidths=1.5,
        )
        ax1.text(
            rx[i], -0.3, f"{label}\n{ctype}  N={n}", ha="center",
            va="top" if False else "center", fontsize=8, color=INK2,
        )
    axon_cols = [
        C_SHARED if a in shared_ab else MUTED for a in range(M_TOY)
    ]
    ax1.scatter(
        range(M_TOY), [1.0] * M_TOY, s=26, color=axon_cols, zorder=4,
        edgecolors=SURFACE, linewidths=1.0,
    )
    ax1.text(
        M_TOY / 2, 1.24,
        f"one axon pool per cortical region:  M = {M_TOY} axons "
        f"(toy;  real dlPFC: 275 000)",
        ha="center", fontsize=9, color=INK, fontweight="semibold",
    )
    ax1.text(
        -1.2, 1.0, "axons", ha="right", va="center", fontsize=8, color=MUTED,
    )
    ax1.text(
        -1.2, 0.0, "receivers", ha="right", va="center", fontsize=8,
        color=MUTED,
    )
    ax1.text(
        M_TOY - 0.5, 0.5,
        f"A and B share {len(shared_ab)} axons\n"
        f"E[shared] = N²/M = {8 * 8 / M_TOY:.1f}",
        ha="right", va="center", fontsize=8, color=C_SHARED,
    )

    # --- panel 2: membership matrix
    R = len(RECEIVERS_1A)
    img = np.zeros((R, M_TOY, 4))
    from matplotlib.colors import to_rgba

    for i, ((ctype, _, _), mem) in enumerate(zip(RECEIVERS_1A, membership)):
        img[i, :, :] = to_rgba(SURFACE)
        img[i, mem, :] = to_rgba(C_TYPE[ctype])
    ax2.imshow(img, aspect="auto", interpolation="nearest")
    for a in shared_ab:
        ax2.add_patch(
            mpatches.Rectangle(
                (a - 0.5, -0.5), 1.0, 2.0, fill=False, edgecolor=C_SHARED,
                lw=1.8, zorder=5,
            )
        )
    ax2.set_yticks(range(R))
    ax2.set_yticklabels(
        [f"{lbl} ({ct})" for ct, lbl, _ in RECEIVERS_1A], fontsize=8
    )
    ax2.set_xlabel("axon index 0 … M−1")
    ax2.set_title(
        "the same thing as a membership matrix — "
        "which the real code never builds",
        fontsize=9,
    )
    ax2.tick_params(colors=MUTED, labelsize=8)
    for i in range(R + 1):
        ax2.axhline(i - 0.5, color=SURFACE, lw=2)
    save_fig(fig, "fig_1a_structure.png")
    return shared_ab


def sim_1a_explicit(membership, p_t):
    """Ground-truth explicit construction: every axon simulated."""
    T = p_t.shape[0]
    axon_spikes = RNG.random((M_TOY, T)) < p_t[None, :]
    k_t = axon_spikes.sum(axis=0)
    counts = np.stack([axon_spikes[mem].sum(axis=0) for mem in membership])
    return k_t, counts


def fig_1a_timecourse(membership):
    p_t, p_per_tr, rate = toy_p_trace(n_trs=6)
    k_t, counts = sim_1a_explicit(membership, p_t)
    t = np.arange(p_t.shape[0])

    fig, axes = new_fig(8.4, 7.2, nrows=4, sharex=True)
    fig.subplots_adjust(hspace=0.55)
    ax_r, ax_k, ax_c, ax_f = axes

    ax_r.step(t, p_t, where="post", color=INK2, lw=1.8)
    ax_r.set_ylabel("p(t) = rate·dt/1000")
    ax_r.set_title(
        "step 2: one shared probability trace per stream — a staircase, "
        "one value per TR (real dlPFC snippet, rescaled)"
    )
    for tr in range(1, 6):
        for ax in axes:
            ax.axvline(tr * TOY_BINS_PER_TR, color=GRID, lw=0.8, zorder=0)
    style_axes(ax_r)

    ax_k.plot(t, k_t, color=C_SHARED, lw=1.4)
    ax_k.plot(
        t, M_TOY * p_t, color=MUTED, lw=1.2, linestyle=(0, (4, 3)),
    )
    ax_k.set_ylabel("k(t)")
    ax_k.set_title(
        "step 3a, first draw: k(t) ~ Binomial(M, p(t)) — how many pool "
        "axons fire; drawn ONCE, shared by every receiver"
    )
    ax_k.text(
        t[-1], M_TOY * p_t[-1] + 1.5, "M·p(t)", color=MUTED, fontsize=8,
        ha="right",
    )
    style_axes(ax_k)

    for idx, lbl in ((0, "dSPN A"), (3, "iSPN D"), (5, "FS F")):
        ctype = RECEIVERS_1A[idx][0]
        ax_c.plot(t, counts[idx], color=C_TYPE[ctype], lw=1.3, label=lbl)
    ax_c.set_ylabel("c_i(t)")
    ax_c.set_title(
        "step 3a, second draw: each receiver keeps the firing axons it owns "
        "— all types ride the same k(t)"
    )
    ax_c.legend(loc="upper right", ncols=3)
    style_axes(ax_c)

    # per-TR averages make the co-fluctuation visible through the noise
    per_tr = counts[:, : 6 * TOY_BINS_PER_TR].reshape(len(RECEIVERS_1A), 6, -1).mean(axis=2)
    tr_centers = (np.arange(6) + 0.5) * TOY_BINS_PER_TR
    for idx, lbl in ((0, "dSPN A"), (1, "dSPN B")):
        ax_f.plot(
            tr_centers, per_tr[idx], color=C_DSPN, lw=1.8,
            alpha=1.0 if idx == 0 else 0.45, marker="o", markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1.2, label=lbl,
        )
    ax_f.step(
        t, 8 * p_t, where="post", color=MUTED, lw=1.2,
        linestyle=(0, (4, 3)), label="N·p(t)",
    )
    ax_f.set_ylabel("mean count / bin")
    ax_f.set_xlabel("time bin (toy: 40 bins per TR;  real: 23 100)")
    ax_f.set_title(
        "two dSPNs, averaged per TR: both follow the drive; the residual "
        "co-fluctuation beyond it is the shared input"
    )
    ax_f.legend(loc="upper right", ncols=3)
    style_axes(ax_f)
    save_fig(fig, "fig_1a_timecourse.png")


def stats_from_counts(counts):
    mean = counts.mean(axis=1)
    fano = counts.var(axis=1) / np.maximum(mean, 1e-12)
    cc = np.corrcoef(counts)
    return mean, fano, cc


def run_1a_equivalence(membership):
    """Explicit toy vs real hypergeometric code vs analytic, long run."""
    T = 400_000
    p_const = TOY_P_MEAN
    p_t = np.full(T, p_const)
    _, counts_explicit = sim_1a_explicit(membership, p_t)

    # real generator on identical parameters (rate in Hz at dt=1 ms)
    streams = {}
    r_of_type = {"dSPN": 3, "iSPN": 2, "FS": 2}
    n_of_type = {"dSPN": 8, "iSPN": 8, "FS": 4}
    for ctype in ("dSPN", "iSPN", "FS"):
        streams[ctype] = {
            "filename": os.path.join(SCRATCH, f"toy1a_{ctype}.dat"),
            "R": r_of_type[ctype],
            "N": n_of_type[ctype],
        }
    simulate_cortical_axon_pool_streams_to_memmap(
        streams=streams,
        pool_size=M_TOY,
        rate=p_const * 1000.0,  # dt = 1 ms
        dt=1.0,
        num_bins=T,
        receiver_dtype=np.int16,
        rng=np.random.default_rng(11),
    )
    counts_real = np.concatenate(
        [
            np.array(
                np.memmap(
                    streams[ct]["filename"], dtype=np.int16, mode="r",
                    shape=(r_of_type[ct], T),
                ),
                dtype=np.float64,
            )
            for ct in ("dSPN", "iSPN", "FS")
        ]
    )

    mean_e, fano_e, cc_e = stats_from_counts(counts_explicit.astype(np.float64))
    mean_r, fano_r, cc_r = stats_from_counts(counts_real)

    pairs = {  # (i, j) into the 7-receiver order, ensemble-analytic f
        "dSPN↔dSPN": (0, 1, 8 * 8),
        "dSPN↔iSPN": (0, 3, 8 * 8),
        "dSPN↔FS": (0, 5, 8 * 4),
        "FS↔FS": (5, 6, 4 * 4),
    }
    overlap = {
        key: len(np.intersect1d(membership[i], membership[j]))
        for key, (i, j, _) in pairs.items()
    }
    result = {
        "T": T, "p": p_const,
        "mean": {
            "explicit": [mean_e[0], mean_e[3], mean_e[5]],
            "real": [mean_r[0], mean_r[3], mean_r[5]],
            "analytic": [8 * p_const, 8 * p_const, 4 * p_const],
        },
        "fano": {
            "explicit": [fano_e[0], fano_e[3], fano_e[5]],
            "real": [fano_r[0], fano_r[3], fano_r[5]],
            "analytic": [1 - p_const] * 3,
        },
        "corr": {
            key: {
                "explicit": cc_e[i, j],
                "real": cc_r[i, j],
                "analytic_ensemble": np.sqrt(ninj) / M_TOY,
                "analytic_this_membership": overlap[key] / np.sqrt(ninj),
                "shared_axons": overlap[key],
            }
            for key, (i, j, ninj) in pairs.items()
        },
    }
    return result


def fig_1a_equivalence(eq):
    fig, (ax1, ax2) = new_fig(8.4, 3.1, ncols=2, width_ratios=[1, 1.4])

    # panel 1: mean and Fano per type
    labels = ["dSPN", "iSPN", "FS"]
    x = np.arange(3)
    w = 0.34
    ax1.bar(
        x - w / 2, eq["fano"]["explicit"], w, color=C_DSPN,
        label="explicit toy pool",
    )
    ax1.bar(
        x + w / 2, eq["fano"]["real"], w, color=C_OTHER, label="real generator"
    )
    ax1.plot(
        [x[0] - w, x[-1] + w], [eq["fano"]["analytic"][0]] * 2,
        color=INK, lw=1.4, linestyle=(0, (4, 3)), label="target 1−p",
    )
    ax1.set_xticks(x, labels)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("Fano factor")
    ax1.set_title("marginals match")
    ax1.legend(loc="lower right", fontsize=7)
    style_axes(ax1)

    # panel 2: pairwise correlations
    keys = list(eq["corr"].keys())
    x = np.arange(len(keys))
    exp_v = [eq["corr"][k]["explicit"] for k in keys]
    real_v = [eq["corr"][k]["real"] for k in keys]
    ens = [eq["corr"][k]["analytic_ensemble"] for k in keys]
    this = [eq["corr"][k]["analytic_this_membership"] for k in keys]
    ax2.bar(x - w / 2, exp_v, w, color=C_DSPN, label="explicit toy pool")
    ax2.bar(x + w / 2, real_v, w, color=C_OTHER, label="real generator")
    for xi, (e, t_) in enumerate(zip(ens, this)):
        ax2.plot(
            [xi - w, xi + w], [e, e], color=INK, lw=1.4,
            linestyle=(0, (4, 3)),
            label="√(NᵢNⱼ)/M" if xi == 0 else None,
        )
        ax2.plot(
            [xi - w * 1.15, xi + w * 0.15], [t_, t_], color=C_SHARED,
            lw=2.4, zorder=6,
            label="this membership" if xi == 0 else None,
        )
    ax2.set_xticks(x, keys, fontsize=8)
    ax2.set_ylabel("pairwise correlation")
    ax2.set_title("correlations: overlap realised two ways")
    ax2.legend(loc="upper right", fontsize=7)
    style_axes(ax2)
    save_fig(fig, "fig_1a_equivalence.png")


# ===========================================================================
# Case 1b -- geometric source pools (missing GABA)
# ===========================================================================


def fig_1b_kernels():
    d_um = np.linspace(0, 1300, 400)
    fig, ax = new_fig(6.8, 2.9)
    show = ["iSPN-dSPN", "FS-iSPN", "FS-FS"]
    cols = {"iSPN-dSPN": C_ISPN, "FS-iSPN": C_FS, "FS-FS": INK2}
    label_at = {  # hand-placed so the three labels never collide
        "FS-iSPN": (300, 0.60),
        "iSPN-dSPN": (330, 0.10),
        "FS-FS": (30, 0.185),
    }
    for key in show:
        P0, sig = REAL["kernels"][key]
        ax.plot(d_um, p_exp(d_um, P0, sig), color=cols[key], lw=1.8)
        lx, ly = label_at[key]
        ax.text(
            lx, ly, f"{key.replace('-', '→')}", color=cols[key], fontsize=8,
            fontweight="semibold",
        )
    d_spacing = lattice_spacing_mm() * 1000
    r_in = REAL["lattice_b"] / 2 * d_spacing
    ax.axvspan(0, r_in, color=GRID, alpha=0.55, zorder=0)
    ax.text(
        r_in / 2, 0.86, "simulated\n(< Rin)", ha="center", fontsize=8,
        color=INK2,
    )
    ax.text(
        (r_in + 1200) / 2, 0.86, "missing — replaced by the stream "
        "(Rin ≤ d ≤ Rout = 3σ)", ha="center", fontsize=8,
        color=INK2,
    )
    ax.set_xlabel("distance d (µm)")
    ax.set_ylabel("p(connect) = P₀·exp(−d²/σ²)")
    ax.set_title(
        "the real fitted kernels: connectivity falls with distance, and only "
        "the inner part is simulated"
    )
    style_axes(ax)
    save_fig(fig, "fig_1b_kernels.png")


class Toy2DPool:
    """2D toy version of build_geometric_source_pools (structure only)."""

    def __init__(self, rng):
        n_side = 10
        xs = np.arange(n_side, dtype=float)
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        self.receivers = np.vstack([X.ravel(), Y.ravel()]).T
        self.r_in, self.r_out = 1.5, 6.0
        self.P0, self.sigma = 0.35, 3.0
        density = 4.0
        lo = self.receivers.min(axis=0) - self.r_out
        hi = self.receivers.max(axis=0) + self.r_out
        n_src = int(round(density * np.prod(hi - lo)))
        self.sources = rng.uniform(lo, hi, size=(n_src, 2))
        self.pools = []
        for r in self.receivers:
            d = np.linalg.norm(self.sources - r, axis=1)
            accept = (
                (d >= self.r_in)
                & (d <= self.r_out)
                & (rng.random(n_src) < p_exp(d, self.P0, self.sigma))
            )
            self.pools.append(np.flatnonzero(accept))
        self.degrees = np.array([len(p) for p in self.pools])

    def shared(self, i, j):
        return np.intersect1d(self.pools[i], self.pools[j])

    def f_pair(self, i, j):
        return len(self.shared(i, j)) / np.sqrt(
            self.degrees[i] * self.degrees[j]
        )


def fig_1b_structure(pool):
    idx_a = 4 * 10 + 4  # (4, 4)
    idx_b = 6 * 10 + 4  # (6, 4): distance 2
    idx_c = 4 * 10 + 9  # far in the remaining grid... use (9,9)
    idx_c = 9 * 10 + 9
    fig, axes = new_fig(8.6, 4.1, ncols=2, sharey=True)
    for ax, idx_other, title in (
        (axes[0], idx_b, "a NEAR pair shares many sources"),
        (axes[1], idx_c, "a FAR pair shares almost none"),
    ):
        ax.set_facecolor(SURFACE)
        ax.scatter(
            pool.sources[:, 0], pool.sources[:, 1], s=3, color=GRID,
            zorder=1,
        )
        ax.scatter(
            pool.receivers[:, 0], pool.receivers[:, 1], s=9, color=MUTED,
            marker="s", zorder=2,
        )
        a, b = pool.pools[idx_a], pool.pools[idx_other]
        sh = pool.shared(idx_a, idx_other)
        only_a = np.setdiff1d(a, sh)
        only_b = np.setdiff1d(b, sh)
        ax.scatter(
            pool.sources[only_a, 0], pool.sources[only_a, 1], s=14,
            color=C_DSPN, zorder=3, label="pool of A",
        )
        ax.scatter(
            pool.sources[only_b, 0], pool.sources[only_b, 1], s=14,
            color=C_OTHER, zorder=3, label="pool of B",
        )
        ax.scatter(
            pool.sources[sh, 0], pool.sources[sh, 1], s=42, color=C_SHARED,
            zorder=4, edgecolors=SURFACE, linewidths=1.2,
            label=f"shared ({len(sh)})",
        )
        for idx, col in ((idx_a, C_DSPN), (idx_other, C_OTHER)):
            r = pool.receivers[idx]
            ax.scatter(
                [r[0]], [r[1]], s=120, marker="s", color=col, zorder=5,
                edgecolors=SURFACE, linewidths=1.5,
            )
            for radius, ls in ((pool.r_in, ":"), (pool.r_out, (0, (4, 3)))):
                ax.add_patch(
                    plt.Circle(
                        r, radius, fill=False, color=col, lw=1.0,
                        linestyle=ls, alpha=0.7, zorder=2,
                    )
                )
        dist = np.linalg.norm(
            pool.receivers[idx_a] - pool.receivers[idx_other]
        )
        f_val = pool.f_pair(idx_a, idx_other)
        ax.set_title(f"{title}\nd = {dist:.1f},  realised f = {f_val:.3f}")
        ax.set_aspect("equal")
        ax.set_xlim(-6.8, 15.8)
        ax.set_ylim(-6.8, 15.8)
        ax.legend(loc="lower left", fontsize=7, markerscale=1.0)
        ax.tick_params(colors=MUTED, labelsize=7)
        for side in ax.spines.values():
            side.set_color(BASELINE)
    axes[0].set_ylabel("position (toy units)", fontsize=8)
    save_fig(fig, "fig_1b_structure.png")
    return idx_a, idx_b, idx_c


def fig_1b_fd(pool, idx_a, idx_b, idx_c):
    R = pool.receivers.shape[0]
    iu = np.triu_indices(R, 1)
    dists = np.linalg.norm(
        pool.receivers[iu[0]] - pool.receivers[iu[1]], axis=1
    )
    fs = np.array([pool.f_pair(i, j) for i, j in zip(iu[0], iu[1])])
    fig, ax = new_fig(6.8, 3.0)
    ax.scatter(dists, fs, s=6, color=C_DSPN, alpha=0.18, edgecolors="none")
    bins = np.arange(0, dists.max() + 1, 1.0)
    which = np.digitize(dists, bins)
    centers, means = [], []
    for b in np.unique(which):
        sel = which == b
        if sel.sum() >= 5:
            centers.append(dists[sel].mean())
            means.append(fs[sel].mean())
    ax.plot(centers, means, color=INK, lw=2.0, label="mean over pairs")
    for idx_other, lbl in ((idx_b, "near pair"), (idx_c, "far pair")):
        d = np.linalg.norm(pool.receivers[idx_a] - pool.receivers[idx_other])
        f = pool.f_pair(idx_a, idx_other)
        ax.scatter(
            [d], [f], s=70, color=C_SHARED, zorder=5, edgecolors=SURFACE,
            linewidths=1.5,
        )
        ax.annotate(
            lbl, (d, f), textcoords="offset points", xytext=(8, 6),
            fontsize=8, color=C_SHARED,
        )
    ax.set_xlabel("distance between the two receivers (toy units)")
    ax.set_ylabel("realised shared fraction f")
    ax.set_title(
        "f(d) emerges from pool overlap — it is never computed or imposed"
    )
    ax.legend(loc="upper right")
    style_axes(ax)
    save_fig(fig, "fig_1b_fd.png")


def fig_1b_timecourse(pool, idx_a, idx_b):
    T = 240
    p_const = 0.02
    n_src = pool.sources.shape[0]
    rng = np.random.default_rng(23)
    src_spikes = rng.random((n_src, T)) < p_const
    n_events = src_spikes.sum(axis=0)
    c_a = src_spikes[pool.pools[idx_a]].sum(axis=0)
    c_b = src_spikes[pool.pools[idx_b]].sum(axis=0)
    t = np.arange(T)

    fig, axes = new_fig(8.4, 5.4, nrows=3, sharex=True)
    fig.subplots_adjust(hspace=0.6)
    ax_p, ax_e, ax_c = axes
    ax_p.plot(t, np.full(T, p_const), color=INK2, lw=1.8)
    ax_p.set_ylim(0, p_const * 2)
    ax_p.set_ylabel("p(t)")
    ax_p.set_title(
        "step 2 for missing GABA: the drive is a CONSTANT rate "
        "(firing_rate_dict) — no staircase, no time structure"
    )
    style_axes(ax_p)

    ax_e.plot(t, n_events, color=C_SHARED, lw=1.4)
    ax_e.plot(
        t, np.full(T, n_src * p_const), color=MUTED, lw=1.2,
        linestyle=(0, (4, 3)),
    )
    ax_e.set_ylabel("n_events(t)")
    ax_e.set_title(
        "step 3b: n_events(t) ~ Binomial(S·k_mult, p) spikes land on the "
        "whole cloud; each is assigned to one source"
    )
    style_axes(ax_e)

    ax_c.plot(t, c_a, color=C_DSPN, lw=1.3, label="receiver A")
    ax_c.plot(t, c_b, color=C_OTHER, lw=1.3, label="receiver B (near)")
    both = np.intersect1d(pool.pools[idx_a], pool.pools[idx_b])
    hits = src_spikes[both].sum(axis=0) > 0
    ax_c.scatter(
        t[hits], np.maximum(c_a, c_b)[hits] + 0.6, s=14, color=C_SHARED,
        marker="v", label="a shared source fired", zorder=5,
    )
    ax_c.set_ylabel("c_i(t)")
    ax_c.set_xlabel("time bin")
    ax_c.set_title(
        "a spike of a shared source lands in BOTH receivers in the same bin "
        "— that is the correlation"
    )
    ax_c.set_ylim(-0.4, float(np.maximum(c_a, c_b).max()) + 3.2)
    ax_c.legend(loc="upper right", ncols=3)
    style_axes(ax_c)
    save_fig(fig, "fig_1b_timecourse.png")


def run_1b_real_check():
    """Real 3D generator at reduced-but-realistic scale (memory-safe)."""
    rng = np.random.default_rng(3)
    d_mm = lattice_spacing_mm()
    n_side = 5
    xs = np.arange(n_side) * d_mm
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T  # 125 receivers

    P0, sigma_um = REAL["kernels"]["iSPN-dSPN"]
    sigma_mm = sigma_um * 1e-3
    r_in = 0.5 * REAL["lattice_b"] * d_mm / 2  # keep the toy cube fully inside
    r_in = 0.114  # the real Rin (L/2 of the real lattice)
    r_out = 3 * 0.4003  # 3 sigma of the widest kernel onto dSPN (real Rout)
    rho_ispn = 0.43 / 0.886 * REAL["density_per_mm3"]

    pools = build_geometric_source_pools(
        receiver_positions=positions,
        p_func=lambda d: p_exp(d, P0, sigma_mm),
        r_in=r_in,
        r_out=r_out,
        density_pre=rho_ispn,
        rng=rng,
        multiplicity=REAL["source_multiplicity"],
    )
    stats = simulate_receiver_counts_geometric_to_memmap(
        filename=os.path.join(SCRATCH, "toy1b_real.dat"),
        pools=pools,
        rate=REAL["firing_rate_dict"]["iSPN"],
        dt=REAL["dt_ms"],
        num_bins=50_000,
        receiver_dtype=np.int16,
        rng=rng,
    )
    return {
        "pair": "iSPN->dSPN",
        "n_receivers": int(pools.n_receivers),
        "n_sources": int(pools.n_sources),
        "multiplicity": int(pools.multiplicity),
        "mean_n_eff": float(pools.mean_n_eff),
        "r_in_mm": r_in,
        "r_out_mm": r_out,
        "density_pre": rho_ispn,
        "target": stats["target"],
        "measured": stats["measured"],
        "mean_shared_fraction": stats["mean_shared_fraction"],
    }


# ===========================================================================
# Case 1c -- flat split (CorticalInputs)
# ===========================================================================

N_1C, F_1C, R_1C = 10, 0.3, 4


def fig_1c_structure():
    n_shared = int(round(F_1C * N_1C))
    n_private = N_1C - n_shared
    fig, ax = new_fig(7.6, 3.2)
    ax.set_facecolor(SURFACE)
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.4, 4.4)

    # shared pool box
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.6, 1.6), 1.8, 1.2, boxstyle="round,pad=0.12",
            facecolor="#eceafd", edgecolor=C_SHARED, lw=1.5,
        )
    )
    for s in range(n_shared):
        ax.scatter(
            [1.0 + 0.7 * s], [2.2], s=60, color=C_SHARED,
            edgecolors=SURFACE, linewidths=1.2, zorder=5,
        )
    ax.text(
        1.5, 3.05, f"shared pool\nn_shared = round(f·N) = {n_shared}",
        ha="center", fontsize=8, color=C_SHARED,
    )

    ys = np.linspace(0.3, 3.9, R_1C)
    for i, y in enumerate(ys):
        col = C_DSPN if i == 0 else (C_OTHER if i == 1 else MUTED)
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (4.4, y - 0.34), 2.2, 0.68, boxstyle="round,pad=0.08",
                facecolor=SURFACE, edgecolor=col, lw=1.2,
            )
        )
        for s in range(n_private):
            ax.scatter(
                [4.65 + 0.28 * s], [y], s=22, color=col, alpha=0.75,
                edgecolors="none", zorder=5,
            )
        ax.annotate(
            "", xy=(8.2, y), xytext=(6.75, y),
            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4),
        )
        ax.annotate(
            "", xy=(8.2, y), xytext=(2.55, 2.2 + (0 if i == 1 else 0)),
            arrowprops=dict(
                arrowstyle="-|>", color=C_SHARED, lw=1.4, alpha=0.8,
                connectionstyle="arc3,rad=-0.12",
            ),
        )
        ax.scatter(
            [8.55], [y], s=180, marker="s", color=col, zorder=6,
            edgecolors=SURFACE, linewidths=1.5,
        )
        ax.text(
            9.0, y, f"receiver {i + 1}", va="center", fontsize=8, color=INK2
        )
    ax.set_ylim(-0.75, 4.4)
    ax.text(
        5.5, -0.55,
        f"private per receiver: n_private = N − n_shared = {n_private}",
        ha="center", fontsize=8, color=INK2,
    )
    ax.set_title(
        "1c: every receiver = the ONE shared sub-pool + its own private "
        f"sub-pool   (toy: N = {N_1C}, f = {F_1C};  real today: f = 0)"
    )
    save_fig(fig, "fig_1c_structure.png")


def fig_1c_timecourse():
    p_t, _, _ = toy_p_trace(n_trs=6)
    T = p_t.shape[0]
    t = np.arange(T)
    rng = np.random.default_rng(31)
    n_shared = int(round(F_1C * N_1C))
    n_private = N_1C - n_shared
    S = rng.binomial(n_shared, p_t)
    P1 = rng.binomial(n_private, p_t)
    P2 = rng.binomial(n_private, p_t)

    fig, axes = new_fig(8.4, 5.4, nrows=3, sharex=True)
    fig.subplots_adjust(hspace=0.6)
    ax_s, ax_p, ax_c = axes
    ax_s.step(t, S, where="mid", color=C_SHARED, lw=1.4)
    ax_s.set_ylabel("S(t)")
    ax_s.set_title(
        "S(t) ~ Binomial(n_shared, p(t)) — drawn once per bin, added to "
        "EVERY receiver"
    )
    style_axes(ax_s)
    ax_p.step(t, P1, where="mid", color=C_DSPN, lw=1.2, label="P₁(t)")
    ax_p.step(t, P2, where="mid", color=C_OTHER, lw=1.2, label="P₂(t)")
    ax_p.set_ylabel("P_i(t)")
    ax_p.set_title(
        "P_i(t) ~ Binomial(n_private, p(t)) — per receiver, independent"
    )
    ax_p.set_ylim(-0.3, float(max(P1.max(), P2.max())) + 2.6)
    ax_p.legend(loc="upper right", ncols=2)
    style_axes(ax_p)
    ax_c.step(t, S + P1, where="mid", color=C_DSPN, lw=1.2, label="c₁")
    ax_c.step(t, S + P2, where="mid", color=C_OTHER, lw=1.2, label="c₂")
    ax_c.step(
        t, N_1C * p_t, where="post", color=MUTED, lw=1.2,
        linestyle=(0, (4, 3)), label="N·p(t)",
    )
    ax_c.set_ylabel("c_i = S + P_i")
    ax_c.set_xlabel("time bin")
    ax_c.set_title(
        "the sum is exactly Binomial(N, p) per receiver; corr = f because a "
        "fraction f of each count is literally the same draw"
    )
    ax_c.set_ylim(-0.4, float(max((S + P1).max(), (S + P2).max())) + 3.0)
    ax_c.legend(loc="upper right", ncols=3)
    style_axes(ax_c)
    save_fig(fig, "fig_1c_timecourse.png")


def run_1c_real_check():
    stats = simulate_receiver_counts_homogeneous_to_memmap(
        filename=os.path.join(SCRATCH, "toy1c_real.dat"),
        R=R_1C,
        N=N_1C,
        shared_input=F_1C,
        rate=TOY_P_MEAN * 1000.0,
        dt=1.0,
        num_bins=400_000,
        receiver_dtype=np.int16,
        rng=np.random.default_rng(13),
    )
    return {
        "N": N_1C, "f": F_1C, "R": R_1C,
        "target": stats["target"], "measured": stats["measured"],
    }


# ===========================================================================
# Step 2b -- shared rate modulation (currently OFF)
# ===========================================================================

MOD = {"rate_hz": 100.0, "dt": 1.0, "r_sc": 0.05, "tau_c": 50.0,
       "t_meas": 200.0}


def make_modulated_trace(T, rng):
    from scipy.signal import lfilter
    from scipy.stats import gamma as gamma_dist, norm

    p_drive = MOD["rate_hz"] * MOD["dt"] / 1000.0
    a = np.exp(-MOD["dt"] / MOD["tau_c"])
    sigma = solve_modulation_amplitude(
        r_sc=MOD["r_sc"], t_meas_ms=MOD["t_meas"], tau_c_ms=MOD["tau_c"],
        dt=MOD["dt"], p_bar=p_drive,
    )
    xi = rng.standard_normal(T)
    z = lfilter([np.sqrt(1 - a * a)], [1, -a], xi)
    z += float(rng.standard_normal()) * a ** np.arange(1, T + 1)
    u = np.clip(norm.cdf(z), 1e-12, 1 - 1e-12)
    mod = gamma_dist.ppf(u, a=1.0 / sigma**2, scale=sigma**2)
    p_t = np.clip(p_drive * mod, 0.0, 1.0)
    return xi, z, u, mod, p_t, p_drive, sigma, a


def fig_2b_pipeline():
    T = 1200
    xi, z, u, mod, p_t, p_drive, sigma, _ = make_modulated_trace(
        T, np.random.default_rng(5)
    )
    t = np.arange(T)
    fig, axes = new_fig(8.4, 7.2, nrows=5, sharex=True)
    specs = [
        (xi, MUTED, 0.9, "ξ ~ N(0,1)", "white noise, one value per bin"),
        (z, INK, 1.4, "z(t)",
         f"AR(1) filter — memory τ_c = {MOD['tau_c']:.0f} ms turns "
         "noise into a slowly wandering trace"),
        (u, INK2, 1.4, "u = Φ(z)",
         "the normal CDF maps it onto (0, 1) — a correlated uniform"),
        (mod, C_SHARED, 1.4, "Mod(t)",
         f"Gamma.ppf(u): mean 1, variance σ² = {sigma**2:.2f}, "
         "strictly positive"),
        (p_t, C_SHARED, 1.4, "p(t)",
         "p(t) = p_drive · Mod(t) — every presynaptic neuron now "
         "waxes and wanes together"),
    ]
    for ax, (y, col, lw, ylab, title) in zip(axes, specs):
        ax.plot(t, y, color=col, lw=lw)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9)
        style_axes(ax)
    axes[1].axhline(0, color=BASELINE, lw=0.8)
    axes[3].axhline(1, color=MUTED, lw=1.0, linestyle=(0, (4, 3)))
    axes[4].axhline(
        p_drive, color=MUTED, lw=1.2, linestyle=(0, (4, 3)),
    )
    axes[4].text(
        8, p_drive * 0.985, "p_drive (what ships today, r_sc = 0)",
        ha="left", va="top", fontsize=8, color=MUTED,
    )
    axes[4].set_xlabel("time (ms)")
    save_fig(fig, "fig_2b_pipeline.png")
    return sigma


def fig_2b_window(sigma):
    p_drive = MOD["rate_hz"] * MOD["dt"] / 1000.0
    a = np.exp(-MOD["dt"] / MOD["tau_c"])
    windows = np.unique(np.round(np.logspace(0, 3.3, 40)).astype(int))
    analytic = []
    for m in windows:
        w = ou_sum_variance(int(m), a)
        analytic.append(
            p_drive * sigma**2 * w / (m + p_drive * sigma**2 * w)
        )

    # simulate two presynaptic neurons sharing the modulation
    T = 2_000_000
    rng = np.random.default_rng(17)
    _, _, _, _, p_t, _, _, _ = make_modulated_trace(T, rng)
    s1 = rng.random(T) < p_t
    s2 = rng.random(T) < p_t
    meas_w = [5, 20, 100, 200, 500, 1000]
    measured = []
    for m in meas_w:
        n = (T // m) * m
        a1 = s1[:n].reshape(-1, m).sum(axis=1)
        a2 = s2[:n].reshape(-1, m).sum(axis=1)
        measured.append(np.corrcoef(a1, a2)[0, 1])

    fig, ax = new_fig(6.8, 3.1)
    ax.semilogx(
        windows * MOD["dt"], analytic, color=INK, lw=2.0, label="formula"
    )
    ax.scatter(
        np.array(meas_w) * MOD["dt"], measured, s=64, color=C_SHARED,
        zorder=5, edgecolors=SURFACE, linewidths=1.5,
        label="two simulated neurons",
    )
    ax.axvline(MOD["t_meas"], color=GRID, lw=1.4)
    ax.scatter(
        [MOD["t_meas"]], [MOD["r_sc"]], s=90, color=C_SHARED, marker="D",
        zorder=6, edgecolors=SURFACE, linewidths=1.5,
    )
    ax.annotate(
        f"calibration point:\nr_sc = {MOD['r_sc']} at T_meas = "
        f"{MOD['t_meas']:.0f} ms",
        (MOD["t_meas"], MOD["r_sc"]), textcoords="offset points",
        xytext=(10, -26), fontsize=8, color=C_SHARED,
    )
    ax.set_xlabel("counting window T (ms)")
    ax.set_ylabel("spike-count correlation r_sc(T)")
    ax.set_title(
        "a spike-count correlation is meaningless without its window: it "
        "grows with T and saturates past τ_c"
    )
    ax.legend(loc="upper left")
    style_axes(ax)
    save_fig(fig, "fig_2b_window.png")


# ===========================================================================


def main():
    print("section 0: stream matrix")
    fig_stream_matrix()

    print("case 1a")
    membership = build_membership_1a()
    shared_ab = fig_1a_structure(membership)
    fig_1a_timecourse(membership)
    eq = run_1a_equivalence(membership)
    fig_1a_equivalence(eq)

    print("case 1b")
    fig_1b_kernels()
    pool = Toy2DPool(np.random.default_rng(19))
    idx_a, idx_b, idx_c = fig_1b_structure(pool)
    fig_1b_fd(pool, idx_a, idx_b, idx_c)
    fig_1b_timecourse(pool, idx_a, idx_b)
    real_1b = run_1b_real_check()

    print("case 1c")
    fig_1c_structure()
    fig_1c_timecourse()
    real_1c = run_1c_real_check()

    print("step 2b")
    sigma = fig_2b_pipeline()
    fig_2b_window(sigma)

    dump_results(
        {
            "toy_1a": {
                "M": M_TOY,
                "receivers": [
                    {"type": ct, "label": lbl, "N": n}
                    for ct, lbl, n in RECEIVERS_1A
                ],
                "shared_AB": int(len(shared_ab)),
                "equivalence": eq,
            },
            "real_check_1b": real_1b,
            "real_check_1c": real_1c,
            "mod_2b": {**MOD, "sigma": sigma, "sigma2": sigma**2},
        }
    )
    shutil.rmtree(SCRATCH, ignore_errors=True)
    print("done")


if __name__ == "__main__":
    main()
