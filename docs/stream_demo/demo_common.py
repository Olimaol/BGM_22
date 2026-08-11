"""Shared style, colors and real-model constants for the stream demo.

Snapshot 2026-08-11 -- explains the generators in
CompNeuroPy/striatal_microcircuit/spike_input_cortex.py as of this date.
Not a living document; rerun make_figures.py if it ever drifts.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Palette (dataviz reference palette, light mode; validated 2026-08-11)
# Color follows the entity everywhere:
#   dSPN = blue, iSPN = orange, FS = aqua, shared/pool-level = violet
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

C_DSPN = "#2a78d6"  # blue
C_ISPN = "#eb6834"  # orange
C_FS = "#1baf7a"  # aqua (low contrast: always direct-label FS marks)
C_SHARED = "#4a3aa7"  # violet: shared axons / k(t) / S(t) / modulation
C_TYPE = {"dSPN": C_DSPN, "iSPN": C_ISPN, "FS": C_FS}

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(DEMO_DIR, "figures")
RESULTS_PATH = os.path.join(DEMO_DIR, "results.json")

# scratch space for memmaps the real generators write; session scratchpad if
# present, else a git-ignored subfolder here
SCRATCH = os.environ.get(
    "STREAM_DEMO_SCRATCH", os.path.join(DEMO_DIR, ".scratch")
)


def style_axes(ax, grid_axis="y"):
    """Recessive chart chrome: hairline grid, no top/right spines."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, width=0.8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK2)
    if grid_axis:
        ax.grid(
            True, axis=grid_axis, color=GRID, linewidth=0.8, linestyle="-"
        )
        ax.set_axisbelow(True)


def new_fig(width, height, nrows=1, ncols=1, **kw):
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(width, height), facecolor=SURFACE, **kw
    )
    return fig, axes


def save_fig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, DEMO_DIR)}")


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "semibold",
        "axes.titlecolor": INK,
        "axes.labelsize": 9,
        "axes.labelcolor": INK2,
        "axes.edgecolor": BASELINE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "lines.linewidth": 1.8,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "legend.frameon": False,
        "legend.fontsize": 8,
    }
)

# ---------------------------------------------------------------------------
# Real-model constants (v07, caudate loop, DBS off) for the toy-vs-real tables
# and the benchmarks. Sources: BOLD_optimization/parameters.py,
# Microcircuit.__init__ (props_delRey, density, nx=10, b=10),
# connectivity_fit_data/fitted_params.json, model_v07.md section 7.
# ---------------------------------------------------------------------------
REAL = {
    "dt_ms": 0.1,
    "tr_ms": 2310.0,
    "bins_per_tr": 23100,
    "n_trs_full": 310,
    "shared_fraction": 0.014,  # Kincaid, SPN-SPN
    "N_cortical_inputs": {"FS": 2800, "dSPN": 7000, "iSPN": 7000},
    # 1000-neuron lattice split by props_delRey [0.026, 0.43, 0.43]/0.886
    "type_counts": {"FS": 29, "dSPN": 486, "iSPN": 485},
    "density_per_mm3": 84900.0,
    "lattice_nx": 10,
    "lattice_b": 10,
    "caudate_proportions": {
        "dlPFC": 0.55, "preSMA": 0.15, "PMd": 0.18, "PMv": 0.04,
        "SMA": 0.06, "M1": 0.02, "S1": 0.00,
    },
    # missing-GABA rates (medication-off, Liang et al. 2008 / parameters.py)
    "firing_rate_dict": {"FS": 10.5, "dSPN": 25.0, "iSPN": 33.0},
    "source_multiplicity": 10,
    # fitted kernels: pre-post -> (P0, sigma_um); p(d) = P0*exp(-d^2/sigma^2)
    "kernels": {
        "dSPN-dSPN": (0.10837817679112945, 400.3325717568318),
        "dSPN-iSPN": (0.10320990783452937, 400.9140054701368),
        "FS-dSPN": (0.5987673121691023, 394.2162098831489),
        "FS-FS": (0.1524181937224343, 189.16385747655912),
        "FS-iSPN": (0.9181986396346192, 139.9954921420855),
        "iSPN-dSPN": (0.1406061585964347, 255.35377976845913),
        "iSPN-iSPN": (0.1384660121789243, 313.6847967516534),
    },
    # CorticalInputs (1c)
    "ci_N_total": {"thal": 1000, "gpe_arky": 500, "gpe_cp": 500, "stn": 500},
    "ci_pop_size": 100,
    "ci_shared_fraction": 0.0,
}

RATE_NPZ = os.path.join(
    DEMO_DIR,
    "..", "..", "striatal_microcircuit_requirements", "cortical_firing_rates",
    "cortical_firing_rates_data", "firing_rates_matlab_condition-off.npz",
)


def load_rate_series(key="dlPFC_rate", n_trs=None):
    """One value per TR, mean-5-Hz normalised series from the real npz."""
    with np.load(RATE_NPZ) as data:
        series = np.asarray(data[key], dtype=np.float64)
    if n_trs is not None:
        series = series[:n_trs]
    return series


def lattice_spacing_mm():
    return (1.0 / REAL["density_per_mm3"]) ** (1.0 / 3.0)


def p_exp(d, P0, sigma):
    """The model's own distance kernel (Microcircuit._p_exp)."""
    return P0 * np.exp(-(d**2) / (sigma**2))


def dump_results(payload):
    existing = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            existing = json.load(f)
    existing.update(payload)
    with open(RESULTS_PATH, "w") as f:
        json.dump(existing, f, indent=1, default=float)
    print(f"  updated {os.path.relpath(RESULTS_PATH, DEMO_DIR)}")
