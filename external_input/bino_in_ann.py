"""
Created this file for testing predefine spikes
"""

from ANNarchy import (
    Neuron,
    PoissonPopulation,
    Network,
    TimedArray,
    CurrentInjection,
)
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp, levene


receiving_neuron = Neuron(
    parameters="""
        tau_gaba = 10.0
    """,
    equations="""
        dv/dt = 0
        dg_ampa/dt = -g_ampa / tau_gaba
    """,
    spike="""
        v>1
    """,
)

self_spiking_neuron = Neuron(
    parameters="""
        tau_gaba = 10.0
        weight = 1
        p = 0.1 : population
        N = 1000 : population
    """,
    equations="""
        dv/dt = 0
        spikes = Binomial(N, p)
        dg_ampa/dt = -g_ampa / tau_gaba + weight*spikes
    """,
    spike="""
        v>1
    """,
)


def poisson_network(N_pre=1000, N_post=1, rate=100.0, weight=0.01):
    net = Network()
    poisson_pop = net.create(
        PoissonPopulation(geometry=N_pre * N_post, name="PoissonInput", rates=rate)
    )

    receiving_pop = net.create(
        geometry=N_post, neuron=receiving_neuron, name="ReceivingPopulation"
    )
    for post_idx in range(N_post):
        start_idx = post_idx * N_pre
        end_idx = start_idx + N_pre
        proj = net.connect(
            pre=poisson_pop[start_idx:end_idx],
            post=receiving_pop[post_idx],
            target="ampa",
            name=f"PoissonToReceiving_{post_idx}",
        )
        proj.all_to_all(weights=weight)
    monitor = net.monitor(receiving_pop, variables=["g_ampa"], start=False)
    net.compile("bino_in_ann_poisson_network")
    print("Starting simulation of Poisson network...")
    net.simulate(1000)
    monitor.start()
    net.simulate(9000, measure_time=True)
    data = monitor.get("g_ampa")
    return data


def self_spiking_network(N_pre=1000, N_post=1, rate=100.0, weight=0.01):
    net = Network()
    self_spiking_pop = net.create(
        geometry=N_post, neuron=self_spiking_neuron, name="SelfSpikingPopulation"
    )
    self_spiking_pop.weight = weight
    self_spiking_pop.N = N_pre
    self_spiking_pop.p = float(rate) / 1000.0
    monitor = net.monitor(self_spiking_pop, variables=["g_ampa"], start=False)
    net.compile("bino_in_ann_self_spiking_network")
    print("Starting simulation of Self-Spiking network...")
    net.simulate(1000)
    monitor.start()
    net.simulate(9000, measure_time=True)
    data = monitor.get("g_ampa")
    return data


def store_spike_counts_to_disk(
    N_pre,
    N_post,
    rate,
    num_bins,
    rng: np.random.Generator,
    dtype=np.int64,
    chunk_size=1000,
    filename="big_array.dat",
):
    nrows, ncols = (num_bins, N_post)

    def get_inp_arr(nchunk):
        return rng.binomial(
            n=N_pre, p=rate / 1000.0, size=(nchunk, ncols)
        )  # size equals (timesteps, neurons), p: firing rate in Hz with dt=1 ms (r/1000)

    # preallocate memmap on disk
    mm = np.memmap(filename, dtype=dtype, mode="w+", shape=(nrows, ncols))

    # loop over chunks
    for start in range(0, nrows, chunk_size):
        end = min(start + chunk_size, nrows)
        nchunk = end - start
        chunk = get_inp_arr(nchunk)  # returns shape (nchunk, ncols)
        mm[start:end, :] = chunk  # write straight into memmap
        del chunk  # free local memory

    # flush to disk
    mm.flush()


def iter_memmap_spike_counts(
    filename, num_bins, N_post, dtype=np.int64, chunk_size=1000, copy=False
):
    nrows, ncols = (num_bins, N_post)
    mm = np.memmap(filename, dtype=dtype, mode="r", shape=(nrows, ncols))
    for start in range(0, nrows, chunk_size):
        end = min(start + chunk_size, nrows)
        chunk = mm[start:end, :]
        yield chunk.copy() if copy else chunk


def pre_defined_spiking_network(N_pre=1000, N_post=1, rate=100.0, weight=0.01):
    net = Network()
    rng = np.random.default_rng(seed=42)

    # store inputs on hard drive (prevents excessive memory usage)
    store_spike_counts_to_disk(
        N_pre=N_pre,
        N_post=N_post,
        rate=rate,
        num_bins=10000,
        rng=rng,
        dtype=np.int64,
        chunk_size=1000,
        filename="big_array.dat",
    )

    # iterator for loading data chunks from disk
    inp_iterator = iter_memmap_spike_counts(
        filename="big_array.dat",
        num_bins=10000,
        N_post=N_post,
        dtype=np.int64,
        chunk_size=1000,
        copy=False,
    )

    # first 1000 ms of inputs
    inputs = next(inp_iterator) * weight
    print(f"initial input data: {inputs}")
    # list for tracking all inputs
    all_inputs = [inputs.copy()]

    inp = net.create(
        TimedArray(rates=inputs, period=inputs.shape[0], name="TimedInput")
    )
    pop = net.create(
        geometry=N_post, neuron=receiving_neuron, name="PreDefinedReceivingPopulation"
    )
    proj = net.connect(CurrentInjection(inp, pop, "ampa"))
    proj.connect_current()
    monitor1 = net.monitor(pop, variables=["g_ampa"], start=False)
    monitor2 = net.monitor(inp, variables=["r"], start=False)
    net.compile("bino_in_ann_pre_defined_spiking_network")
    print("Starting simulation of Pre-Defined Spiking network...")
    net.simulate(1000, measure_time=True)
    monitor1.start()
    monitor2.start()
    # loop over data chunks
    for inputs in inp_iterator:
        inputs = inputs * weight
        all_inputs.append(inputs.copy())
        inp.update(rates=inputs, period=inputs.shape[0])
        net.simulate(inputs.shape[0], measure_time=True)
    data = monitor1.get("g_ampa")
    data_r = monitor2.get("r")
    # create full input data array
    all_inputs_arr = np.vstack(all_inputs)
    print(f"Pre-defined input data shape: {all_inputs_arr.shape}\n")
    chunk = 0
    print(f"Chunk {chunk}\n data:     {all_inputs[chunk][:10,0]}\n")
    for chunk in range(1, len(all_inputs)):
        print(
            f"Chunk {chunk}\n data:     {all_inputs[chunk][:10,0]}\n recorded: {data_r[(chunk-1)*1000:(chunk-1)*1000+10,0]}\n"
        )
    return data, data_r


if __name__ == "__main__":
    MAKE_HUGE_TEST = False

    if MAKE_HUGE_TEST == True:
        # First run with multiple receiving neurons
        print("#########################################################")
        print(
            "Running networks with 50 post neurons each receiving from 1000 Poisson inputs..."
        )
        data_poisson = poisson_network(N_post=50)
        data_self_spiking = self_spiking_network(N_post=50)
        data_pre_defined, _ = pre_defined_spiking_network(N_post=50)

        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.title("Poisson Network")
        plt.plot(data_poisson)
        plt.subplot(3, 1, 2)
        plt.title("Self-Spiking Network")
        plt.plot(data_self_spiking)
        plt.subplot(3, 1, 3)
        plt.title("Pre-Defined Network")
        plt.plot(data_pre_defined)
        plt.tight_layout()
        plt.show()

    # Run all three for single receiving neuron and do analyses
    print("#########################################################")
    print("Running networks with 1 post neuron receiving from 1000 Poisson inputs...")
    data_poisson = poisson_network(rate=50)
    data_self_spiking = self_spiking_network(rate=50)
    data_pre_defined, _ = pre_defined_spiking_network(rate=50)

    # Flatten to 1D arrays to be agnostic to (T, 1) vs (T,) shapes
    x_pois = np.ravel(data_poisson)
    x_self = np.ravel(data_self_spiking)
    x_pred = np.ravel(data_pre_defined)

    datasets = {
        "Poisson": x_pois,
        "Self-spiking": x_self,
        "Pre-defined": x_pred,
    }

    # Helper for ECDF
    def ecdf(x):
        x = np.asarray(x)
        x = x[~np.isnan(x)]
        x = np.sort(x)
        y = np.arange(1, x.size + 1) / x.size if x.size else np.array([0.0])
        return x, y

    # NumPy-only KS statistic fallback (no p-value)
    def ks_statistic_only(x, y):
        x = np.sort(np.asarray(x))
        y = np.sort(np.asarray(y))
        # Evaluate ECDFs on the combined sorted sample points
        z = np.concatenate([x, y])
        zx = np.searchsorted(x, z, side="right") / (x.size if x.size else 1)
        zy = np.searchsorted(y, z, side="right") / (y.size if y.size else 1)
        return float(np.max(np.abs(zx - zy)))

    # Build figure with: time series, histogram overlay, ECDF overlay, and a stats box
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 2])
    ax_ts = fig.add_subplot(gs[0, :])
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_ecdf = fig.add_subplot(gs[1, 1])

    # 1) Time courses (overlay)
    colors = {
        "Poisson": "tab:blue",
        "Self-spiking": "tab:orange",
        "Pre-defined": "tab:green",
    }

    for name, x in datasets.items():
        ax_ts.plot(x, label=name, lw=1.0, alpha=0.9, color=colors.get(name))
    ax_ts.set_title("g_ampa time courses")
    ax_ts.set_xlabel("Time (ms)")
    ax_ts.set_ylabel("g_ampa")
    ax_ts.legend(loc="upper right")
    ax_ts.grid(True, alpha=0.2)

    # 2) Histograms (overlay) with common bins
    combined = np.concatenate(list(datasets.values()))
    # Guard against all-NaN or empty
    if combined.size == 0 or np.all(~np.isfinite(combined)):
        bins = 10
    else:
        finite = combined[np.isfinite(combined)]
        bins = np.histogram_bin_edges(finite, bins=50)

    for name, x in datasets.items():
        ax_hist.hist(
            x,
            bins=bins,
            density=True,
            alpha=0.45,
            label=name,
            color=colors.get(name),
            edgecolor="none",
        )
    ax_hist.set_title("Distribution of g_ampa (histogram)")
    ax_hist.set_xlabel("g_ampa")
    ax_hist.set_ylabel("Density")
    ax_hist.legend(loc="upper right")
    ax_hist.grid(True, alpha=0.2)

    # 3) ECDF overlay (useful to visually assess KS-like differences)
    for name, x in datasets.items():
        xs, ys = ecdf(x)
        if xs.size:
            ax_ecdf.step(xs, ys, where="post", label=name, color=colors.get(name))
    ax_ecdf.set_title("Empirical CDF of g_ampa")
    ax_ecdf.set_xlabel("g_ampa")
    ax_ecdf.set_ylabel("ECDF")
    ax_ecdf.legend(loc="lower right")
    ax_ecdf.grid(True, alpha=0.2)

    # 4) Statistical tests: pairwise KS and variance equality; also report mean±std
    def fmt_mean_std(x):
        x = np.asarray(x)
        mu = float(np.nanmean(x)) if x.size else float("nan")
        sd = float(np.nanstd(x, ddof=1)) if x.size > 1 else float("nan")
        return mu, sd

    means_stds = {name: fmt_mean_std(x) for name, x in datasets.items()}

    pairs = [
        ("Poisson", "Self-spiking"),
        ("Poisson", "Pre-defined"),
        ("Self-spiking", "Pre-defined"),
    ]

    ks_results = {}
    var_results = {}

    for a, b in pairs:
        xa, xb = datasets[a], datasets[b]
        ks_stat, ks_p = ks_2samp(xa, xb, alternative="two-sided", mode="auto")
        var_stat, var_p = levene(xa, xb, center="median")
        ks_results[(a, b)] = (float(ks_stat), float(ks_p))
        var_results[(a, b)] = (float(var_stat), float(var_p))

    # Compose text for stats box that will be positioned to the right of the subplots
    text_lines = []
    text_lines.append("Means ± SD:")
    for name, (mu, sd) in means_stds.items():
        text_lines.append(f"  {name}: {mu:.4g} ± {sd:.4g}")
    text_lines.append("")
    text_lines.append("Pairwise KS (stat, p):")
    for (a, b), (stat, p) in ks_results.items():
        if np.isnan(p):
            text_lines.append(f"  {a} vs {b}: D={stat:.4g}")
        else:
            text_lines.append(f"  {a} vs {b}: D={stat:.4g}, p={p:.2g}")
    text_lines.append("")
    text_lines.append("Levene equal-variance (stat, p):")
    for (a, b), (stat, p) in var_results.items():
        text_lines.append(f"  {a} vs {b}: {stat:.4g}, p={p:.2g}")

    stats_text = "\n".join(text_lines)

    fig.suptitle(
        "g_ampa comparison: time series, histograms, ECDFs, and stats",
        y=0.98,
        fontsize=14,
    )

    # Reserve space on the right for the stats box and place it there
    fig.tight_layout(rect=[0, 0, 0.75, 0.96])
    fig.text(
        0.78,
        0.5,
        stats_text,
        ha="left",
        va="center",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.8"),
    )
    plt.show()
