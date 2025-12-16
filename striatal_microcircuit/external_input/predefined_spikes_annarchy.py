"""
Created this file for testing predefine spikes
"""

from ANNarchy import (
    Neuron,
    TimedArray,
    CurrentInjection,
    Population,
    Monitor,
    compile,
    simulate,
    setup,
)
import matplotlib.pyplot as plt
import numpy as np


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


def store_spike_counts_to_disk(
    N_pre,
    N_post,
    rate,
    dt,
    num_bins,
    rng: np.random.Generator,
    dtype=np.int64,
    chunk_size=1000,
    filename="big_array.dat",
):
    nrows, ncols = (num_bins, N_post)

    def get_inp_arr(nchunk):
        return rng.binomial(
            n=N_pre, p=rate / (1000.0 / dt), size=(nchunk, ncols)
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


def pre_defined_spiking_network(N_pre=1000, N_post=1, rate=100.0, weight=0.01, dt=1.0):
    rng = np.random.default_rng(seed=42)
    setup(dt=dt)

    chunk_size = 1000  # number of time steps per chunk
    total_steps = 10000  # total number of time steps

    # store inputs on hard drive (prevents excessive memory usage)
    store_spike_counts_to_disk(
        N_pre=N_pre,
        N_post=N_post,
        rate=rate,
        dt=dt,
        num_bins=total_steps,
        rng=rng,
        dtype=np.int64,
        chunk_size=chunk_size,
        filename="big_array.dat",
    )

    # iterator for loading data chunks from disk
    inp_iterator = iter_memmap_spike_counts(
        filename="big_array.dat",
        num_bins=total_steps,
        N_post=N_post,
        dtype=np.int64,
        chunk_size=chunk_size,
        copy=False,
    )
    # list for tracking all inputs
    rates = np.zeros((chunk_size, N_post)) * np.arange(chunk_size)[:, None]
    all_inputs = [rates]

    inp = TimedArray(
        rates=rates,
        name="TimedInput",
    )

    print(f"schedule of input:\n {inp.schedule}\n")
    print(f"rates of input:\n {inp.rates.shape}\n")
    print(f"period of input:\n {inp.period}\n")
    pop = Population(
        geometry=N_post, neuron=receiving_neuron, name="PreDefinedReceivingPopulation"
    )
    proj = CurrentInjection(inp, pop, "ampa")
    proj.connect_current()
    monitor1 = Monitor(pop, variables=["g_ampa"], start=True)
    monitor2 = Monitor(inp, variables=["r"], start=True)
    compile("bino_in_ann_pre_defined_spiking_network")

    # set schedule and period in c by my own
    schedule = dt
    value = [float(schedule * i) for i in range(rates.shape[0])]
    val_int = np.rint(np.atleast_1d(value) / dt).astype(np.int64)
    inp.cyInstance.set_schedule(val_int)
    value = -1
    period_steps = int(np.rint(value / dt))
    inp.cyInstance.set_period(period_steps)

    print("Starting simulation of Pre-Defined Spiking network...")
    simulate(chunk_size * dt, measure_time=True)
    monitor1.start()
    monitor2.start()
    # loop over data chunks
    n = 0
    for inputs in inp_iterator:
        inputs = inputs * weight
        if n < 5:
            inp.reset()
            inp.update(rates=inputs)

            # set schedule and period in c by my own
            schedule = dt
            value = [float(schedule * i) for i in range(rates.shape[0])]
            val_int = np.rint(np.atleast_1d(value) / dt).astype(np.int64)
            inp.cyInstance.set_schedule(val_int)
            value = -1
            period_steps = int(np.rint(value / dt))
            inp.cyInstance.set_period(period_steps)

        all_inputs.append(inputs.copy())
        simulate(chunk_size * dt, measure_time=True)
        n += 1
    data = monitor1.get("g_ampa")
    data_r = monitor2.get("r")
    # create full input data array
    all_inputs_arr = np.vstack(all_inputs)
    print(f"Pre-defined input data shape: {all_inputs_arr.shape}\n")
    for chunk in range(len(all_inputs)):
        print(
            f"Chunk {chunk}\n data:     {all_inputs[chunk][:10,0]}\n recorded: {data_r[(chunk)*chunk_size:(chunk)*chunk_size+10,0]}\n"
        )
    return data, data_r, all_inputs_arr


if __name__ == "__main__":

    # Run all three for single receiving neuron and do analyses
    print("#########################################################")
    print("Running networks with 1 post neuron receiving from 1000 Poisson inputs...")
    dt = 0.1
    data_pre_defined, data_r, data_inputs = pre_defined_spiking_network(rate=50, dt=dt)

    # Flatten to 1D arrays to be agnostic to (T, 1) vs (T,) shapes
    x_pred = np.ravel(data_pre_defined)
    r = np.ravel(data_r)
    inputs = np.ravel(data_inputs)

    datasets = {
        "Pre-defined": x_pred,
        "Input rates": r,
        "Inputs": inputs,
    }
    # Plot each dataset in its own stacked subplot
    fig, axes = plt.subplots(len(datasets), 1, figsize=(10, 6), sharex=True)
    axes = np.atleast_1d(axes)
    time_axis = np.arange(len(next(iter(datasets.values())))) * dt

    for ax, (label, values) in zip(axes, datasets.items()):
        ax.plot(time_axis, values)
        ax.set_ylabel(label)
        ax.grid(True, linestyle="--", alpha=0.3)

    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle("Pre-defined input and recorded rates")
    fig.tight_layout()
    plt.show()
