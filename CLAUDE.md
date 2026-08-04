# BGM_22

Spiking basal ganglia model in ANNarchy, simulated in resting state, producing
simulated BOLD that is fitted to one subject's experimental BOLD under DBS on and
off. The scientific goal is inference: fit DBS-off, refit DBS-on, and read off
which parameters had to change — i.e. what DBS did inside the basal ganglia.

Read `PLAN.md` for the current plan and where the work stands, and `TODO.md` for
findings we deliberately postponed.

## Three repos, one environment

| repo | path | install | branch |
|------|------|---------|--------|
| BGM_22 (this) | `~/Dokumente/Code/BGM_22` | — | `olimaol_develop` |
| CompNeuroPy | `~/Dokumente/Code/CompNeuroPy` | **editable** | `olimaol_develop` |
| ANNarchy (patched) | `~/Dokumente/Code/ANNarchy_compneuro` | **regular** | `timedarray-fastbuffer` |

Environment: `/home/oliver/miniforge3/envs/compneuro` (Python 3.11.14, ANNarchy 4.7.3b).

**There is no `python` on PATH.** Always call the interpreter explicitly:

```bash
/home/oliver/miniforge3/envs/compneuro/bin/python ...
```

CompNeuroPy is editable, so changes to it take effect immediately. **ANNarchy is
not** — after editing `ANNarchy_compneuro` you must reinstall, and existing
ANNarchy compile folders become stale and have to be deleted so the changed C++
templates are regenerated:

```bash
cd ~/Dokumente/Code/ANNarchy_compneuro && \
  /home/oliver/miniforge3/envs/compneuro/bin/pip install . --no-build-isolation
```

Do not import ANNarchy with the shell sitting in the ANNarchy source directory —
the source tree shadows site-packages and you will silently test the wrong build.

Other clones exist and are **not** what this project uses: `~/Dokumente/Code/ANNarchy`
is the ANNarchy 5.0 line (525 commits ahead, nanobind), kept only because Oliver's
upstream `timedarray` fixes live there.

## Two model versions

Both are defined in
`CompNeuroPy/src/CompNeuroPy/full_models/bgm_22/model_creation_functions.py`, with
parameters in the `BGM_v07_p01` / `BGM_v08_p01` columns of `parameters.csv`.

- **v07 — the real model.** Striatum built by `Microcircuit` (3D lattice, 1000
  neurons, distance-dependent connectivity fitted from data, correlated cortical
  input, missing-GABA compensation). `CorticalInputs` drives thal/gpe_arky/gpe_cp/stn.
  Both stream **precomputed spike counts** from disk caches via `TimedArray`s that a
  Python `update()` refreshes every `update_time` (100 ms). 28 projections.
  Striatal populations are named `caudate_dSPN`, `putamen_FS`, … (created by
  `Microcircuit`, so no `:loop` appendix).
- **v08 — a reduced fallback** written under time pressure in Dec 2025. Plain
  100-neuron populations, random connectivity, and the entire cortical drive
  collapsed into one `TimedArray` per loop scaled by `exp_input_weight`. 35
  projections (adds the intra-striatal ones that v07 keeps inside the microcircuit).
  Populations are `str_d1:caudate`, … Kept **only** as a fast end-to-end smoke test
  of the optimization pipeline — it needs no caches and runs in minutes.

Two independent BG loops, `caudate` and `putamen`, are simulated side by side.
They share no projections; they meet only when pooled into the shared GPi/GPe/STN
BOLD monitors. Only the putamen loop is stimulated by DBS.

## Data

- Experimental BOLD: `experimental_data/berlin_data/bold_data_roi/sub-01/sub-01_subdiv_results.h5`
  — 310 TRs at TR = 2.31 s, 19 ROIs, conditions `on`/`off`. The simulated regions
  are GPi, GPe, STN, Cau, Put, MD, VAp.
- Cortical drive:
  `striatal_microcircuit_requirements/cortical_firing_rates/cortical_firing_rates_data/firing_rates_matlab_condition-{on,off}.npz`
  — cortical BOLD deconvolved with an SPM HRF into firing rates, one value per TR,
  plus `caudate_rate`/`putamen_rate` mixed by anatomical proportion. Requires MATLAB
  (`matlabengine`) to regenerate.
- Full run length: 310 TRs x 2.31 s = **716,100 ms**.

## Input caches (v07 only)

`Microcircuit` and `CorticalInputs` write raw memmaps to
`<storage_dir>/inputs/receiver_counts_<pre>_<post>.dat`, under
`{mc,ci}_{caudate,putamen}_cache_{off,on}/`.

Build them with `build_input_caches.py`, which takes its settings from
`get_loss.v07_model_creation_kwargs` so a cache always matches what an evaluation
will demand:

```bash
python build_input_caches.py --dbs off --n-trs 5 --cache-dir <abs path>
```

**A cache is only loadable if its `n_steps` equals `int(t.duration/dt)` exactly**,
so it is built for one `--n-trs` and usable only at that `--n-trs`. It must also
cover the fixed 9900 ms firing-rate probe, which is the binding constraint below 5
TRs; the script refuses that case up front.

Two caches exist: `mc_ci_cache/` (12,000 steps, 1.2 s — the old short tests, too
short for anything) and `mc_ci_cache_5tr/` (115,500 steps, 21 GB), which is what
`--n-trs 5` smoke tests use. Cost on the laptop: **24 min per loop for 5 TRs**, so
~25 h per loop at full length. Size at full length would be ~1.25 TiB per DBS
condition in the current `float64`/per-region layout — see `PLAN.md` for the agreed
smaller one.

`storage_dir` is resolved **relative to the working directory you launch from**,
which is why both `mc_ci_cache/` and `mc_caudate_off_cache/` exist with overlapping
data. Pass absolute paths — but note the cortical rate path inside the cache state
is compared verbatim, so build and evaluate from `BOLD_optimization/` either way
(`TODO.md` §13).

## Running things

Everything below is run from `BOLD_optimization/`.

```bash
# compile only (per-individual folder, so parallel jobs do not race)
python get_loss.py --compile --dbs off --model-version v08 --compile-appendix test <params>

# one evaluation; --n-trs shortens the run for smoke tests
python get_loss.py --dbs off --model-version v08 --n-trs 20 <params>

# a v07 evaluation against the short cache, with the gate disabled
python get_loss.py --dbs off --model-version v07 --n-trs 5 \
  --cache-dir <abs path>/mc_ci_cache_5tr --gate-threshold 1.0 <19 params>

# the optimization
python deap_cma_opt.py --dbs off --model-version v07 --optimization-run 1
python deap_cma_opt.py --dbs off --model-version v07 --optimization-run 1 --resume
```

Base parameter count is 19 for v07 and 21 for v08 (see `n_opt_params`). The full
vector layouts are defined by `get_loss.split_param_list`: off takes the base
vector (3 trailing DBS slots tolerated and ignored); on takes base + 3 DBS, or the
staged base + one putamen weight scaling per cluster + 3 DBS. The DBS parameters
are always the **last three** — they used to be read at fixed indices 21-23, which
only lined up with v08. `--dbs` is required on all three scripts on purpose.

A rate probe gates the BOLD run: above `firing_rate_gate` (0.5) BOLD is skipped and
charged 1.0, since both loss terms are in [0, 1]. `--gate-threshold 1.0` disables it.
The threshold is not yet calibrated (`TODO.md` §10).

The loss file `data_BOLD_optimization/loss_<appendix>.json` records the loss
components, per-region BOLD correlations, all firing rates, BOLD sample counts, the
parameter vector and whether the gate fired — not just the total.

`deap_cma_opt.py` runs its own subprocesses rather than CompNeuroPy's
`run_script_parallel`: one log per individual under
`data_BOLD_optimization/individual_logs/<run tag>/`, exit codes kept, a dead
individual penalised with loss 10.0 instead of killing the generation, an abort past
half a generation, and a CMA-ES checkpoint per generation for `--resume`.

## Gotchas

- **ANNarchy resolves `compile(directory=...)` relative to the cwd**, so an absolute
  path gets joined onto it and fails. Pass a relative name and `chdir` first.
- `CurrentInjection` requires a **spiking** post-synaptic population.
- `BoldMonitor` records every `dt` by default. Set
  `bold_monitor._monitor.period = TR_S * 1000.0` **after compile**, or a full run
  stores millions of samples per region and several GB per process.
- Random distributions in ANNarchy equations only accept **global** parameters —
  `Binomial(N, p)` with a variable `p` fails at parse time. There are no
  inverse-CDF functions either.
- `OMP_NUM_THREADS=4` is set in Oliver's shell profile on the workstations.
  ANNarchy ignores it (it sets its own count, default 1) but **numpy's OpenBLAS does
  not**. `nproc` reports 4 there for this reason; the machines really have 40/48
  logical cores.
- **v07 simulates in `update_time` chunks**, because its inputs are handed to
  ANNarchy one chunk at a time. Every simulated stretch — the ramp-up, the rest of
  the run, the firing-rate probe — must be a whole number of chunks or
  `simulate_model` raises. `update_time` is 110 ms because it divides the TR
  (2310 ms, 21 chunks), the full run and the 9900 ms probe. The old 100 ms divided
  none of them, which is why the ramp-up would have failed on the first call.
- The cortical drive has one value per TR, coarser than `dt`. `Microcircuit` and
  `CorticalInputs` repeat each value `TR/dt` times; a drive finer than `dt`, or one
  whose spacing is not an integer multiple, is rejected.
- The striatal `exp_input_weight` bounds are orders of magnitude too wide in **v08**
  (see `TODO.md` §1) — do not start a real fit before resolving that. v07's drive
  weight is better behaved but its bounds are still provisional (`TODO.md` §9).
- Never make assumptions about ANNarchy, CompNeuroPy, and BGM_22 code purely from memory. Always verify your findings against the codebase.

## Workstations

`hinton` (Xeon Gold 6248, 20 physical / 40 logical cores, 125 GB RAM) and `waikiki`
(EPYC 7352, 24 / 48, 251 GB). Both have local ext4 `/scratch` with 5.8 TB and 8.9 TB
free. Oliver works in `/scratch/olmai/` and runs long jobs inside `screen`. Nothing
has been installed there yet for the current work. Prepare everything that is possible
on the laptop first and then move on to the working machines.

## Conventions

- Commit key changes.
- Before changing behaviour that has a reference output, capture a baseline first
  and prove the change bit-identical. That is how the truncating-schedule bug in
  ANNarchy was found rather than shipped.
- Deferred findings go in `TODO.md`, with the caveats that make them provisional.
