# BGM_22

Spiking basal ganglia model in ANNarchy, simulated in resting state, producing
simulated BOLD that is fitted to one subject's experimental BOLD under DBS on and
off. The scientific goal is inference: fit DBS-off, refit DBS-on, and read off
which parameters had to change — i.e. what DBS did inside the basal ganglia.

Read `TODO.md` for everything still to do — its Roadmap section holds the
intended order, its entries the substance, and its Resolved section the
project's history (`PLAN.md` was dissolved into it on 2026-08-11; `TODO.md`
§33 maps the old step numbers). Read `DBS.md` for exactly what differs between
the DBS-off and DBS-on model. `model_v07.md` walks through how the real model is
built, step by step from `setup()` to `compile()`; `model_v08.md` does the same
for the reduced model, as a delta against it.

`community_review/` holds the community-conventions survey (`TODO.md` §34,
resolved 2026-08-13): seven reviews of this model written in character as seven
research lineages from their own published standards, plus
`community_review/synthesis.md` — **24 findings F1–F24 ranked by how many seats
raised each, with a change proposal apiece.** Start there before changing the
model: its four top-tier findings all bear on whether the project's central
claim can be made, and the triage that decides which are accepted is the next
Roadmap item. The findings are not themselves referenceable — anything accepted
becomes its own `TODO.md` entry. PDFs live beside the documents, untracked.

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
  Python `update()` refreshes every `update_time` (110 ms). 28 projections.
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
  plus `caudate_rate`/`putamen_rate` mixed by anatomical proportion. Regenerate with
  `cortical_drive_by_bold_run.py`, which needs MATLAB (`matlabengine`) and, on the
  laptop, an **interactive** MathWorks sign-in — `start_matlab()` hangs unattended
  (`TODO.md` §21, resolved 2026-08-06). The folder is **git-ignored**, so back it up before rerunning:
  `create_data_raw_folder` deletes it after a `y/n` prompt.
- Cortical proportions: `experimental_data/cortical_proportions/README.md` — how the
  per-region caudate/putamen input mixes were derived, which numbers are measured
  and which are our assumptions, and what is weakest (putamen PMv, range
  0.10–0.24). The anchor is macaque tracer counts (Borra et al. 2021, 2022);
  human tractography cannot answer this.
- **The proportions live in exactly one place**,
  `BOLD_optimization/parameters.py: cortical_proportions_dict`. They split each
  striatal neuron's cortical afferents per region *and* weight the
  `caudate_rate`/`putamen_rate` mix above, so a second copy could silently
  disagree; `Microcircuit`/`CorticalInputs` have no defaults and validate what they
  are given. Changing them means regenerating the rate `.npz` — the file records
  the weights it was built with and `get_loss` raises on a mismatch — and
  rebuilding every v07 cache. Only v08 is driven by the mixed series; v07 uses the
  per-region ones.
- Striatal firing rates: `experimental_data/activity_striatum/README.md` — where the
  dSPN/iSPN rates and the `get_firing_rate_loss` bands come from, which assumption
  they rest on, and what was rejected. The values are the **medication-off** state of
  Liang et al. 2008; changing them invalidates every v07 input cache.
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
TRs; the script refuses that case up front. A cache also has to match everything
it was drawn at — `firing_rate_dict`, `correlation_dict`, `shared_fraction`,
`shared_fraction_dict`, `cortical_correlation`, `correlation_window_ms`,
`correlation_timescale_ms`, `source_multiplicity` — and a state file missing
**any** compared field is refused outright (hardened 2026-08-13, `TODO.md`
§29: `CorticalInputs` used to record none of the shared-fraction/correlation
fields, so changing them silently reused a stale CI cache). Each stream is
checked against the statistics its own parameters imply as it is written,
**raising** on a mismatch, and the measured mean, Fano factor and pairwise
correlation are stored in the state file (by `Microcircuit` *and*
`CorticalInputs`) so a cache can be audited without regenerating it. The
missing-GABA streams' realised `f(d)` is additionally checked at build time
against the analytic double quadrature (`TODO.md` §27, resolved 2026-08-13).

**What the streams are required to reproduce is written down** in
`experimental_data/input_streams/README.md`, together with what this approach
deliberately cannot represent and what the model may therefore be used to claim.
Read it before changing anything about the inputs.

**No cache currently exists.** `mc_ci_cache/`, `mc_ci_cache_5tr/` and
`mc_caudate_off_cache/` were deleted on 2026-08-06 when the striatal rates moved to
the medication-off values, which invalidated the missing-GABA streams in all of
them. `mc_ci_cache_dir` in `parameters.py` still points at the now-absent
`../mc_ci_cache`, so a v07 run without `--cache-dir` fails immediately rather than
loading something stale. Since the generator was rebuilt on 2026-08-07
(`TODO.md` §22, resolved 2026-08-07) generation is **~6x faster**: roughly **4 min per loop per DBS
condition** at `--n-trs 5` and ~7 h per DBS condition at full length, against 24
min and ~42 h before. Sizes are unchanged — ~22 GB per DBS condition (~44 GB for
all four caches) at `--n-trs 5` and ~1.25 TiB per DBS condition at full length,
in the current `float64`/per-region layout; see `TODO.md` §3 for the agreed
smaller one.

`storage_dir` is resolved **relative to the working directory you launch from**,
which is how `mc_ci_cache/` and `mc_caudate_off_cache/` came to hold overlapping
data. Pass absolute paths. The cortical rate path inside the cache state is no
longer launch-directory-sensitive: it is recorded and compared as a resolved
absolute path, and `parameters.py` builds it from its own file location
(`TODO.md` §13, resolved 2026-08-13) — which also means a cache records the
machine it was built on and will not validate after being copied to a host
where the repo lives at a different absolute path. The *other* paths in
`parameters.py` are still relative, so run everything from
`BOLD_optimization/` regardless.

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

The loss is the per-region BOLD time-course correlation — not functional
connectivity: the cortical drive comes from the same recording, so the model is
asked to reproduce what this subject's basal ganglia actually did, TR by TR —
plus a firing-rate plausibility term. The cluster scalings multiply the
literature weights rather than freeing them, which preserves their relative
balance and conditions the search. A rate probe gates the BOLD run: above
`firing_rate_gate` (0.5) BOLD is skipped and
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
  the full run and the probe but not the TR, which is why the ramp-up (one TR)
  would have failed on the first call.
- The cortical drive has one value per TR, coarser than `dt`. `Microcircuit` and
  `CorticalInputs` repeat each value `TR/dt` times; a drive finer than `dt`, or one
  whose spacing is not an integer multiple, is rejected.
- The striatal `exp_input_weight` bounds are orders of magnitude too wide in **v08**
  (see `TODO.md` §1) — do not start a real fit before resolving that. v07's drive
  weight is better behaved but its bounds are still provisional (also `TODO.md` §1).
- **A parameter set after `compile()` does not survive `reset()`.**
  `Population.__setattr__` writes to `pop.init` while the population is
  uninitialized and to the C++ instance afterwards, and `Population.reset()` is
  `self.set(self.init)`. So anything assigned post-compile silently reverts at the
  next reset — and CompNeuroPy's `CompNeuroExp.reset()` defaults to
  `parameters=True`. This is exactly how DBS-on evaluations ran with DBS switched
  off, without an error and with a plausible loss file. Set such state **before**
  compile, or re-apply it after every reset site.
- **Assigning an attribute a Population or Projection does not have is not an
  error.** `__setattr__` falls through to `object.__setattr__`, so a wrong or
  missing parameter name quietly becomes a plain Python attribute and the intended
  effect is dropped. Nothing warns. Any code that walks a list of objects setting
  parameters should check `name in pop.attributes` first and raise otherwise.
- **ANNarchy's RNG is not per population.** Random variables in equations draw
  from one global `std::vector<std::mt19937> rng` — `rng[0]` for global variables,
  `rng[thread_id]` under OpenMP. Adding or removing a random variable *anywhere*
  shifts the numbers every other population receives, including populations in the
  other, unconnected BG loop. The "capture a baseline and prove it bit-identical"
  convention below cannot be satisfied locally for such a change; expect the whole
  model to move and record the delta instead.
- **Two unrelated things in this project are called `dbs`.**
  `model_creation_kwargs["dbs"]` only selects the cortical firing-rate file and the
  cache directory — it changes no equation, weight or connectivity. The actual
  stimulation is `DBSstimulator` plus the equation terms added by
  `add_dbs_mechanisms`. `DBS.md` has the full account. Do **not** reach for
  `DBSstimulator(auto_implement=True)` even though it is what the CompNeuroPy
  example uses: it clears and recreates the entire network, which cannot
  reconstruct `TimedArray`/`CurrentInjection` (v07 dies with a `KeyError` on
  `connector_name == "Specific"`) and invalidates every Python pointer into the
  model.
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
- All future work goes in `TODO.md` — deferred findings and upcoming tasks
  alike, with the caveats that make them provisional. Its entries are
  append-only chronological logs with numbers that are never reused; its
  Roadmap section (freely rewritten, ordering only, never itself referenced)
  must always account for every open entry; resolved entries move to its
  Resolved section and leave a stub. The maintenance rules are at the top of
  that file — follow them when adding, updating or resolving an entry.
- **Documentation maintenance** (from `TODO.md` §18, resolved 2026-08-11). The
  living documents are `CLAUDE.md`, `DBS.md`, `model_v07.md`,
  `model_v08.md`, the `experimental_data/` READMEs, and code comments.
  - **Targeted cross-reference check.** Whenever an artifact changes — a code
    file, a function, a doc section, a TODO entry — grep the living documents
    for references to it (filename, symbol names, `§N`) and update what the
    change invalidated. Proportional, not a full re-read of everything.
  - **Cite code by file + symbol** (function, class, method), never by bare line
    numbers — they rot silently. Where no symbol exists (template strings,
    generated code), cite the nearest named thing plus a short greppable quote.
  - **Section numbers are stable identifiers** in `TODO.md`,
    `model_v07.md` and `model_v08.md`: never renumber existing sections; insert
    with sub-numbers or append. A forced renumbering is itself a change under
    the first rule and triggers a `§N` sweep of all living documents.
  - **`TODO.md`'s historical Opened/Update/Resolved blocks are exempt** — they
    stay as written; citations in them are accurate as of their timestamp.
