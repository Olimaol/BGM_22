# Plan: reviving BGM_v07 for the BOLD optimization

Started 2026-08-03. Read alongside `CLAUDE.md` (orientation), `TODO.md`
(deferred findings) and `DBS.md` (what DBS changes in the model, and why the
off and on conditions now compile the same network).

## Why this plan exists

The optimization was written in Dec 2025, ran on the workstations, and **stopped
without a message**. The cause was a compound failure:

1. `infer_max_sim_time_ms` computed `len(rates) * 2.31 * 1000 / dt_ms` = 7,161,000 ms
   instead of 716,100 — every evaluation simulated **10x too long**.
2. `BoldMonitor` recorded every 0.1 ms, so a run stored ≥4 GB of BOLD per process.
3. `run_optimization.sh` launched 2 x lambda=12 = **24 processes**; on hinton that is
   ~96 GB against 125 GB of RAM before the vector-of-vector overhead and before
   `get()` doubles it. The OOM killer took them.
4. CompNeuroPy's `_ScriptRunner.run` turns any non-zero child exit into a bare
   `exit(1)` — no message, no traceback — and the children's output went to a
   console that `run_optimization.sh` had backgrounded with `&`.

Separately, v07 had been abandoned as "too slow". It was not: its C++ simulation
runs at 1.81 s per simulated second versus v08's 1.16. **96% of v07's runtime was
Python overhead** in the input-delivery machinery, and the projected 1.25 TiB of
cache was the same design showing up as storage.

## Decisions taken (with reasons)

**Model: v07, all three components** — structured/correlated cortical input, the
distance-dependent microcircuit with realistic FS/dSPN/iSPN proportions, and the
missing-GABA compensation. v08 was only ever a time-pressure fallback. It stays
alive purely as a fast end-to-end test of the pipeline.

**Speed: patch ANNarchy, keep the statistics exact.** The exact beta-binomial
copula generator *cannot* be moved inside ANNarchy (random distributions need
global arguments; no inverse-CDF functions), and regenerating on the fly is 181 s
per simulated second — 4.5x worse than the precompute. Oliver's original design was
right. So the fix was the transfer, not the algorithm.

**Cache layout: pre-summed per postsynaptic type, `uint16`, on `/scratch/olmai`.**
Summing the cortical regions is exactly lossless while every region shares one
weight per post-type, which is what `set_opt_params` does. ~138 GiB per DBS
condition. Also store `(time, receivers)` rather than `(receivers, time)`: the
current order forces a strided transpose on every chunk, measured 8.2x more
expensive.

**Loss: per-region BOLD time-course correlation**, not functional connectivity —
because the cortical drive comes from the same recording, the model can be asked to
reproduce what this subject's basal ganglia actually did, TR by TR. Plus the
firing-rate plausibility term, with the cheap 10 s firing-rate run **gating** the
expensive BOLD run: if `firing_rate_loss` exceeds a threshold, skip BOLD and return
`firing_rate_loss + 1.0`. CMA-ES is rank-based, so that ordering stays consistent.

**Parameters: 19 for v07** — 3 striatal cortical input weights, 4 CorticalInputs
weights, 2 baseline currents (snr, gpe_proto have no cortical input), and 10
projection-cluster scalings. Clusters scale the literature weights rather than
freeing them, which preserves their relative balance and conditions the search.

**DBS-on staging: only the putamen loop's weights move**, plus the 3 DBS
parameters (13 total). The caudate loop is excluded from all DBS effects and has no
connection to the putamen loop, so refitting its weights would assert a mechanism
the model does not contain. This also gives a free control: caudate's on-vs-off
BOLD change must then be explained entirely by its cortical drive.

**DBS: retrofit the mechanisms before compile, in both conditions.** The DBS
equations are added to the six putamen populations DBS can reach by swapping
`neuron_type`/`synapse_type` on the already-built objects, not by
`DBSstimulator(auto_implement=True)`, which clears and recreates the network and
therefore cannot survive v07's `TimedArray`/`CurrentInjection` inputs. Off and on
then differ only in parameter values — the claim the inference rests on. The
price is that off-condition numerics changed: ANNarchy's RNG is one global
stream, so two extra `Uniform` draws anywhere shift every population. Taken
deliberately, before any real fit had been run. Full account in `DBS.md`.

**Seeds: fixed at 42 during fitting** (common random numbers, so CMA-ES sees a
deterministic objective), then the winning parameter vector re-run across ~10 seeds
to report stability.

**Robustness: penalize and continue.** Per-individual log files with exit codes; a
dead individual gets a worst-case loss rather than killing the run; hard abort if
more than half a generation fails; CMA-ES state pickled every generation so a killed
run resumes.

**Deferred to a later session:** the DBS inference design (single-mechanism scan vs
multi-start vs sparsity path). For now, one straightforward DBS-off fit and one
DBS-on fit, just to get the pipeline working. See `TODO.md` §2.

## Agreed sequence

1. ~~Patch ANNarchy's `TimedArray`, prove bit-identical output.~~ **done**
2. ~~Remove the `cyInstance` workarounds the patch makes redundant.~~ **done**
3. ~~Rebuild `get_loss.py`: bug fixes + `--model-version`, smoke-tested on v08.~~ **done**
4. ~~Build a short v07 cache and run the v07 path end-to-end.~~ **done**
5. ~~Firing-rate gate, logging, checkpointing, failure policy; update `deap_cma_opt.py`.~~ **done**
6. ~~Make the DBS-on path work, and prove both conditions end-to-end.~~ **done**
7. **Resolve the parameter bounds (`TODO.md` §1) before any real fit** — on the
   post-step-6 numerics, since step 6 moved them. ← next
8. Workstation setup; build the full DBS-off **and** DBS-on caches; time one
   evaluation per machine.
9. Five-generation mini-run with checkpointing. **This green is the milestone.**
10. Launch the DBS-off fit, then DBS-on.

Step 6 came first because it changes every number the model produces, so any
bounds measured before it would have to be redone.

## Where it stands

Steps 1-5 are done and verified. Measured on the laptop, v07 went from **41.87 to
~3.25 s per simulated second** — a full 716 s evaluation from ~500 min to **~39 min**,
against a C++ floor of ~21 min.

**The v07 path has now actually been executed** (2026-08-04), which is what steps 1-3
could only claim structurally. Running it required fixing three things that had never
been reachable before:

- `CorticalInputs` still demanded a cortical drive sampled at `dt`, while the real
  drive has one value per TR. `Microcircuit` had been given the repeat-by-expansion
  handling in `c0ad10f`; `CorticalInputs` had not, so no CI cache could be built at
  all. Fixed the same way.
- **`update_time = 100 ms` could never have worked.** v07 hands its inputs over one
  `update_time` chunk at a time, so every simulated stretch has to be a whole number
  of chunks — and the ramp-up is one TR, 2310 ms, which is 23.1 chunks. The first
  `simulate_model` call would have raised. `update_time` is now **110 ms**, which
  divides 2310 (21 per TR), the full 716,100 ms run (6510) and the firing-rate probe.
  The probe moved from 10,000 to **9900 ms** to stay a whole number of chunks; that
  changes the v08 rate loss by 1e-4 (0.9010 → 0.9011), so v08 is no longer
  bit-identical to its pre-refactor reference, by that much.
- The three DBS parameters were read at fixed indices 21-23, which only lines up with
  v08's 21 base parameters. v07 has 19. They are now the last three, whatever the
  version.

The end-to-end v07 evaluation (5 TRs, DBS off, all seven drive weights at their 0.001
default, base currents 100, cluster scalings 1) gives

    loss 1.5430 = firing_rate 0.8678 + bold 0.6752

with all 18 population names resolving, 4 BOLD samples per region on the TR grid, and
firing rates in the right order of magnitude everywhere. Sweeping the drive weight
proves it reaches both `mc` and `ci`:

| weight | caudate_dSPN | caudate_FS | thal:caudate | stn:caudate |
|--------|--------------|------------|--------------|-------------|
| 0.0005 |  7.06 Hz     | 14.91 Hz   |  3.94 Hz     | 18.10 Hz    |
| 0.001  | 22.62        | 25.44      |  6.59        | 19.16       |
| 0.002  | 69.88        | 51.44      | 14.54        | 21.75       |

Unlike v08, v07's drive parameter is **not** saturated at its default: the plausible
dSPN band was crossed between 0.001 and 0.002. See `TODO.md` §1 (the v07
measurements live in resolved §9, merged into §1) — the band has since
moved to the medication-off values (12.67-37.33 Hz), which shifts the useful weight
range down without changing the conclusion.

Commits, by plan step:

| step | repo | commit |
|------|------|--------|
| 1 patch ANNarchy | `ANNarchy_compneuro` | `2a11e858` fast buffer + backported `update()` semantics |
| 1 patch ANNarchy | `ANNarchy_compneuro` | `f215694e` round instead of truncate when converting schedule/period |
| 2 drop workarounds | CompNeuroPy | `c0ad10f` (also carries the coarser-cortical-drive change) |
| 2 drop workarounds | BGM_22 | `9716591` |
| 3 rebuild `get_loss.py` | BGM_22 | `3b65e8f` bug fixes + `--model-version` |
| 4 run v07 | CompNeuroPy | `CorticalInputs`: accept a coarser cortical drive |
| 4 run v07 | BGM_22 | `update_time` 110 ms, `build_input_caches.py`, `--cache-dir` |
| 5 optimizer | BGM_22 | firing-rate gate, robust `deap_cma_opt.py`, DBS-on staging |
| — documentation | BGM_22 | `2db3872` `CLAUDE.md`, `PLAN.md`, `TODO.md` |

Unrelated January work committed at the same time: BGM_22 `a0cd700`
(`test_microcircuit.py`) and CompNeuroPy `5fcc6b9` (`spike_input_cortex.py` demo),
both produced for the SPP-2041 meeting.

BGM_22 and CompNeuroPy are committed but **the 2026-08-04 commits have not been
pushed** — push them before step 7, since that is how the code reaches the
workstations.
**`ANNarchy_compneuro` is local only** and must stay that way: its `origin` is
`github.com/ANNarchy/ANNarchy`, the upstream project rather than a fork, so the
patch needs another route to the workstations (see `TODO.md` §6).

Step 5 is written and exercised on v08:

- **Firing-rate gate.** Above `firing_rate_gate` (0.5, in `parameters.py`) the BOLD
  run is skipped and charged 1.0. `--gate-threshold` overrides it per run; 1.0
  disables it. The threshold is **not calibrated** — see `TODO.md` §10.
- **`deap_cma_opt.py` rewritten.** Version switch and per-version bounds, its own
  parallel runner instead of `run_script_parallel` (per-individual log files,
  exit codes kept, `sys.executable` rather than whatever `python` resolves to),
  penalize-and-continue with a hard abort past half a generation, and a CMA-ES
  checkpoint written every generation with `--resume`.
- **DBS-on staging.** `split_param_list` in `get_loss.py` defines the vector
  layouts; the staged one carries the off-fit base parameters into both loops and
  then re-scales the putamen weights only. `deap_cma_opt.py --dbs on` searches
  exactly that: 10 (v07) putamen cluster scalings plus the 3 DBS parameters.
  **Written, but never executed** — see step 6.

## Step 6: the DBS-on path was never real

Started 2026-08-04. The staging above was written and the vector layouts defined,
but no DBS-on evaluation had ever run: no DBS-on loss file, result, checkpoint,
log or cache existed anywhere in the tree. `--dbs on` reached compilation on v08
in Dec 2025 (`annarchy_folders/bgm_v08_on*`) and stopped there. Two defects were
waiting:

1. **v07 + `--dbs on` could not build the model at all.**
   `DBSstimulator(auto_implement=True)` clears the network (`mf.cnp_clear` in
   `_CreateDBSmodel.__init__`) and recreates every projection through `dbs.py`'s
   module-level `_connector_methods_dict`,
   which has no `"Specific"` key — and v07's `CurrentInjection` inputs keep
   `connector_name = "Specific"`. `KeyError`. The rate-coded rewrite would also
   have raised on the `TimedArray`s. v08 escaped only because its inputs are added
   *after* the stimulator.
2. **Even on v08, DBS was silently inert.** `on()` ran after `compile()`, so it
   wrote to the C++ instance and not to `pop.init`; the first `reset()` in
   `Spikes10s.run` and the bare `reset()` in `get_BOLD_full` then restored
   `dbs_on = 0`. Nothing re-applied it. An on-evaluation would have run with DBS
   off, not errored, and still written `"dbs": "on"` into the loss JSON.

Both are fixed by the retrofit decision above plus calling `on()` **before**
`compile()`, so the on-state is the compile-time state and every `reset()`
restores it. `DBS.md` records the mechanism and the limitations found on the way
(axon spikes bypass synaptic delays; the hyperdirect cortical afferent to STN
cannot be activated at all; the DBS constants are unvalidated single-subject
values).

### What step 6 produced

**The off-condition regression is small.** v07, 5 TRs, DBS off, the same vector as
the reference run:

| | before | after |
|---|---|---|
| total | 1.5430 | **1.5442** |
| firing rate | 0.8678 | 0.8705 |
| BOLD | 0.6752 | 0.6737 |

The RNG stream did move, as predicted, but by little: v07's DBS footprint excludes
the 1000-neuron microcircuit, which is what dominates RNG consumption.

**The DBS effect is real and confined to putamen.** v07 at 5 TRs, off vs on
(`dbs_depolarization` 3.0, `passing_fibres_strength` 0.5,
`axon_spikes_per_pulse` 0.5), Hz:

| population | off | on | Δ | | caudate twin | Δ |
|---|---|---|---|---|---|---|
| snr:putamen | 139.11 | 112.63 | **−26.48** | | snr:caudate | +0.71 |
| gpe_proto:putamen | 111.90 | 88.97 | **−22.93** | | gpe_proto:caudate | +0.79 |
| stn:putamen | 18.53 | 10.69 | **−7.84** | | stn:caudate | −0.79 |
| thal:putamen | 4.56 | 3.12 | −1.44 | | thal:caudate | −1.18 |

The sub-1 Hz caudate changes are the cortical drive differing by condition, not a
DBS leak. On v08, where the two on-runs can be compared on the *same* drive with
only the DBS parameters differing, every caudate population is identical to 2 dp
while nine putamen populations move — so the free control holds exactly.

**The whole fitting loop runs in both conditions**, on both model versions:
a `--dbs off` mini-run, then a `--dbs on` mini-run seeding from its pickle
through `load_best_off_fit`, then `--resume`. v07 on carried over
`deap_cma_result_v07_off_run_98` and searched 13 free parameters against a
19-slot fixed base; v08 on searched 15 against 21. `test_dbs_on.py` covers the
mechanism, the negative validation case, and the free control; all 24 checks pass.

**The rate gate would have gated everything.** v07 off scored 0.8705 and on
0.8742, both far above the 0.5 gate, so all four mini-runs ran with
`--gate-threshold 1.0`. `TODO.md` §10 has the numbers.

**A laptop limit, not a code one:** `--lambda 4` on v07 lost two of four
`cc1plus` processes to the OOM killer during the per-individual compile; 16 GB is
not enough for four v07 compiles at once. `--lambda 2` was fine. Noted in
`TODO.md` §6 as a thing to measure before choosing lambda on the workstations —
it is a different limit from the December failure, which was OOM during
*simulation*.

## Immediate next actions

1. Step 7 — the bounds. v07's are provisional: `[0, 0.01]` for the seven drive
   weights with `p0 = 0.001`, the class default. The sweep above says the useful
   range is roughly `[5e-4, 2e-3]`, so the bound is ~5x above the useful top —
   far better conditioned than v08's `[0, 500]` but still guessed. Measure the
   band-crossing per population and set tight bounds, for both versions. Redo the
   sweep first: step 6 moved the numbers.
2. Step 8 — get the patched ANNarchy and the two repos onto hinton/waikiki
   (`TODO.md` §6), then build the caches there. Budget from the laptop, **after
   the generator rebuild of `TODO.md` §22 (resolved 2026-08-07)**: 2.9 min (caudate) and 3.2 min
   (putamen) for 5 TRs means ~3.5 h per loop serially at 310 TRs, so ~7 h and
   ~138 GiB **per DBS condition** — and both conditions are needed, which step 9
   used to assume without ever saying. That is ~6x faster than the 24 min per
   loop this budget used to quote. Generation is embarrassingly parallel over
   (pre, post) pairs, and `TODO.md` §3's smaller layout should land at the same
   time.
3. Step 9 — the five-generation mini-run.
