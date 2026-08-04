# Plan: reviving BGM_v07 for the BOLD optimization

Started 2026-08-03. Read alongside `CLAUDE.md` (orientation) and `TODO.md`
(deferred findings).

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
6. **Resolve the parameter bounds (`TODO.md` §1) before any real fit.** ← next
7. Build the full DBS-off cache on a workstation; time one evaluation per machine.
8. Five-generation mini-run with checkpointing. **This green is the milestone.**
9. Launch the DBS-off fit, then DBS-on.

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
dSPN band (20.45-53.69 Hz) is crossed between 0.001 and 0.002. See `TODO.md` §9.

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
  disables it. The threshold is **not calibrated** — see `TODO.md` §9.
- **`deap_cma_opt.py` rewritten.** Version switch and per-version bounds, its own
  parallel runner instead of `run_script_parallel` (per-individual log files,
  exit codes kept, `sys.executable` rather than whatever `python` resolves to),
  penalize-and-continue with a hard abort past half a generation, and a CMA-ES
  checkpoint written every generation with `--resume`.
- **DBS-on staging.** `split_param_list` in `get_loss.py` defines the vector
  layouts; the staged one carries the off-fit base parameters into both loops and
  then re-scales the putamen weights only. `deap_cma_opt.py --dbs on` searches
  exactly that: 10 (v07) putamen cluster scalings plus the 3 DBS parameters.

Not yet done: the bounds (step 6), everything on the workstations (step 7 on).

## Immediate next actions

1. Step 6 — the bounds. v07's are provisional: `[0, 0.01]` for the seven drive
   weights with `p0 = 0.001`, the class default. The sweep above says the useful
   range is roughly `[5e-4, 2e-3]`, so the bound is ~5x above the useful top —
   far better conditioned than v08's `[0, 500]` but still guessed. Measure the
   band-crossing per population and set tight bounds, for both versions.
2. Step 7 — get the patched ANNarchy and the two repos onto hinton/waikiki
   (`TODO.md` §6), then build the full DBS-off cache there. Budget from the
   laptop: 24 min per loop for 5 TRs means ~25 h per loop serially at 310 TRs.
   Generation is embarrassingly parallel over (pre, post) pairs, and `TODO.md` §3's
   smaller layout should land at the same time.
3. Step 8 — the five-generation mini-run.
