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
4. **Build a short v07 cache and run the v07 path end-to-end.** ← next
5. Firing-rate gate, logging, checkpointing, failure policy; update `deap_cma_opt.py`.
6. Resolve the parameter bounds (`TODO.md` §1) before any real fit.
7. Build the full DBS-off cache on a workstation; time one evaluation per machine.
8. Five-generation mini-run with checkpointing. **This green is the milestone.**
9. Launch the DBS-off fit, then DBS-on.

## Where it stands

Steps 1-3 are done and verified. Measured on the laptop, v07 went from **41.87 to
~3.25 s per simulated second** — a full 716 s evaluation from ~500 min to **~39 min**,
against a C++ floor of ~21 min. v08 reproduces its pre-refactor loss exactly.

Commits: `ANNarchy_compneuro` `2a11e858` (fast buffer + backported `update()`
semantics) and `f215694e` (rounding fix); CompNeuroPy `c0ad10f`; BGM_22 `9716591`.

**The v07 code path has never been executed** — `Microcircuit` refuses a cache whose
`n_steps` differs from `int(t.duration/dt)`, and the caches on disk hold 12,000 steps
while even a 2-TR run needs 46,200. Everything claimed about v07's parameter mapping
is structural (cluster/projection cross-check against `parameters.csv`, name
mapping), not observed.

`deap_cma_opt.py` is largely untouched: it still hard-codes 21 parameters and the
v08 bounds, and its DBS-on branch still refits all weights for both loops.

## Immediate next actions

1. Build a v07 cache for a short duration (e.g. `t.duration = 4620` ms = 2 TRs, so
   `n_steps = 46,200`), then run
   `get_loss.py --model-version v07 --dbs off --n-trs 2` and check: cache loading,
   `set_opt_params_v07` reaching `mc`/`ci` `mean_weights_by_type`, the chunked
   `simulate_model` loop, the `caudate_dSPN`-style names resolving in the BOLD
   monitors and firing-rate loss, and a plausible loss.
2. Then step 5: the gate, logging, checkpointing, and `deap_cma_opt.py`.
