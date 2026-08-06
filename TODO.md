# TODO

Things found while working on the project that we deliberately postponed. Each
entry says what was observed, why it matters, and what still has to be decided
or done. Newest section at the bottom.

---

## From the session on 2026-08-03 (reviving BGM_v07 for the BOLD optimization)

### 1. Validate the optimization bounds — for both v07 and v08

**Observed (v08).** Sweeping the striatal `exp_input_weight` with everything else
held at an arbitrary vector:

| weight | str_d1 | str_d2 | str_fsi | firing-rate loss |
|--------|--------|--------|---------|------------------|
| 0.5    | 1462 Hz | 1456 Hz | 1144 Hz | 0.900 |
| 0.2    | 1014    | 1001    |  469    | 0.804 |
| 0.05   |  533    |  519    |  241    | 0.795 |
| 0.01   |  293    |  270    |  167    | 0.868 |
| 0.002  |  247    |  222    |  154    | 0.872 |

The configured bound is `[0, 500]` (`[0, 800]` for FSI) but the rate has already
saturated by ~0.2 and the loss is flat across the whole sweep. Mechanism:
`exp_input = Exponential(1/0.7) * exp_input_weight * g_cor` enters as
`g_ampa += exp_input/dt`, so `g_ampa` gains `weight * 7 * rate` per step. With a
cortical drive of ~20 Hz the natural scale for the weight is ~1e-3 to 1e-2.

**Why it matters.** `p0` is 0 for these nine parameters and the default `sig0` is
25% of the range, so early CMA-ES generations would sample almost exclusively in
the saturated regime where the loss carries no gradient information.

**Caveats — do not treat the table as measured truth.**
- The sweep held all twelve cluster scalings at 1.0, base currents at 100 and the
  other four `exp_input_weight` values at 0.01. In that configuration `gpe_arky`
  and `gpe_cp` were silent, so the striatum got no pallidal inhibition, and `thal`
  drove `str_d1` at 74 Hz through a weight of 7. The absolute rates are therefore
  artefacts; only the saturation/flatness conclusion is safe.
- The `sig0` claim was read from the `DeapCma` signature, not verified against
  what `_prepare` actually does after rescaling to [0, 1]. **Check this.**
- The docstring derives 500 from `g_eff = g/(1 + g*dt/C)` with C=50. If
  `exp_input_weight` was deliberately meant to live on a conductance scale, the
  inferred 1e-3 scale is what is wrong, not the bound. Worth resolving.
- v07 parameterizes the striatal drive through `mc.mean_weights_by_type`, which
  **defaults to 0.001** — already in the range the arithmetic suggests. So this may
  be a v08-only problem. Validate both versions separately.

**Options discussed:** optimize in log space (e.g. `log10(weight)` over [-5, 0]);
measure the useful range per population and set tight linear bounds; do both; or
keep the bounds and hand-pick `p0`/`sig0`. Not decided.

### 2. Design the DBS-on inference properly

Deferred by decision: for now, run one straightforward DBS-off fit and one DBS-on
fit with all free parameters, just to get the pipeline working.

The goal is inference — *which* parameters had to move, i.e. what DBS changed
inside the BG. A single argmin will not support that, because the 13 free
parameters contain near-degenerate pairs: `axon_spikes_per_pulse` versus the
`stn__gpe`/`stn__snr` scalings, and `passing_fibres_strength` (DBS activation of
`snr__thal:putamen`) versus the `snr__thal` cluster scaling.

Options costed against 20 (hinton) + 24 (waikiki) physical cores at ~25-40 min per
evaluation: exhaustive single-mechanism scan (ten 4-parameter fits, ~2 days) then a
free refit; multi-start free refit (~5 restarts, ~4 days); L1-regularized "minimal
change" refit with a penalty sweep (~4-5 days). All estimates are CMA-ES rules of
thumb, not measured — treat as ±2x.

Settled already: in the on condition the caudate loop keeps its off-condition
weights, because caudate is excluded from all DBS effects and there are no
cross-loop projections. Only putamen weights + the 3 DBS parameters move.

### 3. Regenerate the input caches with a transposed layout

The memmaps are stored `(receivers, time)`, so `chunk.T` in `Microcircuit.update()`
and `CorticalInputs.update()` is non-contiguous and forces a strided copy.
Measured **8.2x** more expensive than an already `(time, receivers)` layout for a
486-wide chunk (2.49 ms vs 0.30 ms per 486k elements).

Do this when the caches are regenerated anyway, together with the other agreed
changes: `uint16` instead of `float64`, cortical regions pre-summed per
postsynaptic type, stored under `/scratch/olmai/`. Expected ~138 GiB per DBS
condition. Generation is embarrassingly parallel over (pre, post) pairs and over
time chunks — serial it is ~70 h per condition, ~3.5 h across 20 cores.

Note: pre-summing forecloses giving different cortical regions different weights
without regenerating. Currently `set_opt_params` assigns one weight per
postsynaptic type, so it is exactly lossless today.

### 4. Check the missing-GABA self-consistency after the first fit

`Microcircuit._simulate_distance_dependent_spike_counts` generates the missing
local GABA input assuming the surrounding striatal neurons fire at
`{"FS": 10.0, "dSPN": 25.0, "iSPN": 33.0}` Hz. Those are baked into the cache.

`get_firing_rate_loss` scores the simulated rates against bands centred on exactly
those values (12.67-37.33 and 21.22-44.78), so the design is coherent at the band
centre. But the loss is a smooth logistic, so a fit sitting near a band edge would
have neurons firing at e.g. 37 Hz while being inhibited by a surround the cache
assumes fires at 25 Hz.

**Decided:** fit first, then compare the fitted rates against the assumption and
report the mismatch. Only if they drift to an edge, consider iterating to a fixed
point (regenerate the 7 local pairs, ~30 min parallelized, then refit).

### 5. Truncating schedule conversion still present in other ANNarchy classes

Fixed for `TimedArray` (ANNarchy_compneuro `f215694e`): `__setattr__` converted the
schedule to integer steps with a truncating cast, so ~5% of entries landed one step
early (47 of 1000 for a dt-spaced schedule) — `4.3 / 0.1 == 42.99999999999999`. The
period had the same problem via `int(value / dt)`.

The identical conversion still exists in `TimedPoissonPopulation` and the CUDA
`TimedArray` path. Left alone to keep that commit reviewable. Fix if those are ever
used.

### 6. Workstation setup, when we move off the laptop

- Install the patched ANNarchy (`ANNarchy_compneuro`, branch `timedarray-fastbuffer`)
  on hinton and waikiki. Currently both have a plain 4.7.3 in site-packages.
  **This cannot be done with `git push`**: that repo's `origin` is
  `github.com/ANNarchy/ANNarchy`, the upstream project, not a fork of Oliver's. So
  the branch has to reach the workstations another way — push to a personal fork,
  `rsync` the directory across, or build a wheel and copy that. BGM_22 and
  CompNeuroPy do point at `Olimaol/...` repos and can be pushed normally.
- Unify the CompNeuroPy checkout locations: hinton has it at
  `/scratch/olmai/Projects/PhD/CompNeuroPy`, waikiki at `/scratch/olmai/CompNeuroPy`
  (`3d91dbc`, matching the laptop). hinton's commit was never checked.
- Confirm the two Python environments match before running anything long.
- `OMP_NUM_THREADS=4` is set in the shell profile on both. ANNarchy ignores it
  (it sets its own thread count, default 1), but **numpy's OpenBLAS does not** —
  export `OMP_NUM_THREADS=1` / `OPENBLAS_NUM_THREADS=1` in the optimization
  launcher so lambda processes do not each grab a 4-thread pool. Leave it at 4+
  when *building* the caches, where the SVD and ppf calls dominate.
- Test whether lambda should equal physical cores (20/24) or logical (40/48):
  run one generation each way on hinton and compare wall time.
- Measure the real per-evaluation time on both machines; all current budgets come
  from laptop timings.
- **lambda may be bound by compilation, not simulation.** On the laptop (16 GB)
  a v07 mini-run at `--lambda 4` had two of its four `cc1plus` processes killed
  by the OOM killer during the per-individual compile; `--lambda 2` was fine.
  This is a different limit from the December failure, which was OOM during
  *simulation* with 24 processes. Measure the peak RSS of one v07 `g++` before
  choosing lambda on hinton (125 GB) and waikiki (251 GB) — and note the
  per-individual compile folders mean lambda compilations run at once. If it
  binds, compile once with `--compile` and pass `--skip-compile` to the run, or
  stagger the compile phase.

### 7. Where the rebuild stands (state at the end of the session)

Done and verified:
- ANNarchy patched and committed (`ANNarchy_compneuro`, branch
  `timedarray-fastbuffer`, `2a11e858` + `f215694e`), installed into the laptop's
  `compneuro` env. v07 went from 41.87 to ~3.25 s per simulated second, i.e. ~500
  to ~39 min for a full 716 s evaluation.
- `cyInstance` workarounds removed from `Microcircuit`, `CorticalInputs`,
  `get_loss.py` and `test_microcircuit_bgm.py` (CompNeuroPy `c0ad10f`,
  BGM_22 `9716591`). Verified bit-identical: 175311 spikes, 18 populations.
- `get_loss.py` bug fixes: the 10x duration, BOLD monitor recording on the TR grid
  (19 samples per region instead of 462000), per-loop cortical drive, `--dbs`
  required, per-run plot file, `--n-trs` for short runs, and a loss file that now
  records the loss components, per-region correlations and all firing rates.
- `--model-version {v07,v08}` implemented. v08 reproduces the pre-refactor loss
  exactly (1.4266 = 0.9001 + 0.5265). Cluster definitions check out against
  `parameters.csv`: v07 covers all 28 projections in 10 clusters (19 params),
  v08 all 35 in 12 clusters (21 params), none missing or unused.

Not yet done:
- **The v07 path has never actually been executed.** `Microcircuit` and
  `CorticalInputs` refuse a cache whose `n_steps` differs from
  `int(t.duration/dt)`, and the existing `mc_ci_cache` holds 12000 steps (1.2 s)
  while even a 2-TR run needs 46200. A short cache has to be built first.
- The firing-rate gate, per-individual logging, checkpointing and the
  penalize-and-continue failure policy are designed but not written.
- `deap_cma_opt.py` still assumes 21 parameters and the v08 bounds; it needs the
  version switch, the per-version parameter count, and the DBS-on staging where
  only putamen weights move.

### 8. Smaller cleanups

- `compute_bold_correlation_loss` still takes a `dt_ms` argument that is unused now
  that the BOLD monitors record on the TR grid.
- `Spikes10s.run` hard-codes a 10 s firing-rate simulation.
- `BoldMonitor(normalize_input=2000)` computes its baseline over the first 2000 ms
  of recording while `t.rampup` is 2310 ms — check the two are intended to differ.
- `deap_cma.run.max_evals = 2000` is commented as "approx 14 days"; update once the
  real per-evaluation time is known.
- `test_microcircuit_bgm.py` still has `duration_ms = 200  # TODO currently for
  testing` and `normalize_input=500  # TODO change back to 2000`.
- `parameters.py`: `ci.n_thal` etc. are marked "TODO use lit motivated values".
- The FC matrices in `sub-01_subdiv_results.h5` are **not** plain
  `corrcoef(time_series)` — they look regularized or partial. Worth confirming with
  whoever produced them if the FC is ever used as a fit target.
- The user doesn't want dbs to appear in `model_creation_kwargs["dbs"]`, passed to `Microcircuit(dbs_condition=...)` and `CorticalInputs(dbs_condition=...)`. The model creation, i.e. creating the BGM model, should be independent of DBS. DBS is added after model creation. Currently, the dbs information during the model creation only selects the cortical firing-rate files. SO better give directly the locations of the files.
- rename dbs_depolarization, it actually hyperpolarizes the neurons, currently it's just a wrong naming

---

## From the session on 2026-08-04 (first execution of the v07 path, plan steps 4-5)

### 9. The bounds question, now with v07 measured

Resolves two caveats from §1 and adds numbers for v07. §1 stays open; this is the
evidence to settle it with.

**`sig0` is confirmed.** `DeapCma._prepare` scales the bounds to [0, 1] first and
*then* sets `sigma = 0.25 if sig0 is None`, so the default really is 25% of the
parameter range, as §1 assumed.

**v07 is not saturated at its default.** With everything else held at base currents
100 and cluster scalings 1, sweeping all seven drive weights together (5 TRs,
DBS off, seed 42):

| weight | caudate_dSPN | caudate_FS | thal:caudate | stn:caudate | rate loss |
|--------|--------------|------------|--------------|-------------|-----------|
| 0.0005 |  7.06 Hz     | 14.91 Hz   |  3.94 Hz     | 18.10 Hz    | —         |
| 0.001  | 22.62        | 25.44      |  6.59        | 19.16       | 0.868     |
| 0.002  | 69.88        | 51.44      | 14.54        | 21.75       | 0.837     |

So the dSPN plausible band (20.45-53.69 Hz) is crossed between 0.001 and 0.002, and
the response is steep rather than flat — the opposite of v08's behaviour in §1. This
also confirms the optimized parameters really reach `mc.mean_weights_by_type` and
`ci.mean_weights_by_type`.

> **Superseded on 2026-08-06, in part.** The measured rates in the table stand, but
> the bands they were read against do not: the striatal targets moved to the
> medication-off values (dSPN band now 12.67-37.33, iSPN 21.22-44.78) — see
> `experimental_data/activity_striatum/README.md`. Under the new dSPN band, 22.62 Hz
> at weight 0.001 sits comfortably inside rather than near the lower edge, and 69.88
> at 0.002 is well outside rather than just outside. The useful weight range
> therefore shifts *down*, which if anything better centres the provisional
> `[0, 0.01]` bounds with `p0 = 0.001`. **The rate-loss column (0.868, 0.837) was
> computed against the old bands and is stale** — it would have to be re-measured,
> and doing so needs a rebuilt cache.

**Provisional bounds are now in `deap_cma_opt.search_space`:** `[0, 0.01]` for the
seven v07 drive weights with `p0 = 0.001` (the class default, i.e. the model as its
author configured it). That is ~5x above the useful top rather than v08's ~250,000x,
but it is still a guess.

**Caveats.** The sweep moved all seven weights together, so the striatal and the
CorticalInputs weights are confounded; the rates it produces are the joint effect.
`gpe_arky` (1.9-4.3 Hz against a 15-20 band), `gpe_cp` (10-15 against 75-85) and
`snr` (132-139 against 21-93) are far outside their bands at every weight, which is
what keeps the rate loss near 0.85 — those are driven by the base currents and the
cluster scalings, not by this parameter. **Per-population bounds still have to be
measured one parameter at a time.**

### 10. The firing-rate gate threshold is not calibrated

`parameters.py: firing_rate_gate = 0.5`, i.e. "roughly band-edge plausible or
better", derived from the shape of the loss (≈0.02 when every population sits at its
band centre, 0.5 at the edges) — not from observed v07 losses.

The one real v07 evaluation so far scored **0.868**, well above the gate. If early
CMA-ES generations all sit there, every individual is gated and the search sees only
the rate term until the rates come good. That may be exactly the intent, or it may
stall the fit. **Decide from the first mini-run**: log how many individuals per
generation are gated (`bold_skipped` is in every loss file) and raise the threshold
if the answer is "all of them, for many generations".

**Extension (2026-08-04): the bands are condition-independent, and DBS is on
during the probe.** `get_firing_rate_loss`'s `plausible_ranges` has no DBS switch,
and `dbs_stimulator.on()` now precedes the rate probe, so an on-condition
individual is judged against off-condition bands while STN and its targets are
being driven harder. The risk of gating everything is strictly worse in the on
condition than the 0.868 above suggests. The step 6 mini-runs therefore ran with
`--gate-threshold 1.0` and logged the off and on rate losses side by side.

**Measured (2026-08-04), v07 at 5 TRs, the default parameter vector:** off
**0.8705**, on **0.8742**. So the on condition is barely worse, but *both* are far
above the 0.5 gate — at the default vector every individual would be gated in
either condition, and the search would see only the rate term. The gate is
therefore untenable as configured, and the question is not on-vs-off but whether
0.5 is reachable at all once the bounds (§1, §9) let the rate term come good.
Decide together with the bounds, from a real mini-run at fitted parameters rather
than at the defaults. Inventing on-condition bands now would bake guessed values
into an expensive fit.

### 11. `run_script_parallel` is no longer used by the optimization

`deap_cma_opt.py` now runs its individuals itself. CompNeuroPy's version is
unchanged and still has the behaviour that hid the December failure: any non-zero
child exit sets an error flag, which terminates every sibling and calls a bare
`exit(1)`. It also spawns `["python", script]`, i.e. whatever `python` resolves to
rather than the running interpreter.

Left alone deliberately — it is shared code with other users. Fix it there if anyone
else hits the same wall.

### 12. Cost of building the input caches, measured

Laptop, 5 TRs (115,500 steps at dt = 0.1 ms), DBS off, current `float64`
`(receivers, time)` layout:

- **24.2 min for the caudate loop, 23.8 min for putamen**, 21 GB on disk for both.
- Extrapolated to the full 310 TRs: **~25 h per loop, ~50 h per condition** serially.
  Consistent with the ~70 h in §3, which was an estimate.

The short cache lives in `mc_ci_cache_5tr/` (gitignored) and is only loadable at
`--n-trs 5`, because `Microcircuit` and `CorticalInputs` require the stored `n_steps`
to equal `int(t.duration/dt)` exactly.

Note the constraint that only bites on short runs: the cache must also cover the
**firing-rate probe**, which is a fixed 9900 ms. At full length the BOLD run dwarfs
it; below 5 TRs the probe is the longer of the two and would run off the end of the
cache. `build_input_caches.py` refuses that case up front.

### 13. Cache state files store the cortical rate path as a bare string

`Microcircuit` and `CorticalInputs` compare the saved `cortical_rate_path` verbatim
against the one they are given, so a cache built from `../striatal_.../x.npz` is
rejected when the same file is later named through a different path. This is why
`build_input_caches.py` has to be run from `BOLD_optimization/`, like `get_loss.py`.
Harmless once known; worth normalizing to a resolved absolute path if the caches are
ever built from somewhere else.

---

## From the session on 2026-08-04 (making the DBS-on path real)

The mechanism itself is documented in `DBS.md`; this section only records what we
found and chose **not** to act on yet.

### 14. Axon spikes bypass the synaptic delay line

Because the DBS `pre_axon_spike` string differs from the synapse's `pre_spike`,
ANNarchy takes a separate-loop transmission branch that reads `pop.axonal` at the
current step, while regular spikes read `_delayed_spike[delay-1]`
(`ANNarchy/generator/Projection/SingleThreadGenerator.py:983-1013`, whose own
comment calls it "quite hacky").

For `stn__snr` and `stn__gpe_*`, which carry multi-ms delays, the DBS-evoked
volley therefore arrives earlier than a natural one from the same axon. Whether
that matters at BOLD timescales is unknown — it is a millisecond-scale effect
being read out through a 2.31 s TR — but it is a real deviation from the intended
physiology and it is in ANNarchy, not in our configuration. Revisit only if the
fitted `axon_spikes_per_pulse` turns out to carry a lot of the explanation.

### 15. The hyperdirect cortical afferent to STN cannot be activated

`afferents=True` is set, but `ann.projections(post=stn:putamen)` reaches the
cortical drive through `TimedArray` → `CurrentInjection`. A `TimedArray` has no
soma, no `dmp/dt`, and the rate-coded DBS rewriter raises on it, so it is excluded
from the DBS footprint. In practice **afferent DBS in this model means
`gpe_proto→stn` only.**

Cortical fibre activation is one of the most-discussed DBS mechanisms, so its
absence is a genuine limit on what the inference can conclude — a fitted
"afferent" effect here is a pallidal one. Representing it would mean giving the
cortical drive a spiking soma, which is a model change, not a bug fix.

### 16. The DBS constants are unvalidated single-subject values

`population_proportion = (35+23)/(70+75) = 0.4` (VTA, Berlin subject 1),
125 Hz, and `snr__thal:putamen` as the single passing fibre (after Miocinovic et
al. 2006) are hard-coded in `get_loss.py` and documented only inline. The pulse
width is 100 µs against 60 µs in the data, raised because 60 µs is below dt.

None of these is fitted, so each is an assumption the inference inherits.
Sensitivity to at least `population_proportion` and `dbs_pulse_width_us` should be
checked once a DBS-on fit exists — before any of it is written up.

### 17. `dbs_depolarization` scales with `C` in Izhikevich-2007 models

The term is appended to the right-hand side, so in models written as
`C * dv/dt = ...` (the Izhikevich-2007 striatal ones) it is implicitly divided by
`C`, while in the Izhikevich-2003 BG models `dv/dt = ...` it is not. The optimizer
treats `dbs_depolarization` as one scalar in `[0, 10]` regardless.

It does not bite today — every population in the DBS footprint is
`Izhikevich2003NoisyBaseNonlin` — but it would the moment a striatal population
entered the footprint, and it would do so silently. Worth a guard in
`add_dbs_mechanisms` if that ever becomes possible.

## From the session on 2026-08-05 (DBS.md corrections)

### 18. DBS.md's line references drift silently

Two claims in `DBS.md` were wrong and were corrected on 2026-08-05: `c`/`d` were
described as existing for the DBS axon reset (they are the ordinary Izhikevich
reset parameters, used by every regular spike), and `post.dbs_on` was described
as confining an axon volley to the VTA (it is 1 on every non-excluded population,
so it only bites on projections into the stimulated population and into excluded
ones — efferents and passing fibres carry the volley outside the VTA, which is
the orthodromic effect).

While fixing them, **every one of the 15 `dbs.py:NNN` references in `DBS.md` was
stale** — some by −345 lines, some by +155, so `dbs.py` had been reordered as well
as grown. The document was committed (`a8e0222`, 15:42) *after* the last `dbs.py`
commit (`73481ab`, 15:14), so the numbers were already wrong when it claimed
"everything cited here was checked against the code". They have been recomputed
against the current 2100-line file. Re-verify them, or replace them with function
names, whenever `dbs.py` changes.

## From the session on 2026-08-05 (documenting the model creation)

### 19. `parameters.py` labels the *planned* cache size as the current one

The `mc_ci_cache_dir` comment says "the caches are ~138 GiB per DBS condition".
That number is §3's estimate for the **layout we have not built yet** — `uint16`
instead of `float64`, cortical regions pre-summed per postsynaptic type. The
caches that exist today are ~11x larger. Anyone provisioning `/scratch` from that
comment will under-allocate by an order of magnitude.

Derivation, cross-checked against the cache that exists. The streams total
**24 084 receiver rows** across both loops in the current layout — caudate 11 342
(2942 compensation + 6000 cortical-striatal + 2400 `CorticalInputs`), putamen
12 742 (2942 + 7000 + 2800) — each row `n_steps` values:

| layout | rows | `n_steps` | per DBS condition |
|--------|--------|--------|--------|
| current, `--n-trs 5` | 24 084 | 115 500 | 22.28 GB derived / **22.28 GB measured** |
| current, 310 TRs | 24 084 | 7 161 000 | 1 380 GB = **1.26 TiB** |
| planned (`uint16`, pre-summed), 310 TRs | 8 684 | 7 161 000 | 124 GB = 116 GiB |

The derived and measured 5-TR figures agree to four digits, so the extrapolation
is sound, and the planned-layout row confirms that §3's ~138 GiB is the right
order for what it describes. `CLAUDE.md` and `PLAN.md` already carry the correct
current figure (~1.25 TiB); `parameters.py` is the only outlier.

**What to do.** Either qualify the comment ("~1.26 TiB today, ~120 GiB after the
layout change of TODO §3") or leave it until §3 actually lands and the number
becomes true. Not corrected here, because changing it in isolation invites the
opposite confusion.

## From the session on 2026-08-06 (striatal firing rates)

### 20. The FS rate is not on a stated dopamine condition

`firing_rate_dict["FS"] = 10.0` Hz cites Yamada et al. 2016, Marche & Apicella
2021, Adler et al. 2013, Hernandez et al. 2013 and He et al. 2024. **Nothing in
this repository records whether those recordings are from dopamine-depleted
animals.** The dSPN and iSPN rates were just put on an explicit condition — the
parkinsonian Off state of Liang et al. 2008, see
`experimental_data/activity_striatum/README.md` — and FS is now the only striatal
rate that is not.

This matters because dopamine depletion is generally reported to change striatal
FSI activity, and because the rate is what the missing-GABA cache draws the
surrogate FS spike trains at. It is also the pre-synaptic rate for `FS->dSPN`,
`FS->iSPN` and `FS->FS`, i.e. all the feedforward inhibition the lattice receives
from outside itself.

Secondary: the `str_fsi` band `(5, 15)` in `get_firing_rate_loss` is hand-set, not
derived from any reported SD, unlike `str_d1` and `str_d2`.

**What to do.** Check each of the five sources for species and dopamine state, then
either confirm 10 Hz for the Off condition or replace it. Changing it invalidates
the v07 caches exactly as the SPN rates did.

**Caveat.** The planned-layout row assumes pre-summing collapses all cortical
regions to one stream per postsynaptic type and nothing else changes. It has not
been built, so treat it as arithmetic, not measurement.
