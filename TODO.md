# TODO

Things found while working on the project that we deliberately postponed. Each
entry says what was observed, why it matters, and what still has to be decided
or done. Open entries come first, grouped by the session that produced them,
newest at the bottom; closed entries live in the **Resolved** section at the
bottom of the file.

## Maintaining this file

- **Numbers are never reused.** An entry keeps its number for life, so `§N` is
  a stable reference everywhere, including for resolved entries. A gap in the
  active numbering means the entry was resolved; a one-line stub marks its old
  place.
- **Entry bodies are append-only chronological logs.** The first block is
  `**Opened <YYYY-MM-DD HH:MM>:**`; every later change is a new
  `**Update <YYYY-MM-DD HH:MM>:**` block appended at the end. Never rewrite
  earlier blocks — a correction is a new Update stating what changed
  (typo-level wording fixes exempt). Timestamps carry the time of day because
  one day can hold several updates.
- **The dateline** directly under each heading summarizes the lifecycle at a
  glance: opened date, resolved date (if resolved), number of updates and the
  date of the last one.
- **Resolving an entry:** append a `**Resolved <YYYY-MM-DD HH:MM>:**` block,
  move the whole entry to the Resolved section (ordered by number), and leave
  a one-line stub heading in its session position. Then check the
  cross-references: `§N` mentions in the living documents — `CLAUDE.md`,
  `PLAN.md`, `DBS.md`, `model_v07.md`, `model_v08.md`, the
  `experimental_data/` READMEs, code comments — get annotated
  "(resolved <date>)". Historical Opened/Update blocks inside this file are
  never edited retroactively, so references in them stay as written.

All Opened/Update/Resolved timestamps up to 2026-08-10 were reconstructed on
2026-08-11 from the git history of this file; they are commit times, which can
lag the session that produced the finding by a few hours.

---

## From the session on 2026-08-03 (reviving BGM_v07 for the BOLD optimization)

### 1. Validate the optimization bounds — for both v07 and v08

*Opened 2026-08-04 06:24 · 1 update, 2026-08-11 06:54*

**Opened 2026-08-04 06:24:**

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

**Update 2026-08-11 06:54:**

**§9 merged in.** §9 ("The bounds question, now with v07 measured") was opened
the same day as evidence toward this entry — its own opening line said "§1
stays open; this is the evidence to settle it with" — and is now resolved into
it; the measurement tables live with the full §9 entry in the Resolved
section. What it established:

- The `sig0` caveat above is settled: `DeapCma._prepare` rescales the bounds
  to [0, 1] first and *then* sets `sigma = 0.25` when `sig0` is `None`, so the
  default really is 25% of the parameter range.
- v07 is **not** saturated at its default. Sweeping all seven drive weights
  together (5 TRs, DBS off, seed 42), `caudate_dSPN` went 7.06 → 22.62 →
  69.88 Hz for weights 0.0005 / 0.001 / 0.002 — a steep response, the opposite
  of v08's flat sweep above. This also confirmed the optimized parameters
  really reach `mc.mean_weights_by_type` and `ci.mean_weights_by_type`.
- Provisional v07 bounds `[0, 0.01]` with `p0 = 0.001` (the class default) are
  in `deap_cma_opt.search_space` — ~5x above the useful top rather than v08's
  ~250,000x, but still a guess.
- The striatal target bands moved to the medication-off values on 2026-08-06
  (dSPN 12.67–37.33 Hz), which shifts the useful weight range *down* and if
  anything better centres those provisional bounds; §9's rate-loss numbers
  were computed against the old bands and are stale.
- Still open, for both versions: per-population bounds measured one parameter
  at a time. The §9 sweep confounded the seven weights, and `gpe_arky`,
  `gpe_cp` and `snr` were far outside their bands throughout — driven by the
  base currents and cluster scalings, not by the drive weights.

### 2. Design the DBS-on inference properly

*Opened 2026-08-04 06:24*

**Opened 2026-08-04 06:24:**

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

*Opened 2026-08-04 06:24 · 1 update, 2026-08-11 08:31*

**Opened 2026-08-04 06:24:**

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

**Update 2026-08-11 08:31:**

The serial time budget above predates the §22 generator rebuild (2026-08-07)
and is stale: measured after the rebuild, 5 TRs takes 2.9 min (caudate) /
3.2 min (putamen) per loop, so the serial budget is **~7 h per DBS condition**
at full length rather than ~70 h, with the parallel figure shrinking
accordingly (PLAN.md step 8, commit `f661e88`). Everything this entry actually
argues is unchanged — the 8.2x transpose measurement, the ~138 GiB target, the
pre-summing losslessness. As PLAN.md notes, cheaper generation makes the
smaller layout *more* worth doing, not less: storage, not generation time, is
now the bottleneck.

*Opened 2026-08-04 06:24 · 2 updates, last 2026-08-06 13:52*

**Opened 2026-08-04 06:24:**

`Microcircuit._simulate_distance_dependent_spike_counts` generates the missing
local GABA input assuming the surrounding striatal neurons fire at
`{"FS": 10.0, "dSPN": 37.07, "iSPN": 29.07}` Hz. Those are baked into the cache.

`get_firing_rate_loss` scores the simulated rates against bands centred on exactly
those values (20.45-53.69 and 12.99-45.15), so the design is coherent at the band
centre. But the loss is a smooth logistic, so a fit sitting near a band edge would
have neurons firing at e.g. 50 Hz while being inhibited by a surround the cache
assumes fires at 37 Hz.

**Decided:** fit first, then compare the fitted rates against the assumption and
report the mismatch. Only if they drift to an edge, consider iterating to a fixed
point (regenerate the 7 local pairs, ~30 min parallelized, then refit).

**Update 2026-08-06 11:38:** the striatal targets moved to the medication-off
values of Liang et al. 2008 (see
`experimental_data/activity_striatum/README.md`): the surround is now
`{"FS": 10.0, "dSPN": 25.0, "iSPN": 33.0}` Hz and the bands are 12.67-37.33 and
21.22-44.78. The edge scenario reads accordingly: neurons firing at e.g. 37 Hz
while inhibited by a surround the cache assumes fires at 25 Hz. The decision
above is unchanged.

**Update 2026-08-06 13:52:** FS moved 10.0 -> 10.5 Hz (the pooled estimate,
§20), so the surround is `{"FS": 10.5, "dSPN": 25.0, "iSPN": 33.0}` Hz.

### 5. Truncating schedule conversion still present in other ANNarchy classes

*Opened 2026-08-04 06:24*

**Opened 2026-08-04 06:24:**

Fixed for `TimedArray` (ANNarchy_compneuro `f215694e`): `__setattr__` converted the
schedule to integer steps with a truncating cast, so ~5% of entries landed one step
early (47 of 1000 for a dt-spaced schedule) — `4.3 / 0.1 == 42.99999999999999`. The
period had the same problem via `int(value / dt)`.

The identical conversion still exists in `TimedPoissonPopulation` and the CUDA
`TimedArray` path. Left alone to keep that commit reviewable. Fix if those are ever
used.

### 6. Workstation setup, when we move off the laptop

*Opened 2026-08-04 06:24 · 1 update, 2026-08-04 17:26*

**Opened 2026-08-04 06:24:**

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

**Update 2026-08-04 17:26:**

**lambda may be bound by compilation, not simulation.** On the laptop (16 GB)
a v07 mini-run at `--lambda 4` had two of its four `cc1plus` processes killed
by the OOM killer during the per-individual compile; `--lambda 2` was fine.
This is a different limit from the December failure, which was OOM during
*simulation* with 24 processes. Measure the peak RSS of one v07 `g++` before
choosing lambda on hinton (125 GB) and waikiki (251 GB) — and note the
per-individual compile folders mean lambda compilations run at once. If it
binds, compile once with `--compile` and pass `--skip-compile` to the run, or
stagger the compile phase.

### 7. Where the rebuild stands (state at the end of the session) — resolved 2026-08-04, moved to Resolved

### 8. Smaller cleanups

*Opened 2026-08-04 06:24 · 1 update, 2026-08-05 08:13*

**Opened 2026-08-04 06:24:**

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

**Update 2026-08-05 08:13:** two more cleanups:

- The user doesn't want dbs to appear in `model_creation_kwargs["dbs"]`, passed to `Microcircuit(dbs_condition=...)` and `CorticalInputs(dbs_condition=...)`. The model creation, i.e. creating the BGM model, should be independent of DBS. DBS is added after model creation. Currently, the dbs information during the model creation only selects the cortical firing-rate files. SO better give directly the locations of the files.
- rename dbs_depolarization, it actually hyperpolarizes the neurons, currently it's just a wrong naming

---

## From the session on 2026-08-04 (first execution of the v07 path, plan steps 4-5)

### 9. The bounds question, now with v07 measured — resolved 2026-08-11, merged into §1, moved to Resolved

### 10. The firing-rate gate threshold is not calibrated

*Opened 2026-08-04 09:39 · 2 updates, last 2026-08-04 17:26*

**Opened 2026-08-04 09:39:**

`parameters.py: firing_rate_gate = 0.5`, i.e. "roughly band-edge plausible or
better", derived from the shape of the loss (≈0.02 when every population sits at its
band centre, 0.5 at the edges) — not from observed v07 losses.

The one real v07 evaluation so far scored **0.868**, well above the gate. If early
CMA-ES generations all sit there, every individual is gated and the search sees only
the rate term until the rates come good. That may be exactly the intent, or it may
stall the fit. **Decide from the first mini-run**: log how many individuals per
generation are gated (`bold_skipped` is in every loss file) and raise the threshold
if the answer is "all of them, for many generations".

**Update 2026-08-04 15:42:**

**Extension: the bands are condition-independent, and DBS is on
during the probe.** `get_firing_rate_loss`'s `plausible_ranges` has no DBS switch,
and `dbs_stimulator.on()` now precedes the rate probe, so an on-condition
individual is judged against off-condition bands while STN and its targets are
being driven harder. The risk of gating everything is strictly worse in the on
condition than the 0.868 above suggests. The step 6 mini-runs therefore run with
`--gate-threshold 1.0` and log the off and on rate losses side by side; calibrate
the threshold — and decide whether the bands need an on-condition variant — from
those numbers, not before. Inventing on-condition bands now would bake guessed
values into an expensive fit.

**Update 2026-08-04 17:26:**

**Measured, v07 at 5 TRs, the default parameter vector:** off
**0.8705**, on **0.8742**. So the on condition is barely worse, but *both* are far
above the 0.5 gate — at the default vector every individual would be gated in
either condition, and the search would see only the rate term. The gate is
therefore untenable as configured, and the question is not on-vs-off but whether
0.5 is reachable at all once the bounds (§1, §9) let the rate term come good.
Decide together with the bounds, from a real mini-run at fitted parameters rather
than at the defaults. Inventing on-condition bands now would bake guessed values
into an expensive fit.

### 11. `run_script_parallel` is no longer used by the optimization — resolved 2026-08-11, not a BGM_22 task, moved to Resolved

### 12. Cost of building the input caches, measured — resolved 2026-08-11, superseded by the §22 rebuild, moved to Resolved

### 13. Cache state files store the cortical rate path as a bare string

*Opened 2026-08-04 09:39*

**Opened 2026-08-04 09:39:**

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

*Opened 2026-08-04 15:42*

**Opened 2026-08-04 15:42:**

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

*Opened 2026-08-04 15:42*

**Opened 2026-08-04 15:42:**

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

*Opened 2026-08-04 15:42*

**Opened 2026-08-04 15:42:**

`population_proportion = (35+23)/(70+75) = 0.4` (VTA, Berlin subject 1),
125 Hz, and `snr__thal:putamen` as the single passing fibre (after Miocinovic et
al. 2006) are hard-coded in `get_loss.py` and documented only inline. The pulse
width is 100 µs against 60 µs in the data, raised because 60 µs is below dt.

None of these is fitted, so each is an assumption the inference inherits.
Sensitivity to at least `population_proportion` and `dbs_pulse_width_us` should be
checked once a DBS-on fit exists — before any of it is written up.

### 17. `dbs_depolarization` scales with `C` in Izhikevich-2007 models

*Opened 2026-08-04 15:42*

**Opened 2026-08-04 15:42:**

The term is appended to the right-hand side, so in models written as
`C * dv/dt = ...` (the Izhikevich-2007 striatal ones) it is implicitly divided by
`C`, while in the Izhikevich-2003 BG models `dv/dt = ...` it is not. The optimizer
treats `dbs_depolarization` as one scalar in `[0, 10]` regardless.

It does not bite today — every population in the DBS footprint is
`Izhikevich2003NoisyBaseNonlin` — but it would the moment a striatal population
entered the footprint, and it would do so silently. Worth a guard in
`add_dbs_mechanisms` if that ever becomes possible.

## From the session on 2026-08-05 (DBS.md corrections)

### 18. DBS.md's line references drift silently — resolved 2026-08-11, moved to Resolved

## From the session on 2026-08-05 (documenting the model creation)

### 19. `parameters.py` labels the *planned* cache size as the current one

*Opened 2026-08-05 10:12 · 1 update, 2026-08-07 15:28*

**Opened 2026-08-05 10:12:**

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

**Caveat.** The planned-layout row assumes pre-summing collapses all cortical
regions to one stream per postsynaptic type and nothing else changes. It has not
been built, so treat it as arithmetic, not measurement.

**Update 2026-08-07 15:28:** the figures above were corrected against the cache:
the derived 5-TR size is **22.25 GB** (the ~30 MB gap to the 22.28 GB measured is
state pickles and connectivity files), so the agreement is ~0.1 %, not "four
digits"; and the full-length figure is 1 380 GB = **1.25 TiB**, not 1.26.

## From the session on 2026-08-06 (striatal firing rates)

### 20. The FS rate is not on a stated dopamine condition — resolved 2026-08-06, moved to Resolved

## From the session on 2026-08-06 (cortical proportions)

### 21. The cortical proportions had no source, and PMv was wrong by ~3.5x — resolved 2026-08-06, moved to Resolved

### 22. The spike-count generator runs at `concentration = 1.0`, and nothing checks the result — resolved 2026-08-07, moved to Resolved

## From the session on 2026-08-07 (rebuilding the input-stream generator)

*(This session's main outcome — the generator rebuild — is the resolution of
§22; see Resolved. §23–§26 record what it deliberately deferred.)*

### 23. `N_cortical_inputs_dict` (7000 SPN / 2800 FS) has no written derivation

*Opened 2026-08-07 10:55*

**Opened 2026-08-07 10:55:**

`microcircuit.py` says only *"default based on my calculations (see
zotero/google/notebooks)"*. These set the absolute scale of all cortical drive
and are baked into every cache. Oliver has the derivation and the sources.

**To do:** recheck the derivation, then write it up as an `experimental_data/`
document in the style of `activity_striatum/README.md` and
`cortical_proportions/README.md`.

Kincaid et al. 1998 (already cited for the 0.014) carries directly relevant
figures worth checking it against: ~2840 spiny neurons within one spiny cell's
dendritic volume, ~380 000 cortical axons innervating that volume, ~30.5 million
asymmetric synapses in it of which about half are cortical. Note the axon pool
size the model now derives, `M = 7000 / 0.014 = 500 000`, is the same order as
that 380 000 — so the two numbers are consistent, which is mild evidence for both.

**Not a blocker for the generator**: `N` enters the drive as a product with the
synaptic weight, which is fitted, so an error in `N` is partly absorbed. The
distribution was not absorbable by any weight.

### 24. A larger, sparser cube would make `f(d)` vary meaningfully

*Opened 2026-08-07 10:55*

**Opened 2026-08-07 10:55:**

The lattice is 227.6 µm across while the kernel reaches `Rout` = 0.57-1.2 mm, so
`f(d)` barely varies over it — 1.1x for the streams that carry >99 % of the
input. The distance dependence is derivable purely from geometry and connection
probabilities, both of which are data, so it is worth having properly.

Holding the neuron count at 1000 and lowering the density (a subsample of factor
`s`), for dSPN to dSPN with `E_total` = 1595 afferents:

| cube side | `s` | `Rin` | simulated afferents | % of GABA input simulated | `f(d)` range |
|---|---|---|---|---|---|
| 228 µm (now) | 1.00 | 114 µm | 26.2 | 1.65 % | 1.1x |
| 350 µm | 0.27 | 175 µm | 24.6 | 1.54 % | 1.3x |
| 500 µm | 0.094 | 250 µm | 21.9 | 1.37 % | 1.7x |
| 800 µm | 0.023 | 400 µm | 15.7 | 0.98 % | **4.4x** |
| 1200 µm | 0.0068 | 600 µm | 8.6 | 0.54 % | **29x** |

The cost is small *because the model is already almost entirely open-loop*: going
to 800 µm gives up 0.67 percentage points of a circuit that is only 1.65 % to
begin with. Oliver's own design for it works and is the right one — compensate
the **inner** shell as well as the outer, supplying `(1-s)*E(Rin) + E_outer`
synthetically.

**The generator no longer blocks this.** The geometric source pools reproduce
`f(d)` at any lattice size or density, so this is now a parameter change plus the
inner-shell compensation, not a rewrite.

**But state the limitation.** van Albada, Helias & Diesmann 2015
(*PLoS Comput Biol* 11:e1004490) show downscaling **cannot** generally preserve
second-order statistics: *"limitations already arise if also second-order
statistics are to be maintained… the reducibility of asynchronous networks is
fundamentally limited"*. Mean activity and correlation structure can be held by
scaling synaptic weights, but *"only over a range… limited by the variance of
external inputs to the network"*. Correlations are exactly what the larger cube
is meant to buy, so this is a warning, not a licence.

Also: the FS to SPN target count falls from 308 per FS at 228 µm to 100 at
800 µm, and FS to iSPN drops below one connection per iSPN. The FS cortical input
no longer depends on that (it is drawn now), but the simulated feedforward
inhibition does.

### 25. Open-loop compensation forecloses active decorrelation

*Opened 2026-08-07 10:55 · 1 update, 2026-08-07 15:28*

**Opened 2026-08-07 10:55:**

98 % of an SPN's GABAergic input is a precomputed stream that cannot respond to
the cortical drive, so the mechanism that keeps real striatal neurons decorrelated
despite massively shared input cannot operate here. This is *the* reason `rho` and
the cortical correlation cannot be set from the literature and left alone.

Verified sources (abstracts only):

- Tetzlaff et al. 2012, *PLoS Comput Biol* 8:e1002596 — inhibitory feedback
  actively suppresses pairwise correlations *"and hence population-rate
  fluctuations"*.
- Tetzlaff et al. 2010, *BMC Neurosci* 11(S1):P57 — the suppression acts *mainly
  below 20 Hz*, which is the band a shared cortical fluctuation occupies.
- Helias et al. 2013, *PLoS Comput Biol* 10:e1003428 — it works *"even if neurons
  receive identical external inputs"*, and what gets cancelled is *"correlations
  between the summed inputs to pairs of neurons"* — i.e. the BOLD variable.
- Bernacchia & Wang 2011, *Neural Comput* 23:1732 — same mechanism developed for
  *"striatum and globus pallidus"*; predicts zero-lag correlations ~`K^(-1/2)`
  and longer-timescale ~`K^(-1)`. At `K ≈ 2640` that is 0.019 and 3.8e-4, which
  brackets Adler's 0.004.
- Baker et al. 2019, *Phys Rev E* 99:052414 — the caveat: with *correlated*
  feedforward input a recurrent network produces **much larger** correlations
  than the asynchronous-state results suggest.

**Candidate fix**, for later: scale the recurrent striatal weights to carry the
missing mean and add an independent noise stream for the missing variance, so the
network can decorrelate. Subject to §24's van Albada limitation.

**Meanwhile**, `mc.correlation_dict` and `mc.cortical_correlation` are **0**, and
the intended procedure is the scan in `input_streams/README.md` §5: sweep input
correlation against simulated SPN output correlation and simulated BOLD
amplitude, and choose from that, with Adler's 0.004 as a validation target on the
**output**. Note this is the same self-consistency condition as §4 for the rates:
the missing-GABA presynaptic pool *is* striatal neurons of the kind being
simulated, so its assumed correlation must equal the one the model produces.

**Update 2026-08-07 15:28:**

**Addendum — the Adler values are signal correlations, not `r_sc`.**
Read from the local PDF (Methods + Fig 4): the 0.004 (MSN–MSN, Fig 4A right,
± 0.0003 SEM) and 0.06 (FSI–FSI, Fig 4C right, ± 0.009) are correlations between
trial-averaged PSTH vectors (100 ms bins across all behavioral events), computed
over all pairs *including non-simultaneously recorded ones* — tuning similarity,
not moment-to-moment spike-count co-fluctuation. The paper's spike-to-spike CCH
analyses (Figs 5–8) cover only MSN–TAN, MSN–FSI and TAN–TAN, so it contains no
MSN–MSN or FSI–FSI spike-count correlation at all. Consequence for the scan:
0.004/0.06 can serve as order-of-magnitude anchors or output-side plausibility
checks, but not as calibration targets for `ρ`, unless the simulated analysis is
deliberately matched to the paper's (signal correlation over matched windows).
The "validation target" phrasing above is thereby downgraded to an
order-of-magnitude anchor on the output. This also resolves where the FS 0.06
came from — it was never untraceable, just uncited;
`input_streams/README.md` §5 now carries the precise citation and the caveat.

### 26. Two shared fractions have no measurement behind them

*Opened 2026-08-07 10:55*

**Opened 2026-08-07 10:55:**

**`ci.shared_fraction_dict` is `{thal: 0, gpe_arky: 0, gpe_cp: 0, stn: 0}`**, so
neighbouring STN neurons receive fully independent cortical drive. That is
certainly wrong — they share corticosubthalamic axons — but there is no published
overlap fraction for corticosubthalamic or corticothalamic afferents to put
there. Kincaid's 1.4 % was measured on corticostriatal axons onto spiny neurons
and there is no reason the subthalamic arbor has the same geometry. It is a
parameter rather than a hardcoded zero so the assumption stays visible.

**The FS-to-SPN cortical shared fraction is now derived, not chosen.** With one
axon pool per region, `M = N_SPN/f`, it follows that
`corr(FS, SPN) = sqrt(N_FS N_SPN)/M = f sqrt(N_FS/N_SPN) = 0.00885` and
`corr(FS, FS) = f N_FS/N_SPN = 0.0056`. A nested block construction cannot
represent this at all — FS-FS overlap (15.7 axons) is *smaller* than FS-SPN
(39.2) — which is why the pool is realised explicitly.

Worth revisiting: Ramanathan et al. 2002 (*J Neurosci* 22:8158) and Choi et al.
2018 (*Eur J Neurosci* 48:2833) both report cortical convergence onto FS
interneurons is *higher* than onto SPNs, which the current derivation does not
capture — it assumes FS and SPN sample the same pool with the same per-axon
contact probability. If FS sample more broadly, `N_FS` is right but `M` should be
smaller for FS, raising both FS correlations.

## From the session on 2026-08-07 (verifying model_v07.md against the code)

### 27. The realised `f(d)` of the geometric pools is not checked at build time

*Opened 2026-08-07 15:28*

**Opened 2026-08-07 15:28:**

The geometric source pools of the missing-GABA streams are checked on exactly
one prediction: the realised mean degree against `E_outer`, 20 % tolerance,
raising (`microcircuit.py → _simulate_distance_dependent_spike_counts`). The
realised shared fractions are computed (`realised_shared_fractions`) but only to
feed the step-4 statistics check their mean; their agreement with the analytic
`f(d)` was verified **once, manually**, at the current 10×10×10 geometry
(commits `fd2cbde`/`148aec2` — analytic 0.0372→0.0318, realised
0.0364/0.0344/0.0314 for dSPN→dSPN), and no analytic `f(d)` code remains in the
tree.

**Why it matters:** the whole point of the geometric construction is that `f(d)`
emerges correctly at *any* geometry, which is what keeps §24 (the larger,
sparser cube) reachable. That property is currently only verified at one
geometry; a future `nx`/`density` change would silently trust it.

**Proposal:** re-add the analytic double quadrature (it exists in git history,
pre-`fd2cbde`) as a build-time check — it runs once per pair per build, so the
cost is negligible against the stream generation itself.

**Caveats:** the 20 % degree check already catches gross pool errors (a degree
that is right and a shared fraction that is badly wrong requires a subtle bug,
not a gross one); and the tolerance for the `f(d)` comparison would need the
same care as the step-4 tolerances — the realised values scatter across source
clouds, so a naive tight bound would fire on good draws.

### 28. The striatal populations start at `v = 0` — a synchronous spike at t = 0

*Opened 2026-08-07 15:28*

**Opened 2026-08-07 15:28:**

`Microcircuit.create_populations_annarchy` passes no `init`, and nothing else
assigns `v`/`u` to the striatal populations (`_set_params` skips mc components),
so all 2000 striatal neurons start at ANNarchy's defaults `v = 0, u = 0` —
80 mV above `v_r`, and above `v_peak` for the FSI. Every striatal neuron fires
one synchronous spike on the first step, `reset()` restores exactly that state,
and the volley therefore recurs at the start of **every** evaluation. The six BG
populations, by contrast, get deliberate `v_init`/`u_init` from `parameters.csv`.

**Effect:** the BOLD run absorbs it in the 2310 ms ramp-up. The 9900 ms
firing-rate probe does not — the artefactual spike sits inside the probed window
(~0.1 Hz upward bias per neuron, small against the 12.67–37.33 Hz bands), and
the synchronous volley kicks the whole recurrent circuit at t = 0.

**Fix, when wanted:** give the striatal populations inits near `v_r` (and
`u ≈ 0`) like every other population. Constraints: (a) it is a behavior change —
capture a baseline first per the convention, and expect every striatal spike
train to shift; (b) the init must be set **before** compile or re-applied after
every reset site, per the reset trap (`model_v07.md` §11); (c) bounded benefit —
the probe bias is ~0.1 Hz, so this is hygiene, not a suspect for bad fits.

### 29. `CorticalInputs`' cache validation is strictly weaker than `Microcircuit`'s

*Opened 2026-08-07 15:28*

**Opened 2026-08-07 15:28:**

`CorticalInputs._save_cortical_input_state` records neither
`shared_fraction_dict` nor any correlation field
(`cortical_correlation`/`correlation_window_ms`/`correlation_timescale_ms`), so
`_load_cortical_input_state` cannot check them: **changing any of those
parameters silently reuses a stale CI cache**, while the same change correctly
invalidates the MC cache (`microcircuit.py` records and checks all of them, and
hard-fails on state files that predate the fields). Also, the CI state's "key
set" comparison checks keys taken from the same pickle against themselves — it
catches a corrupted state file, not a configuration change.

**Fix:** record the missing fields in `_save_cortical_input_state` and compare
them in `_load_cortical_input_state`, mirroring `microcircuit.py`, including the
hard fail on their absence. **Do it before the next cache build:** no cache
currently exists, so adding fields now invalidates nothing; every day it waits,
the next cache is one parameter change away from being silently stale.

## From the session on 2026-08-10 (model_v07.md §6 follow-up)

### 30. `phi_1 = phi_2 = 0` — the striatal dopamine terms are switched off, unchecked against the source paper

*Opened 2026-08-10 11:25*

**Opened 2026-08-10 11:25:**

The three striatal neuron models
(`Izhikevich2007Humphries2009SPND1`/`SPND2`/`FSI` in CompNeuroPy's
`izhikevich_2007_like_nm.py`) carry dopamine-modulation terms — `phi_1` scales
the D1 effects (NMDA boost `1 + beta_1*phi_1`, the `phi_1*c_da*(v - E_da)`
current, the FSI `eta*phi_1` shift of `v_r`), `phi_2` the D2 effects (AMPA
attenuation `1 - beta_2*phi_2`, the `1 - alpha*phi_2` quadratic scaling, the FSI
`epsilon*phi_2` GABA attenuation). Both are class defaults `0.0`, nothing in
either model version overrides them (`parameters.csv` striatal cells are empty
and `_set_params` skips Microcircuit components; v08 instantiates the same
classes with defaults), so **every dopamine term is inert** in v07 and v08 alike
(`model_v07.md` §6.2–6.3).

**Why this is not obviously right.** The subject is a PD patient, and the
striatal target rates are deliberately the **medication-off** state of Liang et
al. 2008 (§20). In the source paper's convention `phi` is the tonic dopamine
level, so `phi = 0` means *no dopamine at all* — plausibly more extreme than the
dopamine-depleted-but-not-zero PD off-medication state the rest of the model is
calibrated to. Whether 0 is the right value for our condition, or the paper
suggests something specific for dopamine depletion, has never been checked.

**To do:** read the source paper — the docstrings link
DOI `10.1016/j.neunet.2009.07.018` (labelled "Humphries et al. (2007)" in the
docstring, but the class names and that DOI say Humphries, Wood & Gurney,
*Neural Networks* 2009; resolve the label against the actual paper, do not trust
memory) — and extract what `phi_1`/`phi_2` values it uses or suggests for a
dopamine-depleted / PD state. Then decide whether to adopt them.

**Constraints if the values change:** (a) behavior change — capture a baseline
first per the convention; every striatal spike train will move. (b) The input
caches stay *loadable* (their state checks stream parameters, not neuron
parameters), but the missing-GABA compensation and the whole rate calibration
assume the current neuron responses, so the realised rates will shift against
the Liang bands — recheck the probe, and expect §4 (missing-GABA
self-consistency) to be affected. (c) `phi` values must be set at class
instantiation or before compile — the post-compile reset trap applies
(`model_v07.md` §11).

---

## Resolved

Entries keep their numbers and full history; the stubs above mark where each
one lived. Ordered by number.

### 7. Where the rebuild stands (state at the end of the session)

*Opened 2026-08-04 06:24 · resolved 2026-08-04 09:39 (recorded retroactively on 2026-08-11, dates from git history)*

**Opened 2026-08-04 06:24:**

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

**Resolved 2026-08-04 09:39 (recorded retroactively on 2026-08-11):** every
item in the "Not yet done" list was closed by commit `23b6c79` the same
morning: the first actual execution of the v07 path (a 5-TR cache built by the
new `build_input_caches.py`, `update_time` moved to 110 ms so every simulated
stretch divides into whole chunks), plus the firing-rate gate, per-individual
logging, the CMA-ES checkpoint with `--resume`, the penalize-and-continue
failure policy, and the `deap_cma_opt.py` rewrite with the version switch,
per-version parameter counts and the DBS-on staging. The findings from that
execution are §9–§13. The "Done and verified" half was a state snapshot and has
since been superseded by `PLAN.md`.

### 9. The bounds question, now with v07 measured

*Opened 2026-08-04 09:39 · resolved 2026-08-11 06:54 (merged into §1) · 1 update, 2026-08-06 11:38*

**Opened 2026-08-04 09:39:**

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

**Update 2026-08-06 11:38:**

**Superseded in part.** The measured rates in the table stand, but
the bands they were read against do not: the striatal targets moved to the
medication-off values (dSPN band now 12.67-37.33, iSPN 21.22-44.78) — see
`experimental_data/activity_striatum/README.md`. Under the new dSPN band, 22.62 Hz
at weight 0.001 sits comfortably inside rather than near the lower edge, and 69.88
at 0.002 is well outside rather than just outside. The useful weight range
therefore shifts *down*, which if anything better centres the provisional
`[0, 0.01]` bounds with `p0 = 0.001`. **The rate-loss column (0.868, 0.837) was
computed against the old bands and is stale** — it would have to be re-measured,
and doing so needs a rebuilt cache.

**Resolved 2026-08-11 06:54:**

**Merged into §1.** This entry was never a task of its own — its opening line
already framed it as the evidence for §1, and keeping both open split the
chronology of one decision across two entries (the 2026-08-06 band update
landed here but bears directly on §1). §1's Update of 2026-08-11 summarizes
what this entry established; the measurement tables remain here. Living-
document references were repointed to §1.

### 11. `run_script_parallel` is no longer used by the optimization

*Opened 2026-08-04 09:39 · resolved 2026-08-11 08:11*

**Opened 2026-08-04 09:39:**

`deap_cma_opt.py` now runs its individuals itself. CompNeuroPy's version is
unchanged and still has the behaviour that hid the December failure: any non-zero
child exit sets an error flag, which terminates every sibling and calls a bare
`exit(1)`. It also spawns `["python", script]`, i.e. whatever `python` resolves to
rather than the running interpreter.

Left alone deliberately — it is shared code with other users. Fix it there if anyone
else hits the same wall.

**Resolved 2026-08-11 08:11:**

**Closed as not a task of this project.** The entry recorded a decision already
taken ("left alone deliberately") rather than anything BGM_22 still owes, and
its condition — the optimization no longer depending on `run_script_parallel` —
was already true when it was opened. Nothing here can be acted on from this
repo: the fix would be a CompNeuroPy change for its other users.

Re-checked today, both observations still hold in CompNeuroPy
(`system_functions.py`, `_ScriptRunner`): `run_script` spawns
`["python", self.script_path]`, and a non-zero child return sets `error_flag`,
which makes `run()` call `signal_handler`, terminate every sibling and
`exit(1)`. `git log` shows the file untouched since.

One correction to the heading, worth keeping: `run_script_parallel` is unused
by the *optimization*, not by the repo. `cortical_drive_by_bold_run.py` and
`connectivity_fit_run.py` still call it. Both are preprocessing scripts run by
hand, and since there is no `python` on PATH they only work from a shell with
the `compneuro` env activated — a trip hazard if either is ever rerun (the
cortical drive regeneration in §21 is exactly such a case), but not a defect to
fix here.

### 12. Cost of building the input caches, measured

*Opened 2026-08-04 09:39 · resolved 2026-08-11 08:31*

**Opened 2026-08-04 09:39:**

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

**Resolved 2026-08-11 08:31:**

Everything this entry measured is gone. The 24-min-per-loop timings were taken
on the copula generator that the §22 rebuild (2026-08-07) replaced — measured
after the rebuild: 2.9 min (caudate) / 3.2 min (putamen) at 5 TRs, ~7 h per
DBS condition at full length extrapolated (PLAN.md step 8, commit `f661e88`).
And `mc_ci_cache_5tr/` itself was deleted on 2026-08-06 when the striatal
rates moved to the medication-off values (§20); **no cache exists anywhere
right now**. The two constraints this entry recorded — a cache is loadable
only at the exact `n_steps` it was built for, and must cover the fixed 9900 ms
probe — remain true and live in CLAUDE.md.

The next cost measurement belongs to the §3 layout rebuild on the workstations
(PLAN step 8), which waits on the §1 bounds (step 7); a fresh cost record
should be taken there, not extrapolated from the laptop.

### 18. DBS.md's line references drift silently

*Opened 2026-08-05 08:24 · resolved 2026-08-11 08:56*

**Opened 2026-08-05 08:24:**

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

**Resolved 2026-08-11 08:56:**

Generalized into a project-wide documentation-maintenance convention in
`CLAUDE.md` Conventions: a targeted cross-reference check on every change (grep
the living documents for the changed filename, symbols, `§N`), code citations by
file + symbol instead of bare line numbers, section numbers as stable
identifiers in `TODO.md`/`PLAN.md`/`model_v07.md`/`model_v08.md` (never
renumber; insert with sub-numbers or append), and this file's historical blocks
exempt. Manual convention only, no checker script — revisit if refs rot again.

All line references migrated to symbol citations, each verified against the
current code first: `DBS.md`'s ~28 (`dbs.py` was unchanged since the 2026-08-05
recompute, so they were still accurate; the ANNarchy ones checked against
`ANNarchy_compneuro` on `timedarray-fastbuffer`, template-string cites now name
the containing template variable, e.g. `built_in_functions`, `spike_specific`,
`cpp_11_rng`) and `PLAN.md`'s two, of which `dbs.py:73` had already rotted again
— it pointed into `get_line_is_dvdt`; the clear is `mf.cnp_clear` in
`_CreateDBSmodel.__init__`. The six line refs in this file's historical blocks
stay as written per the exemption. `model_v07.md`, `model_v08.md`, `CLAUDE.md`
and the `experimental_data/` READMEs already cited by symbol and needed nothing.

### 20. The FS rate is not on a stated dopamine condition

*Opened 2026-08-06 11:38 · resolved 2026-08-06 13:47 · 1 update, 2026-08-06 13:52*

**Opened 2026-08-06 11:38:**

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

**Resolved 2026-08-06 13:47:**

All five sources were read. **10.0 Hz is confirmed for the unmedicated
parkinsonian condition** and the derivation is written up in
`experimental_data/activity_striatum/README.md` §"The FS rate". In short:

- The three primate sources (Marche & Apicella 2021, Yamada 2016, Adler 2013) are
  all **normal** animals; n-weighted over 151 FSIs they give 10.6 Hz (SEM ~0.7).
- The two rodent sources are the dopamine-condition correction, and — together
  with Mallet et al. 2006, which was *not* in the folder and which both He and
  Yamada cite as the reference for FSI changes in parkinsonism — they agree the
  FSI **baseline** rate is unaltered by chronic depletion. He's large drop is
  transient (weeks 2–3, gone by >3 weeks). Factor 1.0.
- The `str_fsi` band was `(5, 15)`, hand-set. It is now `(3.25, 16.75)` —
  mean ± 1 SD like the SPN bands, with the relative spread from the only source
  that reports an SD (Marche, CV 0.675).

**What is left, and it is not small.** The dopamine condition is *imported from
rodents*, not measured in primates: no parkinsonian-primate striatal FSI recording
exists. So FS is structurally weaker than the SPN rates, which are measured in
chronically parkinsonian monkeys. The specific worry is consistency — chronic
denervation lifts primate MSNs ~20-fold (Liang) and we assert FS is unmoved in the
same striatum. Hernandez shows exactly that dissociation within one dataset, but in
rat, where the MSN elevation is far smaller. If the striatal rates ever become a
suspect in a bad fit, this is a place to look, alongside §4.

Sensitivity bound: 8–12 Hz. Changing the value invalidates the v07 caches exactly
as the SPN rates did — free right now, since none exist.

**Update 2026-08-06 13:52:** the value moved 10.0 -> **10.5 Hz** after all — the
pooled estimate, set because no cache existed to invalidate and a later move
would not be free. The `str_fsi` band follows: `(3.42, 17.58)`, recomputed as
10.5 ± 10.5 · (8.5/12.6) (Marche, CV 0.6746). Any further change invalidates the
v07 caches exactly as the SPN rates did — still free while none exist.

### 21. The cortical proportions had no source, and PMv was wrong by ~3.5x

*Opened 2026-08-06 14:29 · resolved 2026-08-06 19:40 · 2 updates, last 2026-08-06 19:59*

**Opened 2026-08-06 14:29:**

`model_v07.md` §7.5 lists the per-region cortical proportions without a citation,
because there is none. They are hand-set round numbers, duplicated verbatim in
three places that have to agree:

- `CompNeuroPy/.../striatal_microcircuit/microcircuit.py:150`
- `CompNeuroPy/.../striatal_microcircuit/cortical_inputs.py:260`
- `striatal_microcircuit_requirements/cortical_firing_rates/cortical_drive_by_bold.py:27`
  (`MIXING_FACTORS`)

They do two different jobs. In the microcircuit and `CorticalInputs` they set
`N_eff = round(p * N_cortical_inputs)`, i.e. how many of a receiver's 7000 (SPN)
or 2800 (FS) cortical afferents come from each region, and a region with `p <= 0`
is skipped entirely. In `cortical_drive_by_bold.py` they are the weights of the
linear mix that produces the stored `caudate_rate` / `putamen_rate` series. That
third use is the reason this cannot be changed casually: the committed
`firing_rates_matlab_condition-{on,off}.npz` was generated with the current
weights — verified, `corr(stored caudate_rate, old mix) = 1.0000` — and
regenerating it needs MATLAB (`matlabengine`). Editing the constant without
regenerating leaves code and data silently disagreeing.

**What the quantity should be.** For loop L and cortical ROI r, the fraction of
corticostriatal afferents onto a striatal neuron in L that originate in r,
renormalised over just the seven ROIs the Berlin data provides. Everything else
that projects to the striatum — cingulate, insula, temporal, posterior parietal,
orbital and ventrolateral prefrontal — has no ROI here and is renormalised away.
In the macaque tracer data those excluded sources are 30-60% of all corticostriatal
cells, so the renormalisation is not a rounding detail; the seven ROIs stand in
for the whole cortex.

**Primary evidence — quantitative macaque retrograde tracing.** These are the only
sources that report corticostriatal input as a percentage of labelled cells rather
than as a topography.

Borra et al. 2022, *J Neurosci* 42:7060 (doi:10.1523/JNEUROSCI.0071-22.2022),
Table 2, % of ipsilateral labelled cells by region group:

| injection | rostral cing. | prefrontal | motor | parietal | insula | temporal | caudal cing. |
|---|---|---|---|---|---|---|---|
| caudate, lateral head | 21.5 | 37.8 | 8.4 | 6.1 | 4.8 | 13.5 | 7.0 |
| caudate, medial head | 30.6 | 48.5 | 4.1 | 0.1 | 2.3 | 9.4 | 4.4 |
| caudate, body | 7.0 | 4.8 | 74.1 | 11.5 | 0 | 0 | 2.6 |
| putamen, rostral | 23.0 | 17.5 | 37.8 | 8.1 | 7.0 | 4.3 | 2.3 |
| putamen, dorsal motor | 15.3 | 0.7 | 61.9 | 16.6 | 1.4 | 0 | 4.1 |
| putamen, middle motor | 8.3 | 1.0 | 64.5 | 21.6 | 2.0 | 0.8 | 1.8 |
| putamen, middle motor | 9.4 | 0 | 72.1 | 15.1 | 0.5 | 0.1 | 2.8 |
| putamen, midventral motor | 2.6 | 0.5 | 75.5 | 18.6 | 1.2 | 0.8 | 0.8 |

Borra et al. 2021, *J Neurosci* 41:1455-1469 (doi:10.1523/JNEUROSCI.1475-20.2020),
Table 3, splitting that "motor" column per area for the motor putamen (F1 = M1,
F2 = PMd, F3 = SMA, F4/F5 = PMv, F6 = preSMA, F7 = pre-PMd):

| case | 24c/d | F6 | F7 | F3 | F2 | front. operc. | F5 | F4 | F1 |
|---|---|---|---|---|---|---|---|---|---|
| 75 dorsal | 14.4 | 0.8 | 0.3 | 12.7 | 7.1 | 2.2 | 2.5 | 2.0 | 34.1 |
| 71r middle | 14.4 | 0.8 | 0.5 | 13.2 | 6.5 | 2.4 | 7.9 | 3.3 | 26.9 |
| 71l midventral | 2.5 | 0.1 | — | 6.9 | 0.7 | 7.6 | 33.4 | 8.2 | 18.6 |
| 61 | 12.3 | 3.7 | 1.2 | 10.6 | 9.3 | 16.5 | 10.0 | 3.0 | 2.7 |

Supporting topography, used to split the bins the tables leave grouped: Takada et
al. 1998 *Exp Brain Res* 120:114 (M1 lateral putamen, SMA medial putamen, PMd/PMv
dorsomedial); Inase et al. 1999 *Brain Res* 833:191 (preSMA to rostral caudate and
the cell bridges, segregated rostral to the SMA zone); Calzavara et al. 2007
*Eur J Neurosci* 26:2005 (areas 9 and 46 to caudate head, caudal 46 extending into
rostral putamen; PMdr to dorsal and lateral caudate); Flaherty & Graybiel 1995
*J Neurophysiol* 74:2638 (M1's striatal projection magnification ~2x that of each
individual S1 subarea, so summed S1 lands well below M1).

**Secondary — human.** Human tractography is nearly all qualitative (Leh et al.
2007, Lehericy et al. 2004, Draganski et al. 2008 report topography, not
fractions). The one human source with per-pathway percentages is the connectomic
analysis of Cacciola et al. 2017, *Front Neuroanat* 11:85
(doi:10.3389/fnana.2017.00085, n = 15, CSD tractography), in the coarse Desikan-Killiany
parcellation, which cannot separate M1/PMd/PMv (all "precentral") or
SMA/preSMA/dlPFC (all "superior frontal"). Two ratios survive that coarseness and
both say the macaque numbers need a nudge: caudate rostral-middle-frontal 25.8%
against precentral 3.7% (more prefrontal in the human caudate than in the
macaque), and putamen postcentral 4.6% against precentral 9.0% (more S1). This
matches the reported expansion of prefrontal corticostriatal projections in humans
(Neggers et al. 2015 *J Neurophysiol* 113:2164-2172; Balsters et al. 2020 *eLife*
9:e53680).

**Recommended replacement**, macaque tracer percentages as the anchor, nudged
toward the human ratios above. Caudate weighted 0.75 head / 0.25 body, putamen
0.7 motor / 0.3 rostral:

| region | caudate (now -> rec.) | putamen (now -> rec.) |
|---|---|---|
| dlPFC | 0.45 -> **0.55** | 0.05 -> **0.10** |
| preSMA | 0.25 -> **0.15** | 0.10 -> **0.05** |
| PMd | 0.15 -> **0.18** | 0.15 -> **0.11** |
| PMv | 0.10 -> **0.04** | 0.05 -> **0.18** |
| SMA | 0.04 -> **0.06** | 0.25 -> **0.15** |
| M1 | 0.01 -> **0.02** | 0.30 -> **0.28** |
| S1 | 0.00 -> **0.00** | 0.10 -> **0.13** |

Both columns still sum to 1, which matters: the seven rate series are each
normalised to mean 5.0 Hz, so any mix that sums to 1 leaves the mean drive
unchanged and only the variance and timing move. Caudate S1 stays at exactly 0 —
S1 to caudate is absent in the tracer data, and 0 also keeps that stream from
being built at all, so the caudate keeps one stream fewer than the putamen.

**The one entry to argue about is putamen PMv, 0.05 -> 0.18.** It rests on F4+F5
being a major putaminal input, which is true but strongly zone-dependent: F5 is
2.5% and 7.9% of labelled cells in the two arm/hand motor cases and 33.4% in the
midventral orofacial case. A whole-putamen ROI contains that ventral sector, which
is why the average is high, but the value is sensitive to how the sectors are
weighted. Range 0.10-0.24. Every other entry is stable to within about 0.03.

**Measured sensitivity — this may not be worth spending MATLAB on.** The seven
deconvolved cortical series are only moderately correlated (off-diagonal r: min
0.16 for dlPFC-M1, median 0.46, max 0.90 for PMd-PMv), so the regions are
genuinely distinguishable. But the *mixed* drive barely moves:
`corr(old mix, new mix)` = **0.995** for the caudate and **0.982** for the
putamen, with the standard deviation changing by 1-5% and the mean not at all. So
for v08, which sees nothing but the mixed series, the correction is close to a
no-op. It bites in v07, where the proportions also set the per-region stream
sizes: caudate PMv 700 -> 280 afferents, putamen PMv 350 -> 1260, putamen dlPFC
350 -> 700, caudate dlPFC 3150 -> 3850 (per SPN, out of 7000).

One consequence cuts against us and should be stated: the corrected proportions
make the two loops *more* alike, `corr(caudate mix, putamen mix)` rising from
**0.798** to **0.843**. The proportions are the only physical difference between
the loops (`model_v07.md` §7.5), so this shrinks the contrast the inference
depends on. It is the
honest number, not a reason to keep the old one, but it means the loop separation
is doing less work than the current table implies.

**Not applied.** Changing it means editing all three sites *and* regenerating both
`firing_rates_matlab_condition-{on,off}.npz` under MATLAB, and it invalidates
every v07 input cache (none exist right now, so that part is free — same window
that made the striatal-rate change in §20 cheap). Given the 0.98-0.995 correlation
above, the defensible order is: do it when MATLAB is next in hand, before the
caches are built, not as a standalone errand.

**Update 2026-08-06 15:10:** the recommended table was applied, and the
duplication removed. The proportions now live once, in
`BOLD_optimization/parameters.py` under `cortical_proportions_dict`, and all
three former sites read them from there: `get_loss` threads
`mc.cortical_proportions_dict` into both `Microcircuit` and `CorticalInputs`,
and `cortical_drive_by_bold.py` imports the same dict instead of defining
`MIXING_FACTORS`. Both CompNeuroPy classes lost their defaults and call
`spike_input_cortex.validate_cortical_proportions()`, which raises on a missing
mapping, a negative share, or a sum other than 1. Verified end to end: both
columns sum to 1, and the kwargs produce the intended per-region afferent counts
(caudate dSPN 3850 dlPFC / 1260 PMd / 1050 preSMA / 420 SMA / 280 PMv / 140 M1 /
0 S1). No v07 cache existed, so nothing was invalidated — the same free window
as §20.

Note the rate `.npz` is *not* in git — `.gitignore:25` excludes
`cortical_firing_rates_data/`. It is a regenerable artefact with no committed
copy to fall back on, so back the folder up before rerunning
`cortical_drive_by_bold_run.py` (`create_data_raw_folder` deletes it after a
`y/n` prompt with a 60 s timeout).

**Still open at this point: the rate `.npz` had not been regenerated.** MATLAB
R2025b is installed on the laptop but `matlab.engine.start_matlab()` does not
come up unattended — it hangs for 8+ min with `MathWorksServiceHostWindow`
spinning at ~79% CPU off-screen, i.e. waiting on an interactive MathWorks
sign-in, and a `-nodisplay` attempt returned `License Error: Licensing
shutdown`. So `firing_rates_matlab_condition-{on,off}.npz` still held
`caudate_rate` / `putamen_rate` mixed with the **old** proportions. That affects
v08 only — the mixed series is its entire cortical drive — while v07 is driven
by the seven per-region series and reads `caudate_rate` only for its length. A
guard was added so the mismatch cannot go unnoticed: `cortical_drive_by_bold.py`
writes `cortical_proportions_json` into both `.npz` files, and
`get_loss.infer_max_sim_time_ms` raises if that record disagrees with
`parameters.py` (a printed warning for older files that predate the key).

**Resolved 2026-08-06 19:40:**

**The rate `.npz` was regenerated by Oliver on 2026-08-06**, interactively —
the only way it works here, per the MATLAB behaviour above. Worth remembering:
**regenerating this data needs an interactive MATLAB session**, it cannot be
scripted from a headless shell.

Verified against a backup of the pre-regeneration folder, for both conditions:

- `cortical_proportions_json` is present in all four `.npz` files and matches
  `parameters.py`, so `get_loss.infer_max_sim_time_ms` now runs silently.
- The seven per-region `<region>_rate` series are **bit-identical** to before.
  This is the important one: v07 is driven by those, so the whole change is a
  no-op for it, and the regeneration introduced no incidental drift from the
  MATLAB HRF or the deconvolution.
- `caudate_rate` / `putamen_rate` equal the new mix exactly (atol 1e-12), and
  moved by `corr(old, new)` = 0.9949 / 0.9816 (off) and 0.9961 / 0.9909 (on) —
  matching the 0.995 / 0.982 predicted from the sensitivity analysis above.
  These feed v08 only.

So there is nothing left open here. What remains is judgement, not work: putamen
PMv is the entry to revisit if the fit misbehaves (range 0.10-0.24), and the
loop-similarity consequence above still stands.

**Update 2026-08-06 19:59:** the measured-sensitivity paragraph above measured
the amplitude effect on the off condition only, and understated it. Corrections:
the off-condition `corr(old mix, new mix)` = 0.995 / 0.982 stands, and the on
condition later came out at 0.996 / 0.991. The mean cannot move — every column
sums to 1 and every series has mean 5 Hz — so in *timing* the correction really
is close to a no-op for v08. Amplitude is the exception: the standard deviation
moves +1.0 % / +4.6 % (off caudate / putamen) and +10.7 % / **−11.7 %** (on).
The on-condition putamen drive is now ~12 % less modulated than before —
absorbable by the fitted input weight, but asymmetric between the two DBS
conditions the inference compares, which is worth remembering. The change
accordingly bites hardest in v07, through the per-region stream sizes listed
above.

### 22. The spike-count generator runs at `concentration = 1.0`, and nothing checks the result

*Opened 2026-08-06 14:31 · resolved 2026-08-07 10:55*

**Opened 2026-08-06 14:31:**

`ReceiverSimulator.generate_p_matrix` draws each receiver's per-bin spike
probability from `Beta(p·c, (1−p)·c)`, where `c` is the `concentration` argument
of `spike_input_cortex.simulate_receiver_counts_*`. **No caller ever passes it**,
so `c = 1.0` everywhere; in both call sites an explicit `concentration=1000.0`
sits commented out one line below the call:

- `CompNeuroPy/.../striatal_microcircuit/microcircuit.py:1303` (cortical)
- `CompNeuroPy/.../striatal_microcircuit/microcircuit.py:1670` (missing GABA)

`CorticalInputs` never had the line at all.

At `c = 1` the Beta variance is `p(1−p)/2`, so with `p ≈ 5·10⁻⁴` the per-receiver
probability has a standard deviation ~30x its own mean. The mean count is
preserved exactly — which is why nothing downstream has ever complained — but the
distribution around it is not the intended near-Binomial. Measured (see
`model_v07.md` §7.4 for both tables), on a cortical stream of `N_eff = 3150`
sources at 5 Hz, `dt = 0.1 ms`, expected 1.575 counts per bin:

| | mean | SD | max | bins exactly 0 | Fano | realized corr (target 0.014) |
|---|---|---|---|---|---|---|
| `c = 1.0` | 1.540 | 48.65 | 3131 | 99.6 % | 1537 | 0.0002 |
| `c = 1000.0` | 1.574 | 2.55 | 42 | 49.1 % | 4.2 | 0.0083 |

Two things follow. **The drive is delivered as rare, enormous conductance
jumps** — bins in which all 3150 presynaptic neurons fire within one 0.1 ms step
are routine — rather than as a dense input stream. And **the shared fractions are
largely destroyed**: the 0.014 of Kincaid et al. arrives as 0.0002, and on a
missing-GABA stream a target of 0.5 arrives as 0.15. The §7.3 machinery that
computes `E_shared(d)` by double integration is doing careful work that the
final draw then discards.

Which value is right is genuinely open — `c = 1000` is a guess that happens to
look sane, not a calibration, and even there the Fano factor is 4.2 rather than
~1. The honest fix is to state a target input statistic first (Fano factor and
pairwise count correlation of the real presynaptic pools), then choose `c` to
meet it, then check the striatal rates still land in their bands. Until then this
is an unvalidated free parameter sitting under every input the model receives.

**Cost of changing it:** it invalidates every v07 input cache, exactly as the
striatal rates in §20 did. None exist right now, so the change is free today and
expensive after the caches are built. If it is going to be looked at, it should be
looked at before the rebuild, together with §3 and §21.

**Resolved 2026-08-07 10:55:**

The item as written was "`concentration = 1.0`, and nothing checks the result".
Investigating it turned up three more faults and one consequence that was not
visible from the item, all now fixed. The contract the generator has to satisfy
is written up in `experimental_data/input_streams/README.md`; that document did
not exist and is the "state a target statistic first" half of the item.

**What was actually wrong.**

1. `concentration = 1` is not a mis-set constant. It is a *per-receiver private*
   rate fluctuation of variance factor `1/(c+1) = 0.5`, i.e. a third dispersion
   source with no biological counterpart, 125x larger than `rho = 0.004`.
   Measured on dlPFC to caudate dSPN: **Fano 1922**, 99.6 % of bins exactly zero,
   single 0.1 ms bins holding 3131 of 3850 possible spikes.
2. The sharing was injected twice, through two independent Gaussian copulas, with
   the shared fraction fed in as a *Gaussian* correlation coefficient. Measured:
   **0.014 in, 0.00009 out**.
3. `rho` swamped `f` regardless. `corr = (f(1-rho)+N rho)/((1-rho)+N rho)`, and at
   `N rho = 6.3` that is **0.868 whatever `f` is**. FS to dSPN was worse: Fano 30.9,
   corr 0.974. The 50-point nested quadrature computing `E_shared(d)` was
   decorative.
4. Correlations had no timescale. `rho` was white at `dt`, so the model's `r_sc`
   was identical at 0.1 ms and 3 s, while every value it is calibrated against
   (Cohen & Kohn 2011 Table 1) is measured at 66-3000 ms.

**Why it mattered more than it looked.** `get_loss.py` builds
`BoldMonitor(mapping={"I_CBF": "I_v"})` and `I_v` is the *net* synaptic current,
so `BOLD ∝ sum_i I_v,i` and `Var(sum) = N v (1 + (N-1) r)`. At `N = 486`, going
from `r = 0` to `r = 0.99` is a **481x** change in BOLD variance. The input
correlation is the dominant determinant of the model's only output.

**What replaced it.** Every stream is now generated by realising the presynaptic
pool explicitly and letting overlaps produce the correlations, instead of
computing a correlation and imposing it through a copula:

- cortical streams: all receiver types of a region sample one pool of
  `M = N_eff / f` axons, drawn per bin as `k ~ Binomial(M, p)` then
  `Hypergeometric(M, N_i, k)` per receiver;
- missing-GABA streams: virtual sources scattered around the lattice, connected
  with the pair's own distance kernel, so `f(d)` emerges;
- the shared rate modulation is an OU process parameterised by
  `(tau_c, r_sc, T_meas)`, so a correlation is always accompanied by the window
  it was measured at;
- FS cortical input is drawn, not derived from the FS to SPN weight matrix;
- every stream is checked against the closed forms at build time and **raises**,
  with the measured values written into the cache state.

Deleted along the way: `_expected_shared_for_d`,
`_build_distance_dependent_shared_fraction_matrices`, `_derive_fs_cortical_inputs`,
`_calculate_max_chunk_size`, the FS debug validators, and the entire Beta/copula
layer of `spike_input_cortex`.

Generation is also **~6x faster**: ~7 h per DBS condition at full length against
~42 h before, because `beta.ppf`, `binom.ppf` and both copula draws are gone.
