# TODO

The single home for future work on this project: findings we deliberately
postponed *and* the tasks still ahead. Each entry says what was observed or
what is to be done, why it matters, and what still has to be decided. The
**Roadmap** section below holds the intended order; open entries follow,
grouped by the session that produced them, newest at the bottom; closed
entries live in the **Resolved** section at the bottom of the file, which
doubles as the project's historical record.

## Maintaining this file

- **Numbers are never reused.** An entry keeps its number for life, so `§N` is
  a stable reference everywhere, including for resolved entries. A gap in the
  active numbering means the entry was resolved; a one-line stub marks its old
  place.
- **Adding an entry:** take the next unused number and open it under a
  `## From the session on <date> (<topic>)` heading at the bottom of the open
  section (reuse the heading if the session already has one), first block
  `**Opened <YYYY-MM-DD HH:MM>:**`. Then place it in the Roadmap — at an
  ordered position with the reason, or in the no-assigned-order list. An open
  entry that appears nowhere in the Roadmap is a rule violation.
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
  a one-line stub heading in its session position. Rewrite the Roadmap:
  remove the entry and adjust whatever its removal unblocks. Then check the
  cross-references: `§N` mentions in the living documents — `CLAUDE.md`,
  `DBS.md`, `model_v07.md`, `model_v08.md`, the
  `experimental_data/` READMEs, code comments — get annotated
  "(resolved <date>)". Historical Opened/Update blocks inside this file are
  never edited retroactively, so references in them stay as written.
- **The Roadmap section is exempt from the append-only rules.** It is
  rewritten freely in place as the ordering changes. It may only *order* the
  entries and state blocking/enabling relations between them — the substance
  stays in the entries, and nothing may reference the roadmap itself: only
  `§N` entries are referenceable.

All Opened/Update/Resolved timestamps up to 2026-08-10 were reconstructed on
2026-08-11 from the git history of this file; they are commit times, which can
lag the session that produced the finding by a few hours.

## Roadmap

The one part of this file that is rewritten freely in place (see the exemption
above). It holds only ordering — which entries block which — with the reasons
for the order; the substance lives in the entries. It replaced `PLAN.md`'s
step sequence when that file was dissolved into this one on 2026-08-11; §33
maps the old "step N" numbers that historical blocks and commit messages still
use.

Reordered 2026-08-13 into three phases: finalize the model, then make it
fast, then clarify and run the fits. The reason for the order: nearly any
model change invalidates the input caches, and a structural finding from the
community review would invalidate the fits — so nothing expensive
(full-length caches, workstation runs, fits) is built until the model is
final. This also repositions §1 by its own logic ("before anything
expensive"): the expensive part now starts in phase 3.

**Phase 1 — finalize the model.** (Its first item — §29 + §27 + §13, the
cache-validation hardening — was completed 2026-08-13, before any cache was
built through the weaker path. Its second — §34, the community-conventions
review — was completed the same day, and its step 3 was rerun as round 2 on
2026-08-14/15 under hardened evidence requirements; the 32 round-2 findings
now feed the triage below.)

1. **The community-review triage.** §34 is resolved and its output — after
   the step-3 rerun of 2026-08-14/15 — is
   `community_review/round2/synthesis.md`: 32 findings F1–F32 (numbers local
   to that document; the round-1 synthesis under `round1/` is superseded,
   its numbering included), ranked in four tiers by how many of the seven
   seats raised each, each carrying an evidence class (experimentally
   grounded / methodological / difference), with change proposals only where
   the first two classes license them. This step decides which are accepted.
   Every accepted finding **spawns its own numbered entry**, which then joins
   the verdict pass below; a rejected one is recorded as rejected with its
   reason in the spawning entry or, where no entry is warranted, nowhere —
   the synthesis is not itself referenceable, so anything meant to survive
   must become an entry. Runs before any model verdicts because the findings
   bear on exactly what the verdicts rule on (§23, §24, §25, §26, §28, §30).
   Hard blocker on everything downstream, deliberately without a timebox.

   Orderings the round-2 synthesis's reading guide implies, kept here because
   they are ordering and nothing else: the input-correlation scan stays the
   one hard blocker on §32 and is already carried by **§25**; the rate-band
   derivation document (accepted 2026-08-26 as **§35**) must precede §10's
   gate calibration; five findings
   would invalidate input caches if accepted (the striatal-kernel state
   refit — accepted 2026-08-27 as **§43**, which depends on §39 —
   the STN cortical-proportion table — accepted the same day as **§45** —
   the `ci.n_*` derivation of §8,
   the cube size already carried by §24, and the medication state — accepted
   2026-08-27 as **§39**, which therefore precedes every full-length build
   and feeds §30), so they must be settled inside this phase rather than
   after phase 2 builds them; **§46**'s decision rule must likewise be fixed
   here, before its bracketing comparison rides along with phase 2's build;
   and the acceptance criterion for "DBS changed
   this parameter" must be fixed in §2 before any on-fit is interpreted
   (accepted 2026-08-26 into §2's update of that date).
2. **The verdict pass** — §14, §15, §16, §17, §23, §24, §25, §26, §28, §30,
   plus everything the triage spawned (so far: §35, §36, §37, §38, §39,
   §40, §41, §42, §43, §44, §45, §46, §47). Each entry gets an explicit
   verdict:
   **fix now** (implemented within this phase) or **accepted limitation**
   (rationale documented; the entry stays open on its own trigger). §16 is
   pulled into the model phase deliberately — unvalidated DBS constants are
   a model gap, not a write-up-time sensitivity check.
3. **The validation run pair — the phase's exit criterion.** v07, DBS off
   *and* on, short caches (`--n-trs 5`, built on the laptop and rebuilt
   cheaply after model changes), each producing firing rates and BOLD.
   Checked against the targets: rates inside the `get_firing_rate_loss`
   bands or deviations explained, stream statistics matching
   `experimental_data/input_streams/README.md`, §36's regime diagnostics
   computed and recorded, and a demonstrated on-vs-off
   difference in the putamen loop. Outputs kept as artefacts — in this
   project "written" has not meant "run". At 5 TRs the BOLD side proves only
   that the signal is produced, not that it is meaningful; that is accepted.

**Phase 2 — make it fast.** Short-cache builds are phase-1 laptop work; only
the full-length builds belong here.

4. **§6 + §3 as one pass.** Workstation setup (push the repos, carry the
   patched ANNarchy across), implement §3's smaller layout, build the
   full-length caches for both DBS conditions directly in that layout —
   building 2 x 1.25 TiB in the old layout just to redo it is waste — and
   time one full evaluation per machine. The timing is the decision gate:
   further speedup work opens as new entries only if the fits would be
   infeasibly slow. Exit: validated full caches for both conditions on a
   workstation, evaluation cost known.

**Phase 3 — clarify and run the fits.**

5. **§1 — the bounds.** The fits must not start on unvalidated bounds, and
   the per-population sweep has to be redone anyway because the DBS retrofit
   moved every number it was measured on.
6. **§31 — the five-generation mini-run.** After §1, so the numbers that
   calibrate the **§10** gate threshold come from sensible sampling rather
   than the saturated regime. Also the first end-to-end CMA-ES exercise on a
   workstation — phase 2's timed evaluation covers everything below the
   orchestration layer.
7. **§32 — the fits: DBS-off, then DBS-on.** Blocked by everything above.
   The first pair stays as §2 already decided: all free parameters,
   pipeline-proving.
8. **After the first fit:** **§4** (missing-GABA self-consistency against
   the fitted rates), **§2** (the DBS-on inference design — decides what the
   on-fit may claim).

No assigned order — each entry states its own trigger: §5, §8. Verdict-pass
entries whose verdict is "accepted limitation" rejoin this list on their own
triggers. (By definition this is every open entry not ordered above; keep
the enumeration complete when entries open or resolve.)

---

## From the session on 2026-08-03 (reviving BGM_v07 for the BOLD optimization)

### 1. Validate the optimization bounds — for both v07 and v08

*Opened 2026-08-04 06:24 · 2 updates, last 2026-08-11 10:04*

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

**Update 2026-08-11 10:04:**

Carried over from the dissolution of `PLAN.md` (§33): the sweep has to be
**redone on the current numerics** before bounds are read off it — the DBS
retrofit (old plan step 6) moved every number the model produces (off-condition
total 1.5430 → 1.5442 at the reference vector), and all tables in §9 predate
it. The useful range those tables suggest for the seven v07 drive weights,
roughly [5e-4, 2e-3], carries the same caveat.

### 2. Design the DBS-on inference properly

*Opened 2026-08-04 06:24 · 3 updates, last 2026-08-27 13:51*

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

**Update 2026-08-11 10:04:**

Carried over from the dissolution of `PLAN.md` (§33), the flip side of the
staging decision above: it buys a **free control**. Since the caudate loop is
untouched by DBS and keeps its off-condition weights, its on-vs-off BOLD change
must be explained entirely by its cortical drive — a falsifiable prediction the
inference gets for free. Proven exact on v08 (same drive, only DBS parameters
differing: every caudate population identical to 2 dp while nine putamen
populations move; see §33's step-6 record).

**Update 2026-08-26 11:51:**

The community-review triage accepted round-2 F2 of
`community_review/round2/synthesis.md` (six seats; evidence class
methodological, uncontested) into this entry: the central claim currently
has no acceptance criterion. The planned ~10-seed stability rerun of one
winning vector (§32) measures simulator stochasticity, not optimizer
multi-modality — the lab's own predecessor (Maith et al. 2021 §2.4, §3.2)
ran twenty optimization processes per fitted dataset and asserted parameter
differences only across groups with t-tests, FDR and effect sizes, and
Liénard 2024 §5.4 / Girard 2021 §2.5 document >1000 equally-plausible
parameterizations reducible to 15 base solutions, all carried through every
analysis. The off-fit's solution set enters the on-vs-off difference exactly
as the on-fit's does.

What this entry must now decide and fix **before §32's on-fit is
interpreted** (the review's seven commitments; 1–3 change how many fits are
needed, so they are decided first):

1. **N independent restarts per condition** — both conditions, not just on.
2. **Per-parameter restart scatter** reported beside every fitted vector.
3. **The pre-stated acceptance rule**: a parameter is reported as "changed
   by DBS" only where its on-off delta exceeds that scatter (plus a
   seed-robustness re-evaluation of the final vectors).
4. **The 2×2 cross-condition evaluation** (off-parameters on on-data and
   vice versa; Maith 2021 Table 4's discriminability check, single-subject
   analogue): if the on-fit does not beat the off-parameters on on-data by
   more than the restart scatter, the refit captured noise.
5. **Parameter recovery on synthetic data**: simulate BOLD at known DBS
   parameters, refit, report which of the 13 on-stage parameters are
   recoverable at all at this noise level — the cheapest test of the
   central claim (Dunovan 2019's resampled-pipeline discipline).
6. **A held-out-TR split**, with per-region correlation uncertainties
   reported next to every on-vs-off delta (a 309-sample correlation has
   SE ≈ 0.06; no artifact currently records this).
7. **The caudate free control pre-registered as pass/fail**: state before
   the on-fit what caudate BOLD correlations and rates must do for the
   inference to stand (it is currently a prediction without a threshold).

The option family in the Opened block (single-mechanism scans, multi-start,
L1) remains the solution space for 1–3; the review adds no new option there,
it adds the commitment and the checks 4–7.

**Update 2026-08-27 13:51:**

The triage's acceptance of round-2 F12 (§44, opened today) bears directly on
item 7 above: **the caudate free control is an upper bound on channel
independence, not a clean reading.** Two measured facts, both restated with
their sources in §44 — the reviewed anatomy's strongest cross-channel route
runs associative → motor, precisely the direction the control assumes away,
and the subject's own VTA overlaps the associative STN at ≈ 0.09, which the
two-loop design rounds to zero. Whatever pass/fail this entry pre-registers
for the caudate loop must therefore be stated as a bound, and §44 carries
the work of quantifying the 0.09 component so the bound is a number rather
than a caveat.

### 3. Regenerate the input caches with a transposed layout

*Opened 2026-08-04 06:24 · 2 updates, last 2026-08-11 09:16*

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

**Update 2026-08-11 09:16:** the expected size converged on **~120 GiB per DBS
condition** — §19's cross-checked arithmetic (8 684 rows x 7 161 000 steps x
2 B = 124 GB = 116 GiB; arithmetic, not measured, with §19's pre-summing
caveat). The ~138 GiB above had no recorded derivation and is superseded. §19
(resolved today) propagated the figure to `parameters.py` and `PLAN.md`.

### 4. Check the missing-GABA self-consistency after the first fit

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

*Opened 2026-08-04 06:24 · 2 updates, last 2026-08-11 10:04*

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

**Update 2026-08-11 10:04:**

Push status, verified today while dissolving `PLAN.md` (§33): **nothing since
2026-08-04 has been pushed** — `olimaol_develop` is 33 commits ahead of
`origin` in BGM_22 and 8 in CompNeuroPy. Pushing both is how the code reaches
the workstations, so it is the first item of this entry's move; the patched
ANNarchy still needs the separate route described above.

### 7. Where the rebuild stands (state at the end of the session) — resolved 2026-08-04, moved to Resolved

### 8. Smaller cleanups

*Opened 2026-08-04 06:24 · 2 updates, last 2026-08-27 11:25*

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

**Update 2026-08-27 11:25:**

The `normalize_input` item above was worked out while triaging the round-2
review into §42, and splits in two. The window-alignment question — 2000 ms
against a 2310 ms TR and a 2310 ms ramp-up — stays here as the cleanup it
is. The *consequence* does not: because the balloon model receives exactly
zero drive during the baseline window and then a step onto relative
deviations, its haemodynamic onset transient lands inside the scored
window. That belongs to the BOLD chain and is carried by **§42**.

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

### 13. Cache state files store the cortical rate path as a bare string — resolved 2026-08-13, moved to Resolved

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

*Opened 2026-08-04 15:42 · 1 update, 2026-08-27 08:14*

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

**Update 2026-08-27 08:14:**

The community-review triage accepted round-2 F5 of
`community_review/round2/synthesis.md` (four seats; evidence class
experimentally grounded — Kumaravelu 2018: STN-DBS-evoked cortical
potentials in awake rats decompose into R1 at 1.35 ± 0.07 ms, direct
antidromic activation of L5 axons, anaesthesia-resistant; Chen et al. 2020:
human DBS-evoked antidromic responses over prefrontal cortex at 6 ms) into
this entry. The review separates three consequences, and the triage decided
a remedy for each:

- **The rate half of the cortical DBS effect is not missing.** The DBS-on
  drive is deconvolved from the subject's own recording *under
  stimulation*, so the net cortical rate change DBS produced — antidromic
  cortical effects included, for every cortical stream, striatal ones too —
  is already in the input. What the fitted DBS parameters therefore
  estimate is the **intra-BG effect given the observed cortex**: a
  defensible quantity but a narrower one than "what DBS did", and the
  write-up must say so verbatim wherever fitted DBS parameters are
  interpreted (this bound shrinks under the mechanism below but does not
  vanish).
- **The orthodromic terminal volley into STN is missing, and is now to be
  built** — *accepted, in a form modified from the review's proposal*. The
  review proposed a pulse-locked spike-count component in the STN cortical
  stream with its amplitude as a fourth fitted DBS parameter. Decision: the
  amplitude is instead **tied to the existing `axon_spikes_per_pulse`**, no
  new free parameter. Rationale: `DBS.md`'s `_set_orthodromic` already sets
  `proj.pre.prob_axon_spike` from `axon_spikes_per_pulse` for *every*
  reachable afferent of the stimulated population — the pallidal afferent
  `gpe_proto→stn` already emits per-pulse axon spikes at exactly that
  probability, and only the cortical afferent falls out because a
  `TimedArray` has no soma. Scaling a pulse-locked component in the
  cortical spike-count stream by the same parameter is therefore the
  faithful extension of the model's existing shared-activation-probability
  assumption, not a new assumption; a fourth parameter would add a fresh
  degeneracy instead. Side effect on identifiability: §2 lists
  `axon_spikes_per_pulse` as near-degenerate with the `stn__gpe`/`stn__snr`
  scalings — coupling it to a distinct cortical-input effect can only
  sharpen it. Caveat to record with the implementation: hyperdirect axons
  are large myelinated pyramidal-tract collaterals and plausibly *more*
  excitable than the fibre classes the shared probability was written for —
  the shared probability is the model's standing assumption, now stretched
  one class further. Implementation is cheap by construction: the
  count→current conversion passes through Python every 110 ms chunk and
  pulse times are deterministic, so no cache, neuron model or compiled
  network is touched (the Opened block's "spiking soma" remedy is heavier
  than needed and is superseded).
- **The remaining inexpressible part is bounded, not represented.** The
  antidromic soma invasion also synchronises cortex (R2/R3, and via
  pyramidal-tract collaterals potentially striatal input) at unchanged
  deconvolved rate — a fine-timescale synchrony BOLD deconvolution cannot
  see, while `experimental_data/input_streams/README.md` §3 shows input
  correlation is the dominant determinant of simulated BOLD amplitude.
  Generic stream correlation is not pulse-locked, so raising it in DBS-on
  would be a confound, not a representation. Instead: **one sensitivity
  run** — DBS-on once with a non-zero cortical shared modulation
  (`make_global_p_trace` machinery exists and is merely set to zero) — and
  the simulated BOLD amplitude change reported as the bound on whatever
  synchrony effect the model cannot express.

### 16. The DBS constants are unvalidated single-subject values

*Opened 2026-08-04 15:42 · 1 update, 2026-08-27 14:26*

**Opened 2026-08-04 15:42:**

`population_proportion = (35+23)/(70+75) = 0.4` (VTA, Berlin subject 1),
125 Hz, and `snr__thal:putamen` as the single passing fibre (after Miocinovic et
al. 2006) are hard-coded in `get_loss.py` and documented only inline. The pulse
width is 100 µs against 60 µs in the data, raised because 60 µs is below dt.

None of these is fitted, so each is an assumption the inference inherits.
Sensitivity to at least `population_proportion` and `dbs_pulse_width_us` should be
checked once a DBS-on fit exists — before any of it is written up.

**Update 2026-08-27 14:26:**

The community-review triage accepted round-2 F15 of
`community_review/round2/synthesis.md` (one seat; evidence class
experimentally grounded, scope) into this entry, because it bears directly
on what this entry's planned checks can establish.

**The frequency constant cannot be checked by a sweep, and the reason is
structural.** Every DBS term is gated by `pulse(t)` —
`ite(modulo(time_ms*1000, 1000000./dbs_pulse_frequency_Hz) < dbs_pulse_width_us, 1., 0.)`,
one timestep every 8 ms at 125 Hz — with no adaptation, depression or
pulse-to-pulse interaction anywhere in the equation set. The expected
perturbation over any interval is therefore per-pulse effect × pulse count:
**proportional to frequency, with no threshold and no saturation.** A
frequency sweep would provably return a straight line, so it tests nothing.

That contradicts the best-established quantitative fact about STN DBS. The
measured profile — no effect below ~40 Hz, decline between 50 and 130 Hz,
saturation above 150 Hz — is reproduced by Kumaravelu 2016 Fig. 11 (and Su
2019 Fig. 5) *without* being fitted to it, against the parallel frequency
dependence of symptom suppression.

**Why this is scope rather than a defect to fix.** The fit runs at 125 Hz,
squarely therapeutic, so the fitted result is untouched. And the model
could not measure a dose–response even if the mechanism supported one: it
has no pathological oscillation to suppress and reads out at 2.31 s. A
mechanistic frequency dependence would need a memory term this equation set
does not have — a much larger change than this project needs.

**What is owed.** The scope limitation is now recorded in `DBS.md` (known
limitation 6). It must also appear in the write-up: the DBS representation
is a **per-pulse perturbation calibrated at 125 Hz, linear in frequency by
construction, and not to be extrapolated to other stimulation settings**.
That half stays open here, since there is no write-up yet, and it joins the
sensitivity checks above as things this entry owes before publication.

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

### 19. `parameters.py` labels the *planned* cache size as the current one — resolved 2026-08-11, moved to Resolved

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

*Opened 2026-08-07 10:55 · 1 update, 2026-08-12 10:27*

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

**Update 2026-08-12 10:27:**

**Why the CI streams use a different construction than the striatal cortical
streams, and what a future nonzero `f` would and would not change.**
Shared+private with fraction `f`
(`spike_input_cortex.simulate_receiver_counts_homogeneous_to_memmap`) is
pairwise-identical to an axon pool of `M = N/f` — the pool construction subsumes
it — except at `f = 0`, which corresponds to `M = ∞` and is exactly what is in
force here. That, plus the absence of any cross-population requirement, is why
`CorticalInputs` uses the split and `Microcircuit` the explicit pool.

**Putting a nonzero value into `ci.shared_fraction_dict` is by itself no reason
to switch constructions**: shared+private handles any `f` with the same
exactness (Binomial marginal, correlation exactly `f`) and cheaper draws
(Binomial instead of hypergeometric). Switching to one pool per cortical region
would buy exactly two things, both currently unwanted or unmeasured:

1. **Derived cross-structure coupling.** Receivers of different populations
   (thal/GPe/STN) sampling one pool would correlate at `sqrt(N_i N_j)/M` —
   physically plausible via corticofugal collaterals (hyperdirect STN afferents
   are collaterals of corticofugal axons that also reach thalamus), but with no
   measurement behind it. The same move forces `f_i = N_i/M`, so the per-type
   fractions of this dict would stop being independently settable.
2. **A natural parameterization for anatomy-shaped data.** If an overlap
   measurement ever arrives as an axon count `M` (as Kincaid's effectively did)
   rather than as a fraction, the pool takes it directly — though `f = N/M` can
   just as well be computed by hand and fed to the present construction, so this
   is convenience, not capability.

The present-tense description of the construction is in
`input_streams/README.md` §4.6 and the `spike_input_cortex` module docstring;
this block holds the forward-looking part.

## From the session on 2026-08-07 (verifying model_v07.md against the code)

### 27. The realised `f(d)` of the geometric pools is not checked at build time — resolved 2026-08-13, moved to Resolved

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

### 29. `CorticalInputs`' cache validation is strictly weaker than `Microcircuit`'s — resolved 2026-08-13, moved to Resolved

## From the session on 2026-08-10 (model_v07.md §6 follow-up)

### 30. `phi_1 = phi_2 = 0` — the striatal dopamine terms are switched off, unchecked against the source paper

*Opened 2026-08-10 11:25 · 1 update, 2026-08-27 09:37*

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

**Update 2026-08-27 09:37:**

The community-review triage extends this entry's scope twice, from round-2
findings F7 and F1 of `community_review/round2/synthesis.md`:

- **The φ decision cannot be made against an unknown medication state.**
  §39 (opened today from F7) records that nothing documents whether the
  Berlin subject was scanned on or off medication. "Dopamine-depleted"
  has no determinate value until that is known, so §39 comes first.
- **φ is striatal only, and the review shows that is a stated scope, not
  a complete dopamine account.** From Wichmann 2019 (macaque, human):
  GPi/STN rates depend on disease *stage* through **extrastriatal**
  dopamine — D2-like receptors on striatopallidal terminals in primate
  GPe, D1/D2 on STN afferents, D1-like in GPi/SNr — with "comparatively
  maintained pallidal and nigral dopamine levels in early parkinsonism"
  able to hold GPi/SNr rates near normal. The model has no extrastriatal
  dopamine term at all, so whatever that effect is in this subject is
  absorbed by the fitted `base_mean` parameters. This entry's remit stays
  striatal; the point is recorded here so the eventual verdict states the
  limitation rather than implying φ settles the model's dopamine state.
  It also bears on §35's non-striatal band derivation, which cannot cite
  a single "parkinsonian" rate without a stage qualifier.

## From the session on 2026-08-11 (dissolving PLAN.md into this file)

### 31. Run the five-generation mini-run on a workstation

*Opened 2026-08-11 10:04*

**Opened 2026-08-11 10:04:**

Formerly `PLAN.md` step 9 (§33). A five-generation CMA-ES mini-run of
`deap_cma_opt.py` on hinton or waikiki, with checkpointing exercised
(`--resume` after a kill), against the full-length caches. **This green is the
milestone**: it is the first time the whole pipeline — caches, gate, logging,
penalize-and-continue, checkpoint — runs together at scale.

What it must produce beyond a green:

- **The §10 gate numbers.** Log how many individuals per generation are gated
  (`bold_skipped` is in every loss file); the run is what calibrates
  `firing_rate_gate`, which is untenable as configured (0.5 against measured
  0.87 at the default vector). Run with `--gate-threshold 1.0` until decided.
- **The lambda choice** (§6): physical vs logical cores, compile-phase peak
  RSS, and the real per-evaluation time on both machines — every current
  budget is a laptop extrapolation.

Blocked by §1 (bounds — a mini-run on saturated or unmeasured bounds proves
nothing about the search) and by §6 + §3 (the code, the environment and the
caches have to be on the machines first).

### 32. Launch the fits: DBS-off, then DBS-on

*Opened 2026-08-11 10:04*

**Opened 2026-08-11 10:04:**

Formerly `PLAN.md` step 10 (§33). One straightforward DBS-off fit, then a
DBS-on fit — deliberately simple; the proper inference design is §2 and comes
after.

Decisions already taken (from `PLAN.md`, recorded here so they survive its
deletion):

- **Seeds: fixed at 42 during fitting** — common random numbers, so CMA-ES
  sees a deterministic objective. The winning parameter vector is then re-run
  across ~10 seeds to report stability.
- **The on fit is staged**: it seeds from the off fit's pickle
  (`load_best_off_fit`) and searches only the putamen cluster scalings plus
  the 3 DBS parameters (13 free for v07) against the fixed off-condition base.
  The caudate loop keeps its off weights and serves as the free control —
  §2's update of 2026-08-11 has the rationale.
- Budgeted at ~25-40 min per evaluation on the workstations (laptop
  extrapolation; §31 measures the real number).

Blocked by §31. Feeds §4 (compare fitted rates against the missing-GABA
surround assumption), §2 (inference design over the fitted result) and §16
(DBS-constant sensitivity) — see the Roadmap.

### 33. PLAN.md dissolved into this file — resolved 2026-08-11, moved to Resolved

## From the session on 2026-08-13 (community conventions review)

### 34. Survey the BG-modeling community's conventions via reviewer personas — resolved 2026-08-13, moved to Resolved

## From the session on 2026-08-26 (community-review triage)

### 35. Derive and document the six non-striatal firing-rate bands

*Opened 2026-08-26 11:45*

**Opened 2026-08-26 11:45:**

From the round-2 community review (accepted from
`community_review/round2/synthesis.md`, its F1 — the unanimous finding of the
round, raised by all seven seats; the synthesis is not referenceable, so the
substance is restated here). Evidence class: experimentally grounded.

`get_loss.get_firing_rate_loss` scores nine populations per loop against its
`plausible_ranges` table. The three striatal bands are derived, sourced and
caveated in `experimental_data/activity_striatum/README.md`; the six
non-striatal ones are not: `stn` (28, 80) and `snr` (21, 93) carry only the
code comment "from [Li et al., 2015]" — a citation with no journal, DOI,
species, preparation or disease state, which resolves to nothing anywhere in
the repository — and `gpe_proto` (75, 85), `gpe_arky` (15, 20), `gpe_cp`
(75, 85) and `thal` (15, 30) carry no citation at all. These bands are half
the total loss and the whole of the gate, and §10 records that both DBS
conditions score ~0.87 against them — the gate fires on everything, and
nobody can say whether the bands or the model are at fault.

Three compounding problems, with the review's measured anchors:

- **Provenance.** An unresolvable citation on half the loss cannot be
  audited by anyone, including this project's future self — while the
  striatal side of the same function shows exactly how it should be done.
- **Disease state.** No band states whether it is healthy or parkinsonian.
  Measured directions: in MPTP macaques GPe *falls* from 65.1 ± 25.6 to
  41.1 ± 22.3 Hz (Tachibana et al. 2011, as recomputed in Shouno et al. 2017
  Table 1(A)); McGregor & Nelson 2019 Fig. 4 tabulates prototypic GPe
  *down*, STN and GPi/SNr *up*, thalamus "?" in parkinsonism. The model is
  required to hold `gpe_proto` and `gpe_cp` at 75–85 Hz — at or above the
  *normal*-macaque band and roughly double the parkinsonian-macaque mean.
  The arky band (15–20 Hz) excludes the rodent in-vivo average (~10 Hz,
  Giossi et al. 2024). `gpe_cp` — an Npas1-class population by the model's
  own BOLD-weight citation — gets the prototypic band although the one
  condition-relevant datum for Npas1⁺ cells is *hypoactivity* under
  depletion (Pamukcu et al. 2020, mouse).
- **Width.** The two GPe bands are the narrowest in the table (±6 % of
  centre, against ±49 % for dSPN and SDs of ±25 Hz in any monkey GPe
  sample), with no width rule stated — the least-sourced bands bind the
  gate hardest.

Two additional tensions recorded by the review: four of the six bands
exclude or sit at the edge of the operating point at which the inherited
weight table was validated (GPe-Proto ≈ 40 Hz, GPe-Arky ≈ 10–12 Hz,
STN ≈ 15 Hz, thal ≈ 10 Hz; Goenner et al. 2021 Fig. 7), so the gate pushes
the network away from the regime the frozen weight ratios were tuned in.
And GPi/STN rates depend on disease *stage* through extrastriatal dopamine
(Wichmann 2019, macaque/human); the model has no extrastriatal dopamine
term, so the fitted `base_mean` parameters absorb whatever that effect is —
this bears on §30, which frames the dopamine question as striatal only.

**The task.** Write the derivation document for the six non-striatal bands
in the style of `experimental_data/activity_striatum/README.md` — per band:
resolvable source, species, preparation, medication state, and a stated
width rule; resolve or replace "[Li et al., 2015]"; check each band's
parkinsonian *direction* against Tachibana 2011 (via Shouno 2017 Table 1)
and McGregor & Nelson 2019 Fig. 4; mark the thalamic band as unconstrained
in parkinsonism (both physiology reviews mark it so); state explicitly
where a band is an assumption. Update `plausible_ranges` to whatever the
document derives, and point the code comment at the document.

**Blocking:** precedes §10's threshold calibration (calibrating a threshold
against undocumented bands sets one unknown from another) and any fit whose
gate is enabled. Whether the bands need a DBS-on variant stays with §10.

### 36. Report the fitted models' dynamical regime from the probe spikes

*Opened 2026-08-26 11:58*

**Opened 2026-08-26 11:58:**

From the round-2 community review (accepted from
`community_review/round2/synthesis.md`, its F3 — five seats; evidence class
experimentally grounded; the synthesis is not referenceable, so the
substance is restated here).

The only spiking statistics the pipeline ever computes are mean rates over
the 9.9 s probe (`get_loss.Spikes10s` → `get_firing_rate_10s`, last 80 % of
the probe). Whether a fitted model is asynchronous or oscillatory — whether
its STN–GPe loop sits below or above the oscillation boundary — is never
measured, and the BOLD loss cannot see it (the drive is per-TR, the balloon
model low-passes the rest). But pattern, not rate, is the field's
parkinsonian marker: in Shouno et al. 2017 (Fig. 3, Table 1, recomputed
from Tachibana 2011 monkey spike data) the parameter regions whose *mean
rates* match the normal and parkinsonian states overlap, while the states
separate cleanly on oscillation and burst measures (STN oscillatory cells
5.5 % normal vs 36.3 % parkinsonian); both physiology reviews the round
consulted (Wichmann 2019, McGregor & Nelson 2019) make the rate-to-pattern
shift their organising theme.

Nobody demands *fitting* these statistics — the subject's electrophysiology
does not exist. The demand is a **reported diagnostic**, and it is cheap:
the probe already records every spike (`get_loss` builds
`monitor_dictionary = {pop_name: ["spike"] ...}` over all non-TimedInput
populations), so the statistics cost one analysis function and no new
simulation. It catches three failure modes the inference would otherwise
inherit silently:

1. An off-fit that is not recognisably parkinsonian in the one currency the
   field trusts (a result to *report* either way, not to hide).
2. An off-fit whose oscillation frequency is an artefact of the rat delay
   set (the review's F26 carries the delay question itself).
3. An on-vs-off delta that works by flipping the network across the
   oscillation boundary while being reported as a drive-weight change —
   and, relatedly, an uninterpretable `dbs_depolarization`: the review's
   numerical check found it acts as a suppress-and-entrain knob that
   phase-locks the stimulated STN at high amplitude, which only a spectral
   diagnostic surfaces.

**The task.** Compute per-population burst fractions, 8–35 Hz spectral
power and a pairwise spike-count synchrony summary from the existing probe
recordings; write them into `data_BOLD_optimization/loss_<appendix>.json`
beside the firing rates; include them in the phase-1 validation-run
artefacts; and for the accepted off- and on-fits, report them against the
stated references (Shouno 2017 Table 1; the directions of McGregor & Nelson
2019 Fig. 4). Caveat to record when implementing: the probe is 9.9 s, so
low-beta resolution and burst statistics are limited — state the window
alongside the numbers rather than extending the probe silently (a longer
probe would change cache divisibility constraints).

**Blocking:** none for its implementation (one analysis function); the
reporting side joins §2's interpretation checklist and the phase-1
validation run.

### 37. Write the parameter-provenance companion for parameters.csv

*Opened 2026-08-26 12:57*

**Opened 2026-08-26 12:57:**

From the round-2 community review (accepted from
`community_review/round2/synthesis.md`, its F4 — four seats; evidence class
methodological: auditability, and internal consistency with the project's
own standard; the synthesis is not referenceable, so the substance is
restated here).

`parameters.csv` is a value table with no reference column, and the only
lineage record is the `BGM_v07` docstring. The known provenance gaps:

- `get_loss.py` calls the projection weights "the literature values"
  without saying which literature (the answer — Goenner et al. 2021's
  table, rescaled — was recovered by the review's archaeology, carried as
  its F9).
- The GPe neuron refit is recorded in one CSV header cell ("refitted, data
  from Bogacz et al. 2016") with no fit protocol or quality comparison, and
  "Bogacz et al. 2016" is not resolved to a citable reference.
- The synaptic constants (`tau_ampa` 2 ms, `tau_gaba` 10 ms, `E_gaba`
  −70 mV) differ from *both* predecessors' published sets (10/20/−90 in
  Goenner 2021, 10/10/−90 in Maith 2021) with the departure recorded
  nowhere.
- The uniform in-degree (`connect_fixed_number_pre` with `number = 10` on
  every BG-internal projection) has no stated basis — presumably inherited
  lineage convention (the v01–v06 columns carry the same 10), but nothing
  says so. The *substantive* adequacy of 10 is a separate question (the
  review's F25); this entry only owes the origin sentence.
- The delays' provenance was recovered only by the previous review round.

The community anchor is a reporting convention, admitted as such by all
four seats: per-row source columns with tuned values named as tuned
(Corbit 2016 Table 2; Lindahl 2016 Tables 7–9 with "n.d., estimated"
written out; Girard 2021 Tables 1–2; Kumaravelu 2016 Table 1). Its value
was demonstrated by the round itself: three of its sharpest findings (the
weight-table lineage, the kernel state question, the delay set) required
archaeology that a reference column would have made one-line checks. And
the repository's own `experimental_data/` READMEs are the counterexample to
its own CSV — the project demonstrably knows the format. The project's
product is a statement *about parameters*; a fitted scaling on a base
weight of unstated origin transmits no interpretable meaning.

**The task.** A provenance companion document for the `BGM_v07_p01` column
in Corbit-Table-2 form: per row (or per CSV section) the source —
"Goenner et al. 2021 Table 4/5, ×C rescaled" where that is the answer,
"refit on Abdi/Bogacz step-current data" with the fit record and an f–I
comparison (Goenner 2021 Fig. 2 is the template), "no source" where that is
the truth; one sentence on the origin of `number = 10`; the
synaptic-constant departure from both predecessors recorded next to the
values; "Bogacz et al. 2016" resolved to citable form. "[Li et al., 2015]"
is resolved by §35 and referenced from here.

**Blocking:** nothing blocks it; pure documentation and archaeology. Doing
it early makes the verdict-pass entries it feeds (§35, and the eventual
weight/delay verdicts) one-line checks instead of digs.

### 38. The pooled GPi/GPe/STN ROIs weight the two loops by population size

*Opened 2026-08-27 09:33*

**Opened 2026-08-27 09:33:**

From the round-2 community review (accepted from
`community_review/round2/synthesis.md`, its F6 — three seats; evidence
class methodological; seat 4 rates its own half "difference, not
deficiency"; the synthesis is not referenceable, so the substance is
restated here). Accepted on a **narrower cut than the review proposed** —
see the "what this is not" paragraph.

`get_loss`'s `bold_region_compartments` pools both loops' copies of each
nucleus into the `GPi`, `GPe` and `STN` monitors. `GPi` and `STN` pass
`scale_factor=None`, so `BoldMonitor` falls back to weighting by
population size — both copies are 100 neurons, hence **50/50**. `GPe` does
pass `scale_factors`, but they are the cell-type abundances
(`gpe_proportions`), identical in both loops, so its loop split is 50/50
too. That ratio is set by `stn.size = 100`, not chosen.

The subject's own files (`experimental_data/berlin_data/vta/`, with the
arithmetic quoted by seat 4) give the STN territory volumes in voxels,
both hemispheres: motor 70+75 = 145, associative 68+68 = 136, limbic
54+55 = 109, total 390. The motor territory — what the putamen loop stands
for — is therefore 145/390 ≈ **37 %** of the real STN, against 50 % in the
pooled monitor: the model over-weights the DBS-carrying compartment by
≈ 0.50/0.37 ≈ 1.34.

**What this is not.** Two readings were considered and rejected in the
triage:

- *Not a problem with the 0.4 VTA proportion.* `population_proportion =
  (35+23)/(70+75) = 0.4` is this subject's measured VTA overlap **of the
  motor territory**, exactly the quantity the putamen loop needs; it lives
  inside the loop and is untouched by the pooling weights. Seat 4's
  observation that the DBS-affected share of the pooled STN BOLD is
  0.4 × 0.5 = 20 %, close to the whole-STN overlap of 76/390 = 19.5 %, is
  labelled by seat 4 itself as a coincidence rather than a derivation.
- *Not "half the ROI is DBS-blind, so the signal is diluted".* The measured
  BOLD pools territories too, and in the real brain DBS also acts mostly in
  the motor territory, so dilution as such is symmetric and cancels. What
  does not cancel is the mismatch between the two weightings: to reproduce
  a measured pooled on-off change, the model's putamen STN need only
  produce ≈ 0.74 of what the real motor STN produced, and the fitted DBS
  parameters absorb that factor. The defect is an unjustified ≈ 1.3 scaling
  sitting in exactly the three parameters the project's claim is about —
  not a structural blindness.

**The task**, in three parts:

1. **Set the loop weights from the subject's territory volumes** instead of
   from population size. For `STN` the numbers are in the repository
   already; for `GPi` and `GPe` this entry has to establish what is
   available (the same VTA files, another subject-level source, or a stated
   assumption) and say which it used. Mechanically this is the existing
   `scale_factor` path — for `GPe` the loop weight multiplies into the
   cell-type factors rather than replacing them. No cache, network or
   neuron model is touched.
2. **Report, for fitted models,** each loop's share of every pooled ROI's
   variance and of its on-off change, so the compensation that actually
   happened is visible rather than inferred.
3. **Document it in `model_v07.md` §3.6**, beside the GPe abundance
   discussion — including seat 4's second point from the same files: the
   subject's *associative* VTA overlap is 12/136 = **8.8 %**, set to
   exactly zero by excluding the caudate loop. That exclusion is
   deliberate (it is what makes the caudate loop §2's free control), but
   the caveat should quote the 8.8 % rather than assert that DBS reaches
   only the motor territory.

**Blocking:** nothing blocks it, and it blocks nothing structurally — but
part 1 changes what the fitted DBS parameter values mean, so it belongs
before §32's on-fit is interpreted, alongside §2's checklist.

### 39. Record the Berlin subject's provenance: medication state, protocol, acquisition, hemispheres

*Opened 2026-08-27 09:37*

**Opened 2026-08-27 09:37:**

From the round-2 community review (accepted from
`community_review/round2/synthesis.md`, its F7 — three seats; evidence
class methodological: the scientific question is already planned in §30 and
the Roadmap, the gap is recorded provenance; the synthesis is not
referenceable, so the substance is restated here).

The model asserts a **medication-off** subject throughout: the striatal
surround and the `get_firing_rate_loss` bands are the medication-off state
of Liang et al. 2008 (§20), and `phi_1 = phi_2 = 0` switches the striatal
dopamine terms off entirely (§30). But no document records what the Berlin
subject's medication state during scanning actually *was*. Nor is the
hemisphere convention of the ROI series written down (the VTA arithmetic
in `get_loss` pools two electrodes), nor the acquisition parameters
(field strength, TE), nor the on/off session protocol.
`experimental_data/berlin_data/` is the only data directory without a
README, while `activity_striatum/`, `cortical_proportions/`,
`input_streams/` and `Connectivity_intrinsic_striatum/` all have one — the
project's own standard is the counterexample.

**Why this is not bookkeeping.** Seat 5's predecessor drew from the same
clinical population and recorded the state precisely — patients scanned
DBS-OFF *with* their usual medication — then *needed* that fact to
interpret a fitted result (Maith et al. 2021 §4.1 explains an absent
STN/GPi rate increase by the medicated state). If this subject was scanned
on medication, the Liang anchors, `phi = 0` and every input cache built on
them target the wrong physiological state, in **both** conditions of the
inference. Seat 6, which carries the panel's only graded-dopamine
machinery, reports that medication state *inverts* effects in its own
models — this is not a small correction. Seat 1 frames the requirement as
internal state consistency among the fixed ingredients: rates (chronic
med-off), the striatal connectivity kernel (currently state-less; the
review's F11) and φ (§30) must all describe the same physiological state.

**The task.** Write `experimental_data/berlin_data/README.md` in the style
the other data directories already follow: the subject's medication state
during scanning, the on/off session protocol, field strength and TE, and
the hemisphere convention of the ROI series — each marked as measured,
reported by the source, or assumed. Where the answer is not in the
materials at hand, it is a question to the Berlin collaborators, not an
assumption to be made here. If the answer turns out to be "on medication",
that is a finding, and §20's rate anchors, §30's φ decision and every v07
cache follow from it.

**Blocking:** must be settled **before phase 2 builds any full-length
cache** — it is one of the five cache-invalidating findings the Roadmap
orders inside phase 1. It also feeds §30: the φ decision cannot be made
against an unknown medication state.

### 40. Audit the GPe three-way split: `gpe_cp`'s identity, the cell types and their connectivity

*Opened 2026-08-27 10:24*

**Opened 2026-08-27 10:24:**

From the round-2 community review (accepted from
`community_review/round2/synthesis.md`, its F8 — three seats), **widened
by decision beyond what the review licensed**: F8 is classed "difference,
not deficiency" and proposes documentation only, because every targeting
fact available to the panel is marker-defined mouse work that both Courtney
2023 and Wichmann 2019 say is untested in primate. The decision here is
that the split is *our own* construct — introduced in Goenner et al. 2021 by
this lab — so it is ours to re-examine rather than merely to describe: this
entry additionally audits whether the cell-type division and its
connectivity still hold against current empirical work, **and whether
Goenner 2021 read its own cited literature correctly**. An audit may
legitimately conclude "no change licensed"; what it may not do is leave the
question unasked.

**What the model has.** Three GPe populations per loop, 100 neurons each:
`gpe_proto`, `gpe_arky`, `gpe_cp`. `gpe_cp` shares `gpe_proto`'s neuron
parameters exactly (`parameters.csv`, "just like gpe proto" — verified: the
Izhikevich constants are identical) and its 75–85 Hz band, projects to all
three striatal populations (0.5 / 0.5 / 0.8) and within GPe, and receives
fitted cortical drive via `CorticalInputs`. It weights the GPe BOLD at 0.10,
matched to Courtney 2023's NPAS1⁺NKX2.1⁺ abundance (`model_v07.md` §3.6).

**The three tensions the review documents** (as reported in the round-2 seat
reviews; the primary papers are to be read, not recalled, when this entry is
worked):

1. *The namesake efferent cannot exist here* (seat 5, which owns the
   provenance). Goenner 2021 introduced GPe-Cp **for** the
   cortico-pallido-cortical loop — the paper's central novel claim, with
   GPe→cortex tabulated from Abecassis 2020, Chen 2015, Saunders 2015 and
   cortex→GPe from Karube 2019, Naito & Kita 1994, Milardi 2015, Smith &
   Wichmann 2015. In BGM_22 there is no simulated cortex, so that efferent
   is structurally absent and `gpe_cp` is functionally a third GPe
   population distinguished only by its afferent mix and fitted drive
   weight. No document says so.
2. *The identity does not close* (seat 7). Courtney 2023's NPAS1⁺NKX2.1⁺
   class — whose ≈12 % abundance the BOLD weight was matched to —
   "project[s] exclusively to the midbrain, the cortex and the reticular
   nucleus of the thalamus", not to the striatum, while the model's
   `gpe_cp` is a striatum-projecting population. Box 1 leaves room for a
   second striatum-projecting class, so the model is not contradicted — but
   nothing identifies its population, and `gpe_cp` is expanded nowhere in
   the repository. Band and BOLD weight currently borrow from two different
   identities.
3. *The frozen wiring ratios point against the subtype-resolved
   measurements* (seats 2 and 7). Aristieta et al. 2021 (mouse, via Giossi
   2024) measured iSPN→arkypallidal 85 % *weaker* than iSPN→prototypic and
   STN→arkypallidal 74 % weaker than STN→prototypic, while the model has
   `str_d2 → gpe_arky` at *twice* `str_d2 → gpe_proto` and `stn →` equal
   across all three GPe types. Courtney 2023 has iSPNs strongly targeting
   the STN-projecting prototypic class and arkypallidal neurons making few
   local collaterals, while the model gives `gpe_arky →` a third of the
   prototypic collateral weight — all inside frozen optimizer clusters
   (`gpe_laterals`, `str_d2__bg`), so no fitted scaling can repair a ratio.
   In the model's favour, also from Courtney: the direct-pathway collateral
   into GPe exists here at all (`str_d1 → gpe_cp`), is routed to the
   non-prototypic population, and is kept an order of magnitude weaker than
   the iSPN weights — the direction of the measured bouton asymmetry.

**The task**, in two parts:

1. **The audit** (the widening). Re-read Goenner 2021 and the primary
   sources it tabulates for the GPe division, and check: does the
   three-way split as parameterised here still match current empirical
   work, and did Goenner 2021 interpret those sources correctly? Then the
   same question for the intra-GPe and striatopallidal connectivity against
   Courtney 2023, Aristieta 2021 (via Giossi 2024) and Wichmann 2019.
   Sources must be read, not recalled. Outcome is one of: a licensed
   change, or a reasoned "no change — the evidence is marker-defined mouse
   work with no established primate translation", recorded either way.
2. **The documentation** (the review's own proposal, owed regardless of the
   audit's outcome). State in `model_v07.md` what `gpe_cp` denotes, which
   experimental population it is meant to be, that its namesake efferent is
   structurally absent, and the resulting claim boundary — a fitted change
   in `gpe_cp` parameters under DBS must **not** be narrated as a
   pallido-cortical pathway effect. Reconcile or flag the band/abundance
   identity tension when §35's band document is written.

**Relations.** The band half of the identity tension is §35's. The
"weights the BOLD by realistic abundances while simulating 100 neurons
each" question is the review's F22, not yet triaged. Tension 3 above
overlaps **§41** (opened 2026-08-27 from the review's F9, the full
28-weight provenance and cluster-ratio finding). The scope was settled when
§41 opened: this entry owns the GPe-specific empirical audit and feeds its
findings into §41's per-cluster decision, while §41 owns the mechanism-level
question — whether frozen within-cluster ratios are defensible, the
cluster-splitting decision, the non-GPe ratios and the post-fit sensitivity
check. Provenance recording as such is §37.

**Blocking:** the documentation half blocks nothing. The audit half could
license a connectivity change, which would move every GPe spike train and
invalidate nothing cache-side (weights are not cache parameters) — but per
the repository convention a baseline must be captured first, and the
ANNarchy global-RNG caveat in `CLAUDE.md` applies if any random variable is
added or removed.

### 41. The weight table is task-tuned and its within-cluster ratios are frozen

*Opened 2026-08-27 10:50*

**Opened 2026-08-27 10:50:**

From the round-2 community review (accepted from
`community_review/round2/synthesis.md`, its F9 — two seats; evidence class
experimentally grounded, plus methodological for the provenance half; the
synthesis is not referenceable, so the substance is restated here).

**What was established.** Two seats independently diffed all 28 v07
projection weights against Goenner et al. 2021 Tables 4–5: every value
matches — verbatim for non-striatal targets, ×C (50 for SPNs, 80 for FSIs)
for striatal ones. `get_loss.py` calls them "the literature values" and
nothing in the repository records the lineage (the recording itself is
§37's). The source characterises them differently: "the weight strengths…
are rather abstract and were determined mainly by functional constraints"
(Goenner 2021 §4.4) — i.e. tuned until a rat stop-signal network stopped
correctly. They were tuned under synaptic constants 10/20/−90 and run here
under 2/10/−70 (seat 5).

**Why this is more than provenance.** Where subtype-resolved measurements
exist, they contradict the frozen ratios (as reported in the round-2 seat
reviews; primary papers to be read, not recalled, when this entry is
worked):

- Aristieta et al. 2021 (mouse, via Giossi 2024): iSPN→arkypallidal 85 %
  *weaker* than iSPN→prototypic, STN→arkypallidal 74 % weaker than
  STN→prototypic. The table has `str_d2__gpe_arky` (0.08) at **twice**
  `str_d2__gpe_proto` (0.04), and `stn__gpe_proto/arky/cp` all **equal**
  (0.001).
- Corbit et al. 2016 (mouse slice, ChR2 — seat 2's own lineage): GPe→FSI
  IPSCs 566 ± 560 pA in every FSI sampled against 28–108 pA in SPNs,
  modelled there as a 12–40× conductance ratio. BGM_22's aggregate ratio at
  equal in-degree is ~1.4–2.3×, with the prototypic-vs-arkypallidal order
  onto FSIs reversed relative to Giossi 2024's account.

**And the mechanism forecloses the repair.** The affected ratios sit inside
`PROJ_CLUSTERS_COMMON` entries — `str_d2__bg`, `stn__gpe`, `gpe_striatum`,
`gpe_laterals` — where the optimizer fits one common scale per cluster.
Cluster scaling preserves relative balance to condition the search, which
is sound exactly when the encoded balance is trustworthy; here it is
task-tuning for a different task under different kinetics, and **no fitted
scaling can move a frozen ratio**.

**The task.** Decide, per affected cluster, between three outcomes —
recording the reasoning either way:

1. **Re-derive** the within-cluster ratios from the reported measurements,
   with species caveats documented (all of it is marker-defined mouse work
   that Courtney 2023 and Wichmann 2019 both call untested in primate).
2. **Split** the cluster so the fit can move the ratio. Note the cost the
   review does not price: splitting raises the free-parameter count
   (`n_opt_params`, currently 19 for v07) and so degrades exactly the
   conditioning the clusters exist to provide — this is a trade-off to
   argue, not a free improvement.
3. **Reject with reasons**, and then, after the first fit, check the
   conclusions' sensitivity to the unanchored ratios *before* interpreting
   any cluster scaling.

**Scope, against §40.** §40 owns the GPe-specific empirical audit — the
cell-type identity, the three-way division, the intra-GPe and
striatopallidal connectivity, and whether Goenner 2021 read its own cited
sources correctly. This entry owns the mechanism-level question for the
whole table: the provenance status of the weights as task-tuned, whether
frozen within-cluster ratios are defensible at all, the cluster-splitting
decision, the non-GPe ratios (notably GPe→FSI vs GPe→SPN), and the post-fit
sensitivity check. §40's findings feed this entry's per-cluster decision;
neither entry re-does the other's work.

**Blocking.** Weights are not cache parameters, so nothing here invalidates
a cache. Outcomes 1 and 2 change the fitted model, so they belong before
§32; outcome 3's sensitivity check belongs with §2's interpretation
checklist. A baseline must be captured before any weight changes, per the
repository convention.

### 42. Drive the BOLD from synaptic conductances, and document the neural→BOLD chain

*Opened 2026-08-27 11:25*

**Opened 2026-08-27 11:25:**

From the round-2 community review (accepted from
`community_review/round2/synthesis.md`, its F10 — two seats; evidence
class methodological), **restructured by decision**: the review presents
the chain as a stack of untested choices and proposes comparing 2–3
mappings. Working through it settled most of them outright, so this entry
records one decided change, three deliberate choices that were mistaken for
omissions, and one genuinely open sub-question. Note the standing
limitation the synthesis states: no seat on the panel covers the BOLD
pipeline, so this is the nearest thing to a review of it.

**Decided: the hemodynamic model stays ours.** The monitor in force is
ANNarchy's `BoldMonitor` default `balloon_RN` — revised Stephan et al. 2007
coefficients, non-linear BOLD equation — verified against the installed
extension, and never overridden in `get_loss`. This is deliberate: the lab
published this BOLD monitor separately, and this project uses that method,
not the earlier `balloon_maith2021` convention (which the extension also
ships). What is owed is only that no project document says so. The exact
citation of the monitor paper must be taken from the paper, not from
memory. While documenting, check `balloon_RN`'s acquisition-dependent
constants — `TE = 40 ms`, `v_0 = 40.3`, `epsilon = 1.43` — against the
Berlin acquisition, which §39 is separately recording.

**Decided: the drive becomes conductance-based.** The intended neurovascular
coupling drives the haemodynamics with *synaptic input*, and the current
mapping (`I_CBF` → `I`, or `I_v` in the striatum) does that only
imperfectly: `I` is the **signed net** current, so inhibitory input enters
negatively and cancels against excitation, where the intended quantity is
"how much synaptic input arrived" and should be non-negative. The
information lives in the conductances — the factors multiplied onto the
driving potentials. With `stabilize=True` (all BG populations) the current
is

    I = I_app - g_ampa*(-50)/(1 + g_ampa*dt) - g_gaba*(v - E_gaba)/(1 + g_gaba*dt)

and the striatal (Humphries) models have the same shape with `g/(1 + g*dt/C)`
and an additional `g_nmda`.

*The open sub-question*: raw `g_ampa`/`g_gaba` (+ `g_nmda`), or the
stabilized factors `g/(1 + g*dt)` resp. `g/(1 + g*dt/C)`. Recommendation
recorded here, decision still to make: **the raw conductances**, because
`1/(1 + g*dt)` is an integration artefact that depends explicitly on `dt` —
a neurovascular drive that changes when the timestep changes is hard to
defend — while the conductance itself is the quantity transmitter binding
and its energetic cost track. Two further choices come with it: whether
`g_nmda` enters the striatal drive, and how excitatory and inhibitory
conductances are weighted against each other — `tau_ampa = 2` against
`tau_gaba = 10` means they accumulate on different timescales, so a plain
sum weights them implicitly rather than by decision.

**Deliberate, and to be documented rather than changed** (the review reads
these as gaps; they are choices):

- **The GPe monitors read raw `I`, not `f(I, nonlin)`.** The nonlinearity
  belongs to the *neuron model* — everything from input current to spikes —
  while the neurovascular drive is about the *synaptic input* that produces
  that current. The two are separate things that ANNarchy happens to define
  in one equation block.
- **`I_base` and the somatic DBS term are invisible to the monitors.** Same
  rationale: neither is synaptic input.
- Under the conductance-based drive above, all three exclusions follow **by
  construction** instead of by what `I` happens to contain — which is an
  argument for the change beyond the sign question.

**Still open, and separate from the mapping: the baseline and the onset
transient.** `normalize_input=2000` means the monitor accumulates the mapped
variable for 2000 ms after `start()`, feeds **zero** into the balloon model
during that window, then normalises everything afterwards as
`(x - baseline_mean)/(|baseline_mean| + 1e-7)` (verified in the extension's
`AccProjection` template). In this project the monitors start at
`t.rampup = 2310 ms`, so the baseline window runs 2310–4310 ms. Two
consequences:

- The window aligns with nothing: 2000 ms against a 2310 ms TR and a
  2310 ms ramp-up, ending 1730 ms inside the second recorded TR. That is
  §8's existing cleanup item ("check the two are intended to differ") and
  stays there.
- The balloon gets zero drive for ~0.87 TR and then a step onto relative
  deviations, so its haemodynamic onset transient lands **inside the scored
  window**: `compute_bold_correlation_loss` trims `ceil(rampup/TR) = 1` TR
  from the *experimental* series only and scores the simulated series from
  its first sample. Decide whether to trim the affected leading TRs from
  both series, extend the ramp-up so the baseline window closes before
  scoring starts, or accept and document it.

**The task.**

1. Decide the open sub-question, implement the conductance-based `I_CBF`
   mapping, and — per the repository convention — capture a baseline first:
   evaluate one parameter set under the current signed-`I` mapping and the
   new one, and record the per-region BOLD correlation shifts. The
   striatal regions are where the two differ most: a v07 striatal neuron
   receives ~61,000–69,000 GABAergic spikes/s from the missing-GABA
   compensation streams, which under the signed mapping pull the drive
   down and under a conductance mapping push it up.
2. Decide the transient question above.
3. Document the chain end to end in `model_v07.md` §3.6 — model,
   coefficients and their source, the mapping and why it is what it is, the
   three deliberate exclusions and their rationale, the baseline
   mechanism, and the acquisition constants. Maith et al. 2021 Table 2 is
   the house template for the parameter table.

**Blocking.** Nothing cache-side — the mapping is a monitor setting, not a
stream parameter. But it changes what every BOLD correlation means, so it
belongs before §32 and before §10's gate calibration is read against BOLD
losses.

### 43. The striatal connectivity kernels pool healthy and depleted data

*Opened 2026-08-27 11:45*

**Opened 2026-08-27 11:45:**

From the round-2 community review (accepted from
`community_review/round2/synthesis.md`, its F11 — one seat, but grounded in
this repository's own condition-labelled data and called the round's
clearest new discovery; evidence class experimentally grounded; the
synthesis is not referenceable, so the substance is restated here). Every
claim below was re-verified against the files during triage.

The seven `p(d)` kernels in
`striatal_microcircuit_requirements/connectivity_parameters/connectivity_fit_data/fitted_params.json`
— the entire intrinsic striatal connectivity, and through `E_outer` the
size of the missing-GABA compensation in **every cache** — are
maximum-likelihood fits over **pooled condition rows**. The conditions are
labelled explicitly in `connectivity_fit.py`'s `datasets` dict, and the
pools mix them:

| group | depleted rows in the pool | baseline rows |
|---|---|---|
| dSPN→dSPN | 6-OHDA 0/7, reserpine 0/8 | 5/19, 3/7, … |
| iSPN→dSPN | 6-OHDA 3/12 (25 %), reserpine 1/10 (10 %) | 13/24 (54 %) |
| iSPN→iSPN | 6-OHDA 3/17, reserpine 5/18 | 14/39, 4/9, … |
| FS→dSPN | 6-OHDA 43/80 (54 %) | 58/96 (60 %) |
| FS→iSPN | 6-OHDA **66/86 (77 %)** | 42/108 (39 %) |

The last two rows are the crux and reproduce the measured Gittis 2011
effect: under depletion FS→iSPN roughly doubles while FS→dSPN barely
moves. Pooling both into one fit **reverses the FS target preference** —
every healthy dataset in the pool prefers FS→dSPN (60 % against 39 %),
while the fitted kernels prefer FS→iSPN at short range (amplitude 0.918
against 0.599). And the two targets of the same FS axon come out with
σ = 394.2 µm (dSPN) against σ = 140.0 µm (iSPN), a **2.8-fold** difference
with no anatomical reading. The SPN–SPN collapse under depletion (Taverna
2008 — the very rows in this repository's own spreadsheet) dilutes the
dSPN→dSPN and iSPN→ kernels the other way.

**What is not affected.** Both DBS conditions share the kernel, so the
on-vs-off contrast is not directly biased. The damage is to the absolute
state: the kernels correspond to no preparation that exists, `E_outer` and
therefore every cache inherits it, and §4's rate self-consistency is
computed against it.

**Why it is a consistency problem, not a preference.** The project chose
its rate anchors by state deliberately — the medication-off values of
Liang et al. 2008 (§20) — and §30 asks the same question of φ. §39 records
that the subject's actual medication state is undocumented, and seat 1
frames all of these as one requirement: rates, connectivity kernel and φ
must describe the same physiological state. The connectivity is currently
the only one of the three chosen by no state at all.

**The task.**

1. Refit the kernels **per condition** from the already-labelled rows —
   the fit itself costs seconds — and adopt the state that matches the
   answer §39 produces, either as a directly fitted depleted kernel or as
   the healthy kernel plus explicit, cited depletion factors (the
   lineage's own encoding, e.g. Lindahl 2016 Table 9). Record the
   rodent-acute-model caveat: the state difference is measured, but
   chronic human transfer is not established.
2. Ask the same state question of `get_weights.py`'s IPSC-amplitude
   mixtures (CompNeuroPy `striatal_microcircuit/get_weights.py`).
   Verified during triage: those components carry **no condition labels
   at all** — they are counts from a spreadsheet with no state recorded —
   so the question there is open rather than answered wrongly.
3. Document the choice where the kernels are documented, and rebuild the
   short caches.

**Blocking.** Cache-invalidating: it changes `E_outer` and every
missing-GABA stream, so it is one of the findings the Roadmap orders
inside phase 1, **before any full-length build**. It also depends on §39 —
which state to adopt cannot be settled before the subject's state is
known — and feeds §4.

### 44. Cross-channel convergence: the caudate control is an upper bound

*Opened 2026-08-27 13:51*

**Opened 2026-08-27 13:51:**

From the round-2 community review (accepted from
`community_review/round2/synthesis.md`, its F12 — two seats; evidence
class experimentally grounded, both halves measured; the synthesis is not
referenceable, so the substance is restated here).

The two BG loops share no projection, only the putamen loop is stimulated,
and the caudate loop's on-vs-off BOLD change is the designed free control
(§2). Two independently measured facts say the model's channel separation
is cleaner than the subject's:

- **The anatomy has cross-channel convergence, running the direction that
  matters here.** Corticostriatal terminal fields from areas 5 mm apart
  overlap by 50 % (Averbeck 2014, macaque tracing). Through GPe and STN,
  segregation holds for associative regions but *not* for motor ones,
  which receive from associative territories (Shink 1996, EM-level). And
  parkinsonism further degrades functional segregation (four
  receptive-field despecification studies; within-list consensus across
  Haber 2016, Emmi 2020, McGregor & Nelson 2019). The strongest route the
  reviewed anatomy supports is therefore associative → motor — exactly the
  one the model would need — and the model wires zero.
- **The subject's own VTA reaches the associative STN.** Verified during
  triage against `experimental_data/berlin_data/vta/sub-01/`: associative
  overlap 5/68 (lh) and 7/68 (rh), i.e. 12/136 ≈ **0.09**, which the
  two-loop design rounds to zero. The real caudate-territory on-off change
  therefore contains a small *direct* DBS component that the model will
  attribute entirely to cortical drive. (The limbic overlap, 6/109 ≈ 0.055,
  has no representation in the model at all — it stands for neither loop.)

**The consequence.** Any caudate-versus-putamen contrast is an **upper
bound on channel independence**, not a clean control reading, and the
pre-registered pass/fail that §2 now owes inherits both cracks.

**The task.**

1. **Document both facts where the control is defined** — §2 and
   `model_v07.md` §1. The 8.8 %/0.09 figure is already owed by §38 part 3
   as a caveat on the pooled-ROI loop weights; this entry states what it
   means *for the control* rather than restating the arithmetic.
2. **Bound the 0.09 effect**, analytically or with one sensitivity run
   that gives `stn:caudate` its measured coverage. Note the implementation
   cost: the caudate loop currently sits in `get_loss`'s
   `excluded_populations_list`, so a run like this is a compile-level
   extension of the DBS retrofit, not a parameter change.
3. **Open option, deliberately not decided here:** seat 7's testable
   version — add the single associative-GPe → motor-STN projection the
   reviewed anatomy supports most directly, in its own optimizer cluster
   so the fit can drive it to zero. A fitted non-zero value would be the
   model's own estimate of channel leakage: a result rather than an
   assumption. Against it: it breaks the two-independent-loops
   architecture that `CLAUDE.md` documents and that makes the free control
   possible at all, and it adds a parameter (§41 records the same
   conditioning trade-off). Decide when parts 1 and 2 are in hand.

**Blocking.** Nothing cache-side. Part 1 belongs before §32's on-fit is
interpreted, with §2's checklist. Part 3, if ever taken, is a model change
and needs a captured baseline first.

### 45. The corticostriatal proportion table is reused for STN, GPe and thalamus

*Opened 2026-08-27 13:56*

**Opened 2026-08-27 13:56:**

From the round-2 community review (accepted from
`community_review/round2/synthesis.md`, its F13 — one seat; evidence class
experimentally grounded; the synthesis is not referenceable, so the
substance is restated here).

Verified during triage: `BGM_v07` passes the same
`mc.cortical_proportions_dict` to `Microcircuit` **and** to
`CorticalInputs` (`model_creation_functions.py`, both construction sites
read the same key), so one table derived for corticostriatal afferents
also sets the cortical mixes of `thal`, `gpe_arky`, `gpe_cp` and `stn`.
`experimental_data/cortical_proportions/README.md` describes corticostriatal
afferents throughout — "the model splits each striatal neuron's cortical
afferents…" — and never claims another target. The reuse is documented
nowhere. With the current numbers a putamen `stn` neuron draws **13 % of
its 500 cortical afferents from S1**, a region for which the one primate
study that looked reports no subthalamic projection, and a caudate `stn`
neuron draws 55 % from dlPFC.

**Why it reaches the inference.** The corticosubthalamic drive is the only
excitatory input `stn` receives, and the hyperdirect route is the
DBS-relevant one (§15). The mechanism is the one `parameters.py` states
itself: each region's rate series is normalised to mean 5 Hz and the mix
sums to 1, so the mix does not change the mean drive — it changes variance
and **timing**. Timing is what the loss is made of.

**The evidence, graded.** Consensus that the mixes differ (Emmi 2020 for
the corticosubthalamic topography, Haber 2016 for the corticostriatal one),
and the established corticosubthalamic topography — M1 dorsolateral,
SMA/ventral-premotor medial with inverse somatotopy (Nambu 1996–2000,
confirmed Miyachi 2006) — does not resemble the striatal proportions. The
specific S1 negative is weaker: one voice (Von Monakow 1978, macaque
autoradiography), and the seat says plainly that a single-study negative
from 1978 counts for less than a replicated positive.

**The task**, in two parts of different standing:

1. **Owed regardless: document the reuse** in both
   `experimental_data/cortical_proportions/README.md` and `model_v07.md`
   §8 — that a table derived for one target is currently setting the
   cortical mixes of three others, and what that implies for the timing of
   the drive.
2. **Attempt a separate `CorticalInputs` table, if the evidence licenses
   one.** Minimally, per the review: S1 → 0 for STN, motor/premotor/SMA
   dominant per the established topography, dlPFC marked uncertain (Von
   Monakow and Haynes & Haber 2013 disagree). But note the asymmetry that
   made the striatal table possible: Borra et al. report quantitative
   *fractions*, while the corticosubthalamic literature is largely
   topography. If no comparable quantitative source exists, then "no
   defensible separate table — the striatal one stays as a stated
   assumption" is a legitimate outcome of this entry, provided part 1 makes
   the assumption explicit.

**Implementation notes.** `get_loss` already reads
`mc.cortical_proportions_dict` and `ci.cortical_proportions_dict` as
separate attributes, so the structure for two tables exists; both are
currently fed the same dict. The cortical rate `.npz` is **not** affected:
only v08 is driven by the mixed `caudate_rate`/`putamen_rate`, while v07
uses the per-region series (`get_loss`'s own note on
`cortical_proportions_json` says the same).

**Blocking.** Cache-invalidating for the **CI caches** if part 2 changes
the table, so it sits in Roadmap phase 1 before any full-length build. Part
1 blocks nothing and can be done immediately.

### 46. Renormalising over seven ROIs asserts the omitted cortex covaries with the retained

*Opened 2026-08-27 14:14*

**Opened 2026-08-27 14:14:**

From the round-2 community review (accepted from
`community_review/round2/synthesis.md`, its F14 — one seat; evidence class
experimentally grounded, narrow; within-list consensus that the omitted
input is large and concentrated. The synthesis is not referenceable, so
the substance is restated here.) Distinct from §45, which concerns the same
table being reused for targets it was not derived for; this entry concerns
what the renormalisation itself asserts.

The model has exactly seven cortical ROIs, fixed by
`sub-01_subdiv_results.h5`: **dlPFC, preSMA, PMd, PMv, SMA, M1, S1**. Each
carries one deconvolved firing-rate series per TR. The proportion columns
renormalise over these seven and sum to 1, so *all* of a receiver's
cortical afferents are driven by the seven retained series. The omitted
afferents are therefore not dropped — they are **reassigned** to the
retained series in proportion to the retained shares. In the caudate loop,
where dlPFC holds 0.55, dlPFC absorbs most of the omitted limbic and
associative input: an afferent from rostral cingulate is given the dlPFC
time course.

That is the assumption: not "we ignore these regions" but "**these regions
fluctuate, TR by TR, like the retained ones**". The loss is a time-course
correlation, so the assumption is about precisely the fitted quantity, and
the omitted regions are limbic and associative — functionally distinct from
the retained seven. `experimental_data/cortical_proportions/README.md`
states the omission ("relative shares among seven regions, not absolute
shares") but not this consequence.

**The asymmetry, computed during triage from the README's own Borra 2022
Table 2.** The mapping from Borra's region groups to our seven ROIs is not
clean — the "motor" group is retained wholesale (F1–F7), but of "prefrontal"
only dlPFC is retained (orbital and ventrolateral are not) and of "parietal"
only S1 — so exact retained fractions cannot be read off the table. What can
be stated is a **lower bound** on the omitted fraction, from the four groups
that are wholly absent (rostral cingulate, caudal cingulate, insula,
temporal):

| injection site | at least omitted |
|---|---|
| caudate, lateral head | **46.8 %** |
| caudate, medial head | **46.7 %** |
| putamen, rostral | 36.6 % |
| putamen, dorsal motor | 20.8 % |
| putamen, middle motor | 12.8–12.9 % |
| putamen, midventral motor | 5.4 % |
| caudate, body | 9.6 % |

Plus the non-dlPFC prefrontal and non-S1 parietal remainders. The caudate
head omits at least ~47 % against 5–21 % at the motor putamen sites, so the
caudate loop's drive is the more heavily reconstructed — and the loop
contrast is what the inference reads. Rostral cingulate alone, the largest
omitted category, is 21.5/30.6 % at the caudate head against 2.6–15.3 % at
motor putamen. (The caudate *body* is the exception, being motor-dominated;
`Cau` and `Put` are whole nuclei, so the README averages over
subterritories.)

**The task**, three parts:

1. **Owed now: document the assumption** and the bounds above in
   `experimental_data/cortical_proportions/README.md`, beside the existing
   renormalisation caveat — including that renormalisation reassigns the
   omitted mass proportionally, so in the caudate loop dlPFC carries it.
2. **The bracketing test, attached to phase 2.** The review proposes
   columns summing to the retained fraction with the remainder driven flat
   at the same 5 Hz mean. Rejected as designed: a flat remainder removes
   the *variance* of ~half the afferents as well as their covariance, and
   `experimental_data/input_streams/README.md` §3 shows input correlation
   dominates simulated BOLD amplitude — so the comparison would largely
   measure the variance loss. Use instead a **variance-matched but
   decorrelated surrogate** for the remainder (e.g. phase-randomised
   versions of the series), which isolates the covariance claim. The two
   runs then bracket it: perfect correlation with the retained regions
   against none. Cost: the proportions are baked into the caches, so this
   needs a rebuild — build short first to confirm the drive statistics
   move at all, and attach the full-length BOLD comparison to phase 2's
   build rather than paying for it separately.
3. **A pre-stated decision rule for the data request.** Asking Berlin for
   the missing ROIs is the real fix, and it is expensive: new deconvolution
   (`cortical_drive_by_bold_run.py`, which needs MATLAB and an interactive
   MathWorks sign-in — §21), a re-derived proportion table for the new
   region set (Borra's groups are coarse, so more ROIs does not
   automatically mean better numbers), a regenerated rate `.npz`, and every
   v07 cache rebuilt. So part 2 is the decision procedure for part 3: fix
   a threshold beforehand — if the per-region BOLD correlations shift by
   less than it, the request is not warranted and the omission is recorded
   as a quantified limitation; if more, the request is made **with a
   number attached**. The request stays a named, untaken option here until
   then.

**Blocking.** Nothing blocks part 1. Parts 2–3 are cache-touching and
belong with the phase-1/phase-2 boundary; the decision rule must be fixed
before the comparison is run, not after.

### 47. The afferent `antidromic_prob`: its size is underived, and its subset is redrawn every pulse

*Opened 2026-08-27 15:15*

**Opened 2026-08-27 15:15:**

Opened from the round-2 community review (its F16 — one seat, evidence
class experimentally grounded; `community_review/round2/synthesis.md`, not
itself referenceable), **with a second and more consequential problem found
during triage** and made the leading half. Both live at the same code site,
`DBSstimulator._set_antidromic`'s afferent branch.

**Half 1 — the magnitude is not derived (found in triage, not in the
review).** `_set_antidromic` treats its three cases differently: the
stimulated population itself gets `antidromic_prob = 1` gated by its own
`dbs_on_array` (coherent — an axon of a neuron inside the VTA is certainly
activated), passing fibres get the summed branch strengths, and **afferent
populations get `antidromic_prob = np.mean(stim_pop.dbs_on)` = 0.4**. Only
that last case makes an inference, and the inference does not follow:

- 0.4 is measured as the **tissue fraction of the STN inside the VTA**
  (58/145 motor voxels, §16). The quantity needed here is different: what
  fraction of *afferent somata* is reached antidromically.
- Antidromic invasion needs only **one** activated branch. If a `gpe_proto`
  neuron has *n* terminal branches spread over the STN and 40 % of the STN
  is in the field, the probability that at least one is hit is
  1 − 0.6ⁿ — 99.4 % at n = 10, effectively 1 for a realistic arborisation.
  Axons merely *passing* through the field are activated too, which raises
  it further.
- So 0.4 holds only under **strong topography with compact terminal
  fields** — each pallidal axon arborising in a restricted STN subregion,
  so that ~40 % of those fields fall wholly inside the VTA. That is a
  substantial anatomical assumption and it is stated nowhere.

**And the model assumes the opposite.** `gpe_proto__stn` is
`connect_fixed_number_pre` with `number = 10`: each `gpe_proto` neuron
contacts ~10 STN neurons drawn at random across the whole population, with
no topography anywhere in the BG populations (only the striatal
microcircuit has a lattice). Under the connectivity the model actually
implements, the defensible figure is ~99 %, not 40 %. The connectivity
assumption and the `antidromic_prob` assumption contradict each other.

Why it matters for the inference rather than only for realism: if the true
fraction is near 1, the afferent antidromic effect is ~2.5× larger than
represented, and the free parameters that could absorb the difference are
`axon_spikes_per_pulse` and the `gpe_proto__stn` cluster scaling — i.e.
exactly the quantities the project intends to interpret. Note this is a
step beyond §16, where 0.4 is listed as a measured subject value: for the
afferent case it is a *derived* value carrying its own untested assumption.

**Half 2 — the subset is redrawn every pulse (the review's F16).**
`unif_var_dbs2` is a per-neuron, per-timestep uniform, so a different
subset of `gpe_proto:putamen` is invaded at each pulse; only the stimulated
STN's set is fixed (`_create_dbs_on_array`). The invasion is in fact doubly
stochastic — the axon spike must fire first (`unif_var_dbs1` against
`prob_axon_spike`), then the soma is invaded (`unif_var_dbs2` against
`antidromic_prob`) — so the expected invaded fraction per pulse is
`axon_spikes_per_pulse × 0.4`, over a set that changes every time.

The measurement says fixed: the pulse-triggered cortical evoked potential
is stable pulse-by-pulse across ~215 pulses in all six rats (Kumaravelu
2018 Fig. 5), a reproducibility only the same axons every time can produce;
and the same source shows antidromic propagation is unreliable at 130 Hz
(R1 reduced against 9 Hz, Fig. 4B1) — **stochastic thinning within a fixed
axon population**, not resampling across the nucleus. A fixed subset
produces persistent pallidal heterogeneity (a strongly perturbed minority
beside an untouched majority, as Kumaravelu 2016 §4.2 describes); a
per-pulse redraw applies a diluted, identical-in-expectation perturbation
to every neuron and cannot produce it.

**A connectivity-derived mask is not available, and the reason is the same
as half 1's.** Computing which `gpe_proto` neurons contact stimulated STN
neurons gives ~99 % of them (0.6¹⁰ ≈ 0.6 % contact none), because the
model's connectivity is diffuse and carries no topographic content. Under
that connectivity the `gpe_proto` neurons are **exchangeable**, so the
identity of the mask is meaningless and only its persistence matters — a
fixed random draw, exactly as `_create_dbs_on_array` already is for the
STN, is the faithful implementation. No connectivity analysis is needed or
useful.

**The task.**

1. Settle the afferent `antidromic_prob`: derive it, or replace it, or
   record it as an explicit assumption with the topography it presumes —
   and say how it relates to the connectivity the model implements.
2. If a mask is adopted, gate **both** mechanisms with it. Currently
   `prob_axon_spike` is `: population`, so every `gpe_proto` neuron emits
   axon spikes regardless. Coherently, a neuron either has its axon in the
   field — then it fires orthodromically *and* antidromically with the
   propagation-reliability probability — or it does not. That also makes
   the measured 125 Hz propagation failure expressible as reliability
   instead of conflating it with coverage.
3. Report the consequence with §36's diagnostics: a fixed mask predicts a
   **bimodal** `gpe_proto:putamen` rate distribution, a per-pulse redraw a
   unimodal one.

**Cost and risk.** Small: `antidromic_prob` is `: population` today and
would become a LOCAL 0/1 mask, exactly as `dbs_on` already is, with
`unif_var_dbs2` retained for reliability — so **the number of random
variables is unchanged** and the global-RNG caveat in `CLAUDE.md` is not
triggered. The change is invisible to `get_firing_rate_loss`, which scores
population means, and visible only to §36 — so it cannot degrade the fit.
A baseline must still be captured first, per the repository convention.

**Blocking.** Not cache-side. Half 1 should be settled before §32's on-fit
is interpreted, since it changes what a fitted `axon_spikes_per_pulse` or
`gpe_proto__stn` scaling means; it also belongs with §16's write-up
obligations.

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

### 13. Cache state files store the cortical rate path as a bare string

*Opened 2026-08-04 09:39 · resolved 2026-08-13 11:36*

**Opened 2026-08-04 09:39:**

`Microcircuit` and `CorticalInputs` compare the saved `cortical_rate_path` verbatim
against the one they are given, so a cache built from `../striatal_.../x.npz` is
rejected when the same file is later named through a different path. This is why
`build_input_caches.py` has to be run from `BOLD_optimization/`, like `get_loss.py`.
Harmless once known; worth normalizing to a resolved absolute path if the caches are
ever built from somewhere else.

**Resolved 2026-08-13 11:36:** Both classes now resolve `cortical_rate_path`
to an absolute path at construction (`Path(...).expanduser().resolve()` in
their `__init__`s), record the resolved string in the state file, and refuse a
state file that lacks it. `parameters.py` builds `mc.cortical_rate_path` from
its own file location instead of storing `../striatal_...`, so the comparison
no longer depends on the launch directory at all (the *other* paths in
`parameters.py` are still relative — everything keeps being run from
`BOLD_optimization/` for those). Trade-off, accepted: the recorded path is
machine-specific, so a cache copied to a host where the repo sits at a
different absolute path is refused instead of matching by coincidence; the
full-length caches are built directly on the workstations anyway (§6).
Covered by `CompNeuroPy/src/CompNeuroPy/test/test_striatal_state_validation.py`
(reload through a different cwd + relative spelling passes; absent field
refused), which ran green on 2026-08-13. Done together with §29's hardening
pass, before any cache exists, so nothing was invalidated.

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

### 19. `parameters.py` labels the *planned* cache size as the current one

*Opened 2026-08-05 10:12 · 1 update, 2026-08-07 15:28 · resolved 2026-08-11 09:16*

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

**Resolved 2026-08-11 09:16:**

The comment was replaced with the qualified form drafted above: ~1.25 TiB per
DBS condition in the current `float64`/per-region layout, ~120 GiB after the
TODO §3 relayout (arithmetic, not yet measured). Carrying both numbers is what
defuses the "opposite confusion" that kept this entry from acting in isolation.
Acting now rather than waiting for §3: every cache was deleted on 2026-08-06
(the striatal-rate change), so the 22.28 GB measurement above survives only as
recorded here; the §22 generator rebuild left storage, not generation time, as
the bottleneck (§3, update 2026-08-11); and the workstation move — where
`/scratch` actually gets provisioned from this comment — is the next phase.

Two figure convergences landed with it. The planned-layout size is now stated
as **~120 GiB per DBS condition** (the 116 GiB arithmetic above, rounded) in
`parameters.py`, `PLAN.md` (both mentions), §3's new update block,
`model_v07.md` §7.6 and `model_v08.md`'s comparison table, superseding the
underived ~138 GiB. And `CLAUDE.md`'s "~42 GB for `--n-trs 5`",
whose both-conditions scope ("all four") was dropped in commit `c0cbd0e` and
which had become ambiguous next to the per-condition TiB figure beside it, was
rescoped to ~22 GB per DBS condition (~44 GB for all four caches) — consistent
with the measurement above.

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

### 27. The realised `f(d)` of the geometric pools is not checked at build time

*Opened 2026-08-07 15:28 · resolved 2026-08-13 11:36*

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

**Resolved 2026-08-13 11:36:** Implemented as proposed.
`_expected_shared_for_d` (the nested quadrature) was restored from pre-`fd2cbde`
history into `microcircuit.py`, and a new
`Microcircuit._check_realised_shared_fractions` runs at every build, directly
after the `E_outer` degree check and before any stream is written: it evaluates
the analytic `f(d) = E_shared(d) / E_outer` on a 16-point grid (interpolated to
pair distances), compares the mean realised shared fraction of the ~30–500
receiver pairs around three probe distances (5th/50th/90th percentile of pair
distances below `2·r_out`), and raises beyond a 25 % relative tolerance. Probes
with analytic `f < 1e-3` are recorded but not judged — a relative test on a
vanishing value only measures noise. The probe table is stored in
`stream_statistics["<pre>-<post>"]["f_d_check"]`, so every cache carries the
comparison. On the caveats: the tolerance was set against measured scatter —
across 6 seeds x 3 probes on a synthetic lattice the worst deviation of the
binned means was ~12 % (largest where `f` is smallest), while a wrong `r_in` or
kernel sigma moves the analytic value far outside 25 % and fires. Verified at a
10³ lattice *and* a larger, sparser 14³ one (the §24 scenario), in
`CompNeuroPy/src/CompNeuroPy/test/test_striatal_f_d_check.py`, green on
2026-08-13. Note the check is deliberately density-blind: `rho_pre` cancels out
of `f(d)`, so density errors remain the degree check's job. Cost: a few seconds
per build.

### 29. `CorticalInputs`' cache validation is strictly weaker than `Microcircuit`'s

*Opened 2026-08-07 15:28 · resolved 2026-08-13 11:36*

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

**Resolved 2026-08-13 11:36:** Implemented as specified, and the same pass
hardened all three state-file loaders beyond the entry's minimum, since no
cache exists and stricter costs nothing:

- `CorticalInputs._save_cortical_input_state` now records
  `shared_fraction_dict`, `cortical_correlation`, `correlation_window_ms`,
  `correlation_timescale_ms` **and** `stream_statistics` (audit parity with
  the MC states); `_load_cortical_input_state` compares them all.
- **Every** compared field in all three loaders
  (`CorticalInputs._load_cortical_input_state`,
  `Microcircuit._load_cortical_input_state`,
  `Microcircuit._load_missing_input_state`) is now hard-required — the old
  `payload.get(field, current)` / `if saved is not None` patterns treated an
  absent field as a match.
- The self-referential "key set" check (in both cortical-input loaders) was
  replaced by a comparison against the pairs the **current configuration**
  would generate, reproducing the loop logic of the respective
  `_simulate_*_spike_counts`.
- Each stream's stored receiver count `R` is checked against the current
  population size (`pop.size` / `type_counts`), which nothing verified before.

Covered by
`CompNeuroPy/src/CompNeuroPy/test/test_striatal_state_validation.py` — a real
tiny CI cache built with mock populations plus crafted MC state files; every
mismatch and every stripped field is proven to raise — green on 2026-08-13.
Done together with §13 (the rate-path normalization), before any cache was
built through the weaker path.

### 33. PLAN.md dissolved into this file

*Opened 2026-08-11 10:04 · resolved 2026-08-11 10:04*

**Opened 2026-08-11 10:04:**

`PLAN.md` (started 2026-08-03, deleted today) mixed four kinds of content:
project history, still-binding decisions, an ordered step sequence, and
near-duplicates of open entries here. Oliver decided that everything
forward-looking lives in this file alone: the ordering became the Roadmap
section, decisions that describe current state moved to `CLAUDE.md`, decisions
that were really inputs to open work merged into their entries (§1, §2, §3,
§32), and the history is preserved below. This entry is resolved on arrival —
it records history, not work owed.

**The old step numbers**, for the historical "PLAN.md step N" references in
resolved entries and commit messages: steps 1-6 are the completed work recorded
below; step 7 = §1 (bounds); step 8 = §6 + §3 (workstations, full caches in the
new layout); step 9 = §31 (mini-run); step 10 = §32 (the fits).

**Why the plan existed.** The optimization was written in Dec 2025, ran on the
workstations, and stopped without a message. The cause was a compound failure:

1. `infer_max_sim_time_ms` computed `len(rates) * 2.31 * 1000 / dt_ms` =
   7,161,000 ms instead of 716,100 — every evaluation simulated **10x too
   long**.
2. `BoldMonitor` recorded every 0.1 ms, so a run stored ≥4 GB of BOLD per
   process.
3. `run_optimization.sh` launched 2 x lambda=12 = **24 processes**; on hinton
   that is ~96 GB against 125 GB of RAM before the vector-of-vector overhead
   and before `get()` doubles it. The OOM killer took them.
4. CompNeuroPy's `_ScriptRunner.run` turns any non-zero child exit into a bare
   `exit(1)` — no message, no traceback — and the children's output went to a
   console that `run_optimization.sh` had backgrounded with `&`.

Separately, v07 had been abandoned as "too slow". It was not: its C++
simulation runs at 1.81 s per simulated second versus v08's 1.16. **96% of
v07's runtime was Python overhead** in the input-delivery machinery, and the
projected 1.25 TiB of cache was the same design showing up as storage.

**Decisions taken (with reasons), and where each now lives:**

- **Model: v07, all three components** — structured/correlated cortical input,
  the distance-dependent microcircuit, the missing-GABA compensation. v08 was
  only ever a time-pressure fallback and stays alive purely as a fast
  end-to-end pipeline test. → `CLAUDE.md` "Two model versions".
- **Speed: patch ANNarchy, keep the statistics exact.** The generator cannot
  move inside ANNarchy (random distributions need global arguments; no
  inverse-CDF functions), and regenerating on the fly measured 181 s per
  simulated second — 4.5x worse than the precompute. Oliver's original
  precompute design was right; the fix was the transfer, not the algorithm.
  → history (here); the ANNarchy limitation is a `CLAUDE.md` gotcha.
- **Cache layout: pre-summed per postsynaptic type, `uint16`,
  `(time, receivers)`, on `/scratch/olmai`.** → §3.
- **Loss: per-region BOLD time-course correlation** (not functional
  connectivity — the cortical drive comes from the same recording, so the
  model is asked to reproduce what this subject's basal ganglia actually did,
  TR by TR) plus the firing-rate plausibility term, with the cheap rate probe
  gating the expensive BOLD run; CMA-ES is rank-based, so the skip-and-charge
  ordering stays consistent. → `CLAUDE.md` "Running things"; the gate
  calibration is §10.
- **Parameters: 19 for v07** — 3 striatal cortical input weights, 4
  `CorticalInputs` weights, 2 baseline currents (snr, gpe_proto have no
  cortical input), 10 projection-cluster scalings. Clusters scale the
  literature weights rather than freeing them, preserving their relative
  balance and conditioning the search. → `CLAUDE.md` "Running things".
- **DBS-on staging: only the putamen loop's weights move**, plus the 3 DBS
  parameters. → §2 (with the free-control rationale) and §32.
- **DBS: retrofit the mechanisms before compile, in both conditions**, never
  `DBSstimulator(auto_implement=True)`; off and on then differ only in
  parameter values, which is the claim the inference rests on. The price:
  off-condition numerics changed (one global RNG stream), taken deliberately
  before any real fit had run. → `DBS.md`; the before/after values are in the
  step-6 record below.
- **Seeds: fixed at 42 during fitting**, winner re-run across ~10 seeds. → §32.
- **Robustness: penalize and continue** — per-individual logs with exit codes,
  worst-case loss 10.0 for a dead individual, hard abort past half a
  generation, CMA-ES checkpoint every generation with `--resume`.
  → `CLAUDE.md` "Running things".
- **Deferred: the DBS inference design.** → §2.

**Progress: steps 1-6, all done and verified (2026-08-03 to 2026-08-04).**
Step 1 patched ANNarchy's `TimedArray` and proved the output bit-identical;
step 2 removed the `cyInstance` workarounds; step 3 rebuilt `get_loss.py`
(bug fixes + `--model-version`, v08 reproducing its pre-refactor loss exactly);
step 4 built a short v07 cache and executed the v07 path for the first time;
step 5 added the gate, logging, checkpointing and failure policy and rewrote
`deap_cma_opt.py`; step 6 made the DBS-on path real. Measured on the laptop,
v07 went from **41.87 to ~3.25 s per simulated second** — a full 716 s
evaluation from ~500 min to **~39 min**, against a C++ floor of ~21 min.

Running the v07 path (step 4) required fixing three things that had never been
reachable: `CorticalInputs` had to learn the coarse per-TR drive
(`Microcircuit` already had it); `update_time` moved 100 → 110 ms so every
simulated stretch divides into whole chunks (the ramp-up TR is 21 chunks; the
probe moved 10,000 → 9900 ms, changing the v08 rate loss by 1e-4, so v08 is no
longer bit-identical to its pre-refactor reference by exactly that much); and
the 3 DBS parameters moved from fixed indices 21-23 to the last three slots.
The first end-to-end v07 evaluation (5 TRs, DBS off, drive weights 0.001, base
currents 100, scalings 1) gave `loss 1.5430 = firing_rate 0.8678 + bold
0.6752`; the drive-weight sweep proving the parameters reach `mc` and `ci` is
§9's table (resolved, merged into §1).

**Step 6: the DBS-on path was never real.** The staging was written but no
DBS-on evaluation had ever run — no loss file, result, checkpoint, log or
cache existed. Two defects were waiting: (1) v07 + `--dbs on` could not build
the model at all — `DBSstimulator(auto_implement=True)` clears the network and
recreates projections through `_connector_methods_dict`, which has no
`"Specific"` key, so v07's `CurrentInjection` inputs die with a `KeyError`;
(2) even on v08, DBS was silently inert — `on()` ran after `compile()`, wrote
to the C++ instance and not `pop.init`, and the first `reset()` restored
`dbs_on = 0` while the loss JSON still said `"dbs": "on"`. Both fixed by the
retrofit decision plus calling `on()` **before** `compile()`.

The off-condition regression from the retrofit (v07, 5 TRs, reference vector):
total 1.5430 → **1.5442** (rate 0.8678 → 0.8705, BOLD 0.6752 → 0.6737) — the
RNG stream moved, but little, because v07's DBS footprint excludes the
1000-neuron microcircuit that dominates RNG consumption.

The DBS effect is real and confined to putamen (v07, 5 TRs,
`dbs_depolarization` 3.0, `passing_fibres_strength` 0.5,
`axon_spikes_per_pulse` 0.5), Hz:

| population | off | on | Δ | | caudate twin | Δ |
|---|---|---|---|---|---|---|
| snr:putamen | 139.11 | 112.63 | **−26.48** | | snr:caudate | +0.71 |
| gpe_proto:putamen | 111.90 | 88.97 | **−22.93** | | gpe_proto:caudate | +0.79 |
| stn:putamen | 18.53 | 10.69 | **−7.84** | | stn:caudate | −0.79 |
| thal:putamen | 4.56 | 3.12 | −1.44 | | thal:caudate | −1.18 |

The sub-1 Hz caudate changes are the cortical drive differing by condition,
not a DBS leak; on v08, comparing the two on-runs on the *same* drive with
only the DBS parameters differing, every caudate population is identical to
2 dp while nine putamen populations move — the free control holds exactly
(§2's update of 2026-08-11). The whole fitting loop ran in both conditions on
both versions (off mini-run → on mini-run seeding via `load_best_off_fit` →
`--resume`), and `test_dbs_on.py`'s 24 checks pass. Both mini-runs needed
`--gate-threshold 1.0` (§10). A laptop limit surfaced on the way: `--lambda 4`
lost two of four `cc1plus` processes to the OOM killer during the
per-individual compiles; `--lambda 2` was fine (§6's update of 2026-08-04).

**Commits, by plan step:**

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
(`test_microcircuit.py`) and CompNeuroPy `5fcc6b9` (`spike_input_cortex.py`
demo), both produced for the SPP-2041 meeting.

**Resolved 2026-08-11 10:04:**

Resolved on arrival — this entry is the historical record of `PLAN.md`, not
work owed. Its forward-looking remainder went into §1 and §6 (as Updates of
today), the new §31/§32, and the Roadmap; `PLAN.md`'s "Immediate next
actions" were checked against §1/§3/§6 and found already recorded there
(the ~7 h / ~120 GiB cache budget verbatim in §3), except the push status,
which was re-verified and is §6's update of today. `CLAUDE.md` now states the
loss design and parameter rationale as current state, and every living-document
reference to `PLAN.md` was repointed in the same sweep.

### 34. Survey the BG-modeling community's conventions via reviewer personas

*Opened 2026-08-13 10:38 · resolved 2026-08-13 20:54 · 5 updates, last 2026-08-15*

**Opened 2026-08-13 10:38:** We do not know whether the model violates
conventions of the active basal ganglia neurocomputational community — things
every comparable model does that ours silently doesn't. The task is a
structured survey, in three steps, producing review documents; implementing
any accepted change is explicitly *not* part of this entry — each accepted
proposal spawns its own numbered entry, and §34 resolves when the review
documents exist.

**Step 1 — the panel.** Select 5-8 research groups/lineages (the unit is a
group sharing one modeling approach — e.g. Rubin-Terman counts once — not an
individual). Primary ranking: similarity of the modeling approach to ours
(mesoscopic, populations of point neurons, possibly spiking, multiple
functionally connected BG regions) weighted with citation impact. Detailed
single-cell/morphology modeling is dissimilar regardless of citations.
Secondary criterion: the panel must include at least 1-2 groups that model
DBS at the network level, so the mechanisms of §14-§17 get a reviewer with
standing (Jonathan Rubin qualifies on both counts and is the seed
suggestion). The Hamker/Chemnitz lineage is included as one persona, judged
against its own published standards. **Strict recency:** both the groups and
the papers must be from roughly the last 10 years (2016+). Accepted
consequence: foundational papers (Terman & Rubin 2002/2004, Humphries 2006)
are excluded, so each persona is reconstructed from recent work only —
conventions stated long ago and silently assumed since may be missed. No
seat for BOLD/whole-brain (mean-field) groups: they fail the
approach-similarity criterion; instead the synthesis must explicitly record
that the BOLD pipeline had no peer reviewer, as a known limitation.
Checkpoint: the group list is confirmed by Oliver before step 2.

**Step 2 — the reading list.** 2-4 papers per group (~15-25 total): the
flagship network-model paper of the recent era plus the most recent relevant
one, more only if the group's approach shifted. Delivered as a DOI list;
Oliver downloads the PDFs, and step 3 reads the saved full texts, not
abstracts. Checkpoint: the list goes to Oliver for download before any
review is written.

**Step 3 — the reviews.** One review per group, written in character as that
group reviewing our model, applying the group's *full* standards unfiltered
— structure (regions, neuron types), connectivity, dynamics, validation
data, DBS representation. Every raised point is tagged with whether it
plausibly matters for our stated goal (single-subject resting-state BOLD
fitting and DBS inference), so nothing is pre-filtered but triage is
pre-structured. Then one synthesis document: merge overlapping points, rank
by how many personas raise them (convergence across groups = community
convention; a single voice = one lab's taste), and attach one concrete
change proposal per point — exactly what we would modify.

**Location:** new top-level `community_review/` — committed README (group
list with selection rationale, DOI reading list) and committed review
documents; PDFs saved beside them but untracked (the remote is public; cite
DOIs, never commit publisher PDFs — the `experimental_data/` pattern).

**Ordering:** the survey is laptop reading work and runs in parallel with
the workstation/cache track (§6, §3), but the fits (§32) must not launch
until the synthesis is triaged — a structural finding discovered after the
fits would mean paying for them twice. See the Roadmap.

**Update 2026-08-13 (step 1 done):** panel selected, written up with
selection rationale in `community_review/README.md`, and confirmed by
Oliver — all six seats: Kumar–Hellgren Kotaleski (KTH), Rubin/Verstynen
(Pittsburgh/CMU, DBS standing, the seed), Girard–Doya (ISIR/OIST), Grill
(Duke, DBS standing), Hamker (Chemnitz, own lineage), Chakravarthy (IIT
Madras, optional-seat-made-firm). Considered and excluded, with rationale
recorded in the README: Bogacz (mean-field/oscillator approach), Rubchinsky
(thin post-2016 full-network output), Humphries (recency rule), McIntyre
(biophysical/axonal), whole-brain/TVB groups (by design — the README also
records the resulting BOLD-has-no-reviewer limitation, and that Meier et
al. 2022's TVB co-simulation of the Hamker-lineage BG model is the nearest
published precedent for our BOLD pipeline, for the synthesis to note).
Candidates were verified against web searches (lineage activity and
in-window output), not memory; no full texts read yet. Next: step 2, the
DOI reading list.

**Update 2026-08-13 (step 1 amended: seat 7, the experimentalist):** the
panel as confirmed was all modeling lineages, so the review would only
catch what the *modeling* community already models — experimental findings
the modeling literature has not yet absorbed (most prominently the GPe
reorganization of roughly 2015–2024: arkypallidal/prototypic cell types,
pallido-striatal projections, bridging collaterals — directly relevant
since v07 already contains `gpe_arky`/`gpe_cp`) would be invisible. A
seventh seat is added: **the experimentalist**, a *review-defined
composite* persona — a deliberate, recorded exception to the
one-seat-one-lineage rule, since no single experimental lab covers
whole-BG structure, connectivity and organization; its voice is what the
selected reviews collectively assert, and it has no single lab's published
standard to be judged against. Mandate: structure, connectivity and
organization, **plus** auditing the model's empirical validation anchors
(the Liang et al. 2008 medication-off firing-rate bands, the
Borra-tracer-based cortical proportions) — experimental claims no modeling
seat audits; the DBS representation stays with the Grill/Rubin seats.
Reading list (extends step 2): 4–6 reviews — above the 2–4 per-seat norm
because the seat covers a literature, not one lab's output — same 2016+
window, preferring the most recent authoritative synthesis per topic, with
two mandatory slots: at least one dedicated GPe review and at least one
whole-BG circuit-organization review; the remaining slots are chosen at
step 2 against what the model actually contains. Reviews-first with a
narrow escape hatch: a primary paper may take a slot only where step 2
finds no in-window review covering a mandated topic, the substitution and
the failed search recorded in the README. Two step-3 consequences: (a) the
synthesis convergence rule is amended for this seat — a point raised only
by the experimentalist is weighted by convergence *within* its reading
list (asserted by multiple independent reviews → literature consensus,
comparable to multi-persona convergence; a single review → one voice), and
the synthesis states which case applies; (b) every structural claim in the
experimentalist's review carries a species-provenance tag (mouse / rat /
macaque / human) beside the goal-relevance tag, because the recent GPe
literature is overwhelmingly mouse while the model is a human-subject fit
anchored on macaque tracer data. Seat definition written into
`community_review/README.md` and confirmed by Oliver on 2026-08-13.

**Update 2026-08-13 (step 2 done):** the DOI reading list is compiled,
verified and written into `community_review/README.md` — 25 papers: 19
across the six lineage seats (Kumar–Hellgren Kotaleski 3, Rubin/Verstynen
4, Girard–Doya 3, Grill 3, Hamker 4, Chakravarthy 2; the four-paper seats
are justified in the README by a documented shift in the lineage's
approach) plus 6 reviews for the experimentalist seat. Both mandatory
seat-7 slots were filled by in-window reviews (GPe: Courtney/Pamukcu/Chan
2023 Nat Neurosci; whole-BG organization: McGregor & Nelson 2019 Neuron),
so the primary-paper escape hatch was not needed. Every DOI was verified
against Crossref and/or the publisher page on 2026-08-13; per-seat
selections were drawn from PubMed author listings, not memory — which
caught one wrong provisional pointer from step 1 (seat 6's 2016 Frontiers
DBS paper is Mandali & Chakravarthy, not Muralidharan et al.; corrected in
the README, the historical step-1 text above left as written). Notable
selection decisions, with rationale in the README: Meier et al. 2022
(virtual DBS, Exp Neurol) sits in seat 5's list as a Hamker-lineage paper,
partially mitigating the BOLD-has-no-reviewer limitation from inside the
panel; Giossi et al. 2024 (EJN GPe review) was rejected for seat 7 because
it is authored by the seat-2 modeling lineage, whose unfiltered
experimental counterpart seat 7 exists to provide. The list is handed to
Oliver for download (checkpoint); step 3 starts only once the PDFs are
saved under `community_review/` and reads only those full texts.

**Update 2026-08-13 (download checkpoint passed; list amended by Oliver):**
the PDFs are saved under `community_review/` (untracked) and the set was
verified complete and valid against the list — 27 PDFs: 26 papers plus the
Bahuguna 2025 correction. At the checkpoint Oliver amended the reading
list in three places, all recorded with rationale in the README: (a)
**Hjorth et al. 2020** (PNAS, `10.1073/pnas.2000671117`, verified against
Crossref) added to seat 1 — the lineage's own striatal-microcircuit
standard, relevant to v07's `Microcircuit` striatum although
multi-compartment work sits outside the seat-selection similarity
criterion; (b) **Giossi et al. 2024** added to seat 2 — the seat-7
rejection (modeler-authored) stands, but its GPe findings should be taken
into account, and in seat 2 they inform the persona that authored them;
(c) **Meier et al. 2022** removed from seat 5 — TVB-based BOLD is a
different approach from ours, its DBS implementation is simpler than ours,
and its BG model is the same as Maith et al. 2021 (already listed).
Consequence of (c), recorded in the README: the partial in-panel audit of
the simulated-BOLD/DBS side is gone, so the BOLD-has-no-reviewer
limitation now holds without mitigation; Meier et al. 2022 stays noted as
the nearest published precedent for the synthesis to cite. Seat totals are
now 4/5/3/3/3/2 + 6 = 26. Next: step 3, the reviews, from these full
texts only.

**Resolved 2026-08-13 20:54 (step 3 done; the review documents exist,
which is what this entry resolves on):** seven reviews plus a synthesis are
committed under `community_review/`, each written from the full texts of
the saved PDFs — 26 papers across the seven seats, read page by page, no
abstracts. Point counts: seat 1 eleven, seat 2 ten, seat 3 eight, seat 4
nine, seat 5 eleven, seat 6 six, seat 7 sixteen. Every point carries a
goal-relevance tag, and seat 7's structural claims additionally carry a
species-provenance tag and a statement of within-reading-list convergence,
as its seat definition requires. `synthesis.md` merges them into 24
findings **F1–F24**, ranked in four tiers by how many seats raised each,
with one concrete change proposal per finding.

The four Tier-1 findings — five or more seats each — all bear on whether
the project's central claim can be made at all: (F1) nothing separates a
fitted parameter change from optimiser noise, reached independently by six
seats as degeneracy, as identifiability, as a regression against Maith et
al. 2021's twenty runs and 59-parameter sensitivity analysis, and as
sweep-versus-fit; (F2) nothing validates the model except the loss it is
fitted to, six seats, each naming a different held-out statistic; (F3)
input correlation is zero nearly everywhere while
`experimental_data/input_streams/README.md` §3 already computes that it
dominates the observable — the one hard blocker on §32, and §25 already
carries it; (F4) each of the three DBS parameters has a distinct problem,
with both DBS-standing seats leading.

Two facts the review established that the project did not know, both now
recorded in `community_review/README.md`. The GPe BOLD pooling factors
(0.5 / 0.17 / 0.10), which `model_v07.md` §3.6 records as having no source,
are GPe cell-type abundances — PV⁺ 50 %, arkypallidal 18 %,
cortex-projecting 12 %, the missing 0.23 being cell types the model does
not contain (seat 7, from Courtney et al. 2023) — confirming that
document's own conjecture. And the seven subcortical delays in
`parameters.csv` are Kumaravelu et al. 2016 Table 1 exactly, i.e. rat
values with real provenance the project does not record (seat 4). Two
further findings came from opening the data directory rather than the
papers: **three** subjects have BOLD and VTA data, not one (bearing on
F1), and `experimental_data/berlin_data/vta/` resolves the VTA against all
three STN functional subdivisions while `get_loss.py` uses only the motor
row — with a unit inconsistency between the overlap and volume files that
should be settled before the 0.4 is relied on further.

The BOLD-pipeline-has-no-peer-reviewer limitation stands as recorded
above, without mitigation; `synthesis.md` states it at the top and F11 is
the only finding that reaches it from the model side.

Per this entry's own terms, implementing anything is **not** part of §34.
The triage — the Roadmap item that follows — decides which findings are
accepted, and each accepted one spawns its own numbered entry. §34 closes
here because the review documents exist.

**Update 2026-08-15 08:25 (step 3 rerun as round 2; round 1 archived):**
Oliver had step 3 redone: round 1's reviews held BGM_22 against the seats'
own modeling practice without asking what that practice rests on, and their
phrasing left the BGM_22-versus-lineage contrast implicit. The rerun imposed
two requirements on every seat. (1) **Evidence basis**: every appeal to a
lineage's own practice states the experimental evidence behind the
lineage's own choice — source, species, preparation, what was measured, as
documented in its papers — or admits the choice is a convention, estimate
or tuned value, in which case the point is a difference, not a deficiency
of BGM_22, and may carry no change proposal; change proposals require
concretely cited experimental findings or a self-standing methodological
argument. (2) **Explicit contrast**: every point is structured as labelled
parts (what BGM_22 does / what we do / evidence behind our choice / why
that would or would not be better for this project's goal), ending in
exactly one verdict — experimentally grounded deficiency / methodological
deficiency / difference, not deficiency — plus the goal-relevance tag;
seat 7 keeps its species tags and within-list convergence statements. The
2026-08-13 documents moved to `community_review/round1/` (pure renames;
living-document path references updated in the same commit), and the
round-2 documents were written into `community_review/round2/` by seven
parallel sessions, each reading only its own seat's PDFs in full plus the
repository — no round-1 document, no other seat's file, and not the shared
README — with the synthesis then written from the seven finished reviews.
Result: 98 points (13/13/12/14/16/14/16 per seat; 22 experimentally
grounded, 33 methodological, 43 differences, 19 of those explicitly in
BGM_22's favour), merged into `community_review/round2/synthesis.md` as
**32 findings F1–F32** in four convergence tiers with evidence classes, a
section recording the thirteen proposals the evidence rule caused seats to
withdraw, and a triage reading guide. Round-2 finding numbers are local to
that document; round 1 is superseded, its numbering included. The
unanimous Tier-1 finding — all seven seats — is that the six non-striatal
firing-rate bands are unsourced and state-blind ("[Li et al., 2015]"
resolves to nothing) while gating every evaluation. Facts round 2
established beyond round 1's, recorded in the README: all 28 v07
projection weights are Goenner et al. 2021 Tables 4–5 (verbatim, or ×C on
striatal targets), self-described there as "determined mainly by
functional constraints" — verified independently by seats 2 and 5; and the
striatal connectivity kernel fit pools Taverna 2008's 6-OHDA/reserpine
rows and Gittis 2011's 6-OHDA rows with baseline rows into one state-less
kernel (seat 1, from the condition-labelled spreadsheet and
`connectivity_fit.py`). The resolution above stands — the review documents
exist; the triage (Roadmap phase 1 item 1) now runs on the round-2
synthesis.
