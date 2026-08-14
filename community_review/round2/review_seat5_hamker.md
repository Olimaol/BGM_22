# Seat 5 — Hamker (TU Chemnitz), our own lineage

Written in character as the Hamker lab (TU Chemnitz), applying our own published
standards to our own successor project: spiking basal ganglia networks in
ANNarchy fitted to human imaging data. BGM_22's owner is the first author of
Maith et al. 2021, so this review holds the project to a standard its author
himself put in print. This is round 2 of the `TODO.md` §34 survey (step 3 rerun
2026-08-14; the panel definition and reading list are in `../README.md`). It was
written independently of the round-1 review, which was not consulted; where
`model_v07.md` cites round-1 documents as provenance for a fact, that fact is
treated as project documentation. Round-2 requirement one: every appeal to our
own practice states the experimental evidence our published choice rests on — or
admits explicitly that it was a convention, estimate, or unjustified decision.
Round-2 requirement two: every point is structured as an explicit contrast
(what BGM_22 does / what we do / evidence behind our choice / why that would or
would not be better here) ending in exactly one verdict and a goal-relevance
tag.

**Read for this review (full texts, from the PDFs in this directory's parent):**

- Schroll H, Hamker FH (2016). Basal Ganglia dysfunctions in movement disorders:
  What can be learned from computational simulations. *Movement Disorders*
  31(11):1591–1601. DOI 10.1002/mds.26719
- Maith O, Villagrasa Escudero F, Dinkelbach HÜ, Baladron J, Horn A, Irmen F,
  Kühn AA, Hamker FH (2021). A computational model-based analysis of basal
  ganglia pathway changes in Parkinson's disease inferred from resting-state
  fMRI. *European Journal of Neuroscience* 53(7):2278–2295. DOI 10.1111/ejn.14868
- Goenner L, Maith O, Koulouri I, Baladron J, Hamker FH (2021). A spiking model
  of basal ganglia dynamics in stopping behavior supported by arkypallidal
  neurons. *European Journal of Neuroscience* 53(7):2296–2321. DOI
  10.1111/ejn.15082

**What was reviewed:** `model_v07.md` (in full), `model_v08.md`, `DBS.md`,
`experimental_data/input_streams/README.md`,
`experimental_data/activity_striatum/README.md`,
`experimental_data/cortical_proportions/README.md`,
`BOLD_optimization/get_loss.py`, `BOLD_optimization/parameters.py`,
`BOLD_optimization/deap_cma_opt.py` (search-space and run-tag machinery),
`CompNeuroPy/src/CompNeuroPy/full_models/bgm_22/parameters.csv` (columns
`BGM_v07_p01`/`BGM_v08_p01` against the older columns), `TODO.md` (Roadmap and
open entries; §34 skipped per the survey rules), and — because the project's
own documents nowhere name the hemodynamic model in force — the installed
ANNarchy extension (`ANNarchy_compneuro/ANNarchy/extensions/bold/BoldMonitor.py`,
`PredefinedModels.py`, `AccProjection.py`).

## The standard this seat applies

These are the recurring demands of our own papers, each with its evidence basis
stated — or admitted as house convention.

1. **Assumptions must be explicit, distinguishable from evidence, and
   accessible.** Schroll & Hamker 2016 (Limitations): "The validity of
   neuro-computational simulations depends on their underlying assumptions.
   Most of these assumptions can today be based on anatomical or physiological
   evidence. Some assumptions, however, cannot… one should be aware of their
   underlying assumptions." And (Final Remarks): "it will be up to us
   theoreticians to make our assumptions and results more easily accessible."
   This is a stated evaluative position, not an experimental result.
2. **Fixed neuron parameters carry a citation trail ending in measurements.**
   Maith 2021 §2.2: striatal parameters from Humphries, Wood & Gurney 2009,
   which replicated Moyer, Wolf & Finkel 2007, which matched in vitro
   whole-cell recordings of rat nucleus accumbens MSNs (Wolf et al. 2005);
   GPi/GPe/STN from Thibeault & Srinivasa 2013, recreating Humphries et
   al. 2006, matched to rat in vivo single-cell recordings. We admit the trail
   ends in rat data used for a human model without comment in either paper, and
   that the cortical 4:1 excitatory:inhibitory ratio in Maith §2.2 is asserted
   ("following the observed proportion") with no citation at all — our own
   application of this standard was imperfect.
3. **Where no published parameters exist, fit the neuron model to published
   physiology and show the fit.** Goenner 2021 §2.2 and Figure 2: GPe-Arky and
   GPe-Proto parameters obtained by scipy.optimize RMSE fits of step-current
   responses to Abdi et al. 2015 / Bogacz et al. 2016 (rat GPe), with the added
   5 ms refractory period documented; GPe-Cp explicitly *assumed* identical to
   GPe-Proto on shared Npas1 expression (Abecassis et al. 2020).
4. **Included projections carry projection-level evidence tables.** Goenner
   2021 Tables 1–2 list, per GPe projection, the studies supporting it;
   assumptions are labelled as assumptions in the text (e.g. the stop-related
   cortical input to GPe-Arky/GPe-Cp, §2.1: "we assume"). The synaptic
   *weights*, by contrast, were admitted to be "rather abstract and were
   determined mainly by functional constraints" (Goenner §4.4) — a point that
   matters repeatedly below.
5. **Multi-start optimization with selection, and inference only with a
   statistical criterion.** Maith 2021 §2.4: 20 BADS optimization processes per
   fitted dataset, lowest loss selected; §3.2: fitted differences interpreted
   only across groups of models (two-tailed t-tests, FDR, Cohen's d), plus a CV
   analysis of heterogeneity (§3.2, Tables 7–10). These are methodological
   devices with no experimental content; they exist to make a fitted difference
   interpretable, which is exactly what this seat's instructions say cannot be
   dropped without replacement.
6. **Fitted models are checked for discriminability and for plausible emergent
   activity, and implausible outcomes are flagged.** Maith 2021 §3.1/Table 4
   (control models fit the control mean FC significantly better than the
   patient mean FC, and vice versa); §2.5/§4.1 (rest-state firing rates
   compared post hoc against animal recordings); §4.3 (the fitted iSN increase
   is "likely an overestimation", with the recommendation to "add additional
   constraints to the model or add more data to the optimization procedure").
   Goenner 2021 §3.1 (z-scored population responses compared against Mallet et
   al. 2016 rat recordings).
7. **Sensitivity of conclusions to fixed parameters is quantified.** Maith 2021
   §2.6: all 59 predefined non-zero parameters varied ±5% on the fitted
   models, loss slopes reported (Figures 6–7) — with the admitted limitation
   that models were not refitted because that would have taken a month.
8. **The model's claim scope is bounded in print.** Maith 2021 §4.2 ("we do not
   want to draw any direct conclusions about the symptoms"), §4.3 (data
   caveats, including electrode artifacts in STN BOLD).

Finally, the frame: Maith 2021 §4.3 states that "the method provides individual
models of patients, a potential that has not been exploited in our present
study… comparable to the 'virtual epileptic patient'". BGM_22 is that
exploitation. We therefore review it as the continuation our own paper called
for, held to the standards that call implies.

## Points

### 5.1 The "literature weights" are our hand-tuned functional values, and our paper says so

**What BGM_22 does:** The 28 core projections
(`model_v07.md` §5; `parameters.csv` column `BGM_v07_p01`) use
`connect_fixed_number_pre(number=10)` with fixed scalar weights, which
`get_loss.py` (`PROJ_CLUSTERS_COMMON`, `_set_cluster_weights`) scales by one
fitted factor per functional cluster, "preserv[ing] their relative balance"
(comment above `PROJ_CLUSTERS_COMMON`; `CLAUDE.md` calls them "the literature
weights"). We verified the provenance directly: every BG-targeting weight in the
v07 column equals Goenner 2021 Table 4 verbatim (str_d1→snr 0.06, str_d2→gpe_proto
0.04, str_d2→gpe_arky 0.08, stn→snr 0.04, stn→gpe 0.001, gpe_proto→stn 0.001,
gpe_proto→snr 0.015, gpe laterals 0.025/0.008, snr→thal 0.06 …), and every
striatum-targeting weight equals the Goenner value multiplied by the
postsynaptic capacitance (gpe_arky→str_d2 6 = 0.12·50; gpe_proto→str_fsi 1.6 =
0.02·80; thal→str_d1 7 = 0.14·50; all seven checked) — the rescaling that
converts Goenner's `dV/dt` conductances into the Humphries models'
`C·dv/dt` convention. The 10-to-1 in-degree is Goenner's "Ten-to-One pattern"
(§2.2).

**What we do:** Goenner 2021 published exactly these values (Tables 4–5) as
hand-tuned constants of a stop-signal-task model, chosen "such that a Stop cue
with an SSD of 250 ms leads to 70%–80% of Stop trials to correct stopping"
(§3).

**Evidence behind our choice:** Explicitly none, for the magnitudes: Goenner
§4.4 states "the weight strengths between model nuclei, the cortical inputs to
the model, and the baseline inputs are rather abstract and were determined
mainly by functional constraints." A few *ratios* have experimental anchors we
cited: GPe-Arky→StrD2 stronger than →StrD1 (Glajch et al. 2016, per Goenner
§2.1 and Table 1), preserved in v07 as 6 vs 3.25 (same 1.85 ratio). Most
within-cluster ratios (e.g. gpe_cp→striatum at 0.01 vs gpe_arky→striatum at
0.065–0.12) were pure tuning.

**Why that would (or would not) be better here:** Scaling clusters instead of
freeing 28 weights is a sound identifiability decision — our own Maith 2021
fitted 38 free connectivity parameters to 15 FC entries and could not have
claimed identifiability without group statistics. But the preserved
within-cluster balance is not "literature": it is the balance we tuned so a rat
stop-signal network stops correctly, now silently imposed as a prior on a human
resting-state DBS inference — and imposed under different synaptic kinetics
than it was tuned for (see 5.3). Nothing in `model_v07.md` §5 or
`parameters.csv` records this provenance (the delays' provenance is recorded;
the weights' is not). A change proposal is licensed on methodological grounds:
document the weights as Goenner-2021 functional values; identify which
within-cluster ratios carry experimental support (via Goenner Table 1) and
which are tuning; and, once a fit exists, check the conclusions' sensitivity to
the unsupported ratios before interpreting any cluster scaling. Verdict:
**methodological deficiency**.

**Goal relevance:** high — the fitted cluster scalings are among the parameters
the DBS inference will read out, and their meaning depends on what the
preserved ratios encode.

### 5.2 The GPe neuron refit is recorded in one CSV cell, where we published a figure

**What BGM_22 does:** The v07 gpe_proto/gpe_arky/gpe_cp parameters
(`model_v07.md` §6.1) differ from Goenner 2021 Table 3, and the neuron model
adds a fitted power-law input compression (`nonlin ≈ 1.24`,
`Izhikevich2003NoisyBaseNonlin` with `use_nonlin=True`). The only provenance is
the `parameters.csv` section-header cell "refitted, data from Bogacz et
al. 2016" (GPe-Proto and GPe-Arky rows), with GPe-Cp marked "just like gpe
proto". No document states the fit protocol, the fit quality, or why the
`nonlin` term was introduced.

**What we do:** Goenner 2021 §2.2 documents the same kind of refit end to end:
scipy.optimize minimizing RMSE between simulated and published step-current
responses (Abdi et al. 2015; Bogacz et al. 2016), the added 5 ms refractory
period and its reason (concave f–I), and Figure 2 showing model against
experimental data. GPe-Cp = GPe-Proto is declared as an assumption grounded in
shared Npas1 expression (Abecassis et al. 2020).

**Evidence behind our choice:** Abdi et al. 2015 / Bogacz et al. 2016 are rat
GPe electrophysiology — the citation trail ends in measurements, and the
"GPe-Cp like GPe-Proto" step is an admitted assumption, both stated in the
paper.

**Why that would (or would not) be better here:** The v07 refit apparently
follows our own practice in substance (same data family), and inheriting the
GPe-Cp assumption is consistent with what we published. What is missing is the
demonstration: without the fit record, a reader cannot tell whether the new
parameter set plus `nonlin` reproduces the Abdi/Bogacz responses better or
worse than our published set, and `model_v07.md` §6.1 documents the equations
without their provenance. Verdict: **methodological deficiency** — the change
proposal is documentation: record the fit procedure, data targets, and
resulting f–I comparison (our Figure 2 is the template), and the motivation for
the `nonlin` term, in `model_v07.md` §6.1 or a dedicated
`experimental_data/`-style note.

**Goal relevance:** medium — the parameters plausibly are fine; the inability
to verify them is the problem.

### 5.3 Synaptic time constants and E_GABA depart from both of our published sets, unrecorded

**What BGM_22 does:** The six v07/v08 BG populations use `tau_ampa = 2 ms`,
`tau_gaba = 10 ms`, `E_gaba = −70 mV` (`parameters.csv` rows `stn.tau_ampa` …
`thal.E_gaba`, v07/v08 columns; `model_v07.md` §6.1). The *older columns of the
same file* — which still carry the cortex-Go/Stop/Pause and Integrator sections
of our stopping model — hold 10/20/−90 throughout.

**What we do:** Goenner 2021 §2.2: τ_AMPA = 10 ms, τ_GABA = 20 ms, E_AMPA = 0,
E_GABA = −90 mV at all projections. Maith 2021 §2.2/Eq. 1: τ_AMPA = τ_GABA =
10 ms, E_GABA = −90 mV.

**Evidence behind our choice:** Admitted: neither paper cites experimental
sources for these constants; they are conventions we stated in print but did
not justify. The v07 values are therefore not "wrong against evidence we hold"
— but they are a third convention, recorded nowhere, and they interact with
5.1: the Goenner weights being reused were tuned under 10/20/−90, and a
five-times-faster AMPA decay, halved GABA decay and a 20 mV weaker GABA driving
force change what every one of those weights does. `model_v07.md` §6.1 also
notes that with `stabilize=True` the GABA term is no longer rectified, so at
E_gaba = −70 the term turns depolarizing below −70 mV — closer to rest than
with our −90.

**Why that would (or would not) be better here:** Faster AMPA kinetics may well
be defensible (AMPAR decay of ~2 ms is nearer to physiology than our 10 ms),
but our own standard (Schroll & Hamker 2016, Limitations; standard item 1) is
that the choice and its reason be findable. Today the docs mark these values as
CSV overrides (†) without a source, and no living document records that they
depart from both predecessors. Since the fitted cluster scalings can absorb
static gain but not kinetics, the departure is not neutral for the inherited
balance. Verdict: **difference, not deficiency** — no change proposal; we ask
that the departure and its rationale be documented next to the values, and that
5.1's sensitivity check include it.

**Goal relevance:** medium — kinetics shape the network's transfer of the
cortical drive into the slow fluctuations the loss scores.

### 5.4 The fitting target is a genuine advance over our own published method

**What BGM_22 does:** The loss (`get_loss.compute_bold_correlation_loss`) is
the mean per-region Pearson correlation between simulated and experimental
BOLD *time courses* — 7 regions × 309 TRs of this subject — made meaningful by
driving the model with the subject's own deconvolved cortical BOLD
(`cortical_drive_by_bold_run.py` products; `model_v07.md` §7.5). The base
parameter vector is 19 (v07).

**What we do:** Maith 2021 §2.4 fitted the Frobenius norm of the difference
between 6×6 FC matrices — 15 independent correlations — with 38 free
connectivity parameters, and a synthetic cortex (600+150 neurons, fitted
cortical connections, noise devices `I = 50`, `M_SN = uniform(−5,5)`, Table 3)
whose drive had no relation to the subject's actual cortical activity.

**Evidence behind our choice:** Our own admission carries this point: Maith
2021 §4.3 concedes the free parameters were "tuned to match BOLD correlation
data… no direct access to firing rates or local field potentials exists that
could further constrain the model", and §2.4's initialization/bounds were
unmotivated ranges. The FC target was a device forced on us by having no
subject-specific input; it is why identifiability had to be rescued at group
level.

**Why that would (or would not) be better here:** Conditioning the model on
measured cortical activity and scoring the actual trajectory raises the
constraint per parameter by orders of magnitude (≈2100 data points against 19
parameters, versus 15 against 38) and removes our free cortical populations
entirely. It also converts the fit into a within-subject on/off comparison,
which eliminates the confounds our §4.3 had to admit (two scanners, 3T vs 1.5T,
group age-matching). Two conventions ride along and should simply be named in
the docs: the SPM-HRF deconvolution assumes a canonical cortical HRF (the same
class of assumption as the balloon model on the output side), and the per-region
mean-5 Hz normalization (`model_v07.md` §7.5) discards between-region
differences in mean drive, leaving only fluctuation shape. Verdict:
**difference, not deficiency** — in BGM_22's favour; it exceeds our published
practice.

**Goal relevance:** high — this is the design choice that makes single-subject
DBS inference conceivable at all.

### 5.5 The firing-rate constraint implements our own recommendation — but its BG bands are unsourced and clash with the operating point the inherited network was validated at

**What BGM_22 does:** `get_loss.get_firing_rate_loss` scores 18 populations
against plausibility bands and gates the BOLD run (`firing_rate_gate`, 0.5,
uncalibrated per `TODO.md` §10). The striatal bands are derived and documented
to an exemplary standard (`experimental_data/activity_striatum/README.md`:
Liang et al. 2008 medication-off, the response-direction assumption quoted from
the source, SEM→SD conversion shown, rejected alternatives recorded). The BG
bands are another matter: the code comment sources stn and snr to "[Li et al.,
2015]" (not on this seat's reading list; unverified here), and gpe_proto
(75–85 Hz), gpe_arky (15–20), gpe_cp (75–85) and thal (15–30) carry no citation
at all.

**What we do:** Maith 2021 had no rate term in the loss — and §4.3 flagged the
consequence (the fitted iSN rate increase "is likely an overestimation") and
recommended "additional constraints… or more data in the optimization
procedure". BGM_22's rate term is that recommendation implemented, and we
record that in its favour. But our lab's own validated GPe operating point is
elsewhere: Goenner 2021 Figure 7 (read from the panels) shows baseline
GPe-Proto ≈ 40 Hz, GPe-Arky ≈ 10–12 Hz, STN ≈ 15 Hz, thalamus ≈ 10 Hz in the
model whose responses matched Mallet et al. 2016's rat recordings — the same
network, neuron fits and weights v07 inherits.

**Evidence behind our choice:** Goenner's rates are anchored in rat data (Abdi
2015 / Bogacz 2016 f–I fits; Mallet 2016 in vivo comparisons, §3.1). Whether
75–85 Hz is right for human/primate prototypic GPe we cannot judge from our
read sources; if the bands are human intraoperative or primate values, no
document says so.

**Why that would (or would not) be better here:** Four of the six BG bands
(stn, thal, gpe_proto, gpe_arky) exclude or sit at the edge of the operating
point at which the inherited weights were validated in our own paper. A fit
that satisfies the gate must therefore push the network far from the regime the
Goenner balance was tuned for — while the cluster design of 5.1 tries to
preserve that balance. That tension is nowhere acknowledged, and the bands that
create it are uncited. The gate threshold calibration is already planned
(`TODO.md` §10, correctly ordered after the bounds work); band *provenance* is
not part of any entry. Change proposal, on internal-consistency grounds: source
every BG band in the code comment or a data README (the striatal READMEs are
the house template), state the species/state frame, and reconcile — either the
bands move to the frame the network is parameterized in, or the docs record why
the human resting frame overrides it. Verdict: **methodological deficiency**.

**Goal relevance:** high — the gate decides which individuals get a BOLD
evaluation at all, so unsourced bands steer the entire search.

### 5.6 Single-subject inference has no replacement for our multi-start-plus-statistics device

**What BGM_22 does:** `TODO.md` §32 plans one straightforward DBS-off fit, then
a staged DBS-on fit (off base frozen, putamen-only cluster scalings + 3 DBS
parameters); seeds are fixed at 42 during fitting, and the winning vector is
re-run across ~10 seeds "to report stability". §2 (the inference design)
correctly names near-degenerate pairs (`axon_spikes_per_pulse` vs the
`stn__gpe`/`stn__snr` scalings; `passing_fibres_strength` vs the `snr__thal`
scaling) and costs options — including a ~5-restart multi-start — but defers
the decision until after the first fits. `deap_cma_opt.py` supports run tags
(`--optimization-run`), so the infrastructure exists.

**What we do:** Maith 2021 §2.4: twenty optimization processes per fitted
dataset, best-of-20 selected; §3.2: a parameter difference was asserted only
when it survived a t-test across 30 control vs 40 patient models with FDR
correction, with effect sizes reported.

**Evidence behind our choice:** A methodological device, admitted as such —
multi-start against local minima of a non-convex loss, group statistics as the
criterion for "this parameter really differs". No experimental content. But
this is precisely the case flagged in our brief: the device exists to guarantee
that a fitted difference is interpretable, and dropping it without replacement
removes the warrant for the project's central claim.

**Why that would (or would not) be better here:** With n = 1 per condition
there is no group; the only available null distribution for "parameter X moved
because of DBS" is the scatter of that parameter across independent restarts of
the same fit. Seed-stability of a single winning vector measures simulator
stochasticity, not optimizer multi-modality — twenty restarts of Maith 2021
would not have been needed if one BADS run always found the same optimum. The
staged design is good (its generation 0 reproduces the off fit apart from
stimulation, `DBS.md`), but §2 currently costs the options in CPU-days rather
than by the statistical criterion they must satisfy. Change proposal: fix, as
part of §2 and before the on fit, (i) N restarts per condition (we used 20;
even 5 is informative), (ii) per-parameter restart scatter reported alongside
the fit, and (iii) the acceptance rule that an on-vs-off parameter move is
interpreted only if it exceeds that scatter. Verdict: **methodological
deficiency**.

**Goal relevance:** high — this is the warrant for "read off which parameters
had to change".

### 5.7 No analogue of our discriminability check (Table 4)

**What BGM_22 does:** Nothing in `get_loss.py`, `TODO.md` §2 or §32 evaluates a
fitted parameter set against the *other* condition's data. The planned free
control — the caudate loop keeps off weights, so its on-vs-off BOLD change must
be explained by its cortical drive alone (§2 update of 2026-08-11) — is a
genuine internal control our own papers lacked, and we credit it.

**What we do:** Maith 2021 §3.1/Table 4: after fitting, control models were
shown to fit the control-group mean FC significantly better than the
patient-group mean FC, and vice versa. That cross-check is what licensed
calling the two groups different models rather than different noise.

**Evidence behind our choice:** A methodological device; its logic — the fit
must discriminate the condition it was fitted to — is independent of group
size.

**Why that would (or would not) be better here:** The single-subject analogue
is cheap and well defined: evaluate the off-fitted parameters against the on
data (with the on drive and, separately, with DBS mechanisms active at zero
strengths) and the on-fitted parameters against the off data — a 2×2 of
evaluations per candidate fit. If the on-fit does not beat the off-parameters
on on-data by more than the restart scatter of 5.6, the refit captured noise,
and the "what DBS did" readout is empty. This also cleanly separates the two
things that change between conditions — the measured cortical drive (an input,
not an inference) and the fitted parameters — which the project's central claim
must not conflate. Change proposal: add the 2×2 cross-evaluation to §2's design
as an acceptance criterion. Verdict: **methodological deficiency**.

**Goal relevance:** high.

### 5.8 No planned sensitivity analysis over the fixed parameters

**What BGM_22 does:** Sensitivity information exists in fragments — the data
READMEs carry honest ranges for their own inputs (dSPN 22.4–25 Hz, FS 8–12 Hz,
putamen PMv 0.10–0.24), `TODO.md` §16 plans a sensitivity check of the DBS
constants, §25 plans the input-correlation scan. But the fixed neuron
parameters, synaptic constants, `N_cortical_inputs_dict`, and hemodynamic
parameters have no sensitivity plan, and no entry corresponds to Maith 2021
§2.6 as a whole.

**What we do:** Maith 2021 §2.6 varied all 59 predefined non-zero parameters
±5% on every fitted model and reported the loss slopes (Figures 6–7), finding
that the cortex and thalamus quadratic-equation parameters — the drive side —
dominated. We admitted the limitation that models were not refitted under the
variations (a month of compute for 70 models).

**Evidence behind our choice:** A methodological device, with an honest
shortcut we documented.

**Why that would (or would not) be better here:** With one subject there is no
group replication to absorb fixed-parameter error, so the case for the analysis
is *stronger* than in our paper, and our own headline result transfers as a
warning: the loss was most sensitive to the parameters shaping the drive, and
BGM_22's drive side is exactly where its boldest fixed choices sit (input
correlations pinned at 0, `N_cortical_inputs` underived per §23, proportions
per §21). The loss-only version we ran is cheap here too — one evaluation per
varied parameter at the fitted optimum. Change proposal: open an entry for a
Maith-§2.6-style loss-sensitivity pass over the fixed parameters once the first
fit exists, ordered with §4 and §16 in the post-fit block. Verdict:
**methodological deficiency**.

**Goal relevance:** medium — it does not block the fit, but it decides how much
of the fitted story survives scrutiny.

### 5.9 The neural→BOLD proxy silently flipped from our sign-blind synaptic activity to a signed net current

**What BGM_22 does:** The BOLD monitors map `I_CBF` to `I` (BG populations; the
raw current, not the `nonlin`-compressed one) or `I_v` (striatum) —
`get_loss.py` BOLD block, `model_v07.md` §3.6. Both are *net* synaptic
currents: the GABA term enters negatively
(`experimental_data/input_streams/README.md` §3 says so explicitly and builds
its amplitude argument on it). Strong inhibitory input therefore *reduces* the
BOLD drive.

**What we do:** Maith 2021 §2.3, Eq. 2: the BOLD input was a synaptic-activity
trace incremented by 1/n_aff per presynaptic action potential "no matter if
excitatory or inhibitory" — deliberately sign-blind — summed over populations
with an added noise term φ_R.

**Evidence behind our choice:** The motivation we cited (Logothetis et
al. 2001, anesthetized monkey V1; Mathiesen et al. 1998, 2000, rat cerebellum)
supports "synaptic activity, not spiking output, predicts the BOLD signal"; it
does not adjudicate signed versus unsigned summation. Our sign-blind increment
was a choice, but a stated one, aligned with the metabolic reading of those
sources (inhibitory transmission also consumes energy).

**Why that would (or would not) be better here:** In v07 the choice is not
cosmetic: a striatal neuron receives on the order of 61 000–69 000 GABAergic
spikes/s from the compensation streams (`model_v07.md` §7.3), so under the
signed proxy the striatal BOLD reflects cortical drive *minus* a large
inhibitory term, where under our convention the two would add. The simulated
Cau/Put time courses — two of seven scored regions — can differ qualitatively
between the conventions, and the fitted drive weights will absorb the
difference invisibly. Neither `model_v07.md` nor any README records that this
deviates from the predecessor's published, explicitly-argued convention.
Change proposal: record the deviation and its rationale; before interpreting
striatal parameters, run one fit-or-evaluation comparison with a sign-blind
input variable (the absolute-value analogue of Eq. 2) to establish whether the
inference is robust to the proxy. Verdict: **methodological deficiency**.

**Goal relevance:** high for the Cau/Put regions and every striatal parameter.

### 5.10 The hemodynamic model in force is undocumented; our practice was a parameter table with sources

**What BGM_22 does:** `BoldMonitor(...)` is called without a `bold_model`
argument, so the ANNarchy default applies. No BGM_22 document names it — we had
to read the installed extension: the default is `balloon_RN`
(`ANNarchy/extensions/bold/PredefinedModels.py`), a Stephan et al. (2007)
revised-coefficient balloon model with κ = 1/1.54, γ = 1/2.46, τ = 0.98 s,
E₀ = 0.34, V₀ = 0.02, ν₀ = 40.3 s⁻¹, r₀ = 25 s⁻¹, TE = 40 ms.
`normalize_input=2000` makes each pooled input a *relative deviation from a
2000 ms baseline* accumulated after monitor start (verified in
`AccProjection.py`: `(lsum − baseline_mean)/(|baseline_mean| + 1e−7)`, with the
balloon receiving zero during the baseline window). `TODO.md` §8 already notes
the 2000 ms baseline vs 2310 ms ramp-up mismatch as an open cleanup.

**What we do:** Maith 2021 §2.3, Table 2: the Balloon model per Friston et
al. (2000) with each parameter tabulated and sourced (Friston et al. 2003), and
a 15 s initial simulation discarded before the 250 s scored window.

**Evidence behind our choice:** Friston's parameter estimates are conventions
of the human fMRI modeling literature, not our measurements — but we documented
which convention we used.

**Why that would (or would not) be better here:** Three consequences. First,
the loss is a temporal correlation at TR 2.31 s, and κ, γ, τ shape the
hemodynamic impulse response, so the (undocumented) model choice directly
shapes the score — more directly than it shaped our FC target. Second, the
field-strength-dependent parameters (ν₀, r₀, TE) are the extension's defaults
and are nowhere checked against the subject's acquisition. Third, the scored
window includes the onset: after ramp-up (1 TR) plus the 2000 ms zero-input
baseline, the balloon state variables relax over several further TRs, and
`compute_bold_correlation_loss` trims only 1 TR from the experimental side —
where we discarded 15 s. With 309 TRs the contamination is small but
systematic and identical in every evaluation. Change proposal: document the
BOLD model and parameters in `model_v07.md` §3.6 with sources (our Table 2 is
the template); check TE/ν₀/r₀ against the acquisition or state why they do not
matter for a correlation loss; resolve §8's baseline/ramp question; and either
exclude the transient TRs from the correlation or show they are negligible.
(The absence of our φ_R noise term is fine for a correlation target — our
uniform(0, 0.05) was itself unjustified in the paper — but note it while
documenting.) Verdict: **methodological deficiency**.

**Goal relevance:** medium-high — it sits between every simulated current and
every scored number.

### 5.11 Our own paper discounted STN BOLD in implanted patients; here it enters the loss at full weight

**What BGM_22 does:** STN is one of the seven regions in the mean correlation
(`bold_region_compartments` in `get_loss.py`), with no artifact caveat anywhere
in the living documents (grep over `TODO.md`, `model_v07.md`, `DBS.md`,
`CLAUDE.md`, `get_loss.py` finds none).

**What we do:** Maith 2021 §4.1: "the DBS electrodes implanted in the patients
despite being switched off probably caused artifacts in the BOLD signal,
especially in the STN signal, which reduces the validity of our results about
the STN." We said this about the same clinical population this subject comes
from, and we let it limit our own conclusions.

**Evidence behind our choice:** A stated concern about susceptibility artifacts
near the electrode, not a measurement we made — but it is our published
position on exactly this data situation, in both conditions (the electrode is
present whether stimulation is on or off).

**Why that would (or would not) be better here:** If the subject's STN series
is artifact-contaminated, the fit will bend STN-coupled parameters to chase
artifact — and those are precisely the DBS-adjacent parameters
(`ci` stn drive weight, `stn__gpe`/`stn__snr`/`gpe_proto__stn` scalings,
`axon_spikes_per_pulse`) that the inference most wants to read. The equal
1/7 weight of STN in the mean correlation makes this a first-order concern, not
a footnote. Change proposal: carry our own caveat into the project's documents;
once fits exist, report the fit and the inferred parameter moves with and
without the STN region in the loss, and treat disagreement as a red flag on any
STN-mediated conclusion. Verdict: **methodological deficiency** (grounded in
our published caveat; the artifact itself is unquantified here).

**Goal relevance:** high — STN is the stimulated nucleus; the inference's
centrepiece runs through it.

### 5.12 The GPe-Cp population is inherited, but the projection that motivated it cannot exist in this model

**What BGM_22 does:** v07 keeps the three-way GPe division (gpe_proto,
gpe_arky, gpe_cp, 100 neurons each) with gpe_cp projecting to striatum and
within GPe (`model_v07.md` §5), receiving fitted cortical drive
(`CorticalInputs`), and contributing to the GPe BOLD at weight 0.10 (Courtney
et al. 2023 abundances, documented via round-1 provenance in `model_v07.md`
§3.6). There is no GPe-Cp→cortex projection — there is no simulated cortex; the
drive is prerecorded.

**What we do:** Goenner 2021 included GPe-Cp *because of* the
cortico-pallido-cortical loop (§1, §2.1; evidence in Tables 1–2: Abecassis et
al. 2020, Chen et al. 2015, Saunders et al. 2015 for the GPe→cortex direction;
Karube et al. 2019, Naito & Kita 1994, Milardi et al. 2015, Smith & Wichmann
2015 for cortex→GPe), and its function in our model was the long-term
cancellation of cortical go-processes via that efferent — the paper's central
novel claim.

**Evidence behind our choice:** The anatomical projections are experimentally
supported (rodent tracing and physiology, as tabulated in Goenner Tables 1–2);
the stop-related routing of cortical input to GPe-Arky/GPe-Cp was an admitted
assumption (§2.1).

**Why that would (or would not) be better here:** Keeping the cell type is
anatomically right, its afferent cortical drive is supported by the same
evidence we tabulated, and the resting-state generalization of those inputs
(fitted weights instead of task pulses) is legitimate. But without its
cortical efferent, "gpe_cp" here is functionally a third GPe population
distinguished only by its afferent mix and fitted drive weight — its namesake
feature, and everything our paper showed it does, is structurally absent. No
document says so. The consequence for the goal: a fitted change in gpe_cp
parameters under DBS must not be narrated as a pallido-cortical pathway effect,
because that pathway is not in the model. Verdict: **difference, not
deficiency** — no change proposal; we ask that `model_v07.md` record the absent
efferent and the resulting claim boundary, alongside the open question §3.6
already carries about simulating equal-sized GPe populations while weighting
BOLD by realistic abundances.

**Goal relevance:** medium.

### 5.13 The striatal input calibration exceeds anything we have published, and its scope limit is honestly drawn

**What BGM_22 does:** The v07 striatum (`model_v07.md` §7) is a 1000-neuron 3D
lattice at measured density, del Rey et al. (2022) cell-type proportions,
distance-dependent connectivity from a fitted kernel, individually sampled
literature weight mixtures, a geometrically realised surround
(missing-GABA compensation) whose statistics are checked at build time against
closed forms, and a Kincaid-1998-derived shared cortical axon pool. The
contract document (`experimental_data/input_streams/README.md`) states the
target statistics, what the approach deliberately cannot represent (no
feedback, no decorrelation, no single-neuron temporal structure), and what the
model may and may not be used to claim ("this is not a striatal circuit
model"; legitimate inference: input gains, drive amplitudes; not legitimate:
emergent circuit dynamics).

**What we do:** Maith 2021 §2.2: two 200-neuron SPN populations, no FSI, no
space, stochastic connectivity with fitted scalar weights. Goenner 2021 adds a
100-neuron StrFSI with hand-set weights. Neither has input-statistics
calibration; our striatal detail was the minimum the task needed. One
consistency worth recording: v07's absence of SPN→FS projections
(`fitted_params.json` has no such pair) matches our own Goenner Table 4, which
also has no StrD1/D2→StrFSI row — the asymmetry is not a v07 invention, and
its consequences are documented in the input-streams README §4.1.

**Evidence behind our choice:** Admitted: our striatal simplifications were
never justified against data; they were affordable defaults. The successor's
calibration chain cites primary measurements (Kincaid et al. 1998 contact
fractions and axon counts; Liang et al. 2008 rates with the response-direction
assumption quoted from the source; the fitted kernel).

**Why that would (or would not) be better here:** For a BOLD fit the striatal
input statistics *are* the striatal model — the README's variance argument
(input correlation sets BOLD amplitude) makes that explicit — so investing the
detail exactly there is the right allocation, and the documentation discipline
(assumption chains, sensitivity ranges, rejected alternatives, claim
boundaries) meets the demand we made in Schroll & Hamker 2016's Final Remarks
better than our own papers did. Verdict: **difference, not deficiency** — in
BGM_22's favour.

**Goal relevance:** medium — it does not decide the inference, but it is the
part of the project we would show others as the new house standard.

### 5.14 Species and state mixing continues our own habit — with better bookkeeping, and one composite-level test still owed

**What BGM_22 does:** The assembled model combines rat GPe neuron fits (Bogacz
et al. 2016 per `parameters.csv`), rat subcortical delays (Kumaravelu et
al. 2016, per the round-1 provenance recorded in `model_v07.md` §5), macaque
tracer proportions (Borra et al. 2021/2022,
`experimental_data/cortical_proportions/README.md`, with each assumption
tagged M/A/J), MPTP-primate medication-off striatal rates (Liang et al. 2008),
mouse GPe abundances in the BOLD pooling (Courtney et al. 2023, with the
mouse-to-primate caveat recorded), and human single-subject BOLD.

**What we do:** Maith 2021 fitted rat-derived neuron models to human FC with no
species caveat anywhere in the paper — admitted. Goenner 2021 was at least
internally consistent (rat physiology, rat behavior, rat validation data).

**Evidence behind our choice:** None; using rat parameters for human BG was an
unexamined convention of ours, inherited from the availability of data.

**Why that would (or would not) be better here:** BGM_22 does what we did, but
documents each border crossing — which is the most that can be done short of
data that do not exist. The one thing the composite still owes is its first
end-to-end test: the Roadmap's phase-1 validation run pair (rates inside bands
or deviations explained, stream statistics matching the contract, a
demonstrated on-vs-off difference) is exactly that test, and its ordering
before any expensive cache or fit is correct. Verdict: **difference, not
deficiency** — with the explicit note that the validation pair should be
treated as the composite's acceptance test, not a formality.

**Goal relevance:** medium.

### 5.15 The fitted dataset's basic provenance — medication state, hemisphere handling — is not recorded

**What BGM_22 does:** The model asserts a medication-off subject
(`activity_striatum/README.md`: "This project simulates a patient without
dopaminergic medication"; φ₁ = φ₂ = 0, with `TODO.md` §30 tracking whether 0 is
the right dopamine-depleted value in the Humphries source convention), and the
Roadmap makes establishing "the subject's medication state… before any
full-length cache is built" a phase-1 blocker. But no reviewed document records
what the Berlin subject's medication state during scanning actually *was*, and
none records how left and right hemispheres enter the seven ROI series (the VTA
arithmetic in `get_loss.py`, `(35+23)/(70+75)`, suggests two electrodes are
being pooled).

**What we do:** Maith 2021 §2.1 recorded the acquisition state precisely —
patients scanned DBS-OFF *with* their usual medication, 5–15 min after
switch-off — and §4.1 used exactly that fact (Levy et al. 2001, apomorphine
lowering GPi/STN rates) to explain an otherwise puzzling fitted result. We also
fitted left and right hemispheres as separate models (§2.4).

**Evidence behind our choice:** The medication caveat is a documented property
of the dataset; its interpretive use in §4.1 shows why it must be on record
before fitting, not discovered after.

**Why that would (or would not) be better here:** If the subject was scanned on
medication — as our cohort was — then the Liang medication-off rates, the φ = 0
setting and every cache built on them target the wrong state; the Roadmap knows
this and orders it correctly, and §30's plan (read the source convention before
choosing φ) is adequate. What is missing is smaller and cheaper: a data README
for `experimental_data/berlin_data/` recording the subject's medication state,
the on/off session protocol, field strength/TE, and the hemisphere convention
of the ROI series, in the style the other data directories already follow.
Change proposal: write it before phase 2. Verdict: **methodological
deficiency** (a documentation gap on the project's own template; the underlying
scientific question is already correctly planned).

**Goal relevance:** high — a wrong medication assumption invalidates every
cache; the Roadmap says so itself.

### 5.16 v08 is reviewed as what it claims to be, and its claim is honest

**What BGM_22 does:** `model_v08.md` presents v08 as a cache-free smoke test of
the optimization pipeline — plain 100-neuron populations, one mixed-drive
`TimedArray` per loop, `exp_input` noise drive — and both it and `CLAUDE.md`
say plainly that it is kept only for that purpose, with its known-broken
optimization bounds recorded (`TODO.md` §1).

**What we do:** We have no published analogue of a maintained reduced pipeline
model; our practice was ad-hoc test scripts that never survived. A documented,
honest fallback with its limits stated is better practice than ours.

**Evidence behind our choice:** Not applicable; nothing in our papers bears on
it.

**Why that would (or would not) be better here:** The one risk — that v08
results leak into scientific claims — is fenced by its own documentation, and
§1's bounds work is correctly ordered before any real fit. Verdict:
**difference, not deficiency**.

**Goal relevance:** low.

## What we would ask for before publication

1. Before the DBS-on fit is interpreted: fix in `TODO.md` §2 the restart count
   per condition, the per-parameter restart-scatter report, and the rule that
   an on-vs-off parameter move is claimed only if it exceeds that scatter
   (5.6); add the 2×2 cross-condition evaluation (off/on parameters × off/on
   data) as an acceptance criterion (5.7).
2. Source every BG band in `get_firing_rate_loss` (species and state named),
   and reconcile the gpe/stn/thal bands with the operating point the inherited
   Goenner network was validated at — or document why the human resting frame
   overrides it (5.5).
3. Document the projection weights as Goenner 2021's functionally tuned values
   (C-rescaled), mark which within-cluster ratios have experimental anchors,
   and check conclusion sensitivity to the unanchored ratios after the first
   fit (5.1).
4. Document the BOLD chain end to end in `model_v07.md` §3.6: the `balloon_RN`
   model and parameters with sources (our Table 2 as template), the
   field-strength parameters against the acquisition, the resolution of the
   2000 ms baseline vs 2310 ms ramp-up question, handling of the onset
   transient in the scored window (5.10), and the deviation from Maith 2021's
   sign-blind synaptic-activity proxy, with one robustness comparison before
   striatal parameters are interpreted (5.9).
5. Carry Maith 2021 §4.1's electrode-artifact caveat into the project, and
   report the fits with and without the STN region in the loss before any
   STN-mediated conclusion (5.11).
6. Record the GPe neuron refit (protocol, targets, f–I comparison, motivation
   for `nonlin`) per Goenner Figure 2 (5.2).
7. After the first fit: a Maith-§2.6-style loss-sensitivity pass over the fixed
   parameters, ordered with §4 and §16 (5.8).
8. Write the `experimental_data/berlin_data/` README: the subject's medication
   state during scanning, session protocol, acquisition parameters, hemisphere
   convention of the ROI series — before phase 2 builds anything expensive
   (5.15).
9. Documentation-only notes licensed as differences: the synaptic-constant
   departure from both predecessors (5.3), the absent GPe-Cp cortical efferent
   and its claim boundary (5.12).
