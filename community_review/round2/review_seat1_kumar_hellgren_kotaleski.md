# Seat 1 — Kumar / Hellgren Kotaleski (KTH Stockholm)

Written in character as the Kumar–Hellgren Kotaleski lineage (KTH Stockholm /
Freiburg), applying the standards of our own published work: spiking
point-neuron models of the full basal ganglia with biologically constrained
connectivity, simulated in healthy and dopamine-depleted states and validated
against in-vivo firing statistics and oscillation phenomena. This is round 2 of
the `TODO.md` §34 survey (step 3 rerun 2026-08-14); the panel definition and
reading list are in `../README.md`. This review was written independently of the
round-1 review, which was not consulted. Per the round-2 requirements: every
appeal to our own practice states the experimental evidence that practice rests
on, or admits it is a convention; and every point is structured as an explicit
contrast — what BGM_22 does, what we do, the evidence behind our choice, whether
ours would be better *for this project's stated goal*, ending in one of three
verdicts, plus a goal-relevance tag.

**Read for this review (full texts, from the PDFs in this directory's parent):**

- Lindahl M, Hellgren Kotaleski J (2016). Untangling basal ganglia network
  dynamics and function: role of dopamine depletion and inhibition investigated
  in a spiking network model. *eNeuro* 3(6). DOI 10.1523/ENEURO.0156-16.2016.
- Bahuguna J, Sahasranamam A, Kumar A (2020). Uncoupling the roles of firing
  rates and spike bursts in shaping the STN-GPe beta band oscillations. *PLOS
  Comput Biol* 16(3):e1007748. DOI 10.1371/journal.pcbi.1007748. The 2025
  correction (*PLOS Comput Biol* 21(11):e1013638, DOI
  10.1371/journal.pcbi.1013638) adds a missing author affiliation only and
  changes nothing scientific.
- Chakravarty K, Roy S, Sinha A, Nambu A, Chiken S, Hellgren Kotaleski J, Kumar
  A (2022). Transient response of basal ganglia network in healthy and
  low-dopamine state. *eNeuro* 9(2). DOI 10.1523/ENEURO.0376-21.2022.
- Hjorth JJJ, Kozlov A, Carannante I, Frost Nylén J, Lindroos R, Johansson Y,
  Tokarska A, Dorst MC, Suryanarayana SM, Silberberg G, Hellgren Kotaleski J,
  Grillner S (2020). The microcircuits of striatum in silico. *PNAS*
  117(17):9554–9565. DOI 10.1073/pnas.2000671117.

**What was reviewed:** `model_v07.md` (in full), `model_v08.md`, `DBS.md`,
`experimental_data/input_streams/README.md`,
`experimental_data/activity_striatum/README.md`,
`experimental_data/cortical_proportions/README.md`,
`BOLD_optimization/get_loss.py`, `BOLD_optimization/parameters.py`, `TODO.md`
(Roadmap and open entries; §34 skipped per the round-2 protocol),
`CompNeuroPy/.../bgm_22/model_creation_functions.py` (the `BGM_v07` creation
function and docstring), `parameters.csv` (column `BGM_v07_p01`, extracted
programmatically),
`striatal_microcircuit_requirements/connectivity_parameters/connectivity_fit.py`
and `connectivity_fit_run.py`,
`experimental_data/Connectivity_intrinsic_striatum/connectivity_probabilities.ods`
(text-extracted, including its reference list),
`CompNeuroPy/.../striatal_microcircuit/get_weights.py`, and spot checks in
`microcircuit.py`.

---

## The standard this seat applies

These are the demands our own papers repeatedly make of ourselves. Each is
stated with the evidence it rests on, or with the admission that it is a house
convention.

1. **Every parameter value carries a stated source, or an explicit admission
   that none exists.** Lindahl & Hellgren Kotaleski 2016 print source columns in
   their synapse and dopamine tables (Tables 7–9) and mark unsupported entries
   "n.d."; Chakravarty et al. 2022 cite a source per row in Tables 1–10. This is
   a reporting convention, not itself an experimental result — but it is what
   makes a model auditable, and the underlying values rest on named experiments
   (paired recordings: Taverna et al. 2008, Planert et al. 2010, Gittis et al.
   2010/2011; stereology: Oorschot 1996; in-vivo units: Mallet et al. 2008/2012
   — all as cited in those tables).
2. **Population sizes and fan-ins from anatomy where anatomy exists.** Lindahl
   2016 (Methods) sizes populations by the relative proportions of Oorschot
   1996's stereological rat counts and derives MSN fan-ins from
   axonal/dendritic-overlap geometry times Taverna 2008's paired-recording
   probabilities; where only bouton counts exist (GPe TA→MSN, Mallet et al.
   2012) the interpolating assumptions are stated as assumptions in the text.
3. **Disease states are explicit parameter sets keyed to experiments.** The
   dopamine-depletion maps (Lindahl 2016 Table 9; Chakravarty 2022 Table 10)
   modify each affected parameter by a factor whose direction and rough
   magnitude come from a cited experiment (e.g. MSN–MSN IPSC and connectivity
   reduction from Taverna et al. 2008; FSI→D2-SPN connectivity increase from
   Gittis et al. 2011, as recited in Chakravarty's Methods). The α_dop scale
   itself (0.8 = normal, 0 = PD) is an admitted modeling convention.
4. **Validation against data the fit never saw, including second-order
   statistics.** Lindahl 2016 (Figs 2–3) validate rates *and* coefficients of
   variation *and* coherence *and* phase relations against Mallet et al. 2008's
   anesthetized-rat recordings, in two brain states and two dopamine states;
   Chakravarty 2022 validate transient-response zones (against Sano & Nambu
   2019, as cited there), oscillation index and Fano factor; Bahuguna et al.
   2020 validate beta-burst length (~0.2 s in healthy mice, their ref 62),
   burst peak frequency and the burst amplitude–length correlation (their refs
   20, 63).
5. **Robustness before conclusions.** Bahuguna 2020 rerun 10,000
   parameter-perturbed networks (Gaussian, SD 20 % of each mean; Methods and S2
   Fig) and explicitly call even that "a preliminary robustness analysis and
   [in] no way a comprehensive sensitivity analysis"; Lindahl 2016 perturb
   delays and fan-ins and report which validated features survive.
6. **Explicit nulls.** Bahuguna 2020 define their beta-burst threshold against
   an ensemble of rate-matched Poisson processes (Fig 6A caption) — network
   effects are claimed only above what structureless spiking of the same rate
   produces.
7. **Admitted simplifications of our own:** all three network papers drive the
   model with *uncorrelated Poisson input tuned to reproduce target rates*
   (Lindahl 2016 Methods and Table 1; Chakravarty 2022 Methods, "External
   inputs"; Bahuguna 2020 Methods), and Bahuguna 2020's limitations section
   names non-Poissonian input statistics as unexplored; Chakravarty 2022 use
   static synapses with an argument, list homogeneity, missing interneuron
   types and absent spatial structure as limitations; the SSBN neuron emits a
   fixed number of spikes per burst by construction. Where BGM_22 differs from
   us on such a point, that difference is not a deficiency of BGM_22.

---

## Points

### 1.1 The striatal connectivity kernel pools healthy and dopamine-depleted datasets into one state-less kernel

**What BGM_22 does:** The seven distance kernels `p(d) = P0·exp(−d²/σ²)` in
`fitted_params.json` — the entire intrinsic striatal connectivity of v07, and
via `E_outer = 4πρ∫p(r)r²dr` also the size of the missing-GABA compensation
streams (`model_v07.md` §7.2–7.3) — are fitted by
`connectivity_fit.py` (`__main__` `datasets` dict) as one maximum-likelihood fit
per pair over *pooled* condition rows. For dSPN→dSPN, iSPN→dSPN and iSPN→iSPN
the pooled rows include Taverna et al. 2008's 6-OHDA and reserpine rows (0/7
and 0/8 connected) next to its baseline rows (labels `[1] 6-OHDA`,
`[1] reserpine`, `[1] baseline`; the bracket numbers resolve to full citations
in `connectivity_probabilities.ods`, whose extraction is otherwise careful and
condition-labelled). For FS→dSPN and FS→iSPN the pool includes Gittis et al.
2011's 6-OHDA rows next to its baseline rows — for FS→iSPN that is 66/86
(77 %) depleted against 42/108 (39 %) baseline.

**What we do:** We parameterize connectivity in the healthy state and represent
dopamine depletion as explicit multiplicative factors on exactly these
quantities: Lindahl 2016 Table 9 reduces MSN–MSN connectivity and IPSC
amplitude at depletion, source Taverna et al. 2008; Chakravarty 2022 Methods
("Dopamine effects on synaptic weights") state that depletion "increases the
number of connections between FSI and D2-SPN (Gittis et al., 2011), but not
D1-SPN". The two states are never mixed into one likelihood.

**Evidence behind our choice:** The state difference is a measured result, not
a convention: Taverna et al. 2008 (rodent slice paired recordings, 6-OHDA and
reserpine models — the very rows in BGM_22's own spreadsheet) found SPN–SPN
recurrent connectivity collapses under depletion; Gittis et al. 2011 (mouse,
6-OHDA) found FSI→iSPN connectivity roughly doubles while FSI→dSPN does not
move. Our α_dop mapping of those effects is a convention; their existence and
direction are data. Caveat we state against ourselves: both are acute rodent
depletion models, and whether chronic human PD striatum shows the same
remodeling is not established by anything we have read — BGM_22's own
`activity_striatum/README.md` documents (via Liang et al. 2008) that chronic
primate depletion behaves differently from acute depletion for *rates*.

**Why that would (or would not) be better here:** BGM_22's subject is a
chronically dopamine-depleted patient off medication, and the project's own
striatal rate calibration deliberately chose that state (the Liang med-off
values, baked into every cache). The connectivity should be chosen by the same
principle, and pooling forecloses that: the pooled kernel corresponds to no
preparation that exists. Concretely, the dSPN→dSPN amplitude P0 = 0.108 sits
below even the baseline extraction (the `.ods` rows give 13.2 % and 21.4 %
one-way within 50 µm) because the two zero-connectivity depleted rows dilute
it, while FS→iSPN is *pulled up* by its depleted row — at short range the
fitted kernels reverse the FS target preference (FS→iSPN P0 = 0.918 >
FS→dSPN P0 = 0.599 at d = 0) relative to every healthy dataset in the pool
(Planert et al. 2010: 89 % vs 67 % within 100 µm, as used in Hjorth 2020;
Gittis et al. 2010: 27 % vs 18 %, as used in Lindahl 2016; Gittis et al. 2011
baseline: 60 % vs 39 %). So the kernel set encodes a depletion signature in one
pair and dilutes it in others — an incoherent mixture, whichever state is
intended. A second, anatomical symptom of the underconstrained per-pair fit:
FS→dSPN gets σ = 394 µm while FS→iSPN gets σ = 140 µm, yet both are served by
the same presynaptic FS axonal arbor, one physical object (in Hjorth 2020 this
consistency is automatic, because one reconstructed axon morphology serves
both targets and only the pruning differs); a 2.8× difference in spatial reach
between the two targets of the same axon has no anatomical reading and is more
plausibly fit noise amplified by the state mixture. Two mitigations to state
honestly: the comparison of P0 = 0.108 against "Taverna 26 %" that Hjorth 2020
uses would be unfair — the `.ods` correctly converts Taverna's two-way-tested
pairs to one-way probabilities (13 %), the same reading Lindahl 2016 used, so
our own two papers differ by 2× on the same source and BGM_22's extraction
agrees with one of us; and because both DBS conditions share the kernel, the
on-vs-off *contrast* is not directly biased — the damage is to the absolute
state the model claims to represent, to the missing-GABA stream sizes
(`E_outer` enters every cache), and to the §4 rate self-consistency.
**Experimentally grounded deficiency.** Proposal: refit the kernels per
condition from the already-condition-labelled `.ods` rows; adopt the depleted
kernel for this patient (or the healthy kernel with explicit, cited depletion
factors in the style of Lindahl 2016 Table 9), document the choice, and rebuild
the short caches — this belongs in phase 1 of the Roadmap, before any
full-length cache. The same state question applies to the IPSC-amplitude
mixtures in `get_weights.py` (Taverna 2008 also reports amplitude reduction
under depletion, as encoded in Lindahl 2016 Table 9).

**Goal relevance:** High — these kernels set both the simulated GABA circuit
and, through `E_outer`, the dominant (98 %) synthetic GABA input to every
striatal neuron, in every cache, in both DBS conditions.

### 1.2 The non-striatal firing-rate bands have no recorded provenance and no stated disease state

**What BGM_22 does:** `get_loss.get_firing_rate_loss` scores 18 populations
against `plausible_ranges`. The striatal bands are exemplary — derived,
sourced, and caveated in `experimental_data/activity_striatum/README.md`, with
the disease state chosen deliberately. The other six are not: `gpe_proto`
(75, 85), `gpe_arky` (15, 20), `gpe_cp` (75, 85) and `thal` (15, 30) carry no
citation anywhere in the repository (verified by grep), and `stn` (28, 80) and
`snr` (21, 93) carry only the bare comment "from [Li et al., 2015]" with no
resolvable reference. Nothing states which disease state (healthy or
parkinsonian) or species any of them describe. These bands feed the gate
(`firing_rate_gate` = 0.5) that can veto the BOLD run entirely, and the
project's own measurements show the uncited bands are what the gate bites on:
the resolved §9 tables in `TODO.md` record `gpe_cp` at 10–15 Hz against its
75–85 band and `gpe_arky` at 1.9–4.3 Hz against 15–20, dominating the measured
rate loss of ~0.87.

**What we do:** Every rate target is cited with species, preparation and state:
Chakravarty 2022 Methods ("External inputs") tune to D1/D2-SPN ∈ [0.01, 2.0] Hz
(Miller et al. 2008), FSI ∈ [10, 20] (Gage et al. 2010), STN ∈ [10, 13], SNr ∈
[20, 35], GPe-TA 11.8 ± 1.1 Hz and GPe-TI 24.2 ± 0.7 Hz (Mallet et al. 2008,
2012) for the normal state, and separately to PD-state ranges — GPe-TA
[12, 16], GPe-TI [17, 20], STN [26, 29] (de la Crompe et al. 2020, as cited
there) — with the SNr left unchanged after weighing conflicting reports, an
explicitly argued decision. Lindahl 2016 match GPe TA/TI and STN rates to
Mallet et al. 2008 per state and per cortical regime.

**Evidence behind our choice:** In-vivo extracellular recordings — Mallet et
al. 2008/2012 (urethane-anesthetized rats, control and 6-OHDA), de la Crompe et
al. 2020 (as cited in Chakravarty 2022) — which establish that GPe and STN
rates *differ between the healthy and dopamine-depleted state* (STN up, GPe-TI
down in the rat data above). Admissions against ourselves: these are rodent
numbers, largely under anesthesia, and their translation to an awake human
patient is an assumption we also make; BGM_22 fitting a human cannot simply
adopt them, and I cannot state the correct human values from my read sources.

**Why that would (or would not) be better here:** The deficiency claimed is not
that BGM_22's numbers are wrong — I cannot show that — but that they are
unauditable and state-blind in a pipeline where they act as a hard gate on a
parkinsonian patient's fit. If, for instance, 75–85 Hz for `gpe_proto` is a
healthy-primate value, the gate pushes both fits toward a healthy pallidal
regime and biases exactly the parameters the DBS inference will read; nobody
can check, because the provenance is absent. Two sharper sub-points: the two
GPe bands are also the *narrowest* (10 Hz wide, against 52 Hz for STN and 72 Hz
for SNr), so they are the binding constraint of the gate; and `gpe_cp` (the
cortex-projecting NPAS1 class) is given the same band as `gpe_proto` although,
to our knowledge, no in-vivo rate literature exists for that class as a
separate entity [from memory — unverified], which would make its band an
assumption that must be flagged as such. `TODO.md` §10 already covers the
DBS-condition-independence of the bands and correctly defers threshold
calibration to the mini-run; the disease-state axis and the missing provenance
are not covered there and are prior to it. **Experimentally grounded
deficiency** (the state-dependence of GPe/STN rates is a cited experimental
finding; the provenance gap is checkable fact). Proposal: write the missing
derivation document in the style of `activity_striatum/README.md` — per band:
source, species, preparation, disease state, and whether an on-condition
variant is needed — and resolve "Li et al., 2015" to a full citation, before
the gate is enabled in any fit; fold the result into the §10 calibration.

**Goal relevance:** High — the gate can silently steer both fits toward an
unspecified physiological state, which is upstream of every inference claim.

### 1.3 No null model for the BOLD-correlation loss

**What BGM_22 does:** The loss (`get_loss.compute_bold_correlation_loss`) is
1 minus the mean per-region Pearson correlation between simulated and
experimental BOLD across seven ROIs. The cortical drive is deconvolved from the
same subject's cortical BOLD in the same recording, so cortex–BG BOLD
correlations of non-neuronal origin (global fluctuations, motion, vascular
signals shared across ROIs) are inherited by the drive. Nothing in the
repository computes what score a structureless transformation would achieve —
for example, each ROI predicted by an HRF-convolved weighted mix of the seven
cortical series, with no basal ganglia in between. `TODO.md` §2 (the inference
design) discusses degeneracy between fitted parameters but not a null baseline
for the loss itself.

**What we do:** Where we quantify a network effect we first ask what no-network
produces: Bahuguna 2020 define the beta-burst detection threshold against an
ensemble of rate-matched Poisson processes (Fig 6A caption) so that only
structure beyond rate-matched noise counts as a burst.

**Evidence behind our choice:** This is a statistical-methodology practice, not
an experimental finding; it stands on its own — an effect size is
interpretable only against the score of a model with the mechanism of interest
removed.

**Why that would (or would not) be better here:** Directly transferable and
cheap. The danger is specific to this design: because drive and target come
from one recording, a high mean correlation may be achievable with *any*
smoothing dynamics, in which case the fitted parameters carry little
information and the DBS-on-vs-off parameter deltas even less. Conversely, if
the full model clearly beats the null, that number is the strongest possible
answer to the obvious referee question. The null costs no simulation — it is a
regression of the experimental ROI series on the (HRF-convolved) cortical
series — and should be reported per region alongside every fitted loss.
**Methodological deficiency.** Proposal: compute and record the null-model
per-region correlations (cortical-mix regression, and additionally the trivial
best-single-region predictor) before the first fit, and require the fitted
model to be interpreted only where it exceeds them.

**Goal relevance:** High — it decides whether the loss contains leverage for
parameter inference at all.

### 1.4 The inference lacks a stated criterion for "this parameter changed"

**What BGM_22 does:** The goal is to "read off which parameters had to change".
`TODO.md` §2 already recognizes that a single argmin will not support this,
names two near-degenerate parameter pairs, and costs three remedies
(single-mechanism scans, multi-start refits, L1-regularized refit) — a plan we
endorse. What no document states is the acceptance criterion: how large an
on-vs-off delta must be, relative to what spread, before it is reported as "what
DBS did". All evaluations also run at one fixed seed (`parameters.py: seed`
= 42, applied via `setup()` and `set_seed`), so the fitted optimum is
conditioned on a single noise realization of a stochastic model.

**What we do:** We report conclusions at the level that survives perturbation:
Bahuguna 2020 rerun 10,000 networks with all parameters drawn at SD 20 % of
their means and keep only findings that survive (Methods, S2–S3 Figs), while
explicitly calling this preliminary; Chakravarty 2022 vary each of six key
connections over seven values and report coefficients of variation of the
response features (Fig 6M–N) before naming D2-SPN→GPe-TI as *the* crucial
connection; conclusions are stated as regime-level, not point estimates.

**Evidence behind our choice:** Methodological, self-standing: a point estimate
from a stochastic, degenerate objective is not evidence of a parameter change
without a spread against which to judge it. Our own 10,000-network scale is
admittedly infeasible here at ~25–40 min per evaluation; the criterion, not the
scale, is what transfers.

**Why that would (or would not) be better here:** The multi-start option in §2
already produces the needed spread as a by-product. What is missing is the
commitment, made before the on-fit runs, that a parameter is reported as
changed only if its on-off delta exceeds the across-restart spread of that
parameter in the off-fit (and, for a handful of final vectors, the across-seed
spread — a few extra evaluations, since only the final candidates need
re-evaluating under two or three seeds). Fixing the criterion afterwards
invites reading noise as mechanism; this is the same discipline as the
project's own pre-registration of the staged on-fit layout in
`get_loss.split_param_list`. **Methodological deficiency.** Proposal: add the
acceptance criterion to §2 before the DBS-on fit is launched: report every
fitted delta with its restart spread; declare "changed" only above it; and
re-evaluate the final off- and on-vectors under at least two additional seeds.

**Goal relevance:** High — this is the project's stated claim; without a
criterion the claim is not falsifiable.

### 1.5 The fitted model's dynamical regime is never examined

**What BGM_22 does:** The only spiking statistics ever computed are mean rates
over the 9900 ms probe (`get_loss.get_firing_rate_10s`). Whether a fitted model
is asynchronous or oscillatory — whether its STN–GPe loop, which exists in the
model (`stn__gpe_proto`, `gpe_proto__stn`), sits below or above the oscillation
boundary — is never measured, and the BOLD loss cannot see it: the drive has
one value per 2.31 s TR, and the Balloon readout low-passes everything faster.
Two parameter vectors with equal BOLD correlation and in-band rates can
therefore differ qualitatively in regime, and the DBS inference would be read
off whichever one CMA-ES happens to return.

**What we do:** Regime is a validation target in its own right: Lindahl 2016
validate coherence and phase relations per state (Figs 2–3); Bahuguna 2020 map
the oscillatory/non-oscillatory boundary of the STN-GPe network in the rate
plane and locate the healthy and PD states relative to it (Figs 2–3); the whole
lineage's PD phenomenology is that dopamine depletion moves STN-GPe into or
toward the beta-oscillatory regime (experimental anchors as cited in Bahuguna
2020's introduction: Mallet et al. 2008 for the 6-OHDA rat; Tinkhauser et al.
2017 for the human STN off medication).

**Evidence behind our choice:** Excessive beta-band synchronization in STN/GPe
in the dopamine-depleted state, off medication, is one of the most replicated
electrophysiological findings in this field (rat: Mallet et al. 2008; human
LFP: Tinkhauser et al. 2017, Brown and colleagues — all as cited in Bahuguna
2020, refs 13, 20, 10). For this subject no electrophysiology exists, so the
check can only be a plausibility comparison against that literature, not a fit
target — we state this limit plainly.

**Why that would (or would not) be better here:** A reported check costs one
extra analysis of spike data the probe already records: the power spectrum of
STN and GPe population activity at the fitted off-vector and on-vector. It
cannot validate the model against this patient, but it can catch two failure
modes that would poison the inference: a fitted off-state with strong
oscillations at a frequency set by the model's rat delay set (the delays match
Kumaravelu et al. 2016's rat table, per `model_v07.md` §5, and delay values
shape loop-oscillation frequency), or an on-vs-off parameter delta that in fact
operates by flipping the network across the oscillation boundary — which our
Bahuguna 2020 results say small rate changes near the boundary can do — while
the report attributes it to a drive weight. **Methodological deficiency**
(reporting requirement, not a model change). Proposal: add STN/GPe spectra (and
a pairwise spike-count correlation summary) of the fitted vectors to the
validation-run artefacts of Roadmap phase 1 item 3, and state in the write-up
whether the fitted PD-off model is or is not beta-oscillatory.

**Goal relevance:** Medium — it does not block the fit, but it decides how the
fitted deltas may be interpreted mechanistically.

### 1.6 The input-correlation scan (§25): the plan is right; its output anchor is weaker than stated

**What BGM_22 does:** All presynaptic correlations ship at zero
(`mc.correlation_dict`, `mc.cortical_correlation`,
`ci.shared_fraction_dict`), deliberately, pending the scan described in
`input_streams/README.md` §5 and `TODO.md` §25: sweep input correlation against
simulated SPN output correlation and simulated BOLD amplitude, with Adler et
al. 2013's 0.004 downgraded — correctly, after reading the paper — to an
order-of-magnitude anchor on the output. The documentation of why a literature
value cannot simply be injected (the open-loop compensation cannot decorrelate;
Tetzlaff/Helias/Bernacchia citations) is exactly right.

**What we do:** We have no equivalent machinery to hold this against — our
models use uncorrelated Poisson input and say so as a limitation (Bahuguna 2020
limitations; Chakravarty 2022 Methods). On the *validation* side, however, we
require an experimental quantity on the output end of any calibration (standard
item 4).

**Evidence behind our choice:** Admission first: input correlation structure is
a place where our own practice is weaker than BGM_22's. The remaining demand —
an experimental anchor for the scan's chosen value — is methodological.

**Why that would (or would not) be better here:** One gap in the plan as
written: the scan's two readouts are the simulated output correlation
(anchored, weakly, by Adler) and the simulated BOLD *amplitude* — but the loss
is a Pearson correlation and therefore amplitude-invariant, so nothing
downstream constrains the amplitude the scan selects. The subject's own data
contain an amplitude: the per-ROI variance of the experimental BOLD series. If
the preprocessing of `sub-01_subdiv_results.h5` preserves physical percent
signal change (we could not verify the units), comparing simulated against
experimental BOLD variance per ROI would give the scan a second, genuinely
experimental anchor at no simulation cost; if the units are normalized away,
the plan should say explicitly that ρ is set by self-consistency alone.
**Methodological deficiency** (narrow; the plan itself is sound). Proposal:
determine the experimental BOLD units, and either add the per-ROI variance
comparison to the §25 scan or record in `input_streams/README.md` §5 that no
amplitude anchor exists.

**Goal relevance:** Medium — the scan is already the declared hard blocker on
the fits; this decides how well its outcome is grounded.

### 1.7 The cortical drive is measured, and the input streams carry a written, checked contract — in BGM_22's favour

**What BGM_22 does:** The cortical drive is deconvolved from the subject's own
cortical BOLD, split by tracer-derived per-region proportions, and delivered as
spike-count streams whose mean, Fano factor and pairwise correlation are
required to match closed-form targets, checked at build time with raising
errors, and stored for audit (`input_streams/README.md` §2;
`spike_input_cortex.check_stream_statistics`). What the streams deliberately
cannot represent — no feedback, no decorrelation, no cross-pathway identity, no
single-neuron temporal structure — is written down next to what the model may
therefore claim (§4–§5 of that README).

**What we do:** All three of our network papers drive the basal ganglia with
uncorrelated, stationary Poisson processes whose rates are tuned until target
firing rates come out (Lindahl 2016 Methods/Table 1; Chakravarty 2022 Methods;
Bahuguna 2020 Methods), and Bahuguna 2020's limitations section concedes that
real inputs are "richer in their statistics and dynamics, e.g. bursty,
periodic, correlated".

**Evidence behind our choice:** Our Poisson choice is an admitted convention of
convenience; the tuning targets are experimental (in-vivo rates), but the input
statistics themselves are not constrained by data in our models.

**Why that would (or would not) be better here:** For a single-subject BOLD fit
the measured drive is not merely acceptable, it is the point of the design —
the model is asked to reproduce this subject's BG time course given this
subject's cortex, which is a stronger question than our generic stationary
drive could pose. The statistical contract with build-time enforcement (born of
the project's own Fano-factor-1922 incident) exceeds anything in our own
methods sections, and we would adopt the practice, not criticize it.
**Difference, not deficiency** — in BGM_22's favour; no proposal.

**Goal relevance:** High — the drive is the model's main input and the loss's
main source of signal.

### 1.8 The striatal microcircuit against our microcircuit standard

**What BGM_22 does:** v07's striatum (`Microcircuit`) is 1000 Izhikevich-type
point neurons on a periodic lattice at measured density, with distance-Gaussian
connectivity fitted to the paired-recording literature, individually sampled
weights from literature-derived mixtures, and everything outside the cube
replaced by the compensation streams; SPN→FS connections are absent because
`fitted_params.json` has no such pair (`model_v07.md` §7.1–7.3).

**What we do:** Hjorth et al. 2020 build the striatum from reconstructed
morphologies with touch-detection synapse placement pruned to the same
paired-recording probabilities (Taverna et al. 2008: 26/6/36/28 % within 50 µm,
their ref 81; Planert et al. 2010: FS→dSPN 89 % vs FS→iSPN 67 % within 100 µm,
their ref 43), plus ChIN and LTS interneurons, short-term plasticity fitted to
20 Hz optogenetic trains (Fig 9), and dopamine modulation validated per cell
type (Fig 6) — and still list absent interneuron subtypes, absent
striosome/matrix structure and absent plasticity as limitations.

**Evidence behind our choice:** The pruning targets are the same experiments
BGM_22's `.ods` extracts — including Planert et al. 2010, a paper from our own
group. The morphological detail is grounded in reconstructions and
patch-clamp; its necessity for any given question is *not* itself
experimentally established, and our own network-level papers (Lindahl 2016:
MSN D1, MSN D2, FSN only; Chakravarty 2022: FSIs only, with the omission
listed as a limitation) use point neurons and no ChIN/LTS, exactly like
BGM_22.

**Why that would (or would not) be better here:** For fitting a 2.31 s-resolved
BOLD signal, Hjorth-level detail would be unusable (the fit needs thousands of
evaluations) and nothing we have read shows it necessary for this readout. Three
specifics deserve stating rather than judging. First, the absence of SPN→FS
connections matches our own standard — Hjorth 2020's connectivity scheme
(Fig 1C) contains no SPN→FS connection either — and BGM_22's
`input_streams/README.md` §4.1 spells out the consequence honestly. Second,
FS–FS gap junctions, present in Hjorth 2020 (with references there), are absent
in BGM_22; with 29 simulated FS driven almost entirely open-loop, their absence
cannot matter at this readout. Third, the apparent factor-2 conflict between
BGM_22's dSPN→dSPN amplitude and Hjorth's Taverna numbers dissolves on the
directional-probability convention (point 1.1) — the extraction is more careful
than a surface comparison suggests. The one substantive kernel problem is the
state pooling, which is point 1.1, not the reduction itself. **Difference, not
deficiency.** It would be reasonable to document in `model_v07.md` that the
FS-only interneuron set and absent gap junctions match the network-model
convention of the field (our papers included) rather than an oversight.

**Goal relevance:** Medium — the striatum is the largest simulated structure
and half the BOLD ROIs, but the reduction level is appropriate to the readout.

### 1.9 Equal 100-neuron BG nuclei against stereology-proportional sizes

**What BGM_22 does:** All six BG populations have 100 neurons per loop
(`parameters.csv` `*.size`), so STN, the three GPe classes, SNr and thalamus
are numerically equal, and the three GPe classes are equal to each other; the
BOLD pooling then reweights the GPe classes by measured abundances
(`get_loss.py: gpe_proportions`, sourced in `model_v07.md` §3.6 to Courtney et
al. 2023) while the dynamics run at equal sizes — a mismatch `model_v07.md`
§3.6 itself flags as an open question (three times too many arkypallidal
neurons relative to prototypic in the dynamics).

**What we do:** Population sizes proportional to stereological counts and
measured class fractions: Lindahl 2016 (Methods) scale Oorschot 1996's rat
counts to 80,000 neurons, giving STN 388, GPe-TA 329, GPe-TI 988, SNr 754
(carried unchanged into Chakravarty 2022 Table 1), with the TA/TI split from
Abdi et al. 2015 as cited there.

**Evidence behind our choice:** Stereology (Oorschot 1996, rat) and molecular
classification (Abdi et al. 2015; Mallet et al. 2012) are measurements; that
*relative nucleus size* must be preserved in a point-neuron model is, however,
our convention — what sizes buy is correct convergence ratios and finite-size
noise, both of which interact with fitted weights.

**Why that would (or would not) be better here:** With fixed in-degree
(`connect_fixed_number_pre(number=10)`) and per-cluster fitted weight
scalings, mean drives are absorbed by the fit, and the size choice is common to
both DBS conditions, so it largely cancels in the contrast. The
arky-vs-proto imbalance is the one place it could leak into the inference (GPe
lateral inhibition is disproportionately arkypallidal in the dynamics while
the BOLD readout weights it down), and that is already documented with a
pending proposal in `model_v07.md` §3.6; we defer to that triage rather than
re-propose. **Difference, not deficiency.** Documenting in `model_v07.md` §5
that the equal sizes are a computational convention, not an anatomical claim,
would complete the picture.

**Goal relevance:** Low-medium — mostly absorbed by the fit; the documented GPe
class imbalance is the part worth resolving.

### 1.10 The dopamine state is implicit in the fit rather than an explicit parameter map

**What BGM_22 does:** The subject's parkinsonism enters through the striatal
rate targets (Liang med-off, baked into caches and loss bands) and through
whatever values the fitted weights take; the striatal neuron models' own
dopamine terms are inert (`phi_1 = phi_2 = 0`, `model_v07.md` §6.2–6.3), with
`TODO.md` §30 tracking the unresolved question of whether 0 is the right value
for a patient off medication, scheduled in the phase-1 verdict pass. No
network-level parameter carries an explicit dopamine tag.

**What we do:** Explicit dopamine maps: every affected parameter is modified by
an experiment-keyed factor per state (Lindahl 2016 Table 9; Chakravarty 2022
Table 10 — e.g. CTX→STN and STN→GPe strengthening at depletion, sources cited
therein), so "healthy" and "PD" are reproducible parameter sets and the
difference between them is itself an object of study.

**Evidence behind our choice:** The individual effects are experimental (slice
and in-vivo studies cited per row in those tables); the composite α_dop scale
is a convention, and Chakravarty 2022 (Discussion) themselves found the
homogeneous α_dop insufficient — restoring a triphasic response in PD required
hand-adjusting three connections beyond the map, i.e. our own instrument is
admitted to be approximate.

**Why that would (or would not) be better here:** For this project's contrast
the explicit map is unnecessary: both fitted conditions are the same disease
state, so any static dopamine-dependent parameter shift is common mode and
absorbed by the base fit — the on-vs-off delta is DBS, not dopamine. What
remains load-bearing is internal state consistency among the fixed
(non-fitted) ingredients: rates (chosen: chronic med-off), connectivity kernel
(currently mixed — point 1.1), and the neuron models' phi terms (§30, pending).
The §30 plan — read the Humphries source, decide, capture a baseline before
changing — is adequate and correctly ordered before full-length caches.
**Difference, not deficiency.**

**Goal relevance:** Medium — via the state-consistency thread, which points 1.1
and §30 carry.

### 1.11 Provenance of the BG-level parameters stops at a docstring

**What BGM_22 does:** The 28 projections' weights and the six populations'
neuron parameters live in `parameters.csv` (column `BGM_v07_p01`), a pure value
table with no source column. `get_loss.py` calls them "the literature values"
(comment above `PROJ_CLUSTERS_COMMON`); the only recorded trail is the
`BGM_v07` docstring in `model_creation_functions.py` ("difference to Goenner et
al. (2021)... GPe parameters were refitted"), which names the parent model but
not which values survive from it, which were refit, or against what. The
uniform in-degree (`number` = 10 on all 28 projections) has no stated basis.
The seven delays were traced to Kumaravelu et al. 2016's rat table only by the
previous review round (`model_v07.md` §5), i.e. the project itself had not
recorded even that.

**What we do:** Standard item 1: per-value source columns (Lindahl 2016 Tables
7–9 with "n.d." where none exists; Chakravarty 2022 Tables 1–10), and
anatomically derived fan-ins (standard item 2) rather than a uniform
convergence.

**Evidence behind our choice:** The reporting convention is ours, not an
experimental finding; the fan-in evidence is stereology plus paired-recording
probabilities as cited in Lindahl 2016's Methods. Honesty requires noting that
a uniform low in-degree with rescaled weights preserves mean drive, and at a
2.31 s readout the granularity noise it changes is largely invisible — we
cannot show the value 10 harms this fit.

**Why that would (or would not) be better here:** The project's claim is that
fitted parameter *changes* are physiologically readable. The on-off delta of a
cluster scaling is baseline-free, so the inference survives uncited baselines
— but the moment the write-up says which pathway a scaling stands for and
whether its fitted value is plausible, the baseline weights and the in-degree
need a stated origin, exactly as the project's own CLAUDE.md conventions demand
for data. The fix is documentation, not refitting. **Methodological
deficiency** (confined to auditability). Proposal: a provenance note per
`BGM_v07_p01` row — "from Goenner et al. 2021", "refit on <target>", or "no
source" — in the CSV or a companion document, in the project's own
`activity_striatum` style; and one recorded sentence on the origin of
`number` = 10.

**Goal relevance:** Medium — it does not change any number, but the inference's
interpretability rests on it.

### 1.12 Reproducibility and audit discipline — in BGM_22's favour

**What BGM_22 does:** Caches refuse to load on any mismatched or absent
generation parameter; measured stream statistics are stored for post-hoc audit;
behaviour changes require a captured baseline proven bit-identical; loss files
record every component, the parameter vector and the gate state; the living
documents cite code by symbol and are kept in sync by convention
(`model_v07.md` §7.6; CLAUDE.md Conventions).

**What we do:** We publish code and models (Lindahl 2016 on GitHub, as stated
in the paper; Chakravarty 2022 on GitHub with NEST 2.20 and integration details;
Hjorth 2020 on EBRAINS and GitHub), which made our results rerunnable — but
none of our papers enforces input-provenance checks at load time or stores
generation-time statistics for audit; our reproducibility is repository-level,
not artefact-level.

**Evidence behind our choice:** Convention; the community's, and ours.

**Why that would (or would not) be better here:** BGM_22's artefact-level
discipline exceeds our published practice and is well matched to a project
whose expensive artefacts (terabyte caches, multi-day fits) outlive the code
state that produced them. One boundary worth noting: this discipline currently
lives in a private ecosystem (BGM_22 + a patched ANNarchy + editable
CompNeuroPy); at publication time, our repository-level standard — a tagged,
installable code state that reproduces the fits — still has to be met on top of
it. **Difference, not deficiency** — in BGM_22's favour.

**Goal relevance:** Medium — it protects the validity of everything the fits
will produce.

### 1.13 v08 as a smoke test

**What BGM_22 does:** `model_v08.md` presents v08 as a cache-free end-to-end
pipeline test only — collapsed drive, random connectivity, known-invalid
optimization bounds (`TODO.md` §1) — and no document we read leans on v08 for
any scientific claim.

**What we do:** We build reduced models deliberately and say what they are for
(Bahuguna 2020's two-population STN-GPe network is a minimal model "sufficient
to dissociate" its stated question; its limitations section bounds the claims).

**Evidence behind our choice:** Convention — a reduced model is legitimate
exactly to the extent its purpose is stated, which both sides here do.

**Why that would (or would not) be better here:** As long as v08 results are
never quoted as model results, the arrangement is sound; the honest labelling
is already in place. **Difference, not deficiency.**

**Goal relevance:** Low — by design.

---

## What we would ask for before publication

Only demands licensed by the deficiency verdicts above.

1. **Re-fit the striatal connectivity kernels per dopamine condition** from the
   already-labelled rows of `connectivity_probabilities.ods`, choose the state
   consistently with the Liang med-off rate calibration, document the choice,
   and rebuild the short caches (point 1.1; Roadmap phase 1, before any
   full-length cache).
2. **A provenance document for the six non-striatal firing-rate bands** —
   source, species, preparation, disease state per band, "Li et al., 2015"
   resolved to a citation, and the `gpe_cp` band flagged as an assumption if no
   class-specific recording exists — before the gate is enabled in any fit
   (point 1.2, folded into the §10 calibration).
3. **A null-model baseline for the BOLD loss** (cortical-mix regression per
   ROI), reported alongside every fitted loss, with fitted parameters
   interpreted only where the model beats it (point 1.3).
4. **A pre-stated acceptance criterion for "parameter changed"** in `TODO.md`
   §2 — deltas reported with across-restart spreads, plus a seed-robustness
   check of the final vectors — fixed before the DBS-on fit launches (point
   1.4).
5. **A regime report of the fitted models** — STN/GPe spectra and a pairwise
   correlation summary at the fitted off- and on-vectors, stated against the
   PD-off beta literature — as part of the phase-1 validation artefacts (point
   1.5).
6. **A units decision for the §25 scan's amplitude anchor** — either the
   experimental per-ROI BOLD variance enters the scan, or the README records
   that ρ is set by self-consistency alone (point 1.6).
7. **Per-value provenance for `BGM_v07_p01`** (inherited from Goenner et al.
   2021 / refit / unsourced) and one sentence on the uniform in-degree of 10
   (point 1.11).
