# Seat 3 — Girard / Doya (ISIR Sorbonne Université / OIST)

Review of BGM_22 written in character as the Girard–Doya lineage, applying that
lineage's own published standards unfiltered. Part of the community-conventions
survey defined in `TODO.md` §34; the panel and the reading list are in
`README.md`.

**Read for this review (full texts, from the PDFs in this directory):**

- Girard B, Liénard J, Gutierrez CE, Delord B, Doya K (2021), Eur J Neurosci
  53(7):2254–2277, `10.1111/ejn.14869`
- Shouno O, Tachibana Y, Nambu A, Doya K (2017), Front Neuroanat 11:21,
  `10.3389/fnana.2017.00021`
- Liénard JF, Aubin L, Cos I, Girard B (2024), Eur J Neurosci 59(7):1657–1680,
  `10.1111/ejn.16271`

**What was reviewed:** the same project documents as the other seats —
`model_v07.md`, `model_v08.md`, `DBS.md`, the three `experimental_data/`
READMEs, `CompNeuroPy`'s `parameters.csv` column `BGM_v07_p01`, and
`BOLD_optimization/get_loss.py`.

Every point carries a **goal-relevance** tag against the project's stated goal
(single-subject resting-state BOLD fitting and DBS inference).

---

## The standard this seat applies

This is the panel's seat for *data-constrained parameterisation of a primate
basal ganglia model*, and our standards are procedural:

1. **Two separable objectives, both scored.** Girard 2021 keeps an *anatomical*
   objective (bouton counts, dendritic synapse counts, mean synapse location
   along the dendrite, all within published ranges) apart from a *physiological
   value* objective — 14 discrete checks: resting firing rates per nucleus
   against plausible ranges statistically aggregated from 20 macaque studies,
   **plus** the model's response to nine pharmacological receptor-blockade
   experiments (AMPA, NMDA, GABA_A, and their combinations, in GPe and GPi).
2. **The plausible ranges are derived, published, and revised when data
   improves.** Girard 2021 tightened FSI from [0, 20] to [7.8, 14] Hz and MSN
   from [0, 1] to [0.05, 1] Hz, each with the reasoning and the underlying
   recordings named.
3. **Keep the whole solution set, not one point.** Fifteen parameterisations
   satisfy all constraints and all fifteen are carried forward; where a single
   operating point is needed, it is the centre of the largest hypersphere fitting
   inside the plausible domain (their Eq. 8) — deliberately the most robust point,
   not the best-scoring one.
4. **Delays are measured quantities with an error bar and a systematic estimation
   procedure.** Liénard 2024 estimates the full delay set from stimulation
   latencies across ten macaque studies, reports that the data are mutually
   *inconsistent*, publishes two alternative sets rather than one, and shows
   (their Fig. 9) that the oscillation frequency of the network is set by the sum
   of the STN→GPe and GPe→STN delays.
5. **Primate anatomy, and primate anatomy says the pathways overlap.** Girard
   2021's central structural commitment: more than 80 % of macaque striatal
   neurons project to both GPe and GPi/SNr (Parent et al. 1995; Lévesque & Parent
   2005), so a strictly segregated direct/indirect model is at odds with the
   species BGM_22 is fitting.

We note at the outset what we recognise as good practice here: the striatal
microcircuit's connectivity is fitted from data with published kernels, the
cortical proportions are derived from macaque tracer counts with an explicit
audit trail marking each step as measured, assumed or judged, and the FS rate is
pooled from the same three primate datasets we used. That last one is a real
convergence and we say so in point 3.4.

---

## Points

### 3.1 Strict pathway segregation, in a model of a human subject

`model_v07.md` §5's projection table: `str_d1 → snr, gpe_cp` and
`str_d2 → gpe_proto, gpe_arky, gpe_cp`. So dSPNs do reach one GPe population,
but iSPNs reach no output nucleus at all, and the arrangement is essentially the
Albin–DeLong segregation with a small dSPN→GPe addition.

Our lineage's position, stated in Girard 2021's Introduction, is that this is the
"second problem with the basal ganglia" (Nambu 2008): anatomical results in
cynomolgus and squirrel monkeys show the pathways' boundaries "are quite blurred,
as more than 80 % of striatal neurons simultaneously project to nuclei supposed
to belong to segregated pathways". Our 2021 paper's whole point is that a model
with *overlapping* pathways still performs action selection — segregation is not
needed to get the function, so it should not be assumed to get the anatomy.

For BGM_22 the argument is sharper than for us, because BGM_22 fits a human and
takes its cortical proportions from macaque tracer data (Borra et al. 2021,
2022) — the same experimental tradition that produced the overlap finding. It is
inconsistent to accept macaque tracer counts as the anchor for corticostriatal
proportions and reject macaque tracer counts on striatofugal collateralisation.

**Goal relevance: medium.** A missing iSPN→SNr collateral changes how a change in
indirect-pathway drive propagates to the GPi ROI's BOLD, which is one of the
seven fitted regions. It is not fatal — it is a structural prior the inference
inherits.

### 3.2 The delays are round numbers where measured values exist, and the
cortical drive arrives with no delay at all

Liénard 2024 Table 1 is the closest thing the field has to a measured primate
delay set. Placing BGM_22's `parameters.csv` delays beside our two solutions
(*Cmpr* = best compromise over all ten studies; *NoPlkv* = the same excluding the
one mutually inconsistent study):

| pathway | BGM_22 | Cmpr | NoPlkv |
|---|---|---|---|
| Ctx → Str | **0** (`CurrentInjection`) | 6 | 9 |
| Ctx → STN | **0** (`CurrentInjection`) | 4 | 4 |
| Str → GPe | 5 | 8 | 6 |
| Str → GPi/SNr | 4 | 11 | 8 |
| STN → GPe | 2 | 9 | 2 |
| STN → GPi/SNr | 1.5 | 4 | 4 |
| GPe → STN | 4 | 1 | 7 |
| GPe → GPi/SNr | 3 | 1 | 3/4 |

Two observations. The striatofugal delays are roughly half our estimates in both
solutions, which is the one place where BGM_22's numbers fall outside the range
our data admit at all. And the cortical drive has **no delay**: `CurrentInjection`
is wired with `connect_current()` and injects into the current timestep
(`model_v07.md` §7.7), so cortex reaches striatum, STN, GPe and thalamus
simultaneously. The hyperdirect pathway's defining property is that it is
*faster* than the trans-striatal route — 4 ms against 6–9 ms in our estimates —
and in this model that difference is zero.

**Goal relevance: low for the BOLD fit, medium for DBS.** At a 2310 ms TR a
millisecond-scale delay cannot move the fitted time course. But `DBS.md`'s known
limitation 1 already records that DBS axon spikes bypass the delay lines
entirely, so the DBS volley's arrival times are set by a mechanism that is
doubly detached from measurement. If any conclusion is drawn about *where* in
the circuit DBS acts, this matters.

### 3.3 One weight per projection and one drive weight per population, with no
per-neuron variability — and we know from our own model what that costs

Girard 2021 reports its own failure honestly: our simulated coefficients of
variation are lower than experimental ones and our rate distributions narrower,
and we attribute this to "a too homogeneous construction of the models: all cells
have the same number of input synapses from the same number of neurons, the same
constant input, the same threshold... Adding individual variability around these
mean values would probably enlarge the spread."

BGM_22 is more homogeneous than that model in every respect we can check. All 28
projections use `connect_fixed_number_pre(number=10)` with one scalar weight, so
every neuron has exactly 10 afferents of each type at exactly the same strength.
The fitted cortical drive weight is one number per postsynaptic *type* applied to
every region's stream (`model_v07.md` §11), so every dSPN in the caudate has the
identical drive gain. `base_mean` is a single value per population. The only
heterogeneity anywhere in the model is the microcircuit's distance-sampled
intrinsic weights and the per-neuron noise term.

We raise this here rather than as a general realism complaint because of what
BGM_22's observable is. The BOLD signal is a *sum over the population* of
synaptic current; a homogeneous population sums to a scaled single neuron, and
its variance is set almost entirely by the input correlation
(`input_streams/README.md` §3 makes exactly this calculation). Heterogeneity in
gains and thresholds is one of the mechanisms that decorrelates a real
population, and it is absent alongside the recurrent decorrelation the same
README says is absent (§4.1).

**Goal relevance: high.** It compounds with the correlation question every seat
has raised, and it acts on the amplitude of the fitted signal.

### 3.4 The firing-rate bands: one is derived from our own sources, the rest are
looser than the data support

`get_firing_rate_loss`'s FS band (3.42, 17.58) Hz is built from Marche &
Apicella, Yamada et al. and Adler et al. — the three primate datasets we used to
set our own FSI constraint. We derived [7.8, 14] Hz from them by the confidence
interval methodology of Liénard & Girard 2014; BGM_22 derives ±1 SD across
neurons from a single study's CV and lands three times wider. The project's own
README states the reasoning and calls it "a plausibility band, not a confidence
interval", which we accept as honest — but the consequence is that a gate built
on it admits FS rates from 3.4 to 17.6 Hz where the same data constrain the
population mean to about half that width.

The comparison across all bands, with our own plausible ranges (macaque,
normal, from Girard 2021 Fig. 2):

| population | BGM_22 band | our range | note |
|---|---|---|---|
| STN | (28, 80) | ~15–25 (model lands 17–19) | BGM_22's floor is above our ceiling |
| SNr/GPi | (21, 93) | ~60–90 (model ~75) | BGM_22's band is 3.5× wider |
| GPe | proto (75, 85), arky (15, 20), cp (75, 85) | ~50–70 (model ~60) | see below |
| FSI | (3.42, 17.58) | 7.8–14 | same sources, 3× wider |
| MSN | dSPN (12.67, 37.33), iSPN (21.22, 44.78) | 0.05–1 | different condition; see seat 7 |

The MSN row is not a criticism — those are parkinsonian medication-off primate
values and ours are normal — but the STN row is, because the same reasoning does
not obviously apply. Our normal-macaque STN plausible range peaks around 25 Hz;
Shouno 2017's Table 1, recomputed from Tachibana et al. 2011's recordings, gives
STN 19.9 ± 9.5 Hz normal and **27.6 ± 11.3 Hz parkinsonian**, and GPe 65.1 ± 25.6
normal against **41.1 ± 22.3 parkinsonian**. So in the primate MPTP data this
lineage calibrates against, dopamine depletion *raises* STN modestly and
*lowers* GPe substantially. BGM_22's bands do the opposite for GPe: 75–85 Hz for
two of its three GPe populations, above even the normal value, in a model of a
parkinsonian patient.

**Goal relevance: high.** These bands gate every BOLD evaluation.

### 3.5 A single fitted vector per condition, where our experience is that the
solution set is large

Girard 2021 carries fifteen parameterisations, all of which pass the same
anatomical and physiological constraints, and reports that they differ in their
internuclei connection strengths while producing similar models. Faced with a
choice, we do not take the best-scoring point; we take the centre of the largest
hypersphere that fits inside the plausible region, precisely because the
best-scoring point is not robust to model stochasticity.

BGM_22's design (`deap_cma_opt.py`, CMA-ES, one run per condition) returns one
vector per DBS condition and reads the inference off the difference between them.
If the DBS-off solution set is a manifold rather than a point — which our
experience says it will be, at 19 parameters against one scalar loss — then the
difference between two arbitrary points on two manifolds is not an inference about
DBS. Seat 2 raises the same concern from the identifiability side; we raise it
from the degeneracy side, and they are the same problem.

The minimum we would want: multiple independent CMA-ES restarts per condition,
with the spread of the resulting vectors reported alongside the DBS-on/off
difference. If a parameter's between-restart spread exceeds its on-minus-off
change, no claim can be made about it.

**Goal relevance: high. This is the inference.**

### 3.6 Two inputs our lineage treats as essential are absent

**CM/Pf.** In Girard 2021 the centromedian/parafascicular thalamic input reaches
*every* simulated nucleus — MSN, FSI, STN, GPe and GPi/SNr — and our sensitivity
analysis (their Table 3 and Fig. 6) makes it the input that "modulates the
responsiveness of action selection", i.e. a global gain on the circuit. BGM_22
has one thalamic population per loop, and it is the BG *target* (`snr → thal`,
`thal → striatum`); nothing plays the CM/Pf role of a thalamic drive onto STN,
GPe and the output nuclei. Since the model does contain `CorticalInputs` streams
onto `thal`, `gpe_arky`, `gpe_cp` and `stn`, the machinery to add it exists — but
those streams are labelled cortical and mixed by cortical proportions.

**Subthalamo-striatal.** Girard 2021 includes STN→MSN and STN→FSI (17 % of STN
neurons projecting, Table 1, from Nakano et al. 1990 / Parent & Smith 1987);
BGM_22 has no STN→striatum projection.

**Goal relevance: low–medium.** Neither is likely to change a BOLD correlation
much, but the CM/Pf omission removes a global gain term that would otherwise
compete with the fitted drive weights for the same explanatory role.

### 3.7 Short-term depression at GPe→STN, and what its absence means for the DBS
result specifically

Shouno 2017's mechanism is worth stating in full because it bears on BGM_22's DBS
half. In that model, dopamine depletion reduces GPe autonomous activity; because
GPe→STN synapses carry strong short-term depression, *lower* GPe firing means
*less* depressed synapses and therefore paradoxically *stronger* effective
inhibition of STN; that inhibition de-inactivates the STN T-current and produces
post-inhibitory rebound bursts, which the recurrent loop shapes into 8–15 Hz
oscillations. Their Fig. 6C shows the 8–15 Hz power is a monotone function of the
depression parameter alone.

BGM_22 has neither the T-current (Izhikevich point neurons) nor short-term
depression, so this mechanism is unavailable — which is fine for a BOLD model.
The consequence to record is about DBS: the same GPe→STN synapse is the one
`DBS.md` drives with an orthodromic axon volley at 125 Hz, at a static weight.
In Shouno's calibration (their Fig. 2, fitted to Atherton et al. 2013) a GPe→STN
synapse stimulated at 100 Hz falls to a transmission probability far below its
resting value within a second or two. A model without that depression delivers
the full weight on every one of the 125 pulses per second.

**Goal relevance: high for the DBS inference.** Seat 2 reaches this from Rubin
2017 and the STN-axon side; we reach it from the GPe→STN side and the same
correction is needed.

### 3.8 The model's own honesty is a strength — provided the claims inherit it

We want to record that `experimental_data/input_streams/README.md` §5 does
something we have not seen in a submission before: it states which inferences the
model may support ("input gains, drive amplitudes, cellular and synaptic
parameters") and which it may not ("claims about emergent circuit dynamics, about
synchronisation arising from striatal connectivity, or about anything that
requires the recurrent loop to be doing work"). That is the correct instinct and
the correct division.

Our request is that the same discipline be applied to the *parameter* claims.
"Which parameters had to change under DBS" is on the legitimate side of that
line only if the parameters are identifiable (point 3.5). At present the document
licenses a class of inference that the fitting procedure has not yet been shown
to support.

**Goal relevance: high.**

---

## What we would ask for before publication

1. Multiple independent optimisation restarts per DBS condition, with the
   between-restart parameter spread reported alongside the on-minus-off
   difference; no per-parameter claim where the spread exceeds the change
   (point 3.5).
2. Either the striatofugal delays brought into the range of Liénard 2024's two
   solutions, or a statement that delays are not identifiable at BOLD resolution
   and are therefore not interpreted (point 3.2).
3. A source and a derivation for the STN, SNr and GPe rate bands, with the
   parkinsonian direction checked against Tachibana et al. 2011 as recomputed in
   Shouno 2017 Table 1 (point 3.4).
4. Per-neuron heterogeneity in at least the drive gain, or an explicit argument
   that the population sum is insensitive to it (point 3.3).
5. A statement of what strict direct/indirect segregation assumes, given that the
   model's own cortical proportions come from the macaque tracer literature that
   reports the overlap (point 3.1).
6. Short-term depression on GPe→STN, or an explicit statement that the fitted
   `axon_spikes_per_pulse` absorbs it and is therefore not an axonal recruitment
   probability (point 3.7).
