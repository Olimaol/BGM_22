# Seat 6 — Chakravarthy (IIT Madras)

We write in character as the Chakravarthy computational neuroscience lineage
(IIT Madras), applying the standards of our own published basal ganglia
program: a long-running series of spiking and rate-coded cortico-BG models
with explicit STN-DBS simulation — electrode position, current spread,
antidromic activation, medication effects on behaviour — and, in its current
multiscale form, explicit dopamine pharmacology. We are the panel's third
DBS-capable voice, and the DBS representation (`DBS.md`) is in our scope with
authority alongside the network itself. This is our contribution to **round 2**
of the `TODO.md` §34 community-conventions survey (step 3 rerun 2026-08-14;
the panel definition and reading list are in `../README.md`). It was written
independently of the round-1 review, which we did not consult. Per the first
round-2 requirement, every appeal to our own practice below states the
experimental evidence that practice rests on as documented in our own papers,
or admits explicitly that it is a convention, an estimate, or a tuned value —
and a convention of ours is reported as a difference, never as a deficiency of
BGM_22. Per the second, every point is structured as an explicit contrast —
what BGM_22 does, what we do, the evidence behind our choice, whether ours
would be better *for this project's stated goal*, and a goal-relevance tag —
ending in exactly one of the three permitted verdicts.

**Read for this review (full texts, from the PDFs in this directory's parent):**

- Mandali A, Chakravarthy VS (2016). Probing the Role of Medication, DBS
  Electrode Position, and Antidromic Activation on Impulsivity Using a
  Computational Model of Basal Ganglia. *Front Hum Neurosci* 10:450.
  doi:10.3389/fnhum.2016.00450. (Below: M&C16.)
- Nair SS, Muddapu VR, Chakravarthy VS (2022). A Multiscale, Systems-Level,
  Neuropharmacological Model of Cortico-Basal Ganglia System for Arm Reaching
  Under Normal, Parkinsonian, and Levodopa Medication Conditions. *Front
  Comput Neurosci* 15:756881. doi:10.3389/fncom.2021.756881. (Below: N22.)

**What was reviewed:** `model_v07.md` (full), `model_v08.md` (full), `DBS.md`
(full), `CLAUDE.md`, `experimental_data/berlin_data/vta/` (`stim_settings.csv`
and all three subjects' overlap and volume files), the three
`experimental_data/` READMEs (`input_streams`, `activity_striatum`,
`cortical_proportions`, full), `BOLD_optimization/get_loss.py` (full),
`BOLD_optimization/parameters.py` (full), `BOLD_optimization/deap_cma_opt.py`
(`search_space`, `load_best_off_fit`), `TODO.md` (Roadmap and §§1–5, 10,
14–17, 28, 30–32; §34 skipped per the survey rules), and in CompNeuroPy:
`dbs.py` (`add_dbs_mechanisms`, `add_DBS_to_spiking_neuron_model` region,
`DBSstimulator.__init__`, `_create_dbs_on_array`, `_set_constants`) and
`full_models/bgm_22/model_creation_functions.py` (`BGM_v07` head) with
`parameters.csv`. Code claims in the documents were spot-checked against the
code, per the project's own rule.

## The standard this seat applies

Our lineage's recurring demands, each with its evidence basis stated:

1. **Simulate the therapy the patient actually receives, at clinical
   parameter values.** In M&C16 the DBS frequency (130 Hz) and pulse duration
   (100 µs) were "chosen such that they are comparable to that used in a
   clinical setting (Garcia et al., 2005)" (Materials and Methods, DBS
   Stimulation); in N22 levodopa is dosed in milligrams through a
   two-compartment pharmacokinetic model adapted from Baston et al. 2016 and a
   three-compartment terminal model from Reed et al. 2012 (Levodopa Medication
   section). The *electrical* internals of our DBS — amplitude ~220 pA at
   point neurons, Gaussian width σ = 5 lattice units — are model-scale
   constructions with no measured referent, and we say so here.
2. **Represent DBS's spatial selectivity: electrode position and partial
   coverage of the nucleus.** The motivating evidence is behavioural:
   electrode position alters behaviour in patients (Hershey et al. 2004, 2010,
   as cited in M&C16's Electrode position section); the mechanistic
   attribution to "difference in pattern and volume of STN activation" rests
   on a computational study (Miocinovic et al. 2006, same citation trail). Our
   own spatial substrate — a 50×50 lattice with electrode centres at lattice
   points — has no anatomical coordinates: M&C16 states outright that the
   quadrant mapping "is a modeling assumption that has to be made in the
   absence of experimental data" and that "the four quadrants in the modules
   do not correspond to the well-known basal ganglia loops", and lists the
   spatial definition of STN territories as an open limitation (Conclusions).
3. **Represent antidromic activation.** Our basis is explicitly "theories
   that stimulation of STN could result in antidromic activation of GPe, GPi,
   or cortical neurons (Hauptmann and Tass, 2007; Montgomery and Gale, 2008)"
   (M&C16, Antidromic activation) — not a measurement. The percentages we
   used (10/50/75 %) were swept, and judged only by behavioural consistency
   with Frank et al. 2007's DBS patients.
4. **Anchor intermediate observables where a measured range exists.** N22
   checks the model's striatal extracellular dopamine (~264 nM in control)
   against the measured 150–400 nM range cited to Schultz 1998 (Results,
   Effect of SNc Cell Loss). Our STN synchrony observables, by contrast, are
   validated only qualitatively against the healthy-desynchronized /
   parkinsonian-synchronized contrast (Bergman et al. 1998; Wilson & Bevan
   2011, as cited in M&C16) — a direction, not a number.
5. **Validate at the level of the patient-facing outcome.** M&C16 against
   probabilistic-learning-task accuracy and reaction times of PD patients
   ON/OFF/DBS (Frank et al. 2004, 2007) and human STN activity under conflict
   (Zaghloul et al. 2012); N22 against arm-reaching kinematics of PD patients
   (Majsak et al. 1998) and UPDRS-III time courses under levodopa (Nomoto et
   al. 2018).
6. **Probe the model by forward parameter variation and report the map, not
   one configuration** — electrode positions, antidromic percentages, cell
   loss 25–75 %, dosages 50–300 mg. This is our practice; its justification
   is methodological (it is what exposed our own degeneracies, point 6.7), not
   experimental.
7. **Name the unconstrained structure.** Our lattice sizes (50×50 in M&C16;
   15×15 and 8×8 in N22) carry no anatomical justification in either paper;
   the STN↔GPe wiring in M&C16 is one-to-one, a simplification on top of
   Plenz & Kital 1999 (which our paper cites for the bidirectional coupling,
   from organotypic cultures, not for the topology); the lateral radii
   r_s = 1.4, r_g = 1.6 were tuned to make STN-GPe desynchronize in the
   healthy condition; the PD-condition weights (w_Str→D1 4→3, w_STN→GPi
   1.5→2) are hand-set. We hold BGM_22 to nothing these values could not
   survive themselves.

## Points

### 6.1 VTA coverage as a scalar neuron fraction versus our Gaussian current spread

**What BGM_22 does:** DBS coverage of the stimulated population is a single
scalar, `population_proportion = (35+23)/(70+75) = 0.4`, derived from the
fitted subject's own volume-of-tissue-activated overlap with the motor STN
subdivision (`experimental_data/berlin_data/vta/sub-01/sub-01_overlap.csv`:
35/70 left, 23/75 right), applied by `DBSstimulator._create_dbs_on_array`
(CompNeuroPy `dbs.py`) as a randomly shuffled 0/1 array over the 100
`stn:putamen` neurons. Neurons inside the set get the full somatic and
axonal effect; outside, none. There is no electrode position and no spatial
falloff, because the STN population has no spatial coordinates.

**What we do:** M&C16 gives the stimulation current to the 50×50 STN lattice
as a Gaussian, `I_DBS = A·exp(−((i−i_c)²+(j−j_c)²)/σ²)` (Eq. 6), with the
electrode centre at a lattice point and σ = 5, and shows that moving the
centre between three positions flips the model between reward- and
punishment-biased learning (Fig. 7) and changes reaction time via STN
activity (Figs. 10–11).

**Evidence behind our choice:** the position-dependence is motivated by human
behavioural studies (Hershey et al. 2004, 2010, as cited in M&C16) and the
spread form by computational work (Hauptmann & Tass 2007; Foutz & McIntyre
2010, same citation trail). But our lattice coordinates have no anatomical
referent — our own paper admits the quadrants correspond to no real BG
territory — so our spatial DBS is an abstract construction, and σ = 5 is a
chosen value, not a measurement.

**Why that would (or would not) be better here:** it would not. For a
single-subject fit, a coverage fraction computed from that subject's own
imaging-derived VTA is better grounded than a Gaussian on a lattice with no
coordinates; BGM_22's scalar is the anatomically honest reduction of what its
100-neuron, geometry-free STN can carry. What our experience does license is a
sharpening of the already-planned sensitivity check (`TODO.md` §16): the
subject's own data supply the principled range — per-hemisphere motor
coverage is 0.50 (left) and 0.31 (right), so the sensitivity scan over
`population_proportion` should span at least [0.31, 0.50] rather than an
arbitrary ± band around 0.4. That is a methodological refinement of an
existing plan, checkable from the repository's own files. **Verdict:
difference, not deficiency** (with the §16 range suggestion).

**Goal relevance:** high — the coverage constant scales every DBS effect the
inference will read.

### 6.2 The associative STN's coverage is rounded to zero, against the subject's own data

**What BGM_22 does:** only `stn:putamen` is stimulated; the caudate loop is
excluded from every DBS mechanism by design and serves as the free control
(`DBS.md`, "The DBS footprint in this model"; `get_loss.py`,
`DBS_LOOP = "putamen"`). The caudate-loop STN receives the caudate cortical
mix (dlPFC-dominated, `parameters.py: cortical_proportions_dict`), i.e. it
stands for the associative STN territory in all but name.

**What we do:** M&C16 has no functional subdivisions to protect — one STN
lattice serves all four choices, and we stated that its quadrants correspond
to no real territories — so we have no superior practice to offer on the
subdivision itself.

**Evidence behind our choice:** none; this point is grounded not in our
papers but in the reviewed project's own data. The same overlap files that
yield the 0.4 motor coverage show non-zero VTA overlap with the associative
subdivision: 5/68 (left) and 7/68 (right), pooled ≈ 0.09, and limbic 6/109 ≈
0.055 (`sub-01_overlap.csv` together with `sub-01_volumes.csv`).

**Why that would (or would not) be better here:** the free-control design —
caudate's on-vs-off BOLD change must be explained entirely by its cortical
drive — is an inference instrument we endorse (our own designs have nothing
as clean). But the subject's data say the real associative STN receives a
small, non-zero direct stimulation (~9 % of the subdivision), which the
design rounds to zero. To the extent that matters, the real caudate-territory
on-off change contains a DBS component the model will attribute to cortical
drive, slightly biasing the control's prediction by construction. The
magnitude is plausibly small (0.09 versus 0.40, and the associative loop's
STN contributes to shared GPi/GPe/STN monitors only alongside the putamen
loop), but it is a measured number in the project's own files, currently
stated nowhere in `DBS.md` or `TODO.md` §16. We propose: document the
rounding as an explicit assumption of the free control, and bound its effect
— either analytically or with one sensitivity run giving `stn:caudate` its
measured ~0.09 coverage (which requires extending the DBS retrofit to that
population, a compile-level change) — before the control is used to certify
the on-fit. **Verdict: experimentally grounded deficiency** (minor; the
grounding is the subject's own VTA data).

**Goal relevance:** medium — the free control is a load-bearing instrument of
the planned inference, and this is a known-magnitude crack in it.

### 6.3 The somatic DBS term: hyperpolarizing shunt versus our depolarizing current

**What BGM_22 does:** the somatic effect is
`+ pulse(t)*dbs_on*dbs_depolarization*neg(-90 - v)` appended to every
`dv/dt` line (`dbs.py`, `add_DBS_to_spiking_neuron_model`; documented in
`DBS.md`) — a shunting pull toward −90 mV, active for one 0.1 ms step every
8 ms, with fitted amplitude in [0, 10]. Despite its name the term
hyperpolarizes, which the code comments and `DBS.md` state plainly. Axonal
excitation is a separate, independent channel (point 6.4).

**What we do:** M&C16 applies a *depolarizing* current (A ≈ 220 pA at the
Gaussian peak) to STN somata (Eq. 6); reduced STN activity under DBS emerges,
where it emerges, as a network effect (Fig. 11: significantly lower STN
activity for one electrode position during the first 600 ms).

**Evidence behind our choice:** none that decides the sign. Our parameters
were "comparable to a clinical setting (Garcia et al., 2005)" in frequency
and pulse width; the amplitude, sign and spatial form of the somatic current
are conventions of ours. The excitation-versus-inhibition question for the
stimulated soma is, in our own citation trail, an open debate rather than a
settled measurement.

**Why that would (or would not) be better here:** BGM_22's structure is the
better one for an inference goal: somatic suppression and axonal excitation
are separate fitted parameters, so the on-fit can in principle allocate
between the two hypotheses instead of inheriting a modeller's choice of sign
— our fixed depolarizing current could not. Whether BOLD at TR = 2.31 s can
actually separate them is an identifiability question (point 6.7), not a
structural fault. The naming hazard (`dbs_depolarization` hyperpolarizes) is
already documented in three places; we only ask that the eventual write-up
carry it, since a reader of the fitted value will otherwise invert its
meaning. **Verdict: difference, not deficiency** — structurally in BGM_22's
favour.

**Goal relevance:** medium — the parameterization shapes what "what DBS did"
can even be expressed as.

### 6.4 Antidromic activation: fitted axon spikes and soma invasion versus our swept current fraction

**What BGM_22 does:** axonal DBS is a per-pulse probability
(`axon_spikes_per_pulse`, fitted in [0, 1]) of an axon-spike event that
drives efferent synapses without somatic spiking, bypassing the delay line
(known, `TODO.md` §14); antidromic invasion is a full Izhikevich reset
(`v → c`, `u += d`) applied without threshold crossing, with probability 1 on
the stimulated STN and probability equal to the VTA coverage (0.4) on the
spiking afferent `gpe_proto` (`dbs.py`, axon reset; `DBS.md`, "What `on()`
writes"). The afferent volley also travels orthodromically into the covered
STN neurons via `gpe_proto__stn`.

**What we do:** M&C16 models antidromic activation of GPe by adding a fixed
percentage of the STN-bound DBS current directly to GPe membrane potentials,
sweeping 10/50/75 % and comparing the resulting behaviour with DBS patients
(Figs. 8, 12): at 50–75 % the accuracy pattern resembled the experimental DBS
group, and 75 % reduced reaction times as reported clinically (Frank et al.
2007, as cited).

**Evidence behind our choice:** the mechanism is grounded in "theories"
(Hauptmann & Tass 2007; Montgomery & Gale 2008, as cited in M&C16); the
percentage was never measured, only swept, and our support for any particular
value is indirect behavioural consistency. We cannot claim experimental
superiority for our current-fraction implementation.

**Why that would (or would not) be better here:** BGM_22's representation is
at least as mechanistic as ours — spike-event antidromics with soma invasion
rather than a smeared current — and, decisively for this project, the
antidromic strength is *fitted* rather than assumed, which is what the
DBS-off/DBS-on inference needs. Tying the afferent antidromic probability to
the VTA coverage fraction is an assumption, but a stated and sensible one.
The zero-delay arrival of DBS volleys (§14) is a real deviation whose
BOLD-timescale impact is plausibly negligible, as that entry argues; its
revisit trigger (if `axon_spikes_per_pulse` carries much of the explanation)
is the right one. **Verdict: difference, not deficiency** — in BGM_22's
favour on both counts.

**Goal relevance:** medium.

### 6.5 Pulse train constants: the subject's own settings, and what the 100 µs substitution does

**What BGM_22 does:** 125 Hz is the subject's programmed frequency
(`stim_settings.csv`: sub-01 `stim_f` 125, pulse width 60 µs); the model uses
a 100 µs pulse width because 60 µs is below dt = 0.1 ms (`DBS.md`, known
limitation 3; `TODO.md` §16 plans a sensitivity check). The pulse function
(`_set_constants`) is 1 for exactly one timestep every 8 ms.

**What we do:** M&C16 uses 130 Hz / 100 µs as generic clinically-comparable
values (Garcia et al. 2005, as cited) — not any particular patient's
settings.

**Evidence behind our choice:** the clinical comparability citation covers
the order of magnitude only; we had no subject to match. BGM_22's use of the
fitted subject's own frequency is strictly closer to the data than our
practice.

**Why that would (or would not) be better here:** our examination of the
implementation can, we believe, largely close §16's pulse-width worry by
argument rather than simulation, and we offer the argument for checking. At
dt = 0.1 ms, `pulse(t) = ite(modulo(t·1000, 8000) < width_µs, 1, 0)` is ON
for exactly one timestep per period for *any* width in (0, 100] µs — a 60 µs
width would produce the identical realized pulse train, so the somatic term
is unaffected by the substitution. The only thing the width touches is the
conversion `np.clip(axon_spikes_per_pulse · 1000 · dt / width, 0, 1)`
(`_axon_spikes_per_pulse_to_prob`): identity at 100 µs, a factor 1.67 at
60 µs. Since `axon_spikes_per_pulse` is fitted on [0, 1] and the result is
clipped to [0, 1], the *reachable* per-pulse spike probabilities are the same
interval either way — the substitution restricts nothing the fit can express;
it only rescales the fitted parameter's nominal meaning. The residual cost is
purely interpretive: the fitted value is "spikes per 100 µs pulse" for
hardware delivering 60 µs pulses, so any cross-subject or cross-study
comparison of the number must renormalize. We suggest recording this argument
in §16 (or `DBS.md`) so the planned sensitivity run can be narrowed to the
interpretation caveat. **Verdict: difference, not deficiency** — in BGM_22's
favour on the frequency; the width substitution is benign in expressiveness
and needs only the documentation note.

**Goal relevance:** medium — it decides what the fitted DBS numbers mean when
read off.

### 6.6 The hyperdirect pathway cannot be activated by DBS — a limitation our own models share

**What BGM_22 does:** the cortical drive to STN is a `TimedArray` →
`CurrentInjection` chain with no soma, so it is excluded from the DBS
footprint; `afferents=True` in practice means `gpe_proto→stn` only
(`TODO.md` §15; `DBS.md`, known limitation 2). A fitted "afferent" DBS
effect is therefore a pallidal one, and cortical fibre activation — a
much-discussed DBS mechanism — is outside the model's expressible space.
Unlike a DBS-activatable pathway, the corticosubthalamic *drive itself* is
present (the `CorticalInputs` streams), per-region and subject-derived.

**What we do:** M&C16 contains no hyperdirect pathway at all — "The input
from the cortex to STN, also known as the hyper-direct pathway (Nambu, 2015),
and the GABAergic projection from GPe to GPi were not included in the model"
— and our antidromic mechanism covered GPe only; N22 likewise lists the
absence of the hyperdirect pathway as a limitation of its own model (Nambu et
al. 2002; Cai et al. 2019, as cited there). Both of our papers defer it to
future work.

**Evidence behind our choice:** our omission was justified only by "the
functional significance of these connections is not fully understood"
(M&C16) — an admission, not evidence. We therefore have no standing to
demand the pathway's DBS activation from BGM_22, and the round-2 rules bar us
from proposing our (non-existent) alternative.

**Why that would (or would not) be better here:** BGM_22 is ahead of our own
practice — it has the corticosubthalamic drive we lacked, and it has already
written down exactly what the DBS gap does to the inference ("a fitted
'afferent' effect here is a pallidal one", §15). That bound must survive into
any write-up of the inference result, because the project's stated goal is
precisely to name what DBS did, and one leading candidate mechanism is
excluded by construction. §15's assessment that representing it means a model
change (a spiking cortical relay), not a bug fix, is correct. **Verdict:
difference, not deficiency** — with the request that §15's scope bound be
stated wherever the inference is reported.

**Goal relevance:** high — it truncates the hypothesis space the inference
can select from.

### 6.7 The loss sees no dynamics faster than one TR; our own results show why that underdetermines DBS mechanism

**What BGM_22 does:** the objective is the per-region BOLD time-course
correlation at TR = 2.31 s plus a firing-rate plausibility term
(`get_loss.compute_bold_correlation_loss`, `get_firing_rate_loss`); no
spectral, synchrony or timing observable exists anywhere in the pipeline.
`TODO.md` §2 already names near-degenerate pairs among the 13 DBS-on free
parameters (`axon_spikes_per_pulse` vs the `stn__gpe`/`stn__snr` scalings;
`passing_fibres_strength` vs the `snr__thal` scaling) and defers the proper
inference design until after the first pipeline-proving fits.

**What we do:** we probe DBS parameters forward and separate mechanisms by
intermediate neural observables. In M&C16, electrode position and antidromic
percentage — two distinct mechanisms — produced overlapping signatures at
the behavioural readout: position 3 flipped accuracy and cut reaction time
(Figs. 7B, 10), and 75 % antidromic GPe activation also cut reaction time
(Fig. 12), with mid-range antidromic values resembling position 2's accuracy.
What disambiguated them was not the endpoint but the STN activity time course
(Fig. 11: position 3 significantly reduced STN activity in the first
600 ms). N22 likewise separates tremor-like from rigidity-like outcomes not
by the endpoint (both slow the arm) but by STN spectrogram and synchrony
(Figs. 6iii–iv, 7).

**Evidence behind our choice:** the degeneracy demonstration is our own
model result, not an experimental measurement — but that is exactly its
force here: it is read-source evidence that scalar readouts underdetermine
DBS mechanism even in a model far smaller than BGM_22, and that a cheap
intermediate observable can break the tie. The healthy-vs-parkinsonian
synchrony contrast our observables lean on is experimentally grounded
(Bergman et al. 1998; Wilson & Bevan 2011, as cited in M&C16), though our
uses of it are qualitative.

**Why that would (or would not) be better here:** BGM_22 cannot put
electrophysiology into its loss — the Berlin dataset is BOLD — and we do not
propose it should. But the inference step ("read off which parameters had to
change") will face exactly the degeneracy structure we met, with a coarser
readout. Two additions stand on methodological ground alone: (a) when the §2
design runs, report the fitted models' STN/GPe population spectra and
synchrony per condition as a *diagnostic* (not a loss term) — parameter sets
indistinguishable in BOLD but different in oscillatory state should be
reported as unresolved rather than silently collapsed to the argmin; (b) the
spike recordings the rate probe already collects make this nearly free. As a
data note: `stim_settings.csv` lists the subject's IPG as "Percept", a device
family capable of chronic LFP recording **[from memory — unverified]**; if
beta-band LFP exists for this subject it would be an independent check worth
asking the Berlin group about, though we build no proposal on an unverified
device property. **Verdict: methodological deficiency** (mild — the plan in
§2 is right; the diagnostic reporting is the missing piece).

**Goal relevance:** high — this is the difference between "the fit chose
these parameters" and "the data determined them".

### 6.8 Rate-band provenance is exemplary for the striatum and absent for pallidum and thalamus

**What BGM_22 does:** `get_firing_rate_loss` scores 9 populations per loop
against `plausible_ranges`. The striatal bands are derived in an exemplary
audit trail (`experimental_data/activity_striatum/README.md`: Liang et al.
2008 med-off, SEM→SD reconstruction, stated assumption, sensitivity bounds,
what was rejected). The STN and SNr(GPi) bands carry a bare comment "from
[Li et al., 2015]" with no title, DOI, species or condition; the gpe_proto
(75–85 Hz), gpe_arky (15–20 Hz), gpe_cp (75–85 Hz) and thal (15–30 Hz) bands
carry no citation at all, in the code or in any document we read (`TODO.md`
tracks the gate *threshold* in §10, not the band provenance).

**What we do:** worse, in honesty: M&C16 validates no firing rates at all
(the striatal 2–40 Hz figure it uses for weight normalization cites a single
optogenetics study, Kravitz et al. 2010, with no derivation), and N22
validates DA concentration and behaviour but no BG population rates. We
cannot demand from BGM_22 a practice we lack.

**Evidence behind our choice:** not applicable — our lineage has no band
practice; this point rests entirely on BGM_22's own internal standard.

**Why that would (or would not) be better here:** the argument is
self-standing consistency, not our habit. The project has demonstrated, on
the striatal bands, exactly what a defensible band derivation looks like —
including the warning that a ±1 SD across-neurons band is a plausibility
window, not a confidence interval. The pallidal and thalamic bands enter the
same loss with equal weight (18 population scores averaged), gate the entire
BOLD evaluation, and are the narrowest in relative terms (75–85 Hz is ±6 %,
versus ±49 % for dSPN) — so they dominate the gate's behaviour precisely
where their provenance is weakest. A fit steered by an unsourced 75–85 Hz
window cannot be audited the way the project audits everything else. We
propose: write the derivation note for the six non-striatal bands to the
`activity_striatum` standard (species, condition — parkinsonian off-state or
healthy — and the identity of "Li et al., 2015" with DOI), or mark them
provisional in the code and widen them deliberately until sourced.
**Verdict: methodological deficiency** (documentation-grade, but with
gate-level consequences).

**Goal relevance:** medium — the bands steer the gate and the rate term of
every evaluation of both fits.

### 6.9 Medication state: explicit graded dopamine in our program versus φ = 0 here

**What BGM_22 does:** the striatal dopamine-modulation terms are inert
(`phi_1 = phi_2 = 0`, class defaults; `model_v07.md` §6.2–6.3), which in the
source model's convention means zero tonic dopamine — flagged by the project
itself as unchecked against the source paper (`TODO.md` §30), while the
striatal rate targets are deliberately the medication-off (not
dopamine-zero) state of Liang et al. 2008. The Roadmap separately requires
the subject's medication state during scanning to be established before any
full-length cache is built.

**What we do:** N22 treats parkinsonian dopamine as *graded and non-zero*:
SNc cell loss is simulated at 25/37/50/62/75 %, and the model's striatal
extracellular DA falls gradually (~264 nM control → ~148/154 nM at 25 % loss
→ ~13–51 nM at 75 %; Results, Figs. 6v), with the control value checked
against the measured 150–400 nM range (Schultz 1998, as cited). M&C16
handles medication as an added tonic term (δ_med = 2 on a clamped
δ_lim = −0.1) — hand-set values.

**Evidence behind our choice:** the only *measured* anchor in our practice is
the control DA range (Schultz 1998); the graded depletion curves are our
model's outputs, and δ_med/δ_lim are tuned. So our support for "PD
off-medication is not zero dopamine" is directional, not quantitative — and
we therefore do not propose our values.

**Why that would (or would not) be better here:** the project's own §30 plan
— read the Humphries source paper and extract what φ values it uses or
suggests for a dopamine-depleted state, then decide — is exactly right, and
its constraints paragraph (baseline capture, cache implications, pre-compile
setting) is more careful than anything in our own methods sections. What we
add is emphasis on ordering: our M&C16 results show medication state
*inverting* behavioural effects (PD-ON reversing PD-OFF learning biases,
Figs. 4–5), i.e. the dopamine level is not a small correction; and the
Roadmap's requirement that the subject's scan-time medication state be
established before cache builds is, in our view, the single most important
unresolved data question in the project — if the subject was medicated
during either scanning condition, the Liang off-state anchors and φ = 0 are
wrong together, in both conditions of the inference. **Verdict:
methodological deficiency** — already tracked in §30 and the Roadmap; the
plan and its ordering are adequate; we ask only that it not slip past phase 1.

**Goal relevance:** high — it conditions both fits and every rate anchor.

### 6.10 The DBS-on striatal surround is frozen at off-state behaviour

**What BGM_22 does:** 98 % of a simulated SPN's GABA input is an open-loop
stream drawn at fixed med-off rates (25/33/10.5 Hz), identical in the on and
off conditions except for the cortical rate file
(`experimental_data/input_streams/README.md` §4.1: "No DBS effect on the
surround"). Meanwhile the *simulated* putamen striatum does feel DBS — the
pallidostriatal projections (`gpe_arky__str_*`, `gpe_cp__str_*`,
`gpe_proto__str_fsi`) deliver the DBS-shifted pallidal activity through
ordinary synapses even though those projections carry no axonal DBS terms.
So under DBS-on the simulated neurons shift while the surround, standing for
the same tissue, cannot. `TODO.md` §4 plans a self-consistency check of
fitted rates against the surround assumption, after the first fit.

**What we do:** our models have no surround to freeze — everything is
simulated (M&C16) or rate-coded end-to-end (N22) — so this failure mode
cannot arise in our designs, and its absence there is a property of scale,
not a superior method. We built nothing at BGM_22's striatal realism.

**Evidence behind our choice:** not applicable; our alternative (simulate
everything) is not available at this project's fidelity and is not proposed.

**Why that would (or would not) be better here:** the limitation is known,
documented, and priced into the model's claim boundary (input_streams §5:
input gains and drive amplitudes are legitimate inference targets; emergent
circuit dynamics are not). Our addition is to the §4 plan, on self-standing
methodological grounds: the check must be run **per condition**. The
surround assumption is condition-independent while the fitted state is not,
so agreement in the off condition does not imply agreement in the on
condition — the pallidostriatal cascade above shifts the simulated putamen
SPN/FSI rates under DBS-on, and it is precisely the on-condition mismatch
that would contaminate the DBS inference. As currently worded, §4 compares
"the fitted rates" against one assumption; it should compare off-fit rates
and on-fit rates against it separately and report both gaps. **Verdict:
methodological deficiency** (a plan gap, one sentence to fix now, one extra
comparison to run later).

**Goal relevance:** medium — it decides whether the on-fit's striatal state
is internally consistent, which the inference implicitly assumes.

### 6.11 Condition discipline: same compiled network, guarded state, seeded search — beyond our own practice

**What BGM_22 does:** both DBS conditions compile the same network, differing
only in parameter values, verified once by byte-identical `report()` tables
(`model_v07.md` §10; `DBS.md`); `assert_dbs_state` re-checks the DBS state
immediately before every probe and BOLD run after the reset trap silently
disabled DBS once; seeds are fixed (common random numbers for CMA-ES) with a
planned multi-seed stability re-run of the winning vector (`TODO.md` §32);
and the RNG-stream hazard of the retrofit is documented with its measured
off-condition delta (§33 record, cited from `DBS.md`).

**What we do:** in M&C16, conditions differ by hand-edited weight constants
(w_Str→D1 4→3, w_STN→GPi 1.5→2 for PD-OFF) plus the added current; neither
of our papers reports RNG seeds, state assertions, or a same-structure proof
between conditions; variability is reported as SE over trials.

**Evidence behind our choice:** none — our practice here is simply the field
convention of our publication years, and it is weaker.

**Why that would (or would not) be better here:** BGM_22's discipline is the
correct standard for a difference-based inference — the claim "off and on
differ only in the values of the DBS parameters" is exactly what the
inference rests on, and this project has *measured* it rather than asserted
it. We record this as the project exceeding our lineage's own published
practice, and we intend to import the `assert_dbs_state` idea into our own
work. **Verdict: difference, not deficiency** — in BGM_22's favour.

**Goal relevance:** high — it is the precondition of the central claim.

### 6.12 The empirical per-condition cortical drive versus our synthetic cortex

**What BGM_22 does:** the cortical drive is the subject's own cortical BOLD,
deconvolved to firing rates per ROI and per DBS condition
(`model_creation_kwargs["dbs"]` selects the rate file and nothing else;
`DBS.md`). Consequently, whatever DBS did to cortex — including antidromic
cortical activation the model cannot generate (point 6.6) — is present
*implicitly in the input*, and the fitted DBS parameters capture only the
intra-BG portion of the effect; the caudate free control is designed to test
exactly this split.

**What we do:** we synthesize cortex — Poisson trains with learned weights
(M&C16) or a CANN motor cortex in a closed sensorimotor loop (N22). That
buys us closed-loop behaviour, which BGM_22 does not need (resting state),
and costs us any subject-specificity: our cortical input is the same
construction for every simulated "patient".

**Evidence behind our choice:** our cortical constructions are validated only
through downstream behaviour (Frank et al. 2004/2007; Majsak et al. 1998, as
cited); no element of them is any individual's data.

**Why that would (or would not) be better here:** for fitting one subject's
resting-state BOLD TR-by-TR, the empirical drive is the right instrument and
ours would be wrong. The cost — DBS's cortical action is folded into the
input rather than the parameters — is inherent and already documented; the
reading of the fitted DBS parameters as "intra-BG changes given the observed
cortex" should be stated verbatim in the write-up, since a casual reader of
"what DBS did" will otherwise assume totality. **Verdict: difference, not
deficiency** — in BGM_22's favour, with the framing caveat already implicit
in `DBS.md` made explicit at write-up time.

**Goal relevance:** high — it defines the scope of the inference's answer.

### 6.13 Scale and composition conventions

**What BGM_22 does:** 100 neurons per BG nucleus (`parameters.csv`,
`stn.size` etc.), a 1000-neuron striatal cube at measured density
(84 900 /mm³) with del Rey et al. 2022 type proportions and Kincaid-anchored
cortical pool sharing (`model_v07.md` §7); two loops with distinct,
tracer-derived cortical mixes (`cortical_proportions/README.md`).

**What we do:** 50×50 = 2500 neurons per nucleus in M&C16 and 15×15 = 225
(SNc 8×8) in N22, with no anatomical justification for either size in either
paper; no cell-type composition anywhere; no per-territory cortical mixes
(our M&C16 quadrants explicitly correspond to no territory).

**Evidence behind our choice:** none — our lattice sizes are conventions, as
stated in the standard section.

**Why that would (or would not) be better here:** nothing in our practice
licenses criticizing 100-neuron nuclei; our own sizes are equally
unconstrained, and BGM_22's striatal composition and density are anchored
where ours never were. The one place where size interacts with a measured
number — the 0.4 coverage applied to 100 STN neurons standing for a
functional territory — is covered in 6.1/6.2. **Verdict: difference, not
deficiency** — on the striatum, in BGM_22's favour.

**Goal relevance:** low.

### 6.14 v08 as the smoke test it claims to be

**What BGM_22 does:** v08 collapses the drive into one `TimedArray` per loop
and 100-neuron populations, carries the same DBS retrofit, and is documented
everywhere as a cache-free pipeline test only, with its known-broken drive
bounds tracked (`model_v08.md`; `TODO.md` §1).

**What we do:** our own programs likewise maintain reduced companions (the
M&C16 network is itself the reduced sibling of a larger program); we have no
published criterion for them beyond end-to-end executability.

**Evidence behind our choice:** none; convention.

**Why that would (or would not) be better here:** as a smoke test v08 is
adequate and honestly walled off — the documents repeatedly bar it from
carrying claims, and its one substantive use (the free-control demonstration
in §2's update, measured on v08) is labelled as such. We only note that any
number measured on v08 and cited in support of a v07 design decision (as the
free-control exactness was) should eventually be re-measured on v07, since
the two differ in precisely the striatal machinery the BOLD reads out.
**Verdict: difference, not deficiency.**

**Goal relevance:** low.

## What we would ask for before publication

1. Document the associative-STN coverage the free-control design rounds to
   zero (the subject's own ≈ 0.09 pooled overlap), and bound its effect on
   the caudate control analytically or with one sensitivity run (6.2).
2. Report identifiability diagnostics with the inference: fitted-model
   STN/GPe spectra and synchrony per condition, and the §2 degeneracy design
   executed before any mechanistic claim about what DBS did (6.7).
3. Write the provenance note for the six non-striatal firing-rate bands to
   the standard the striatal note already sets, or mark them provisional and
   widen them deliberately; identify "[Li et al., 2015]" (6.8).
4. Resolve the medication-state pair before full-length caches: the
   subject's scan-time state (Roadmap phase 1) and the φ = 0 check against
   the source paper (§30) (6.9).
5. Amend `TODO.md` §4 to run the missing-GABA self-consistency comparison
   per DBS condition, off-fit and on-fit separately (6.10).
6. State in the write-up: the sensitivity range for `population_proportion`
   from the per-hemisphere values [0.31, 0.50] (6.1); that fitted
   `axon_spikes_per_pulse` is defined with respect to the 100 µs model pulse
   (renormalize by 1.67 for the hardware's 60 µs before comparing across
   studies) and that the realized pulse train is identical for any width in
   (0, 100] µs at dt = 0.1 ms (6.5); the §15 hyperdirect scope bound (6.6);
   and the reading of the fitted DBS parameters as intra-BG changes given
   the observed cortical drive (6.12).
