# Seat 4 — Grill (Duke) — DBS standing

Review of BGM_22 written in character as the Grill lineage, applying that
lineage's own published standards unfiltered. Part of the community-conventions
survey defined in `TODO.md` §34; the panel and the reading list are in
`README.md`. This seat reviews the DBS representation with the most standing on
the panel.

**Read for this review (full texts, from the PDFs in this directory):**

- Kumaravelu K, Brocker DT, Grill WM (2016), J Comput Neurosci 40(2):207–229,
  `10.1007/s10827-016-0593-9`
- Kumaravelu K, Oza CS, Behrend CE, Grill WM (2018), J Neurophysiol
  120(2):662–680, `10.1152/jn.00862.2017`
- Su F, Kumaravelu K, Wang J, Grill WM (2019), Front Neurosci 13:956,
  `10.3389/fnins.2019.00956`

**What was reviewed:** `DBS.md` in full, plus `model_v07.md`, `model_v08.md`,
the three `experimental_data/` READMEs, `CompNeuroPy`'s `dbs.py` as described in
`DBS.md`, and `BOLD_optimization/get_loss.py`.

Every point carries a **goal-relevance** tag against the project's stated goal
(single-subject resting-state BOLD fitting and DBS inference).

---

## The standard this seat applies

1. **A network model earns the right to be used for DBS by being validated
   first, on responses it was not tuned to produce.** Kumaravelu 2016 §3
   validates against Kita & Kita 2011's cortically evoked PSTHs in striatum, STN,
   GPe and GPi — the shape of the early excitation / short inhibition / late
   excitation sequence, separately in control and 6-OHDA — *and* against firing
   rates in both states, *and* the low-frequency oscillations emerge rather than
   being imposed. Only then is DBS applied. Su 2019 reuses that model explicitly
   "because it replicated a wide range of electrophysiological data from the
   unilateral 6-OHDA lesioned rat model of PD."
2. **The DBS result that counts is the frequency–response curve.** Our central
   finding is that STN DBS below 40 Hz does not reduce low-frequency GPi
   oscillatory power, that power falls gradually between 50 and 130 Hz, and that
   the effect saturates above 150 Hz (Kumaravelu 2016 Fig. 11E) — matching the
   clinical and behavioural frequency dependence. A single-frequency DBS
   simulation is not a DBS model result; it is one point on a curve whose shape
   is the actual claim.
3. **Which pathway carries the effect is an empirical question with an answer.**
   Kumaravelu 2018 decomposed the cortical evoked potential and found the
   short-latency R1 arises from **antidromic activation of layer-5 pyramidal
   axons** — the corticosubthalamic (hyperdirect) fibres — and that the R2/R3
   components follow from intracortical and cortico-thalamo-cortical
   interactions, with orthodromic propagation through STN→GPi→thalamus→cortex
   **not required**.
4. **Robustness before interpretation.** Kumaravelu 2016 randomised the three
   parameters that define the PD state and re-ran with stochastic connectivity,
   ten trials each, and showed the peak oscillatory frequencies were unchanged
   (their Fig. 8D–I).

We note where BGM_22 already meets us. The delays are ours.

---

## Points

### 4.1 The delays are this lineage's delays, and that resolves an open question
in this review

Placing `parameters.csv` beside Kumaravelu 2016 Table 1:

| pathway | BGM_22 | Kumaravelu 2016 | our source |
|---|---|---|---|
| dStr → GPi/SNr | 4 | 4 | Nakanishi et al. 1987 |
| idStr → GPe | 5 | 5 | Kita & Kitai 1991 |
| STN → GPi/SNr | 1.5 | 1.5 | Nakanishi et al. 1987 |
| STN → GPe | 2 | 2 | Kita & Kitai 1991 |
| GPe → STN | 4 | 4 | Fujimoto & Kita 1993 |
| GPe → GPi | 3 | 3 | Nakanishi et al. 1991 |
| GPi → TH | 5 | 5 | Xu et al. 2008 |

Seven of seven match exactly. So BGM_22's delays are not unsourced, as they
appear to be from the CSV alone — they are the rat delay set of this lineage.
Two consequences. The provenance should be written into `parameters.csv` or
`model_v07.md` §5, because at present nothing in the project records it. And the
species should be stated: these are rat values, in a model fitted to a human
subject, and seat 3 has tabulated Liénard et al. 2024's macaque estimates, which
differ from ours by up to a factor of two on the striatofugal pathways. That
disagreement is between the source literatures, not something BGM_22 introduced.

**Goal relevance: low.** At a 2310 ms TR the delays cannot move the fit. We
raise it because provenance is cheap to record and because the model is
otherwise a rat/primate/human chimera whose species mix is nowhere stated
(point 4.7).

### 4.2 DBS is simulated at one frequency, so the model's DBS behaviour has no
shape to check

`DBS.md`: `dbs_pulse_frequency_Hz = 125`, from Berlin subject 1, fixed. There is
no frequency sweep anywhere in the project.

This is the single most important thing missing from the DBS half of BGM_22, and
it is also the cheapest to add. The frequency–response of STN DBS is the one
model-level DBS prediction that is quantitatively established, in rats
(Kumaravelu 2016 Fig. 11; So et al. 2012b; Li et al. 2012; Ryu et al. 2013) and
in patients (frequencies above ~100 Hz relieve symptoms, below ~50 Hz usually do
not; Fogelson et al. 2005; Timmermann et al. 2004). A model of DBS that responds
to 20 Hz exactly as it responds to 125 Hz has not represented DBS; it has
represented a constant.

BGM_22 can run this test without changing anything structural: take the fitted
DBS-on vector, sweep `dbs_pulse_frequency_Hz` from 5 to 200 Hz, and report the
model's BOLD change and its GPi/STN rate change against frequency. If the curve
is flat, or monotone with no low-frequency floor, the three fitted DBS parameters
are not doing what DBS does. We would regard this as a *precondition* for
publishing a DBS inference, not as a supplementary figure.

Note that the model's own construction predicts a specific failure here. The
somatic shunt is proportional to `pulse(t)`, which is 1 for one timestep every
8 ms, and the axon spikes are drawn per pulse at a fixed probability. Both
effects are therefore very nearly *linear in frequency* by construction, with no
mechanism producing the saturation above 130 Hz that the data show and no
mechanism producing the ineffectiveness below 40 Hz. In our model the saturation
comes from the interaction of the pulse train with the intrinsic firing rate of
the stimulated neurons — the stimulation must be fast enough to mask the neuron's
own activity — and the low-frequency floor comes from the network having time to
recover between pulses. Neither mechanism is excluded by BGM_22's architecture,
so the sweep is a genuine test rather than a foregone conclusion.

**Goal relevance: high. This is the DBS half of the project.**

### 4.3 The one DBS pathway with a direct link to symptom relief is the one the
model cannot represent

`DBS.md` limitation 2 states that the hyperdirect cortical afferent to STN is a
`TimedArray`→`CurrentInjection` with no soma, so it is outside the DBS footprint
and `afferents=True` means `gpe_proto→stn` only.

We want to put on the record how much weight that pathway carries in the current
evidence, because "much-discussed" understates it:

- Kumaravelu 2018 established by model-based decomposition that the R1 cortical
  evoked potential — the earliest, most robust, and most stimulation-locked
  cortical signature of STN DBS, present in rats and in humans (our Fig. 7) —
  arises from antidromic activation of L5 pyramidal axons, confirming the
  collision-test result of Li et al. 2007.
- Li et al. 2012 found a significant correlation between the probability of
  antidromic spikes in the hyperdirect pathway and the relief of parkinsonian
  symptoms in 6-OHDA rats.
- Gradinaru et al. 2009 reversed parkinsonian motor symptoms by optogenetically
  driving M1 layer-5 projection neurons directly.
- Our decomposition further showed that the *orthodromic* route
  STN→GPi→thalamus→cortex was **not necessary** to generate the evoked cortical
  response.

BGM_22 represents the orthodromic efferent volley and the antidromic invasion of
the STN soma, and cannot represent antidromic invasion of the cortical afferent.
So the model's DBS repertoire is, in our terms, the pathway our data say is not
required, plus a somatic effect, minus the pathway with the symptom correlation.

We are not asking for cortical neurons to be added — that is a different model.
We are asking that the DBS-on/off parameter differences be reported with an
explicit statement that they are the DBS effects *expressible in this model*, and
that a fitted `axon_spikes_per_pulse` is not an estimate of DBS recruitment in
the patient.

**Goal relevance: high. It is a prior on the inference's conclusion.**

### 4.4 Antidromic transmission fails at high frequency, and a fixed per-pulse
probability at 125 Hz cannot express that

Kumaravelu 2018, reporting Li et al. 2012: antidromic spike propagation in
hyperdirect-pathway axons "was faithful only for low stimulation frequencies";
we invoke exactly this to explain why the R1 amplitude falls at 130 Hz relative
to 4.5 and 9 Hz (our Fig. 4B1, P = 0.011). The same paper's Fig. 6 shows the
130 Hz response can be reconstructed by summing time-shifted low-frequency
responses — i.e. the per-pulse response is *not* independent of the pulse train.

BGM_22's `axon_spike = pulse(t)*dbs_on*unif_var_dbs1 > 1-prob_axon_spike` draws
independently at every pulse with a constant probability, and `DBS.md` records
that `axon_spikes_per_pulse` *is* that per-pulse probability at this project's
numbers. There is no history dependence, no refractory interaction, and no
frequency dependence. Combined with point 4.2, this means the fitted probability
will take whatever value best matches the BOLD at 125 Hz, and cannot be read as a
physiological recruitment fraction at any frequency.

**Goal relevance: high for the interpretation of the fitted DBS parameters.**

### 4.5 The VTA coverage and the per-pulse probability are the same parameter
twice

`DBS.md`: `population_proportion = 0.4` sets `dbs_on = 1` on 40 % of
`stn:putamen`, and `axon_spikes_per_pulse ∈ [0, 1]` is fitted. The axon-spike
condition multiplies both, so the expected number of DBS-evoked STN axon spikes
per pulse is `0.4 × p` over the population. Nothing in a BOLD time course can
separate the two factors — a fitted `p = 0.5` at 40 % coverage is
indistinguishable from `p = 0.25` at 80 % coverage.

That would be unremarkable if the 0.4 were solid, but `DBS.md` limitation 3
already flags it as an "unvalidated single-subject value" — `(35+23)/(70+75)`
from one VTA estimate. So a fitted quantity is being reported against a fixed
factor with unstated uncertainty. Our practice is the opposite: we hold the
recruitment deterministic (every pulse evokes one action potential in every
stimulated model neuron; Kumaravelu 2016 §2.6) and vary the *proportion* of
elements activated explicitly, in steps, as the free variable — Kumaravelu 2018
Fig. 11B sweeps the activated L5 axon fraction from 20 % to 60 % and reports how
the response components change with it.

Concretely, we would recommend fixing `axon_spikes_per_pulse = 1` and fitting the
VTA proportion instead, or fitting the product and reporting it as one number.

**Goal relevance: high.** One of the three DBS parameters is currently
unidentifiable by construction.

### 4.6 The somatic DBS term is hyperpolarising and has no cited basis

`DBS.md` documents the term honestly:
`+ pulse(t)*dbs_on*dbs_depolarization*neg(-90 - v)`, a shunting pull toward
−90 mV, "despite the parameter being called `dbs_depolarization`", inert below
−90 mV.

In our models DBS is a depolarising intracellular current pulse (300 µA/cm²,
0.3 ms) applied to every STN neuron so that each pulse evokes one action
potential — the opposite sign, and directly motivated by the fact that
extracellular stimulation near a soma drives it above threshold. A
hyperpolarising somatic term is not unmotivated in the literature (depolarisation
block and stimulation-evoked GABAergic input have both been proposed), but
`DBS.md` cites nothing for it, and it is one of only three fitted DBS parameters.

The interaction with `neg()` deserves a second look too. `dbs_depolarization`
has bounds [0, 10] and multiplies `(-90 - v)`; for an STN neuron sitting near
−60 mV that is a 30 mV driving force, so at the upper bound the term is 300 units
on the `dv/dt` line of an Izhikevich-2003 model whose entire quadratic term at
that voltage is of order 100. We would want a plot of what the fitted value
actually does to the STN membrane trajectory, at the fitted value, before
believing any interpretation of it.

**Goal relevance: high.** It is a third of the DBS parameter vector.

### 4.7 The model is a species chimera, and the DBS frequency window depends on
which species you are in

Collecting what the review has established: the delays are rat (point 4.1); the
striatal firing rates are MPTP macaque, medication-off (Liang et al. 2008); the
FS rate is normal macaque times a rodent depletion factor; the cortical
proportions are macaque tracer counts; the GPe rate bands are primate-like at
75–85 Hz; the striatal cell-type proportions are del Rey et al. 2022; the subject
and the target data are human; the DBS frequency is the human clinical 125 Hz.

Kumaravelu 2016 §5.2 makes the reason this matters concrete: firing rates of STN,
GPe and GPi in rats are much lower than in non-human primates in both normal and
parkinsonian conditions, and *"the differences in firing rates likely underlie
the variations in the frequency-dependent effects of DBS between the animal
models. While low frequency stimulation (~50 Hz) was sufficient to mask and
regularize the intrinsic activity of a model neuron firing at a low rate, higher
frequency stimulation (>100 Hz) was necessary to achieve similar effects in a
neuron that fired at a higher rate."*

So the DBS frequency window is a function of the target's firing rate, and
BGM_22 sets its firing rates from one species mix and its DBS frequency from
another. The model may well be fine — 125 Hz against 75–85 Hz GPe rates is the
primate-consistent combination — but the reasoning has not been done anywhere in
the documentation, and the rat delays sit inside the same circuit.

**Goal relevance: medium.** Mostly a documentation and interpretation problem,
but it becomes a real one the moment point 4.2's frequency sweep is run, because
the sweep's shape depends on exactly this.

### 4.8 The cortical drive already contains the DBS effect, which is a strength
and a confound at once

This is the point we would most want the authors to think about, and we did not
find it discussed anywhere in the documentation.

`model_creation_kwargs["dbs"]` selects the cortical rate file:
`firing_rates_matlab_condition-on.npz` versus `-off.npz`, deconvolved from the
same subject's cortical BOLD in the two conditions
(`experimental_data/cortical_proportions/README.md`, `model_v07.md` §2). So the
DBS-on model is driven by cortex *as it was under DBS*, and the DBS-off model by
cortex as it was without.

The strength: DBS's well-documented cortical effects — which this model cannot
generate internally (point 4.3), and which in our decomposition are the dominant
route to cortex — enter the model empirically, through the subject's own
recording. That is a genuinely elegant way around a structural limitation, and it
is closer to the truth than any model-internal cortical loop we have built.

The confound: it means the DBS-on and DBS-off models differ in their *inputs*
before any DBS parameter is set. Whatever part of the measured BG BOLD change is
explained by the changed cortical drive is not available to be explained by the
three DBS parameters, and the fitted DBS parameters are therefore estimates of
the *residual* DBS effect after the cortical route has been accounted for. That
is a defensible and even desirable definition — but it is a different quantity
from "what DBS did inside the basal ganglia", which is how `CLAUDE.md` states the
project's goal.

There is a clean way to quantify it, and we would ask for it: evaluate the
DBS-off fitted parameter vector with the DBS-on cortical drive and the DBS
parameters at zero. The resulting BOLD change is the part of the on-vs-off
difference that the cortical drive alone produces. `DBS.md` already identifies
the caudate loop as "the free control" on exactly this logic; the caudate carries
no DBS terms at all, so its entire on-vs-off change *is* this quantity, and it is
measurable in the same run.

**Goal relevance: high. It bears directly on what the inference means.**

### 4.9 No robustness analysis, and the DBS parameters are the place it matters
most

Kumaravelu 2016 Fig. 8D–I: we randomised the three PD-defining conductances over
a uniform range and re-ran with stochastic connectivity, ten trials, and showed
the oscillatory peak frequencies did not move while the spectral power magnitude
did. `TODO.md` §16 tells us a DBS-constant sensitivity analysis is planned. We
would raise its priority: the three DBS parameters are the entire product of the
DBS-on fit, one of them is unidentifiable by construction (point 4.5), one has no
cited mechanism (point 4.6), and one cannot express its known frequency
dependence (point 4.4).

**Goal relevance: high.**

---

## What we would ask for before publication

1. The DBS frequency sweep, 5–200 Hz, on the fitted DBS-on vector, with the
   resulting curve compared against the established saturation above ~130 Hz and
   ineffectiveness below ~40 Hz (point 4.2). This is a precondition, not a
   supplement.
2. Either fix `axon_spikes_per_pulse` at 1 and fit the VTA proportion, or fit and
   report the product; the two cannot both be free (point 4.5).
3. A citation for the hyperpolarising somatic term, and a membrane-trajectory
   figure at the fitted value (point 4.6).
4. The cortical-drive-only control quantified — the caudate loop's on-vs-off BOLD
   change, and a DBS-off vector evaluated against the DBS-on drive — reported
   alongside every DBS parameter difference (point 4.8).
5. A sentence in the DBS results stating that antidromic activation of the
   corticosubthalamic pathway, which carries the strongest evidence linking DBS
   to symptom relief, is outside the model (point 4.3).
6. The delay provenance recorded, with the species stated (point 4.1).
