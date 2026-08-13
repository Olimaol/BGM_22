# Seat 2 — Rubin / Verstynen (Pittsburgh / CMU) — DBS standing, panel seed

Review of BGM_22 written in character as the Rubin–Verstynen lineage, applying
that lineage's own published standards unfiltered. Part of the
community-conventions survey defined in `TODO.md` §34; the panel and the reading
list are in `README.md`.

**Read for this review (full texts, from the PDFs in this directory):**

- Corbit VL, Whalen TC, Zitelli KT, Crilly SY, Rubin JE, Gittis AH (2016),
  J Neurosci 36(20):5556–5571, `10.1523/JNEUROSCI.0339-16.2016`
- Rubin JE (2017), Curr Opin Neurobiol 46:127–135, `10.1016/j.conb.2017.08.011`
- Dunovan K, Vich C, Clapp M, Verstynen T, Rubin J (2019), PLOS Comput Biol
  15(5):e1006998, `10.1371/journal.pcbi.1006998`
- Clapp M, Bahuguna J, Giossi C, Rubin JE, Verstynen T, Vich C (2025), PLOS ONE
  20(1):e0310367, `10.1371/journal.pone.0310367`
- Giossi C, Rubin JE, Gittis A, Verstynen T, Vich C (2024), Eur J Neurosci
  60(10):6129–6144, `10.1111/ejn.16348`

**What was reviewed:** the same project documents as seat 1 —
`model_v07.md`, `model_v08.md`, `DBS.md`, the three `experimental_data/`
READMEs, and `BOLD_optimization/get_loss.py`.

Every point carries a **goal-relevance** tag against the project's stated goal
(single-subject resting-state BOLD fitting and DBS inference).

---

## The standard this seat applies

Rubin 2017 is this seat's explicit statement of what a BG dysfunction model owes
its reader, and three of its demands bear directly on BGM_22:

1. **Know what your model's observable actually is.** On simulated LFPs: "Simulated
   LFPs signal clearly depend on which model outputs are used to compute them"
   (Fig. 2 shows voltage-derived and synaptic-current-derived LFPs from the *same*
   model giving different spectra), and the paper concludes it is important "to
   characterize better what the LFPs measured in the BG really represent."
2. **Simplification is legitimate, but the omitted detail must be argued, not
   assumed away.** "Biological details, such as particular ionic currents,
   patterns of synaptic connections, and synaptic delays, may significantly
   affect model dynamics. A balance between detail and abstraction is needed."
   The paper is pointed about integrate-and-fire-class models: they "cannot
   capture critical contributions of particular currents to neuronal spike
   patterns," and "caution should be used with predictions from firing-rate or
   integrate-and-fire models until they can be linked with plausible biophysical
   mechanisms that can be tested in subsequent experiments."
3. **Connectivity structure, not just connection existence.** "The spatial spread
   and density of synapses from individual neurons, or the prevalence of local
   coupling structures (motifs), can strongly affect network dynamics" — citing
   Rosenbaum et al. 2016 on the spatial structure of correlated variability.

Corbit 2016 and Giossi 2024 add the seat's GPe standard; Dunovan 2019 and
CBGTPy add the seat's standard for what a complete CBGT network contains and how
free parameters are handled.

**Where we start from agreement.** BGM_22's `I_CBF` mapping is the *synaptic
current*, which is the observable our own Fig. 2b endorses over voltage; the
model's three GPe populations reflect exactly the reorganisation Giossi 2024
argues the field must absorb; and the documentation's own honesty about what the
streams cannot do (`input_streams/README.md` §4–§5) is the kind of statement we
wish more submissions contained. The points below are where the model departs
from our standards, or where a stated caveat needs to become a constraint on the
claims.

---

## Points

### 2.1 The model has arkypallidal neurons and does not use them for anything
arkypallidal

Giossi 2024 — this seat's own review — argues the GPe is a bidirectional hub, not
a way-station: descending information via prototypical cells to STN and GPi, and
**ascending** information from STN and cortex, relayed *up* to striatum by
arkypallidal cells, which is what allows reactive and proactive control to
interact. BGM_22 has the anatomy for this (`gpe_arky → str_d1/str_d2/str_fsi`)
but the ascending limb cannot function as an ascending limb, for a specific
reason: the model's striatum is 98 % driven by precomputed streams
(`input_streams/README.md` §4.1), so a change in arkypallidal firing modulates
the ~2 % of striatal input that is simulated. The ascending pathway is present
in the wiring diagram and numerically negligible in the dynamics.

This is a stronger objection than "the model is simplified". The GPe→striatum
projection is precisely the one Corbit 2016 found to be *functionally strong* —
we recorded GPe-evoked IPSCs in FSIs "more than six times as strong" as in MSNs
in the mean and maximal responses, and stronger and more frequent under dopamine
depletion — and it is the pathway on which our whole pallidostriatal β mechanism
rests.

**Goal relevance: medium–high.** For the BOLD fit, the GPe→striatum weight
cluster (`gpe_striatum`) is one of the 19 fitted parameters; if the pathway
cannot influence striatal firing against the stream background, that parameter
is weakly identified and its post-fit change should not be interpreted.

### 2.2 The pallidostriatal loop, which this lineage identified as a β generator,
is open in this model

Corbit 2016's result is a *loop*: GPe→FSI→MSN→GPe, with the model developing
significant β power under dopamine depletion, and with the mechanism running
through FSI pauses translating GPe synchrony into MSN synchrony. In BGM_22 the
return limb is severely weakened in two independent ways. First, FS neurons
receive GABA only from other FS neurons — `fitted_params.json` has no SPN→FS
pair, so the FSI population has essentially no striatal inhibitory input
(`input_streams/README.md` §4.1). Second, the MSN→GPe limb exists but the MSN
firing it carries is dominated by the open-loop drive.

We would not ask this project to reproduce β — the BOLD observable cannot see it.
We would ask that the documentation state that **no oscillation-generating loop
in the model is closed**, so that any DBS-related change the fit reports cannot be
attributed to a change in oscillatory dynamics.

**Goal relevance: low for the fit, high for the interpretation.**

### 2.3 `gpe_cp` is in the model because of a projection the model does not contain

We read Goenner et al. 2021 (this project's own lineage, our reference [24] in
CBGTPy) to establish what `gpe_cp` denotes: cortex-projecting GPe neurons, the
Npas1⁺-Nkx2.1⁺ population of Abecassis 2020, whose entire functional role in that
model is to close a **cortico-pallido-cortical loop** — in Goenner's Table 4,
GPe-Cp's efferents are the striatal populations, the other GPe populations, *and*
the Integrator-Stop that suppresses the cortical Go input.

BGM_22 has no cortex. The cortical drive is a precomputed stream that nothing in
the model can influence (`model_v07.md` §7.5, §8). So the population is retained
with its striatal and intra-pallidal projections while the projection that
motivated its existence is structurally absent. It is now a duplicate of
`gpe_arky` in connectivity type — both project to all three striatal populations
and to the other two GPe populations — differing only in its Izhikevich
parameters (identical to `gpe_proto`'s, per `parameters.csv`) and in receiving
`stn` input.

Note the interaction with the firing-rate gate: `gpe_cp` is scored against the
same (75, 85) Hz band as `gpe_proto` while carrying arkypallidal-type
connectivity. Giossi 2024 records the rodent numbers as prototypical 10–100 Hz
(mean ~55) and arkypallidal 1–30 Hz (mean ~10). Whatever `gpe_cp` is, the model
asserts it fires like a prototypical cell and wires like an arkypallidal one.

**Goal relevance: medium.** Three GPe populations pooled into one BOLD ROI with
fixed weights; the fitted `gpe_striatum` cluster spans all seven of their
striatal projections, so the redundancy is absorbed into one number.

### 2.4 Every neuron in the model is an Izhikevich point neuron, and this seat's
position on that class of model is on the record

Rubin 2017 states it directly: integrate-and-fire-class models "cannot capture
critical contributions of particular currents to neuronal spike patterns", and
for STN specifically the T-type calcium current is "crucial for parkinsonian STN
bursting and sets the interburst frequency" while HCN "opposes STN bursting".
Our own STN–GPe and pallidostriatal work is conductance-based for this reason;
CBGTPy uses integrate-and-fire, and we said in its Discussion that this is a
deliberate trade for extensibility and scale.

We do not think this sinks BGM_22 — a BOLD fit at 2.31 s resolution has no
access to burst structure — but two consequences follow that should be written
down. First, the model cannot support any claim about DBS acting through STN
burst suppression, which is one of the main mechanistic hypotheses in the field.
Second, `DBS.md`'s `antidromic` reset (a full Izhikevich `v → c; u += d` fired by
a pulse, without threshold crossing and without refractoriness) is a
*phenomenological* stand-in for somatic invasion whose effect on a
conductance-based STN would differ; the value fitted for it is a fitted number,
not a measured antidromic efficacy.

**Goal relevance: medium.** Bounds the claims, not the fit.

### 2.5 The BOLD observable is a modelling choice presented as a given

This is our point 1 above, applied. `get_loss.py` maps `I_CBF` to `I` for the BG
populations and to `I_v` for the striatal ones, and `model_v07.md` §3.6 records
three consequences of that choice that we think are under-argued:

- for the three GPe populations the quantity actually entering `dv/dt` is
  `f(I, nonlin)` (the compressed input), but the monitor takes the **raw** `I`;
- `I_base` — the entire drive of `snr` and `gpe_proto`, which receive no cortical
  input — sits outside `I` and is therefore invisible to the monitor;
- the GPe pooling factors are uncited and sum to 0.77.

Our Fig. 2 makes the general point with two LFP proxies from one model; here the
proxy differs *between populations within one ROI*, and one population's dominant
drive term is excluded by construction. At minimum this deserves the sensitivity
analysis we would demand of an LFP proxy: refit with `I_base` included and report
whether the fitted vector moves.

**Goal relevance: high.** This is the definition of the model's only output.

### 2.6 Nineteen free parameters, one 310-point target, and no
parameter-identifiability analysis

CBGTPy's Discussion says plainly what we believe about this: "Spiking network
models like those used in CBGTPy have an immense number of free parameters. The
nature of both the scale and variety of parameters in spiking neural networks
makes the fitting problem substantially more complex than that faced by more
abstracted neural network models... To the best of our knowledge, there is no
established solution to simultaneously fitting both constraints together in these
sorts of networks."

BGM_22 goes further than we have: it fits 19 parameters by CMA-ES against a
single subject's BOLD correlation plus a rate term. Two structural risks follow,
and neither is currently addressed in the documentation.

First, **the drive weights and the weight-cluster scalings are not obviously
separable.** Parameters 0–6 scale the streamed input counts; parameters 9–18
scale the recurrent weights. In a network where 98 % of striatal input is
streamed, a change in a `gpe_striatum` scaling and a compensating change in the
dSPN drive weight will produce nearly the same striatal output. The reported
"which parameters had to change under DBS" is a statement about a ridge in
parameter space unless the ridge is characterised.

Second, **CMA-ES returns one point.** The inference in `TODO.md` §2 asks what DBS
changed; that requires a posterior or at least a confidence region, not a single
best vector from each condition. We would point to the approach cited in Rubin
2017 ([51,52], probabilistic parameter estimation) as the accepted way to make
this kind of claim.

**Goal relevance: high.** This is the inference itself.

### 2.7 DBS is modelled without cortical fibre activation, which is the mechanism
the field most argues about

`DBS.md`'s own limitation 2 states it: the hyperdirect cortical afferent to STN is
a `TimedArray`→`CurrentInjection` with no soma, so `afferents=True` in practice
means `gpe_proto→stn` only, and "cortical fibre activation — a real and
much-discussed DBS effect — is not represented in this model."

We want to underline how much this costs, because this seat carries the panel's
DBS standing. Antidromic activation of corticosubthalamic fibres is a leading
candidate mechanism for STN-DBS's effect; it is what Kumaravelu 2018 (seat 4)
decomposes and what Mandali & Chakravarthy 2016 (seat 6) put in their title. A
model that fits DBS-on by moving three parameters, none of which can express
cortical antidromic drive, will attribute whatever DBS did to the mechanisms it
*can* express — the STN somatic shunt, the orthodromic axon volley, and the
passing fibre. That is not a neutral limitation; it is a prior on the answer.

**Goal relevance: high.** Directly shapes the inference's conclusion.

### 2.8 Short-term synaptic plasticity is absent, and this seat regards it as
central specifically for DBS

Rubin 2017's DBS section: high-frequency axonal drive may cause "massive
neurotransmitter release" such that synapses "assume a state in which subsequent
release becomes coupled to stochastic reuptake events, rather than to the
transmission of potentially pathological patterns of spiking through the BG"; a
model combining stochastic short-term plasticity with axonal failure reproduced
observed changes in GPi rates and timing and in GPi–GPi and STN–GPi coherence.
The section closes by noting the need for experimental work on "how short-term
synaptic plasticity may affect information flow through these populations even in
baseline conditions."

BGM_22 delivers the DBS volley through static synapses at 125 Hz. Under the
mechanism above, that is the one regime where a static synapse is most wrong: the
model will transmit every pulse at full weight where a real synapse would deplete.
The fitted `axon_spikes_per_pulse` will therefore absorb the depletion as a lower
per-pulse probability, which is a defensible modelling choice but must not then
be read as a measurement of axonal recruitment.

**Goal relevance: high for the DBS half of the inference.**

### 2.9 Correlated variability has spatial structure, and here it is imposed
rather than derived — except where it is derived and then switched off

Rubin 2017 cites Rosenbaum et al. 2016 for the point that "synaptic connection
profiles influence the spatial pattern of pairwise neural correlations". BGM_22's
missing-GABA construction does exactly the right thing — realise the source pool
geometrically and let `f(d)` emerge (`model_v07.md` §7.3) — and we single this out
as the best-engineered part of the input machinery.

But the cortical streams do not: `ci.shared_fraction_dict` is 0 for all four BG
populations, so neighbouring STN neurons receive fully independent cortical drive
(`input_streams/README.md` §4.6), and the striatal cortical pool has one flat
Kincaid fraction with no distance dependence (§4.7). Meanwhile
`mc.correlation_dict` and `mc.cortical_correlation` are 0. The model's own §3
computes that this choice moves the BOLD variance by up to a factor of 481.

**Goal relevance: high.** Same conclusion as seat 1 reached independently: the
`TODO.md` §25 scan is a prerequisite for the fits, not a follow-up.

### 2.10 What we would take from this project

Two things in BGM_22 exceed our own practice and should be said.

The **input-stream contract** (`experimental_data/input_streams/README.md`) is a
document we do not have an equivalent of: a written statement of what the
surrogate input must reproduce, checked in code at build time, with the
closed-form target statistics and a recorded history of the bug that motivated it.
CBGTPy's external input is Poisson at a tuned rate, and we do not check it against
anything.

The **subject-specific drive** is likewise a step past our design. Our cortical
input is a tuned Poisson rate; here the drive is the same subject's own cortical
BOLD, deconvolved, so the model is asked to track that individual's basal ganglia
rather than a population average.

We record this partly to sharpen point 2.6: a stronger, more falsifiable target
raises the value of the fit and simultaneously raises the cost of not knowing how
well the parameters are determined.

---

## What we would ask for before publication

1. A statement, in the model documentation, that no oscillation-generating loop in
   the model is closed and that the GPe→striatum ascending pathway acts on ~2 % of
   striatal input (points 2.1, 2.2).
2. A sensitivity analysis of the BOLD observable — at minimum, `I` vs `I` + `I_base`
   for `snr`/`gpe_proto`, and raw vs compressed `I` for the GPe populations — plus
   a source or a replacement for the GPe pooling factors (point 2.5).
3. A parameter-identifiability analysis: which of the 19 parameters the loss
   actually constrains, and which trade off against which (point 2.6). Without it,
   we would not accept a "these parameters changed under DBS" conclusion.
4. An explicit statement that cortical fibre activation is outside the model's DBS
   footprint, positioned where the DBS results are reported, not only in `DBS.md`
   (point 2.7).
5. Either a justification of `gpe_cp` in a model without cortex, or its removal
   (point 2.3).
6. The input-correlation scan before the fits (point 2.9).
