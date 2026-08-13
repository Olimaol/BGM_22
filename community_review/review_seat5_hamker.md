# Seat 5 — Hamker (TU Chemnitz) — our own lineage

Review of BGM_22 written in character as the Hamker/Chemnitz lineage, judging
the model **against the bar this lab set in print**. Part of the
community-conventions survey defined in `TODO.md` §34; the panel and the reading
list are in `README.md`. Per §34 this seat exists to ask one question: does this
model meet the standards its own lab has already published?

**Read for this review (full texts, from the PDFs in this directory):**

- Schroll H, Hamker FH (2016), Mov Disord 31(11):1591–1601, `10.1002/mds.26719`
- Maith O, Villagrasa Escudero F, Dinkelbach HÜ, Baladron J, Horn A, Irmen F,
  Kühn AA, Hamker FH (2021), Eur J Neurosci 53(7):2278–2295, `10.1111/ejn.14868`
- Goenner L, Maith O, Koulouri I, Baladron J, Hamker FH (2021), Eur J Neurosci
  53(7):2296–2321, `10.1111/ejn.15082`

A fourth slot, Meier et al. 2022, was removed from this seat's list at the
download checkpoint (see `README.md`); the consequence — that the panel now has
no in-house auditor of the simulated-BOLD side at all — is recorded there and
picked up in point 5.9 below.

**What was reviewed:** `model_v07.md`, `model_v08.md`, `DBS.md`, the three
`experimental_data/` READMEs, `TODO.md`, and `BOLD_optimization/get_loss.py`.

Every point carries a **goal-relevance** tag against the project's stated goal
(single-subject resting-state BOLD fitting and DBS inference).

---

## The standard this seat applies

Maith et al. 2021 is the direct predecessor of this project — same lab, same
first author, same simulator, the same class of question. It is therefore the
sharpest available yardstick, and it published four practices that BGM_22 has
not carried forward:

1. **Twenty optimisation runs per fitted unit, with different random
   initialisations, and the lowest-loss solution selected** (§2.4).
2. **A sensitivity analysis over the parameters that were *not* fitted** — each
   of 59 predefined neuron parameters varied from −5 % to +5 % in 0.3125 %
   increments, across all 70 fitted models, with the loss sensitivity summarised
   as a per-parameter regression slope (§2.6, Fig. 7). The conclusion mattered:
   the cortical and thalamic `n0`/`n1`/`n2` parameters dominated the loss.
3. **A group-level statistical test as the vehicle of the inference** — 30
   control and 40 patient models, two-tailed *t* tests with FDR correction and
   Cohen's *d* on every connection strength and firing rate (Tables 5, 6).
4. **A discrimination check that the fits captured group-specific structure** —
   control models fit the control mean FC significantly better than the patient
   mean FC, and vice versa (Table 4), before any parameter difference was
   interpreted.

Schroll & Hamker 2016 adds the lineage's evaluative standard for disease models:
a model of a movement disorder is judged by whether it *explains symptoms* from
mechanism, and the review is organised around which BG models predict which
symptom classes.

**Where BGM_22 improves on its predecessor, and we say so first.** Maith 2021
fitted a *functional connectivity matrix* — the Frobenius norm between the
model's and the subject's 6×6 BOLD correlation matrices. BGM_22 fits the BOLD
**time course**, TR by TR, driven by the same subject's own deconvolved cortical
activity. That is a strictly harder and more falsifiable target: an FC matrix is
15 numbers, a time course is 310 per region, and a model can match an FC matrix
with the wrong dynamics far more easily. The striatal microcircuit, the derived
cortical proportions and the input-stream contract are all likewise well past
what we published. The points below are where the project has nonetheless
dropped standards this lab had already met.

---

## Points

### 5.1 One optimisation run per condition, where this lab published twenty

Maith 2021 §2.4: "we carried out 20 optimization processes with BADS... In each
of the 20 optimization processes, different initial values were randomly drawn
for the 38 parameters... for each subject, we selected the 38 connectivity
parameters of the optimization process that resulted in the lowest loss value."

BGM_22 runs one CMA-ES optimisation per DBS condition
(`deap_cma_opt.py --optimization-run 1`) and reads the inference off the
difference between the two resulting vectors. There is no restart, so there is no
estimate of how much of any parameter's on-minus-off difference is optimiser
noise rather than DBS.

We note that seats 2 and 3 arrived at the same objection from identifiability and
from solution-set degeneracy respectively. We raise it as a **regression**: the
lab has already run this experiment and already decided that one run is not
enough, on a problem with a smoother loss surface and a cheaper evaluation.

**Goal relevance: high. Without it there is no inference, only a difference.**

### 5.2 A single subject, where this lab's inference rested on a group test

Maith 2021's conclusions — dSPN→GPi up 69.4 %, cortex→STN up 22.9 %,
cortex→iSN up 110.2 %, GPe→STN up 81.5 % — are all *t*-tests over 30 versus 40
fitted models with FDR correction and effect sizes, and the paper is careful to
say which of them contradict the rate model and which are probably
overestimates. BGM_22 has one subject and two conditions: n = 1 per cell, no
variance, no test.

This is a deliberate design choice and we do not object to it as such — a
single-subject, within-subject on/off contrast controls for everything a group
comparison cannot. But the analysis then has to come from somewhere else, and at
present nothing plays the role the *t*-test played. The available substitutes,
in increasing order of cost: restarts (point 5.1) to get an optimiser-noise
floor; a parametric bootstrap over the model's own stochasticity, refitting from
several ANNarchy seeds; and, if the Berlin dataset has more subjects with both
conditions, a small group.

**Goal relevance: high.**

### 5.3 No sensitivity analysis over the fixed parameters, which this lab did
publish and which found the fixed parameters to matter

Maith 2021 §2.6 and Fig. 7. Their finding is directly transferable: the loss was
most sensitive to the quadratic membrane parameters `n0`, `n1`, `n2` of the
*cortex* and *thalamus* populations — i.e. to parameters that were **not**
fitted, and whose values came from other papers.

BGM_22 has far more fixed parameters than Maith 2021 did, and several are known
to be weakly grounded by the project's own documentation: the GPe BOLD pooling
factors (uncited, sum to 0.77, `model_v07.md` §3.6), the six GPe/STN/SNr
firing-rate band edges, `N_cortical_inputs_dict` = 7000/2800 (`TODO.md` §23,
derivation unwritten), the striatal `phi_1`/`phi_2` = 0 (`TODO.md` §30), the
0.4 VTA proportion, and `shared_fraction` = 0.014. Any of these could be doing
what cortex `n0` did in the predecessor.

**Goal relevance: high.** It is also the cheapest of our asks: the machinery is
a loop over `parameters.csv` re-evaluating an already-fitted vector.

### 5.4 No discrimination check that the two fits are actually different fits

Maith 2021 Table 4 is a small, elegant control we would want repeated here:
before interpreting any parameter difference, show that the control-fitted models
match control data better than patient data, and vice versa.

The BGM_22 analogue is exact and cheap: evaluate the DBS-off fitted vector
against the DBS-on target, and the DBS-on vector against the DBS-off target. If
each fits its own condition better, the two vectors have captured something
condition-specific. If they cross over, or if the difference is within the
restart spread of point 5.1, the inference has no basis.

Note this is *not* the same as the cortical-drive control seat 4 asks for in its
point 4.8 — that one asks how much of the on/off BOLD change the changed cortical
input alone produces. Both are needed and they answer different questions.

**Goal relevance: high.**

### 5.5 The BOLD input variable was changed from the lab's published one, and the
change moves the fitted weights into the observable

This is the point we did not expect to find, and it is the most technical.

Maith 2021 §2.3 computes BOLD from **synaptic activity**: each neuron carries
`τ_syn ds_j/dt = −s_j` with `τ_syn = 1 ms`, and *"whenever a presynaptic action
potential reaches a neuron (no matter if excitatory or inhibitory), its synaptic
activity `s_j` is increased by 1 divided by the number of all afferent synaptic
contacts `n_aff,j` of this neuron"*, so `s_j` is bounded by 1 and is a
**normalised presynaptic event rate**. Crucially, `s_j` does not contain the
synaptic weights or the driving force. The regional signal is the population
mean, summed over the region's populations, plus a noise term, and that is what
enters the Balloon model.

BGM_22 instead maps `I_CBF` to `I` (BG populations) or `I_v` (striatal
populations) — the **net synaptic current**, which is `g · (E − v)` and therefore
carries the weights, the conductance, and the membrane potential
(`model_v07.md` §3.6, §6.1–6.2).

The consequence for the inference is direct. In Maith 2021, a fitted weight
changed the BOLD only *through the network* — by changing how many spikes
arrived somewhere. In BGM_22, a fitted weight scales the observable **directly**:
raising a `gpe_striatum` cluster scaling raises `g_gaba` in the striatal
populations, which changes `I_v` in the very same timestep, whether or not any
firing rate changes at all. The optimiser therefore has a path to the BOLD
amplitude that bypasses the model's dynamics entirely, and the ten weight-cluster
parameters sit on that path.

Two further wrinkles the documentation already notes and we would connect to
this: the GPe populations' monitor takes the raw `I` while the membrane equation
uses the compressed `f(I, nonlin)`, and `I_base` — the *entire* drive of `snr`
and `gpe_proto` — is outside `I` and so invisible to the monitor, meaning
parameters 7 and 8 have the *opposite* property, reaching the BOLD only through
the network.

We are not asserting the choice is wrong. `I_v` is a defensible BOLD proxy and
CompNeuroPy supports both. We are saying it is a **deviation from this lab's
published method that is nowhere recorded as one**, and that it changes what a
fitted weight means. At minimum: state the change and its rationale, and refit
one condition with the Maith-2021-style normalised synaptic activity to show the
conclusions do not depend on it.

**Goal relevance: high. It is the coupling between the fitted parameters and the
loss.**

### 5.6 `gpe_cp` is inherited from Goenner et al. 2021 with the projection that
justified it removed

Goenner 2021 introduced GPe-Cp into this lab's models as the **cortex-projecting**
GPe population (Abecassis et al. 2020's Npas1⁺-Nkx2.1⁺ neurons), and in that
model it projects to the striatal populations, the other GPe populations, *and*
to the cortical Integrator-Stop — which is how it participates in stopping. Its
whole functional identity is the cortico-pallido-cortical loop.

BGM_22 has no cortex: the cortical drive is a precomputed stream that nothing can
influence. So `gpe_cp` retains its striatal and intra-pallidal projections and
has lost the one that made it a distinct population. Its Izhikevich parameters
in `parameters.csv` are identical to `gpe_proto`'s, its connectivity type is
arkypallidal, and its firing-rate band is `gpe_proto`'s. Seats 1 and 2 raise the
same population from the outside literature; we raise it from inside: the lab
introduced it for a reason that no longer applies here.

**Goal relevance: medium.** Either drop it, or state why a cortex-projecting
population is retained in a model without cortex.

### 5.7 The medication state of the subject is not recorded anywhere, and the
striatal rate anchor depends on it

The project's entire striatal calibration is the **medication-off** state of
Liang et al. 2008 — the missing-GABA streams are drawn at 25/33 Hz and the rate
gate is centred there (`experimental_data/activity_striatum/README.md`,
`TODO.md` §20). We searched the repository and found no statement of whether the
Berlin subject was scanned on or off dopaminergic medication.

This matters because of what happened in the predecessor. Maith 2021 §2.1 used
Horn et al. 2019's patients, who were scanned "with their usual medication ON
(Levodopa)", and our own Discussion §4.1 invokes exactly that to explain a
discrepancy: *"Levy et al. (2001) showed that DA agonist apomorphine
administration leads to a generally lower GPi rate and a lower STN rate during
movement in PD patients. This could be a reason why we did not find an increased
STN or GPi rate in our models."* So the lab has already been bitten by the
medication state of scanned patients, and has already written that it changes the
expected subcortical rates.

If the Berlin subject was scanned on medication — which, given the shared Berlin
provenance with Horn et al. 2019, is at least likely enough to check — then the
med-off rate bands are the wrong anchor, every v07 cache is drawn at the wrong
rates, and `TODO.md` §30's question about `phi_1 = phi_2 = 0` changes character
entirely.

**Goal relevance: high, and it is a factual question with a cheap answer.**
Nothing else in this review can be resolved by reading one line of a scanning
protocol.

### 5.8 The DBS electrode artefact in the BOLD signal, which this lab flagged in
the same kind of data

Maith 2021 §4.3, verbatim: *"In addition, the patients received a DBS electrode
which may cause artifacts in the BOLD signal even when switched off, especially
in the STN signal, which reduces the validity of our results about the STN."*

BGM_22 fits seven ROIs including STN, in a DBS patient, in both the on and the
off condition — so the artefact is present in both, and in the on condition the
device is actively pulsing. We found no mention of it in the project's
documentation. Since STN is both an ROI of the loss and the stimulated
population, an artefact there contaminates precisely the measurement the DBS
inference depends on.

**Goal relevance: high.** At minimum this needs the same sentence the
predecessor wrote, and preferably a check of the STN ROI's signal quality
against the other six.

### 5.9 We are the only seat that can review the BOLD pipeline, and we are not
qualified to

`README.md` records the panel-composition consequence: no BOLD or whole-brain
lineage has a seat, so the balloon model, the `normalize_input=2000` baseline,
the HRF deconvolution producing the cortical drive, and the fit-to-fMRI
methodology have **no reviewer**. Removing Meier et al. 2022 from this seat's
list removed the last partial mitigation.

We can say only this much from inside the lineage: the balloon parameterisation
BGM_22 inherits is the one in Maith 2021 Table 2 (Friston et al. 2000, 2003), it
was used there for a different input variable (point 5.5), and this lab has
never validated the simulated BOLD against a measured BOLD *time course* — only
against correlation matrices. BGM_22 is doing something the lineage has not done
before, with a pipeline calibrated for the thing it did do before. That is not a
criticism, it is a statement of where the unreviewed risk sits.

**Goal relevance: high, and unresolvable by this panel.** It should be recorded
in the paper as a limitation in exactly those terms.

### 5.10 The model does not yet meet Schroll & Hamker 2016's stated bar, and it
may not need to

Our 2016 review judges BG disease models by whether they explain *symptoms* from
mechanism, and surveys models by which symptom classes they account for.
BGM_22 explains no symptom: it fits a resting BOLD time course and reads off
parameter changes. `TODO.md` §2 shows the inference design is still open.

We think that is defensible — the project's question is "what did DBS change
inside the BG", not "why does DBS relieve bradykinesia" — but the lineage's
published standard is the other one, and a paper from this lab will be read
against it. The bridge, if one is wanted, is Maith 2021's move: compare the
fitted changes against the rate model's predictions and against the experimental
literature, and report where they agree and where they contradict it. That gave
our predecessor paper its interpretive spine, and BGM_22 currently has no
equivalent plan.

**Goal relevance: medium.** It shapes what the result can be written up as.

---

## What we would ask for before publication

1. Multiple optimisation restarts per DBS condition, with the between-restart
   spread reported as the noise floor against which every on-minus-off parameter
   change is judged (point 5.1) — the lab's own published practice.
2. The fixed-parameter sensitivity analysis of Maith 2021 §2.6, repeated
   (point 5.3).
3. The cross-condition discrimination check: off-vector on on-data and vice
   versa (point 5.4).
4. **Find out and record whether the Berlin subject was scanned on or off
   medication** (point 5.7). If on, the striatal rate anchor and every v07 cache
   are wrong.
5. Record the change of BOLD input variable from normalised synaptic activity to
   synaptic current as a deliberate deviation, and show one condition refitted
   the predecessor's way (point 5.5).
6. State the DBS-electrode BOLD artefact as a limitation, as the predecessor did
   (point 5.8), and the unreviewed-BOLD-pipeline limitation in the terms of
   point 5.9.
