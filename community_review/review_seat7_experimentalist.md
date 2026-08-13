# Seat 7 — The experimentalist (review-defined composite seat)

Review of BGM_22 written as the composite experimentalist persona defined in
`README.md` §7 and `TODO.md` §34 (the step-1 amendment of 2026-08-13). Unlike
seats 1–6 this seat is not a lineage and has no single lab's published standard
to be judged in character against: its voice is what its six reviews
*collectively assert*.

**Mandate:** structure, connectivity and organisation of the basal ganglia,
**plus** an audit of the model's empirical validation anchors — the Liang et al.
2008 firing-rate bands and the macaque-tracer-based cortical proportions. **The
DBS representation is out of scope** for this seat and stays with seats 2 and 4.

**Read for this review (full texts, from the PDFs in this directory):**

1. Courtney CD, Pamukcu A, Chan CS (2023), Nat Neurosci 26:1147–1159,
   `10.1038/s41593-023-01368-7` — *mandatory GPe slot*
2. McGregor MM, Nelson AB (2019), Neuron 101(6):1042–1056,
   `10.1016/j.neuron.2019.03.004` — *mandatory whole-BG organisation slot*
3. Tepper JM, Koós T, Ibáñez-Sandoval O, Tecuapetla F, Faust TW, Assous M (2018),
   Front Neuroanat 12:91, `10.3389/fnana.2018.00091`
4. Wichmann T (2019), Mov Disord 34(8):1130–1143, `10.1002/mds.27741`
5. Haber SN (2016), Dialogues Clin Neurosci 18(1):7–21,
   `10.31887/DCNS.2016.18.1/shaber`
6. Emmi A, Antonini A, Macchi V, Porzionato A, De Caro R (2020), Front Neuroanat
   14:13, `10.3389/fnana.2020.00013`

**Tagging.** Per the seat definition every structural claim carries a
**species-provenance tag** — `[mouse]`, `[rat]`, `[macaque]`, `[primate]`
(where a review pools NHP species), `[human]` — beside the goal-relevance tag,
because the recent GPe literature is overwhelmingly mouse while this model is a
human-subject fit anchored on macaque tracer data.

**Convergence.** Per the amended synthesis rule, each point states whether it is
asserted by **multiple independent reviews** in this list (→ literature
consensus, weighted comparably to multi-persona convergence) or by **one**
(→ one voice).

---

## A. The external globus pallidus

### 7.1 The three GPe populations are equal in size; the literature says 50 / 18 / 12 — and that is where the model's uncited BOLD weights came from

Courtney 2023's composition, from mouse molecular and genetic work
`[mouse]`: PV⁺ ≈ 50 % of GPe neurons, NPAS1⁺ ≈ 30 %, ChAT⁺ ≈ 5 %, and
PV⁻NPAS1⁻ ≈ 15 %. NPAS1⁺ splits into NPAS1⁺FOXP2⁺ (arkypallidal, ≈ 60 % of
NPAS1⁺, so ≈ 18 % of GPe) and NPAS1⁺NKX2.1⁺ (≈ 40 %, so ≈ 12 % of GPe) — the
latter being the cortex-projecting cells the model calls `gpe_cp`.

BGM_22 simulates `gpe_proto`, `gpe_arky` and `gpe_cp` at **100 neurons each**
(`parameters.csv`), i.e. 33 / 33 / 33.

**But the BOLD pooling factors are 0.5 / 0.17 / 0.10** (`model_v07.md` §3.6),
which the project records as having "no recorded source", summing to 0.77 with
the deficit unexplained. Against the numbers above: 0.50 ≈ PV⁺ 50 %, 0.17 ≈
arkypallidal 18 %, 0.10 ≈ NKX2.1⁺ 12 %, and the missing 0.23 ≈ ChAT⁺ 5 % +
unclassified 15 % = 20 % — **the GPe cell types the model does not contain.**
The document's own conjecture ("would fit fractions of all GPe cells with the
rest belonging to types the model does not have") is therefore correct, and the
factors are GPe cell-type abundances. Seats 1 and 2 both flagged these as
uncited; this seat identifies what they are.

Two consequences. The provenance can now be written down and a source cited.
And the model is internally inconsistent: it *weights the BOLD* by realistic
abundances while *simulating* equal population sizes, so in the network dynamics
there are three times as many arkypallidal neurons inhibiting the striatum,
relative to prototypic neurons, as the abundances imply.

**Species caveat, and it is the seat's central one.** Wichmann 2019 states it
plainly: the molecular markers used to define these subtypes are rodent work and
"it is not clear whether and how these protein expression patterns translate to
the primate GPe" `[mouse → primate, untested]`.

**Convergence: multiple** (Courtney 2023 and Wichmann 2019 both assert the
arkypallidal/prototypic division as established; Courtney supplies the
proportions). **Goal relevance: high** — the pooling factors define one of the
seven fitted BOLD signals.

### 7.2 The model's within-GPe wiring is the inverse of what is reported

Courtney 2023 `[mouse]`: GPe neurons inhibit each other through local axon
collaterals which "give rise to a large number of local synapses, in contrast to
the relatively few synapses typically formed by inputs from individual SPNs";
**PV⁺ neurons provide the largest local input**, while **NPAS1⁺FOXP2⁺
(arkypallidal) neurons "do not produce appreciable levels of local
connections."**

BGM_22 has no within-population connection anywhere (`model_v07.md` §5): no
`gpe_proto → gpe_proto`. What it does have is `gpe_arky → gpe_proto` and
`gpe_arky → gpe_cp` at 0.008. So the model contains the pallidal local
connections the literature says are negligible and omits the ones it says
dominate.

Seats 5 and 6 reach the absence of within-nucleus recurrence from the modelling
side (the project's own predecessor had them and found GPe–GPe significantly
changed in PD). This seat adds which specific connection is missing and which
present connection is doubtful.

**Convergence: one voice** for the PV⁺-dominant/arky-negligible detail
(Courtney only); the existence of dense local collaterals is uncontested across
the list. **Goal relevance: medium–high.**

### 7.3 GPe input balance: ~80 % of GPe synapses are striatal GABAergic; in the model the GPe populations are cortically driven

Courtney 2023 `[mouse, with rat anatomy]`: GABAergic synapses constitute ≈ 80 %
of all synapses in the GPe, and the dorsal striatum contributes the large
majority of that projection, at least an order of magnitude more than any other
source.

In BGM_22, `gpe_arky` and `gpe_cp` each receive **500 cortical afferents per
neuron** through `CorticalInputs` (`model_v07.md` §3.3) against **10** striatal
afferents from `str_d2` via `connect_fixed_number_pre(number=10)`. `gpe_proto`
receives no cortical stream but also only 10 striatal afferents plus a fitted
baseline current. So the model's GPe is driven overwhelmingly by cortex where
the anatomy says it is driven overwhelmingly by striatum.

**In the model's favour**, and worth stating because it is a place where BGM_22
is ahead of the textbook: Courtney 2023 emphasises that cortical input to GPe is
real and that "the spatially distributed cortical inputs together form the
largest source of excitatory inputs to the GPe, in contrast to the traditional
model which assumes dominant glutamatergic input from the STN." The model having
a direct cortico-pallidal drive at all is correct and unusual. The objection is
only to the balance against the striatal projection.

**Convergence: one voice** (Courtney), but the underlying anatomy is not
contested elsewhere in the list. **Goal relevance: high** — the fitted drive
weights for `gpe_arky` and `gpe_cp` (parameters 4 and 5) are the model's
principal control over two of the three GPe populations.

### 7.4 Striatal input to GPe is cell-type-specific, and the model's assignment is partly reversed

Courtney 2023 `[mouse]`: "iSPNs strongly target canonical STN-projecting PV⁺
(PV⁺KCNG4⁺) neurons, whereas dSPNs largely target NPAS1⁺ neurons, Pf-projecting
PV⁺ (PV⁺LHX6⁻) neurons and ChAT⁺ neurons." The review also stresses that dSPN
axons collateralise in GPe as well as SNr — "these studies solidify the fact
that the GPe receives notable direct-pathway inputs" — against the classical
scheme in which the direct pathway bypasses GPe.

BGM_22: `str_d1 → gpe_cp` only, at weight 0.005 (the smallest weight in the
whole table), and `str_d2 → gpe_proto, gpe_arky, gpe_cp` at 0.04 / 0.08 / 0.08.
So the dSPN→GPe projection exists but is negligible and reaches only one of the
two NPAS1⁺ populations, while the iSPN projection is spread evenly over all
three instead of preferentially reaching the prototypic population.

**Convergence: one voice** (Courtney) for the target specificity; the *existence*
of substantial dSPN→GPe is asserted by Courtney and consistent with McGregor &
Nelson's account of the pathways being less segregated than the classical model.
**Goal relevance: medium** — the whole `str_d1__bg` and `str_d2__bg` clusters
are fitted, so the balance is partly absorbed, but the topology is not.

---

## B. The striatum and its interneurons

### 7.5 One interneuron class out of at least eight, at a fraction that is a rodent value applied to a human

Tepper 2018 `[rat/mouse, with primate notes]` catalogues the striatal GABAergic
interneurons now identified: FSI (PV⁺), LTS (NPY/NOS/SOM), calretinin, TH,
neurogliaform (NGF), fast-adapting (FAI), spontaneously-active bursty (SABI) —
"and it is likely that we have not yet found all of them." BGM_22's microcircuit
has **FS only**, at 2.9 % of striatal cells (del Rey proportions,
`model_v07.md` §7.1), with dSPN + iSPN making up the remaining 97.1 %.

Haber 2016 `[macaque/human]` supplies the species correction: medium spiny
neurons "in nonprimate species... account for over 90 % of the cells, and
probably far less in the primates." So a 97.1 % projection-neuron fraction is
not merely a rodent value — it is above even the rodent figure, applied to a
human model.

Two specific omissions matter more than the head-count:

**The NGF interneuron.** Tepper 2018 `[mouse]`: NGF cells evoke a GABA_A,slow
IPSC in SPNs whose decay is roughly an order of magnitude slower than the
conventional fast GABA_A IPSC, and they connect to a very high proportion of
SPNs within their axonal field; the review calls this "an extremely powerful
source of inhibition to the SPNs, not only because of its amplitude but also
because of the extremely long duration and slow decay of the IPSC." BGM_22's SPN
`tau_gaba` is 4 ms and there is nothing slower anywhere in the striatum. For a
model whose observable is a *time-integrated synaptic current*, a missing
inhibitory conductance an order of magnitude slower than the one present is not a
detail about spike timing — it is a missing low-pass term in the very quantity
being fitted.

**FSI subtypes with pathway-specific targeting.** Tepper 2018, citing Garas et
al. 2016 `[rat and primate — explicitly not mouse]`: FSIs divide by secretagogin
expression into Scgn⁺ cells preferentially targeting dSPN somata and Scgn⁻ cells
preferentially innervating iSPN axons, with different spike timing relative to
the two pathways. BGM_22's single FS population is fitted with one drive weight.
Note the model's kernels do encode differential targeting geometrically
(FS→dSPN σ = 394 µm against FS→iSPN σ = 140 µm), which is a partial capture.

**Convergence: multiple** (Tepper for the diversity and the NGF conductance,
Haber independently for the primate projection-neuron fraction).
**Goal relevance: high** for the NGF point specifically.

### 7.6 Cortical afferents from different regions are drawn independently; in the primate their terminal fields overlap by 50 % at 5 mm cortical separation

`model_v07.md` §7.5: "Regions, by contrast, are drawn **apart**: each has its own
axon pool, so within a bin the streams of different regions are independent."

Haber 2016 `[macaque]`, reporting Averbeck et al. 2014: "convergence of
cortico-striatal terminals from cortical areas that are separated by 5 mm is
50 % in nonhuman primates. The overlap decreased to below 20 % for regions
separated by 30 mm." The review's whole thesis is that "convergence of inputs
from different functional regions" is extensive and occurs in specific striatal
interface zones.

Our seven ROIs include several pairs well inside 5 mm of cortical separation
(M1/PMd, PMd/PMv, SMA/preSMA). The model treats those as independent axon pools
while the anatomy says they overlap by about half. Note that this is *not* the
same quantity as the Kincaid 0.014 within-region shared fraction the model does
represent; it is cross-region sharing, which the model sets to zero by
construction.

There is one mitigating factor the project should get credit for: the streams'
rates come from seven *correlated* deconvolved BOLD series (median off-diagonal
r = 0.46, up to 0.90 for PMd–PMv, per
`experimental_data/cortical_proportions/README.md`), so cross-region
co-fluctuation is present at the slow timescale even though the pools are
disjoint. What is missing is the within-bin sharing.

**Convergence: one voice** (Haber), but with a published quantitative estimate
from the field's authority on this exact measurement. **Goal relevance: high** —
this is the same input-correlation quantity that `input_streams/README.md` §3
identifies as the dominant determinant of the simulated BOLD amplitude.

### 7.7 Cortical afferents converge onto single interneurons more than onto SPNs

Haber 2016 `[rat, cited into a primate review]`: "axons from each area synapse
onto single fast-spiking GABAergic interneurons. Interestingly, these
interneurons are more responsive to cortical input than the MSNs. This suggests
a potentially critical role for interneurons in integrating information from
different cortical areas before passing that information on to the medium spiny
projection cells."

This is exactly the concern `TODO.md` §26 records — the model derives
`f_FS↔SPN` = 0.00885 from a single axon pool under the assumption that FS and SPN
sample it with equal per-axon contact probability, while Ramanathan 2002 and
Choi 2018 report higher convergence onto FS. This seat's reading confirms it from
a second direction and adds the functional claim: the cross-region integration
the model cannot represent (point 7.6) is asserted to happen *specifically at
the interneuron*, which is the cell type the model has 29 of.

**Convergence: multiple** (Haber asserts it; Tepper's account of FSI cortical
and thalamic afferents is consistent). **Goal relevance: medium.**

---

## C. Audit of the validation anchors

### 7.8 The Liang et al. 2008 rate bands: contested in the literature, but defensible, and the model's ordering is independently supported

This seat's mandate includes auditing
`experimental_data/activity_striatum/README.md`. Our verdict is that the
document is unusually careful and that its conclusion survives, with two
additions it should record.

**The anchor is on one side of an open disagreement, and the project does not
say so.** McGregor & Nelson 2019 `[primate/human review]`: "evidence from humans
and nonhuman primates is conflicting, showing either marked increases in MSN
firing (Liang et al., 2008; Singh et al., 2016) or no change (Deffains et al.,
2016)." So Liang is not an isolated outlier — Singh et al. 2016 is human
intraoperative striatal recording supporting it — but neither is it settled.
The project's README cites Deffains 2016 elsewhere (in
`input_streams/README.md` §4.3, on beta entrainment) without connecting it to the
rate anchor.

**The magnitude, against a normal-primate baseline.** Haber 2016 `[macaque]`:
MSNs "have a very low spontaneous discharge rate (0.5–1 spike/s), but a
relatively high firing rate (10–40 spikes/s) associated with behavioral tasks."
So Liang's 25/33 Hz parkinsonian resting rates sit *above* the normal
primate's task-driven range and 25–50× its resting rate. The README's "why
these rates are so high" section anticipates this and attributes it to chronic
denervation, which is the paper's own claim; we record the comparison because a
reader will make it.

**The direction is independently supported by identified-cell data, which is
the assumption the README says it cannot remove.** McGregor & Nelson `[rat]`:
"In anesthetized parkinsonian rats, antidromically identified dMSNs showed
decreased firing as compared to healthy animals, while presumed iMSNs showed
elevated firing (Mallet et al., 2006; Kita and Kita, 2011)." BGM_22 assigns
dSPN 25 Hz and iSPN 33 Hz — iSPN above dSPN, the same ordering, derived by Liang
from levodopa response direction. That the two independent routes agree on the
sign is meaningful support for the response-direction assumption the README
correctly identifies as its weakest link.

**The FS rate.** We find no fault with the derivation. Tepper 2018 adds one
confirmation of a caveat the README already lists `[primate]`: three
structurally distinct calretinin populations exist in primates, and "the 'large'
CR interneurons also express ChAT, a difference between primates and rodents" —
so a primate extracellular "FSI" waveform class is genuinely heterogeneous, and
the model's PV⁺-specific FS population is a subset of what those recordings
measured.

**Convergence: multiple** (McGregor & Nelson and Haber both bear on it).
**Goal relevance: high** — the bands gate every evaluation, and the same rates
are baked into the caches. **Recommended action: none to the values; add the
conflicting-literature citation to the README.**

### 7.9 The cortical proportions: the topography is supported; the renormalisation loss is larger than the document estimates

Auditing `experimental_data/cortical_proportions/README.md` against Haber 2016
`[macaque, with human DWI]`:

**Supported.** M1 projections "terminate almost entirely in the putamen, in the
dorsolateral and central region" — consistent with putamen M1 = 0.28 against
caudate M1 = 0.02. dPFC "projects primarily to the rostral striatum, including
the rostral central region of the caudate nucleus... primarily in the head of the
caudate and the medial, ventral, and central parts of the rostral putamen. The
dorsolateral portion of the putamen contains fewer terminating axons" —
consistent with caudate dlPFC = 0.55 against putamen dlPFC = 0.10. "The rostral
premotor areas terminate in both the caudate and putamen, bridging the two with a
continuous projection" — consistent with PMd being substantial in both (0.18 /
0.11). The derivation's qualitative structure holds up well.

**The renormalisation, quantified.** The README's caveat 2 says cingulate,
insula, temporal, parietal, orbital and ventrolateral prefrontal input is 25–60 %
of labelled cells and "has nowhere to go". Haber puts a spatial number on part of
it: "the volume occupied by the collective dense terminal fields from the vmPFC,
dACC and OFC is approximately 22 % of the striatum, a larger cortical input than
would be predicted by the relative cortical volume of these areas." So roughly a
fifth of the striatum's dense terminal territory belongs to three regions with no
ROI in this dataset — and those are the regions with the *strongest*
overrepresentation relative to cortical volume. The seven ROIs are not a
representative sample being renormalised; they are a sample that systematically
excludes the densest projections.

**Convergence: one voice** (Haber), but on his own primary measurements.
**Goal relevance: medium** — it does not change the caudate/putamen contrast,
which is what the inference reads; it bounds what "the cortical drive" means.

---

## D. The subthalamic nucleus

### 7.10 The STN receives the striatum's cortical proportions, and at least one entry is contradicted by tracing

This is the sharpest structural finding in this seat's review.

`CorticalInputs` splits each STN neuron's 500 cortical afferents using
`mc.cortical_proportions_dict` — the *same* per-region mix used for the striatal
populations (`model_v07.md` §3.3, §8). For the putamen loop that gives an STN
whose cortical input is 28 % M1, 18 % PMv, 15 % SMA, **13 % S1**, 11 % PMd, 10 %
dlPFC, 5 % preSMA.

Emmi 2020 `[macaque]`, reporting Von Monakow et al. 1978: Brodmann area 4 (M1)
projects to dorsal and lateral STN, area 6 to central STN, area 8 to ventral STN,
and "the authors were unable to identify any projections arising from Brodmann
areas 9 and 3,1,2." **Areas 3, 1 and 2 are S1.** Haynes & Haber 2013's large
tracing study `[macaque]`, also reported there, later established a dorsal
prefrontal (areas 9 and 46) projection to the medial STN, so the area-9 negative
did not hold — but no source in this seat's reading restores an S1 projection.

The corticosubthalamic projection is also organised quite differently from the
corticostriatal one: M1 to dorsolateral STN with a somatotopy (leg medial, arm
lateral, orofacial dorsolateral, per Parent & Hazrati 1995 / Nambu 1996–2000),
SMA and ventral premotor to the medial portion with an *inverse* somatotopic
distribution, caudal dorsal premotor to ventrolateral STN, dACC to the medial tip
`[macaque]`. Applying the striatal mix to the STN assumes the two projections
have the same regional composition, which no source in this list supports.

**Convergence: one voice** (Emmi), but reporting multiple independent tracing
studies within it. **Goal relevance: medium–high** — the STN is the stimulated
nucleus and one of the seven BOLD ROIs, and its drive weight (parameter 6) is
fitted against a mix that is at minimum wrong in its S1 entry.

### 7.11 The STN is tripartite, overlapping, and contains interneurons; the model's is one homogeneous excitatory population

Emmi 2020 `[macaque, human]`: the classical tripartite subdivision — dorsolateral
and caudal motor, ventrolateral/rostral associative, medial rostral limbic — is
"the main trend in classical literature", confirmed by Haynes & Haber 2013 and
by human DWI clustering (Lambert et al. 2012), **but** "more recent evidence
points toward overlapping subregions and converging axonal afferents", and even
Lambert's own paper describes the associative territory as an overlapping
transition between limbic and motor rather than a distinct subregion.

Also `[human]`: GABAergic interneurons are present in human STN, with smaller
somata (12 µm against 24 µm) and, unlike the projection neurons, lacking both
parvalbumin and calretinin immunoreactivity (Levesque & Parent 2005); and the
proportion of PV-expressing neurons is significantly *higher* in primates and
humans than in rodents (Hardman et al. 2002). Unbiased stereology puts the human
STN at 431 ± 72 to 561 ± 30 ×10³ neurons in 114–240 mm³.

BGM_22's STN is 100 excitatory Izhikevich neurons with no subdivision, no
interneurons and no space. Seat 6 raises the spatial consequence for electrode
placement; this seat adds that the subdivision is *anatomically* real in the two
species that matter here, that its boundaries are gradients rather than borders,
and that the model's own VTA data (`sub-01_overlap.csv`) resolves all three
subdivisions.

**Convergence: one voice** (Emmi is the seat's only STN-dedicated review), but
its tripartite claim is corroborated in passing by McGregor & Nelson's account of
functional-channel organisation. **Goal relevance: medium.**

### 7.12 Two projections the model lacks, each asserted by two independent sources

**STN → striatum.** Emmi 2020 `[squirrel monkey, macaque]`: Smith et al. 1990
and Sato et al. 2000b both report STN projections to the striatum, and Sato's
axon-branching classification puts 17.3 % of STN neurons in the
striatum-projecting class. Seat 3 raises the same omission from Girard 2021,
which includes STN→MSN and STN→FSI explicitly.

**CM/Pf thalamus → STN, GPe and the output nuclei.** Emmi 2020 `[squirrel
monkey]`: Sadikot et al. 1992 traced centromedian → dorsolateral STN and
parafascicular → medial and rostral STN. Tepper 2018 `[rat, with the note that
these inputs are denser in primates than in rodents]` documents the PfN
innervation of striatal interneurons, and Haber 2016 lists thalamus as one of the
three major striatal afferent sources. BGM_22's only thalamic population is the
BG *target* (`snr → thal → striatum`); nothing supplies the intralaminar drive
to STN, GPe and SNr. Seat 3 raises this as the CM/Pf omission and reports that
in its own model this input acts as a global gain on the circuit.

**Convergence: multiple, and cross-seat** (Emmi + Girard for STN→striatum; Emmi +
Tepper + Haber + Girard for the intralaminar input). **Goal relevance: low–medium
for the fit, medium for interpretation** — a missing global thalamic gain
competes for the same explanatory role as the fitted drive weights.

---

## E. Cross-cutting

### 7.13 The two loops share nothing; the anatomy says the caudate/putamen boundary is structural, not functional

Haber 2016 `[macaque/human]`: "there is no clear boundary between the VS and DS,
and the separation between the caudate nucleus and the putamen is merely a
structural one, based solely on the IC separation, not a functional one." The
review's central argument is that "these pathways are not as segregated as once
thought", with dense interweaving terminal fields and specific striatal
convergence zones.

McGregor & Nelson 2019 `[primate/rodent]` add the disease-state version: loss of
functional segregation between sensorimotor, associative and limbic channels is
a documented feature of the parkinsonian basal ganglia.

BGM_22 simulates `caudate` and `putamen` as two loops sharing **no projection at
all**, meeting only inside the pooled BOLD monitors, differing only in their
cortical proportions. Both reviews say that separation is sharper than the
anatomy, and the second says the disease being modelled specifically degrades it.
This bears directly on `DBS.md`'s use of the caudate as "the free control": the
control is clean in the model by construction, and the anatomy does not license
assuming it is clean in the subject.

**Convergence: multiple.** **Goal relevance: medium–high** — the two-loop
contrast is the model's main structural degree of freedom for the inference.

### 7.14 Dopamine is lost outside the striatum too, and the model has no dopamine anywhere

Wichmann 2019 `[macaque, human]`: "there is ample evidence that dopaminergic
midbrain neurons innervate not only the striatum, but also extrastriatal sites"
and this loss "is also well documented to occur in parkinsonian animals and in
PD patients." Specifically: D2-like receptors on striatopallidal terminals in
primate GPe; D1 and D2 receptors on preterminal axons and glutamatergic and
GABAergic terminals in monkey STN, where local D1 agonists lower firing rates but
increase bursting; D1LRs on striatopallidal and striatonigral terminals in
GPi/SNr, where agonists reduce firing. And a functional consequence: "the
comparatively maintained pallidal and nigral dopamine levels in early
parkinsonism may thus compensate for the striatal dopamine loss in early PD,
helping to maintain GPi/SNr firing rates at levels close to normal."

BGM_22's only dopamine representation is `phi_1`/`phi_2` in the SPN models, set
to 0 and therefore inert (`model_v07.md` §6.2, `TODO.md` §30). Seats 1 and 6
raise the striatal half; this seat records that even a correct striatal dopamine
parameterisation would leave the extrastriatal half unrepresented, and that the
quantity most affected — GPi/SNr baseline rate — is one of the model's seven
BOLD ROIs and one of its gate bands.

**Convergence: one voice** (Wichmann), on primate and human data.
**Goal relevance: medium.**

### 7.15 Rate is the model's only validation, and this is the decade the field decided rate was not the story

Wichmann 2019 `[primate/human]` is unambiguous: "Models such as the 'rate' model
are now clearly outdated"; "rate changes in the basal ganglia/thalamocortical
circuits may not be as important for the pathophysiology of movement disorders
than originally thought"; recent optogenetic studies "have cast doubt on the
rate-model notion that parkinsonism arises from an imbalance between the
(antikinetic) actions of striatal neurons that project to the GPe and the
(prokinetic) actions of neurons that project to the GPi", since movement is
accompanied by combined activation of *both* SPN populations. McGregor & Nelson
2019 concur that changes in pattern and synchronisation are more consistent than
changes in rate.

BGM_22's only non-BOLD constraint is `get_firing_rate_loss`, a set of nine rate
bands. Every seat on this panel has asked for a validation the loss does not
already optimise; this seat adds that the *kind* of validation used is the one
the field has spent a decade demoting.

**Two things must be said in the model's defence, and we say them without
qualification.** First, the rate bands are used as a *plausibility gate*, not as
a claim that rate causes symptoms — the model's actual target is a time course.
Second, Wichmann himself endorses the resting-state design: "it is also
remarkable that most of our knowledge of the pathophysiology of parkinsonism was
acquired through studies of neuronal activity at rest — when, strictly speaking,
neither bradykinesia nor akinesia are present," offered as a limitation of the
field, not of this project. BGM_22 studying resting state is standard; using
rate as its sole non-fitted constraint is what we flag.

**Convergence: multiple.** **Goal relevance: high.**

### 7.16 One thing the model gets right that the classical picture does not: fitting weights rather than rates

Wichmann 2019 `[rodent, macaque, human]`: "parkinsonism is associated with
considerable morphological and functional plasticity at synapses within the basal
ganglia, thalamus, and cortex" — loss and remodelling of glutamatergic synapses
in the striatum (including thalamostriatal inputs and the intralaminar nuclei),
glutamatergic remodelling in the STN that triggers subsequent remodelling of the
GABAergic pallidosubthalamic synapses, and a dropout of thalamic innervation of
motor cortex layer 5.

BGM_22's inference is precisely a claim about **synaptic weights** changing
between conditions. That is the right observable to be asking about: the
literature says the parkinsonian and stimulated basal ganglia differ from the
healthy one in its synapses, not only in its rates. We record this because the
rest of this review is critical, and because it is a genuine point of alignment
between the model's design and where the experimental field has moved.

**Convergence: one voice** (Wichmann), a review of primate and human evidence.
**Goal relevance: high, and favourable.**

---

## What this seat would ask for

1. Cite the GPe pooling factors as cell-type abundances, and resolve the
   inconsistency between weighting the BOLD by real abundances and simulating
   equal population sizes (point 7.1).
2. Add the conflicting-literature citation (Singh et al. 2016 supporting, Deffains
   et al. 2016 opposing, via McGregor & Nelson 2019) to
   `experimental_data/activity_striatum/README.md`; record that the
   identified-cell rodent data independently supports the dSPN < iSPN ordering
   (point 7.8). **No change to the values.**
3. Do not apply the striatal cortical proportions to the STN unexamined; at
   minimum remove or justify the 13 % S1 share of the putamen STN's cortical
   afferents (point 7.10).
4. Record in `experimental_data/cortical_proportions/README.md` that the excluded
   regions are not a random remainder — vmPFC, dACC and OFC alone occupy ~22 % of
   striatal dense terminal territory (point 7.9).
5. State that the two loops' strict separation exceeds the anatomy, and that PD
   degrades channel segregation, wherever the caudate is used as a control
   (point 7.13).
6. Consider a slow inhibitory conductance in the striatum standing in for NGF
   input, since the observable is a time-integrated current (point 7.5).
7. Record the species provenance of the GPe three-way division as mouse molecular
   work whose translation to primate is explicitly untested (point 7.1).
