# Seat 7 — The experimentalist (composite seat)

*2026-08-14*

This seat is a review-defined composite and a deliberate exception to the panel's
one-seat-one-lineage rule. It is not a modelling lineage and has no model of its
own. Its voice is what six recent authoritative reviews of basal ganglia
structure, connectivity and organization collectively assert about the anatomy,
held against what BGM_22 contains. Its second mandate is to audit the model's two
empirical validation anchors — the Liang et al. 2008 medication-off striatal
firing-rate bands and the macaque-tracer-based cortical proportions — because
those are experimental claims no modelling seat on this panel is positioned to
check.

This is round 2 of the `TODO.md` §34 survey (step 3 rerun, 2026-08-14); the seat
definition and reading list are in `../README.md`. It was written independently of
the round-1 seat 7 review, which was not consulted, as were the round-1 synthesis
and every other seat's round-2 file. Facts that `model_v07.md` attributes to
`community_review/round1/...` are treated as part of the project's own
documentation and cited as such, without opening the cited files.

**The DBS representation is out of scope** for this seat. DBS appears below only
where an anatomical fact bears on it — chiefly, what pathways exist to be
stimulated.

The two round-2 requirements govern what follows. **(1) Primary evidence for every
claim**: never stopping at "the review asserts it" — each point reports which
primary study the review invokes, in what species and preparation, what was
measured, and whether the review presents it as established or emerging.
**(2) Explicit contrast in every point**: what BGM_22 does, what the literature
reports, the primary evidence, why it matters here, one verdict, and a
goal-relevance rating. Two rules are specific to this seat. **Species-provenance
tags** — every structural claim carries `[mouse]`, `[rat]`, `[macaque]`, `[human]`
or a combination, because the recent GPe literature is overwhelmingly mouse while
BGM_22 is a human-subject fit anchored on macaque tracer data. **Within-list
convergence** — every point closes by stating whether its central claim is
asserted by several independent reviews on the reading list (consensus, strong)
or by one (one voice, weaker).

**Read for this review (full texts, from the PDFs in this directory's parent):**

- Courtney CD, Pamukcu A, Chan CS (2023). Cell and circuit complexity of the
  external globus pallidus. *Nat Neurosci* 26:1147–1159.
  `10.1038/s41593-023-01368-7`
- McGregor MM, Nelson AB (2019). Circuit mechanisms of Parkinson's disease.
  *Neuron* 101(6):1042–1056. `10.1016/j.neuron.2019.03.004`
- Tepper JM, Koós T, Ibáñez-Sandoval O, Tecuapetla F, Faust TW, Assous M (2018).
  Heterogeneity and diversity of striatal GABAergic interneurons: update 2018.
  *Front Neuroanat* 12:91. `10.3389/fnana.2018.00091`
- Wichmann T (2019). Changing views of the pathophysiology of parkinsonism.
  *Mov Disord* 34(8):1130–1143. `10.1002/mds.27741`
- Haber SN (2016). Corticostriatal circuitry. *Dialogues Clin Neurosci*
  18(1):7–21. `10.31887/DCNS.2016.18.1/shaber`
- Emmi A, Antonini A, Macchi V, Porzionato A, De Caro R (2020). Anatomy and
  connectivity of the subthalamic nucleus in humans and non-human primates.
  *Front Neuroanat* 14:13. `10.3389/fnana.2020.00013`

**What was reviewed:** `model_v07.md` (full), `model_v08.md` (full, as the smoke
test it claims to be), the `experimental_data/` READMEs for `activity_striatum`,
`cortical_proportions` and `input_streams`, `DBS.md` (background only),
`BOLD_optimization/get_loss.py` (`get_firing_rate_loss`, `add_TimedInputs`, and
the BOLD-pooling and DBS blocks of `__main__`), `BOLD_optimization/parameters.py`,
`CompNeuroPy/.../bgm_22/model_creation_functions.py` (`BGM_v07`),
`CompNeuroPy/.../bgm_22/parameters.csv` (column `BGM_v07_p01`, read
programmatically), `CompNeuroPy/.../striatal_microcircuit/microcircuit.py`
(constructor defaults and the `props_delRey` / `density` /
`N_cortical_inputs_dict` constants), and `TODO.md` (Roadmap and §1–§33; §34
skipped per protocol).

---

## The standard this seat applies

What counts as evidence here is a claim an authoritative recent review states
**and backs with a named primary study**, where the review is clear about what was
measured, in what preparation, and how firmly it is established. A review sentence
with no primary citation behind it is reported as a summary judgement and never
carries a change proposal.

- **Convergence beats emphasis.** A claim two or more of these six reviews make
  independently is literature consensus; a claim one review makes, however
  forcefully, is one voice.
- **Species match is part of the evidence.** Mouse molecular work is the best
  evidence that exists for GPe cell types and nearly the only evidence for
  cell-type-specific striatopallidal targeting — and both Courtney and Wichmann
  say in their own words that its translation to primate is untested. A mouse-only
  finding cannot by itself license a change to a human-target model; it can
  license documentation, and a change only where the model is *internally*
  inconsistent about which species it follows.
- **Absence of a measurement is not a deficiency.** Where the literature has no
  number, or the reviews disagree, the point is an open question with a request
  that BGM_22 record the uncertainty.

Nothing below is cited from memory. Where a citation is weaker than it looks — a
textbook chapter rather than a measurement, or a review's own remark rather than a
finding — the point says so in place.

---

## Points

### 7.1 The GPe cell-type abundances weight the BOLD signal but not the network

**What BGM_22 does:** `get_loss.py` (`__main__`, `gpe_proportions`) sets
`{"gpe_proto": 0.5, "gpe_arky": 0.17, "gpe_cp": 0.10}`, normalises over all six
pooled populations, and passes them as `scale_factor` to the GPe `BoldMonitor`.
The populations are each exactly 100 neurons — `parameters.csv` column
`BGM_v07_p01` gives `gpe_proto.size = gpe_arky.size = gpe_cp.size = 100`, verified
by reading the column. `model_v07.md` §3.6 states the consequence itself: the model
"weights the BOLD by these realistic abundances while simulating the three
populations at 100 neurons each, so the network dynamics carry three times as many
arkypallidal neurons relative to prototypic ones as the abundances imply."

**What the literature reports:** Courtney et al. 2023 Fig. 2a and Table 1 give the
rodent GPe composition as PV⁺ 50 %, NPAS1⁺ 30 %, ChAT⁺ 5 %, PV⁻NPAS1⁻ 15 %; within
NPAS1⁺, 60 % are FOXP2⁺ (arkypallidal, ≈18 % of the GPe) and 40 % NKX2.1⁺ (≈12 %)
`[mouse]` — the numbers `model_v07.md` §3.6 already cites as the source of
0.5 / 0.17 / 0.10.

**The primary evidence:** the review attributes the 50 % figure to a recent
systematic examination across transgenic lines that "definitively confirmed that
PV⁺ neurons account for approximately 50 % of neurons in the GPe", noting earlier
estimates varied across laboratories through methodological differences (driver
lines, retrograde labeling, fate mapping, immunohistochemistry) and spatial
gradients in PV⁺ distribution `[mouse]`. The 60/40 split rests on `Npas1`, `Foxp2`
and `Npr3`/`Lhx6` driver lines plus single-cell transcriptomics. Courtney presents
the composition as established for rodent and adds its own caveat in the same
section: "we cannot yet directly equate neuron subtypes in rodents to those in
higher-order species owing to methodological limitations in identifying neuron
types in primates." Wichmann 2019 states the same independently: "It is not clear
whether and how these protein expression patterns translate to the primate GPe"
`[primate]`.

**Why it matters (or does not) here:** GPe is one of seven fitted ROIs and the only
one whose pooling weights are not the size-proportional default. The model makes
two incompatible commitments about the same three populations at once: the
observation model says the GPe is half prototypic and a sixth arkypallidal, the
network says a third each. Fitted parameters that trade arkypallidal against
prototypic influence — `gpe_striatum` and `gpe_laterals` are exactly such clusters
(`model_v07.md` §5) — absorb the inconsistency. This is not a claim that the mouse
abundances hold in a human GPe; it is a claim that the model should not assume them
in one place and contradict them in another. The inconsistency is species-neutral,
which is why a change proposal is licensed where the mouse-only provenance would
otherwise block one.
→ **experimentally grounded deficiency.** Proposal: either size the three GPe
populations by the same abundances the BOLD weights use (100 → e.g. 160 / 58 / 38
at the same total), or drop the abundance weighting and let the size-proportional
default carry it, as v07 already does for the striatum (`get_loss.py`,
`str_scaling_factors = None` for v07). Either way, write the Courtney citation into
`get_loss.py` beside `gpe_proportions`, where `model_v07.md` §3.6 records it is
still missing, with the shared caveat that the numbers are mouse.

**Goal relevance: high** — the GPe BOLD time course is one of seven fitted signals,
and the inconsistency sits between the network and the observation model.

**Within-list convergence:** the percentages are **one voice** (Courtney 2023). The
prototypic/arkypallidal division and the warning against assuming it translates to
primate are **consensus** — Courtney 2023, Wichmann 2019 and McGregor & Nelson 2019
all treat the division as real and rodent-derived.

---

### 7.2 GPe internal wiring: local collaterals and SPN-subtype targeting

**What BGM_22 does:** six intra-GPe projections, all in the single optimizer
cluster `gpe_laterals`, so their ratios are frozen and only their common scale is
fitted (`model_v07.md` §5, §11): `gpe_proto → gpe_arky` and `gpe_proto → gpe_cp` at
0.025, and `gpe_arky → gpe_proto`, `gpe_arky → gpe_cp`, `gpe_cp → gpe_proto`,
`gpe_cp → gpe_arky` all at 0.008. Striatopallidal input: `str_d1 → gpe_cp` 0.005
(cluster `str_d1__bg`, shared with `str_d1 → snr` at 0.06) and `str_d2 → gpe_proto`
0.04, `str_d2 → gpe_arky` 0.08, `str_d2 → gpe_cp` 0.08 (all `str_d2__bg`, again a
frozen ratio).

**What the literature reports:** Courtney 2023 on local collaterals `[mouse]`:
"First, PV⁺ neurons provide the largest local input. Second, NPAS1⁺FOXP2⁺ neurons
do not produce appreciable levels of local connections." On striatopallidal input
`[mouse]`: "iSPN axons terminate exclusively in the GPe" while "dSPN axons
collateralize in both the GPe and the SNr"; iSPN boutons are "three to eight times
higher than that formed by dSPNs"; and "iSPNs strongly target canonical
STN-projecting PV⁺ (PV⁺KCNG4⁺) neurons, whereas dSPNs largely target NPAS1⁺
neurons, Pf-projecting PV⁺ (PV⁺LHX6⁻) neurons and ChAT⁺ neurons."

**The primary evidence:** the local-collateral findings come from recent transgenic
and optogenetic work activating individual GPe classes, which the review says only
became feasible recently `[mouse]`. The subtype-targeting claim is sourced to "both
viral-tracing and modeling data" — one of the two strands is modelling, not
measurement, which the review states plainly. dSPN collateralisation in GPe is
presented as settled ("modern techniques have confirmed"); the bouton ratio as a
recent revision of an older estimate. Box 1 adds two qualifications bearing on the
model: "while all PV⁺ neurons are prototypic, not all prototypic neurons are PV⁺.
Furthermore, not all GPe neurons with ascending projections to the dStr are
arkypallidal neurons."

**Why it matters (or does not) here:** three readings. *In the model's favour*: a
direct-pathway collateral into the GPe is a real feature most three-population BG
models omit, and BGM_22 has it (`str_d1 → gpe_cp`); it routes it to the
non-prototypic population, which is what Courtney reports; and it keeps it an order
of magnitude weaker than the iSPN weights (0.005 vs 0.04–0.08), the direction of the
bouton asymmetry. *Against*: Courtney has iSPNs *strongly* targeting the
STN-projecting prototypic class, while the model gives `str_d2 → gpe_proto` half the
weight of the other two and freezes the ratio inside `str_d2__bg`; likewise
arkypallidal neurons make few local connections, yet `gpe_arky →` sits at a third of
the prototypic collateral weight, also frozen. *Why no change is proposed*: every
targeting fact here is mouse, of the marker-defined kind both Courtney and Wichmann
say has not been shown to translate to primate. The model's third population is also
not identifiable against the marker scheme — `gpe_cp` is expanded nowhere in the
repository (grepped across `*.md` and the CompNeuroPy `bgm_22` sources), and if it
means cortex-projecting, Courtney's NPAS1⁺NKX2.1⁺ neurons "project exclusively to
the midbrain, the cortex and the reticular nucleus of the thalamus" — not to the
striatum — whereas the model's `gpe_cp` projects to all three striatal populations
at 0.5 / 0.5 / 0.8. Box 1 leaves room for a second striatum-projecting class, so the
model is not contradicted; but nothing here identifies it, and re-weighting on an
unidentified correspondence would be guesswork.
→ **difference, not deficiency / open question.** Useful step: document what
`gpe_cp` denotes and which experimental population it is meant to be, since its
BOLD weight 0.10 was matched to Courtney's NPAS1⁺NKX2.1⁺ (≈12 %) in `model_v07.md`
§3.6 while its projections are those of a striatum-projecting class.

**Goal relevance: medium** — the frozen ratios shape GPe dynamics and one fitted
BOLD signal, but no defensible replacement values exist.

**Within-list convergence:** **one voice** (Courtney 2023) for every specific claim
here. Wichmann 2019 independently supports only the coarser statement that
"arkypallidal neurons… make abundant connections and strongly inhibit direct and
indirect pathway striatal projection neurons and interneurons".

---

### 7.3 The two loops share no projection; three reviews report cross-channel convergence

**What BGM_22 does:** `caudate` and `putamen` are two complete, disjoint BG loops.
`model_v07.md` §1: they "share no projection; they meet only when their populations
are pooled into the shared GPi/GPe/STN BOLD monitors." The only physical difference
between them is the cortical proportion column (§7.5); all other `parameters.csv`
rows are identical. The claim the project wants to read off — which parameters had
to change under DBS — is read against that contrast, and only the putamen loop is
stimulated.

**What the literature reports:** Haber 2016 `[macaque]`: "convergence of
cortico-striatal terminals from cortical areas that are separated by 5 mm is 50 %
in nonhuman primates. The overlap decreased to below 20 % for regions separated by
30 mm", an exponential decay with cortical distance; and, on the two nuclei the
model splits, "the separation between the caudate nucleus and the putamen is merely
a structural one, based solely on the IC separation, not a functional one."
Emmi et al. 2020 `[macaque]`, on segregation through GPe and STN: "segregation
seems to be maintained with regard to the associative subregions of the striatum
(associative regions being contacted exclusively by other associative regions), but
not at the level of the motor circuits (motor regions being contacted by other
functional division, in particular the associative ones)." McGregor & Nelson 2019,
in the disease state the model fits, devote a subsection to "Loss of Functional
Segregation in Parkinsonian Basal Ganglia Circuits", concluding that "the aberrant
mapping of sensory responses or actions is likely to be another major form of PD
circuit dysfunction."

**The primary evidence:** Haber's figures come from Averbeck, Lehman, Jacobson &
Haber 2014, *J Neurosci* 34:9497, estimating projection overlap in frontal-striatal
circuits from macaque anterograde tracing; presented as established, and her own
lab's data. Emmi's convergence claim rests on Joel & Weiner's 1997 framework as
confirmed by Shink et al. 1996, which used **electron microscopy** to identify GPe
fibres and GPi-projecting STN neurons in the same tissue — a synapse-level
demonstration. Emmi is explicit that the status is unsettled: "Empirical evidence
for the open versus closed loop circuits hypothesis in non-human primates and
humans remains controversial", and offers two competing hypotheses for how
associative and motor circuits meet in the motor STN. McGregor & Nelson synthesise
receptive-field and somatotopy studies in parkinsonian animals (Rothblat &
Schneider 1995; Cho et al. 2002; Boraud et al. 2000; Pessiglione et al. 2005)
reporting diminished spatial specificity; the loss of segregation is presented as
well supported, its causal role as open.

**Why it matters (or does not) here:** this is the point on which the central
inference turns. With zero coupling, a DBS effect on the putamen can never reach the
caudate through the model, and any caudate-vs-putamen difference the fit reports is
guaranteed to trace back to the proportion table alone. The experimental picture is
that the strongest cross-channel route runs exactly the way the model would need it
— associative into motor, at GPe and STN — and that the disease state degrades
segregation further. The model's own pooling already presumes shared substrate:
`snr`, `stn` and the three GPe populations of both loops are summed into single ROIs
(`get_loss.py`, `bold_region_compartments`), so the model treats them as one nucleus
for observation while wiring them as two.
→ **experimentally grounded deficiency.** Proposal, in increasing cost: (a) state in
`model_v07.md` §1 that complete loop separation is a modelling choice contradicted
by macaque anatomy, so any caudate-vs-putamen contrast is an upper bound on channel
independence; (b) add the one convergence Emmi's reviewed anatomy supports most
directly — associative (caudate) GPe onto motor (putamen) STN — as a single weak
projection in its own optimizer cluster, so the fit can zero it; (c) treat a fitted
non-zero value as the model's own estimate of channel leakage, a result rather than
an assumption.

**Goal relevance: high** — it bounds what the caudate/putamen comparison, and hence
the DBS inference, can mean.

**Within-list convergence:** **consensus** — Haber 2016, Emmi et al. 2020 and
McGregor & Nelson 2019 all assert it, from corticostriatal tracing, from
pallidal/subthalamic tracing and EM, and from parkinsonian physiology. The species
base is macaque for both anatomical strands, the right base for a human-target
model.

---

### 7.4 The corticostriatal proportion table is applied unchanged to STN, GPe and thalamus

**What BGM_22 does:** `BGM_v07` in `model_creation_functions.py` passes the same
object to both consumers — `Microcircuit(cortical_proportions_dict=...)` and, thirty
lines later, `CorticalInputs(cortical_proportions_dict=...)`, both reading
`model_creation_kwargs["mc.cortical_proportions_dict"]`. So a putamen STN neuron's
500 cortical afferents split M1 0.28, PMv 0.18, SMA 0.15, S1 0.13, PMd 0.11, dlPFC
0.10, preSMA 0.05, and the same table sets the cortical afferent mix of `thal`,
`gpe_arky` and `gpe_cp`. `experimental_data/cortical_proportions/README.md` derives
the table exclusively from **corticostriatal** tracer counts (Borra et al. 2021,
2022) and defines the quantity as "the fraction of the corticostriatal afferents
onto a striatal neuron"; it never claims to describe another target, and the reuse
is not discussed there.

**What the literature reports:** Emmi et al. 2020 `[macaque]`, summarising Von
Monakow et al. 1978 in Table 2: Brodmann area 4 → dorsal and lateral STN; area 6 →
central; area 8 → ventral; **area 9 → "No connections to STh evidenced"; areas 3,
1, 2 → "No connections to STh evidenced"**. Areas 3/1/2 are S1. Haynes & Haber 2013
`[macaque]`, which Emmi calls "an extensive study on the cortical projections to
the STh in non-human primates", found dorsal prefrontal cortex (areas 9 and 46)
*does* project to the medial half of the STN — contradicting Von Monakow on area 9,
and Emmi reports both without resolving them. Isaacs et al. 2018 `[human]`, DWI plus
resting-state fMRI, via Emmi: "the striatum appeared to receive more projections
from the cortex when compared to the STh, with the exception of the orbitofrontal
cortex and the ventromedial prefrontal cortex."

**The primary evidence:** Von Monakow 1978 is anterograde autoradiography with
radioactive amino acids in *Macaca fascicularis*, 2–8 days post-injection, tabulated
by Emmi with injection site and result per area. Haynes & Haber 2013 is anterograde
plus bidirectional tracing in macaques, 12–14 days. Emmi presents the motor/premotor
topography (M1 → dorsolateral; SMA and ventral premotor → medial with an inverse
somatotopy, after Nambu et al. 1996, 1997, 2000, confirmed by Miyachi et al. 2006)
as established and replicated, and the prefrontal picture as contested. The S1
absence is a single-study negative result from 1978 — weaker than a positive
replication — but no study on Emmi's list reports an S1 → STN projection, and the
review's own conclusion, that the hyperdirect pathway arises chiefly from motor and
premotor cortex, is consistent with it.

**Why it matters (or does not) here:** the corticosubthalamic drive is the only
excitatory drive `stn` receives, so its regional mix sets what the model's STN sees;
fitted parameter 6 scales the whole stream and cannot redistribute it. The putamen
STN neuron currently draws 13 % of its cortical afferents from a region for which
the one primate study here that looked reports no projection. That matters more than
it would in a generic model, because the hyperdirect route is the DBS-relevant one
and `TODO.md` §15 already records that afferent DBS here "means `gpe_proto→stn`
only" — so the cortical afferent to STN is both the weakest-represented part of the
DBS story and mis-composed. Against that, the seven region streams into `stn` are
summed with a single weight, so a misallocation changes the *timing* of the drive,
not its mean — and timing is what the loss is made of.
→ **experimentally grounded deficiency.** Proposal: give `CorticalInputs` its own
proportion table. A defensible minimal version entirely within the evidence above:
zero S1 for STN; keep M1, premotor and SMA dominant per Nambu/Miyachi; treat dlPFC
as uncertain and record that Von Monakow and Haynes & Haber disagree. If the table
is not split, `cortical_proportions/README.md` must state that its numbers are used
for three structures they were not derived for, and `model_v07.md` §8 must repeat
it.

**Goal relevance: high** — for the STN ROI and for anything the DBS-on fit is asked
to say about hyperdirect drive.

**Within-list convergence:** **consensus that the mixes differ** — Emmi et al. 2020
documents the corticosubthalamic topography in detail and Haber 2016 the
corticostriatal one, and no region-share table transfers between them. The specific
S1 negative is **one voice** within Emmi (Von Monakow 1978), which is why the
proposal is "zero S1 or say why not" rather than a settled correction.

---

### 7.5 The intralaminar thalamus is absent, and four reviews make it a player

**What BGM_22 does:** one thalamic population per loop, mapped to the MD and VAp
ROIs (`get_loss.py`, `bold_region_compartments`). It receives `snr → thal` (0.06)
and emits the three largest weights in the CSV: `thal → str_d1` 7, `thal → str_d2`
6, `thal → str_fsi` 9.6 (`model_v07.md` §5). There is no centromedian or
parafascicular population anywhere.

**What the literature reports:** four of the six reviews assign the intralaminar
nuclei a substantive role. Courtney 2023 `[mouse]`: cortical inputs and the
parafascicular nucleus are together the largest sources of excitatory input to the
GPe, and PV⁺ neurons subdivide by whether they project to STN and SNr *or* to Pf
(PV⁺LHX6⁻). Emmi et al. 2020 `[macaque]`: "The thalamo-subthalamic pathway arises
mainly from the parafascicular nucleus of the thalamus and from the centromedian
nuclei", CM to the motor division of the STN and Pf to its medial and rostral
territories; Table 3 places thalamic glutamatergic contacts on STN **distal
dendrites**. Tepper et al. 2018 `[rat, macaque]`: FSIs receive intralaminar thalamic
input, "significantly denser in primates than rodents". Wichmann 2019
`[primate, human]`: thalamostriatal inputs are involved in parkinsonism-associated
remodelling, "likely because of the overall loss of neurons in the intralaminar
nuclei", with a matching dropout of thalamic innervation of motor-cortex layer 5.

**The primary evidence:** Emmi's CM/Pf topography is sourced to Sadikot et al. 1992
and Tandé et al. 2006 (the latter tabulated in Table 2: Fluoro-gold retrograde
injection into the macaque STN labelling the parafascicular nucleus), plus Hamani
et al. 2004 for the terminals' glutamatergic immunoreactivity; presented as
established. Tepper's primate-density claim carries Rudkin & Sadikot 1999 and
Sidibé & Smith 1999 and is stated without hedging. Wichmann's intralaminar-loss
claim carries three references and is presented as established. Courtney's Pf claim
rests on whole-brain viral input maps `[mouse]` and is presented as a recent
revision of the classic picture.

**Why it matters (or does not) here:** partly it does not, for a structural rather
than a scientific reason: the Berlin ROI set has MD and VAp and no CM/Pf, so there
is nothing to fit an intralaminar population against, and adding one would add free
parameters with no data to constrain them. That is a sufficient defence. What is not
defensible is silence, because the single `thal` population does two jobs the
anatomy assigns to different nuclei: it is read out as MD/VAp BOLD, and it carries
the thalamostriatal projection, of which the intralaminar nuclei are a major source
— specifically, per Tepper, the source of thalamic input to FSIs, which here is the
largest weight in the CSV. Haber 2016's reference list also documents convergent
input from thalamic *motor* nuclei to the dorsal striatum (McFarland & Haber, ref
71), so a VA/VL-derived thalamostriatal projection is not wrong — but the
FS-targeting part of it is attributed to the intralaminar nuclei by the one review
covering striatal interneurons.
→ **difference, not deficiency / open question.** Suggestion: `model_v07.md` §5
should state that `thal` stands for the BG-receiving motor/associative thalamus
(MD, VAp), that the intralaminar nuclei — supplying a substantial part of the real
thalamostriatal and all of the thalamosubthalamic input — are not represented, and
that `thal → str_fsi` therefore carries a load the anatomy assigns elsewhere.

**Goal relevance: medium** — two of seven ROIs are thalamic, and the largest weight
in the model depends on the identification.

**Within-list convergence:** **consensus** — Courtney 2023, Emmi et al. 2020,
Tepper et al. 2018 and Wichmann 2019 each independently give the intralaminar
nuclei a role, in four different target structures, across mouse, macaque, rat and
human.

---

### 7.6 The subthalamo-striatal projection exists in primate and the model has none

**What BGM_22 does:** `stn` projects to `snr`, `gpe_proto`, `gpe_arky` and `gpe_cp`
and nowhere else (`model_v07.md` §5). There is no STN → striatum projection in v07
or v08.

**What the literature reports:** Emmi et al. 2020 `[macaque]`: "approximately 17 %
of the STh neurons labeled present a single axon projecting toward the striatum"
(Sato et al. 2000b), with a topography — putamen projections arise mainly from the
dorsolateral (motor) STN, caudate projections from the ventromedial associative and
limbic regions. Morphologically these are "long, varicose axons with few
collaterals, scattered throughout wide areas of both striatal components", likely
exerting "an en passant type of excitatory influence on vast populations of
striatal cells." The same Sato 2000b breakdown of primate STN neurons by axonal
target — SNr+GPi+GPe 21.3 %, SNr+GPe 2.7 %, GPi+GPe 48 %, GPe only 10.7 %, striatum
only 17.3 % — puts the striatum-projecting class on a par with the GPe-only class.

**The primary evidence:** Sato, Parent, Levesque & Parent 2000b, single-axon
reconstruction after tracer injection in macaque — the strongest available method
here, and the same study establishing that primate STN neurons have five branching
patterns rather than the rodent bifurcation. Emmi hedges it twice:
subthalamo-striatal projections are "scarce compared to other subthalamic targets"
(Nauta & Cole 1978; Smith & Parent 1986; Parent & Smith 1987; Sato 2000b), and "the
terminal arborizations of these labeled neurons could not be visualized in this
study". The existence and the fraction of cell bodies are established; the synaptic
weight is not measured.

**Why it matters (or does not) here:** for the BOLD fit, little — 17 % of a
100-neuron STN making sparse en-passant contacts on 1000 striatal neurons would
barely perturb a striatum already 98 % open-loop stream
(`input_streams/README.md` §4.1). Where it bears is on what a DBS-on fit may claim:
this is the one anatomical route by which STN stimulation could reach the striatum
directly, and the model forecloses it, so a fitted "DBS changed the striatum" result
here can only be a downstream, multi-synaptic effect by construction — worth knowing
before the on-fit is interpreted. The review's own "scarce" hedging and the absent
terminal quantification are why this is not a change proposal.
→ **difference, not deficiency / open question.** Suggestion: record it in
`model_v07.md` §5 alongside the note about absent intra-striatal projections, and in
whatever document eventually states what the DBS-on fit may conclude.

**Goal relevance: low** for the fit, **medium** for the DBS interpretation.

**Within-list convergence:** **one voice** (Emmi et al. 2020), citing four
independent primary studies converging on the pathway's existence and its scarcity.

---

### 7.7 One `snr` population stands for both output nuclei, in a species where the other dominates

**What BGM_22 does:** a single `snr` population per loop, 100 neurons, receiving
`str_d1 → snr` (0.06), `stn → snr` (0.04) and `gpe_proto → snr` (0.015), emitting
`snr → thal` (0.06). It is pooled into the **GPi** BOLD ROI (`get_loss.py`,
`"GPi": [("snr","caudate"), ("snr","putamen")]`), and `get_loss.py` says so inline
where it registers the passing fibre: "snr__thal is actually gpi__thal". Its delays
come from the rat set of Kumaravelu, Brocker & Grill 2016 (`model_v07.md` §5).

**What the literature reports:** Emmi et al. 2020, citing Hardman et al. 2002: "the
SNr plays a much more important role as an output structure of the basal ganglia in
rodents compared to non-human primates, while the opposite seems to be true for the
GPi" `[rodent vs macaque vs human]`. Consistently, the Sato 2000b `[macaque]`
breakdown in 7.6 has ~69 % of STN neurons reaching the GPi (21.3 + 48) against
~24 % reaching the SNr (21.3 + 2.7).

**The primary evidence:** Hardman, Henderson, Finkelstein, Horne, Paxinos &
Halliday 2002 is a comparative stereological study of neuron numbers across
rodents, primates and humans — Emmi uses it repeatedly, including for the human STN
counts in Table 1, so it is a quantitative cross-species measurement rather than an
impression. Sato 2000b is single-axon reconstruction, as above. Emmi presents both
as established and draws an inference about the STN's changing role "through
phylogenesis".

**Why it matters (or does not) here:** GPi is one of seven fitted ROIs and `snr`
produces it, so the identification is load-bearing rather than cosmetic. The
population's parameters and delays are inherited from a rat model in which that
nucleus name denotes the dominant output structure, whereas in primate the dominant
output is the one being read out, and their STN inputs differ in proportion by
roughly threefold. And the model's output nucleus receives the direct pathway from
`str_d1` only, whereas in primate both GPi and SNr do, from partly different
striatal territories. Neither is fatal — one output ROI and one output population is
coherent — but the naming hides the choice, and only one inline code comment records
it.
→ **difference, not deficiency / open question**, on evidence grounds: nothing here
says a merged output nucleus cannot reproduce a GPi BOLD time course, and there is
no SNr ROI to separate them against. Suggestion: `model_v07.md` §3.6 and §5 should
state plainly that `snr` is the GPi in every data-facing role, and record Hardman's
rodent/primate asymmetry as a reason to treat rat-derived weights and delays into
and out of it as provisional — compounding the rat/macaque delay disagreement
`model_v07.md` §5 already records.

**Goal relevance: medium** — it does not change what is fitted, but it changes how a
fitted GPi parameter should be read.

**Within-list convergence:** **one voice** (Emmi et al. 2020), resting on two
independent primary macaque/comparative studies.

---

### 7.8 Audit anchor 1a — the Liang elevation is corroborated as a claim and contested as a value

**What BGM_22 does:** `parameters.py` `mc.firing_rate_dict` sets the unsimulated
striatal surround to `{FS: 10.5, dSPN: 25.0, iSPN: 33.0}` Hz, baked into every v07
cache; `get_loss.get_firing_rate_loss` scores simulated rates against bands centred
on the same values, `str_d1` (12.67, 37.33) and `str_d2` (21.22, 44.78).
`activity_striatum/README.md` derives both from Liang, DeLong & Papa 2008 Table 1,
Off state, n = 140, two chronically MPTP-lesioned rhesus monkeys with levodopa
withdrawn, and states the one assumption it rests on — that the direction of the
levodopa response identifies the receptor class — quoting the paper's own admission
that this "cannot be excluded with the available evidence".

**What the literature reports:** McGregor & Nelson 2019 place Liang exactly where
the README places it and add the other side: "evidence from humans and nonhuman
primates is conflicting, showing either marked increases in MSN firing (Liang
et al., 2008; Singh et al., 2016) or no change (Deffains et al., 2016)." They add
the methodological reason the primate picture is weak — "changes in iMSN and dMSN
firing were largely inferred from recordings in downstream nuclei". Separately,
Haber 2016 `[macaque]` gives the normal baseline the README's framing depends on:
MSNs have "a very low spontaneous discharge rate (0.5–1 spike/s)" with task-related
rates of 10–40 spikes/s in awake behaving monkeys.

**The primary evidence:** McGregor & Nelson's contrast is a three-study citation
presented as an open conflict rather than resolved either way; Deffains et al. 2016
is a source `input_streams/README.md` already cites for a different purpose, so the
project has met it. Haber's 0.5–1 spike/s is stated without an inline citation in
that sentence — a review's summary of standard primate electrophysiology, and I flag
it as such rather than as a primary-backed number.

**Why it matters (or does not) here:** the audit's positive finding is the
substantial one. Liang is not an idiosyncratic choice: an independent authoritative
review names it as one of two primary sources for elevated parkinsonian MSN firing,
so the README's central decision survives, and its "Why these rates are so high"
section is corroborated on the normal side by Haber. What the README does not record
is that the elevation is contested in primate by Deffains et al. 2016 — its "What we
did not consider" section lists only identified-cell-type *rodent* alternatives.
Since the rates are baked into every cache and centre the gate bands, a reader
deciding whether to rebuild deserves to know the field treats this as unsettled.
This is not a reason to change the value: no alternative primate number is on offer,
Deffains reports *no change* rather than a competing rate, and the README's own
sensitivity bound (22.4–25.0 Hz for dSPN) is a smaller uncertainty than the conflict
implies.
→ **difference, not deficiency / open question.** Suggestion: add Deffains et al.
2016 and Singh et al. 2016 to `activity_striatum/README.md` as the recorded
conflict, citing McGregor & Nelson 2019 for the framing, and note that if the "no
change" side is right the entire missing-GABA surround is drawn at a rate an order
of magnitude too high — the one scenario in which the caches would need rebuilding
for a reason external to the model.

**Goal relevance: high** — the rates gate every evaluation and are baked into every
cache.

**Within-list convergence:** **one voice** for the conflict (McGregor & Nelson 2019
is the only review here discussing striatal firing rates in parkinsonism at this
resolution); Haber 2016 independently corroborates the normal-primate baseline.

---

### 7.9 Audit anchor 1b — the iSPN > dSPN ordering is independently corroborated

**What BGM_22 does:** dSPN 25.0 Hz, iSPN 33.0 Hz — the indirect-pathway population
fires ~32 % faster, in both the surround streams and the band centres. The README is
explicit that this ordering comes from the response-direction assumption and nothing
else, and that "there is no cell-type identification anywhere in this dataset".

**What the literature reports:** McGregor & Nelson 2019 Fig. 4 schematises
parkinsonian dMSNs as *decreased* and iMSNs as *increased* relative to healthy, and
the text supports it from identified-cell recordings: "In anesthetized parkinsonian
rats, antidromically identified dMSNs showed decreased firing as compared to healthy
animals, while presumed iMSNs showed elevated firing (Mallet et al., 2006; Kita and
Kita, 2011). Recent single-unit recordings of optically and antidromically
identified iMSNs and dMSNs in freely moving parkinsonian mice similarly show
bidirectional changes in firing rate (Ryan et al., 2018; … Sagot et al., 2018)."
`[rat, mouse]`

**The primary evidence:** antidromic identification in anesthetized 6-OHDA rats
(Mallet 2006; Kita & Kita 2011) and optogenetic-plus-antidromic identification in
freely moving 6-OHDA mice (Ryan 2018; Sagot 2018) — precisely the class of study
`activity_striatum/README.md` names under "What we did not consider" as the thing
that "would remove the response-direction assumption entirely". McGregor & Nelson
present the rodent bidirectionality as consistent and the primate picture as
conflicting, so *direction* is the part they treat as established.

**Why it matters (or does not) here:** a point in the model's favour, and a specific
one. The README's weakest link is the inference from levodopa-response direction to
receptor class; the ordering that inference produces — iSPN above dSPN in the
depleted state — is exactly what four identified-cell-type rodent studies find
without needing the inference at all. That does not validate the *magnitudes*
(rodent MSN rates are 1–5 Hz, as the README notes), and the species mismatch is
real; what it validates is the relative arrangement, which is what most affects the
model's direct/indirect balance.
→ **difference, not deficiency / open question**, in BGM_22's favour. Suggestion:
the README's "What we did not consider" section presents identified-cell-type
recordings only as a threat to the current choice; it is worth recording that on the
one question those studies and Liang both answer — which pathway fires faster off
medication — they agree.

**Goal relevance: medium** — it does not change a number, but it removes weight from
the single assumption the README names as its price of admission.

**Within-list convergence:** **one voice** (McGregor & Nelson 2019), citing four
independent primary studies in two rodent species and two preparations.

---

### 7.10 Audit anchor 1c — the FS rate, the waveform-class problem, and one point in the model's favour

**What BGM_22 does:** FS 10.5 Hz with band (3.42, 17.58), derived in
`activity_striatum/README.md` as an n-weighted normal-primate level (Marche &
Apicella 2021; Yamada et al. 2016; Adler et al. 2013) times a chronic
dopamine-depletion factor of 1.0 imported from rodents (Mallet 2006; Hernandez 2013;
He 2024). Its caveat 3 already records that "primate 'FSI' is a waveform class, not
a cell type… The model's `FS` population is PV+ specifically."

**What the literature reports:** Tepper et al. 2018 makes the heterogeneity worse
than that caveat, and better in one respect. Worse: the review's thesis is that the
striatal GABAergic interneuron population comprises at least four classes beyond the
classical FSI/PLTS/CR triad — tyrosine-hydroxylase (THINs), neurogliaform (NGF),
fast-adapting (FAI) and spontaneously active bursty (SABI) neurons — and its Table 1
gives NGF a connection probability to SPNs comparable to the FSI's (~80 %), with a
slow GABA_A response the FSI does not produce; its conclusion is that "previous
hypotheses in which striatal GABAergic interneurons modulate and/or control the
firing of spiny neurons principally by simple feedforward and/or feedback inhibition
are at best incomplete." Even within the PV⁺ class, Garas et al. 2016 split FSIs by
secretagogin in **rat and primate**, with different postsynaptic preferences — Scgn⁺
cells preferentially targeting dSPN somata, Scgn⁻ axons preferentially innervating
iSPNs. Better: FSIs receive intralaminar thalamic input, "significantly denser in
primates than rodents" (Rudkin & Sadikot 1999; Sidibé & Smith 1999) `[rat, macaque]`.

**The primary evidence:** the new interneuron classes each rest on named
electrophysiological and genetic-labelling studies in mouse and rat; Tepper presents
them as established discoveries of the preceding decade and their *functional*
consequences as emerging. The secretagogin split is a single study (Garas et al.
2016) but one of the few striatal-interneuron findings here with a primate arm. The
primate density of thalamic input to FSIs carries two independent citations and is
stated without hedging.

**Why it matters (or does not) here:** the interneuron-diversity finding is real but
has little purchase. The model's 29 FS neurons supply 8.8 of a dSPN's ~509 FS
afferents (`input_streams/README.md` §4.1); everything else is a stream. Adding an
NGF class would mean another synthetic stream with no measured rate, no measured
kernel and no ROI — more free structure, not more constraint. Haber 2016
independently notes that MSNs "account for over 90 % of cells in nonprimate species,
and probably far less in the primates", so the model's 2.9 % FS fraction (del Rey
et al. 2022, via `microcircuit.py` `props_delRey`) may understate primate
interneuron abundance — but Haber gives no number, so this cannot be acted on. The
favourable finding is the thalamic one: `thal → str_fsi` is 9.6, the largest weight
in `parameters.csv`, larger than `thal → str_d1` (7) and `thal → str_d2` (6) despite
FS being 6 % of the striatal neurons. That looks anomalous until Tepper is read —
dense intralaminar thalamic innervation of striatal FSIs is documented, and it is
*denser in primates than rodents*, so a model of a human striatum having its
strongest thalamostriatal weight onto FS is anatomically motivated, whatever route
produced the number.
→ **difference, not deficiency / open question.** Suggestion: record the
Tepper/Rudkin/Sidibé support for the large `thal → str_fsi` weight in `model_v07.md`
§5, where it stands uncommented beside the note about its `ampa` target; and add
Tepper 2018 to `activity_striatum/README.md` caveat 3, since it sharpens the
waveform-class problem into a specific, primate-confirmed subdivision.

**Goal relevance: low** for interneuron diversity, **medium** for the thalamic-input
corroboration.

**Within-list convergence:** the interneuron diversity is **one voice** (Tepper
2018, the only striatal-interneuron review here). Greater FS responsiveness to
cortical and thalamic drive than SPNs is **consensus** — Tepper 2018 and Haber 2016
assert it independently, in rat and macaque (see 7.15).

---

### 7.11 Audit anchor 1d — the non-striatal rate bands have no derivation at all

**What BGM_22 does:** `get_loss.get_firing_rate_loss` scores nine populations per
loop against `plausible_ranges`. Three carry a documented derivation (`str_d1`,
`str_d2`, `str_fsi`, with a fifteen-line comment block pointing at
`activity_striatum/README.md`). Six do not: `gpe_proto` (75, 85), `gpe_arky`
(15, 20), `gpe_cp` (75, 85), `stn` (28, 80), `snr` (21, 93), `thal` (15, 30). Their
entire documentation is one line — "stn and snr (gpi): from [Li et al., 2015]" —
which names no journal, no DOI, no species, no preparation and no disease state, and
which does not cover the three GPe bands or the thalamic band at all. No
`experimental_data/` document exists for them.

**What the literature reports:** two reviews specify the parkinsonian direction for
each of these nuclei, and both mark the thalamus as the one that cannot be
specified. McGregor & Nelson 2019 Fig. 4 tabulates healthy vs parkinsonian rate and
pattern per nucleus with citations: GPe prototypic *decreased* (Pan 1988; Filion
1991; Boraud 1998; Heimer 2002; Soares 2004; Mallet 2008, 2012, 2016), GPe
arkypallidal *increased*, STN *increased with bursts* (Bergman 1994; Benazzouz
2002), GPi *increased* (Miller & DeLong 1988; Hutchison 1994; Boraud 1996, 1998;
Heimer 2002; Starr 2005; Muralidharan 2016), SNr *increased* (Wichmann 1999), and
**thalamus "?"**. Wichmann 2019 states the thalamic case in words: "parkinsonism is
associated with relatively minor changes in firing in the basal ganglia receiving
portion of the thalamus", and notes the traditional prediction that interventions at
the ventral motor thalamus should affect bradykinesia has "not been confirmed in
parkinsonian animals or human patients."

**The primary evidence:** McGregor & Nelson's figure is a citation-dense summary of
primate and rodent single-unit recordings, presented as the field's settled
qualitative picture; the review is explicit that the *causal* status of these changes
is disputed but not their existence. Wichmann's thalamic statement carries five
references and is presented as established. Neither review gives absolute rate ranges
per nucleus, so neither can supply replacement band endpoints — they constrain
direction and confidence, not values.

**Why it matters (or does not) here:** this is the audit's clearest procedural
finding, procedural rather than empirical, which is why the verdict differs from the
rest. Six of the nine bands that gate every evaluation — including the two narrowest,
`gpe_proto` and `gpe_cp` at (75, 85), ±6 % of centre — rest on an unresolvable
citation. The striatal bands were derived with visible care, including the deliberate
statement that they are "a plausibility band, not a confidence interval". The other
six have no such statement and no traceable source, so a reader cannot tell whether
they are healthy-primate, parkinsonian, human intraoperative, or model outputs from
another paper. The disease state is not neutral: per McGregor & Nelson, GPe
prototypic goes *down* while GPi/SNr and STN go *up*, so a healthy-derived and a
parkinsonian-derived band are different bands — and `TODO.md` §10 already records
that both DBS conditions score ~0.87 against these bands at the default vector, i.e.
the gate currently fires on everything and nobody can tell whether the bands or the
model are at fault. §10 plans to calibrate the *threshold*; nothing plans to
establish the *bands*.
→ **methodological deficiency.** Proposal: resolve "[Li et al., 2015]" to a full
citation with species, preparation and medication state; give each of the six
non-striatal bands a source, species and disease state in the style of
`activity_striatum/README.md`, even where that source is "not established, band
chosen wide on purpose"; and mark the thalamic band explicitly as unconstrained in
parkinsonism on the authority of McGregor & Nelson Fig. 4 and Wichmann. Do this
before §10's threshold calibration, since calibrating a threshold against
undocumented bands sets one unknown from another.

One fact for whoever writes that document, from Wichmann 2019: dopamine loss in PD
is not confined to the striatum. D2-like receptors sit on striatopallidal terminals
in the primate GPe, D1 and D2 receptors on preterminal axons and glutamatergic and
GABAergic terminals in the STN, and D1-like receptors on striatopallidal and
striatonigral terminals in GPi and SNr; "the comparatively maintained pallidal and
nigral dopamine levels in early parkinsonism may thus compensate for the striatal
dopamine loss in early PD, helping to maintain GPi/SNr firing rates at levels close
to normal", while in the fully parkinsonian state the active fraction of D1-like
receptors at those sites increases `[macaque, human]`. So a GPi or STN band depends
on disease *stage*, not merely on the presence of PD — and BGM_22 has no
extrastriatal dopamine term at all, so its fitted `base_mean` for `snr` and
`gpe_proto` (parameters 7–8) silently absorbs whatever that effect is. This also
bears on `TODO.md` §30, which frames the dopamine question as striatal only.

**Goal relevance: high** — the gate decides whether the BOLD run happens, and
`TODO.md` §10 records it currently fires on every individual.

**Within-list convergence:** **consensus** on the per-nucleus directions (McGregor &
Nelson 2019 Fig. 4 and Wichmann 2019 agree throughout) and **consensus** that the
thalamic rate is the unconstrained one (both mark it). The extrastriatal-dopamine
addendum is **one voice** (Wichmann 2019).

---

### 7.12 Audit anchor 2a — the cortical-proportions method is what both anatomy reviews prescribe

**What BGM_22 does:** `cortical_proportions/README.md` rejects human diffusion
tractography as a quantitative source — it reports "*topography*…, not what fraction
of a neuron's input each area supplies" — anchors on quantitative macaque retrograde
tracing (Borra et al. 2021, 2022), uses topography papers only to split bins the
tracer tables leave grouped, and admits the one quantitative human study (Cacciola
et al. 2017) "as direction, not as values" because the Desikan-Killiany parcellation
cannot separate M1/PMd/PMv or SMA/preSMA/dlPFC. Every derivation step is tagged (M)
measured, (A) assumed or (J) judgement.

**What the literature reports:** both anatomy reviews independently prescribe this
hierarchy. Emmi et al. 2020 `[human, macaque]`: "the anatomical accuracy of brain
connections derived from diffusion MRI appears to be inherently limited (Jones
et al., 2013; Thomas et al., 2014)… directionality of fibers can not be inferred,
whilst the values regarding the connection strength between structures must be
interpreted with care", concluding that "MRI derived tractography should be based on
the anatomical evidence derived from non-human primate tracing studies". On
tract-strength numbers specifically, via Isaacs et al. 2018: "tract strength" counts
streamlines from the seed region "but does not quantify the actual number of white
matter fibers, and its values are related to the size of the target structure. Hence,
results related to tract strengths should not be over interpreted." Haber 2016
`[macaque]` builds her entire quantitative account of corticostriatal topography from
macaque tracer injections and uses the human material she cites for topography rather
than for fractions.

**The primary evidence:** Emmi's tractography critique carries two dedicated
methodological citations (Jones et al. 2013; Thomas et al. 2014) plus the gyral-bias
caveat from Isaacs et al. 2018, and is stated as the review's own methodological
position rather than in passing — it recurs in three sections. Haber's position is
implicit in her sourcing rather than argued, so I count it as corroborating practice,
not an independent assertion.

**Why it matters (or does not) here:** this is a substantial finding in BGM_22's
favour and the main thing the audit of anchor 2 establishes. The README's central
methodological choice is not a compromise forced by data availability — it is what
the field's own anatomy reviews say the correct procedure is, including the specific
refusal to read connection-strength percentages off tractography, which is exactly
what Cacciola's numbers are. Step 5's use of Cacciola as a directional nudge only
(moving putamen S1 up by ~0.03 rather than adopting the 0.51 ratio tractography
implies) is precisely what Emmi's Isaacs caveat licenses. Caveat 6 — "Tracer counts
are cell counts, not synapse counts" — is also the right caveat, and Emmi
independently confirms that laminar origin and terminal distribution differ across
areas (Coudé et al. 2018: motor-cortex corticosubthalamic projections originate from
layer V; Borra 2021 is itself a laminar-origin study).
→ **difference, not deficiency / open question**, in BGM_22's favour. No change
requested. Worth adding to the README: Emmi et al. 2020 as the explicit
methodological citation for the "human data cannot answer this" position, which
currently rests on the README's own assessment.

**Goal relevance: high** — the proportions are the only physical difference between
the loops, so the credibility of their derivation method is the credibility of the
loop contrast.

**Within-list convergence:** **consensus** — Emmi et al. 2020 argues it explicitly
and repeatedly; Haber 2016 follows it in practice throughout.

---

### 7.13 Audit anchor 2b — renormalising over seven ROIs gives the omitted afferents the retained regions' time course

**What BGM_22 does:** `cortical_proportions/README.md` defines the quantity as the
fraction of corticostriatal afferents from each ROI "**renormalised over just these
seven ROIs**", and says why: the striatum "also receives heavily from cingulate,
insula, temporal, posterior parietal, orbital and ventrolateral prefrontal cortex,
and **none of those has an ROI in the Berlin data**. In the macaque tracer counts
below they are 25–60 % of all labelled corticostriatal cells." Each column sums to
1, enforced by `spike_input_cortex.validate_cortical_proportions()`, so all 7000 of
an SPN's afferents are assigned to the seven retained regions and driven by their
deconvolved rate series.

**What the literature reports:** Haber 2016 quantifies the omitted input and shows
it is neither small nor spatially uniform `[macaque]`: "the volume occupied by the
collective dense terminal fields from the vmPFC, dACC, and OFC is approximately
22 % of the striatum, a larger cortical input than would be predicted by the relative
cortical volume of these areas." Those fields are concentrated rostrally and
ventrally — the caudate head and rostral putamen, exactly the territory the README's
own weighting (caudate head 0.75, rostral putamen 0.3) makes dominant. Emmi et al.
2020, reporting Isaacs et al. 2018 `[human]`, adds that OFC and vmPFC are the two
regions whose tract strength is *stronger* for the STN than for the striatum, so the
omission propagates to the `CorticalInputs` streams as well.

**The primary evidence:** Haber's 22 % is a measured volume fraction from her lab's
macaque anterograde tracing programme (the review's Figs. 3–5 are built from it),
presented as established. Borra's Table 2, reproduced in BGM_22's own README,
independently shows rostral cingulate at 21.5–30.6 % for the caudate head rows and
insula plus temporal at a further 7–18 % — so the omitted mass is documented inside
the project already.

**Why it matters (or does not) here:** the ROI set is fixed by the experimental file
and cannot be extended, so the *omission* is not a deficiency. What is a substantive
and unstated claim is what renormalisation does with the omitted afferents: it does
not drop them, it reassigns them. Because every column sums to 1 and each series is
normalised to mean 5 Hz, an SPN's full 7000 afferents are driven by the seven
retained series — so the model asserts that the 25–60 % of afferents arising in
cingulate, insula, temporal, orbital and ventrolateral prefrontal cortex fluctuate,
TR by TR, exactly as the retained seven do. The loss is a per-region BOLD
*time-course* correlation, so this assertion is about precisely the fitted quantity.
And it is asymmetric between the loops: the omitted mass is largest in the caudate
rows of Borra's table, so the caudate loop's drive is the more heavily reconstructed
of the two, while the loop contrast is what the inference reads. The README's caveat
2 states the omission; it does not state that renormalisation is an assumption about
covariance.
→ **experimentally grounded deficiency**, on the narrow ground that the model makes
a checkable claim about the omitted input's temporal profile which the anatomy says
is unlikely to hold — the omitted regions are limbic and associative, functionally
distinct from the seven retained motor and dorsolateral-prefrontal ones. Proposal,
cheapest first: (a) state the assumption explicitly in
`cortical_proportions/README.md` — renormalisation assigns the omitted afferents the
retained regions' time course, a covariance assumption rather than a normalisation
convenience; (b) if it is to be tested, a strictly weaker alternative costs nothing
anatomically — let the columns sum to the retained fraction (≈0.4–0.75 depending on
the row) and drive the remainder with a flat series at the same 5 Hz mean, which
preserves the mean drive exactly while dropping the covariance claim; (c) either
way, report the caudate/putamen asymmetry of the omitted fraction beside the
existing note that the corrected proportions raised `corr(caudate mix, putamen mix)`
to 0.843.

**Goal relevance: high** — it acts directly on the fitted quantity and
asymmetrically between the two loops.

**Within-list convergence:** **consensus** that the omitted input is large and
concentrated — Haber 2016 quantifies it in the striatum, Emmi et al. 2020 (via
Isaacs 2018) shows the same regions are disproportionately strong for the STN.
BGM_22's own Borra table is a third, independent confirmation.

---

### 7.14 The simulated cube is smaller than one spiny neuron's dendritic field

**What BGM_22 does:** the microcircuit lattice is 1000 neurons on a 227.6 µm cube at
84 900 neurons/mm³ (`model_v07.md` §7.1; `microcircuit.py` `density = 84900.0`). The
connection radius is clamped to `L_max/2` = 113.8 µm for all three post types because
3σ reaches 568 µm–1.2 mm (§7.2), and everything beyond is replaced by synthetic
streams: a dSPN receives ~67 simulated GABAergic afferents against ~2573 synthetic
ones, "about 2 % circuit and 98 % open-loop stream" (§7.3).

**What the literature reports:** Haber 2016 `[macaque]`, in the passage on
corticostriatal convergence: "Since MSNs have a dendritic field 0.5 mm, terminals
from more than one cortical area can converge onto a single striatal cell." The
number is used to license a functional inference, so the review treats it as a
settled anatomical constant.

**The primary evidence:** Haber attributes it to ref 107, Wilson CJ, "The basal
ganglia", in Shepherd GM (ed), *The Synaptic Organization of the Brain*, 5th ed.,
2004, pp. 361–413 — a textbook chapter rather than a primary measurement, a real
limitation on the citation's weight. I did not verify it against a primary source and
treat it as no more than a review's settled constant.

**Why it matters (or does not) here:** `TODO.md` §24 already plans a larger, sparser
cube and tabulates the cost at 350, 500, 800 and 1200 µm, so the task is to review
the plan rather than rediscover the issue. §24 justifies the enlargement entirely by
how much `f(d)` varies over the cube (1.1× now, 4.4× at 800 µm) and by the van Albada
downscaling limitation — a statistics argument. There is an independent anatomical
argument §24 does not record: at 227.6 µm the simulated volume is smaller than the
dendritic field of a single one of the neurons in it, so no simulated SPN's dendritic
tree fits inside the modelled tissue, and the fitted `p(d)` kernel is applied over a
range the cube never covers. That argument also sets a target size §24's table does
not name: ≥500 µm to hold one dendritic field, which in §24's own table costs 0.28
percentage points of simulated GABA input (1.65 % → 1.37 %) and buys a 1.7× `f(d)`
range — cheaper than the 800 µm option §24 highlights, with an anatomical rather than
statistical rationale.
→ **experimentally grounded deficiency**, whose fix is already planned. Proposal:
record Haber's 0.5 mm dendritic-field figure in `TODO.md` §24 alongside the existing
`f(d)` argument and in `model_v07.md` §7.1 beside the 227.6 µm figure. Do not treat
it as licensing a specific new value beyond "at least 500 µm", the citation being a
textbook chapter.

**Goal relevance: medium** — it does not change the fit directly, but it is the
anatomical statement of why 98 % of the striatum is a stream, and it sharpens an
already-planned change.

**Within-list convergence:** **one voice** (Haber 2016), and via a textbook citation
rather than a primary study — the weakest evidential basis of any point here, which
is why the proposal is limited to documentation.

---

### 7.15 Two things the input-stream design gets anatomically right

**What BGM_22 does:** two choices in the cortical stream generator. First,
`model_v07.md` §7.5: "Regions, by contrast, are drawn **apart**: each has its own
axon pool, so within a bin the streams of different regions are independent. Their
co-fluctuation comes only through the correlated BOLD-derived rate series each pool
is driven by." Second, `input_streams/README.md` §4.7: Kincaid's 1.4 % shared
fraction is applied flat, with no distance dependence, because "Corticostriatal
topography is real at larger scales; the cube is too small for it to bite."

**What the literature reports:** Haber 2016 `[macaque]`: overlap of corticostriatal
terminals from areas 5 mm apart is 50 %, falling below 20 % at 30 mm; terminal fields
from different functional areas interdigitate and overlap at their edges and in
embedded clusters. On FS interneurons specifically: "corticostriatal terminals from
sensory and motor cortex converge within the striatum. Here, axons from each area
synapse onto single fast-spiking GABAergic interneurons. Interestingly, these
interneurons are more responsive to cortical input than the MSNs."

**The primary evidence:** the overlap estimates are Averbeck et al. 2014 (Haber ref
15), macaque anterograde tracing. The FS convergence statement carries three
references (Haber refs 104–106): Ramanathan, Hanley, Deniau & Bolam 2002
*J Neurosci* 22:8158, "Synaptic convergence of motor and somatosensory cortical
afferents onto GABAergic interneurons in the rat striatum" `[rat]`; Mallet et al.
2005 *J Neurosci* 25:3857, feedforward inhibition of projection neurons by
fast-spiking GABA interneurons in the rat striatum in vivo `[rat]`; and Takada,
Tokuno, Nambu & Inase 1998 *Brain Res* 791:335, on SMA corticostriatal input zones
overlapping those from contralateral rather than ipsilateral M1 `[macaque]`. Haber
presents all of it as established.

**Why it matters (or does not) here:** three readings, two favourable.
*Independent per-region axon pools are the anatomically correct choice.* Haber's
convergence data concern **terminal-field overlap in the striatum** — that a striatal
neuron in an overlap zone receives from several cortical areas — which BGM_22 already
represents by giving every SPN afferents from all seven regions. What the model
asserts by drawing regions apart is that no single presynaptic neuron belongs to two
regions' pools, which is true by construction: a corticostriatal axon arises in one
cortical area. The design is right, and co-fluctuation through the rate series is the
correct locus for inter-regional correlation — worth saying because it is the one
place in the stream design where "independent" is not a simplification. *The
no-distance-dependence argument survives too*: Averbeck's overlap decays over
centimetres of cortical separation and hence millimetres of striatum, while the cube
is 227.6 µm, two orders of magnitude below the scale at which corticostriatal
topography varies, so §4.7's reasoning is correct and now has a quantitative
anatomical citation behind it rather than an assertion. *The one genuine gap is
already known*: `TODO.md` §26 records that Ramanathan 2002 and Choi 2018 report
higher cortical convergence onto FS interneurons than onto SPNs, which the
single-pool derivation does not capture. Haber 2016 invokes the same Ramanathan study
independently and states the consequence as fact, and Tepper 2018 independently
reports denser primate thalamic innervation of FSIs (7.10). What Ramanathan measured,
per its title and Haber's use of it, is cross-area convergence onto individual
interneurons rather than raw afferent count, and BGM_22's FS neurons already receive
all seven regions — so the finding does not license changing `N_cortical_inputs_dict`
FS = 2800; it licenses recording that the evidence behind §26 is stronger and better
replicated than §26 states.
→ **difference, not deficiency / open question**, largely in BGM_22's favour.
Suggestion: add Haber 2016's convergence numbers to `input_streams/README.md` §4.7
as the citation the argument currently lacks, and note in `TODO.md` §26 that Haber
independently invokes Ramanathan 2002 and that the measured quantity is cross-area
convergence, not afferent count — which bounds what a correction may change.

**Goal relevance: medium** — it consolidates two design choices the input
correlation, and hence the BOLD amplitude (`input_streams/README.md` §3), depends on.

**Within-list convergence:** **consensus** on greater FS responsiveness to cortical
and thalamic drive (Haber 2016 and Tepper 2018, independently, in rat and macaque).
The overlap-decay numbers are **one voice** (Haber 2016, own-lab primary data).

---

### 7.16 Rate is not the field's currency, but rest is the right state

**What BGM_22 does:** the model is simulated in resting state, its loss is a
per-region BOLD time-course correlation, and its only physiological validation is the
firing-rate term of `get_loss.get_firing_rate_loss`, which gates the BOLD run.
`input_streams/README.md` §4.3 states what the streams can carry: "**Population
rhythms are representable** — a shared modulation at 20 Hz gives beta-band correlated
input — but individual neurons bursting is not." Nothing in the pipeline measures a
simulated oscillation, synchrony or burst statistic; the loss file records rates,
per-region BOLD correlations, sample counts and the gate flag (`CLAUDE.md`, "Running
things").

**What the literature reports:** both parkinsonian-physiology reviews put pattern and
synchrony ahead of rate, in their conclusions rather than in passing. Wichmann 2019:
"this view has now given way to studies showing that changes in firing patterns
within the basal ganglia and related nuclei are more important, including the
emergence of burst discharges, greater synchrony of firing between neighboring
neurons, oscillatory activity patterns, and the excessive coupling of oscillatory
activities at different frequencies"; and in the conclusion, "Models such as the
'rate' model are now clearly outdated". McGregor & Nelson 2019: "many investigators
have challenged the idea that static changes in rate causally underlie PD motor
symptoms and instead postulate that changes in pattern are central", with beta-band
LFP power (8–35 Hz, typically ~20 Hz) in the STN of PD patients correlating with
bradykinesia and rigidity (Brown et al. 2001; Kühn et al. 2009) and reduced by both
dopamine replacement and DBS (Giannicola et al. 2013; Quinn et al. 2015). And,
favourably, Wichmann on the state the model simulates: "It is also remarkable that
most of our knowledge of the pathophysiology of parkinsonism was acquired through
studies of neuronal activity at rest—when, strictly speaking, neither bradykinesia
nor akinesia are present."

**The primary evidence:** the beta/bradykinesia correlation is the best-supported
item in either review — Brown 2001 and Kühn 2009 are human intraoperative and
post-implant STN LFP recordings, and both reviews present the correlation as
established while presenting *causality* as open (Wichmann: "It remains an open
question which of the observed abnormalities in basal ganglia activity are causal for
parkinsonism"). The rest-state remark is Wichmann's own methodological observation,
not a primary-backed finding, and I mark it as such.

**Why it matters (or does not) here:** the criticism does not land on the objective —
the loss is BOLD correlation, not rate — but it lands squarely on the validation. The
only evidence BGM_22 will have that a fitted vector produces a parkinsonian basal
ganglia is that nine population rates per loop sat inside nine bands, six of which
have no derivation (7.11), against a literature that says rate is the weakest
available marker. Meanwhile the model *can* represent the marker the field trusts:
the streams carry population-level shared modulation, `mc.correlation_dict` and
`mc.cortical_correlation` are the knobs, and `TODO.md` §25 already plans a scan over
them. A simulated STN or GPe beta measurement would cost one spike-train analysis on
recordings the model already makes (`get_loss.py` monitors `["spike"]` on every
non-`TimedInput` population) and would be a genuinely independent check — it is not
in the loss, so it cannot be fitted to, which is what makes it worth having. The
favourable half is real: the project simulates resting state and fits resting-state
BOLD, and Wichmann says the pathophysiology literature is itself overwhelmingly a
resting-state literature. The mismatch is one level down and
`activity_striatum/README.md` caveat 6 already names it — the FS rate anchors are ITI
and pre-cue windows in trained, food-restricted, task-engaged animals, not rest, and
Liang's baselines are likewise recorded during behavioural sessions. So the model's
state matches the field's, and its rate anchors do not quite match the model's state.
→ **experimentally grounded deficiency**, narrowly on the validation rather than the
objective. Proposal: add a reported diagnostic, not a loss term — compute simulated
STN and GPe population spike-count synchrony and spectral power in the 8–35 Hz band
from the existing spike monitors, write them into
`data_BOLD_optimization/loss_<appendix>.json` beside the rates, and check them
against the parkinsonian direction both reviews assert (elevated beta, increased
synchrony, bursting in STN and GPi). Treat a fit that reproduces the BOLD but shows
no parkinsonian pattern signature as a result to report, not to hide.

**Goal relevance: medium-high** — it is the difference between "the fit reproduced
seven BOLD time courses" and "the fit reproduced them with a basal ganglia in a
parkinsonian state".

**Within-list convergence:** **consensus** — Wichmann 2019 and McGregor & Nelson 2019
both make the rate-to-pattern shift their organising theme, from primate/human and
rodent/human evidence respectively. The rest-state observation is **one voice**
(Wichmann 2019) and is that review's own methodological remark rather than a
primary-backed finding.

---

## What we would ask for before publication

Only demands licensed by the verdicts above.

1. **Resolve the GPe abundance inconsistency** (7.1). Either size `gpe_proto`,
   `gpe_arky` and `gpe_cp` by the same abundances the BOLD monitor weights them
   with, or drop the weighting; and write the Courtney et al. 2023 citation into
   `get_loss.py` beside `gpe_proportions`, with its mouse provenance and both
   Courtney's and Wichmann's caveat that primate translation is untested.
2. **State the loop-independence assumption, and preferably test it** (7.3). At
   minimum, record in `model_v07.md` §1 that complete caudate/putamen separation is
   contradicted by macaque anatomy (Haber 2016; Emmi et al. 2020) and by parkinsonian
   physiology (McGregor & Nelson 2019), so every caudate-vs-putamen contrast is an
   upper bound. Better: add the single associative-GPe → motor-STN projection Emmi's
   reviewed anatomy supports, in its own optimizer cluster, and let the fit set it.
3. **Stop reusing the corticostriatal proportion table for STN, GPe and thalamus**
   (7.4), or say in both `cortical_proportions/README.md` and `model_v07.md` §8 that
   it is being reused for three structures it was not derived for. The minimum
   evidence-backed correction is S1 → 0 for STN.
4. **Document the six non-striatal firing-rate bands** (7.11): resolve
   "[Li et al., 2015]" to a full citation with species, preparation and medication
   state; give `gpe_proto`, `gpe_arky`, `gpe_cp` and `thal` a source and disease state
   each, or mark them explicitly as unsourced; and mark the thalamic band as
   unconstrained in parkinsonism on the authority of McGregor & Nelson 2019 Fig. 4 and
   Wichmann 2019. Do this **before** `TODO.md` §10's gate calibration.
5. **State that renormalising the proportions is a covariance assumption** (7.13), and
   report that the omitted fraction is larger for the caudate than for the putamen. If
   it is to be tested, the flat-remainder alternative (columns summing to the retained
   fraction, remainder driven at a constant 5 Hz) costs nothing anatomically and is
   strictly weaker.
6. **Record the anatomical lower bound on the cube** (7.14): Haber 2016's 0.5 mm MSN
   dendritic field, in `TODO.md` §24 alongside the existing `f(d)` argument and in
   `model_v07.md` §7.1 — noting the citation is a textbook chapter, so it bounds the
   target size rather than fixing it.
7. **Add a pattern diagnostic to the loss file** (7.16): simulated STN and GPe
   spike-count synchrony and 8–35 Hz power from the spike monitors the model already
   runs, reported and not fitted, checked against the parkinsonian direction both
   physiology reviews assert.
8. **Record what the audit confirmed**, not only what it questioned — Liang 2008 as a
   recognised primary source for elevated parkinsonian MSN rates, and the conflict
   with Deffains 2016 (7.8); the independent identified-cell-type support for
   iSPN > dSPN (7.9); the anatomical motivation for the large `thal → str_fsi` weight
   (7.10); Emmi et al. 2020 as the methodological citation for rejecting tractography
   as a quantitative source (7.12); and Haber 2016's convergence numbers behind
   `input_streams/README.md` §4.7 (7.15). A validation anchor that has survived an
   external audit is worth saying so in the paper.
