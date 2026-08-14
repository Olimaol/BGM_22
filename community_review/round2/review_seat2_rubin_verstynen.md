# Seat 2 — Rubin / Verstynen (Pittsburgh / CMU)

We write as the Rubin–Verstynen lineage: biologically constrained spiking models of the
full cortico-basal-ganglia-thalamic (CBGT) loop (dSPN/iSPN/FSI striatum, prototypic and
arkypallidal GPe, STN, GPi, thalamus, cortex), and the network-level modeling of the
parkinsonian and DBS-stimulated basal ganglia that this panel treats as one of its two
seats with DBS standing — so `DBS.md` and the three fitted DBS parameters are inside our
mandate, with authority. This is round 2 of the `TODO.md` §34 survey (step 3 rerun
2026-08-14); the panel definition and reading list are in `../README.md`. We did not
consult the round-1 reviews, any other seat's round-2 file, or `community_review/README.md`;
where `model_v07.md` cites round-1 material for a specific fact, we treat that fact as
project documentation. Round 2 imposes two requirements on us: every appeal to our own
practice must state the evidence our practice actually rests on — cited source, species,
preparation — or admit that it is a convention, an estimate, or a tuned value, and a change
proposal is only permitted when grounded in concretely cited experimental findings or in a
methodological argument checkable without trusting our habits. And every point must carry
the explicit contrast structure: what BGM_22 does, what we do, the evidence behind our
choice, why that would or would not be better *for this project's goal* (single-subject
resting-state BOLD fitting and DBS inference), with exactly one verdict and a goal-relevance
tag.

**Read for this review (full texts, from the PDFs in this directory's parent):**

- Corbit VL, Whalen TC, Zitelli KT, Crilly SY, Rubin JE, Gittis AH (2016). Pallidostriatal
  projections promote β oscillations in a dopamine-depleted biophysical network model.
  *J Neurosci* 36(20):5556–5571. DOI 10.1523/JNEUROSCI.0339-16.2016
- Rubin JE (2017). Computational models of basal ganglia dysfunction: the dynamics is in
  the details. *Curr Opin Neurobiol* 46:127–135. DOI 10.1016/j.conb.2017.08.011
- Dunovan K, Vich C, Clapp M, Verstynen T, Rubin J (2019). Reward-driven changes in
  striatal pathway competition shape evidence evaluation in decision-making. *PLOS Comput
  Biol* 15(5):e1006998. DOI 10.1371/journal.pcbi.1006998
- Clapp M, Bahuguna J, Giossi C, Rubin JE, Verstynen T, Vich C (2025). CBGTPy: An
  extensible cortico-basal ganglia-thalamic framework for modeling biological decision
  making. *PLOS ONE* 20(1):e0310367. DOI 10.1371/journal.pone.0310367 (main text; its S1
  Appendix with the full parameter tables is not in this directory and was not read)
- Giossi C, Rubin JE, Gittis A, Verstynen T, Vich C (2024). Rethinking the external globus
  pallidus and information flow in cortico-basal ganglia-thalamic circuits. *Eur J Neurosci*
  60(10):6129–6144. DOI 10.1111/ejn.16348

Additionally consulted, as a local PDF, for one provenance question raised below: Goenner L,
Maith O, Koulouri I, Baladron J, Hamker FH (2021), *Eur J Neurosci* 53(7):2296–2321, DOI
10.1111/ejn.15082 — Tables 4–5 and the limitations section only.

**What was reviewed:** `model_v07.md` (in full), `model_v08.md`, `DBS.md`,
`experimental_data/input_streams/README.md`, `experimental_data/activity_striatum/README.md`,
`experimental_data/cortical_proportions/README.md`, `BOLD_optimization/get_loss.py`,
`BOLD_optimization/parameters.py`,
`CompNeuroPy/.../bgm_22/parameters.csv` (columns `BGM_v07_p01`/`BGM_v08_p01`), the
`Izhikevich2003NoisyBaseNonlin` definition in CompNeuroPy, ANNarchy's
`extensions/bold/BoldMonitor.py` and `AccProjection.py` (for the `normalize_input`
semantics), and `TODO.md` §1, §2, §4, §10, §14–17, §23–26, §28, §30–32 plus the Roadmap
(§34 skipped per the survey's independence rule). One 40-line numerical check of the v07
STN equations was run for this review (point 2.9).

## The standard this seat applies

These are the recurring demands of our own papers, each with its actual basis stated.

1. **Parameters carry their sources, row by row, with tuned values labelled as tuned.**
   Corbit 2016 Table 2 lists every connection with probability, conductance, kinetics and a
   per-row reference column; its footnote states the rule: "taken directly from experimental
   data when possible and estimated based on related empirical data otherwise." The
   conductances were set by matching simulated IPSCs to the paper's own recordings (Methods:
   unitary data scaled by in-vivo contact counts where available, total optically evoked
   conductance divided by model contacts otherwise). We must also admit the other half of
   our practice: Dunovan 2019's Table 3 efficacies were "adjusted to reflect empirical
   knowledge about local and distal connectivity … as well as resting and task-related
   firing patterns" (Methods, "Network architecture") — i.e. tuned — and CBGTPy is
   described as "tuned to match known neuronal firing rates and connection patterns"
   (Clapp 2025, Section 2). So the standard we can honestly apply is not "all weights
   measured"; it is "the tuned/measured status of every value is visible, and tuned values
   are not frozen against measurements that exist."
2. **Parkinsonism is a disorder of patterns, not only rates.** Rubin 2017 (Discussion):
   "Abnormal activity patterns, including changes in firing rates, enhanced bursting,
   changes in oscillatory power, increases in correlation … have been observed across the
   BG under DD conditions." Corbit 2016 opens from the same fact ("amplification of
   synchronous, rhythmic activity", citing human/rodent LFP work — Brown et al. 2001, Levy
   et al. 2002, Sharott et al. 2005, as cited there). This is experimentally grounded via
   those citation trails.
3. **The proxy that turns model activity into the measured signal is a modeling choice
   with teeth.** Rubin 2017, Figure 2 and the surrounding text: "Simulated LFPs signal
   clearly depend on which model outputs are used to compute them" — voltage-derived and
   synaptic-current-derived spectra of the *same* simulations differ qualitatively — and
   GABAergic contributions to extracellular signals "may be small unless cells are
   depolarized away from the reversal potential" (citing Buzsáki et al. 2012). This is a
   methodological hazard our own review documents with a worked example.
4. **Inferences are shown to survive perturbation before they are interpreted.** Corbit
   2016 (Methods): three connectivity realisations × three runs, averaged; random initial
   voltages; "the same random seed was used in each pair of trials compared" across
   conditions. Dunovan 2019 ("Network sampling procedure"): N = 15 "subject" networks with
   every efficacy resampled at σ = 0.05µ, the full simulation and hierarchical model-fitting
   pipeline repeated, to confirm the CBGT→DDM mapping was not an artifact of one
   parameterisation. Clapp 2025 Fig 5 averages over 50 random seeds. These are
   methodological conventions, not experimental findings — but they are checkable
   statistics, not house taste.
5. **Simplify where the simplification is tested; keep minimal mechanisms that are known
   to be load-bearing.** Corbit 2016 dropped conduction delays *after* checking they did
   not change the qualitative dynamics ("data not shown", Methods). Dunovan 2019's neurons
   are integrate-and-fire-or-burst (Smith et al. 2000), i.e. maximally reduced — except
   that STN and GPe keep a T-type-like rebound current (g_T = 0.06 nS, Eqs 3–5), because
   STN burst mechanisms are current-specific (Rubin 2017: "calcium currents are crucial for
   STN burst firing", citing Beurrier et al. 1999).
6. **Fitting spiking networks to data is an open problem — including for us.** Clapp 2025
   (Discussion): "One issue that has been left unresolved in our toolbox is the problem of
   parameter fitting … there is no established solution to simultaneously fitting both
   [neural and behavioral] constraints together in these sorts of networks"; the same
   Discussion notes CBGTPy is "not … as computationally efficient as rate-based models in
   generating macroscale dynamics, including those observed using fMRI or EEG". BGM_22 is
   attempting, at fMRI scale, something our own framework paper names unresolved and
   expensive. We therefore judge how the known hazards are handled, not the ambition —
   and we note that Clapp 2025 cites the predecessor of this project (Maith et al. 2021,
   its reference [41]) as the exemplar of the cellular-constraint end of that trade-off.

## Points

### 2.1 The parameter-difference readout needs recovery and held-out tests, not only the planned stability rerun

**What BGM_22 does:** The scientific claim is a difference readout: fit DBS-off, refit
DBS-on, "read off which parameters had to change" (`CLAUDE.md`). The machinery is CMA-ES
on 19 (v07) base parameters, a staged on-fit (base frozen from the off fit, one
putamen-only scaling per cluster + 3 DBS parameters; `get_loss.split_param_list`), seed
fixed at 42 during fitting with the winning vector re-run across ~10 seeds (`TODO.md`
§32), and an inference-design entry (`TODO.md` §2) that already names the degeneracies
(`axon_spikes_per_pulse` vs the `stn__gpe`/`stn__snr` scalings;
`passing_fibres_strength` vs the `snr__thal` scaling) and costs three protocols
(single-mechanism scans, multi-start, L1-regularised minimal-change refit). The caudate
loop is a designed free control (§2 update of 2026-08-11).

**What we do:** Before interpreting a mapping we re-derive it under perturbation: Dunovan
2019 resampled every connection efficacy (N = 15 networks, σ = 0.05µ) and re-ran the entire
fitting pipeline hierarchically, confirming the same best-fitting model; model selection
itself was a forward stepwise comparison over which DDM parameters were allowed to vary
(Dunovan 2019, "Model comparison"), not inspection of one best fit. Corbit 2016 averaged
over three connectivity realisations × three runs and seed-matched compared conditions.

**Evidence behind our choice:** Convention, not experiment — but of the checkable kind:
an inference that does not survive 5 % jitter in nuisance parameters, or that a formal
model comparison does not prefer, is about the optimizer, not the brain (standard 4).

**Why that would (or would not) be better here:** §2 is a genuinely good start — the
degeneracy pairs are exactly the right worry, and the staged layout plus caudate control
is an identification design our own work has no analogue of. But two tests that our
practice treats as mandatory are absent from §2's three options and from §32: (i)
**parameter recovery on synthetic data** — simulate BOLD from the model at known DBS
parameters, refit, and report which of the three DBS parameters (and which cluster
scalings) are recoverable at all at this noise level; with 7 region time-courses of ~309
TRs against 13 free on-fit parameters, recoverability is an open empirical question, and
it is the cheapest possible test of the central claim because it needs no new data; (ii)
**held-out validation** — the loss is a time-course correlation, so fitting on one
segment of the 310 TRs and scoring on the rest is nearly free and directly measures
overfitting of the per-region correlations; relatedly, the per-region correlation of a
309-sample series has a standard error of roughly 1/√306 ≈ 0.06, and off-vs-on parameter
deltas should be interpreted against the spread this implies, which no current artifact
records. The ~10-seed stability rerun in §32 varies simulation randomness of the *winner*;
it does not test whether a *refit* finds the same answer — §2's multi-start option covers
that and should not be the option that gets dropped. **Verdict: methodological
deficiency.** Change proposal: add synthetic-data recovery and a held-out split to §2's
protocol, and state in §32 which randomness sources the stability reruns vary.

**Goal relevance:** high — this is the project's central claim.

### 2.2 The BOLD input proxy is this project's LFP-proxy problem, and it is currently a single untested choice

**What BGM_22 does:** All seven `BoldMonitor`s map `I_CBF` to a *net* synaptic current:
`I_v` for the striatal ROIs, `I` for the rest (`get_loss.py`, BOLD block;
`model_v07.md` §3.6). Consequences documented by the project itself: excitation enters
positively and GABA with its driving-force sign, so inhibition *reduces* the BOLD input;
the three GPe populations are monitored on raw `I` although their membrane sees
`f(I, nonlin)`; `I_base` sits outside `I`, so the two fitted baseline currents (snr,
gpe_proto) and the DBS somatic term never reach the monitor. The monitor uses ANNarchy's
default `balloon_RN` model (`BoldMonitor.__init__` default), with `normalize_input=2000`
— a baseline estimated over the first 2000 ms after the monitors start
(`AccProjection` baseline machinery), i.e. inside the first recorded TR, after a one-TR
ramp-up.

**What we do:** For LFPs we showed that the proxy choice changes the answer: Rubin 2017
Fig 2 computes spectra of the same pallidostriatal simulations from voltage-derived and
synaptic-current-derived proxies and gets qualitatively different pictures (γ present in
one and not the other); Corbit 2016 states its pseudo-LFP choice and its motivation
explicitly (low-pass-filtered population voltage, Methods) and Rubin 2017 defends it
against the alternatives while flagging that GABAergic contributions to extracellular
signals may be small near the reversal potential (Buzsáki et al. 2012, as cited there).

**Evidence behind our choice:** The proxy-dependence demonstration is our own published
simulation result (Rubin 2017 Fig 2, built on Corbit 2016's model); the reversal-potential
caveat is experimental literature as cited in Rubin 2017. That LFP lesson transfers to
BOLD as a methodological argument, not as a neurovascular measurement — we do not claim
to know the correct CBF driver.

**Why that would (or would not) be better here:** The BOLD time-course correlation is the
model's only data-facing output, and every one of the choices above (net current vs.
summed magnitudes of excitation and inhibition, `I` vs `f(I, nonlin)` for GPe, excluding
`I_base`, the sub-TR baseline window, the balloon transient in the first TRs) changes
that output before a single parameter is fitted. None of these is wrong on its face —
the `I_base` exclusion is even argued (`model_v07.md` §3.6) — but none has a recorded
sensitivity check, and our own Fig 2 is a documented case of exactly this kind of choice
flipping a conclusion. A fit whose per-region correlations move materially under a
defensible alternative mapping cannot support a parameter-difference readout. **Verdict:
methodological deficiency.** Change proposal: before the production fits, evaluate one
fitted (or default) parameter set under 2–3 defensible `I_CBF` mappings (at minimum: net
current as now; |excitatory| + |inhibitory| current; GPe on `f(I, nonlin)`) and record
the per-region correlation shifts; a baseline window of at least a few TRs should be
tested at the same time.

**Goal relevance:** high — it conditions everything the loss sees.

### 2.3 The DBS-adjacent ROIs pool two non-interacting loop copies at an uncited 50/50

**What BGM_22 does:** GPi, GPe and STN BOLD monitors pool the caudate-loop and
putamen-loop copies of each nucleus (`get_loss.py`, `bold_region_compartments`). The two
loops share no projection (`CLAUDE.md`; `model_v07.md` §1), only the putamen loop carries
DBS terms, and each copy is 100 neurons, so `BoldMonitor`'s size-share fallback weights
the loops 50/50 in GPi and STN; the GPe monitor's cell-type factors are identical per
loop, so its loop split is also 50/50. The striatal and GPe *cell-type* pooling, by
contrast, is deliberately anatomical (del Rey proportions by construction; Courtney 2023
abundances per `model_v07.md` §3.6).

**What we do:** Our channel architectures let the shared nuclei couple the channels:
in Dunovan 2019 the STN→GPi projections are "channel-generic and caused diffuse
excitation in both L- and R-encoding populations", GPe/STN divergence and sparseness are
set from anatomical work (Fox & Rafols 1976; Smith et al. 1998; Steiner et al. 2019 — as
cited in Dunovan 2019's Methods), and FSI and cortical interneuron pools are shared
across channels outright.

**Evidence behind our choice:** The diffuse-STN choice rests on the anatomical divergence
literature as cited in Dunovan 2019; we have not measured it ourselves, and the
channel construct itself is a modeling convention. So we do not propose that BGM_22 wire
its loops together on our authority.

**Why that would (or would not) be better here:** The checkable problem is arithmetic,
not anatomical: the STN, GPi and GPe ROIs — the three ROIs closest to the electrode —
are each half composed of a compartment that is structurally barred from any direct DBS
effect, at a 50/50 ratio that comes from `stn.size = 100` in `parameters.csv` rather
than from the associative/motor territory proportions of the human nuclei. Whatever DBS
does to the putamen copy arrives in these ROI signals diluted by a factor the model
fixed by accident, and the caudate copy's contribution to the very ROIs used to infer
DBS parameters is driven only by its cortical input. This interacts directly with the
inference: an attenuated modeled DBS response in STN/GPi/GPe BOLD will be compensated
somewhere — plausibly in the three DBS parameters themselves. **Verdict: methodological
deficiency.** Change proposal: state and justify the loop shares of the pooled ROIs
(any defensible anatomical number beats a population-size accident; the same
`scale_factor` mechanism already used for GPe cell types can carry it), and report, for
fitted models, each loop's share of every pooled ROI's variance and of its on–off
change.

**Goal relevance:** high — these are the DBS ROIs.

### 2.4 The BG weight skeleton is Goenner 2021's tuned table, uncited, with subtype ratios frozen against later measurements

**What BGM_22 does:** We checked all 28 v07 projection weights in `parameters.csv`
(column `BGM_v07_p01`) against Goenner et al. 2021 Tables 4–5: every one matches, either
verbatim (all non-striatal targets: e.g. `str_d2__gpe_proto` 0.04, `str_d2__gpe_arky`
0.08, `stn__gpe_*` 0.001, `gpe_*__gpe_*` 0.008/0.025) or multiplied by the postsynaptic
membrane capacitance for the Humphries striatal targets (×50 for SPNs, ×80 for FSIs:
`gpe_arky__str_d1` 0.065→3.25, `__str_d2` 0.12→6, `__str_fsi` 0.08→6.4,
`gpe_proto__str_fsi` 0.02→1.6, `thal__str_*` 0.14/0.12/0.12→7/6/9.6); the seven v08
intra-striatal weights match under the same rule. Nothing in the repository records this
origin — the CSV has no reference column, and `get_loss.py`'s cluster comment calls them
"the literature values already in BGM.params". The optimizer then scales whole clusters
(`proj_clusters`), so within-cluster *ratios* — iSPN→arky twice iSPN→proto, STN input
equal across all three GPe types — are frozen into every fit by design
("preserves their relative balance", same comment).

**What we do:** Where subtype-resolved measurements exist we encode their direction. Our
own review consolidates them (Giossi 2024, §3.2): "iSPN projections to arkypallidal
neurons have been estimated to be 85% weaker than those targeting prototypical neurons
and also less numerous …, while STN inputs have been measured to be 74% weaker to
arkypallidal than prototypical cells" (both Aristieta et al. 2021, mouse, in vivo
opto/ephys as reported there).

**Evidence behind our choice:** Aristieta et al. 2021 as reported in Giossi 2024 §3.2
(mouse). And, per requirement (1), the source of BGM_22's numbers says of itself: "the
weight strengths between model nuclei, the cortical inputs to the model, and the baseline
inputs are rather abstract and were determined mainly by functional constraints"
(Goenner et al. 2021, Discussion) — tuned for a stopping task in which strong striatal
engagement of arkypallidal cells is the mechanism under study. Our own Giossi 2024 cites
Goenner 2021 approvingly for that mechanism, so we do not call the inheritance
illegitimate — but its tuned status and task provenance are facts the repository does
not state.

**Why that would (or would not) be better here:** BGM_22's cluster design deliberately
preserves relative balance to condition the search — a sound idea exactly when the
encoded balance is trustworthy. Here the frozen iSPN→GPe and STN→GPe subtype ratios point
the *opposite* way from the only subtype-resolved measurements we know of, and no fitted
scaling can repair them. The species caveat is real (mouse measurements, human model) —
but the current encoding has no species at all, only an uncited inheritance from a
functionally tuned table. For a resting-state PD fit whose GPe BOLD is cell-type-weighted,
the proto/arky balance of striatal and STN drive plausibly matters. **Verdict:
experimentally grounded deficiency.** Change proposal: record the Goenner-2021-with-×C
provenance in the repository; then either re-derive the within-cluster subtype ratios
from Aristieta 2021's reported asymmetries (iSPN→arky ≪ iSPN→proto; STN→arky < STN→proto)
with the species caveat documented, or split the affected clusters so the fit can move
the ratios, or record a reasoned rejection.

**Goal relevance:** medium — fixed structure under everything the fit does; it shapes
what the fitted GPe-related parameters mean.

### 2.5 Pallidostriatal targeting strengths are compressed relative to our own recordings

**What BGM_22 does:** With equal in-degrees (`connect_fixed_number_pre(number=10)` for
all 28 projections), the aggregate GPe→FSI weight (1.6 + 6.4 + 0.8 = 8.8) is only ~1.4×
the aggregate GPe→iSPN weight (6 + 0.5 = 6.5) and ~2.3× GPe→dSPN (3.25 + 0.5 = 3.75).
Within the split, arky→FSI (6.4) is four times proto→FSI (1.6).

**What we do:** We measured this pathway: optogenetic activation of GPe terminals in
mouse slices gave maximal IPSCs of 566 ± 560 pA in *every* FSI sampled, against 28 ± 44 pA
(control; 38 % responding) and 108 ± 73 pA (dopamine-depleted; 73 % responding) in MSNs,
FSI ≫ MSN in both conditions (p < 0.00001) (Corbit 2016, Results and Fig 1). Our model
encoded this as g_syn 0.12 (GPe→FSI) vs 0.003–0.01 (GPe→MSN) at equal connection
probability — a 12–40× ratio (Corbit 2016, Table 2) — and found the GPe→FSI branch, not
GPe→MSN, to be the dynamically load-bearing one (adding GPe→MSN to the DD network changed
no β measure significantly; Results, Fig 4). On subtype targeting, our review states the
prototypic-to-FSI projection signals "with more strength than that with which
arkypallidal neurons signal to FSIs" (Giossi 2024 §3.2, citing Corbit 2016) — the
opposite order from BGM_22's 1.6 vs 6.4.

**Evidence behind our choice:** Our own recordings (mouse, slice, ChR2 — species and
preparation limits acknowledged), and the model-side ratio derived from them by IPSC
matching.

**Why that would (or would not) be better here:** Absolute weights are not commensurable
across neuron models, but ratios at equal in-degree are, and BGM_22's FSI:SPN
pallidostriatal ratio (~1.4–2.3) sits an order of magnitude below what our recordings
support (~8–20× by IPSC amplitude alone, before counting the response-probability
difference). Because these weights are Goenner-tuned (point 2.4) rather than chosen
against pallidostriatal data, this is most likely inheritance, not judgment. In a model
whose 29 FS neurons matter only through strong weights (`model_v07.md` §7.2), and whose
off-state is exactly the DD condition where we showed the GPe→FSI branch shapes striatal
dynamics, the compression is a real modeling risk — attenuated here, we concede, by the
fact that ~98 % of FS GABAergic input is open-loop stream rather than simulated GPe
(`input_streams/README.md` §4.1), which caps how much circuit dynamics this pathway can
carry at all. **Verdict: experimentally grounded deficiency.** Change proposal: document
the pathway's provenance; re-set or bound the GPe→FSI : GPe→SPN ratio (and the
proto-vs-arky order onto FSIs) against Corbit 2016's measurements with the species caveat
stated, or record a reasoned rejection.

**Goal relevance:** medium — it bears on the off-state striatal dynamics and on what the
fitted `gpe_striatum` scaling means, less on the BOLD fit directly.

### 2.6 The GPe and thalamic rate bands are uncited, and two of them sit oddly against the subtype physiology our review consolidates

**What BGM_22 does:** `get_loss.get_firing_rate_loss` scores all populations against
bands: gpe_proto (75, 85), gpe_arky (15, 20), gpe_cp (75, 85), thal (15, 30). The code
comment sources stn and snr to "[Li et al., 2015]" and the striatal bands to the
documented Liang/FS derivations; the GPe and thalamic bands carry no citation. The bands
also gate the BOLD run (`firing_rate_gate`), are condition-independent, and the probe
runs with DBS active — a problem the project knows (`DBS.md` limitation 5; `TODO.md` §10,
whose plan is to run mini-runs at `--gate-threshold 1.0` and calibrate afterwards).

**What we do:** Our review compiles the subtype rates: prototypic neurons fire 10–100
spk/s, ~55 spk/s average, arkypallidal 1–30 spk/s, ~10 spk/s average — rodent,
dopamine-intact, in vivo (Giossi 2024 §3.1, citing Abdi 2015, Dodson 2015, Mallet
2012/2016 among others); it reports Npas1+ neurons as *hypoactive* under dopamine
depletion (Giossi 2024 §5, citing Pamukcu et al. 2020, mouse); and it is explicit that
the rodent-to-primate translation of the subtype scheme is unproven ("the molecular
evidence to support this analogy remains lacking", §3.1, discussing Katabi 2023 and
Nambu & Chiken 2024).

**Evidence behind our choice:** The rodent numbers above, with their citation trail as
reported in Giossi 2024; the primate caveat cuts against us too, and we state it.

**Why that would (or would not) be better here:** Three observations, in decreasing
strength. First, the bands are constraint values that shape every fit and (via the gate)
decide which individuals are even evaluated, yet unlike every striatal number in this
project they have no recorded derivation — by BGM_22's own documentation standard, that
is a gap. Second, the arky band (15–20 Hz) *excludes* the rodent in-vivo average (~10 Hz)
entirely; a human/primate or PD-state justification may exist, but none is written.
Third, gpe_cp — an Npas1-class population by the model's own BOLD-weight citation
(NPAS1+NKX2.1+, `model_v07.md` §3.6) — is given the prototypic band (75–85 Hz) while the
one condition-relevant datum our review reports for Npas1+ cells is hypoactivity under
depletion. We do not claim to know the right human PD numbers; we claim these three
things need sources or reasons. The §10 plan is adequate for the *threshold*; it does not
touch band provenance. **Verdict: experimentally grounded deficiency.** Change proposal:
document the provenance of the GPe and thalamic bands next to the striatal derivations;
reconcile the arky and cp bands with the subtype literature (rodent values, PD-state
data where they exist) or record why the model's abstractions justify different targets.

**Goal relevance:** medium — the bands steer the search and define "plausible" for the
gate.

### 2.7 Nothing reports the fitted model's dynamical state beyond rates

**What BGM_22 does:** The loss and its artifacts record rates, per-region BOLD
correlations, sample counts and gate state (`loss_<appendix>.json`); nothing measures or
records oscillatory structure, synchrony or bursting of a fitted model in either
condition. The input machinery could carry a β-band shared modulation but currently
injects none (ρ = 0; `model_v07.md` §7.4), and the project's claim-boundary document
correctly rules out *claims about* emergent striatal dynamics
(`input_streams/README.md` §5).

**What we do:** Our parkinsonian modeling is organised around the pattern phenotype:
β-band synchronisation "a hallmark of parkinsonian circuit dysfunction" (Corbit 2016,
Abstract/Introduction, citing Brown 2001, Levy 2002, Sharott 2005), and model states are
reported with spectra and synchrony measures in both conditions (Corbit 2016 Fig 3;
Rubin 2017 throughout).

**Evidence behind our choice:** The DD pattern phenotype is experimental, via the
citation trail above (human LFP and rodent recordings as cited in Corbit 2016 and Rubin
2017).

**Why that would (or would not) be better here:** The subject's β is not in the fitted
data (only BOLD is), so we do not propose a β loss term. But the inference's credibility
depends on what dynamical regime the off-fit lands in: a fitted "parkinsonian" model
whose STN-GPe circuit sits in a regime bearing no resemblance to the documented DD
phenotype weakens every mechanistic sentence written about the DBS deltas, and no current
artifact would reveal it. A descriptive diagnostic is nearly free: the rate probe already
records all spikes (`Spikes10s`); adding population spectra/synchrony summaries of the
probe window to the loss JSON costs no extra simulation. **Verdict: experimentally
grounded deficiency.** Change proposal: report (not fit) STN/GPe/striatal population
spectra and a synchrony measure for fitted off and on models, and discuss them against
the DD-pattern literature in any write-up.

**Goal relevance:** medium — it does not change the fit, but it bounds what the fit can
be said to mean.

### 2.8 The DBS mechanism set: what is absent is documented, and mostly tolerable at TR resolution

**What BGM_22 does:** DBS is a somatic shunt toward −90 mV on 40 % of putamen-STN, axon
spikes at a fitted per-pulse probability propagating orthodromically (zero delay) and
antidromically, afferent activation that in practice reaches `gpe_proto→stn` only, and
one passing fibre (`snr__thal`, standing for GPi→thal after Miocinovic et al. 2006 — a
citation we did not verify and treat as the project's). Known gaps are recorded:
hyperdirect/cortical fibre activation is structurally impossible (`TODO.md` §15: "a
fitted 'afferent' effect here is a pallidal one"), axon spikes bypass the delay line
(§14), the constants are single-subject and unfitted (§16, with a planned sensitivity
check).

**What we do:** Our lineage's account of high-frequency STN DBS includes an
activity-dependent mechanism BGM_22 lacks: axonal and synaptic failure under
high-frequency driving, which "can … suppress transfer of firing rate oscillations,
synchrony, and rate-coded information" and lets DBS work "by decoupling BG output from
pathological upstream firing patterns" — a model constrained by parkinsonian non-human
primate recordings (Rosenbaum et al. 2014, as summarized and annotated in Rubin 2017,
including its NHP evidence base as cited there).

**Evidence behind our choice:** Second-hand through Rubin 2017's summary and annotation
of Rosenbaum et al. 2014; we did not re-read the original here, and we say so.

**Why that would (or would not) be better here:** The fitted `axon_spikes_per_pulse` and
`p_axon_spike_trans` are *constant* efficacies: they can absorb the mean transmission
level that depression would produce, but not its pattern-selectivity (a depressing
synapse transmits tonic drive and oscillatory drive differently — that is the point of
the mechanism). Read out through a 2.31 s TR, the mean effect dominates and the constant
approximation is defensible for *fitting*; what it cannot support is interpreting the
fitted axonal parameters as evidence about transmission mechanisms, and the same holds
for the hyperdirect gap (§15) — a DBS-on delta will be attributed to the pathways the
model contains. The project's documentation already carries most of this; §15's framing
is exactly right. **Verdict: difference, not deficiency.** We suggest only that the
misattribution risk stated in §15 and the constant-efficacy simplification be restated
together, prominently, wherever fitted DBS parameters are interpreted — they define the
menu the inference chooses from.

**Goal relevance:** high — not because a change is needed, but because the write-up's
causal language depends on it.

### 2.9 What the somatic DBS term actually does, measured: suppression and entrainment, not rebound

**What BGM_22 does:** The somatic term adds `pulse(t)·dbs_on·dbs_depolarization·neg(−90−v)`
to `dv/dt` — a one-timestep (0.1 ms) shunt toward −90 mV every 8 ms (`DBS.md`). The
project documents the sign inversion (it hyperpolarizes despite the name) but not its
dynamical consequences. We ran a 40-line explicit-Euler reimplementation of the v07 STN
equations (§6.1 parameters, dt = 0.1 ms, tonic drive giving ~16 Hz; noise and
conductances omitted) for this review. Result: the parameter set *does* possess a
rebound analogue — a 500 ms hyperpolarizing step is followed by a ~60 Hz transient
against an 18 Hz baseline (u de-inactivation, τ = 1/a = 200 ms) — but the actual DBS
train never engages it: each 0.1 ms shunt is far too brief to move u, so across
`dbs_depolarization` = 1/5/10 the rate only falls (15.5 → 14.8 → 11.8 → 9.0 Hz) and at
the upper bound the surviving spikes phase-lock to the last millisecond of the
inter-pulse interval.

**What we do:** Even our most reduced networks keep a minimal burst current in STN and
GPe (integrate-and-fire-or-burst, g_T = 0.06 nS; Dunovan 2019 Eqs 3–5), because STN
burst/rebound behavior is current-specific: "calcium currents are crucial for STN burst
firing" (Rubin 2017, citing Beurrier et al. 1999), and the STN-GPe pacemaker mechanism
runs on "STN rebound following offset of GPe inhibition" (Rubin 2017, describing the
Plenz & Kitai 1999 organotypic-culture experiments).

**Evidence behind our choice:** Beurrier et al. 1999 and Plenz & Kitai 1999 via Rubin
2017's citation trail; the IFB inclusion itself is our modeling convention built on
them.

**Why that would (or would not) be better here:** We are *not* claiming the Izhikevich
STN is disqualified — our check shows it expresses inhibition-driven rebound within the
network (GPe inhibition lasts long enough to engage u), which is the loop mechanism that
matters. The finding is narrower: the fitted `dbs_depolarization` is, measurably, a
suppress-and-entrain knob, and at high values it imposes 125 Hz phase-locking on the
stimulated 40 % of STN — a strong pattern effect that nothing in the repository
documents and that interacts with point 2.7 (it would show in a spectral diagnostic).
Whether a real somatic response to brief extracellular pulses engages T-type dynamics
between pulses is a physiological question our sources do not answer at this pulse
width, so we make no correctness claim — only that the parameter's actual effect should
be characterized in the project's own terms before its fitted value is interpreted.
**Verdict: difference, not deficiency.** We suggest documenting the measured behavior
(suppression + entrainment; no train-driven rebound) alongside `DBS.md`'s existing sign
note, and checking a fitted model for STN entrainment at its fitted amplitude.

**Goal relevance:** medium — it determines what a fitted `dbs_depolarization` value can
be said to mean.

### 2.10 Where BGM_22 exceeds our own practice

**What BGM_22 does:** Four things we want on the record as strengths. (i) A three-type
GPe (proto/arky/cp) with the pallidostriatal loop present *constitutively*, at rest, in
both loops. (ii) Subtype directions that match the data our review compiles: arky→iSPN
vs arky→dSPN at 6/3.25 ≈ 1.85:1 against the measured 2:1 (Glajch et al. 2016, as
reported in Giossi 2024 §3.2); proto→striatum onto FSIs only, matching the reported
targeting (Bevan 1998, Gittis 2014, Glajch 2016, Saunders 2016, as cited there);
proto→arky (0.025) ≫ arky→proto (0.008), matching the reported asymmetry of intra-GPe
inhibition (Aristieta 2021; Gast 2021, Ketzef & Silberberg 2021, as cited there). (iii)
Striatal rate anchors from parkinsonian *primates* with the identifying assumption
audited and sensitivity-bounded (`activity_striatum/README.md`: Liang et al. 2008
med-off, response-direction assumption stated verbatim from the source; FS as
normal-primate level × rodent depletion factor with the range given). (iv) A written
claim-boundary contract (`input_streams/README.md` §5) that pre-commits what the model
may and may not be used to claim, plus the caudate free control (`TODO.md` §2) as a
falsifiable internal check.

**What we do:** Our own framework enables the pallidostriatal pathways "only … for the
stop signal task" (Clapp 2025, Fig 1 caption) and Dunovan 2019 excluded arkypallidal
projections outright ("not currently well understood how this pathway contributes to
basic choice behavior", Methods); our DD rate targets are rodent (MSN 2.0→5.0 Hz, GPe
24.5→18.9 Hz, FSI 21.4→23.7 Hz; Corbit 2016, Results, citing Fino 2007, Azdad 2009,
Kita & Kita 2011, Hernández 2013); and our limitation sections, while real, do not
pre-register claim boundaries with BGM_22's precision.

**Evidence behind our choice / theirs:** For (i), our own Corbit 2016 shows the
pallidostriatal circuit shapes DD striatal dynamics — the very reason a resting-state DD
model should carry it always-on, as BGM_22 does and our task framework does not. For
(iii), a human-patient fit anchored on MPTP-primate rates is better matched in species
and chronicity than our rodent numbers would be — the 20-fold rodent/primate MSN
discrepancy is documented in BGM_22's own README.

**Why that would (or would not) be better here:** These are the right calls for this
project's goal, and two of them (constitutive pallidostriatal loop; written claim
boundaries) are practices we should adopt rather than the reverse. **Verdict:
difference, not deficiency** — in BGM_22's favour.

**Goal relevance:** medium — strengths that make the deficiencies above worth fixing.

### 2.11 Delays and the open-loop striatum: simplifications our practice endorses, with the plans in the right order

**What BGM_22 does:** Fixed subcortical delays (the rat set of Kumaravelu et al. 2016,
per the provenance recorded in `model_v07.md` §5), zero-delay cortical drive
(`CurrentInjection` injects into the current step, §7.7), and a striatum that is ~2 %
recurrent circuit and 98 % open-loop calibrated stream, with the missing decorrelation
mechanism named and a scan planned before any real fit (`input_streams/README.md` §3–5;
`TODO.md` §25, carried in the Roadmap as the hard blocker on §32).

**What we do:** Corbit 2016 included conduction delays, checked that they did not change
the qualitative dynamics, and dropped them (Methods, "data not shown"); Dunovan 2019 uses
a uniform 0.2 ms synaptic delay. So our own standard is "test, then simplify", not
"reproduce measured delays" — and we hold BGM_22 to nothing stronger.

**Evidence behind our choice:** The delay omission is our tested convention (Corbit
2016); the decorrelation literature behind §25 is the project's own citation set, which
matches what we know of it.

**Why that would (or would not) be better here:** At a 2.31 s TR, millisecond delay
choices are prima facie irrelevant to the loss; their oscillation-frequency effects fold
into point 2.7's diagnostic rather than needing separate treatment. The open-loop
striatum is the bigger simplification, and the project's handling — a written statement
of what it forecloses, a ρ-scan gating the fits, and the fixed-point argument for input
correlation — is more explicit than anything we have published on an equivalent
simplification. The ordering (scan before fits) is correct. **Verdict: difference, not
deficiency.**

**Goal relevance:** low for delays, high for the open-loop striatum — but the latter is
already governed by an adequate plan, which we endorse rather than duplicate.

### 2.12 Parameter provenance should be an artifact, not an archaeology project

**What BGM_22 does:** `parameters.csv` holds 19 historical model-version columns of bare
numbers with no reference column; the living documents recover provenance piecemeal
(delays via the previous review round, as recorded in `model_v07.md` §5; GPe BOLD
abundances via Courtney 2023 in §3.6) — and the weight skeleton's origin (point 2.4)
was recoverable only by diffing against a 2021 paper's tables, which took this reviewer
an afternoon and a factor-of-C detour.

**What we do:** Corbit 2016 Table 2 is the house form: every connection row carries its
probability, conductance, kinetics *and* its references, with the tuned/measured status
declared in the footnote; Table 1 does the same for cellular parameters. Dunovan 2019
and Clapp 2025 put the equivalent tables in the paper/supplement with their citation
sentences ("adjusted to reflect empirical knowledge …"), which — as requirement (1)
obliges us to admit — is weaker than Corbit's per-row form, but still names what is
tuned.

**Evidence behind our choice:** Convention — but one whose value this very review
demonstrates: points 2.4–2.6 exist because the provenance was absent, and each would
have been a one-line check against a reference column.

**Why that would (or would not) be better here:** For a project whose *product is a
parameter-difference claim*, every fixed number is part of the inference's support, and
BGM_22's own documentation conventions (`CLAUDE.md`: cite by symbol, README audit
trails) already exceed ours everywhere except in the one file the model is actually
built from. **Verdict: methodological deficiency.** Change proposal: a provenance table
for the `BGM_v07_p01` column (and the loss bands) — value, source or "tuned, from
Goenner 2021 ×C", species, and the date checked — in the repository, in Corbit-Table-2
form.

**Goal relevance:** medium — it converts three of this review's findings into
one-line checks for every future reviewer.

### 2.13 v08 as a smoke test

**What BGM_22 does:** v08 collapses the drive to one mixed `TimedArray` per loop with a
random-kick `exp_input` mechanism, keeps the same BG skeleton, and is documented as a
pipeline test only (`model_v08.md`; `CLAUDE.md`), with its known-broken bounds flagged
(`TODO.md` §1).

**What we do:** We validate frameworks on fast reduced configurations before science runs
(CBGTPy's example tasks and 50-seed averages serve that role; Clapp 2025 §3), and we keep
framework papers separate from claims papers.

**Evidence behind our choice:** Convention.

**Why that would (or would not) be better here:** Reviewed strictly as the smoke test it
claims to be, v08 is fit for purpose, and the documentation firewall around it (explicit
"do not start a real fit" warnings) is sound. **Verdict: difference, not deficiency.**

**Goal relevance:** low.

## What we would ask for before publication

1. A parameter-recovery study on synthetic BOLD (known DBS parameters in, refit, report
   recovery) and a held-out-TRs validation of the fitted correlations, added to the
   `TODO.md` §2 protocol; report per-region correlation uncertainties (~0.06 SE) next to
   every off-vs-on delta (2.1).
2. A recorded sensitivity check of the `I_CBF` mapping (net current vs. summed magnitudes;
   GPe raw `I` vs `f(I, nonlin)`; baseline window) on at least one parameter set (2.2).
3. Documented, anatomically justified loop shares for the pooled STN/GPi/GPe ROIs, and a
   report of each loop's contribution to those ROIs' variance and on–off change in fitted
   models (2.3).
4. Provenance recorded for the BG weight skeleton (Goenner 2021 ×C), with the
   within-cluster iSPN→GPe, STN→GPe and GPe→FSI-vs-SPN subtype ratios either reconciled
   with Aristieta 2021 / Corbit 2016 (species caveats stated) or their retention argued
   (2.4, 2.5).
5. Sources or reasons for the GPe and thalamic rate bands, including the arky and cp
   bands' relation to the subtype physiology (2.6).
6. Spectral/synchrony diagnostics of fitted off and on models in the loss artifacts and
   the write-up (2.7).
7. A provenance table for `parameters.csv` column `BGM_v07_p01` and the loss bands, in
   per-row Corbit-Table-2 form (2.12).
8. From the difference verdicts, documentation only: restate §15's misattribution caveat
   and the constant-efficacy simplification wherever fitted DBS parameters are
   interpreted, and record the measured suppress-and-entrain character of the somatic
   term (2.8, 2.9).
