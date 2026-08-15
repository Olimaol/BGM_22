# Synthesis of the community-conventions review — round 2

Merges the seven round-2 seat reviews in this directory into one ranked list of
findings, each with an explicit contrast and, where licensed, a change
proposal. Defined by `TODO.md` §34 step 3, rerun 2026-08-14 under two
additional requirements; the panel, the selection rationale and the reading
list are in `../README.md`. The round-1 documents live in `../round1/` and were
not consulted by any seat; this synthesis supersedes `../round1/synthesis.md`.
**Finding numbers are local to this document** — round 1's F-numbers are not
referenced here and are superseded.

**What this document is not.** It contains no verdicts on the project's behalf.
`TODO.md` §34 is explicit that implementing accepted changes is not part of the
survey — the triage that follows decides which findings are accepted, and each
accepted one spawns its own numbered `TODO.md` entry; this synthesis is not
itself referenceable.

## The two round-2 requirements, and how they shaped what is below

Round 2 was rerun because round 1's seats argued partly from their own
modeling habits. Two rules were imposed:

1. **Evidence basis.** Every appeal a seat makes to its own practice must state
   the experimental evidence that practice rests on — source, species,
   preparation, what was measured — or admit that it is a convention, an
   estimate, or a tuned value. A seat's convention may be reported as a
   *difference between approaches* but never as a deficiency of BGM_22, and no
   change proposal may rest on it.
2. **Explicit contrast.** Every point states what BGM_22 does, what the seat's
   lineage does, the evidence behind the lineage's choice, and why that would
   or would not be better *for this project's goal* — ending in exactly one
   verdict: **experimentally grounded deficiency**, **methodological
   deficiency** (an argument that stands on its own: identifiability,
   statistics, numerics, internal consistency), or **difference, not
   deficiency** (no change proposal; at most a request to document).

The seven reviews carry 98 points — seat 1: 13, seat 2: 13, seat 3: 12,
seat 4: 14, seat 5: 16, seat 6: 14, seat 7: 16 — with 22 experimentally
grounded deficiencies, 33 methodological deficiencies and 43 differences (of
which 19 are recorded explicitly in BGM_22's favour). Every finding below
carries the strongest evidence class among the seat points it merges, and
change proposals appear only where that class licenses them. Findings whose
constituent points are all differences carry documentation requests only.

## How the ranking works

Per §34: convergence across seats is the signal. A point raised by several
independent lineages is a community convention; a point raised by one is one
lab's voice, which may still be right but carries less weight. Tiers: **Tier 1**
five or more seats, **Tier 2** three or four, **Tier 3** two seats or one seat
with strong evidence, **Tier 4** single-seat findings worth recording.

Per the §34 amendment for seat 7 (the experimentalist), a point raised only by
that seat is weighted by convergence *within its own reading list*: asserted by
multiple independent reviews → literature consensus, comparable to multi-seat
convergence; found in a single review → one voice. Every seat-7-derived finding
below states which case applies, and seat 7's structural claims carry species
tags.

## The one limitation this panel cannot address

No BOLD or whole-brain lineage has a seat, by design (rationale in
`../README.md`): **the project's BOLD pipeline has no peer reviewer on this
panel.** Nobody audited the balloon model as such, the deconvolution that
produces the cortical drive, or the fit-to-fMRI methodology from the imaging
side. Meier et al. 2022 (`10.1016/j.expneurol.2022.114111`) remains the nearest
published precedent for the pipeline and should be cited as such in any
write-up. Round 2 reaches into this gap only from the model side: F10 (the
neural→BOLD chain, seats 2 and 5) is where the unreviewed risk is closest to
being covered, and its discovery that not even the hemodynamic model in force
is named in any project document (seat 5 had to read the installed ANNarchy
extension to identify `balloon_RN`) shows how unreviewed that flank is.

---

# Tier 1 — raised by five or more seats

## F1. The non-striatal firing-rate bands: unsourced, state-blind, and steering both fits — all seven seats

**Seats: 1 (1.2), 2 (2.6), 3 (3.1), 4 (4.13), 5 (5.5), 6 (6.8), 7 (7.11).**
The unanimous finding of the round, and every seat reached it against its own
standard.

**What BGM_22 does.** `get_loss.get_firing_rate_loss` scores nine populations
per loop against `plausible_ranges`. The three striatal bands are derived,
sourced and caveated in `experimental_data/activity_striatum/README.md` — every
seat that examined them calls them exemplary. The six others are not: `stn`
(28, 80) and `snr` (21, 93) carry the bare comment "from [Li et al., 2015]" —
a citation with no journal, DOI, species, preparation or disease state, which
resolves to nothing anywhere in the repository — and `gpe_proto` (75, 85),
`gpe_arky` (15, 20), `gpe_cp` (75, 85) and `thal` (15, 30) carry no citation at
all. These bands are half the total loss and the whole of the gate
(`firing_rate_gate` = 0.5), and `TODO.md` §10 records that both DBS conditions
currently score ~0.87 against them — the gate fires on everything, and nobody
can say whether the bands or the model are at fault.

**What the seats do, and on what evidence.** Every modeling seat cites its rate
targets with species, preparation and disease state: seat 1's lineage tunes to
Mallet et al. 2008/2012 (urethane-anesthetized rats, control and 6-OHDA) and
per-state PD ranges from de la Crompe et al. 2020, as cited in Chakravarty
2022; seat 3's macaque baselines are aggregated from ~20 monkeys with the
methodology published, and its parkinsonian values are recomputed from
Tachibana et al. 2011 spike data with n's and significance tests (Shouno 2017
Table 1); seat 4 sources every band-like target in Kumaravelu 2016 to Kita &
Kita 2011 (rat). These are measured, resolvable numbers — the demand that a
source be findable is an auditability convention, but the state-dependence of
BG rates is measurement, not convention.

**Why theirs is better here, concretely.** Three independent problems compound:

- *Provenance*: an unresolvable citation on half the loss cannot be audited by
  anyone, including the project's future self — while the striatal side of the
  same function shows the project knows exactly how to do this (seats 1, 3, 4,
  5, 6, 7 all make this internal-consistency argument).
- *Disease state*: nothing states whether any band is healthy or parkinsonian.
  In MPTP macaques GPe *falls* from 65.1 ± 25.6 to 41.1 ± 22.3 Hz (Tachibana
  2011 as recomputed in Shouno 2017 Table 1(A), seat 3), and McGregor & Nelson
  2019 Fig. 4 tabulates prototypic GPe *down*, STN, GPi/SNr *up*, thalamus
  "?" in parkinsonism (seat 7, within-list consensus with Wichmann 2019). The
  model is required to hold prototypic and cortex-projecting GPe at 75–85 Hz —
  at or above seat 3's *normal*-macaque plausible band and roughly double the
  parkinsonian-macaque mean. Seat 2 adds that the arky band (15–20 Hz) excludes
  the rodent in-vivo average (~10 Hz, Giossi 2024 §3.1) and that `gpe_cp` — an
  Npas1-class population by the model's own BOLD-weight citation — is given the
  prototypic band although the one condition-relevant datum reported for
  Npas1⁺ cells is *hypoactivity* under depletion (Pamukcu et al. 2020, mouse,
  via Giossi 2024 §5).
- *Width*: the two GPe bands are the narrowest in the table (±6 % of centre,
  against ±49 % for dSPN and SDs of ±25 Hz in any monkey GPe sample seat 3
  knows), with no width rule stated — so the least-sourced bands bind the gate
  hardest (seats 3, 6).

Seat 5 adds an internal tension: four of the six bands exclude or sit at the
edge of the operating point (GPe-Proto ≈ 40 Hz, GPe-Arky ≈ 10–12 Hz, STN
≈ 15 Hz, thal ≈ 10 Hz, Goenner 2021 Fig. 7) at which the inherited weight
table was validated in the lab's own predecessor — so the gate pushes the
network away from the regime the frozen weight ratios were tuned in. Seat 7
adds, from Wichmann 2019 [macaque, human], that GPi/STN rates depend on disease
*stage* through extrastriatal dopamine (D2-like receptors on striatopallidal
terminals in primate GPe, D1/D2 on STN afferents, D1-like in GPi/SNr;
"comparatively maintained pallidal and nigral dopamine levels in early
parkinsonism may… maintain GPi/SNr firing rates at levels close to normal") —
the model has no extrastriatal dopamine term, so its fitted `base_mean`
parameters absorb whatever that effect is; this bears on `TODO.md` §30, which
frames the dopamine question as striatal only.

**Evidence class: experimentally grounded** (the state-dependence and the
parkinsonian directions are measured; the provenance gap is checkable fact).

**Change proposal.** Write the derivation document for the six non-striatal
bands in the style of `activity_striatum/README.md` — per band: resolvable
source, species, preparation, medication state, width rule; resolve or replace
"[Li et al., 2015]"; check each band's parkinsonian *direction* against
Tachibana 2011 (via Shouno 2017 Table 1) and McGregor & Nelson 2019 Fig. 4;
mark the thalamic band as unconstrained in parkinsonism (both physiology
reviews mark it so); state where a band is an assumption, which is itself
usable information. Do this **before** `TODO.md` §10's threshold calibration —
calibrating a threshold against undocumented bands sets one unknown from
another. Whether the bands need a DBS-on variant stays with §10; F19 below
carries the related on-stage objective problem.

**Blocking status:** before §10, and before any fit whose gate is enabled.

## F2. No stated criterion separates "DBS changed this parameter" from optimiser and solution-set noise — six seats

**Seats: 1 (1.4), 2 (2.1), 3 (3.3), 4 (4.12), 5 (5.6, 5.7), 6 (6.7,
degeneracy half).** The project's central claim — fit off, refit on, read off
which parameters had to change — currently has no acceptance criterion.

**What BGM_22 does.** One CMA-ES fit per condition at seed 42, a staged on-fit
(`get_loss.split_param_list`), a ~10-seed stability rerun of the winning
vector planned in `TODO.md` §32, and an inference-design entry (`TODO.md` §2)
that already names two near-degenerate parameter pairs and costs three
remedies (single-mechanism scans, multi-start, L1) but defers the decision
until after the first fits. No document states how large an on-vs-off delta
must be, relative to what spread, before it is reported as "what DBS did".

**What the seats do, and on what evidence.** All six argue from published
methodological practice whose content is checkable rather than from taste:

- Seat 5's own predecessor (Maith et al. 2021 §2.4, §3.2) ran twenty
  optimization processes per fitted dataset and asserted a parameter
  difference only across groups with t-tests, FDR and effect sizes — a device
  that exists precisely to make a fitted difference interpretable, and that
  has no replacement here: seed-stability of one winning vector measures
  simulator stochasticity, not optimizer multi-modality.
- Seat 3's lineage documented the degeneracy this guards against in its own
  constraint fits: >1000 parameterizations "equally maximising the
  plausibility scores", reducible to 15 base solutions, all fifteen carried
  through every analysis (Liénard 2024 §5.4; Girard 2021 §2.5) — and notes the
  off-fit's solution set enters the difference exactly as the on-fit's does.
- Seat 2's lineage re-derives a mapping under perturbation before interpreting
  it (Dunovan 2019: N = 15 resampled networks through the full pipeline;
  Corbit 2016: three connectivity realisations × three runs, seed-matched
  comparisons) and adds two tests absent from §2: **parameter recovery on
  synthetic data** (simulate BOLD at known DBS parameters, refit, report which
  of the 13 on-stage parameters are recoverable at all at this noise level —
  the cheapest possible test of the central claim) and **held-out validation**
  (fit on a segment of the 310 TRs, score on the rest; a 309-sample
  correlation has SE ≈ 0.06, and no artifact records it).
- Seat 4 frames it as out-of-sample discipline (its own DBS model was judged
  on a frequency profile it was not fitted to): a mechanism whose every free
  parameter is fitted to the single series it explains cannot be
  distinguished, on that series, from any other mechanism with the same
  degrees of freedom — and asks that the caudate free control become a
  **pre-registered pass/fail** (state before the on-fit what caudate BOLD
  correlations and rates must do for the inference to stand).
- Seat 5 adds the single-subject analogue of its predecessor's
  discriminability check (Maith 2021 Table 4): the **2×2 cross-condition
  evaluation** — off-parameters on on-data and vice versa; if the on-fit does
  not beat the off-parameters on on-data by more than the restart scatter,
  the refit captured noise.

**Why theirs is better here.** These are methodological devices, and every
seat says so — but they are the checkable kind, and the claim they protect is
the project's product. `TODO.md` §2 is the right entry with the right option
family; what is missing is the commitment, fixed *before* the on-fit runs:
restarts per condition, per-parameter restart scatter reported, and the rule
that a parameter is reported as changed only where its on-off delta exceeds
that scatter (plus a seed-robustness re-evaluation of the final vectors).

**Evidence class: methodological** — uncontested across six seats.

**Change proposal.** Extend `TODO.md` §2, before §32's on-fit is interpreted:
(1) N independent restarts per condition (both conditions — the off-fit's
spread enters the difference identically); (2) per-parameter restart scatter
reported beside every fitted vector; (3) the pre-stated acceptance rule; (4)
the 2×2 cross-condition evaluation as an acceptance criterion; (5) a
parameter-recovery study on synthetic BOLD for the 13 on-stage parameters;
(6) a held-out-TR split, with per-region correlation uncertainties (~0.06)
reported next to every on-vs-off delta; (7) the caudate control pre-registered
as pass/fail. Items 1–3 change how many fits are needed, so deciding them
first is cheaper than deciding them after.

**Blocking status:** blocks the *interpretation* of §32, not its launch; §2 is
already ordered before interpretation, and this fixes its content.

## F3. The fitted models' dynamical regime is never measured, while pattern — not rate — is the field's parkinsonian marker — five seats

**Seats: 1 (1.5), 2 (2.7), 3 (3.2), 6 (6.7), 7 (7.16).**

**What BGM_22 does.** The only spiking statistics ever computed are mean rates
over the 9.9 s probe; the loss JSON stores rates, per-region BOLD
correlations, sample counts and the gate flag. Whether a fitted model is
asynchronous or oscillatory — whether its STN–GPe loop sits below or above the
oscillation boundary — is never measured, and the BOLD loss cannot see it (the
drive is per-TR, the balloon low-passes the rest).

**What the seats do, and on what evidence.** Seat 3's lineage shows *why* mean
rates cannot certify the regime: in Shouno 2017 Fig. 3 the parameter regions
whose mean rates match the normal and the parkinsonian state *overlap*, while
the states separate cleanly on oscillation and burst measures recomputed from
monkey spike data (STN oscillatory cells 5.5 % normal vs 36.3 % parkinsonian;
Tachibana 2011, Table 1). Seat 1's lineage validates coherence and phase per
state (Lindahl 2016 Figs. 2–3) and maps the oscillatory boundary of STN–GPe in
the rate plane (Bahuguna 2020 Figs. 2–3), with the experimental anchors cited
there (Mallet 2008 6-OHDA rat; Tinkhauser 2017 human STN off medication).
Seat 2's lineage organises its parkinsonian modeling around the pattern
phenotype ("amplification of synchronous, rhythmic activity", Corbit 2016,
citing Brown 2001, Levy 2002, Sharott 2005). Seat 7 reports both physiology
reviews making the rate-to-pattern shift their organising theme — Wichmann
2019: "Models such as the 'rate' model are now clearly outdated"; McGregor &
Nelson 2019: beta-band STN LFP power correlating with bradykinesia/rigidity
(Brown 2001; Kühn 2009, human intraoperative) — within-list consensus, with
causality explicitly open. Seat 6's own models needed intermediate neural
observables (STN activity time courses, spectrograms) to disambiguate DBS
mechanisms that behavioural endpoints could not separate (M&C16 Figs. 7–12;
N22 Figs. 6–7).

**Why theirs is better here.** Nobody proposes fitting these statistics — the
subject's electrophysiology does not exist, and seat 3's own model missed some
of its targets. The demand is a **reported diagnostic**: the probe already
records every spike (`get_loss.py` monitors `["spike"]` on every non-TimedInput
population), so STN/GPe spectra, a synchrony summary and burst fractions cost
one analysis and no new simulation. It catches three failure modes the
inference would otherwise inherit silently: an off-fit that is not
recognisably parkinsonian in the one currency the field trusts; an off-fit
whose oscillation frequency is set by the rat delay set (F26); and an
on-vs-off delta that operates by flipping the network across the oscillation
boundary while being reported as a drive-weight change (seat 1). It is also
what makes the fitted `dbs_depolarization` interpretable — seat 2's numerical
check (2.9) shows it is measurably a suppress-and-entrain knob that
phase-locks the stimulated STN at high amplitude, which only a spectral
diagnostic would surface.

**Evidence class: experimentally grounded** (the pattern phenotype and its
direction are among the most replicated findings in the field; the diagnostic
itself is a reporting requirement).

**Change proposal.** Compute per-population burst fractions, 8–35 Hz power and
a pairwise spike-count synchrony summary from the existing probe recordings;
write them into `data_BOLD_optimization/loss_<appendix>.json` beside the
rates; report them for the accepted off- and on-fits against the stated monkey
and human references (Shouno 2017 Table 1; the directions of McGregor & Nelson
2019 Fig. 4); and treat "the fit reproduces the BOLD but shows no parkinsonian
pattern signature" as a result to report, not to hide. Add them to the
phase-1 validation-run artefacts.

---

# Tier 2 — raised by three or four seats

## F4. Parameter provenance is an archaeology project, in a repository whose data provenance is exemplary — four seats

**Seats: 1 (1.11), 2 (2.12), 3 (3.9), 5 (5.1–5.3).**

**What BGM_22 does.** `parameters.csv` is a value table with no reference
column. `get_loss.py` calls the projection weights "the literature values";
the only lineage record is the `BGM_v07` docstring. The GPe neuron refit is
recorded in one CSV header cell ("refitted, data from Bogacz et al. 2016") with
no fit protocol or quality shown; the synaptic constants (`tau_ampa` 2 ms,
`tau_gaba` 10 ms, `E_gaba` −70 mV) differ from both predecessors' published
sets (10/20/−90 in Goenner 2021, 10/10/−90 in Maith 2021) with the departure
recorded nowhere; the uniform in-degree of 10 has no stated basis; and the
delays' provenance was recovered only by the previous review round.

**What the seats do, and on what evidence.** Per-row source columns with tuned
values named as tuned: Corbit 2016 Table 2 (probability, conductance, kinetics
*and* references per connection, the tuned/measured rule in the footnote);
Lindahl 2016 Tables 7–9 ("n.d., estimated" written out); Girard 2021 Tables
1–2 (footnotes A–O, per-axon bouton counts and their tracer studies);
Kumaravelu 2016 Table 1 (a citation per delay). All four seats admit this is a
reporting convention, not an experiment — seat 2's own later papers are weaker
than its Corbit form, and says so. The convention's value is what this round
demonstrated: three of the sharpest findings (F9, F11, the delay set) were
recoverable only by archaeology that a reference column would have made
one-line checks.

**Why theirs is better here.** The project's product is a statement *about
parameters*; a fitted scaling on a base weight of unstated origin transmits no
interpretable meaning. And the repository's own `experimental_data/` READMEs
are the counterexample to its own CSV — the project demonstrably knows the
format.

**Evidence class: methodological** (auditability; internal consistency with
the project's own standard).

**Change proposal.** A provenance companion for the `BGM_v07_p01` column in
Corbit-Table-2 form: per row (or per CSV section) the source — "Goenner et al.
2021 Table 4/5, ×C rescaled" (F9), "refit on Abdi/Bogacz step-current data"
with the fit record and f–I comparison (Goenner Fig. 2 is the template), "no
source" where true; one sentence on the origin of `number = 10`; the
synaptic-constant departure from both predecessors recorded next to the
values; "Bogacz et al. 2016" and "[Li et al., 2015]" resolved to citable form.

## F5. The hyperdirect pathway under DBS: absent as a mechanism, present as measured input, inexpressible as synchrony — four seats

**Seats: 2 (2.8), 3 (3.4), 4 (4.4), 6 (6.6, 6.12).** The panel's three
DBS-capable seats plus the nearest-neighbour lineage, converging with
different emphases.

**What BGM_22 does.** The cortical afferents to STN are `TimedArray` →
`CurrentInjection` streams, structurally excluded from the DBS footprint —
"afferent DBS in this model means `gpe_proto→stn` only" (`TODO.md` §15;
`DBS.md` limitation 2). The DBS-on cortical drive is deconvolved from the
subject's own recording *under stimulation*, so the net cortical rate change
DBS produced is already in the input.

**What the seats do, and on what evidence.** Seat 4's lineage built the
evidence base: in awake rats, STN-DBS-evoked cortical potentials decompose
into R1 (1.35 ± 0.07 ms, direct antidromic activation of L5 axons — too fast
for trans-synaptic routes, anaesthesia-resistant), R2 and R3, with "antidromic
activation of the cortico-thalamic-cortical pathway… sufficient" and the
orthodromic BG route "not required" (Kumaravelu 2018, ECoG in 10 analysed
rats, model decomposition; presented with the honest limit that the same paper
does not claim these elements are therapeutic). Seat 3 adds the human datum:
DBS-evoked antidromic responses over prefrontal cortex at 6 ms latency (Chen
et al. 2020, as cited in Liénard 2024 §4). Seat 2's lineage contributes the
mechanism BGM_22's constant efficacies cannot express (activity-dependent
axonal/synaptic failure under high-frequency driving, Rosenbaum et al. 2014 as
summarized in Rubin 2017 — reported second-hand and marked as such). Seat 6's
own models omit the pathway entirely and admit the omission was justified only
by "functional significance… not fully understood" — under the round-2 rules
it therefore proposes nothing, and records that BGM_22 is *ahead* of its own
practice here.

**Why this matters here, precisely.** Three consequences, separated:

- The *rate* half of the cortical DBS effect is **not missing** — it enters
  empirically through the DBS-on drive. Seat 4 calls this a real structural
  advantage of fitting a human under stimulation, and seat 6 (6.12) draws the
  frame: the fitted DBS parameters estimate the **intra-BG effect given the
  observed cortex**, which is a defensible quantity but a narrower one than
  "what DBS did"; the write-up must say so verbatim.
- The *synchrony* half is **inexpressible**: seat 4's own data show a cortical
  volley time-locked to every pulse, i.e. a 125 Hz synchronous event across
  the corticofugal population, while BGM_22's streams are independent per-bin
  draws at `r_sc = 0` — and the project's own
  `input_streams/README.md` §3 shows input correlation is the dominant
  determinant of simulated BOLD amplitude. A DBS-induced rise in cortical
  input synchrony at unchanged mean rate would move the BOLD strongly and
  cannot occur in either condition.
- The *attribution* consequence: whatever the patient's hyperdirect activation
  contributed to the DBS-on BOLD is forced into the pallidal-afferent,
  efferent and passing-fibre parameters (§15 already states this).

**Evidence class: experimentally grounded** (seats 3 and 4), with seats 2 and
6 as differences endorsing the documentation.

**Change proposal.** (1) The cheap bound, from seat 4: the shared-modulation
machinery already exists and is merely set to zero (`make_global_p_trace`) —
run the DBS-on condition once with a non-zero cortical shared modulation and
report the simulated BOLD amplitude change as a sensitivity bound on the
inexpressible effect. (2) The structural option, from seat 3, if the triage
wants the mechanism representable: a pulse-locked spike-count component added
to the STN cortical stream, its amplitude a fourth fitted DBS parameter — the
count→current conversion passes through Python every 110 ms chunk and the
pulse times are deterministic, so this touches no cache, neuron model or
compiled network (contra §15's "spiking soma" remedy, which is heavier than
needed). (3) Either way: restate §15's misattribution bound and the
intra-BG-given-observed-cortex framing wherever fitted DBS parameters are
interpreted.

## F6. The pooled STN/GPi/GPe ROIs mix the two loops by a population-size accident — three seats

**Seats: 2 (2.3), 3 (3.10), 4 (4.10, the arithmetic).**

**What BGM_22 does.** GPi, GPe and STN BOLD monitors pool the caudate-loop and
putamen-loop copies of each nucleus; with equal 100-neuron populations the
loops enter at 50/50 (`BoldMonitor`'s size-share fallback; the GPe cell-type
factors are identical per loop). Only the putamen loop carries DBS, so the
three ROIs closest to the electrode are half composed of a compartment that is
structurally barred from any direct DBS effect, at a ratio set by
`stn.size = 100` rather than by the associative/motor territory proportions of
the human nuclei.

**What the seats do, and on what evidence.** Seat 2's lineage sets shared-
nucleus divergence from anatomical work (Fox & Rafols 1976; Smith et al. 1998,
as cited in Dunovan 2019) but concedes its channel construct is a convention
and does not propose loop coupling on its authority. Seat 3 concedes it has
no measurement that resolves functional-territory volume shares and therefore
licenses no replacement value — the point is reported as an unstated
assumption, not a wrong number. Seat 4 supplies the checkable arithmetic from
the project's own files: the DBS-affected share of the pooled STN BOLD is
0.4 × 0.5 = 20 %, coincidentally near the subject's whole-STN VTA overlap of
19.5 % — but by a different route, since associative plus limbic territory is
63 % of the real volume while the model's caudate stand-in gets 50 %.

**Why this matters here.** An attenuated modeled DBS response in the STN, GPi
and GPe ROIs is compensated somewhere — plausibly in the three fitted DBS
parameters. The 50/50 is not defended anywhere because it was never chosen.

**Evidence class: methodological** (the share is an accident, checkable in
code; no seat can supply the anatomically correct number).

**Change proposal.** State and justify the loop shares of the pooled ROIs (any
defensible anatomical basis beats a population-size accident; the
`scale_factor` mechanism already used for GPe cell types can carry it), and
report, for fitted models, each loop's share of every pooled ROI's variance
and of its on–off change. Document the implied 50:50 in `model_v07.md` §3.6
beside the GPe abundance discussion either way.

## F7. The subject's medication state and the model's dopamine state have to be settled together, and neither is recorded — three seats

**Seats: 1 (1.10), 5 (5.15), 6 (6.9); seat 7 (7.11) adds the extrastriatal
addendum carried in F1.**

**What BGM_22 does.** The model asserts a medication-off subject: striatal
surround and bands from Liang et al. 2008 medication-off, `phi_1 = phi_2 = 0`
with `TODO.md` §30 tracking whether zero is the right depleted value in the
Humphries convention. But no document records what the Berlin subject's
medication state during scanning actually *was*, nor the ROI hemisphere
convention (the VTA arithmetic pools two electrodes), nor the acquisition
parameters. The Roadmap already orders the medication question before any
full-length cache.

**What the seats do, and on what evidence.** Seat 5's predecessor recorded the
acquisition state precisely — patients scanned DBS-OFF *with* their usual
medication — and then *needed* that fact to interpret a fitted result (Maith
2021 §4.1 explains an absent STN/GPi rate increase by the medicated state,
citing apomorphine effects). Seat 6's lineage carries the only graded-dopamine
machinery on the panel and reports its own medication values are tuned, so it
proposes no numbers — but its measured anchor (striatal DA 150–400 nM control,
Schultz 1998 as cited in N22) and its behavioural result that medication state
*inverts* effects support the ordering: this is not a small correction.
Seat 1 frames the requirement as internal state consistency among the fixed
ingredients — rates (chronic med-off), connectivity kernel (currently mixed,
F11), φ (§30) — all of which must describe the same physiological state.

**Why this matters here.** If the subject was scanned on medication — as the
predecessor's cohort from the same clinical population was — the Liang
anchors, φ = 0 and every cache built on them target the wrong state, in both
conditions of the inference.

**Evidence class: methodological** (the scientific question is already
correctly planned in §30 and the Roadmap; the gap is recorded provenance).

**Change proposal.** Write the `experimental_data/berlin_data/` README before
phase 2 builds anything expensive: the subject's medication state during
scanning, the on/off session protocol, field strength/TE, and the hemisphere
convention of the ROI series — in the style the other data directories
already follow. Resolve §30 with its existing plan; extend its scope note
with seat 7's extrastriatal-dopamine point (F1).

## F8. `gpe_cp`: an inherited population whose identity, projections and rate band cannot be reconciled — three seats

**Seats: 2 (2.6, band half), 5 (5.12), 7 (7.2).**

**What BGM_22 does.** v07 keeps the three-way GPe split with `gpe_cp`
projecting to all three striatal populations (0.5/0.5/0.8), receiving fitted
cortical drive, sharing `gpe_proto`'s neuron parameters ("just like gpe
proto", `parameters.csv`) and its 75–85 Hz band, and weighting the GPe BOLD at
0.10 via the NPAS1⁺NKX2.1⁺ abundance (`model_v07.md` §3.6).

**What the seats know, and on what evidence.** Seat 5 owns the provenance:
Goenner 2021 introduced GPe-Cp *for* the cortico-pallido-cortical loop — the
paper's central novel claim, with the anatomical projections tabulated
(Abecassis 2020, Chen 2015, Saunders 2015 for GPe→cortex; rodent) — and that
efferent cannot exist here, since the cortex is a prerecorded stream. Seat 7
adds the identity problem [mouse]: Courtney 2023's NPAS1⁺NKX2.1⁺ class —
the class whose 12 % abundance the BOLD weight was matched to — "project[s]
exclusively to the midbrain, the cortex and the reticular nucleus of the
thalamus", *not* to the striatum, while the model's `gpe_cp` is a
striatum-projecting population; Box 1 leaves room for a second
striatum-projecting class, so the model is not contradicted, but nothing
identifies its population, and `gpe_cp` is expanded nowhere in the repository.
Seat 2 adds the state datum: the one condition-relevant report for Npas1⁺
cells is hypoactivity under depletion (via Giossi 2024), against the 75–85 Hz
band (carried in F1).

**Why this matters here.** A fitted change in `gpe_cp` parameters under DBS
must not be narrated as a pallido-cortical pathway effect — the pathway is not
in the model — and the population's band and BOLD weight currently borrow from
two different identities. All the targeting facts are mouse, of the
marker-defined kind Courtney and Wichmann both say is untested in primate, so
no rewiring is licensed.

**Evidence class: difference, not deficiency** (documentation), except the
band half, which F1 carries.

**Change proposal (documentation only).** State in `model_v07.md` what
`gpe_cp` denotes, which experimental population it is meant to be, that its
namesake efferent is structurally absent, and the resulting claim boundary;
reconcile or flag the band/abundance identity tension when F1's band document
is written.

---

# Tier 3 — two seats, or one seat with strong evidence

## F9. The BG weight skeleton is Goenner 2021's task-tuned table — verified twice — and the frozen subtype ratios point against the only subtype-resolved measurements

**Seats: 2 (2.4, 2.5), 5 (5.1).** Two seats independently diffed all 28 v07
projection weights against Goenner 2021 Tables 4–5 and found every value
matches, verbatim for non-striatal targets and ×C (50 for SPNs, 80 for FSIs)
for striatal ones. Nothing in the repository records this; `get_loss.py` calls
them "the literature values". The source itself says otherwise — "the weight
strengths… are rather abstract and were determined mainly by functional
constraints" (Goenner 2021 §4.4): tuned so a rat stop-signal network stops
correctly. The optimizer then freezes the within-cluster ratios by design.

**The measured contradictions (what makes this experimentally grounded rather
than provenance-only).** Seat 2, from its own lineage's data and review:
iSPN→arkypallidal projections measured 85 % *weaker* than iSPN→prototypic and
STN→arkypallidal 74 % weaker than STN→prototypic (Aristieta et al. 2021,
mouse, as reported in Giossi 2024 §3.2) — while the frozen table has iSPN→arky
at *twice* iSPN→proto and STN equal across all three GPe types. And the
pallidostriatal targeting ratio: the lineage's own recordings gave GPe→FSI
IPSCs of 566 ± 560 pA in every FSI sampled against 28–108 pA in MSNs (Corbit
2016, mouse slice, ChR2; modeled there as a 12–40× conductance ratio), while
BGM_22's aggregate ratio at equal in-degree is ~1.4–2.3×, with the
proto-vs-arky order onto FSIs reversed relative to Giossi 2024's account.
Species caveats (mouse → human) are stated by both seats; but the current
encoding has no species at all, only an uncited inheritance tuned for a
different task under different synaptic kinetics (seat 5's 5.3: the constants
the weights were tuned under were 10/20/−90, here 2/10/−70).

**Why theirs is better here.** Cluster scaling preserves relative balance to
condition the search — sound exactly when the encoded balance is trustworthy.
Here the balance is task-tuning, and where subtype-resolved measurements
exist they point the other way; no fitted scaling can repair a frozen ratio.

**Evidence class: experimentally grounded** (the Aristieta and Corbit
measurements), plus methodological (provenance, carried in F4).

**Change proposal.** Record the Goenner-2021-×C provenance (F4). Then either
re-derive the within-cluster iSPN→GPe, STN→GPe and GPe→FSI-vs-SPN ratios from
the reported measurements with species caveats documented, or split the
affected clusters so the fit can move the ratios, or record a reasoned
rejection. After the first fit, check the conclusions' sensitivity to the
unanchored ratios before interpreting any cluster scaling.

## F10. The neural→BOLD chain is a stack of untested choices, one of them a silent reversal of the predecessor's published convention

**Seats: 2 (2.2), 5 (5.9, 5.10).**

**What BGM_22 does.** `I_CBF` is mapped to *net* synaptic current (`I_v`
striatum, raw `I` elsewhere): GABA enters with its driving-force sign, so
inhibition *reduces* the BOLD drive; the GPe monitors read raw `I` while the
membrane sees `f(I, nonlin)`; `I_base` and the DBS somatic term are invisible
to the monitors; the hemodynamic model in force is ANNarchy's default
`balloon_RN` (Stephan 2007 coefficients) — named in no project document (seat
5 identified it from the installed extension); `normalize_input=2000` sets the
baseline from a sub-TR window inside the first recorded TR, and the balloon
transient sits inside the scored window.

**What the seats do, and on what evidence.** Seat 2's lineage published the
worked example of exactly this hazard class: voltage-derived and
synaptic-current-derived LFP proxies of the *same* simulations give
qualitatively different spectra (Rubin 2017 Fig. 2), with the
reversal-potential caveat on GABAergic contributions from Buzsáki et al. 2012
as cited there. Seat 5 owns the reversal: Maith 2021 §2.3 computed the BOLD
input as a deliberately **sign-blind** synaptic-activity trace ("no matter if
excitatory or inhibitory"), motivated by Logothetis 2001 / Mathiesen 1998,
2000 (synaptic activity, not spiking, predicts BOLD; inhibitory transmission
also consumes energy) — those sources do not adjudicate signed vs unsigned,
and seat 5 says so, but the predecessor's choice was stated and argued while
BGM_22's departure from it is recorded nowhere. The stakes are not cosmetic:
a v07 striatal neuron receives ~61,000–69,000 GABAergic spikes/s from the
compensation streams, so the two conventions can differ qualitatively in two
of seven scored regions, absorbed invisibly by the fitted drive weights.

**Evidence class: methodological** (the proxy-dependence demonstration is a
published model result; no seat claims to know the correct CBF driver).

**Change proposal.** Document the chain end to end in `model_v07.md` §3.6
(model, coefficients, sources — Maith 2021 Table 2 is the house template; the
field-strength parameters against the acquisition; the baseline/ramp
question of `TODO.md` §8; the onset transient in the scored window). Before
the production fits, evaluate one parameter set under 2–3 defensible `I_CBF`
mappings (net current as now; sign-blind |exc| + |inh| as the predecessor;
GPe on `f(I, nonlin)`) and record the per-region correlation shifts; interpret
striatal parameters only if the inference survives the proxy choice.

## F11. The striatal connectivity kernels pool healthy and dopamine-depleted datasets into one state-less fit

**Seat: 1 (1.1) — one seat, but grounded in the repository's own
condition-labelled data, and the round's clearest new discovery.**

**What BGM_22 does.** The seven `p(d)` kernels in `fitted_params.json` — the
entire intrinsic striatal connectivity, and via `E_outer` the size of the
missing-GABA compensation in every cache — are maximum-likelihood fits over
*pooled* condition rows: for dSPN→dSPN/iSPN→dSPN/iSPN→iSPN the pool includes
Taverna 2008's 6-OHDA and reserpine rows (0/7, 0/8 connected) beside its
baseline rows; for FS→dSPN/FS→iSPN it includes Gittis 2011's 6-OHDA rows
(verified against `connectivity_fit.py`'s condition-labelled `datasets` dict).

**What the seat's lineage does, and on what evidence.** Healthy-state
parameterization with depletion as explicit, cited multiplicative factors:
SPN–SPN collapse under depletion (Taverna 2008 — the very rows in BGM_22's own
spreadsheet) and FSI→iSPN roughly doubling while FSI→dSPN does not move
(Gittis 2011, mouse 6-OHDA), encoded in Lindahl 2016 Table 9 and Chakravarty
2022's methods. The state *difference* is measurement; the lineage's α_dop
map of it is convention, and the seat says so — with the honest caveat that
both are acute rodent models and chronic human transfer is not established.

**Why this matters here.** The pooled kernel corresponds to no preparation
that exists: dSPN→dSPN is diluted by depleted zero-rows while FS→iSPN is
pulled up by its depleted row — at short range the fitted kernels *reverse*
the FS target preference relative to every healthy dataset in their own pool
— and the two targets of the same FS axon get σ differing 2.8-fold, which has
no anatomical reading. Both DBS conditions share the kernel, so the on-off
contrast is not directly biased; the damage is to the absolute state, to
`E_outer` (every cache), and to §4's rate self-consistency. The project chose
its rate anchors by state deliberately (Liang med-off); the connectivity
should be chosen by the same principle.

**Evidence class: experimentally grounded.** **Change proposal.** Refit the
kernels per condition from the already-labelled `.ods` rows; adopt the
depleted kernel (or healthy plus explicit cited depletion factors); apply the
same state question to the IPSC-amplitude mixtures in `get_weights.py`;
document; rebuild the short caches. Roadmap phase 1, before any full-length
cache.

## F12. The caudate "free control" is cleaner in the model than in the subject

**Seats: 6 (6.2), 7 (7.3).**

**What BGM_22 does.** The two loops share no projection, only the putamen loop
is stimulated, and the caudate loop's on-vs-off BOLD change is the designed
free control (`TODO.md` §2).

**What the seats know, and on what evidence.** Seat 7 [macaque, human;
within-list consensus — Haber 2016, Emmi 2020, McGregor & Nelson 2019]:
corticostriatal terminal fields from areas 5 mm apart overlap by 50 %
(Averbeck 2014, macaque tracing); through GPe and STN, segregation holds for
associative regions but *not* for motor ones, which receive from associative
territories (Shink 1996, EM-level); and parkinsonism further degrades
functional segregation (four studies of receptive-field despecification). So
the strongest cross-channel route in the anatomy runs exactly the way the
model would need it — associative into motor — and the model wires zero.
Seat 6, from the subject's own files: the VTA overlaps the *associative* STN
at ≈ 0.09 (5/68 + 7/68, `sub-01_overlap.csv`), which the design rounds to
zero, so the real caudate-territory on-off change contains a small direct DBS
component the model will attribute to cortical drive.

**Why this matters here.** Any caudate-vs-putamen contrast is an **upper
bound** on channel independence, not a clean control reading; and the
control's pass/fail (F2) inherits both cracks.

**Evidence class: experimentally grounded** (both halves are measured — one
in the literature, one in the subject's own files).

**Change proposal.** Document both facts where the control is defined (§2) and
in `model_v07.md` §1. Bound the 0.09 effect analytically or with one
sensitivity run giving `stn:caudate` its measured coverage (a compile-level
extension of the DBS retrofit). Optionally, seat 7's testable version: add the
single associative-GPe → motor-STN projection the reviewed anatomy supports
most directly, in its own optimizer cluster, so the fit can zero it — a fitted
non-zero value is then the model's own estimate of channel leakage, a result
rather than an assumption.

## F13. The corticostriatal proportion table is silently reused for STN, GPe and thalamus, where the tracing says the mixes differ

**Seat: 7 (7.4). Within-list convergence: consensus that the mixes differ
(Emmi 2020 documents the corticosubthalamic topography, Haber 2016 the
corticostriatal one); the specific S1 negative is one voice (Von Monakow
1978, macaque autoradiography — no study on the list reports an S1→STN
projection, but a single-study negative from 1978 is weaker than a
replicated positive).**

**What BGM_22 does.** `BGM_v07` passes the same
`cortical_proportions_dict` to `Microcircuit` and to `CorticalInputs`, so a
putamen STN neuron draws 13 % of its 500 cortical afferents from S1 — a region
for which the one primate study that looked reports no STN projection — and
the same striatal table sets the thalamic and pallidal cortical mixes. The
README that derives the table is explicit that it describes corticostriatal
afferents and never claims another target; the reuse is undiscussed.

**Why this matters here.** The corticosubthalamic drive is the only
excitatory drive `stn` receives, the hyperdirect route is the DBS-relevant
one (F5), and a misallocated mix changes the *timing* of the drive — and
timing is what the loss is made of. The corticosubthalamic topography that is
established (M1 dorsolateral, SMA/ventral-premotor medial with inverse
somatotopy; Nambu 1996–2000, confirmed Miyachi 2006) does not resemble the
striatal proportions.

**Evidence class: experimentally grounded.** **Change proposal.** Give
`CorticalInputs` its own proportion table — minimally: S1 → 0 for STN, motor/
premotor/SMA dominant per the established topography, dlPFC marked uncertain
(Von Monakow and Haynes & Haber 2013 disagree) — or document in both
`cortical_proportions/README.md` and `model_v07.md` §8 that the striatal
table is being reused for three structures it was not derived for.
Cache-invalidating (CI caches), so it belongs in Roadmap phase 1.

## F14. Renormalising the proportions over seven ROIs is a covariance assumption, asymmetric between the loops

**Seat: 7 (7.13). Within-list convergence: consensus that the omitted input
is large and concentrated (Haber 2016: dense vmPFC/dACC/OFC terminal fields
occupy ≈ 22 % of the striatum; Emmi via Isaacs 2018: the same regions are
disproportionately strong for the STN; BGM_22's own Borra table is the third
confirmation).**

**What BGM_22 does.** The proportion columns renormalise over the seven
Berlin ROIs and sum to 1, so all ~7000 of an SPN's cortical afferents are
driven by the seven retained regions' deconvolved series — asserting that the
25–60 % of afferents arising in cingulate, insula, temporal, orbital and
ventrolateral prefrontal cortex fluctuate, TR by TR, exactly as the retained
motor/dorsolateral-prefrontal regions do. The omitted mass is largest in the
caudate rows, so the caudate loop's drive is the more heavily reconstructed —
and the loop contrast is what the inference reads. The README's caveat states
the omission but not the covariance assumption.

**Why this matters here.** The assertion is about precisely the fitted
quantity (a time-course correlation), and the omitted regions are limbic and
associative — functionally distinct from the retained seven.

**Evidence class: experimentally grounded (narrow).** **Change proposal.**
State the assumption explicitly in the README; report the caudate/putamen
asymmetry of the omitted fraction; if it is to be tested, the strictly weaker
alternative costs nothing anatomically — columns summing to the retained
fraction with the remainder driven flat at the same 5 Hz mean, preserving
mean drive while dropping the covariance claim.

## F15. The DBS effect is exactly linear in stimulation frequency, by construction

**Seat: 4 (4.2).**

**What BGM_22 does.** Every DBS term is gated by `pulse(t)` with no
adaptation, depression or pulse-to-pulse interaction anywhere, so the expected
perturbation over any interval is per-pulse × pulse count — proportional to
frequency, no threshold, no saturation.

**What the seat's lineage does, and on what evidence.** Its model *produces*
the thresholded, saturating frequency profile (no effect below 40 Hz, decline
50–130 Hz, saturation above 150 Hz; Kumaravelu 2016 Fig. 11, reproduced in Su
2019 Fig. 5) without being fitted to it; the experimental anchor is the
parallel frequency dependence of symptom suppression (rat studies cited in
Kumaravelu 2016 §4.3, human/primate in §5.2 — reported as the paper's
citation trail, not read by the seat).

**Why this matters here.** The fit is at 125 Hz, squarely therapeutic, so the
fit itself is untouched — but the mechanism contradicts the best-established
quantitative fact about STN DBS, and a frequency sweep here would provably
return a straight line, so the one cheap external check is foreclosed. The
seat states its own limit: BGM_22 has no pathological oscillation to suppress
and reads out at 2.31 s, so it could not measure the dose–response even if
the mechanism supported one; a mechanistic frequency dependence would need a
memory term the equation set lacks.

**Evidence class: experimentally grounded (scope).** **Change proposal.**
Record in `DBS.md` and the write-up that the DBS representation is a
per-pulse perturbation calibrated at 125 Hz, linear in frequency by
construction, not to be extrapolated to other stimulation settings.

## F16. The antidromically invaded pallidal and nigral subsets are a fresh random draw at every pulse

**Seat: 4 (4.3).**

**What BGM_22 does.** `unif_var_dbs2` is a per-neuron, per-timestep local
uniform, so a different ~40 % of `gpe_proto:putamen` is antidromically
invaded on each pulse; only the stimulated STN's set is fixed
(`_create_dbs_on_array`).

**What the seat's lineage does, and on what evidence.** Fixed activated sets
(all STN neurons, Kumaravelu 2016; a fixed 60 % of L5 terminals, Kumaravelu
2018). The reliability half is measured: the pulse-triggered cortical evoked
potential is stable pulse-by-pulse across ~215 pulses in all six rats
(Kumaravelu 2018 Fig. 5) — a reproducibility only the same axons every time
can produce. The same source narrows the claim: antidromic propagation is
unreliable at 130 Hz (R1 reduced vs 9 Hz, Fig. 4B1) — *stochastically thinned
within a fixed axon population*, not resampled across the nucleus.

**Why this matters here.** A fixed subset produces persistent pallidal
heterogeneity (a strongly perturbed 40 % and an untouched 60 % — what
Kumaravelu 2016 §4.2 describes); a per-pulse redraw applies a diluted,
identical-in-expectation perturbation to every neuron and cannot produce it.
The pooled BOLD mean is first-order unchanged, but network consequences and
any per-neuron claim differ.

**Evidence class: experimentally grounded.** **Change proposal.** Draw the
invaded sets once (as `_create_dbs_on_array` already does for STN), store as
per-neuron masks, and keep per-pulse propagation reliability as a separate
scalar — which also makes the 125 Hz propagation failure expressible instead
of conflated with coverage.

## F17. `TODO.md` §16's planned DBS-constant checks are wrong as designed: coverage must be profiled, and the pulse-width check is void

**Seats: 4 (4.5, 4.6), 6 (6.1, 6.5) — the two seats converged independently,
including on the arithmetic.**

**What BGM_22 does.** `population_proportion` = 0.4 (subject's motor-STN VTA
overlap) is fixed while three weaker DBS parameters are fitted;
`dbs_pulse_width_us` = 100 substitutes for the hardware's 60 µs; §16 plans
sensitivity checks on both.

**What the seats show, on self-standing grounds.** Seat 4: the coverage enters
each DBS effect multiplicatively with a fitted parameter (orthodromic:
proportion × `prob_axon_spike`; antidromic: `antidromic_prob` = proportion ×
the same; somatic: proportion × `dbs_depolarization` for the pooled mean), so
a naive sensitivity check returns "insensitive" because the optimiser absorbs
the rescaling — exactly the wrong conclusion. The fitted DBS parameters are
interpretable only *conditional on* an assumed coverage, and the project's own
subjects span 0.24–0.40 on the identical quantity (0.32 sub-03, 0.24 sub-04);
seat 6 adds the per-hemisphere range of the fitted subject itself, 0.31–0.50.
On pulse width, both seats independently prove the same theorem: at
dt = 0.1 ms, `pulse(t)` is ON for exactly one timestep for any width in
(0, 100] µs — the realized train is identical, and only
`_axon_spikes_per_pulse_to_prob`'s scale factor changes (×1.67 at 60 µs),
while the reachable probability set stays [0, 1]. The planned check is
provably empty; the residue is interpretive (the fitted value means "per
100 µs pulse" and must be renormalized before cross-study comparison).

**Evidence class: methodological.** **Change proposal.** Reword §16: profile
the loss along `population_proportion` with the three DBS parameters
re-optimised at each value, over at least [0.31, 0.50] (or 0.24–0.50 to span
the cohort), and quote the fitted DBS parameters as coverage-conditional;
drop `dbs_pulse_width_us` from the check with the one-line reason recorded,
and note the ×1.67 renormalization in `DBS.md`.

## F18. Static synapses under 125 Hz stimulation: the fitted DBS-on weight changes conflate depression with plasticity

**Seats: 2 (2.8, mechanism half), 3 (3.5).**

**What BGM_22 does.** No synapse carries short-term plasticity; DBS-evoked
transmission uses the same static `w`.

**What the seats know, and on what evidence.** Seat 3's lineage fitted
GPe→STN short-term depression to measured 1–100 Hz data (Shouno 2017 Fig. 2,
experimental points from Atherton et al. 2013 — species not stated in the
text, flagged): at 100 Hz sustained firing, unitary transmission collapses
below ~0.2 within seconds. Seat 2's lineage carries the corresponding DBS
account (activity-dependent failure decoupling BG output; Rosenbaum 2014 via
Rubin 2017, second-hand and marked).

**Why this matters here.** Under run-long 125 Hz stimulation the model
operates exactly where depression saturates. For the *fit* this is absorbable
— a steady state, standing in the fitted efficacies. For the *reading* it is
not: an on-refit that lowers `gpe_proto__stn` would be reported as "DBS
weakened pallido-subthalamic transmission" when the physiological description
is frequency-dependent depression the model cannot express — a
mechanism-level conflation in precisely the quantity the project wants to
report.

**Evidence class: experimentally grounded (interpretation).** **Change
proposal.** A claim-boundary sentence wherever fitted weight changes under
DBS are interpreted mechanistically; optionally short-term depression on the
six DBS-footprint projections (ANNarchy synapse models can carry it).

## F19. The rate term biases the DBS-on fit toward "DBS changes no rate", and the probe cannot see DBS-evoked spikes

**Seat: 4 (4.7) — one seat, on facts checkable in the code.**

**What BGM_22 does.** One condition-independent band table, *added* to the
objective (not gate-only), with the probe run under active DBS; and the spike
monitors record ANNarchy's `spiked` container while DBS-evoked axon spikes go
to `axonal` — so the measured STN rate excludes the DBS-evoked volley
entirely, and the only DBS terms the rate sees are the hyperpolarizing shunt
and the antidromic reset.

**What the seat's lineage knows, and on what evidence.** Its model imposes
full STN entrainment (a convention), but the pallidal half has an
experimental counterpart: mean GPe/GPi rates roughly unchanged under STN DBS
(McConnell 2012, as cited in Kumaravelu 2016 — unread, reported as trail),
which *supports* condition-independent pallidal bands; the problem
concentrates in the stimulated nucleus.

**Evidence class: methodological.** **Change proposal.** In the DBS-on stage
use the rate term as a gate only, or exempt `stn` from the additive term;
record in `DBS.md` limitation 5 that the probe counts `spiked`, not `axonal`.
(Band provenance is F1; this finding survives even with perfect bands.)

## F20. No null-model baseline for the BOLD-correlation loss

**Seat: 1 (1.3) — one seat, self-standing statistics.**

**What BGM_22 does.** The loss is a mean per-region correlation against a
target whose drive is deconvolved from the same recording's cortex; nothing
computes what score a structureless transformation achieves.

**What the seat's lineage does.** Effects are quantified against
mechanism-removed nulls (Bahuguna 2020's burst threshold against rate-matched
Poisson ensembles) — a statistical practice, not an experiment, and argued as
such.

**Why this matters here.** Because drive and target share a recording, a high
correlation may be achievable by *any* smoothing dynamics — in which case the
fitted parameters carry little information and the on-off deltas less.
Conversely a model that clearly beats the null answers the obvious referee
question. The null costs no simulation: regress each experimental ROI series
on the (HRF-convolved) cortical series.

**Evidence class: methodological.** **Change proposal.** Compute and record
the null-model per-region correlations (cortical-mix regression, plus the
best-single-region predictor) before the first fit; require the fitted model
to be interpreted only where it exceeds them.

## F21. The Liang anchor survives its audit; the opposing measurements should be recorded beside it

**Seats: 3 (3.11), 7 (7.8, 7.9) — convergent audits, both largely
favourable.**

**What the audits found.** Seat 7: McGregor & Nelson 2019 name Liang 2008 as
one of two primary sources for elevated parkinsonian MSN firing (with Singh
2016), against Deffains 2016 reporting no change — an open conflict the README
does not record; Haber 2016 corroborates the normal-primate baseline the
README's "why so high" section leans on. Decisively in the model's favour
(7.9): the iSPN > dSPN ordering — the product of the README's single largest
assumption — is exactly what four identified-cell-type rodent studies find
without the assumption (Mallet 2006; Kita & Kita 2011; Ryan 2018; Sagot 2018,
antidromic/optogenetic identification), so the weakest link has independent
support on the one question it decides. Seat 3 independently supplies the
other counter-datum its lineage carries: unchanged striatal rates after MPTP
(Goldberg et al. 2002, as cited in Liénard 2024 §5.5 — not read directly),
against Shouno 2017 citing Liang approvingly; the README's chronic-vs-acute
reconciliation is plausible but argued without naming the opposing
measurement.

**Evidence class: difference (documentation), in BGM_22's favour on
substance.** **Change proposal (documentation).** Add Deffains 2016, Singh
2016 and Goldberg 2002 to `activity_striatum/README.md` as the recorded
conflict; note that if the "no change" side is right the surround is drawn an
order of magnitude too high — the one scenario in which the caches would need
rebuilding for a reason external to the model; and record the
identified-cell-type corroboration of the ordering, which removes weight from
the README's named weakest link.

## F22. The GPe cell-type abundances weight the BOLD signal but not the network

**Seats: 7 (7.1), 1 (1.9).**

**What BGM_22 does.** `get_loss.py` weights the GPe BOLD pooling by measured
cell-type abundances (`gpe_proportions` = 0.5/0.17/0.10; Courtney et al. 2023
per `model_v07.md` §3.6 — still uncited in the code itself) while the network
simulates the three populations at 100 neurons each. The observation model
says the GPe is half prototypic and a sixth arkypallidal; the dynamics carry a
third each — three times as many arkypallidal neurons relative to prototypic
as the abundances imply. `model_v07.md` §3.6 already records the inconsistency
as an open question.

**What the seats know, and on what evidence.** Seat 7 [mouse; within-list: the
percentages are one voice (Courtney 2023, driver lines plus single-cell
transcriptomics, established for rodent), while the subtype division and the
warning that its primate translation is untested are consensus (Courtney,
Wichmann, McGregor & Nelson)]. The claim is deliberately *not* that mouse
abundances hold in a human GPe — it is that the model should not assume them
in one place and contradict them in another; the inconsistency is
species-neutral, which is what licenses a proposal despite the mouse-only
provenance. Seat 1, from stereology-proportional practice it admits is partly
convention, identifies where the imbalance leaks into the inference: GPe
lateral inhibition is disproportionately arkypallidal in the dynamics while
the BOLD readout weights it down, and the fitted `gpe_striatum` and
`gpe_laterals` clusters absorb the difference.

**Evidence class: experimentally grounded (internal consistency).**
**Change proposal.** Either size the three GPe populations by the same
abundances the BOLD weights use (e.g. 160/58/38 at the same total), or drop
the abundance weighting and let the size-proportional default carry it — as
v07 already does for the striatum (`str_scaling_factors = None`). Either way,
write the Courtney citation into `get_loss.py` beside `gpe_proportions`, with
the mouse caveat both source reviews state.

---

# Tier 4 — single-seat findings worth recording

## F23. In the staged DBS-on fit, two of the seven loss regions are provably constant

**Seat: 4 (4.11).** The 13 on-stage parameters touch nothing in the caudate
loop, and `Cau` and `MD` pool caudate populations only — so 2/7 of the
DBS-on objective is an additive constant, the reported loss is compressed,
and a good caudate fit inherited from the off stage masks a poor putamen fit
in the headline number (CMA-ES itself is rank-based and unharmed).
**Methodological.** Proposal: optimise and report the on-stage mean over the
putamen-sensitive regions (Put, VAp, GPi, GPe, STN); report Cau and MD
separately as the control they already are — which is what the free-control
logic implies and the objective does not implement.

## F24. The passing-fibre parameter's label rests on one unvaried choice

**Seat: 4 (4.8).** `passing_fibres_strength` is interpretable only through
which projection it is attached to; it is attached to exactly one
(`snr__thal`, standing for GPi→thal per a project citation the seat did not
verify), while the candidates its own literature names (striatonigral and
pallidonigral fibres, Bosch 2011 via Kumaravelu 2016 §5.1 — discussion-level,
rat, admitted weak) already exist in the model. **Methodological.** Proposal:
refit with each candidate in `passing_fibres_list` (one run each) and report
either indifference (the choice is harmless, say so) or a difference (the
parameter is not identifiable as labelled).

## F25. The uniform in-degree of 10 sits three orders below measured striatofugal convergence

**Seat: 3 (3.7).** From its own constraint tables (macaque single-axon bouton
counts × neuron counts): ~6–7·10³ distinct MSN afferents per GPe neuron,
against 10 here — while for STN↔GPe the same arithmetic gives ~20–33, the
order BGM_22 uses, so that half is defensible on the seat's own numbers. The
fitted cluster weight absorbs the mean (ν·w), so rates are insensitive; what
the in-degree sets is input *sharing*, i.e. the noise structure of the
pallidal/nigral populations and their pooled-current BOLD, and the firing
statistics F3 asks to be diagnosed. **Experimentally grounded (bounded).**
Proposal: raise `number` toward anatomical proportions on the striatofugal
projections (cheap at 100-neuron targets) or run and record a sensitivity
check; give the `ci.n_*` counts their planned literature derivation (§8).

## F26. The subcortical delays: rat values, now recorded — and the panel splits on whether that matters

**Seats: 3 (3.6), 4 (4.1) — listed here despite two seats because they
disagree.** Both confirm the set is Kumaravelu 2016 Table 1 (rat, measured,
sourced). Seat 3: the macaque estimates its lineage published differ by up to
2× on striatofugal pathways and the STN↔GPe loop sum (6 ms here vs 9–10 ms)
sets the loop's resonance band — proposes re-evaluating accepted fits under
both macaque sets (Cmpr/NoPlkv; a recompile, no cache cost). Seat 4: the one
cross-species latency comparison in its reading (rat vs human DBS-evoked
cortical components, Kumaravelu 2018 Fig. 7) has *human* polysynaptic
latencies 20–30 % **shorter**, so "human is bigger, scale delays up" is
wrong, the transfer is better than feared, and at TR = 2.31 s the loss cannot
see it either way. Both agree it is invisible to the BOLD loss and matters
only through F3's regime diagnostics. **Difference / experimentally grounded
(moderate).** Proposal: the sensitivity pair when the F3 diagnostics exist;
provenance is already recorded.

## F27. The simulated striatal cube is smaller than one SPN dendritic field

**Seat: 7 (7.14). Within-list: one voice, and via a textbook citation
(Wilson 2004 in Shepherd, per Haber 2016) — the weakest evidential basis in
the round, stated as such.** At 227.6 µm the volume is below the ~0.5 mm MSN
dendritic field, so no simulated SPN's tree fits inside the modelled tissue —
an anatomical argument for the already-planned larger cube (`TODO.md` §24)
that §24's statistics-based table does not record, and which sets a natural
target (≥ 500 µm — costing 0.28 percentage points of simulated GABA share in
§24's own table, cheaper than the 800 µm option highlighted there).
**Experimentally grounded; fix already planned.** Proposal: record the
argument and the ≥ 500 µm bound in §24 and `model_v07.md` §7.1.

## F28. The missing-GABA self-consistency check must run per DBS condition

**Seat: 6 (6.10).** The surround is frozen at off-state rates in both
conditions while the *simulated* putamen striatum feels DBS through the
pallidostriatal cascade — so agreement in the off condition does not imply
agreement in the on condition, and it is the on-condition mismatch that would
contaminate the inference. `TODO.md` §4 as worded compares "the fitted rates"
against the assumption once. **Methodological (plan gap).** Proposal: amend
§4 to compare off-fit and on-fit rates against the surround separately and
report both gaps.

## F29. The predecessor's own STN electrode-artifact caveat is absent here

**Seat: 5 (5.11).** Maith 2021 §4.1, about the same clinical population:
implanted electrodes "probably caused artifacts in the BOLD signal,
especially in the STN signal, which reduces the validity of our results about
the STN". BGM_22 fits STN at full 1/7 weight in both conditions with no
caveat anywhere, and the STN-coupled parameters are precisely the DBS-adjacent
ones the inference most wants to read. **Methodological (grounded in the
lab's published caveat; the artifact itself unquantified).** Proposal: carry
the caveat into the project documents; once fits exist, report fit and
parameter moves with and without the STN region in the loss, treating
disagreement as a red flag on any STN-mediated conclusion.

## F30. The §25 correlation scan's amplitude arm has no experimental anchor, and the BOLD units decide whether one exists

**Seat: 1 (1.6).** The scan's readouts are simulated output correlation
(weakly anchored) and simulated BOLD *amplitude* — but the loss is
amplitude-invariant, so nothing downstream constrains the amplitude the scan
selects. The subject's own per-ROI BOLD variance is an anchor **if** the
preprocessing of `sub-01_subdiv_results.h5` preserves physical percent signal
change — unverified. **Methodological (narrow; the plan itself endorsed).**
Proposal: determine the experimental BOLD units; either add the per-ROI
variance comparison to the §25 scan or record in `input_streams/README.md` §5
that ρ is set by self-consistency alone.

## F31. No analogue of the predecessor's fixed-parameter sensitivity analysis is planned

**Seat: 5 (5.8).** Maith 2021 §2.6 varied all 59 fixed parameters ±5 % on the
fitted models and found the loss dominated by the *drive-side* parameters —
a warning that transfers directly, since BGM_22's boldest fixed choices sit
exactly on the drive side (correlations pinned at 0, underived `ci.n_*`,
proportions). With n = 1 there is no group replication to absorb
fixed-parameter error, so the case is stronger than in the predecessor; the
loss-only version costs one evaluation per parameter. **Methodological.**
Proposal: open an entry for a §2.6-style loss-sensitivity pass over the fixed
parameters once the first fit exists, ordered with §4 and §16.

## F32. Structures and characterisations to document, licensed as differences

Collected documentation-only requests whose seats explicitly declined
change proposals; none blocks anything.

- **Intralaminar thalamus** (seat 7, 7.5; within-list consensus across four
  reviews): `thal` stands for the BG-receiving motor/associative thalamus
  (MD, VAp) while also carrying the thalamostriatal projection whose
  FS-targeting part the literature assigns to CM/Pf — state this in
  `model_v07.md` §5; no population is proposed (no ROI could constrain it).
- **STN→striatum** (seat 7, 7.6; one voice, four primary studies): ~17 % of
  macaque STN neurons project to striatum (Sato 2000b, single-axon); absent
  here; negligible for the fit, but it is the one direct route from
  stimulated STN to striatum, so record it where DBS-on conclusions are
  bounded.
- **`snr` is the GPi in every data-facing role** (seat 7, 7.7; one voice, two
  primary studies): in primate the GPi dominates output where rodent SNr
  does (Hardman 2002); the merged population carries rat-derived parameters
  under the other nucleus's name — state it in `model_v07.md` §3.6/§5.
- **Branching, not segregation** (seat 3, 3.8): the measured primate truth
  (≥ 80 % striatofugal branching, Lévesque & Parent 2005) sits between
  BGM_22's segregated pathways and the seat's own admitted-"radical" merged
  MSNs — one sentence so cluster-level findings are not translated into
  cell-type claims.
- **The somatic DBS term** (seats 4, 4.9; 2, 2.9): its sign is one-sided by
  construction (`neg()` clamps; no net somatic excitation can be discovered),
  and its measured character is suppress-and-entrain with no train-driven
  rebound (seat 2's numerical check) — record both beside `DBS.md`'s existing
  sign note, and check a fitted model for STN entrainment at its fitted
  amplitude (folds into F3).
- **`thal → str_fsi`** (seat 7, 7.10): the largest weight in the CSV has
  anatomical motivation — intralaminar input to FSIs is denser in primates
  than rodents (Rudkin & Sadikot 1999; Sidibé & Smith 1999) — record it where
  the weight stands uncommented.
- **Stream-design citations** (seat 7, 7.15; consensus on FS convergence):
  Haber 2016's overlap-decay numbers belong in
  `input_streams/README.md` §4.7 as the citation its no-distance-dependence
  argument currently lacks; `TODO.md` §26 should note the Ramanathan 2002
  evidence is stronger and better replicated than the entry states, and that
  the measured quantity is cross-area convergence, not afferent count.

---

# Proposals round 2 declined under the evidence rule

Recorded so the effect of the round-2 requirements is visible: each of these
is a place where a seat found BGM_22 differing from its own practice, checked
what its practice actually rests on, and withdrew the criticism.

- **Adopt anatomically-scaled population sizes** (seat 1, 1.9): the sizes'
  necessity in a point-neuron model is the lineage's convention; with fixed
  in-degrees and fitted cluster weights the choice largely cancels in the
  contrast. Documentation only.
- **Adopt the explicit dopamine parameter map** (seat 1, 1.10): the lineage's
  own α_dop instrument is admitted approximate in print (Chakravarty 2022 had
  to hand-adjust beyond it), and both fitted conditions share the disease
  state, so a static map is common mode.
- **Match the lineage's striatal microcircuit detail** (seat 1, 1.8): nothing
  read shows morphological detail necessary for a TR-resolution readout —
  and the lineage's own network papers use point neurons and FS-only
  interneuron sets, exactly like BGM_22. The absent SPN→FS connection matches
  Hjorth 2020's own connectivity scheme.
- **Wire the loops together on channel-architecture authority** (seat 2,
  2.3): the diffuse-STN construct is a modeling convention; only the
  anatomy-grounded version survives, as F12's optional projection.
- **Represent Rosenbaum-style synaptic failure in the fit** (seat 2, 2.8):
  second-hand evidence, and the constant-efficacy approximation is defensible
  at TR resolution; survives only as F18's claim boundary.
- **Impose one-spike-per-pulse somatic entrainment** (seat 4, in 4.9): the
  lineage's own rule, justified by citations the seat did not read; it makes
  no claim about the STN soma's true response and records BGM_22's decoupled
  parameterization as the more expressive one.
- **Replace the scalar VTA coverage with a spatial current spread** (seat 6,
  6.1): the lineage's Gaussian lives on a lattice its own paper admits
  corresponds to no anatomy; the subject-specific scalar is better grounded.
- **Adopt graded dopamine values** (seat 6, 6.9): the lineage's medication
  parameters are tuned; only the ordering emphasis survives (F7).
- **Re-weight GPe wiring to the mouse cell-type map** (seat 7, 7.2): the
  targeting data are mouse marker classes whose primate translation both
  source reviews call untested, and the model's third population cannot be
  identified against the scheme. Documentation only (F8).
- **Raise the FS fraction toward primate values** (seat 7, 7.10): Haber gives
  no number; unactionable.
- **Add an intralaminar thalamic population** (seat 7, 7.5): no ROI exists to
  constrain it; free structure without constraint. Documentation only (F32).
- **Adopt either lineage's synaptic time constants** (seat 5, 5.3): neither
  predecessor paper justifies its own constants; the departure needs
  *recording*, not reversal.
- **Move the striatal rate anchors** (seats 3, 7): both audits ended in the
  anchor's favour (F21); the conflicts get recorded, the values stay.

---

# What the panel praised

Places where BGM_22 exceeds the panel's own published practice, each stated
by the seat against its own standard (19 points explicitly in the model's
favour across the seven reviews):

- **The fitting target itself** (seat 5, 5.4; seat 1, 1.7). Conditioning the
  model on the subject's own deconvolved cortical activity and scoring the
  actual 7 × 309-TR trajectory raises the constraint per parameter by orders
  of magnitude over the predecessor's 15-entry FC target with 38 free
  parameters, removes its free synthetic cortex, and converts the fit into a
  within-subject on/off comparison free of the predecessor's two-scanner and
  group-matching confounds.
- **The input-stream contract and cache audit trail** (seats 1, 1.7/1.12; 3,
  3.12). Build-time statistical checks with raising errors, closed-form
  targets, stored audit statistics, refuse-on-mismatch caches — "we would
  adopt this practice, not the reverse" (seat 3), against lineages whose own
  inputs are tuned Poisson processes checked against nothing.
- **Condition discipline for a difference inference** (seat 6, 6.11). Same
  compiled network for both conditions verified byte-identical,
  `assert_dbs_state` guards, seeded search, the RNG-hazard measured and
  recorded — "the claim 'off and on differ only in the DBS parameters' is
  exactly what the inference rests on, and this project has *measured* it."
- **The DBS parameterization** (seats 6, 6.3/6.4/6.5; 4, 4.9/4.10).
  Decoupled somatic and axonal effects (more expressive than the seat's own
  one-spike-per-pulse convention); fitted rather than assumed antidromic
  strength; the subject's own stimulation frequency and imaging-derived
  partial VTA coverage — better grounded than either DBS lineage's own
  spatial constructions.
- **The rate anchors and their audit trail** (seats 2, 2.10; 3, 3.11; 7,
  7.8/7.9). Parkinsonian-primate, state-matched, assumption-quoted — and the
  ordering their weakest assumption produces is independently confirmed by
  identified-cell-type recordings.
- **The cortical-proportions method** (seat 7, 7.12; within-list consensus).
  Anchoring on quantitative macaque tracing and refusing tractography-derived
  strengths is precisely what the anatomy reviews prescribe
  (Emmi 2020's stated methodological position; Haber 2016's practice).
- **The stream design's anatomical choices** (seat 7, 7.15). Independent
  per-region axon pools are anatomically correct (an axon arises in one
  area), and the flat Kincaid fraction is right at 227.6 µm — the topography
  varies at scales two orders larger (Averbeck 2014).
- **The claim-boundary discipline** (seats 2, 2.10; 3, 3.12). A written
  statement of what the model may and may not claim, pre-committed — no seat
  has a published equivalent.
- **Species bookkeeping** (seat 5, 5.14). The model mixes species as its
  predecessors did, but documents each border crossing — "the most that can
  be done short of data that do not exist," with the phase-1 validation pair
  as the composite's acceptance test.
- **The DBS-on cortical drive** (seats 4, 4.4; 6, 6.12). Fitting a human
  *under stimulation* puts DBS's cortical rate effect into the input
  empirically — a structural advantage no seat's synthetic cortex has.

---

# Reading guide for the triage

Not verdicts and not an ordering of `TODO.md` entries — the Roadmap owns
that. What the findings' own content implies about sequencing:

**Fixed before any fit is interpreted (cheap, mostly writing):** F2's
acceptance criterion and cross-checks (into §2), F19's on-stage objective
change, F23's objective bookkeeping, F20's null baselines.

**Before §10's gate calibration:** F1's band derivation document — the
calibration is meaningless against undocumented bands.

**Cache-invalidating if accepted, so settled in Roadmap phase 1 before any
full-length build:** F11 (kernel state refit — MC caches), F13 (STN
proportion table — CI caches), F7 (medication state, if the answer is "on"
— everything), F25's `ci.n_*` derivation (CI caches), F27 (cube size,
already planned in §24). F14's flat-remainder variant would also rebuild
caches if tested.

**Cheap runs on artefacts the project produces anyway:** F3's diagnostics
(probe spikes already recorded), F5's shared-modulation bound (machinery
exists, one run), F10's proxy comparison (one evaluation per mapping), F12's
0.09 bound, F17's coverage profile, F22's population resizing (if that arm
is chosen), F24's candidate refits, F26's delay pair, F28's per-condition
comparison, F29's with/without-STN comparison, F31's sensitivity pass.

**Pure documentation, no computation:** F4, F8, F15, F21, F30 (after one
units check), F32 — and the recording of everything under "What the panel
praised" that belongs in the eventual write-up.

**The standing limitation:** the BOLD pipeline itself remains unreviewed by
any peer on this panel; F10 is the nearest coverage from the model side, and
Meier et al. 2022 remains the citation for precedent.
