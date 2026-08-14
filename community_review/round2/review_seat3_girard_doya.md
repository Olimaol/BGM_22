# Seat 3 — Girard / Doya (ISIR Sorbonne Université / OIST)

Written in character as the Girard–Doya lineage (ISIR Sorbonne Université / OIST),
applying the standards of our own published work: biologically constrained
spiking models of the primate basal ganglia whose parameters are fitted under
quantitative anatomical and electrophysiological constraints. This is round 2 of
the `TODO.md` §34 survey (step 3 rerun 2026-08-14); the panel definition and
reading list are in `../README.md`. It was written independently of the round-1
review, which was not consulted (facts that `model_v07.md` itself attributes to
round-1 documents are treated as project documentation, as the round rules
allow). Per this round's first requirement, every appeal to our own practice
states the experimental evidence our choice rests on — or admits outright that
it is an estimate or a convention — and change proposals are made only where
experimental findings or a self-standing methodological argument license them.
Per the second requirement, every point is structured as an explicit contrast:
what BGM_22 does, what we do, the evidence behind our choice, whether ours would
be better *for this project's goal*, one verdict, and a goal-relevance tag.

**Read for this review (full texts, from the PDFs in this directory's parent):**

- Girard B, Liénard J, Gutierrez CE, Delord B, Doya K (2021). A biologically
  constrained spiking neural network model of the primate basal ganglia with
  overlapping pathways exhibits action selection. *Eur J Neurosci* 53(7):2254–2277.
  DOI 10.1111/ejn.14869.
- Shouno O, Tachibana Y, Nambu A, Doya K (2017). Computational model of recurrent
  subthalamo-pallidal circuit for generation of parkinsonian oscillations.
  *Front Neuroanat* 11:21. DOI 10.3389/fnana.2017.00021.
- Liénard JF, Aubin L, Cos I, Girard B (2024). Estimation of the transmission
  delays in the basal ganglia of the macaque monkey and subsequent predictions
  about oscillatory activity under dopamine depletion. *Eur J Neurosci*
  59(7):1657–1680. DOI 10.1111/ejn.16271.

**What was reviewed:** `model_v07.md` (in full, the primary object), `model_v08.md`
(as the smoke test it declares itself to be), `DBS.md`, `CLAUDE.md`,
`experimental_data/input_streams/README.md`,
`experimental_data/activity_striatum/README.md`,
`experimental_data/cortical_proportions/README.md`,
`BOLD_optimization/get_loss.py` (in full), `BOLD_optimization/parameters.py`,
`TODO.md` (Roadmap and all open entries, §34 skipped per the round rules), and,
in the CompNeuroPy repository, `full_models/bgm_22/model_creation_functions.py`
(the `BGM_v07` docstring and creation sequence) and `parameters.csv` (the
`BGM_v07_p01`/`BGM_v08_p01` columns, parsed programmatically to verify the
projection numbers, weights and delays cited below).

---

## The standard this seat applies

These are the recurring demands of our own papers, each with its evidence basis
stated — including where our own basis is thin.

1. **Every fixed parameter carries a stated, resolvable provenance.** Girard et
   al. 2021 Tables 1–2 reference each row (neuron counts per nucleus from
   Hardman et al. 2002; per-axon bouton counts and branching percentages from
   single-axon tracing, Sato, Parent et al. 2000, Sato, Lavallee et al. 2000,
   Lévesque & Parent 2005; input baselines CSN 2 Hz / PTN 15 Hz / CM/Pf 4 Hz from
   Bauswein et al. 1989, Turner & DeLong 2000, Matsumoto et al. 2001 — all as
   cited in Table 1, footnotes A–O). Liénard et al. 2024 Tables 2–4 list every
   latency datum with its source study. This referencing is an auditability
   convention, not itself an experimental result — but it is what makes a
   parameter-level claim checkable.
2. **Function-agnostic constraint fitting, validated on rest-state physiology.**
   Parameters are optimized to sit inside measured anatomical ranges and to
   reproduce per-nucleus rest firing rates plus fourteen pharmacological
   deactivation experiments (Girard 2021 §2.2, Figure 2); the baselines were
   established "by statistically aggregating activities from 20 different
   macaque monkeys through a literature survey" (§2.2, methodology in Liénard &
   Girard 2014), and updated ranges name their studies and n's (FSI 7.8–14 Hz
   from Adler et al. 2013 n=36/2 monkeys, Yamada et al. 2016 n=42/4, Marche &
   Apicella 2016 n=64/2; MSN floor 0.05 Hz argued from Adler et al. 2013 —
   Girard 2021 §2.4). The function-agnostic stance itself is a methodological
   argument (§4.2), not a measurement.
3. **State-resolved validation beyond mean rates.** Shouno et al. 2017 Table 1
   scores the model against seven statistics per nucleus and state — mean rate,
   fraction of oscillatory cells, 8–15 Hz power (mean and peak), peak frequency,
   percentage of spikes in bursts, bursts per second — recomputed from monkey
   spike data (normal and MPTP-parkinsonian; Tachibana et al. 2011, as recomputed
   in that table), with significance tests, and its Discussion lists what the
   model fails to reproduce (no 3–8 Hz band; burst frequency too low).
4. **Contradictory data are carried, not resolved silently.** The macaque delay
   data are partly inconsistent; Liénard 2024 therefore proposes two delay sets
   (Cmpr and NoPlkv, Table 1), duplicates every simulation over both, and calls
   for experimental replication (§4). Where primate data are absent we refuse to
   model — the GPe is kept whole because "too little information is available
   about the existence of these subpopulations in monkeys" (Liénard 2024 §4) —
   or we import from rodent with an explicit flag and a sensitivity test: the
   redundancy ρ=3 "is arbitrary … it derives from a rat study dealing with
   MSN-to-MSN projections only" (Koos et al. 2004, as used in Girard 2021 §2.3),
   and was re-tested at 2.75 and 3.25 (§4.1).
5. **Fitted-parameter degeneracy is expected and propagated.** Our constraint
   optimization returned more than 1000 parameterizations "equally maximising
   the plausibility scores", reducible to 15 base solutions (Liénard 2024 §5.4);
   all fifteen are carried through every subsequent analysis (Girard 2021 §2.5,
   §3), and the under-constrained tonic inputs are set at hypersphere centres
   precisely so that no result hangs on "a single brittle configuration"
   (Girard 2021 §2.5, §4.1). This is a documented methodological experience of
   our own work, not an experimental datum, and is presented as such below.
6. **Admitted conveniences of our own.** Our neurons are LIF (Girard 2021) or
   single-compartment conductance models (Shouno 2017); our inputs are
   homogeneous Poisson generators at one shared rate per population; Girard 2021
   §3.1 concedes the consequences in print — firing-rate distributions narrower
   than experimental, CVs "lower than the experimental ones and also much less
   widespread", the models "fire much more regularly and are also much more
   homogeneous than the real neural substrate". Our own Introduction warns that
   "the mix of experimental data obtained in different species (rodents and
   primates, usually)" can make a detailed model's veracity questionable — a
   warning our own synaptic parameters (rat STP data in Shouno 2017) do not fully
   escape. None of these conveniences is held against BGM_22 below.

---

## Points

### 3.1 The BG firing-rate bands: unresolvable provenance, unstated state, and a parkinsonian-direction problem

**What BGM_22 does:** `get_loss.get_firing_rate_loss` scores all 18 populations
(9 per loop) against `plausible_ranges`. The three striatal bands are derived and
documented to our standard (see 3.11). The six BG bands are not: `gpe_proto`
(75, 85), `gpe_arky` (15, 20), `gpe_cp` (75, 85) and `thal` (15, 30) carry no
citation at all, and `stn` (28, 80) and `snr` (21, 93) are cited as
"`[Li et al., 2015]`" — a bracket citation with no journal, no DOI, no local PDF,
occurring exactly once in the repository (`get_loss.py`). This term is half of
the total loss and the whole of the gate.

**What we do:** per-nucleus rest baselines aggregated from ~20 macaques with the
aggregation methodology published (Girard 2021 §2.2, Figure 2; Liénard & Girard
2014), updated ranges naming studies and cell counts (§2.4); for the
parkinsonian state, monkey values recomputed from spike data with n's and
significance tests (Shouno 2017 Table 1(A): STN 19.9 ± 9.5 Hz normal /
27.6 ± 11.3 Hz parkinsonian; GPe 65.1 ± 25.6 / 41.1 ± 22.3; from Tachibana et
al. 2011).

**Evidence behind our choice:** measured, species- and state-specific quantities
with resolvable sources. The choice of band *width* (confidence-interval-based
in our case) is a convention; the demand that the source be findable is not.

**Why that would (or would not) be better here:** three concrete problems
follow from the current bands. (a) *Auditability*: an unresolvable citation on
half the loss cannot be checked by anyone, including the project's own future
self — the striatal side of the very same function shows the project knows how
to do this properly. (b) *Width*: the GPe bands are ±5 Hz around 80 Hz, far
narrower than the between-neuron spread of any monkey GPe sample we know
(±25.6 Hz SD in the normal state, Shouno Table 1(A)) and narrower than the
striatal bands' own ±1 SD rule — no width rule is stated. (c) *State*: the
patient is PD, off medication; in MPTP monkeys the GPe rate *falls* (65 → 41 Hz,
Table 1(A)) while the model is required to hold its prototypic and
cortex-projecting GPe at 75–85 Hz — at or above the top of the *normal*-monkey
plausible band our own model was held to (Girard 2021 Figure 2) and roughly
double the parkinsonian monkey mean. The striatal anchor was deliberately moved
to the medication-off state (Liang et al. 2008); the pallidal bands may be
pulling the same model toward a healthy or even hyperactive state. If
"Li et al., 2015" is a human intraoperative dataset the bands may be defensible —
but then it must be named, and its state (on-stim? off-med? anesthesia?) stated
per band. `TODO.md` §10 covers the gate *threshold* and the missing on-condition
variant, but not the provenance or the state of the band contents; that gap is
not carried by any entry we found. **Experimentally grounded deficiency.**
Proposal: a derivation document for the six BG bands in the style of
`experimental_data/activity_striatum/README.md` — resolvable source, species,
dopamine state, and width rule per band; for the parkinsonian state the
Tachibana et al. 2011 monkey values (as recomputed in Shouno 2017 Table 1(A))
are the nearest constrained set we can point to, with the species caveat stated.

**Goal relevance:** high — these bands are half the loss, the entire gate, and
the only physiological anchor of the non-striatal model.

### 3.2 Mean rates cannot certify the parkinsonian regime; the statistics that can are already recorded

**What BGM_22 does:** the only physiological term is the mean-rate loss computed
from the 9.9 s probe (`Spikes10s` / `get_firing_rate_10s`). The probe's
`CompNeuroMonitors` record full spike trains from every population, but no burst
or spectral statistic is computed anywhere; the loss JSON stores means only
(`firing_rates_hz`).

**What we do:** Shouno 2017 validates against seven statistics per nucleus and
state (Table 1), and its Figure 3 shows *why*: the parameter regions whose mean
rates are compatible with the normal state and with the parkinsonian state
overlap (the slate-gray patches of that figure) — mean rate alone
under-determines which regime the circuit is in, while the states separate
cleanly on oscillation and burst measures (STN oscillatory cells 5.5% normal vs
36.3% parkinsonian; bursts/s 1.68 vs 2.73). Girard 2021 likewise reports
distributions and CVs alongside means (§3.1, Figure 4) and uses 10-s simulations
for spectra (§2.6) — the same duration as BGM_22's probe.

**Evidence behind our choice:** monkey recordings in both states, recomputed
with stated n's and Mann-Whitney tests (Tachibana et al. 2011, in Shouno
Table 1(A)); burst detection by the Poisson-surprise method (Wichmann & Soares
2006, as cited in Shouno's Data Analysis). For humans, PD-patient STN
single-unit burst oscillations in the 10–25 Hz band are documented (Levy et al.
2001, 2002, as cited in Shouno's Introduction).

**Why that would (or would not) be better here:** the inference "read off what
DBS did" presupposes that the DBS-off fit sits in a parkinsonian dynamical
regime, since that is the state DBS acts on; a rate-only gate cannot certify
this — our own Figure 3 exhibits rate-matched parameter regions on both sides
of the normal/parkinsonian divide. BGM_22 already records every spike it would
need; computing per-population burst fractions and low-β band power from the
existing probe and writing them into the loss JSON costs nearly nothing and
changes no fit. We are *not* proposing to fit these statistics — our own model
missed some of them (Shouno's Discussion admits no 3–8 Hz band and too-low burst
frequency), and BGM_22's open-loop striatum limits which of them are meaningful
there (`input_streams/README.md` §4.3). We propose to *report* them for the
accepted fits and compare against the stated monkey references before any claim
about the off state is written. **Experimentally grounded deficiency.**

**Goal relevance:** high — it is the difference between "the off-fit reproduces
the subject's BOLD" and "the off-fit is a parkinsonian basal ganglia that
reproduces the subject's BOLD".

### 3.3 One argmin per condition, against the documented degeneracy of constrained BG fits

**What BGM_22 does:** the Roadmap runs one DBS-off fit and one staged DBS-on fit
(`TODO.md` §32: CMA-ES, seed fixed at 42 during fitting, the winning vector
re-run across ~10 seeds), and defers the inference design to §2, which already
names near-degenerate pairs (`axon_spikes_per_pulse` vs the `stn__gpe`/`stn__snr`
scalings; `passing_fibres_strength` vs `snr__thal`) and costs three remedies
(single-mechanism scans, multi-start, L1).

**What we do:** we never interpret a single optimum. Our constraint fit returned
>1000 parameterizations "equally maximising the plausibility scores", reducible
to 15 base solutions (Liénard 2024 §5.4); all fifteen were translated and
analysed in Girard 2021 (§2.5, §3 — including finding that two of them behave
unacceptably and excluding them), and Liénard 2024 additionally duplicated every
simulation over two delay sets. Under-constrained parameters (the tonic inputs)
were fixed at the centre of the maximal hypersphere inside the plausible region,
explicitly to avoid "a single brittle configuration" (Girard 2021 §2.5, §4.1).

**Evidence behind our choice:** this is a documented methodological experience
of our own optimization work, not an experimental measurement — we say so. The
argument that carries the proposal stands on its own: identifiability. Two fits
each returning one vector cannot distinguish "DBS moved parameter k" from "the
two runs landed on different members of the same near-optimal set", and BGM_22's
loss surface (19–22 parameters against one subject's 310-TR correlation plus
rate bands) has no stated reason to be better conditioned than our 50-parameter
fit against ~100 constraints was.

**Why that would (or would not) be better here:** the project's central claim
*is* a difference between two fits, so the within-condition solution spread is
the null distribution of that claim. `TODO.md` §2 is the right entry and its
option list is the right family; the caudate free control (§2's update, and
`split_param_list`'s staged layout) is a genuinely strong design element we have
no equivalent of, and fixing the seed during fitting with a multi-seed re-run of
the winner is sound practice. Two gaps remain: (a) §2 designs the *on*-side
inference, but the *off*-fit's non-uniqueness enters the difference identically —
the off solution set needs the same characterization; (b) the ordering (fits
first, §2 after) is safe only as long as the first fit pair is treated as
pipeline-proving and never quoted as inference. **Methodological deficiency**
(of the currently scheduled procedure; the planned direction is right).
Proposal: extend §2 to both conditions — multi-start (or retained CMA-ES
populations plus independent restarts), report the per-parameter spread across
near-optimal solutions per condition, and claim a DBS-induced change only where
the off–on difference exceeds that spread; settle this in §2 before §32's
results are interpreted, not after.

**Goal relevance:** high — this is the machinery of the central claim itself.

### 3.4 The hyperdirect pathway cannot carry DBS, but in patients it demonstrably does

**What BGM_22 does:** the cortical afferents to STN are `TimedArray` →
`CurrentInjection` streams, structurally excluded from the DBS footprint, so
"afferent DBS in this model means `gpe_proto→stn` only" (`DBS.md` Known
limitations 2; `TODO.md` §15). The three fitted DBS parameters span somatic
hyperpolarization, STN-efferent/pallidal-afferent axon spikes, and one passing
fibre (`snr__thal`).

**What we do:** Shouno 2017 treats the cortico-subthalamic input as a principal
handle on the pathological state: tonic increases of the cortical drive suppress
the 8–15 Hz oscillations (via de-inactivation of the T-current rebound
mechanism), and oscillatory cortical input amplifies or suppresses them
phase-dependently (Results, "Roles of Cortical Excitatory Inputs", Figure 7);
the Discussion links STN-DBS therapy to "high-frequency, direct stimulation of
cortical afferents to the STN" (citing Miocinovic et al. 2006).

**Evidence behind our choice:** the suppression result is our simulation,
consistent with patient data as cited in Shouno (STN oscillations reduced during
voluntary movement: Amirnovin et al. 2004; Brown & Williams 2005). That STN DBS
activates the cortico-subthalamic pathway in humans is experimental: DBS-evoked
antidromic responses recorded with ECoG over prefrontal cortex at 6 ms latency
(Chen et al. 2020, as cited in Liénard 2024 §4).

**Why that would (or would not) be better here:** with this channel structurally
absent, whatever the patient's hyperdirect activation contributes to the DBS-on
BOLD is forced into the pallidal-afferent, efferent and passing-fibre
parameters — §15 already states that "a fitted 'afferent' effect here is a
pallidal one". For an inference whose product is a parameter attribution, a
missing candidate mechanism is not a neutral simplification; it biases the
attribution toward the mechanisms that are representable. Reviewing §15's
adequacy rather than rediscovering it: the entry's remedy ("giving the cortical
drive a spiking soma") is heavier than needed. The count→current conversion
passes through Python every 110 ms chunk (`Microcircuit.update()` /
`CorticalInputs.update()`), and the 125 Hz pulse times are deterministic
(`pulse(t)` is 1 for one timestep every 8 ms, `DBS.md`); an additive,
pulse-locked spike-count component on the STN cortical stream, its amplitude a
fourth fitted DBS parameter, would represent orthodromic hyperdirect activation
without touching caches, neuron models or the compiled network. Antidromic
cortical effects (and their cortical BOLD consequences) would remain absent
either way and belong in the written claim boundary. **Experimentally grounded
deficiency.**

**Goal relevance:** high — it bounds what "what DBS did" can mean.

### 3.5 Static synapses under 125 Hz stimulation, against measured pallido-subthalamic depression

**What BGM_22 does:** no synapse in either model version carries short-term
plasticity — the streams by design (`input_streams/README.md` §4.4), the 28 BGM
projections as plain static-weight projections (`model_v07.md` §5), and the
DBS-evoked transmission uses the same static `w`
(`pre_axon_spike = g_target += ite(unif < p_axon_spike_trans, w*post.dbs_on, 0)`,
`DBS.md`).

**What we do:** Shouno 2017 modelled GPe→STN short-term depression with
parameters "manually tuned to fit experimental data" and overlays model against
experiment at 1–100 Hz (Figure 2, experimental points from Atherton et al.
2013), plus STN→GPe facilitation/depression after Hanson & Jaeger 2002; the
depression is mechanistically load-bearing in that model (it is what lets high
GPe rates shut the 8–15 Hz bursts off).

**Evidence behind our choice:** measured — at 100 Hz sustained presynaptic
firing the unitary GPe→STN transmission probability collapses to below ~0.2
within seconds and recovers over tens of seconds (Shouno Figure 2, experimental
data of Atherton et al. 2013; the species of those slice experiments is not
stated in the text we read, and we flag that). Our equal-probability assignment
of three STP types is a convention on top of the measured heterogeneity.

**Why that would (or would not) be better here:** under continuous 125 Hz DBS
the model's pallido-subthalamic and subthalamo-pallidal synapses operate exactly
where depression saturates. For the *fit* this is largely absorbable: a run-long
DBS-on state is a steady state, and the fitted `axon_spikes_per_pulse` and the
putamen `gpe_proto__stn` scaling can stand in for the depressed efficacies. What
is not absorbable is the *reading*: an on-refit that lowers `gpe_proto__stn`
would be reported as "DBS weakened pallido-subthalamic transmission", when the
physiological description would be frequency-dependent depression that the model
cannot express — a mechanism-level conflation in precisely the quantity the
project wants to report. **Experimentally grounded deficiency** (of
interpretation more than of fit quality). Proposal: at minimum a claim-boundary
sentence wherever fitted weight changes under DBS are interpreted; optionally,
short-term depression on the six DBS-footprint projections, which ANNarchy
synapse models can carry.

**Goal relevance:** medium — it does not block the fit, but it sits directly on
the meaning of the fitted DBS-on parameters.

### 3.6 Subcortical delays: a rat set, a third shorter than the macaque estimates in the loop that sets the oscillation band

**What BGM_22 does:** the seven distinct subcortical delays (`parameters.csv`,
verified: str→snr 4, str→gpe 5, stn→snr 1.5, stn→gpe 2, gpe→stn 4, gpe→snr 3,
snr→thal 5 ms) are the Kumaravelu et al. 2016 rat set, as `model_v07.md` §5 now
records with caveats; the cortical drive arrives with no delay at all
(`CurrentInjection`, §7.7) — which for a per-TR-constant rate with
independently drawn bins is genuinely inconsequential, since there are no
millisecond-scale cortical events whose ordering a delay could change.

**What we do:** Liénard 2024 estimated macaque delays from ten stimulation
studies (every latency with its source, Tables 2–4), searched all 12⁸ ≈ 430
million combinations (§5.3), found the data partly contradictory, and carried
two sets through duplicated simulations (Table 1: Cmpr Str→GPe 8, Str→GPi 11,
STN→GPe 9, GPe→STN 1; NoPlkv 6, 8, 2, 7). Figure 9 shows the STN↔GPe delay sum
sets the oscillation frequency of that loop.

**Evidence behind our choice:** measured stimulation latencies in macaque, with
the method's crudeness admitted in print (§4: summing delays "does not honour
the real complexity" of the dynamics) and the human evidence (Oswal et al. 2016,
2021; Chen et al. 2020, as cited in §4) only partially concordant. We cannot
claim "the" macaque delays exist; we claim two constrained candidates do, and
that both differ from the rat set in the same direction on the striatofugal
pathways (BGM_22's 4–5 ms vs 6–11 ms).

**Why that would (or would not) be better here:** BGM_22's STN↔GPe loop delay
sums to 6 ms against our 9–10 ms; per Figure 9 that resonates near 40–45 Hz
rather than the 32–35 Hz high-β both our sets give (indicative only — the PSP
kinetics differ between the models). At a 2.31 s TR the correlation loss barely
sees this, so we do not claim the fit is compromised; but the regime diagnostics
of 3.2 and any β-band language in the eventual interpretation would inherit a
loop tuned to the wrong band, in a model of a human patient where rat, not
primate, timing was used. The rat values are measured quantities too — the
deficiency is species, not fabrication. **Experimentally grounded deficiency**
(moderate). Proposal: a sensitivity pair in our own style — re-evaluate the
accepted fits under both macaque sets (a delay change touches `parameters.csv`
and a recompile, not the input caches) and report whether the loss, the fitted
parameters, or the 3.2 diagnostics move; adopt or keep with the result recorded.

**Goal relevance:** medium — low for the BOLD loss itself, medium for the
regime and for any oscillation-level interpretation of the DBS effect.

### 3.7 A uniform in-degree of 10, against convergences measured at three orders more on the striatofugal pathways

**What BGM_22 does:** all 28 BGM projections use
`connect_fixed_number_pre(number=10)` (`parameters.csv`, verified on
`str_d1__snr`, `gpe_proto__stn`, `gpe_arky__str_d1` and others; `model_v07.md`
§5), and the cortical afferent counts of the BG populations are flat round
numbers flagged in-code as "TODO use lit motivated values" (`parameters.py`,
`ci.n_*`; `TODO.md` §8).

**What we do:** in-degrees are derived, not chosen: ν = P·n_pre·α/n_post
(Girard 2021 Eq. 4) from per-axon bouton counts and branching percentages
(Tables 1–2; single-axon tracing, refs M/N/O) and per-nucleus neuron counts
(Hardman et al. 2002). For the striatopallidal pathway that gives, per GPe
neuron, α_MSN→GPe = 171–203 boutons per axon × 10,576 MSN / 100 GPe ≈ 2·10⁴
synapses, i.e. ~6–7·10³ distinct MSN afferents at redundancy ρ=3 — three orders
above BGM_22's 10. For the STN↔GPe loop the same arithmetic gives ~33 (STN→GPe)
and ~20 (GPe→STN) distinct afferents — the same order as BGM_22's 10, and as
the sparse connectivity Shouno 2017 grounded in Baufreton et al. 2009 (12 GPe
per STN, 6 STN per GPe).

**Evidence behind our choice:** the bouton counts and percentages are measured
(macaque single-axon tracing); our ρ=3 is an admitted rat generalization with a
published sensitivity check (Girard 2021 §4.1). So the STN↔GPe part of BGM_22's
choice is defensible on our own numbers; the striatofugal part is not.

**Why that would (or would not) be better here:** the fitted cluster weight
absorbs the product ν·w, so mean rates are insensitive to the in-degree. What
the in-degree does set is input *sharing*: two `snr` neurons drawing 10 of 486
dSPN afferents share ~0.2 of them (input correlation ≈ 0.02), where the measured
convergence implies mostly-shared striatal input — and BGM_22's own
`input_streams/README.md` §3 argues that exactly such correlations govern the
pooled-current variance that is the simulated BOLD. To be precise about the
consequence: the subject-locked slow component transfers through the rates
regardless of sharing, so the fit can succeed either way; what the sparse
in-degree changes is the noise structure of the simulated pallidal/nigral BOLD
(likely cleaner than a real GPi's) and the pallidal firing statistics — large
unitary IPSPs from few afferents — which are the very statistics 3.2 asks to be
diagnosed. **Experimentally grounded deficiency** (bounded consequence).
Proposal: raise `number` toward the anatomical proportions on the striatofugal
projections (cheap at 100-neuron targets) or run and record a sensitivity check;
give the `ci.n_*` counts their planned literature derivation.

**Goal relevance:** medium — invisible to the mean-rate fit, but it shapes the
regime statistics and the noise floor of the model's only output.

### 3.8 Pathway segregation, circuit scope, and what the cluster labels may claim

**What BGM_22 does:** the striatum is split into dSPN/iSPN populations with
segregated projections — `str_d1__snr` plus a weak `str_d1__gpe_cp` (weights
0.06 / 0.005), no D1 projection to `gpe_proto` or `gpe_arky` — and the inference
handles are the clusters `str_d1__bg` / `str_d2__bg`
(`get_loss.PROJ_CLUSTERS_COMMON`). The circuit has no CM/Pf-like diffuse
thalamic input to the BG nuclei; it does have cortico-pallidal drive to
`gpe_arky`/`gpe_cp` and a rich pallido-striatal feedback (`gpe_striatum`
cluster).

**What we do:** a single MSN population whose axons branch — the defining
feature of our model line ("overlapping pathways", Girard 2021 title and §1),
with MSN→GPi carried by 82% of striatofugal neurons (Table 1 ref M, Lévesque &
Parent 2005) and the direct/indirect overlap documented in monkeys (Parent,
Charara & Pinault 1995, as cited in Girard 2021 §4). We include CM/Pf as an
external input to all BG nuclei with measured bouton counts and a 4 Hz baseline
(Tables 1–2; Matsumoto et al. 2001) and predict it sets the circuit's overall
excitability (Girard 2021 §3.2, Figure 9). We have no cortico-pallidal
projection.

**Evidence behind our choice:** the axonal branching is measured (macaque
single-axon tracing, as above). But our *functional* merging of D1 and D2 MSNs
is admitted in print to be "a relatively radical modelling choice" resting on an
average-cancellation assumption (Liénard 2024 §5.5, with Goldberg et al. 2002
cited there for unchanged striatal rates after MPTP); and CM/Pf's functional
importance is our model's prediction, not a measurement. D1/D2-resolved
collateral strengths in primates are measured in neither lineage's sources.

**Why that would (or would not) be better here:** BGM_22's segregation is the
opposite simplification to ours, and we cannot propose our convention without
violating this round's first requirement — the honest statement is that the
measured truth (pervasive branching) sits between the two models. The practical
consequence for BGM_22 is semantic: a change the fit assigns to `str_d1__bg`
cannot be read as a cell-type-pure "direct pathway" change, and the cluster
mechanism (which scales `str_d1__snr` and `str_d1__gpe_cp` together) already
prevents the fit from separating the branches anyway. The CM/Pf absence is, for
a resting-state fit, largely degenerate with the fitted baseline currents and
drive weights. **Difference, not deficiency.** Suggested documentation only: one
sentence in the eventual write-up that the pathway labels are model constructs
whose primate anatomical counterpart is branching rather than segregation, so
that cluster-level findings are not translated into cell-type claims.

**Goal relevance:** medium — it does not change the fit, but it constrains how
the headline result may be phrased.

### 3.9 Parameter provenance inside `parameters.csv` is partial where the project's own standard is exemplary

**What BGM_22 does:** the CSV's section headers carry provenance for some
compartments — GPe-proto/arky "refitted, data from Bogacz et al. 2016" (earlier
columns "optimized fit of Abdi, Mallet et al.") — and none for others: the STN,
SNr and thalamus sections have empty comment cells, and the thalamus parameters
(a, b, c, d = 0.02, 0.2, −65, 6 with a hand-set `I_app = 5`) are the generic
Izhikevich-2003 textbook values. The model lineage is anchored only in the
`BGM_v07` docstring ("difference to Goenner et al. (2021): …"), and the delay
provenance was recovered only by last round's review (`model_v07.md` §5).

**What we do:** a reference per parameter row (Girard 2021 Tables 1–2,
footnotes A–O; Liénard 2024 Tables 2–4), and an explicit label where a value is
a modelling artifact rather than a measurement (the tonic inputs V_C, the
redundancy ρ).

**Evidence behind our choice:** none experimental — this is an auditability
convention. The argument that carries a proposal is methodological and
self-standing: the project's end product is a statement about parameters
(which cluster scalings and DBS strengths had to change), and a fitted scaling
on a base weight of unstated origin transmits no interpretable meaning. The
project's own `experimental_data/` READMEs are the counterexample to its own
CSV: it demonstrably knows the format.

**Why that would (or would not) be better here:** the inference is only as
readable as the base values it scales. **Methodological deficiency.** Proposal:
one provenance line per CSV section (or a companion table in `model_v07.md` §6),
naming for each neuron model and each base weight the paper or fit it came from —
including resolving "Bogacz et al. 2016" and "[Li et al., 2015]" to citable
form, and recording the Goenner et al. 2021 inheritance of the weight table in
the file that carries the weights.

**Goal relevance:** medium — no simulation changes, but the central claim's
legibility does.

### 3.10 The shared-ROI BOLD monitors mix the two loops 50:50 by construction

**What BGM_22 does:** the GPi, GPe and STN monitors pool the caudate-loop and
putamen-loop populations; with equal population sizes (100 each) and no
cross-loop scale factors, the two loops enter each shared ROI at equal weight
(`get_loss.py` `bold_region_compartments`; `model_v07.md` §3.6 documents the
normalization mechanics but not this implied anatomical weighting). The
experimental ROI is the whole nucleus, whose associative (caudate-associated)
and sensorimotor (putamen-associated) territories are not equal in general.

**What we do:** we have no BOLD practice to hold against this. Our nearest
quantity — per-nucleus neuron counts (Hardman et al. 2002, as used in Girard
2021 Table 1) — does not resolve functional territories, so we cannot supply
the correct weighting either.

**Evidence behind our choice:** none; stated as such.

**Why that would (or would not) be better here:** the shared-ROI signals are
where the DBS-carrying putamen loop and the control caudate loop are compared
against one experimental series; if the real GPi BOLD is dominated by one
territory, the simulated mix mis-weights the DBS-carrying component. Since no
measurement we can cite replaces the 50:50, no change proposal is licensed.
**Difference, not deficiency.** Suggested documentation only: state the implied
50:50 territorial weighting explicitly in `model_v07.md` §3.6 beside the GPe
abundance discussion, so the triage and the eventual write-up see it as an
assumption rather than an accident of equal population sizes.

**Goal relevance:** medium — it sits inside the loss of three of the seven
fitted ROIs.

### 3.11 In BGM_22's favour: the striatal rate anchor meets our standard and is better state-matched than our own baselines

**What BGM_22 does:** `experimental_data/activity_striatum/README.md` derives
dSPN 25.0 / iSPN 33.0 Hz verbatim from Liang et al. 2008 Table 1 (chronic MPTP
monkeys, levodopa withdrawn — matching the patient's medication state), quotes
the paper's own admission that response-direction-equals-receptor-class is an
inference, reconstructs the SDs, documents a rejected deconvolution with
reasons and a sensitivity bound, and builds the FS rate from three
normal-primate datasets (n-weighted) times a rodent-supported chronic-depletion
factor of 1.0, with the caveats ranked.

**What we do:** the same method family — literature aggregation with stated
provenance and n's (Girard 2021 §2.2, §2.4) — but for the *healthy* rest state.
Our FSI range [7.8–14 Hz] came from an overlapping study set (Adler et al. 2013,
Yamada et al. 2016, Marche & Apicella 2016); BGM_22's FS centre of 10.5 Hz falls
inside it — an independent convergence worth noting. The width philosophies
differ (our CI-style aggregation vs their ±1 SD across neurons), which their
README itself flags ("a plausibility band, not a confidence interval").

**Evidence behind our choice:** the same kind of measured aggregates; our
healthy-state choice is a property of our scientific questions, not superior
evidence.

**Why that would (or would not) be better here:** it would not — for a patient
off medication, BGM_22's state-matched anchor is the more defensible choice, and
the derivation discipline equals or exceeds ours. One tension should be
recorded: our lineage carries the opposing datum that striatal mean rates do
not change after MPTP (Goldberg et al. 2002, as cited in Liénard 2024 §4.1 —
we did not read Goldberg itself), while Shouno 2017 cites Liang et al. 2008
approvingly for elevated striatal inhibition; the chronic-vs-acute
reconciliation the README sketches ("contrasting it with acute depletion") is
plausible but currently argued without naming the opposing measurement.
**Difference, not deficiency** — in BGM_22's favour. Suggested documentation
only: add the Goldberg 2002 counter-datum, and the chronic-denervation argument
against it, to the README's "Why these rates are so high".

**Goal relevance:** high — these values are baked into every v07 cache and
centre the striatal bands.

### 3.12 In BGM_22's favour: the input-stream contract and cache audit trail exceed our published validation practice

**What BGM_22 does:** every synthetic input stream is checked at build time
against closed-form targets (mean, Fano factor, pairwise correlation —
`input_streams/README.md` §2; `model_v07.md` §7.4 step 4), raising on mismatch,
with the measured statistics stored in the cache state so any cache can be
audited without regeneration; the missing-GABA geometry is additionally checked
against the analytic f(d); the README states in one place what the model may
and may not be used to claim (§5: input gains and drive amplitudes yes, emergent
striatal circuit dynamics no) and even discloses which cited sources were read
only as abstracts. The pending input-correlation values are not guessed but
scheduled for a scan against the model's own output correlation and BOLD
amplitude (`TODO.md` §25), correctly placed as a hard blocker before the fits.

**What we do:** our validation operates at the level of network outputs — rest
rates, deactivation tests, state statistics (Girard 2021 Figure 2; Shouno 2017
Table 1) — and we never verified our input generators against distributional
targets in print; our inputs are homogeneous Poisson processes whose unrealism
we concede after the fact (Girard 2021 §3.1). We have no equivalent of the
claim-boundary statement, and no plan equivalent to §25 — our input
correlations are simply zero by construction.

**Evidence behind our choice:** none to defend — the comparison is between
documented practices, and the four-year silent Fano-factor failure that
motivated BGM_22's contract (README preamble) is exactly the error class that
output-level validation like ours does not catch.

**Why that would (or would not) be better here:** it is better, full stop; we
would adopt this practice, not the reverse. **Difference, not deficiency** — in
BGM_22's favour. No proposal.

**Goal relevance:** medium — it does not make the fit succeed, but it protects
everything the fit rests on, and the §25 scan it feeds is a stated blocker on
the fits.

---

## What we would ask for before publication

Only demands licensed by the deficiency verdicts above; the documentation
suggestions from difference verdicts are gathered in item 8.

1. **Band provenance (3.1):** a derivation document for the six BG firing-rate
   bands — resolvable citation, species, dopamine state and width rule per
   band — in the style the project already uses for the striatal ones; resolve
   or replace "[Li et al., 2015]".
2. **Regime diagnostics (3.2):** per-population burst fractions and low-β power
   computed from the existing probe recordings, reported in the loss JSON for
   the accepted fits and compared, with species caveats, against the
   parkinsonian-monkey reference statistics.
3. **Solution sets (3.3):** multi-start characterization of both the off- and
   on-fits, per-parameter spreads reported, and DBS-induced changes claimed only
   where the off–on difference exceeds the within-condition spread; folded into
   `TODO.md` §2 before any fit is interpreted.
4. **Hyperdirect DBS (3.4):** either a pulse-locked component on the STN
   cortical stream as a fourth fitted DBS parameter, or an explicit claim
   boundary stating that the fitted DBS parameters exclude cortico-subthalamic
   activation, antidromic cortical effects included.
5. **STP conflation (3.5):** a claim-boundary sentence wherever DBS-on weight
   changes are interpreted mechanistically (optionally: short-term depression on
   the six DBS-footprint projections).
6. **Delay sensitivity (3.6):** the accepted fits re-evaluated under both
   macaque delay sets (Cmpr and NoPlkv), with the effect on loss, parameters and
   diagnostics recorded.
7. **Convergence (3.7, 3.9):** a sensitivity check or literature-derived
   in-degrees for the striatofugal projections and the `ci.n_*` counts; one
   provenance line per `parameters.csv` section.
8. **Documentation from difference verdicts (3.8, 3.10, 3.11):** the
   branching-vs-segregation caveat on cluster-level claims; the implicit 50:50
   territorial weighting of the shared-ROI monitors; the Goldberg 2002
   counter-datum beside the Liang rates.
