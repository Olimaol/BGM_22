# Synthesis of the community-conventions review

Merges the seven persona reviews in this directory into one ranked list of
findings, each with a concrete change proposal. Defined by `TODO.md` §34
step 3; the panel, the selection rationale and the reading list are in
`README.md`.

**What this document is not.** It contains no verdicts. `TODO.md` §34 is
explicit that implementing accepted changes is not part of the survey — each
accepted proposal spawns its own numbered `TODO.md` entry, and the triage that
decides which are accepted is the Roadmap item that follows §34. The proposals
below are what the panel would change; the decision is Oliver's.

## How the ranking works

Per §34: convergence across personas is the signal. A point raised by several
independent lineages is a **community convention**; a point raised by one is
**one lab's taste**, which may still be right but carries less weight.

Per the §34 amendment for seat 7 (the experimentalist), a point raised only by
that seat is weighted by convergence *within its own reading list*: asserted by
multiple independent reviews → literature consensus, comparable to
multi-persona convergence; found in a single review → one voice. Every seat-7
finding below states which case applies, and every structural claim in that
review carries a species-provenance tag.

Findings are numbered **F1…F24** and grouped in four tiers by convergence.
Within a tier they are ordered by how much they bear on the project's stated
goal. Each carries the seat points it merges, so any finding can be read back
to the reviews it came from.

## The one limitation this panel cannot address

No BOLD or whole-brain lineage has a seat, by design — such groups are
mean-field modellers and fail §34's approach-similarity criterion. **The
project's BOLD pipeline therefore has no peer reviewer on this panel.** Nobody
audited the balloon model, the `normalize_input=2000` baseline, the SPM-HRF
deconvolution that produces the cortical drive, or the fit-to-fMRI methodology
itself.

That limitation was originally partly mitigated: Meier et al. 2022
(`10.1016/j.expneurol.2022.114111`), a TVB co-simulation embedding the
Hamker-lineage spiking BG model in a whole-brain model, sat in seat 5's reading
list and would have provided a partial in-panel audit of the simulated-BOLD and
DBS side. It was removed at the download checkpoint (rationale in `README.md`),
so **the limitation now holds without mitigation.** Meier et al. 2022 remains
the nearest published precedent for this project's BOLD pipeline and should be
cited as such in any write-up.

Seat 5 was asked to say what it could from inside the lineage and could say only
this (its point 5.9): the balloon parameterisation is Maith et al. 2021's
(Friston 2000, 2003), it was used there with a *different* input variable, and
this lab has never validated simulated BOLD against a measured BOLD **time
course** — only against correlation matrices. BGM_22 is doing something the
lineage has not done before, with a pipeline calibrated for what it did do
before. That is where the unreviewed risk sits, and F11 is the one finding that
touches it from the model side.

---

# Tier 1 — raised by five or more seats

These are the panel's consensus. All four bear on whether the project's central
claim — *which parameters had to change under DBS* — can be made at all.

## F1. There is no way to tell a fitted parameter change from optimiser noise

**Seats: 1 (1.10), 2 (2.6), 3 (3.5), 4 (4.9), 5 (5.1, 5.3, 5.4), 6 (6.3) — six
of seven.** The strongest convergence in the review, and every seat reached it
from its own practice rather than from a shared source.

The panel's separate framings are worth keeping distinct, because they call for
different remedies:

- **Degeneracy** (seat 3). Girard et al. 2021 carries *fifteen* parameterisations
  that all satisfy the same anatomical and physiological constraints, and where a
  single point is needed it takes the centre of the largest hypersphere inside
  the plausible region — deliberately the most robust point, not the
  best-scoring one. At 19 parameters against one scalar loss, the DBS-off
  solution set is a manifold; the difference between two arbitrary points on two
  manifolds is not an inference.
- **Identifiability** (seat 2). The drive weights (parameters 0–6) and the weight
  cluster scalings (9–18) are not obviously separable: in a striatum whose input
  is 98 % streamed, a change in a `gpe_striatum` scaling and a compensating
  change in the dSPN drive weight produce nearly the same output.
- **Regression against the lab's own published method** (seat 5). Maith et al.
  2021 ran **twenty** optimisation processes per fitted unit with different
  random initialisations and selected the lowest loss; it also ran a sensitivity
  analysis over 59 parameters it had *not* fitted and found the loss dominated
  by cortical and thalamic parameters that were held fixed. BGM_22 runs one
  optimisation and no sensitivity analysis.
- **Sweep rather than fit** (seats 4, 6). For the DBS parameters specifically,
  the community norm is a reported table across the parameter's range
  (Mandali & Chakravarthy 2016 sweeps antidromic strength at 10/50/75 %;
  Kumaravelu et al. 2018 sweeps activated axon fraction 20–60 %), not a single
  fitted optimum.

**Change proposal.** Four steps, in increasing cost:

1. Multiple independent CMA-ES restarts per DBS condition. Report the
   between-restart spread per parameter as a **noise floor**, and make no claim
   about any parameter whose on-minus-off change is inside it.
2. The fixed-parameter sensitivity analysis of Maith et al. 2021 §2.6, repeated
   over BGM_22's fixed parameters — which are more numerous and several of which
   this review has shown to be weakly grounded (F6, F8, F14, F19).
3. Report the DBS parameters as a swept table across their bounds alongside the
   fitted optimum.
4. If affordable, replace the point estimate with a posterior or a confidence
   region (seat 2 points to the probabilistic parameter-estimation approaches
   cited in Rubin 2017).

**Blocking status: this blocks the interpretation of §32's fits, not their
launch.** Steps 1–3 can be decided after the fits run, but they change how many
fits are needed, so deciding first is cheaper.

## F2. Nothing validates the model except the loss it is fitted to

**Seats: 1 (1.11), 3 (3.8), 4 (4.2), 5 (5.4), 6 (6.5), 7 (7.15) — six of
seven.**

Every lineage on the panel validates on quantities the fit does not optimise,
and each named a different one: firing-rate CV, coherence and pairwise phase
against Mallet 2008 (seat 1); response to nine pharmacological
receptor-blockade experiments (seat 3); cortically evoked PSTH shape and the DBS
frequency–response curve (seat 4); cross-group discrimination of the fits
(seat 5); behavioural accuracy and reaction time (seat 6).

BGM_22's only non-BOLD constraint is `get_firing_rate_loss`, and it is a *gate*
rather than a validation — it decides which parameter vectors get evaluated. Its
GPe entries are the unsourced ones of F6. Seat 7 adds that rate is the *kind* of
constraint the experimental field has spent a decade demoting: Wichmann 2019
`[primate/human]` states that "models such as the 'rate' model are now clearly
outdated" and that rate changes "may not be as important... than originally
thought", and McGregor & Nelson 2019 concur that pattern and synchronisation
changes are the more consistent finding. **Within-list convergence: multiple.**

**Change proposal.** Adopt at least one held-out statistic. Ranked by cost:

1. **The cross-condition discrimination check** (seat 5, from Maith 2021
   Table 4): evaluate the DBS-off fitted vector against the DBS-on target and
   vice versa. If each vector does not fit its own condition better, the two
   fits have captured nothing condition-specific. Cost: two extra evaluations.
2. **The DBS frequency response** (seats 4, 6): sweep `dbs_pulse_frequency_Hz`
   from 5 to 200 Hz on the fitted DBS-on vector and compare against the
   established shape — ineffective below ~40 Hz, falling through 50–130 Hz,
   saturating above ~150 Hz. See F9.
3. **A dynamical statistic** the loss does not see — firing-rate CV, or the
   pairwise correlation of the simulated SPN population, which is also the
   quantity `TODO.md` §25's scan needs (F3).

Seat 7 records two defences of the current design that should be kept: the rate
bands are used as a plausibility gate, not as a causal claim about symptoms; and
Wichmann himself notes that most of what is known about parkinsonism comes from
resting-state recordings, offered as a limitation of the field, not of this
project.

## F3. Input correlation is zero nearly everywhere, and the project's own analysis says it dominates the observable

**Seats: 1 (1.6), 2 (2.9), 3 (3.3), 7 (7.6, 7.7) — five of seven, counting seat
3's homogeneity argument as the same mechanism.**

`experimental_data/input_streams/README.md` §3 already contains the argument:
`Var(Σ I_i) = N·v·(1 + (N−1)r)`, a 481× variance swing at N = 486 between r = 0
and r = 0.99, and the simulated BOLD is exactly that summed synaptic current.
Every source of receiver correlation except the missing-GABA pool overlap is
currently switched off — `mc.correlation_dict`, `mc.cortical_correlation` and
all four entries of `ci.shared_fraction_dict` are 0, cortical sharing is one
flat Kincaid fraction with no distance dependence, and the per-region axon pools
are drawn independently.

Seat 7 adds a quantitative constraint the modelling seats did not have.
Haber 2016 `[macaque]`, reporting Averbeck et al. 2014: corticostriatal terminal
fields from cortical areas **5 mm apart overlap by 50 %**, falling below 20 % at
30 mm. Several of the model's seven ROIs are well inside 5 mm of each other
(M1/PMd, PMd/PMv, SMA/preSMA), and the model treats their axon pools as
disjoint. This is cross-region sharing, a different quantity from the within-
region Kincaid 0.014 the model does represent. **Within-list convergence: one
voice**, but on the field authority's own published measurement. Seat 7's 7.7
adds, from two sources, that cortical convergence onto FS interneurons exceeds
that onto SPNs — the concern already logged as `TODO.md` §26.

Seats 1 and 2 both note in mitigation that the model *is* right about the slow
timescale: every receiver of a region shares one BOLD-derived `p(t)`, so the
slow co-fluctuation is fully present. What is missing is the within-bin sharing.

Seat 3 reaches the same amplitude problem from homogeneity: every neuron has
exactly 10 afferents per projection at one scalar weight, one drive weight per
postsynaptic type, and one `base_mean` per population. Girard et al. 2021
diagnosed this construction in its *own* model as the cause of too-narrow rate
and CV distributions — "all cells have the same number of input synapses from
the same number of neurons, the same constant input, the same threshold" — and a
homogeneous population sums to a scaled single neuron.

**Change proposal.**

1. **Run the `TODO.md` §25 scan before the fits, not after.** Both seats 1 and 2
   made this an ordering demand independently: a fitted drive weight absorbs the
   missing correlation, and no post-hoc analysis can separate them afterwards.
2. Add cross-region sharing to the striatal cortical streams, calibrated against
   Averbeck's 50 %-at-5 mm figure and the model's ROI geometry.
3. Add per-neuron heterogeneity to at least the drive gain, or demonstrate that
   the population sum is insensitive to it.

**Blocking status: F3 step 1 is a hard blocker on §32.** It is the one Tier-1
finding that must be resolved *before* the fits rather than alongside them, and
`TODO.md` §25 already exists to carry it.

## F4. Three DBS parameters, and the review found a problem with each

**Seats: 2 (2.7, 2.8), 3 (3.7), 4 (4.2–4.6), 6 (6.1, 6.3), 7 (indirectly, 7.10,
7.11) — five of seven, with the panel's two DBS-standing seats leading.**

Taking `DBS.md`'s three fitted parameters in turn:

- **`axon_spikes_per_pulse` is not identifiable.** It multiplies the fixed VTA
  coverage 0.4, so the expected DBS-evoked spikes per pulse is `0.4 × p` and no
  BOLD time course can separate the factors (seat 4). The 0.4 is itself flagged
  in `DBS.md` as an unvalidated single-subject value. The same parameter also
  carries three physically distinct mechanisms at once — STN efferent
  orthodromic recruitment, STN somatic antidromic invasion, and `gpe_proto`'s
  orthodromic drive into STN (seat 6).
- **`axon_spikes_per_pulse` also cannot express its known frequency
  dependence.** Antidromic propagation in hyperdirect axons is faithful only at
  low frequencies (Li et al. 2012, via Kumaravelu et al. 2018, which invokes it
  to explain a significant fall in R1 amplitude at 130 Hz versus 4.5 and 9 Hz).
  A constant per-pulse probability at 125 Hz is wrong in a known direction
  (seat 4).
- **`dbs_depolarization` hyperpolarises and has no cited basis.** `DBS.md`
  documents the sign honestly. Seat 4 notes that this lineage's DBS is a
  *depolarising* intracellular pulse evoking one spike per pulse, that a
  hyperpolarising somatic term is not unmotivated in the literature but is
  uncited here, and that at its upper bound of 10 against a ~30 mV driving force
  the term reaches ~300 units on a `dv/dt` line whose quadratic term is of order
  100 at that voltage.
- **`passing_fibres_strength`** drives `snr__thal` at a static weight, at
  125 Hz. Seats 2 and 3 both argue this is the regime where a static synapse is
  most wrong: Rubin 2017 describes high-frequency axonal drive putting synapses
  into a state where release couples to stochastic reuptake rather than
  transmitting spike patterns, and Shouno et al. 2017 calibrated GPe→STN
  short-term depression against Atherton et al. 2013 and found transmission
  falling far below its resting value within seconds at 100 Hz. Whatever
  depression exists is absorbed into the fitted value.

**Change proposal.**

1. Fix `axon_spikes_per_pulse = 1` and fit the VTA proportion instead, **or** fit
   the product and report it as one number.
2. Report all three DBS parameters as a swept table across their bounds
   (this is also F1 step 3).
3. Cite a source for the hyperpolarising somatic term, and plot the STN membrane
   trajectory at the fitted value.
4. State in the DBS results — not only in `DBS.md` — that the fitted parameters
   absorb short-term depression and are therefore not physiological recruitment
   probabilities.

---

# Tier 2 — raised by three or four seats

## F5. No nucleus in the model has recurrent connectivity within itself

**Seats: 5 (5.2b), 6 (6.2), 7 (7.2), with 1 (1.9) and 3 (3.3) adjacent — three
direct, two neighbouring.**

There is no `stn → stn`, no `gpe_proto → gpe_proto`, no `snr → snr`, no
`thal → thal`. The six GPe lateral projections run *between* the three GPe
populations, never within one.

Each seat objects for its own reason and they compound:

- **Seat 5, regression:** Maith et al. 2021 §2.2 included local inhibitory
  connections in GPi, GPe, dSN and iSN, and its Table 5 reports GPe–GPe as one
  of the significantly changed connection strengths in Parkinsonian models
  (+38.9 %, Cohen's *d* = 1.17, *p* < .001). The predecessor had them, fitted
  them, and found them to matter.
- **Seat 6, synchrony:** within-STN and within-GPe lateral radii are what
  distinguish a desynchronised healthy STN–GPe system from a synchronised
  parkinsonian one, and in Mandali & Chakravarthy 2016 two settings reproduce
  two *contradictory* published patient behaviours.
- **Seat 7, anatomy** `[mouse]`: Courtney et al. 2023 reports that GPe local
  axon collaterals "give rise to a large number of local synapses, in contrast to
  the relatively few synapses typically formed by inputs from individual SPNs",
  that **PV⁺ (prototypic) neurons provide the largest local input**, and that
  **arkypallidal neurons "do not produce appreciable levels of local
  connections."** So the model contains the pallidal local connections the
  literature calls negligible (`gpe_arky → gpe_proto/gpe_cp` at 0.008) and omits
  the ones it calls dominant. **Within-list convergence: one voice** for the
  cell-type detail; the existence of dense local collaterals is uncontested
  across the list.
- **Seats 1 and 3, gain:** a population without recurrent inhibition responds to
  its fitted drive weight unopposed. Seat 1 makes the same argument about the FS
  population, which receives GABA only from other FS neurons (no SPN→FS pair in
  `fitted_params.json`) while carrying the largest weights in the striatal table.

**Change proposal.** Add within-population inhibition to `stn`, the three GPe
populations and `snr`, at minimum reinstating what Maith et al. 2021 had. If the
GPe laterals are revised at the same time, invert the current asymmetry:
prototypic-origin local collaterals strong, arkypallidal-origin negligible.

## F6. The GPe rate bands and BOLD pooling factors — and the pooling factors are now explained

**Seats: 1 (1.3), 2 (2.5), 3 (3.4), 7 (7.1) — four of seven.**

Two separate uncited quantities, one of which seat 7 resolved.

**The pooling factors are GPe cell-type abundances.** `model_v07.md` §3.6 records
0.5 / 0.17 / 0.10 for `gpe_proto` / `gpe_arky` / `gpe_cp` as having "no recorded
source", summing to 0.77, and conjectures they are "fractions of all GPe cells
with the rest belonging to types the model does not have". Seat 7 confirms the
conjecture against Courtney et al. 2023 `[mouse]`: PV⁺ ≈ 50 %, NPAS1⁺FOXP2⁺
(arkypallidal) ≈ 18 %, NPAS1⁺NKX2.1⁺ (cortex-projecting) ≈ 12 %, with ChAT⁺ ≈ 5 %
and unclassified ≈ 15 % making up the missing 0.23 — exactly the GPe cell types
the model does not contain. **Within-list convergence: multiple** (Courtney for
the proportions, Wichmann corroborating the subtype division).

That resolves the provenance and exposes an inconsistency: the model **weights
the BOLD** by realistic abundances while **simulating** the three populations at
100 neurons each. In the network dynamics there are three times as many
arkypallidal neurons inhibiting striatum, relative to prototypic neurons, as the
abundances imply.

**The rate bands remain unsourced and are contested.** `get_firing_rate_loss`
scores `gpe_proto` and `gpe_cp` against (75, 85) Hz and `gpe_arky` against
(15, 20) Hz, with nothing cited for any of the six edges. Against the panel:
seat 1's lineage validates on Mallet 2008 `[rat]` — GPe TA 11.8 ± 1.1 Hz, TI
24.2 ± 0.7 Hz in control, with PD targets TA 12–16 and TI 17–20; seat 3's
plausible range from 20 macaque studies is ~50–70 Hz for GPe; seat 7 reports
Giossi et al. 2024 `[rodent]` at prototypical 10–100 Hz (mean ~55) and
arkypallidal 1–30 Hz (mean ~10). The *ordering* (arky slower) is right
everywhere. The prototypic value of 75–85 Hz is above every range the panel
offers, and the proto/arky ratio of 4.6 exceeds the ~2–5.5 implied by those
sources.

Seat 3 adds a direction check the project should run: in Tachibana et al. 2011
as recomputed in Shouno et al. 2017 Table 1 `[macaque]`, dopamine depletion
*raises* STN modestly (19.9 ± 9.5 → 27.6 ± 11.3 Hz) and *lowers* GPe
substantially (65.1 ± 25.6 → 41.1 ± 22.3 Hz). BGM_22's GPe bands move the other
way for a parkinsonian subject. Seat 3 also notes that BGM_22's STN band floor
(28 Hz) sits above its own macaque ceiling (~25 Hz).

**Change proposal.**

1. Cite the pooling factors as GPe cell-type abundances, with the species
   provenance recorded — Wichmann 2019 states explicitly that it is unknown
   whether the rodent molecular subtype markers translate to the primate GPe.
2. Resolve the abundance/size inconsistency: either size the populations by the
   same proportions, or state why the BOLD weighting and the simulated sizes
   disagree.
3. Derive and cite each of the nine firing-rate band edges, checking the
   parkinsonian *direction* against Tachibana 2011.

## F7. Short-term plasticity is absent, on pathways every lineage fits explicitly

**Seats: 1 (1.2), 2 (2.8), 3 (3.7) — three of seven, each naming a different
pathway.**

- Seat 1: Lindahl & Hellgren Kotaleski 2016 puts Tsodyks–Markram dynamics on
  FSN→MSN, FSN→FSN, GPe→SNr, MSN D1→SNr (facilitating), MSN D2→SNr (depressing),
  STN→SNr and both GPe types→FSN, with parameters tabulated; Hjorth et al. 2020
  fitted them to eight-pulse 20 Hz optogenetic protocols for corticostriatal,
  thalamostriatal and every intrastriatal pair.
- Seat 2: Rubin 2017's account of DBS turns on high-frequency depletion at STN
  efferent synapses.
- Seat 3: Shouno et al. 2017's parkinsonian oscillation mechanism runs entirely
  through GPe→STN depression, with the 8–15 Hz power a monotone function of the
  depression parameter.

Seat 1 also disposes of the one argument for omitting it. Chakravarty et al.
2022 dropped short-term plasticity deliberately and defended it on the grounds
that τ_rec ≈ 100 ms lies outside the band of interest — an argument that does
**not** transfer here, because BGM_22's drive varies on a 2.31 s TR grid and the
fitted quantity is a slow BOLD time course, which is exactly the timescale on
which depression and facilitation act as a *gain* on sustained input.

**Change proposal.** Add short-term plasticity to the corticostriatal and
striatopallidal synapses and to GPe→STN, or state explicitly that the
input–output gain is assumed constant across the drive's dynamic range and that
the fitted DBS parameters absorb synaptic depression (this second half overlaps
F4 step 4).

## F8. Dopamine is nowhere in the model, including outside the striatum

**Seats: 1 (1.7), 6 (6.4), 7 (7.14) — three of seven.**

`phi_1 = phi_2 = 0` on the SPN models, so the dopamine machinery is compiled and
inert (`model_v07.md` §6.2, `TODO.md` §30). Against that:

- Seat 1: both Lindahl 2016 and Chakravarty 2022 carry an explicit dopamine
  occupancy parameter α_dop with α_normal = **0.8**, mapped onto ~19 synaptic and
  excitability quantities in a published table. Two consequences: the
  *direction* of the depletion changes is unconstrained in BGM_22, so a fit may
  move a cluster the way dopamine would not; and if φ is defined as tonic
  dopamine level, then φ = 0 is neither the healthy nor the depleted state but a
  third one.
- Seat 6: three representations of the same physiological fact coexist — inert
  φ, fixed dopamine-depleted firing rates, and free weights — so a fitted weight
  change can be "dopamine", "DBS" or "compensation for the inert φ", and nothing
  separates them.
- Seat 7 `[macaque, human]`: Wichmann 2019 documents dopaminergic innervation and
  its parkinsonian loss at *extrastriatal* sites — D2-like receptors on
  striatopallidal terminals in primate GPe, D1/D2 on preterminal axons and
  terminals in monkey STN, D1-like on striatopallidal and striatonigral terminals
  in GPi/SNr — and reports that maintained pallidal and nigral dopamine may hold
  GPi/SNr rates near normal in early PD. So even a correct striatal
  parameterisation would leave this unrepresented, and the quantity most affected
  is a BOLD ROI and a gate band. **Within-list convergence: one voice**
  (Wichmann), on primate and human data.

**Change proposal.** Resolve `TODO.md` §30 by reading the Humphries source and
setting φ to whatever it implies for a dopamine-depleted state, then state in the
documentation that extrastriatal dopamine loss is an accepted omission and that
fitted weight changes conflate dopaminergic and stimulation effects.

## F9. DBS exists at one frequency, so its behaviour has no shape to check

**Seats: 4 (4.2), 6 (6.5, 6.6) — two seats, but both DBS-standing, and it is
also F2's cheapest concrete instance.**

`dbs_pulse_frequency_Hz = 125`, fixed, with no sweep anywhere in the project. The
frequency–response of STN DBS is the one quantitatively established model-level
DBS result — ineffective below ~40 Hz, falling gradually through 50–130 Hz,
saturating above ~150 Hz (Kumaravelu et al. 2016 Fig. 11E `[rat]`, matching the
clinical dependence).

Seat 4 adds that the model's construction predicts a specific failure: the
somatic shunt is proportional to `pulse(t)` and the axon spikes are drawn per
pulse at fixed probability, so both effects are very nearly **linear in
frequency** by construction, with no mechanism producing either the
high-frequency saturation or the low-frequency floor. Neither mechanism is
*excluded* by the architecture, which is what makes the sweep a real test.

Seat 6 notes the dataset makes this partly answerable empirically: the three
Berlin subjects are stimulated at 125, 130 and 100 Hz
(`experimental_data/berlin_data/vta/stim_settings.csv`).

**Change proposal.** Sweep `dbs_pulse_frequency_Hz` from 5 to 200 Hz on the
fitted DBS-on vector and report the model's BOLD change and STN/GPi rate change
against frequency. Seat 4 calls this a precondition for publishing a DBS
inference rather than a supplementary figure.

## F10. `gpe_cp` has lost the projection that defined it

**Seats: 1 (1.4), 2 (2.3), 5 (5.6), with 7 (7.1) supplying the abundance — three
of seven plus corroboration.**

Goenner et al. 2021 — this project's own lineage — introduced GPe-Cp as the
**cortex-projecting** GPe population (Abecassis et al. 2020's Npas1⁺-Nkx2.1⁺
neurons), whose functional role is to close a cortico-pallido-cortical loop: in
that model it projects to the striatal populations, the other GPe populations,
*and* the cortical Integrator-Stop.

BGM_22 has no cortex — the cortical drive is a precomputed stream nothing can
influence. So `gpe_cp` keeps its striatal and intra-pallidal projections and has
lost the one that made it distinct. It now carries `gpe_proto`'s Izhikevich
parameters (identical rows in `parameters.csv`), arkypallidal-type connectivity
(projections onto both SPN types), and `gpe_proto`'s firing-rate band. Seats 1
and 2 add that projections onto both SPN types are an arkypallidal property in
the recent literature and want a citation in a population that is not
arkypallidal.

**Change proposal.** Either remove `gpe_cp` and fold its 12 % abundance into the
BOLD pooling, or state why a cortex-projecting population is retained in a model
without cortex and give its SPN projections a source.

## F11. The BOLD observable is a modelling choice, and it was changed from the lineage's published one without record

**Seats: 2 (2.5), 5 (5.5) — two seats, but seat 5's finding is a documented
deviation from the predecessor's method and seat 2 is the panel's authority on
model observables.**

Rubin 2017's Fig. 2 shows voltage-derived and synaptic-current-derived LFPs from
the *same* model giving different spectra, and concludes that characterising what
the measured signal represents is essential. Seat 2 applies this and finds three
under-argued consequences in `model_v07.md` §3.6: the GPe monitors take the raw
`I` while the membrane equation uses the compressed `f(I, nonlin)`; `I_base` —
the *entire* drive of `snr` and `gpe_proto`, which receive no cortical input —
sits outside `I` and is invisible to the monitor; and the pooling factors were
uncited (now F6).

Seat 5 found the deviation. Maith et al. 2021 §2.3 computes BOLD from **synaptic
activity**: `τ_syn ds_j/dt = −s_j` with `s_j` incremented by `1/n_aff,j` per
arriving presynaptic action potential — a **normalised presynaptic event rate**,
containing neither the synaptic weights nor the driving force. BGM_22 maps
`I_CBF` to `I` / `I_v`, the net synaptic current, which carries the weights, the
conductance and the membrane potential.

The consequence is specific and matters for the inference: in the predecessor a
fitted weight changed the BOLD only *through the network*; in BGM_22 raising a
`gpe_striatum` scaling raises `g_gaba` and therefore `I_v` in the same timestep,
whether or not any firing rate changes. **The optimiser has a path to the BOLD
amplitude that bypasses the model's dynamics, and the ten weight-cluster
parameters sit on it.** Parameters 7 and 8 (`base_mean` on `snr` and
`gpe_proto`) have the opposite property — they reach the BOLD only through the
network, because `I_base` is excluded.

Neither seat asserts the choice is wrong; `I_v` is a defensible proxy and
CompNeuroPy supports both. Both ask that it be recorded as a deliberate change
and that the conclusions be shown not to depend on it.

**Change proposal.** Document the change of BOLD input variable and its
rationale. Refit one condition with the predecessor's normalised synaptic
activity and report whether the fitted vector moves. Separately, run the
sensitivity checks seat 2 asks for: `I` versus `I` + `I_base` for `snr` and
`gpe_proto`, and raw versus compressed `I` for the GPe populations. **This is
the only finding that reaches into the unreviewed BOLD pipeline from the model
side.**

---

# Tier 3 — raised by two seats, or by one seat with strong evidence

## F12. The cortical fibre activation pathway is absent, which is a prior on the DBS conclusion

**Seats: 2 (2.7), 4 (4.3), corroborated by 7 (7.11).** `DBS.md` limitation 2
already states the fact; the panel's contribution is how much it weighs.

Kumaravelu et al. 2018 established by model-based decomposition that the R1
cortical evoked potential — the earliest and most robust cortical signature of
STN DBS, present in rats and humans — arises from **antidromic activation of
layer-5 pyramidal axons**, and that the orthodromic STN→GPi→thalamus→cortex
route was **not required** to generate the cortical response. Li et al. 2012
found a significant correlation between hyperdirect antidromic spike probability
and symptom relief `[rat]`; Gradinaru et al. 2009 reversed parkinsonian motor
signs by driving M1 layer-5 projection neurons directly. Seat 7 adds from Emmi
et al. 2020 `[macaque]` that no subthalamo-cortical projection has been reported,
so DBS's cortical effect must travel antidromically.

BGM_22 represents the orthodromic efferent volley and the antidromic STN somatic
invasion, and cannot represent antidromic invasion of the cortical afferent —
the model's DBS repertoire is the pathway shown to be unnecessary plus a somatic
effect, minus the pathway with the symptom correlation. A fit will attribute
whatever DBS did to the mechanisms it can express.

**Change proposal.** No structural change requested — adding cortical neurons is
a different model. State it where the DBS results are reported, not only in
`DBS.md`, and state that fitted DBS parameters are the effects *expressible in
this model*.

## F13. The DBS-on and DBS-off models differ in their inputs before any DBS parameter is set

**Seat: 4 (4.8) — one voice, and the only finding in the review that no other
seat came near.**

`model_creation_kwargs["dbs"]` selects the cortical rate file: the DBS-on model
is driven by cortex *as recorded under stimulation*, the DBS-off model by cortex
as recorded without it.

Seat 4 calls this both a strength and a confound. The strength: DBS's cortical
effects, which the model cannot generate internally (F12) and which this
lineage's decomposition makes the dominant route to cortex, enter empirically
through the subject's own recording — closer to the truth than any
model-internal cortical loop. The confound: whatever part of the measured BG
BOLD change the changed cortical drive explains is not available to the three
DBS parameters, so those parameters estimate the **residual** DBS effect after
the cortical route is accounted for. That is a defensible quantity, but a
different one from "what DBS did inside the basal ganglia" as `CLAUDE.md` states
the goal.

**Change proposal.** Quantify it, using machinery the model already has:
evaluate the DBS-off fitted vector with the DBS-on cortical drive and the DBS
parameters at zero. `DBS.md` already identifies the caudate loop as "the free
control" on this logic — the caudate carries no DBS terms, so its entire
on-versus-off BOLD change *is* this quantity, measurable in the same run. Report
it alongside every DBS parameter difference, and restate the project's goal to
match what is actually estimated.

## F14. The medication state of the subject is not recorded anywhere

**Seat: 5 (5.7) — one voice, and the cheapest item in the entire review.**

The project's whole striatal calibration is the **medication-off** state of Liang
et al. 2008: the missing-GABA streams are drawn at 25/33 Hz and the rate gate is
centred there. Nothing in the repository states whether the Berlin subject was
scanned on or off dopaminergic medication.

Seat 5 raises it because the lineage has already been bitten by exactly this.
Maith et al. 2021 §2.1 used Horn et al. 2019's patients, scanned "with their
usual medication ON (Levodopa)", and its Discussion §4.1 invokes that to explain
a discrepancy: apomorphine lowers GPi and STN rates in PD patients, "this could
be a reason why we did not find an increased STN or GPi rate in our models."
Given the shared Berlin provenance, this is at least likely enough to check.

If the subject was scanned on medication, the rate bands are the wrong anchor,
every v07 cache would be drawn at the wrong rates, and `TODO.md` §30's φ question
changes character.

**Change proposal.** Find out and record it. One line of a scanning protocol.
**This should be resolved before any full-length cache is built** (Roadmap
phase 2), because a wrong answer invalidates them.

## F15. The STN receives the striatum's cortical proportions, and one entry is contradicted by tracing

**Seat: 7 (7.10) — one voice within seat 7's list, but reporting several
independent tracing studies.**

`CorticalInputs` splits each STN neuron's 500 cortical afferents using
`mc.cortical_proportions_dict` — the *same* per-region mix as the striatal
populations (`model_v07.md` §3.3, §8). For the putamen loop that means an STN
whose cortical input is 28 % M1, 18 % PMv, 15 % SMA, **13 % S1**, 11 % PMd,
10 % dlPFC, 5 % preSMA.

Emmi et al. 2020 `[macaque]` reports Von Monakow et al. 1978 finding projections
to STN from Brodmann areas 4, 6 and 8 but explicitly none "arising from Brodmann
areas 9 and 3,1,2" — areas 3, 1, 2 being S1. (The area-9 negative was later
overturned by Haynes & Haber 2013, which found a dorsal prefrontal projection to
medial STN; nothing in this seat's reading restores an S1 projection.) The
corticosubthalamic projection is also organised quite differently from the
corticostriatal one — M1 to dorsolateral STN with a somatotopy, SMA and ventral
premotor to the medial portion with an *inverse* somatotopy, caudal dorsal
premotor to ventrolateral STN, dACC to the medial tip.

**Change proposal.** Do not apply the striatal proportions to the STN
unexamined. At minimum remove or justify the 13 % S1 share; ideally derive a
separate corticosubthalamic proportion set, which `parameters.py` can carry as a
second dict without touching the striatal one. Note this invalidates the v07
`CorticalInputs` caches, so it belongs in Roadmap phase 1.

## F16. The STN has no spatial structure, and the geometry to give it one is already in the repository

**Seat: 6 (6.1), with 7 (7.11) supplying the anatomy — two seats.**

`_create_dbs_on_array` sets `dbs_on = 1` on a shuffled random 40 % of 100 STN
neurons. There is no lattice, no coordinate, no electrode. Mandali &
Chakravarthy 2016's headline result is that moving the electrode between three
positions in a 50×50 STN lattice *reverses* the behavioural outcome (accuracy
0(100) at position 1 against 100(0) at position 3).

Seat 6 found that the data is present and unused.
`experimental_data/berlin_data/vta/sub-01/sub-01_overlap.csv` records the VTA's
overlap with the three functional STN subdivisions separately — motor 35+23 of
70+75 (= the 0.4 the code uses), associative 5+7 of 68+68, limbic 1+5 of
54+55 — and `stim_settings.csv` shows sub-01 on a directional Sensight lead with
segmented contacts. So the model's homogeneous STN is given the *motor*
subdivision's coverage while the same file records associative and limbic
coverage being discarded. Seat 7 `[human, macaque]` confirms the tripartite
subdivision is real in both relevant species, with overlapping rather than sharp
boundaries, and that human STN contains GABAergic interneurons the model does not
have.

**A consistency check for the authors:** `sub-01_volumes.csv` gives the VTA as
37 + 31 = 68 units while the three overlaps sum to 58 + 12 + 6 = 76. Either the
units differ between the files or the subdivisions overlap; one of the two is
being read in a way the other does not support. This should be resolved before
the 0.4 is relied on further.

**Change proposal.** Resolve the unit inconsistency first. Then either give the
STN spatial structure — the codebase already builds a 3D lattice for the
striatum — and place the VTA using the subdivision data, or state explicitly that
electrode position is not represented and that the DBS conclusions are therefore
position-independent.

## F17. Strict segregation, in two places, where the primate anatomy reports overlap

**Seats: 3 (3.1) and 7 (7.13) — two different segregations, two seats.**

**Direct/indirect** (seat 3): `str_d1 → snr, gpe_cp` and
`str_d2 → gpe_proto, gpe_arky, gpe_cp`, so iSPNs reach no output nucleus.
Girard et al. 2021's central structural commitment is that more than 80 % of
macaque striatal neurons project to *both* GPe and GPi/SNr (Parent et al. 1995;
Lévesque & Parent 2005), and their model shows segregation is not needed to get
action selection. Seat 3 notes the inconsistency of accepting macaque tracer
counts as the anchor for the cortical proportions while rejecting macaque tracer
counts on striatofugal collateralisation.

**Caudate/putamen** (seat 7) `[macaque, human]`: Haber 2016 — "the separation
between the caudate nucleus and the putamen is merely a structural one, based
solely on the IC separation, not a functional one" — with extensive convergence
and specific striatal interface zones; McGregor & Nelson 2019 add that loss of
functional-channel segregation is a documented feature of the parkinsonian basal
ganglia. **Within-list convergence: multiple.** BGM_22's two loops share no
projection at all. This bears on `DBS.md`'s use of the caudate as "the free
control": clean in the model by construction, not licensed as clean in the
subject.

**Change proposal.** Add an iSPN→SNr collateral, or record strict
direct/indirect segregation as an accepted assumption with its cost. Separately,
state wherever the caudate is used as a control that the two loops' independence
exceeds the anatomy and that PD degrades channel segregation.

## F18. Missing inputs: CM/Pf thalamus, STN→striatum

**Seats: 3 (3.6) and 7 (7.12) — two seats, and seat 7's is multi-source within
its own list.**

**Intralaminar thalamus.** In Girard et al. 2021 the CM/Pf input reaches *every*
simulated nucleus — MSN, FSI, STN, GPe, GPi/SNr — and its sensitivity analysis
makes it the input that "modulates the responsiveness of action selection", i.e.
a global gain. Seat 7 corroborates from three of its six reviews: Emmi
`[squirrel monkey]` (centromedian → dorsolateral STN, parafascicular → medial
and rostral STN), Tepper `[rat, denser in primates than rodents]`, and Haber
listing thalamus among the three major striatal afferent sources. BGM_22's only
thalamic population is the BG *target* (`snr → thal → striatum`); nothing plays
the intralaminar role.

**STN→striatum.** Girard et al. 2021 includes STN→MSN and STN→FSI at 17 % of STN
neurons projecting; Emmi `[squirrel monkey, macaque]` reports the same from Smith
et al. 1990 and Sato et al. 2000b. BGM_22 has no such projection.

**Change proposal.** Neither is likely to change a BOLD correlation much. The
CM/Pf omission matters more, because a missing global thalamic gain competes for
the same explanatory role as the fitted drive weights — so it should at minimum
be recorded as an accepted limitation with that reasoning attached.

---

# Tier 4 — single-seat findings worth recording

## F19. The striatum has one interneuron class out of at least eight, at a fraction below even the rodent value

**Seat: 7 (7.5). Within-list convergence: multiple** (Tepper for the diversity
and the NGF conductance, Haber independently for the primate fraction).

Tepper et al. 2018 `[rat/mouse, primate notes]` catalogues FSI, LTS, calretinin,
TH, neurogliaform, fast-adapting and spontaneously-active-bursty interneurons,
"and it is likely that we have not yet found all of them". BGM_22 has FS only, at
2.9 % of striatal cells. Haber 2016 `[macaque/human]` adds that MSNs "in
nonprimate species... account for over 90 % of the cells, and probably far less
in the primates" — so 97.1 % projection neurons is above even the rodent figure,
in a human model.

The specific omission that matters for this project's observable: NGF
interneurons evoke a GABA_A,slow IPSC in SPNs roughly an order of magnitude
slower than the conventional fast GABA_A response, connecting to a very high
proportion of SPNs in their axonal field — "an extremely powerful source of
inhibition to the SPNs, not only because of its amplitude but also because of the
extremely long duration and slow decay of the IPSC" `[mouse]`. BGM_22's SPN
`tau_gaba` is 4 ms with nothing slower anywhere. For a model whose observable is
a **time-integrated synaptic current**, a missing inhibitory conductance an order
of magnitude slower is a missing low-pass term in the fitted quantity itself.

**Change proposal.** Consider a slow inhibitory conductance in the striatum
standing in for NGF input — cheaper than a new population, since it can be a
second GABA time constant rather than new cells. At minimum record the
projection-neuron fraction as a rodent-derived value in a primate model.

## F20. GPe input balance: ~80 % of GPe synapses are striatal, and the model's GPe is cortically driven

**Seat: 7 (7.3). Within-list convergence: one voice** (Courtney 2023
`[mouse, rat anatomy]`), but the underlying anatomy is uncontested in the list.

GABAergic synapses are ≈ 80 % of all synapses in GPe, the large majority from
dorsal striatum. In BGM_22, `gpe_arky` and `gpe_cp` each receive **500 cortical
afferents per neuron** against **10** striatal afferents from `str_d2`.

Recorded in the model's favour: Courtney also emphasises that spatially
distributed cortical inputs "form the largest source of excitatory inputs to the
GPe, in contrast to the traditional model which assumes dominant glutamatergic
input from the STN" — so having a direct cortico-pallidal drive at all is correct
and unusual. The objection is to the balance against the striatal projection.

**Change proposal.** Reconsider the 500 cortical afferents per GPe neuron
against the 10 striatal ones, or record the imbalance. Note this is a cache
parameter, so it belongs in Roadmap phase 1.

## F21. Striatal input to GPe is cell-type-specific and the model's assignment is partly reversed

**Seat: 7 (7.4). Within-list convergence: one voice** for the target specificity
`[mouse]`.

Courtney 2023: iSPNs strongly target STN-projecting PV⁺ neurons; dSPNs largely
target NPAS1⁺ neurons, Pf-projecting PV⁺ neurons and ChAT⁺ neurons. BGM_22 has
`str_d1 → gpe_cp` only, at 0.005 — the smallest weight in the table — and
`str_d2 → gpe_proto, gpe_arky, gpe_cp` spread evenly.

**Change proposal.** Strengthen dSPN→GPe and route it to both NPAS1⁺
populations; weight iSPN→GPe toward `gpe_proto`. Partly absorbed by the fitted
`str_d1__bg` and `str_d2__bg` clusters, but the topology is not.

## F22. The renormalisation of the cortical proportions loses the densest projections

**Seat: 7 (7.9). Within-list convergence: one voice** (Haber 2016 `[macaque]`),
on his own primary measurements.

The audit of `experimental_data/cortical_proportions/README.md` *supports* the
derived topography: M1 terminating almost entirely in putamen; dPFC primarily in
the caudate head and rostral putamen with few terminals dorsolaterally; rostral
premotor bridging both. The caudate/putamen contrast the inference reads is
sound.

What the audit adds is a number for the excluded input. The README's caveat 2
notes that cingulate, insula, temporal, parietal, orbital and ventrolateral
prefrontal cortex are 25–60 % of labelled cells with "nowhere to go". Haber:
"the volume occupied by the collective dense terminal fields from the vmPFC,
dACC and OFC is approximately 22 % of the striatum, a larger cortical input than
would be predicted by the relative cortical volume of these areas." So the seven
ROIs are not a representative sample being renormalised — they systematically
exclude the *densest* projections.

**Change proposal.** Record this in the README's caveat 2. No change to the
values.

## F23. The DBS electrode may contaminate the BOLD signal, in the ROI that matters most

**Seat: 5 (5.8) — one voice, and the lineage's own recorded experience.**

Maith et al. 2021 §4.3, verbatim: "the patients received a DBS electrode which
may cause artifacts in the BOLD signal even when switched off, especially in the
STN signal, which reduces the validity of our results about the STN." BGM_22 fits
seven ROIs including STN, in a DBS patient, in both conditions — with the device
actively pulsing in one of them. Nothing in the project's documentation mentions
it. STN is both a loss ROI and the stimulated population.

**Change proposal.** State it as a limitation, as the predecessor did, and check
the STN ROI's signal quality against the other six.

## F24. The model is a species chimera and nowhere says so

**Seats: 4 (4.1, 4.7), 7 (7.1, 7.5), 3 (3.2) — three seats, though each raises a
different piece.**

Collected: the subcortical delays are rat (seat 4 established that all seven
match Kumaravelu et al. 2016 Table 1 exactly — real provenance the project does
not record, which also resolves seat 3's complaint into a rat-versus-macaque
disagreement between source literatures); the striatal firing rates are MPTP
macaque medication-off; the FS rate is normal macaque times a rodent depletion
factor; the cortical proportions are macaque tracer counts; the GPe three-way
division is mouse molecular work whose primate translation Wichmann says is
explicitly untested; the striatal cell-type proportions are del Rey et al. 2022;
the subject, target data and DBS frequency are human.

Seat 4 makes the mixture consequential rather than merely untidy: Kumaravelu
et al. 2016 §5.2 reports that rat basal ganglia rates are much lower than
primate ones and that *"the differences in firing rates likely underlie the
variations in the frequency-dependent effects of DBS between the animal
models"* — low-frequency stimulation suffices to mask a low-rate neuron, while
>100 Hz is needed for a high-rate one. So the DBS frequency window is a function
of the target's firing rate, and BGM_22 sets its rates from one species mix and
its frequency from another. The combination may well be right; the reasoning has
not been done, and it determines the shape F9's sweep should produce.

**Change proposal.** Add a species-provenance column to the parameter
documentation (`model_v07.md` §5 and the `experimental_data/` READMEs already
have the right structure for it). Record the delay provenance specifically, since
it is currently absent and the values are not arbitrary.

---

# What the panel praised

Recorded because the reviews are otherwise critical and these are places where
BGM_22 exceeds the panel's own practice.

- **The input-stream contract** (`experimental_data/input_streams/README.md`).
  Seat 2: "a document we do not have an equivalent of" — a written statement of
  what the surrogate input must reproduce, checked in code at build time, with
  closed-form target statistics and the history of the bug that motivated it.
  CBGTPy's external input is Poisson at a tuned rate, checked against nothing.
- **The subject-specific drive.** Seats 1, 2 and 5 all noted this independently.
  The model is asked to track one human's basal ganglia TR by TR, driven by that
  same human's cortical activity — a harder and more falsifiable target than the
  pooled population statistics or tuned Poisson rates the panel validates
  against. Seat 5 adds that it is strictly harder than the predecessor's
  functional-connectivity target: 15 numbers against 310 per region.
- **The geometric missing-GABA construction.** Seat 2 singles it out as the
  best-engineered part of the input machinery: the source pool is realised
  explicitly and `f(d)` emerges from the geometry rather than being computed and
  imposed — which is what Rubin 2017 asks for when it cites Rosenbaum et al.
  2016 on the spatial structure of correlated variability.
- **Honesty about scope.** Seat 3 on `input_streams/README.md` §5: "that is the
  correct instinct and the correct division" — stating which inferences the model
  may support and which it may not. Its request is only that the *parameter*
  claims inherit the same discipline (F1).
- **The circuit is more complete than two panel members' own models.** Seat 6
  records that Mandali & Chakravarthy 2016 omits both the hyperdirect pathway and
  GPe→GPi; BGM_22 has both.
- **A measured electrode geometry.** Seat 6: its own DBS current is a Gaussian in
  an abstract lattice whose coordinates have no anatomical referent, while
  BGM_22's coverage fraction comes from a segmented VTA in a real patient against
  a real subdivision atlas — better grounded than the panel's, if F16's spatial
  structure were added.
- **Fitting weights is the right target.** Seat 7 `[rodent, macaque, human]`:
  Wichmann 2019 documents that parkinsonism involves "considerable morphological
  and functional plasticity at synapses within the basal ganglia, thalamus, and
  cortex" — glutamatergic remodelling in striatum and STN, GABAergic remodelling
  of the pallidosubthalamic projection. The literature says the parkinsonian
  brain differs from the healthy one in its *synapses*, not only its rates, so an
  inference about synaptic weights is asking about the right quantity.
- **The Liang firing-rate anchor survives its audit.** Seat 7's mandate included
  auditing it, and the verdict is that no value should change. Three findings:
  it sits on one side of a live disagreement (Singh et al. 2016 `[human]`
  supporting, Deffains et al. 2016 opposing, per McGregor & Nelson 2019) which
  the README should cite; the rates are 25–50× the normal-primate resting rate
  Haber gives (0.5–1 Hz), which the README's own "why these rates are so high"
  section anticipates; and — the useful part — the dSPN < iSPN **ordering** is
  independently supported by antidromically identified rodent recordings (Mallet
  et al. 2006; Kita & Kita 2011, via McGregor & Nelson), which is exactly the
  response-direction assumption the README identifies as its weakest link.
  **Within-list convergence: multiple.**

---

# Reading guide for the triage

Not a verdict and not an ordering of the entries — `TODO.md`'s Roadmap owns
that. This is only what the panel's own remarks imply about sequencing, offered
to make the triage pass cheaper.

**One finding is a hard blocker on the fits (§32):** F3 step 1, the input-
correlation scan, because a fitted drive weight absorbs the missing correlation
and no later analysis can separate them. `TODO.md` §25 already carries it.

**Four findings would invalidate the input caches if accepted**, so they belong
in Roadmap phase 1 before any full-length build: F15 (STN cortical proportions),
F20 (GPe afferent counts), F14 (the medication state, if the answer is "on"),
and F3 steps 2–3 (cross-region sharing, heterogeneity).

**Three are answerable by reading rather than by computing:** F14 (a scanning
protocol), F6 step 1 (citing the pooling factors, now that F6 identifies them),
F24 (recording provenance already established in this review).

**Four are cheap runs on artefacts the project will produce anyway:** F2 step 1
(cross-condition discrimination, two evaluations), F9 (the frequency sweep),
F13 (the caudate control, already measured by any on/off pair), F11's
sensitivity checks.

**The rest are model changes** whose cost is a cache rebuild plus a revalidation
run, and they are what the verdict pass exists to decide.
