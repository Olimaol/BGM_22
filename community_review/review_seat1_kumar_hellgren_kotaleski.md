# Seat 1 — Kumar / Hellgren Kotaleski (KTH Stockholm)

Review of BGM_22 written in character as the Kumar–Hellgren Kotaleski lineage,
applying that lineage's own published standards unfiltered. Part of the
community-conventions survey defined in `TODO.md` §34; the panel and the reading
list are in `README.md`.

**Read for this review (full texts, from the PDFs in this directory):**

- Lindahl M, Hellgren Kotaleski J (2016), eNeuro 3(6), `10.1523/ENEURO.0156-16.2016`
- Bahuguna J, Sahasranamam A, Kumar A (2020), PLOS Comput Biol 16(3):e1007748,
  `10.1371/journal.pcbi.1007748` (+ the 2025 correction, `10.1371/journal.pcbi.1013638`,
  which is an affiliation fix only and carries no scientific content)
- Chakravarty K, Roy S, Sinha A, Nambu A, Chiken S, Hellgren Kotaleski J, Kumar A
  (2022), eNeuro 9(2), `10.1523/ENEURO.0376-21.2022`
- Hjorth JJJ et al. (2020), PNAS 117(17):9554–9565, `10.1073/pnas.2000671117`

**What was reviewed:** `model_v07.md`, `model_v08.md`, `DBS.md`,
`experimental_data/input_streams/README.md`,
`experimental_data/activity_striatum/README.md`,
`experimental_data/cortical_proportions/README.md`, and the loss/gate definition in
`BOLD_optimization/get_loss.py` (`get_firing_rate_loss`, `proj_clusters`,
`set_opt_params_v07`).

Every point carries a **goal-relevance** tag — whether it plausibly matters for
this project's stated goal, single-subject resting-state BOLD fitting and DBS
inference — so that nothing is pre-filtered but the triage is pre-structured.

---

## The standard this seat applies

Four things recur across our four papers and constitute what we would demand of
any BG network model submitted to us:

1. **Every parameter carries a per-row provenance, and the unknowns are labelled
   as unknown.** Lindahl 2016 Tables 7–9 give a source column for each of ~60
   synaptic parameters, with `n.d., estimated` or `n.d., assumed as for X` written
   out where no measurement exists. Table 9 lists all 19 dopamine scaling
   coefficients with their source papers.
2. **Nucleus sizes and fan-ins are anatomically scaled, not round numbers.**
   80,000 neurons distributed as MSN D1 37,971 / MSN D2 37,971 / FSN 1,599 /
   STN 388 / GPe TA 329 / GPe TI 988 / SNr 754, from Oorschot 1996 absolute
   counts. Fan-in per pair derived from arborisation volumes times measured pair
   connection probabilities (2,800 MSNs in an MSN axonal field × Taverna 2008's
   13/3/14/18 % → 364/84/392/504).
3. **Validation is on dynamics, not only on mean rates.** Lindahl 2016 validates
   firing rate *and* CV *and* coherence *and* pairwise phase relations of GPe
   TA/TI and STN against Mallet 2008, in two cortical states. Chakravarty 2022
   validates the *shape* of the cortically evoked triphasic SNr response against
   Nambu-lab data. Bahuguna 2020 validates β-burst duration (~0.24 s) and peak
   frequency against rodent recordings.
4. **A published claim is accompanied by a parameter-perturbation analysis
   showing which connection the claim rests on.** Lindahl 2016 Fig. 4B/8B
   restores each dopamine-depleted parameter one at a time; Chakravarty 2022
   Fig. 6 varies six connections over 7 steps and reports a CV per zone;
   Bahuguna 2020 resimulates 10,000 networks with all nine parameters drawn at
   ±20 %.

We recognise a great deal of care in this project's documentation — the input
stream contract, the cache validation, and the honest `README`s on the firing
rates and cortical proportions are better than what most submissions show us. The
points below are where the model departs from the four standards above.

---

## Points

### 1.1 The striatum is 98 % open-loop, and this is presented as a striatal model

`experimental_data/input_streams/README.md` §4.1 states it plainly: a dSPN
receives ~66.8 simulated GABAergic afferents against ~2573 synthetic ones. We
have no objection to compensating for a small simulated volume — but we would
object to any wording, in a paper or a figure caption, that calls the result a
striatal microcircuit. Hjorth 2020 built a 10,000-neuron striatal network
precisely because the intrastriatal connectivity *is* the object of study; the
1000-neuron cube here is a receiver array with a statistically calibrated input,
and `input_streams/README.md` §5 already says so. Our concern is that
`model_v07.md` §7 and the class name `Microcircuit` do not, and downstream
readers will take the name at face value.

**Goal relevance: low for the fit, high for the writing.** The BOLD prediction
does not need the recurrence; the paper's claims must not exceed it.

### 1.2 No short-term plasticity anywhere in the model

Lindahl 2016 puts Tsodyks–Markram dynamics on FSN→MSN, FSN→FSN, GPe→SNr,
MSN D1→SNr (facilitating, U = 0.0192), MSN D2→SNr (depressing, U = 0.24),
STN→SNr and both GPe types→FSN, with the parameters in Table 8; Lindahl 2013
(the predecessor this lineage builds on) makes signal *enhancement by short-term
plasticity* its central result. Hjorth 2020 fitted Tsodyks–Markram parameters to
eight-pulse 20 Hz optogenetic protocols for corticostriatal, thalamostriatal and
every intrastriatal pair (their Fig. 9), and found facilitation from thalamus and
other dSPNs against depression from contralateral M1. BGM_22 has static synapses
throughout — `model_v07.md` §6 shows plain exponential conductances with no
resource variables.

We note in fairness that our own Chakravarty 2022 dropped short-term plasticity
deliberately and argued the case (short stimuli, τ_rec ~100 ms is outside the
β band, and no experimental evidence ties the triphasic response to STP). That
argument does **not** transfer here: BGM_22's drive varies on a 2.31 s TR grid
and the fitted quantity is a slow BOLD time course, which is exactly the
timescale on which depression and facilitation act as a *gain* on a sustained
input. A depressing corticostriatal synapse would attenuate a sustained
high-rate epoch that the current model passes through linearly.

**Goal relevance: high.** The loss is a correlation between the simulated and
measured BOLD time course; STP is a per-pathway temporal filter sitting directly
in that path, and its absence is not neutral — it is the assumption that the
input–output gain is constant across the drive's dynamic range.

### 1.3 The GPe firing-rate bands are not from any source we recognise, and the
proto/arky ordering conflicts with our own validation data

`get_loss.get_firing_rate_loss` scores `gpe_proto` and `gpe_cp` against
(75, 85) Hz and `gpe_arky` against (15, 20) Hz. Our model is validated against
Mallet 2008: GPe TA (arkypallidal) 11.8 ± 1.1 Hz and GPe TI (prototypical)
24.2 ± 0.7 Hz in the control rat, with PD-condition targets of TA 12–16 and
TI 17–20 Hz (Chakravarty 2022, Materials and Methods, citing de la Crompe 2020).
The *ordering* in BGM_22 agrees with ours — arky slower than proto — but the
prototypical value is three to four times ours, and no source is cited in the
code for any of the six numbers. Awake primate GPe rates of 60–80 Hz do exist in
the literature, and would justify 75–85, but then the arky band should have been
scaled the same way and it visibly was not: the proto/arky ratio is 4.6 here
against ~2.0 in Mallet.

Separately, `model_v07.md` §3.6 records that the GPe BOLD pooling factors
(0.5 / 0.17 / 0.10) "have no recorded source" and sum to 0.77. We would not accept
a figure produced with an uncited weighting.

**Goal relevance: high.** The rate probe gates the BOLD run entirely
(`firing_rate_gate` = 0.5), so an unsourced band decides which parameter vectors
are ever evaluated at all.

### 1.4 `gpe_cp` is a third GPe population with no counterpart in the recent
literature, and it projects where prototypical neurons do not

The recent GPe subdivision our lineage models is two-way: TA/arkypallidal, which
projects to striatum (both SPNs and FSNs) and *not* to STN, and TI/prototypical,
which projects to STN and the output nuclei and — per Glajch 2016 and Saunders
2016 — reaches striatum only onto FSNs, with only ~10 % of TI cells projecting
back to striatum at all (Abdi 2015). BGM_22 has `gpe_proto`, `gpe_arky` **and**
`gpe_cp`, where `gpe_cp` projects to `str_d1`, `str_d2` and `str_fsi` at
weights 0.5/0.5/0.8 and receives from `str_d1`, `str_d2` and `stn`
(`model_v07.md` §5). If `cp` denotes the caudal/prototypic-projecting
subpopulation of Abdi 2015 / Mallet 2012, then its SPN projections need a
citation, because that is the property the literature assigns to TA and
explicitly withholds from TI. If it denotes something else, the document does not
say what.

Our own model's arrangement was itself a bet, and Lindahl 2016 §"Model
robustness" reports what testing it cost us: the TA→MSN synaptic time constant
had to be >6× a normal GABA synapse or the striatum oscillated in the control
state, and the TI–TA phase relation reverses if MSN→TA fan-in is raised from 25
to 100. Those are exactly the tests a third GPe population invites.

**Goal relevance: medium.** For a BOLD fit the three GPe populations are pooled
into one ROI monitor with fixed scale factors, so their internal division
influences the output only through the network dynamics — but the fitted weight
cluster `gpe_striatum` covers seven projections at once, so a wrong assignment is
absorbed silently into one number.

### 1.5 Delays are single fixed values with no stated source, and this lineage has
shown delays are load-bearing for exactly the STN–GPe phase structure

`parameters.csv` gives one delay per projection (1.5–5 ms; `model_v07.md` §5).
Lindahl 2016 Fig. 9C/D reports that stepping the cortex→striatum and cortex→STN
delays from 2.5 to 20 ms *distorts the STN–TI phase relation* and concludes
"these parameters need to be controlled well in the real system." Our delays are
cited per pathway (Jaeger and Kita 2011 for corticostriatal 2.5 ms, Park 1982 for
MSN→GPe 7 ms, Shen and Johnson 2006 / Ammari 2010 for STN→SNr 4.5 ms). We also
note that BGM_22 uses a single scalar delay per projection where our own values
differ by a factor of 4 across pathways within the same nucleus.

**Goal relevance: low–medium.** BOLD integrates over 2.31 s, so millisecond
delays cannot move the fitted signal directly. They matter only if they change
the network's operating regime — which, per our Fig. 9, they can.

### 1.6 The cortical drive is uncorrelated across pathways, and the model's own
documentation identifies this as the dominant determinant of its output

`input_streams/README.md` §3 states that input correlation sets the BOLD
amplitude (a 481× variance swing at N = 486 between r = 0 and r = 0.99) and §4.5
records that the corticostriatal and hyperdirect drives are drawn independently,
so a cortical neuron projecting to both striatum and STN is two uncorrelated
sources in the model. Our own models also use uncorrelated Poisson input, and
Bahuguna 2020's limitations section flags this as a real weakness — "inputs to
STN-GPe are richer in their statistics and dynamics, e.g. bursty, periodic,
correlated ... such non-Poissonian inputs might underlie resonance of the STN-GPe
network at certain frequencies." We flag it here for a stronger reason than we
flagged it for ourselves: in our models the correlation affects a spectral
observable, whereas here it multiplies the amplitude of the only fitted quantity.

We read `TODO.md` §25 and see the scan is planned. Our comment is on ordering,
not on awareness: the scan must precede the fits, because a fitted drive weight
absorbs the missing correlation and no post-hoc analysis can separate them.

**Goal relevance: high.**

### 1.7 The dopamine state is asserted by parameter choice, not represented

Both Lindahl 2016 and Chakravarty 2022 carry an explicit dopamine occupancy
parameter α_dop ∈ [0, 1] with α_normal = 0.8, and a published table mapping it
onto ~19 neuronal and synaptic quantities (resting potentials, spike thresholds,
CTX→MSN D1 NMDA up, CTX→MSN D2 AMPA up, MSN–MSN collaterals down to a quarter,
FSI→MSN D2 connectivity doubled, MSN D2→GPe TI strengthened, CTX→STN
strengthened, GPe excitability reduced). BGM_22 sets `phi_1 = phi_2 = 0` on the
SPN models — the dopamine machinery is present in the equations and inert
(`model_v07.md` §6.2) — and represents the parkinsonian state instead through
striatal firing rates taken from a parkinsonian recording plus free weight
scalings.

We accept the design argument (the weights are fitted, so a dopamine
parameterisation would be partly redundant), but two consequences follow that the
documentation should state. First, the *direction* of the dopamine-depletion
changes is then unconstrained: a fit is free to move a cluster the way dopamine
would not. Second, `TODO.md` §30 already notes that `phi = 0` is unchecked
against Humphries's source paper; with α_normal = 0.8 in our formulation, zero is
not the healthy value either, so `phi = 0` is a *third* state, neither healthy nor
depleted, unless the SPN model's φ is defined differently from ours.

**Goal relevance: medium.** It does not break the fit, but it weakens the
inference: "which parameters had to change" is only interpretable against a
stated baseline.

### 1.8 Weight cluster scalings preserve ratios that were never jointly measured

`get_loss.proj_clusters` ties, for example, all six GPe lateral projections to one
scaling and all seven GPe→striatum projections to another. The documentation
argues this preserves the literature balance and conditions the search, which we
find sound *as an optimisation strategy*. As a *biological* statement it asserts
that the CSV's relative weights within a cluster are jointly correct. In our
tables those weights come from different papers, different preparations and
different temperatures — Lindahl 2016 Table 7 explicitly discusses TA→FSN IPSC
time constants differing between Glajch 2016 (~21 °C) and Saunders 2016 (~31 °C)
by a factor of 4 — and several are marked `n.d., estimated`. Locking an estimated
ratio and fitting only its overall scale hides the estimate inside a fitted
number.

We would want, at minimum, one leave-one-out run per cluster showing the loss is
insensitive to the intra-cluster ratio, or the ratios' provenance listed as
Lindahl 2016 Table 7 does.

**Goal relevance: medium–high.** This is the inference's actual parameterisation.

### 1.9 The FS population receives GABA only from other FS neurons

`fitted_params.json` has no dSPN→FS or iSPN→FS pair, so — as
`input_streams/README.md` §4.1 states — the 29 simulated FS neurons receive
130 spikes/s of inhibition against 61,000 for a dSPN, in both the circuit and the
compensating streams. This is defensible as a data limitation (SPN→FSI
connections are rare in paired recordings) but it interacts badly with point 1.3:
Lindahl 2016 Fig. 5A is precisely the result that FSN and GPe inhibition dominate
MSN rate control at low cortical drive while MSN collaterals dominate at high
drive. A model whose FS population has essentially no inhibitory input will place
FS activity wherever its fitted drive weight puts it, with no negative feedback,
and FS→SPN weights are the largest in the whole striatal weight table (mixture
mean 6.06 against 0.41).

**Goal relevance: medium.** One fitted parameter (`FS` drive weight, index 2)
controls a population with unusually high downstream gain and no local restraint.

### 1.10 There is no reported robustness analysis of any kind

Bahuguna 2020 resimulated 10,000 networks with every network parameter drawn from
a Gaussian at 20 % SD to establish that the firing-rate/burst conclusions survive;
Lindahl 2016 restored 25 dopamine-modified parameters one at a time; Chakravarty
2022 swept six connections over seven values each and reported per-zone
coefficients of variation. BGM_22's documentation contains no equivalent, and
`TODO.md` §16 (DBS-constant sensitivity) suggests one is planned only for the DBS
constants. Since the inference's product is "these parameters changed", the
credibility of that statement rests entirely on how sharply the loss constrains
each one.

**Goal relevance: high.** This is the difference between a fitted vector and an
inference.

### 1.11 A remark in the model's favour, and one caution attached to it

We want to record that the *fit-to-a-single-subject's-own-time-course* design is
stronger than what we do. Our models are validated against pooled population
statistics from anaesthetised rats; BGM_22 asks the model to track one human's
basal ganglia BOLD TR by TR, driven by that same human's cortical activity. That
is a harder target and a more falsifiable one.

The caution: with 19 free parameters, a smooth 310-point target and a drive that
is itself derived from the target recording's cortex, a good correlation is
attainable by a model whose internal dynamics are wrong. Our lineage's response
to that risk is Lindahl 2016's Fig. 2/3 — validate on quantities the fit does not
optimise (CV, coherence, phase). BGM_22's only such quantity is the firing-rate
band, which is a gate rather than a validation, and whose GPe entries are the
unsourced ones of point 1.3.

**Goal relevance: high.**

---

## What we would ask for before publication

1. A source line for each of the nine firing-rate bands and the three GPe BOLD
   pooling factors (points 1.3, 1.4).
2. The input-correlation scan run *before* the fits (point 1.6, `TODO.md` §25).
3. One held-out validation statistic that the loss does not optimise (point 1.11).
4. A per-parameter sensitivity analysis of the fitted vector (point 1.10).
5. Either short-term plasticity on the corticostriatal and striatopallidal
   synapses, or an explicit argument that its timescale is irrelevant to a
   TR-resolution fit (point 1.2).
6. Naming: `Microcircuit` renamed or the documentation's own §5 caveat repeated
   wherever the class is described (point 1.1).
