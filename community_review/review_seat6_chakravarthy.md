# Seat 6 — Chakravarthy (IIT Madras)

Review of BGM_22 written in character as the Chakravarthy lineage, applying that
lineage's own published standards unfiltered. Part of the community-conventions
survey defined in `TODO.md` §34; the panel and the reading list are in
`README.md`. This is the panel's third DBS-capable voice.

**Read for this review (full texts, from the PDFs in this directory):**

- Mandali A, Chakravarthy VS (2016), Front Hum Neurosci 10:450,
  `10.3389/fnhum.2016.00450`
- Nair SS, Muddapu VR, Chakravarthy VS (2022), Front Comput Neurosci 15:756881,
  `10.3389/fncom.2021.756881`

**What was reviewed:** `model_v07.md`, `model_v08.md`, `DBS.md`, the three
`experimental_data/` READMEs, `BOLD_optimization/get_loss.py`, and — because
this seat's central concern turned out to hinge on it —
`experimental_data/berlin_data/vta/`.

Every point carries a **goal-relevance** tag against the project's stated goal
(single-subject resting-state BOLD fitting and DBS inference).

---

## The standard this seat applies

1. **The stimulated nucleus is a spatial object and the electrode has a
   position in it.** Mandali & Chakravarthy 2016 models STN as a 50×50 lattice
   and delivers DBS as a spatial Gaussian,
   `I_DBS(i,j) = A·exp(−((i−i_c)² + (j−j_c)²)/σ²)`, with the electrode centre
   `(i_c, j_c)` and spread `σ` as explicit variables. Our headline result is
   that moving the electrode between three positions in that lattice *reverses
   the behavioural outcome*: at Pos 1 the model's accuracy in choosing A versus
   avoiding B was 0 (100), at Pos 3 it was 100 (0), and at Pos 2 it sat in
   between at 53 (56). Same current, same frequency, same network — different
   position, opposite behaviour.
2. **Lateral connectivity within STN and within GPe is a first-class model
   parameter, because it sets synchrony.** In Mandali 2016 the STN and GPe
   neighbourhood radii `(r_s, r_g)` are chosen so the STN–GPe system is
   desynchronised in the healthy state (Bergman 1998; Wilson & Bevan 2011), and
   two settings — `(3.3, 0.7)` versus `(1.43, 1.7)` — reproduce the two
   *contradictory* patient behaviours reported by Frank et al. 2004 and 2007.
   Nair 2022 keeps the same construction as Gaussian kernels `W_glat`/`W_slat`
   and computes an explicit synchrony measure from the STN population activity
   (0.03 in control, 0.11 at 25 % SNc loss).
3. **Antidromic activation is swept, not fitted.** Mandali 2016 delivers a
   percentage of the DBS current directly to GPe neurons and reports 10 %, 50 %
   and 75 % as three conditions, with the accuracy and reaction-time
   consequences of each.
4. **The dopaminergic state is a mechanism, not a set of weights.** Nair 2022
   goes as far as a three-compartment biochemical model of the SNc terminal —
   calcium-dependent synthesis and release, DAT reuptake, autoreceptors — plus a
   two-compartment pharmacokinetic model of oral levodopa, so that "PD" and
   "medicated" are states of a simulated dopamine system rather than
   parameter labels.

**Where BGM_22 is ahead of us.** Our 2016 model explicitly *omits* the
hyperdirect pathway: "The input from the cortex to STN, also known as the
hyper-direct pathway (Nambu, 2015), and the GABAergic projection from GPe to GPi
were not included in the model as the functional significance of these
connections is not fully understood." BGM_22 has both — `CorticalInputs` drives
`stn` directly, and `gpe_proto → snr` is in the projection table. On circuit
completeness this model is the better one, and we say so before criticising.

---

## Points

### 6.1 The STN has no spatial structure, so the electrode position cannot be
represented — and the data to represent it is already in the repository

`DBS.md`: the stimulated population is `stn:putamen`, 100 neurons, and
`_create_dbs_on_array` sets `dbs_on = 1` on a **shuffled random 40 %** of them.
There is no lattice, no coordinate, and therefore no electrode.

This is the point this seat exists to make, and in BGM_22's case it is sharper
than usual, for three reasons.

**First, the project already builds lattices.** `Microcircuit` places 1000
striatal neurons on a periodic 3D grid at 22.76 µm spacing and wires them by a
fitted distance kernel (`model_v07.md` §7.1–§7.2). The machinery for a spatially
organised STN exists in the same codebase.

**Second, the geometry is measured and sitting unused.**
`experimental_data/berlin_data/vta/sub-01/sub-01_overlap.csv` records the VTA's
overlap with the three functional STN subdivisions separately:

| subdivision | overlap (lh, rh) | subdivision volume (lh, rh) | fraction |
|---|---|---|---|
| STN motor | 35, 23 | 70, 75 | 0.40 |
| STN associative | 5, 7 | 68, 68 | 0.088 |
| STN limbic | 1, 5 | 54, 55 | 0.055 |

`get_loss.py` uses `(35 + 23) / (70 + 75)` — i.e. only the first row. So the
model's single homogeneous STN population is being given the *motor* subdivision's
coverage fraction, while the same file records that this subject's VTA also
reaches associative and limbic STN. The tripartite subdivision that our lineage
would insist on is measured, per subject, and discarded at the point of use.

(A small consistency check we could not resolve and would ask the authors to
run: `sub-01_volumes.csv` gives the VTA as 37 + 31 = 68 units while the three
overlaps sum to 58 + 12 + 6 = 76. Either the units differ between the two files
or the subdivisions overlap; one of the two files is being read in a way the
other does not support.)

**Third, the electrode is directional.**
`experimental_data/berlin_data/vta/stim_settings.csv` records sub-01 on a
Sensight lead with segmented contacts (`2a, 2b, 2c` / `3a, 3b, 3c`) — the
clinical setting is `3a, 3b` on the left and `2a, 2b, 3a, 3b` on the right. A
directional lead is the clinical realisation of exactly the variable our Fig. 7
sweeps. Representing it as an isotropic random 40 % discards the one DBS
parameter this lineage has shown to change outcomes qualitatively.

**Goal relevance: medium for the BOLD fit, high for the DBS interpretation.**
At BOLD resolution a spatially structured 40 % and a random 40 % may well give
the same signal, precisely because the STN population is homogeneous (seat 3's
point 3.3) and has no internal connectivity (point 6.2). But that is the
argument that needs making and it is nowhere made; and if it is right, then the
model cannot address electrode position at all, which should be stated where the
DBS results are.

### 6.2 No nucleus in the model has recurrent connectivity within itself

Checking the 28-projection table of `model_v07.md` §5: there is no `stn → stn`,
no `gpe_proto → gpe_proto`, no `snr → snr`, no `thal → thal`. The six GPe lateral
projections all run *between* the three GPe populations, never within one. The
striatal laterals exist, but inside the microcircuit (v07) or as three
projections (v08).

For our lineage this is the single most consequential structural omission,
because within-nucleus lateral connectivity is what sets synchrony, and synchrony
is what the STN–GPe subsystem does. In Mandali 2016 the lateral radii are what
distinguishes a desynchronised healthy STN from a synchronised parkinsonian one,
and they are what makes two patient groups behave oppositely under the same
medication. In Nair 2022 the same kernels drive the synchrony measure that tracks
SNc cell loss.

We note this is also a departure from the project's *own* lineage: Maith et al.
2021 §2.2 states "local inhibitory connections were included in the GPi, GPe, dSN
and iSN population", and its Table 5 reports GPe–GPe connection strength as one
of the significantly changed parameters in Parkinsonian models (+38.9 %,
Cohen's *d* = 1.17, *p* < .001). The predecessor had them, found them to matter,
and BGM_22 dropped them.

**Goal relevance: high.** Not because BGM_22 needs to produce synchrony — its
observable cannot see it — but because within-population inhibition is a gain
control. Without it, a population's response to its fitted drive weight is
unopposed, which is the same structural concern seat 1 raised about the FS
population and seat 3 raised about homogeneity, arriving from a third direction.

### 6.3 Antidromic strength is fitted where we sweep it, and it is confounded
with the coverage fraction

Mandali 2016 treats antidromic GPe activation as a condition to be varied
(10 %, 50 %, 75 %) and reports what each does. BGM_22 folds antidromic and
orthodromic recruitment into a single fitted `axon_spikes_per_pulse`, with the
antidromic probability on the afferent branch fixed at `np.mean(stn.dbs_on)`,
i.e. at the VTA coverage 0.4 (`DBS.md`, `_set_antidromic`).

So one fitted number carries: STN efferent orthodromic recruitment, STN somatic
antidromic invasion, and `gpe_proto`'s orthodromic drive into STN — three
mechanisms our model separates and which have different signs of effect on the
network. Seat 4 makes the identifiability version of this point; ours is
methodological: these should be *swept conditions* reported as a small table,
not a single fitted scalar, because the reader needs to know what the model does
across the range, not only at its optimum.

**Goal relevance: high for the DBS half.**

### 6.4 The dopaminergic state is a set of free weights, and the model cannot say
what dopamine did

Nair 2022 is at the far end of the spectrum from BGM_22 here — SNc terminal
biochemistry and levodopa pharmacokinetics — and we do not suggest BGM_22 go
there. But there is a large gap between a full pharmacological model and
`phi_1 = phi_2 = 0` with the dopamine terms compiled and inert
(`model_v07.md` §6.2, `TODO.md` §30).

The specific risk for this project: the striatal SPN models *have* dopamine
modulation built in (`beta_1`, `beta_2`, `alpha`, `c_da`, `E_da`) and it is
switched off, while the striatal firing rates are set to a chronically
dopamine-depleted primate's, and the weights that would express depletion are
free. Three different representations of the same physiological fact, one of
which is inert, one of which is fixed data, and one of which is fitted. A fitted
weight change can therefore be "dopamine" or "DBS" or "compensation for the
inert φ", and nothing in the design separates them.

**Goal relevance: medium.** Seats 1 and 6 converge here from different
literatures.

### 6.5 There is no behavioural or symptom-level validation, which is the only
kind our lineage trusts

Every result in both our papers is anchored to behaviour: probabilistic learning
accuracy and reaction time against Frank et al. 2004/2007 and Zaghloul et al.
2012 (Mandali 2016, Table 1 lists which conditions have experimental data and
which are predictions); movement time, peak velocity, time-to-peak velocity and
average velocity against Majsak et al. 1998 (Nair 2022, Fig. 4).

BGM_22 has no behaviour — it is a resting-state fit — and this is a deliberate
design choice we do not object to. The consequence we would want stated: the
model's DBS parameters cannot be checked against anything the patient did. In our
framework a wrong electrode position shows up as a reversed learning bias; here
a wrong DBS parameterisation shows up as nothing, because the only observable is
the BOLD correlation the parameter was fitted to.

The cheapest partial substitute we can suggest is the one seat 4 also asks for
from a different angle: the DBS frequency response. It is not behaviour, but it
is a curve the model did not see during fitting and whose shape is known.

**Goal relevance: high.** Combined with the absence of any held-out statistic
(seat 1's point 1.11), the model currently has no check that is independent of
its own loss.

### 6.6 One thing we would adopt from this project

Our DBS current is a Gaussian in an abstract lattice with `A` and `σ` chosen so
the results look right; the lattice coordinates have no anatomical referent, and
we say as much — "the four quadrants in the modules do not correspond to the
well-known basal ganglia loops like sensorimotor, associative, limbic."

BGM_22's coverage fraction comes from a segmented VTA estimate in a real
patient's imaging, against a real subdivision atlas. That is a better-grounded
number than ours, and if the spatial structure of point 6.1 were added, this
project would have something our lineage does not: a *measured* electrode
geometry rather than a schematic one. We would encourage that combination
specifically, and note that the three subjects in
`experimental_data/berlin_data/vta/` differ in stimulation frequency (125, 130
and 100 Hz), which would make the frequency question of point 6.5 answerable
within the dataset rather than only in simulation.

---

## What we would ask for before publication

1. Either give the STN spatial structure and place the electrode using the VTA
   subdivision data already in the repository, or state explicitly that electrode
   position is not represented and that the model's DBS conclusions are therefore
   position-independent (point 6.1).
2. Resolve the VTA overlap/volume unit inconsistency before the 0.4 is used
   further, and say what happens to the associative and limbic overlap
   (point 6.1).
3. Add within-population lateral inhibition to at least STN and the GPe
   populations, as the project's own predecessor had, or argue why a population
   without recurrent inhibition is adequate (point 6.2).
4. Report the DBS mechanisms as a swept table — orthodromic and antidromic
   recruitment across their range — alongside the single fitted optimum
   (point 6.3).
5. One validation the loss did not see. The frequency response is the cheapest
   candidate and the dataset itself spans three stimulation frequencies
   (points 6.5, 6.6).
