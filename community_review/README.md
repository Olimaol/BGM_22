# Community-conventions review (TODO.md §34)

A structured survey of the active basal ganglia neurocomputational community's
conventions, run as a persona review: one review of our model per research
lineage, written in character as that lineage, plus a synthesis. The task
definition, the three steps and their checkpoints are in `TODO.md` §34.
Implementing accepted changes is not part of this survey — each accepted
proposal spawns its own `TODO.md` entry.

**Status: all three steps done (2026-08-13); step 3 rerun as round 2 on
2026-08-14.** The seven reviews and the synthesis of the first round are in
`round1/`; every one was written from the full texts of the PDFs saved here,
not from abstracts. The round-2 documents are being written into `round2/`
under two additional requirements recorded in `TODO.md` §34. The survey is
complete and `TODO.md` §34 is resolved. What remains is the **triage** — the Roadmap item
that follows §34 — which decides which of the synthesis's 24 findings are
accepted; each accepted proposal spawns its own `TODO.md` entry.

## The documents

| file | contents |
|---|---|
| `round1/review_seat1_kumar_hellgren_kotaleski.md` | seat 1, 11 points |
| `round1/review_seat2_rubin_verstynen.md` | seat 2, 10 points |
| `round1/review_seat3_girard_doya.md` | seat 3, 8 points |
| `round1/review_seat4_grill.md` | seat 4 (DBS standing), 9 points |
| `round1/review_seat5_hamker.md` | seat 5, our own lineage, 11 points |
| `round1/review_seat6_chakravarthy.md` | seat 6, 6 points |
| `round1/review_seat7_experimentalist.md` | seat 7, 16 points, species-tagged |
| `round1/synthesis.md` | 24 merged findings F1–F24, ranked in four tiers by cross-seat convergence, one change proposal each |

Every point in every review carries a goal-relevance tag; seat 7's structural
claims additionally carry a species-provenance tag and a within-reading-list
convergence statement, per the §34 amendment.

Two facts the reviews established that the project did not know, recorded here
because they are referenced from several documents:

- **The GPe BOLD pooling factors (0.5 / 0.17 / 0.10) are GPe cell-type
  abundances** — PV⁺ 50 %, arkypallidal 18 %, cortex-projecting 12 %, with the
  missing 0.23 being ChAT⁺ and unclassified cells the model does not contain
  (seat 7, from Courtney et al. 2023). `model_v07.md` §3.6's own conjecture was
  right; the factors are no longer unsourced.
- **The seven subcortical delays in `parameters.csv` are Kumaravelu et al. 2016
  Table 1 exactly** — rat values with real provenance the project does not
  record (seat 4).

### Step-2 history

The PDFs were downloaded by Oliver and saved here on 2026-08-13, and the set was
verified complete against the list (27 valid PDFs: 26 papers + the Bahuguna
correction). At the checkpoint Oliver amended the list in three places —
Hjorth et al. 2020 added to seat 1, Giossi et al. 2024 added to seat 2, Meier
et al. 2022 removed from seat 5 — each recorded with rationale at the entry it
touches.

## Selection criteria (from §34)

- The unit is a **lineage** sharing one modeling approach, not an individual.
- Primary ranking: **similarity to our approach** (mesoscopic, populations of
  point neurons, possibly spiking, multiple functionally connected BG regions)
  weighted with citation impact. Detailed single-cell/morphology modeling is
  dissimilar regardless of citations.
- At least 1–2 lineages that model **DBS at the network level**, so the
  mechanisms of `TODO.md` §14–§17 get a reviewer with standing.
- The Hamker/Chemnitz lineage is one persona, judged against its own
  published standards.
- **Strict recency:** lineages and papers from roughly 2016 onward. Accepted
  consequence: personas are reconstructed from recent work only; conventions
  stated in foundational papers (Terman & Rubin 2002/2004, Humphries 2006)
  and silently assumed since may be missed.
- **No seat for BOLD/whole-brain (mean-field) lineages** — they fail the
  approach-similarity criterion. Consequence, recorded here as a known
  limitation: **the BOLD pipeline of this project has no peer reviewer on
  this panel.** No persona will audit the balloon model, the BOLD-signal
  source assumptions, or the fit-to-fMRI methodology; the panel reviews the
  neuronal model that feeds it.
- **One deliberate exception to all of the above (added 2026-08-13): seat 7,
  the experimentalist** — not a lineage and not a modeling approach, but a
  review-defined composite persona representing the recent experimental
  literature on basal ganglia structure, connectivity and organization. See
  its section below for the full definition; the amendment is logged in
  `TODO.md` §34.

## Proposed panel

Seats 1–6 ordered by approach similarity; seat 7 sits outside that
ordering by design. Paper pointers below are provisional orientation only
— the reading list is fixed in step 2.

### 1. Kumar–Hellgren Kotaleski lineage (KTH Stockholm / Freiburg)

The closest match to our approach in the field: spiking point-neuron
populations of the full basal ganglia with biologically constrained
connectivity, simulated in health and dopamine-depleted states, validated
against in-vivo firing rates and oscillation phenomena. Lindahl &
Hellgren Kotaleski (2016, eNeuro) is the flagship; the Bahuguna/Kumar line of
STN–GPe beta-oscillation papers (through ~2020, PLOS Comput Biol) carries it
forward. High citation impact in exactly our niche. (The lineage's separate
detailed striatal microcircuit work — Hjorth et al., multi-compartment — is
outside the similarity criterion for *seat selection*; Oliver nonetheless
added Hjorth et al. 2020 to the reading list at the download checkpoint,
since it is this lineage's own striatal-microcircuit standard and v07's
striatum is built by `Microcircuit` — see the step-2 list.)

### 2. Rubin lineage (Pittsburgh / CMU, with Verstynen) — **DBS standing, panel seed**

Jonathan Rubin qualifies on both §34 counts: the recent Rubin–Verstynen
cortico-basal-ganglia-thalamic (CBGT) line builds biologically constrained
spiking networks of the full loop (dSPN/iSPN/FSI striatum, GPe, STN, GPi,
thalamus, cortex; e.g. Dunovan et al. 2019, the CBGTPy framework paper, and
successors through 2025), and Rubin's parkinsonian/DBS network modeling
lineage is the community's reference point for network-level DBS mechanisms.
Counts once as one persona.

### 3. Girard–Doya lineage (ISIR Sorbonne Université / OIST)

Biologically constrained spiking model of the primate basal ganglia with
parameters fitted under quantitative anatomical constraints (Girard, Liénard,
Gutierrez, Delord & Doya 2021, EJN — the spiking successor of the
Liénard–Girard population model), reproducing rest firing rates per nucleus.
The OIST side adds a spiking subthalamo-pallidal model of parkinsonian
oscillations with DBS exploration (Shouno, Tachibana, Nambu & Doya 2017,
Front Neuroanat). Methodologically the nearest neighbour to our
"fit a constrained spiking BG network to data" program.

### 4. Grill lineage (Duke) — **DBS standing**

The cortex–basal-ganglia–thalamus network model of the 6-OHDA rat
(Kumaravelu, Brocker & Grill 2016, J Comput Neurosci): small populations of
single-compartment conductance-based point neurons across striatum, STN,
GPe, GPi, thalamus and cortex, calibrated to a parkinsonian animal model,
used to study STN-DBS frequency dependence, and since adopted widely as the
evaluation platform for closed-loop DBS controllers. More biophysical per
neuron than we are, but network-level and DBS-centred — this seat reviews
our DBS representation (`DBS.md`, §14–§17 mechanisms) with the most
standing.

### 5. Hamker lineage (TU Chemnitz) — **our own lineage**

Included per §34 to judge the model against its own published standards:
spiking basal ganglia networks in ANNarchy fitted to human data, including
the direct predecessors of this project (the Maith et al. parkinsonian
BG models fitted to fMRI, the Baladron/Villagrasa BG-loop models). The
persona asks: does this model meet the bar the lab itself set in print?

### 6. Chakravarthy lineage (IIT Madras) — *optional sixth seat*

Long-running spiking (Izhikevich) basal ganglia network program with
explicit STN-DBS simulation, including medication + DBS effects on behaviour
(e.g. Muralidharan et al. 2016, Front Hum Neurosci; book-length synthesis
since). In-window and approach-similar, but lower citation impact than seats
1–5 and flagship papers sit at the 2015/2016 boundary. Included as the only
non-US/EU lineage and a third DBS-capable voice; Oliver confirmed the seat
on 2026-08-13.

### 7. The experimentalist — *review-defined composite seat* (added 2026-08-13)

**Why the seat exists.** Seats 1–6 are all modeling lineages, so the panel
would only catch what the modeling community already models. Experimental
findings the modeling literature has not yet absorbed — most prominently
the GPe reorganization of roughly 2015–2024 (arkypallidal/prototypic cell
types, pallido-striatal projections, bridging collaterals), directly
relevant since v07 already contains `gpe_arky`/`gpe_cp` — would otherwise
be invisible to the review.

**What the persona is.** A deliberate, recorded exception to the
one-seat-one-lineage rule: no single experimental lab covers whole-BG
structure, connectivity and organization (the GPe work, striatal cell
types, and primate circuit anatomy are different lineages). The persona is
therefore defined entirely by its step-2 reading list; its voice is what
those reviews *collectively assert*, and — unlike the lineage seats — it
has no single lab's published standard to be judged in character against.

**Mandate.** Structure, connectivity, and organization of the basal
ganglia, **plus** auditing the model's empirical validation anchors — the
Liang et al. 2008 medication-off firing-rate bands
(`experimental_data/activity_striatum/README.md`) and the
macaque-tracer-based cortical proportions
(`experimental_data/cortical_proportions/README.md`) — which are
experimental claims no modeling seat audits. The DBS representation is
**out of scope** for this seat; it stays with the Grill and Rubin seats.

**Reading list rules (extend step 2).**

- **4–6 reviews** — above the 2–4 per-seat norm, justified because the
  seat covers an entire literature rather than one lab's output.
- Same **2016+** window as the rest of the panel, preferring the most
  recent authoritative synthesis per topic.
- **Two mandatory slots:** at least one dedicated GPe review and at least
  one whole-BG circuit-organization review. The remaining slots are chosen
  at step 2 against what the model actually contains (striatal cell types,
  cortico-BG anatomy/proportions, STN, firing-rate physiology).
- **Reviews-first with a narrow escape hatch:** a primary paper may take a
  slot only where step 2 finds no in-window review covering a mandated
  topic; the substitution and the failed search are recorded here.

**Step-3 consequences.**

- **Synthesis weighting:** the convergence rule (many personas = community
  convention, one voice = one lab's taste) is amended for this seat. A
  point raised only by the experimentalist is weighted by convergence
  *within its own reading list*: asserted by multiple independent reviews →
  treated as literature consensus, comparable to multi-persona
  convergence; found in a single review → one voice. The synthesis states
  which case applies for each such point.
- **Species-provenance tags:** every structural claim in the
  experimentalist's review carries a species tag (mouse / rat / macaque /
  human) beside the goal-relevance tag — the recent GPe literature is
  overwhelmingly mouse, while this model is a human-subject fit anchored
  on macaque tracer data.

## Step 2 — the reading list (DOIs)

Compiled 2026-08-13, amended by Oliver at the download checkpoint the
same day (three changes, marked at the entries they touch). Per §34: the
flagship network-model paper of the recent era plus the most recent
relevant one per lineage seat, more only where the lineage's approach
shifted; 4–6 reviews for seat 7. All entries are in-window (2016+).
**26 papers total.** Every DOI was verified against Crossref and/or the
publisher page on 2026-08-13; no entry is cited from memory.

### Seat 1 — Kumar–Hellgren Kotaleski (4)

1. Lindahl M, Hellgren Kotaleski J (2016). *Untangling basal ganglia
   network dynamics and function: role of dopamine depletion and inhibition
   investigated in a spiking network model.* eNeuro 3(6).
   DOI: `10.1523/ENEURO.0156-16.2016` — the flagship: full spiking BG,
   healthy vs dopamine-depleted, validated on in-vivo rates and oscillations.
2. Bahuguna J, Sahasranamam A, Kumar A (2020). *Uncoupling the roles of
   firing rates and spike bursts in shaping the STN-GPe beta band
   oscillations.* PLOS Comput Biol 16(3): e1007748.
   DOI: `10.1371/journal.pcbi.1007748` — the STN–GPe beta line. (A 2025
   correction exists, `10.1371/journal.pcbi.1013638` — download both.)
3. Chakravarty K, Roy S, Sinha A, Nambu A, Chiken S, Hellgren Kotaleski J,
   Kumar A (2022). *Transient response of basal ganglia network in healthy
   and low-dopamine state.* eNeuro 9(2).
   DOI: `10.1523/ENEURO.0376-21.2022` — the most recent full-BG paper with
   both lineage PIs, validated against Nambu-lab cortical-stimulation data.
4. Hjorth JJJ, Kozlov A, Carannante I, et al. (2020). *The microcircuits of
   striatum in silico.* PNAS 117(17):9554–9565.
   DOI: `10.1073/pnas.2000671117` — **added by Oliver at the download
   checkpoint**: the lineage's own striatal-microcircuit standard, directly
   relevant to auditing v07's `Microcircuit` striatum even though its
   multi-compartment approach sits outside the seat-selection similarity
   criterion (verified against Crossref 2026-08-13).

### Seat 2 — Rubin/Verstynen (5 — the shift from parkinsonian-dynamics networks to the CBGT decision framework justifies the fourth slot; the fifth was added at the checkpoint)

1. Corbit VL, Whalen TC, Zitelli KT, Crilly SY, Rubin JE, Gittis AH (2016).
   *Pallidostriatal projections promote β oscillations in a dopamine-depleted
   biophysical network model.* J Neurosci 36(20).
   DOI: `10.1523/JNEUROSCI.0339-16.2016` — the in-window flagship of the
   parkinsonian-network line; directly on our `gpe_arky`/`gpe_cp` content.
2. Rubin JE (2017). *Computational models of basal ganglia dysfunction: the
   dynamics is in the details.* Curr Opin Neurobiol 46:127–135.
   DOI: `10.1016/j.conb.2017.08.011` — the persona's stated standards for
   PD/DBS network models; carries the seat's DBS standing in-window.
3. Dunovan K, Vich C, Clapp M, Verstynen T, Rubin J (2019). *Reward-driven
   changes in striatal pathway competition shape evidence evaluation in
   decision-making.* PLOS Comput Biol 15(5): e1006998.
   DOI: `10.1371/journal.pcbi.1006998` — the CBGT-line flagship.
4. Clapp M, Bahuguna J, Giossi C, Rubin JE, Verstynen T, Vich C (2025).
   *CBGTPy: an extensible cortico-basal ganglia-thalamic framework for
   modeling biological decision making.* PLOS ONE 20(1): e0310367.
   DOI: `10.1371/journal.pone.0310367` — the most recent: the lineage's
   published framework, i.e. what it considers a complete CBGT network.
5. Giossi C, Rubin JE, Gittis A, Verstynen T, Vich C (2024). *Rethinking
   the external globus pallidus and information flow in cortico-basal
   ganglia-thalamic circuits.* Eur J Neurosci 60(10).
   DOI: `10.1111/ejn.16348` — **added by Oliver at the download
   checkpoint**: not a modeling study, but its findings on GPe information
   flow should still be taken into account; it sits in this seat (not
   seat 7) because it is authored by this lineage and articulates its
   perspective — see "Considered for the list, not selected" below.

### Seat 3 — Girard–Doya (3)

1. Girard B, Liénard J, Gutierrez CE, Delord B, Doya K (2021). *A
   biologically constrained spiking neural network model of the primate
   basal ganglia with overlapping pathways exhibits action selection.*
   Eur J Neurosci 53(7):2254–2277. DOI: `10.1111/ejn.14869` — the flagship:
   parameter fitting under quantitative anatomical constraints, primate.
2. Shouno O, Tachibana Y, Nambu A, Doya K (2017). *Computational model of
   recurrent subthalamo-pallidal circuit for generation of parkinsonian
   oscillations.* Front Neuroanat 11:21. DOI: `10.3389/fnana.2017.00021` —
   the OIST spiking STN–GPe model with DBS exploration.
3. Liénard J, et al. (2024). *Estimation of the transmission delays in the
   basal ganglia of the macaque monkey and subsequent predictions about
   oscillatory activity under dopamine depletion.* Eur J Neurosci.
   DOI: `10.1111/ejn.16271` — the most recent: the same lineage's
   data-constrained fitting program carried forward.

### Seat 4 — Grill (3)

1. Kumaravelu K, Brocker DT, Grill WM (2016). *A biophysical model of the
   cortex-basal ganglia-thalamus network in the 6-OHDA lesioned rat model
   of Parkinson's disease.* J Comput Neurosci 40(2):207–229.
   DOI: `10.1007/s10827-016-0593-9` — the flagship and the community's
   closed-loop-DBS evaluation platform.
2. Kumaravelu K, Oza CS, Behrend CE, Grill WM (2018). *Model-based
   deconstruction of cortical evoked potentials generated by subthalamic
   nucleus deep brain stimulation.* J Neurophysiol 120(2):662–680.
   DOI: `10.1152/jn.00862.2017` — how this lineage represents DBS at the
   pathway level (orthodromic/antidromic decomposition); the sharpest
   standard to hold `DBS.md`'s mechanisms against.
3. Su F, Kumaravelu K, Wang J, Grill WM (2019). *Model-based evaluation of
   closed-loop deep brain stimulation controller to adapt to dynamic
   changes in reference signal.* Front Neurosci 13:956.
   DOI: `10.3389/fnins.2019.00956` — the most recent network-level use of
   the model in its DBS-controller role.

### Seat 5 — Hamker, our own lineage (3 — a fourth slot was compiled but removed at the checkpoint, see below)

1. Schroll H, Hamker FH (2016). *Basal ganglia dysfunctions in movement
   disorders: what can be learned from computational simulations.*
   Mov Disord 31(11). DOI: `10.1002/mds.26719` — the lab's own stated
   evaluative standards for BG models of disease.
2. Maith O, Villagrasa Escudero F, Dinkelbach HÜ, Baladron J, Horn A,
   Irmen F, Kühn AA, Hamker FH (2021). *A computational model-based
   analysis of basal ganglia pathway changes in Parkinson's disease
   inferred from resting-state fMRI.* Eur J Neurosci 53(7):2133–2153.
   DOI: `10.1111/ejn.14868` — the direct predecessor of this project:
   spiking BG fitted to human resting-state fMRI, patient data.
3. Goenner L, Maith O, Koulouri I, Baladron J, Hamker FH (2021). *A spiking
   model of basal ganglia dynamics in stopping behavior supported by
   arkypallidal neurons.* Eur J Neurosci 53(7):2296–2321.
   DOI: `10.1111/ejn.15082` — the lab's own bar for modeling
   arkypallidal/prototypic GPe, which v07 inherits.
A fourth slot — Meier JM, et al. (2022), *Virtual deep brain stimulation:
multiscale co-simulation of a spiking basal ganglia model and a
whole-brain mean-field model with The Virtual Brain,* Exp Neurol 354
(`10.1016/j.expneurol.2022.114111`) — was compiled but **removed by
Oliver at the download checkpoint**: it simulates BOLD via TVB, a
different approach from ours; its DBS implementation is simpler than
ours, leaving nothing further to draw inspiration from; and its BG model
is the same as Maith et al. (2021), already slot 2. Consequence: the
partial from-inside-the-panel audit of the simulated-BOLD/DBS side that
this slot would have provided is gone, so the
**BOLD-pipeline-has-no-peer-reviewer limitation** in the selection
criteria above now holds without mitigation. Meier et al. 2022 remains
noted under "Considered and excluded" as the nearest published precedent
of our BOLD pipeline, for the synthesis to cite.

### Seat 6 — Chakravarthy (2)

1. Mandali A, Chakravarthy VS (2016). *Probing the role of medication, DBS
   electrode position, and antidromic activation on impulsivity using a
   computational model of basal ganglia.* Front Hum Neurosci 10:450.
   DOI: `10.3389/fnhum.2016.00450` — the in-window flagship: Izhikevich
   spiking BG with explicit STN-DBS including antidromic effects. (This
   corrects the provisional pointer above: the 2016 Frontiers DBS paper is
   Mandali & Chakravarthy, not Muralidharan et al.)
2. Nair SS, Muddapu VR, Chakravarthy VS (2022). *A multiscale,
   systems-level, neuropharmacological model of cortico-basal ganglia
   system for arm reaching under normal, parkinsonian, and levodopa
   medication conditions.* Front Comput Neurosci 15:756881.
   DOI: `10.3389/fncom.2021.756881` — the most recent: the lineage's
   current multiscale cortico-BG program including medication state.

### Seat 7 — the experimentalist (6 reviews)

All slots were filled by in-window reviews — the primary-paper escape
hatch was **not needed**. Dominant species noted per entry (the review
itself will tag every claim, per the seat definition).

1. **[mandatory GPe slot]** Courtney CD, Pamukcu A, Chan CS (2023). *Cell
   and circuit complexity of the external globus pallidus.* Nat Neurosci
   26:1147–1159. DOI: `10.1038/s41593-023-01368-7` — the most recent
   authoritative synthesis of the GPe reorganization
   (arkypallidal/prototypic, pallido-striatal projections, bridging
   collaterals); overwhelmingly mouse.
2. **[mandatory whole-BG organization slot]** McGregor MM, Nelson AB
   (2019). *Circuit mechanisms of Parkinson's disease.* Neuron
   101(6):1042–1056. DOI: `10.1016/j.neuron.2019.03.004` — whole-BG
   circuit organization read through the disease state the model actually
   fits; largely rodent, with primate/human anchors.
3. Tepper JM, Koós T, Ibáñez-Sandoval O, Tecuapetla F, Faust TW, Assous M
   (2018). *Heterogeneity and diversity of striatal GABAergic
   interneurons: update 2018.* Front Neuroanat 12:91.
   DOI: `10.3389/fnana.2018.00091` — audits the microcircuit's cell-type
   choices (dSPN/iSPN/FS and what is deliberately absent); rodent.
4. Wichmann T (2019). *Changing views of the pathophysiology of
   parkinsonism.* Mov Disord 34(8). DOI: `10.1002/mds.27741` — firing
   rates vs patterns in parkinsonism, nonhuman primate and human; the
   direct audit anchor for the Liang et al. 2008 firing-rate bands
   (`experimental_data/activity_striatum/README.md`).
5. Haber SN (2016). *Corticostriatal circuitry.* Dialogues Clin Neurosci
   18(1):7–21. DOI: `10.31887/DCNS.2016.18.1/shaber` — primate
   corticostriatal topography from the field's authority; the audit anchor
   for the Borra-tracer-based cortical proportions
   (`experimental_data/cortical_proportions/README.md`); macaque/human.
6. Emmi A, Antonini A, Macchi V, Porzionato A, De Caro R (2020). *Anatomy
   and connectivity of the subthalamic nucleus in humans and non-human
   primates.* Front Neuroanat 14:13. DOI: `10.3389/fnana.2020.00013` —
   STN afferents/efferents and internal organization in the species that
   matter for a human-subject fit; human/macaque.

### Considered for the list, not selected

- **Giossi C, Rubin JE, Gittis A, Verstynen T, Vich C (2024),** *Rethinking
  the external globus pallidus and information flow in cortico-basal
  ganglia-thalamic circuits,* Eur J Neurosci (`10.1111/ejn.16348`) — a GPe
  review, but authored by the seat-2 modeling lineage; seat 7 exists
  precisely to hear the experimental literature unfiltered by modelers,
  and seat 2's persona already carries this perspective. *Amended at the
  download checkpoint:* the seat-7 rejection stands, but Oliver added the
  paper to **seat 2's** list instead — its findings should be taken into
  account, and there they inform the very persona that authored them.
- **Hegeman DJ, Hong ES, Hernández VM, Chan CS (2016),** *The external
  globus pallidus: progress and perspectives,* Eur J Neurosci
  (`10.1111/ejn.13196`) — the classic dedicated GPe review; superseded for
  our purpose by the same group's 2023 Nature Neuroscience synthesis
  (rule: most recent authoritative synthesis per topic).
- **Whalen/Gittis delta-oscillation and SNr papers (Rubin co-authored,
  2020–2024)** — primary experimental papers; the seat-2 selection keeps
  to network models per §34.

## Considered and excluded

- **Bogacz lineage (Oxford)** — high-impact network-level DBS theory
  (beta-oscillation generation, phase-locked/adaptive DBS), but the models
  are Wilson–Cowan / coupled-oscillator / mean-field, not populations of
  point neurons: fails the primary similarity criterion the same way the
  whole-brain groups do. Strongest overrule candidate if a fourth DBS voice
  is wanted.
- **Rubchinsky lineage (Indiana)** — network-level DBS and synchrony, but the
  in-window output centres on temporal patterning of synchrony in data and
  small conductance-based motifs; thin on full-BG network models since 2016.
- **Humphries lineage (Nottingham)** — the canonical spiking BG and striatum
  models predate the 2016 recency cut; no in-window full-network successor
  found. Excluded by the recency rule (an accepted consequence recorded
  in §34).
- **McIntyre lineage (Cleveland)** — DBS modeling with maximal citation
  impact, but biophysical axon/field models: detailed-morphology,
  dissimilar regardless of citations.
- **Whole-brain / BOLD lineages (e.g. Ritter's TVB group, Berlin)** — excluded
  by design (see the limitation above). Noted for the synthesis: the
  "virtual DBS" TVB co-simulation work (Meier et al. 2022, Exp Neurol)
  embeds the Hamker-lineage spiking BG model in a whole-brain model, so the
  nearest thing to a published peer precedent for our BOLD pipeline lives
  outside this panel.

## Verification note

Panel candidates were checked against web searches on 2026-08-13 (lineage
activity and in-window network-model output); no full texts have been read
yet. Reviews (step 3) will be written **only** from PDFs Oliver downloads
against the step-2 DOI list, not from abstracts or search summaries.

The step-2 DOI list was verified on 2026-08-13: every DOI was resolved
through the Crossref API and/or the publisher page (Wiley, PLOS, Frontiers,
Elsevier/Cell, Springer, SfN, Nature) and matched against title, authors and
journal. The lineage publication records behind the per-seat selections were
taken from PubMed author listings (Rubin JE, Grill WM, Hamker FH,
Chakravarthy VS), not from memory. One provisional pointer from step 1 was
corrected in the process (seat 6: Mandali & Chakravarthy 2016, previously
remembered as "Muralidharan et al. 2016").

## PDFs

Saved beside these documents, **untracked** (the remote is public: cite
DOIs, never commit publisher PDFs — the `experimental_data/` pattern).
