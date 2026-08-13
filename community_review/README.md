# Community-conventions review (TODO.md §34)

A structured survey of the active basal ganglia neurocomputational community's
conventions, run as a persona review: one review of our model per research
lineage, written in character as that lineage, plus a synthesis. The task
definition, the three steps and their checkpoints are in `TODO.md` §34.
Implementing accepted changes is not part of this survey — each accepted
proposal spawns its own `TODO.md` entry.

**Status: step 1 done — panel confirmed by Oliver on 2026-08-13 (all six
lineage seats), amended the same day with seat 7 (the experimentalist,
also confirmed). Next: step 2, the DOI reading list.**

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
outside the similarity criterion and not what this seat reviews with.)

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
against the step-2 DOI list, not from abstracts or search summaries — and
the DOI list itself will be verified against the publisher pages when it is
compiled.

## PDFs

Saved beside these documents, **untracked** (the remote is public: cite
DOIs, never commit publisher PDFs — the `experimental_data/` pattern).
