# Cortical proportions: where the caudate and putamen input mixes come from

The model splits each striatal neuron's cortical afferents across the seven
cortical ROIs the Berlin dataset provides. Those seven shares, per loop, are the
**only physical difference between the caudate and the putamen loop** — the two
loops share no projection and are otherwise identical, so every difference the
inference reads out between them traces back to this table.

> **The same table is also used for four non-striatal populations.** Everything
> derived below is corticostriatal, but `BGM_v07` passes this dict to
> `CorticalInputs` as well, so it sets the cortical mixes of `thal`,
> `gpe_arky`, `gpe_cp` and `stn` too — a putamen `stn` neuron currently draws
> 13 % of its afferents from S1, and a caudate one 55 % from dlPFC. The
> corticosubthalamic topography reported in the literature does not resemble
> these proportions, and because each region's series is normalised to mean
> 5 Hz, the mix sets the *timing* of the drive rather than its mean.
> `TODO.md` §45 carries the question of whether a separate table is
> derivable; until it is resolved, treat the non-striatal use as an
> undefended assumption.

Two places consume it, and they have to agree:

| consumer | what it does with the proportions |
|---|---|
| `Microcircuit._simulate_cor_input_spike_counts`, `CorticalInputs` | `N_eff = round(p · N_cortical_inputs)` — how many of a receiver's 7000 (SPN) or 2800 (FS) cortical afferents come from each region. A region at `p ≤ 0` gets no stream at all. **Baked into the v07 input caches.** |
| `cortical_drive_by_bold.py` | weights of the linear mix that produces `caudate_rate` / `putamen_rate` in the cortical rate `.npz`. **Only v08 is driven by these**; v07 uses the per-region series and reads `caudate_rate` for its length alone. |

The values, set in `BOLD_optimization/parameters.py` under
`cortical_proportions_dict` and defined **nowhere else**:

| region | caudate | putamen |
|---|---|---|
| dlPFC | **0.55** | **0.10** |
| preSMA | **0.15** | **0.05** |
| PMd | **0.18** | **0.11** |
| PMv | **0.04** | **0.18** |
| SMA | **0.06** | **0.15** |
| M1 | **0.02** | **0.28** |
| S1 | **0.00** | **0.13** |

Each column sums to 1. That matters: the seven rate series are each normalised to
mean 5 Hz, so a mix summing to 1 leaves the mean drive untouched and moves only
the variance and the timing. A sum off 1 would silently rescale the drive, which
is why `validate_cortical_proportions` rejects it.

No PDFs are kept here. Every source is cited by DOI below; the two that carry the
actual numbers are open access.

---

## What the quantity is

For loop *L* and cortical ROI *r*:

> the fraction of the corticostriatal afferents onto a striatal neuron in *L*
> that originate in *r*, **renormalised over just these seven ROIs**.

The renormalisation is not a detail. The striatum also receives heavily from
cingulate, insula, temporal, posterior parietal, orbital and ventrolateral
prefrontal cortex, and **none of those has an ROI in the Berlin data**. In the
macaque tracer counts below they are 25–60% of all labelled corticostriatal
cells. Renormalising means the seven ROIs we do have stand in for the entire
cortical input — so these are *relative* shares among seven regions, not absolute
shares of the cortical drive.

**What that asserts about time courses.** The omitted afferents are not
dropped; renormalising *reassigns* them to the retained series in proportion
to the retained shares. In the caudate column, where dlPFC holds 0.55, dlPFC
therefore carries most of the omitted limbic and associative input — an
afferent from rostral cingulate is given the dlPFC time course. The claim
being made is not "we ignore these regions" but "**they fluctuate, TR by TR,
like the retained ones**", and since the loss is a time-course correlation,
that is a claim about precisely the fitted quantity.

**It is asymmetric between the loops.** Reading the four wholly absent groups
off Borra 2022 Table 2 below (rostral cingulate, caudal cingulate, insula,
temporal) gives a lower bound on the omitted fraction: **≈ 47% at the caudate
head** (46.8 lateral, 46.7 medial) against 5–21% at the motor putamen sites
(20.8 dorsal, 12.8–12.9 middle, 5.4 midventral), with 36.6% at rostral
putamen and 9.6% at the motor-dominated caudate body. Rostral cingulate
alone, the largest omitted category, is 21.5–30.6% at the caudate head
against 2.6–15.3% at motor putamen. The caudate loop's drive is thus the more
heavily reconstructed of the two — and the loop contrast is what the
inference reads out. (Only a bound: Borra's groups do not map cleanly onto
our seven, since of "prefrontal" only dlPFC is retained and of "parietal"
only S1, while "motor" is retained wholesale.)

`TODO.md` §46 carries the bracketing test that would bound how much this
matters, and the pre-stated rule for whether it justifies asking Berlin for
the missing ROIs.

The seven ROIs are fixed by the experimental file: `sub-01_subdiv_results.h5`
labels them `M1, PMd, PMv, preSMA, SMA, S1, dlPFC` alongside `Cau` and `Put`.
`Cau` and `Put` are whole anatomical nuclei, which is why the derivation below
has to average over striatal subterritories rather than pick one.

---

## The sources

Human data cannot answer this. Diffusion tractography of the corticostriatal
system is almost entirely qualitative — Leh et al. 2007, Lehéricy et al. 2004 and
Draganski et al. 2008 report *topography* (which cortical area reaches which part
of the striatum), not what fraction of a neuron's input each area supplies. So
the anchor has to be quantitative macaque retrograde tracing, which is the only
method that counts labelled cells per cortical area and reports them as
percentages.

**Primary — Borra et al. 2022.** *Crossed corticostriatal projections in the
macaque brain.* J Neurosci 42(37):7060–7076.
[doi:10.1523/JNEUROSCI.0071-22.2022](https://doi.org/10.1523/JNEUROSCI.0071-22.2022)

Tracer injections at eight striatal sites; Table 2 gives the distribution of
ipsilateral labelled cells by cortical region group (% of total):

| injection | rostral cing. | prefrontal | motor | parietal | insula | temporal | caudal cing. | cells n |
|---|---|---|---|---|---|---|---|---|
| caudate, lateral head | 21.5 | 37.8 | 8.4 | 6.1 | 4.8 | 13.5 | 7.0 | 42 183 |
| caudate, medial head | 30.6 | 48.5 | 4.1 | 0.1 | 2.3 | 9.4 | 4.4 | 140 790 |
| caudate, body | 7.0 | 4.8 | 74.1 | 11.5 | 0 | 0 | 2.6 | 33 082 |
| putamen, rostral | 23.0 | 17.5 | 37.8 | 8.1 | 7.0 | 4.3 | 2.3 | 99 341 |
| putamen, dorsal motor (75r) | 15.3 | 0.7 | 61.9 | 16.6 | 1.4 | 0 | 4.1 | 119 306 |
| putamen, middle motor (77l) | 8.3 | 1.0 | 64.5 | 21.6 | 2.0 | 0.8 | 1.8 | 75 292 |
| putamen, middle motor (71r) | 9.4 | 0 | 72.1 | 15.1 | 0.5 | 0.1 | 2.8 | 8 360 |
| putamen, midventral motor (71l) | 2.6 | 0.5 | 75.5 | 18.6 | 1.2 | 0.8 | 0.8 | 36 628 |

**Primary — Borra et al. 2021.** *Laminar origin of corticostriatal projections
to the motor putamen in the macaque brain.* J Neurosci 41(7):1455–1469.
[doi:10.1523/JNEUROSCI.1475-20.2020](https://doi.org/10.1523/JNEUROSCI.1475-20.2020)

Same lab, overlapping cases. Table 3 splits that single "motor" column into
individual areas (% of total labelled cells). This is what makes a seven-region
table possible at all:

| case | 24c/d | F6 | F7 | F3 | F2 | front. operc. | F5 | F4 | F1 |
|---|---|---|---|---|---|---|---|---|---|
| 75, dorsal | 14.4 | 0.8 | 0.3 | 12.7 | 7.1 | 2.2 | 2.5 | 2.0 | 34.1 |
| 71r, middle | 14.4 | 0.8 | 0.5 | 13.2 | 6.5 | 2.4 | 7.9 | 3.3 | 26.9 |
| 71l, midventral | 2.5 | 0.1 | — | 6.9 | 0.7 | 7.6 | 33.4 | 8.2 | 18.6 |
| 61 | 12.3 | 3.7 | 1.2 | 10.6 | 9.3 | 16.5 | 10.0 | 3.0 | 2.7 |

Macaque area names map to our ROIs as **F1 = M1, F2 = PMd, F3 = SMA, F4+F5 =
PMv, F6 = preSMA, F7 = pre-PMd** (folded into PMd). The frontal operculum is
area 44/opercular cortex, closer to the inferior frontal gyrus than to the
precentral ventral bank, so it is **excluded** rather than folded into PMv.

**Supporting topography**, used only to split bins the tables leave grouped:

- Takada et al. 1998, *Exp Brain Res* 120:114–128 — M1 to lateral putamen, SMA to
  medial putamen, PMd/PMv to the dorsomedial sector.
- Inase et al. 1999, *Brain Res* 833:191–201 — preSMA to the rostral caudate and
  the striatal cell bridges, segregated rostral to the SMA zone.
- Calzavara et al. 2007, *Eur J Neurosci* 26:2005–2024
  ([doi](https://doi.org/10.1111/j.1460-9568.2007.05825.x)) — areas 9 and 46 to
  the caudate head, caudal 46 extending into rostral putamen; PMdr to dorsal and
  lateral caudate.
- Flaherty & Graybiel 1995, *J Neurophysiol* 74:2638–2648
  ([doi](https://doi.org/10.1152/jn.1995.74.6.2638)) — M1's striatal projection
  magnification is ~2× that of *each individual* S1 subarea, so summed S1 sits
  below M1 rather than beside it.

**Secondary — the one quantitative human study.** Cacciola et al. 2017, *A
connectomic analysis of the human basal ganglia network.* Front Neuroanat 11:85
([doi:10.3389/fnana.2017.00085](https://doi.org/10.3389/fnana.2017.00085)),
n = 15, constrained spherical deconvolution. It reports per-pathway connectivity
percentages, but in the Desikan-Killiany parcellation, which **cannot separate
M1/PMd/PMv** (all "precentral") or **SMA/preSMA/dlPFC** (all "superior frontal").
Only two ratios survive that coarseness, and both are used below as direction,
not as values.

---

## How the numbers were derived

Read this section as the audit trail. Each step is marked **(M)** measured and
published, **(A)** an assumption we had to make because no source resolves it, or
**(J)** a judgement call in rounding.

### Putamen

**Step 1 (M+J): weight the motor-putamen cases.** Cases 75, 71r and 71l get
weights 0.4 / 0.4 / 0.2. Cases 75 and 71r are the arm/hand sector, which occupies
most of the sensorimotor putamen; 71l is the midventral orofacial/grasping zone,
real but smaller. **(J)** Case 61 is dropped — its F1 is 2.7% against 34.1/26.9
in the others, so it is sampling a different zone. Weighted composition:

```
F6 0.66   F7 0.32   F3 11.74   F2 5.58   F.Op 3.36
F5 10.84  F4 3.76   F1 28.12   parietal 16.40   prefrontal 0.38
```

**Step 2 (A): S1's share of the parietal bin.** Borra reports "parietal" as one
number; areas 3/1/2 are not split from PE/PEc/PF/PFG/PG/AIP/SII. We take **half**
→ S1 = 8.20. This is the single least constrained assumption in the putamen
column; Flaherty & Graybiel bound it loosely from above.

**Step 3 (A): the rostral putamen's motor split.** Case 77r gives prefrontal 17.5
and motor 37.8, but no per-area breakdown. Split from the topography sources as
F1 2, F3 7, F2 8, F7 3, F5 6, F4 3, F.Op 3, F6 6 (sums to 38); S1 ≈ 2 of its 8.1
parietal.

**Step 4 (M+A): combine sensorimotor and associative putamen, 0.7 / 0.3.** A
whole-putamen ROI is mostly postcommissural sensorimotor territory with a
precommissural associative part. **(A)** The 0.7/0.3 weighting is our estimate.

```
              0.7·motor  +  0.3·rostral  =  raw     normalised
M1      0.7·28.12 + 0.3·2      = 20.28    0.3117
SMA     0.7·11.74 + 0.3·7      = 10.32    0.1586
PMd     0.7· 5.90 + 0.3·11     =  7.43    0.1142
PMv     0.7·14.60 + 0.3·9      = 12.92    0.1986
preSMA  0.7· 0.66 + 0.3·6      =  2.26    0.0348
dlPFC   0.7· 0.38 + 0.3·17.5   =  5.52    0.0848
S1      0.7· 8.20 + 0.3·2      =  6.34    0.0974
                                 ------
                                  65.07
```

**Step 5 (J): the human nudge and rounding.** Cacciola's putamen shows postcentral
4.6% against precentral 9.0% — an S1-to-precentral ratio of ~0.51, where the
macaque-derived table gives 0.097/0.62 ≈ 0.16. Human tractography over-weights
short adjacent pathways and cannot exclude passing fibres, so we do not adopt
0.51; we move S1 up by ~0.03 and take it out of M1 and PMv. dlPFC and preSMA go
up slightly for the same reason (human prefrontal expansion, below). Final:

```
M1 0.3117 -> 0.28    PMv 0.1986 -> 0.18    SMA 0.1586 -> 0.15
PMd 0.1142 -> 0.11   S1 0.0974 -> 0.13     dlPFC 0.0848 -> 0.10
preSMA 0.0348 -> 0.05
```

### Caudate

**Step 1 (M+A): weight head against body, 0.75 / 0.25.** The head is the bulk of
the nucleus. The tail has no tracer row here; its inputs are temporal and
occipital, which renormalise away entirely, so folding it into the body costs
little. Head = mean of the lateral and medial head rows.

```
prefrontal  0.75·43.15 + 0.25·4.8  = 33.56
motor       0.75· 6.25 + 0.25·74.1 = 23.21
parietal    0.75· 3.10 + 0.25·11.5 =  5.20
```

The caudate **body** row is the surprise in this dataset: 74.1% motor. That is
not M1 — the caudate body sits dorsally and receives from dorsal premotor and
area 9/46, exactly the F7/F2 territory Calzavara describes. It is why the caudate
column has a substantial PMd share and still almost no M1.

**Step 2 (A): split the caudate's motor bin.** No per-area table exists for
caudate injections, so this comes from topography: PMd 11, preSMA 7, SMA 3,
PMv 1.5, M1 0.7 (sums to 23.2). **(A)** S1 ≈ 0.2 of the 5.20 parietal — caudate
parietal input is posterior parietal (PG/PGm/7a), not areas 3/1/2.

```
        raw     normalised
dlPFC   33.56   0.5892
PMd     11.00   0.1931
preSMA   7.00   0.1229
SMA      3.00   0.0527
PMv      1.50   0.0263
M1       0.70   0.0123
S1       0.20   0.0035
        ------
         56.96
```

**Step 3 (J): correcting the prefrontal over-assignment.** Borra's "prefrontal"
bin is *all* prefrontal cortex — dorsolateral, ventrolateral and orbital. Our
dlPFC ROI is dorsolateral only, so handing it the whole 0.589 over-assigns.
Two corrections pull in opposite directions here and should be named plainly:

- assigning the entire prefrontal bin to dlPFC is **too much** (bin is broader
  than the ROI) → push down;
- the human data says the caudate is **more** prefrontal than the macaque
  suggests (Cacciola: rostral middle frontal 25.8% vs precentral 3.7%, a 7:1
  ratio against the macaque-derived 2.6:1; consistent with the reported expansion
  of prefrontal corticostriatal projections in humans, Neggers et al. 2015
  *J Neurophysiol* 113:2164–2172, Balsters et al. 2020 *eLife* 9:e53680) → push
  up.

They roughly cancel. We settled on **0.55**, slightly below the raw 0.589, with
the freed mass going to preSMA and the small motor entries. **This means the
caudate dlPFC value is not strongly determined** — anything in 0.50–0.62 is
defensible on the same evidence.

**Step 4 (J): S1 to exactly 0.** The tracer data show essentially no S1 → caudate
projection, and 0 is also load-bearing in the code: a region at `p ≤ 0` gets no
stream built at all, so the caudate keeps one cortical stream fewer per receiver
type than the putamen. Rounding 0.0035 to 0.00 rather than 0.01 is deliberate.

---

## Caveats

1. **The anchor is macaque, the model is human.** Every quantitative number above
   is from two macaque papers out of one lab. The human evidence enters only as
   the direction of two nudges, because no human study reports these fractions.
2. **The seven ROIs stand in for the whole cortex.** Cingulate input is 15–30% of
   labelled cells in most rows and simply has nowhere to go. If the cortical
   drive later gains ROIs, this table must be recomputed, not extended.
3. **Whole-nucleus ROIs average over strongly differentiated territories.** The
   caudate head is 38–49% prefrontal and the caudate body 74% motor; we collapse
   them with a weight we chose. The same applies to the putamen's motor/rostral
   split. Both weightings (**A**) are estimates, not measurements.
4. **Putamen PMv is the entry to argue about.** F5 is 2.5% and 7.9% of labelled
   cells in the two arm/hand cases but 33.4% in the midventral one, so the value
   swings with how the sectors are weighted. Defensible range **0.10–0.24**.
   Every other entry is stable to about ±0.03 under the same re-weighting.
5. **S1's share of the parietal bin is a coin-flip assumption** (Step 2). It sets
   the putamen S1 value almost single-handedly.
6. **Tracer counts are cell counts, not synapse counts.** We use them as if they
   were proportional to afferent numbers per receiver neuron. Laminar origin and
   terminal density differ across areas (that is Borra 2021's actual subject), so
   this is an approximation.
7. **The corrected proportions make the two loops more alike.**
   `corr(caudate mix, putamen mix)` rises from 0.798 under the previous hand-set
   numbers to 0.843. Since these proportions are the only physical difference
   between the loops, that shrinks the contrast the inference depends on. It is
   the honest number, not a reason to prefer the old one — but it should be known
   before reading any caudate-vs-putamen result.

---

## What this replaced, and how much it matters

Until 2026-08-06 the table was hand-set round numbers (caudate dlPFC 0.45,
preSMA 0.25, PMd 0.15, PMv 0.10, SMA 0.04, M1 0.01, S1 0.00; putamen dlPFC 0.05,
preSMA 0.10, PMd 0.15, PMv 0.05, SMA 0.25, M1 0.30, S1 0.10) with no citation,
copied into three files that could drift apart. The topography they encoded was
right — caudate associative, putamen sensorimotor — and M1 in the putamen was
almost exactly right at 0.30 against the derived 0.31. The large corrections are
**putamen PMv 0.05 → 0.18**, putamen SMA 0.25 → 0.15, caudate preSMA 0.25 → 0.15
and caudate PMv 0.10 → 0.04.

Measured effect of the change, on the real drive series:

- The seven deconvolved cortical series are only moderately correlated
  (off-diagonal *r*: 0.16 dlPFC–M1, median 0.46, 0.90 PMd–PMv), so the regions
  are genuinely distinguishable.
- But the **mixed** drive barely changes shape: `corr(old mix, new mix)` = 0.995
  (caudate) and 0.982 (putamen) for the off condition, 0.996 / 0.991 for on. The
  mean is unchanged by construction — every column sums to 1 and every series has
  mean 5 Hz. **For v08 the correction is close to a no-op in timing.**
- Its *amplitude* is a different story, and worth noting: the standard deviation
  moves +1.0% / +4.6% (off caudate / putamen) but +10.7% / **−11.7%** (on). So
  the on-condition putamen drive is now ~12% less modulated than before. Since
  the striatal input weight is a fitted parameter this is largely absorbable, but
  it is not nothing, and it is asymmetric between the two DBS conditions the
  inference compares.
- It bites in v07, where the proportions set the per-region stream sizes. Per SPN,
  out of 7000 afferents: caudate dlPFC 3150 → 3850, caudate PMv 700 → 280,
  putamen PMv 350 → 1260, putamen dlPFC 350 → 700.

---

## Changing these values

They are baked into the v07 input caches (the cortical streams are drawn at
exactly these `N_eff`) **and** into the cortical rate `.npz` (the same weights mix
`caudate_rate`/`putamen_rate`). Changing them means:

1. Edit `BOLD_optimization/parameters.py` → `cortical_proportions_dict`. **This is
   the only place they are defined.** `Microcircuit` and `CorticalInputs` have no
   defaults — they raise via `spike_input_cortex.validate_cortical_proportions()`
   if the mapping is missing, negative, or does not sum to 1.
2. Regenerate the rate file with `cortical_drive_by_bold_run.py` from
   `striatal_microcircuit_requirements/cortical_firing_rates/`. It needs
   **MATLAB with an interactive sign-in** — `matlab.engine.start_matlab()` will
   not come up in a headless shell — and it records the proportions in
   `__data_raw_meta__`. The data folder is **git-ignored, so back it up first**:
   `create_data_raw_folder` deletes it after a `y/n` prompt.
3. Rebuild every v07 input cache with `build_input_caches.py`.

Step 2 is guarded: the `.npz` files carry a `cortical_proportions_json` record and
`get_loss.infer_max_sim_time_ms` raises if it disagrees with `parameters.py`, so
code and data cannot silently diverge. Steps 1 and 3 are guarded by the cache
state check. Nothing guards step 3 being *skipped* if you also skip step 1 —
rebuild caches whenever this file changes.

See `TODO.md` §21 (resolved 2026-08-06) for the change history and `model_v07.md` §7.5 for how the
proportions are consumed inside the microcircuit.
