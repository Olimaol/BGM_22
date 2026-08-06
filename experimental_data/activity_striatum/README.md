# Striatal firing rates: where dSPN 25 Hz and iSPN 33 Hz come from

The model needs a resting firing rate for each striatal cell type. Two places
consume it, and they have to agree:

| consumer | what it does with the rate |
|---|---|
| `Microcircuit._simulate_distance_dependent_spike_counts` | draws the surrogate GABA spike trains from the striatal neurons that surround the 1000-neuron lattice but are not simulated. **Baked into the v07 input caches.** |
| `get_loss.get_firing_rate_loss` | scores the simulated population rates against a plausible band centred on the same value |

The values, set in `BOLD_optimization/parameters.py` under `mc.firing_rate_dict`
and mirrored as the `Microcircuit` default:

| cell type | rate | band used by the loss | source |
|---|---|---|---|
| dSPN | **25.0 Hz** | (12.67, 37.33) | Liang et al. 2008, Table 1 |
| iSPN | **33.0 Hz** | (21.22, 44.78) | Liang et al. 2008, Table 1 |
| FS | 10.0 Hz | (5, 15) | Yamada 2016; Marche & Apicella 2021; Adler 2013; Hernandez 2013; He 2024 |

Files here: `Liang_etal_2008_extraction - MSN.csv` is the spreadsheet transcription
of the paper's tables ([online copy](https://docs.google.com/spreadsheets/d/1FYXBhNQJZx-MvGt7IpQZi1EVFxsDKHEMx73uF5hI4pE/edit?usp=sharing)),
and `extract_from_liang_etal_2008.py` is a deconvolution that we
**do not use** — see [What we rejected](#what-we-rejected). The paper itself is not
in the repository; get it from the DOI below. Publisher PDFs in this directory are
gitignored.

---

## The source

Liang L, DeLong MR, Papa SM (2008). *Inversion of dopamine responses in striatal
medium spiny neurons and involuntary movements.* J Neurosci 28(30):7537–7547.
[doi:10.1523/JNEUROSCI.1176-08.2008](https://doi.org/10.1523/JNEUROSCI.1176-08.2008)

Two rhesus monkeys rendered parkinsonian with MPTP more than a year earlier,
chronic, severe, late-stage, with stable levodopa-induced dyskinesias. 223 units
recorded, 188 classified as MSNs. Each unit was held continuously while the animal
passed through three motor states:

- **Off** — parkinsonian disability at its worst. Oral levodopa maintenance was
  withdrawn on experiment days. This is a dopamine-denervated striatum with no
  drug on board.
- **On** — 20 min after subcutaneous levodopa, disability reversed.
- **On-with-dyskinesias** — peak of the levodopa effect, ~40 min in.

This project simulates a patient **without dopaminergic medication**, so the
**Off** state is the one we take. That also keeps the rates consistent with the SPN
equations, which run at `phi_1 = phi_2 = 0` — the dopamine-modulation terms are
fully switched off (`model_v07.md` §5).

### Why these rates are so high

Normal primate MSNs fire at 0.5–2 Hz. Liang's parkinsonian Off baselines are
20–35 Hz. That is the paper's headline finding, not an artefact: chronic
denervation raises MSN rates across the board, which the authors attribute to
compensatory and plastic changes, contrasting it with acute depletion. If a rate
of 25 Hz for a resting dSPN looks wrong, this is why.

The paper does flag one sampling concern honestly: direct-pathway neurons silenced
by depletion (<0.5–2 Hz) would not have been picked up by the electrode at all, so
the recorded sample could be biased upward. The authors argue against it — 64% of
units *increased* under levodopa, which is the D1-type response, so the direct
pathway was clearly sampled — but they cannot exclude it.

---

## What we take, and the one assumption behind it

Table 1 of the paper, Off state, n = 140 (the dyskinetic-dose experiments):

| group | share of units | baseline rate |
|---|---|---|
| Increased activity (D1) | 64% | 25 ± 1.3 Hz (SEM) |
| Decreased activity (D2) | 34% | 33 ± 1.7 Hz (SEM) |

Taken verbatim: **dSPN 25.0 Hz, iSPN 33.0 Hz.**

**The assumption:** that the *direction of the levodopa response* identifies the
receptor class — cells that speed up under dopamine are D1-bearing (direct
pathway, dSPN), cells that slow down are D2-bearing (indirect pathway, iSPN).

This is the paper's own reading, and the paper is explicit that it is an
inference rather than a measurement (p. 7542):

> Increases and decreases of MSN activity in response to dopamine are here
> considered indicative of the excitatory action on D1 receptors and the
> inhibitory action on D2 receptors, respectively. […] However, the alternative
> possibility of changes of activity in either direction depending on other
> factors than the activation of a certain receptor subtype, particularly in
> chronic pathological conditions, cannot be excluded with the available
> evidence and, particularly, the lack of data from identified neurons in
> behaving animals.

There is no cell-type identification anywhere in this dataset: no optogenetic
tagging, no antidromic identification, no peptide labelling. Anything derived from
it inherits this assumption. **It is the price of using this paper at all** — see
[What we did not consider](#what-we-did-not-consider) for the alternative.

### The bands

`get_firing_rate_loss` uses mean ± 1 SD. Table 1 reports SEM, so the SD is
recovered as SEM × √n with n = 64% and 34% of 140:

```
dSPN:  1.3 * sqrt(90) = 12.33  ->  (12.67, 37.33)
iSPN:  1.7 * sqrt(48) = 11.78  ->  (21.22, 44.78)
```

> **Caveat on what this band means.** The SD here is the spread *across individual
> monkey neurons*. The quantity being scored, `get_firing_rate_10s`, is the *mean*
> over ~430 simulated neurons — for which the corresponding statistical band would
> be ~±1.3 Hz, roughly ten times narrower. So this is a plausibility band, not a
> confidence interval, and it is deliberately loose: different species, two
> animals, and the sampling concern above. It is not a defensible statistical
> statement about the population mean, and it should not be read as one.

---

## What we rejected

`extract_from_liang_etal_2008.py` implements a mixture deconvolution. Its premise
is the paper's own arithmetic problem: 64% of units increased and 34% decreased,
but anatomy says the two pathways are about 50/50. The paper resolves this by
proposing that the increaser group contains ~14% of indirect-pathway neurons that
co-express D1 and D2 and therefore respond like D1 cells. The script takes that
literally and un-mixes them out:

1. 136 units, so true D1 = true D2 = 68.
2. Putative D1 (88 increasers) = 68 true D1 + 20 D2 cells that increased.
   Putative D2 (48) = the D2 cells that decreased.
3. Those 20 have the *decreasers'* Off baseline (33.5 Hz).
4. Their levodopa change equals D1's (+14.7 Hz).
5. Their On-state SD equals the measured decreaser SD.

Running it:

```
              putative   deconvolved
dSPN off        24.90       22.37
iSPN off        33.50       33.50
dSPN on         39.60       37.07
iSPN on         21.10       29.07
```

**Three reasons we do not use it.**

*It does not remove the assumption; it stacks on it.* The deconvolution's own
inputs are the increaser and decreaser groups. It still needs response direction
to mean receptor class, and then adds four more unverifiable assumptions on top.
It buys a contamination correction, not identified cell types.

*For the Off state it is nearly a no-op.* iSPN is unchanged — 33.50 either way —
because `mu_off_D2 = mu_off_P2` is an *assumption* of the method (step 3), not a
result of it. The only thing that moves is dSPN, 24.90 → 22.37, about 10%.

*And that one shift has undetermined sign.* It exists only because step 3 gives
the 20 contaminants the high baseline. But the paper's reading of Table 1 is that
the baseline difference **tracks the response direction** — increasers are low,
decreasers are high, which is presented as the differential effect of depletion on
the two pathways. Under that reading the contaminants, being increasers, carry
~25 Hz, and the correction is exactly zero. So the 10% is not a known bias being
removed; it is a bias whose direction the paper's own data argue against.

**Sensitivity bound:** if the contamination correction is real and step 3 is right,
dSPN is 22.37 Hz rather than 25.0. Treat 22.4–25.0 Hz as the range, and note that
25.0 is the conservative end (higher rate → more inhibition from the unsimulated
surround).

Note how much bigger this all was for the On state, which is what the code used
before: iSPN moved 21.10 → 29.07, a 38% change, driven entirely by 20 hypothetical
neurons posited to fire at 48.2 Hz. Choosing Off is what makes the method's
weakness mostly irrelevant.

---

## What we did not use

### Region-specific rates

The model runs two microcircuits, `caudate` and `putamen`, and both get the same
rates. Liang does report the regions separately for *all* units in the Off state —
putamen 28.2 ± 1.5 Hz (n = 96), caudate 23.5 ± 1.3 Hz (n = 92) — but Tables 1
and 2, the ones that split by response type, **pool the two regions**. So no
per-region dSPN/iSPN numbers exist.

They could be manufactured by scaling the pooled values by the regional ratio
(pooled high-dose Off mean 27.9 → ×1.072 putamen, ×0.922 caudate), giving dSPN
26.8/23.0 and iSPN 35.4/30.4. We don't, because a uniform scaling is the wrong
model of the regional difference: the paper reports that the *composition* differs
too — 57.5% of putamen MSNs increased versus 70% in caudate. Level and mix move
together, so splitting properly would need a per-region version of exactly the
deconvolution rejected above.

### Dose

The response-split tables cover only the dyskinetic-dose experiments (n = 140).
The nondyskinetic-dose set (n = 48) is not broken down by response type. This
costs nothing for our purpose: the Off state is the same physiological condition
in both — parkinsonian baseline with maintenance levodopa withdrawn — and the
paper reports the dose sets showed "similar distributions of changes".

### The FS rate

10 Hz does not come from Liang at all; that study excluded interneurons by design
("Units that could be classified as interneurons […] were not selected"). It comes
from a separate literature (Yamada 2016; Marche & Apicella 2021; Adler 2013;
Hernandez 2013; He 2024), and **this repository does not record whether those
recordings are from dopamine-depleted animals**. So unlike the SPN rates, FS is
not on a stated dopamine condition. The `(5, 15)` band is hand-set, not derived
from any reported SD. Logged in `TODO.md`.

---

## What we did not consider

Identified-cell-type recordings do exist — optogenetically tagged dSPN and iSPN in
6-OHDA-lesioned rodents (e.g. Sharott et al. 2017; Parker et al. 2018; Ryan et al.
2018). Those would remove the response-direction assumption entirely, which is the
one weakness no choice within Liang can fix.

They are not a drop-in replacement. Rodent MSN rates are on the order of 1–5 Hz
against Liang's 20–35 Hz, so adopting them would mean recalibrating the whole
microcircuit — connectivity weights, missing-GABA compensation, cortical drive —
not swapping two constants. If the striatal rates ever become a suspect in a bad
fit, this is the direction to look.

---

## Changing these values

They are baked into the v07 input caches: the missing-GABA spike counts are drawn
at exactly these rates. `Microcircuit._save_missing_input_state` records
`firing_rate_dict` and `correlation_dict` in the cache state, and
`_load_missing_input_state` refuses any cache whose values differ — or that predates
those fields and therefore cannot be checked. Change the rate and every v07 cache
must be rebuilt with `build_input_caches.py`.

Keep three places in sync:

- `BOLD_optimization/parameters.py` — `mc.firing_rate_dict`, the source of truth,
  threaded through `get_loss.v07_model_creation_kwargs` so evaluation and cache
  building cannot disagree
- `CompNeuroPy/src/CompNeuroPy/striatal_microcircuit/microcircuit.py` — the class
  default, for any other caller
- `BOLD_optimization/get_loss.py` — `get_firing_rate_loss`, whose `str_d1` and
  `str_d2` bands are centred on these values

`TODO.md` §4 tracks the self-consistency question this raises: the surround is
*assumed* to fire at 25/33 Hz while the simulated neurons are free to land
anywhere in the band.
