# Input streams: what they must reproduce, and what they cannot

Almost everything a v07 striatal neuron receives is synthetic. The simulated cube
holds 1000 neurons in 227.6 µm; the cortex that drives them is not simulated at
all, and the striatum around them is five times wider than the cube in every
direction. Both are replaced by **spike-count streams** drawn from a statistical
model.

This document states what those streams are required to reproduce, what they are
knowingly not able to reproduce, and therefore what the model may and may not be
used to claim. It exists because `TODO.md` §22 found the generator silently
producing a Fano factor of 1922 where 1 was intended, for four years, with
nothing in the codebase able to notice.

The values themselves live in `BOLD_optimization/parameters.py`. The mechanism is
described build-order in `model_v07.md` §7.3–§7.5. This file is the contract
between them.

---

## 1. What a stream stands in for

One stream is an `(R, n_steps)` matrix for one `(pre, post)` pair. Row `i` is
receiver neuron `i`, column `t` is one `dt = 0.1 ms` step, and the entry is a
**count**: how many of that pair's presynaptic neurons spiked into receiver `i`
in that bin.

The presynaptic neurons are never represented individually. There are no spike
trains, no individual synapses and no per-synapse weights inside a stream. At
simulation time the count is multiplied by **one scalar mean weight** for the
pair and injected as conductance:

```
count × mean_weights_by_type[(pre, post)]  →  g_glut / g_ampa / g_gaba
```

So a stream stands in for exactly this: **`N_eff` presynaptic neurons, firing at
a stated rate, whose afferent pools overlap between receivers by a stated
fraction.** Everything in §2 follows from that sentence and nothing else does.

Streams are **never combined**. Each `(pre, post)` pair gets its own `TimedArray`
and `CurrentInjection`, so a caudate dSPN receives nine separate streams — six
cortical (`glut`) and three missing-GABA (`gaba`).

---

## 2. Required — the target statistic

If `N_eff` neurons each fire independently with probability `p(t) = rate(t)·dt/1000`,
and two receivers share a fraction `f_ij` of their pools, and the presynaptic
neurons are themselves pairwise correlated at `ρ`, then three things follow with
no modelling freedom:

```
mean = N·p(t)

Fano = (1 − p) · ((1 − ρ) + N·ρ)

corr = (f·(1 − ρ) + N·ρ) / ((1 − ρ) + N·ρ)
```

**These are the target statistics.** They are not a choice — they are what the
definition arithmetically implies. Any generator standing in for that pool must
reproduce them, and `build_input_caches.py` now checks each stream against them
and raises rather than warning. The measured values are written into the cache
state file so every cache carries an audit trail of what it actually contains.

All three were confirmed against direct simulation to within 1 %:

| stream | Fano predicted / simulated | corr predicted / simulated |
|---|---|---|
| dSPN→dSPN | 7.25 / 7.23 | 0.868 / 0.868 |
| FS→dSPN | 30.91 / 31.37 | 0.976 / 0.976 |
| iSPN→iSPN | 4.77 / 4.73 | 0.802 / 0.800 |
| dlPFC→dSPN | 1.00 / 1.00 | 0.014 / 0.014 |

### The five inputs, and where each comes from

| quantity | value | source |
|---|---|---|
| `N_eff`, cortical | 7000 per SPN, 2800 per FS, split by `cortical_proportions_dict` | **derivation not yet written up — `TODO.md`** |
| `N_eff`, missing GABA | 12–1568, from `E_outer = 4πρ∫p(r)r²dr` | `fitted_params.json` kernel × del Rey et al. 2022 density |
| `rate(t)`, cortical | one value per 2.31 s TR per ROI, normalised to mean 5 Hz | the subject's own cortical BOLD, deconvolved |
| `rate`, missing GABA | FS 10.5, dSPN 25.0, iSPN 33.0 Hz | Liang et al. 2008 med-off — see `../activity_striatum/README.md` |
| `f_ij`, cortical | 0.014, flat | Kincaid et al. 1998 |
| `f_ij`, missing GABA | 0.022–0.206, distance-dependent | emerges from the same kernel and density |
| `ρ` + timescale `τ_c` | FS 0.06, SPN 0.004 | Adler et al. 2013 — **but see §5** |

**Kincaid's 0.014 is well founded and worth spelling out.** Kincaid, Zheng &
Wilson 1998 (*J Neurosci* 18:4722) report that *"each axon must contact ≤1.4 % of
all cells in its axonal arborization"*. The shared fraction between two SPNs
equals that same figure, because each of receiver A's afferents independently
contacts B with probability 0.014. The same paper also settles a worry about
`N_eff`: ~40 synapses within a dendritic volume containing 2840 spiny cells,
contacting ≤1.4 % ≈ 40 of them, i.e. **about one synapse per axon per SPN**. So
`N_eff = 7000` is correctly read as 7000 *distinct presynaptic neurons*, not 7000
synapses from a smaller number of axons.

### Correlations have a timescale, and it is not optional

`ρ` is a shared rate fluctuation. The old implementation redrew it independently
every 0.1 ms bin, which makes the spike-count correlation `r_sc` **identical at
every measurement window**. Real `r_sc` grows with the window and saturates once
the window exceeds the correlation timescale.

This matters because every measurement we calibrate against is taken at a window
of 66–3000 ms (Cohen & Kohn 2011, *Nat Neurosci* 14:811, Table 1 — the PDF is in
`../cortical_correlations/`). Nothing is measured at 0.1 ms. A white-noise `ρ`
therefore cannot be right at both ends: calibrate it at the measured window and
the fine timescale is wrong, and vice versa.

The shared fluctuation is therefore an Ornstein–Uhlenbeck process,

```
ε(t+dt) = a·ε(t) + sqrt(1 − a²)·ξ(t),        a = exp(−dt/τ_c)
```

parameterised by `(τ_c, r_sc, T_meas)`: the amplitude `σ` is solved so the
*presynaptic* pair correlation equals `r_sc` at the window `T_meas` the source
study used, via `V(T) = 2τ²(T/τ − 1 + e^(−T/τ))`. Setting `τ_c = 0` recovers the
old white behaviour exactly.

The relevant Cohen & Kohn rows for our seven ROIs:

| study | area | window | `r_sc` |
|---|---|---|---|
| Averbeck & Lee 2006 | SMA | 66 / 200 ms | 0.013 |
| Averbeck & Lee 2003 | SMA | 200 ms | 0.02 |
| Stark et al. 2008 | premotor | 400 ms | 0.02 |
| Lee et al. 1998 | motor/parietal | 1000 ms | 0.02–0.04 |
| Maynard et al. 1999 | M1 | 600 ms | 0.1–0.2 |
| Constantinidis & Goldman-Rakic 2002 | prefrontal | 3000 ms | 0.08 |

---

## 3. Why this is not a detail: the correlation sets the BOLD amplitude

`get_loss.py` builds `BoldMonitor(mapping={"I_CBF": "I_v"})` for the striatal
regions, and `I_v` is the **net** synaptic current — excitation positive,
`g_gaba·(E_gaba − v)` negative. So the simulated BOLD is driven by the summed
synaptic input current, and for `N` receivers

```
Var(Σ I_i) = N · v · (1 + (N − 1)·r)
```

With `N = 486` receivers, `r = 0` gives a variance factor of 1 and `r = 0.99`
gives **481** — a 22× larger BOLD fluctuation from an identical mean drive.

**The input correlation is the dominant determinant of the model's only output.**
It is not a second-order detail about spike statistics. This is why it cannot be
left as an unvalidated constant, and why §5 treats it as something to be measured
rather than assumed.

Note also that `Σ I_v` is precisely the quantity Helias et al. 2013
(*PLoS Comput Biol* 10:e1003428) identify as what inhibitory feedback cancels in a
real circuit: *"the cancellation of correlations between the summed inputs to
pairs of neurons"*. That cancellation is a mechanism this model cannot run — see
§4.1.

---

## 4. Deliberately simplified

Each of these is a known cost, accepted for a reason. They bound what the model
can be used to claim.

### 4.1 No feedback of any kind

A stream is precomputed. It cannot respond to what the simulated neurons do.
Three consequences:

- **No active decorrelation.** Tetzlaff et al. 2012 (*PLoS Comput Biol* 8:e1002596)
  show inhibitory feedback actively suppresses pairwise correlations *"and hence
  population-rate fluctuations"*; Tetzlaff et al. 2010 (*BMC Neurosci* 11(S1):P57)
  add that this acts *mainly below 20 Hz*, which is exactly the band a shared
  cortical fluctuation occupies. Bernacchia & Wang 2011 (*Neural Comput* 23:1732)
  develop the same mechanism specifically for *"striatum and globus pallidus"*.
  **None of it can operate here.**
- **No surround self-consistency.** The unsimulated striatum is assumed to fire at
  fixed rates and to be correlated at a fixed `ρ`, and nothing forces those to
  agree with what the simulated neurons actually do (`TODO.md` §4).
- **No DBS effect on the surround.** Only the simulated populations are
  stimulated; the 98 % of input arriving as streams is identical in DBS-on and
  DBS-off except for the cortical rate file.

**How much circuit is left:** a dSPN receives **66.8 simulated GABAergic
afferents against 2573 synthetic ones — about 2 %**. Even the FS pathway is
open-loop: only 8.8 of a dSPN's 509 FS afferents come from the 29 simulated FS
neurons.

### 4.2 One mean weight per pathway

The whole outer shell — up to 1568 neurons — collapses onto a single scalar,
while the inner shell keeps individually sampled weights. FS→SPN weights are
drawn from a mixture spanning 0–58.6, one to two orders of magnitude above
SPN→SPN. All of that heterogeneity is lost in the compensation.

### 4.3 No single-neuron temporal structure

Bins are drawn independently. There is no refractoriness, no bursting, no spike
adaptation in the presynaptic pool. **Population rhythms are representable** —
a shared modulation at 20 Hz gives beta-band correlated input — but individual
neurons bursting is not. This matters for the parkinsonian state: Raz et al. 2001
(*J Neurosci* 21:RC128) report the MPTP basal ganglia synchronises strongly, though
Deffains et al. 2016 (*eLife* 5:e16443) find beta does *not* entrain SPN spiking
even under MPTP.

### 4.4 No short-term plasticity

Corticostriatal synapses facilitate and depress. A fixed mean weight cannot.

### 4.5 No shared identity across pathways

A cortical neuron projecting to both striatum and STN is one cell in reality. In
the model the striatal cortical streams and the `CorticalInputs` streams for
thal / gpe_arky / gpe_cp / stn are **independent draws**, so the corticostriatal
and hyperdirect drives are uncorrelated. For a study about DBS, which is thought
to act partly through the hyperdirect pathway, this is a real limitation.

### 4.6 No overlap between BG receivers at all

`ci.shared_fraction_dict` is `{thal: 0.0, gpe_arky: 0.0, gpe_cp: 0.0, stn: 0.0}`,
so neighbouring STN neurons receive fully independent cortical drive. This is
certainly wrong physically; there is simply no measurement of corticosubthalamic
or corticothalamic afferent overlap to put there. It is a parameter rather than a
hardcoded zero so the assumption stays visible.

### 4.7 Cortical sharing has no distance dependence

Kincaid gives one flat 1.4 % with no geometry attached, so two SPNs 20 µm apart
share exactly as much as two 200 µm apart. Corticostriatal topography is real at
larger scales; the cube is too small for it to bite.

### 4.8 Beyond-pairwise structure, and dendrites

Only pairwise correlations are controlled — no triplet correlations, no synchrony
events. And every receiver is a point neuron, so synapse location on the dendrite
is not represented.

---

## 5. What follows for the model's claims

Put §4.1 next to the numbers: **this is not a striatal circuit model.** It is a
population of striatal neurons driven by a statistically calibrated input, with a
roughly 2 % recurrent skeleton.

That is the right instrument for the question BGM_22 actually asks — fit DBS-off,
refit DBS-on, and read off what changed in the drive. But it means:

- **Legitimate inference:** input gains, drive amplitudes, cellular and synaptic
  parameters.
- **Not legitimate:** claims about emergent circuit dynamics, about
  synchronisation arising from striatal connectivity, or about anything that
  requires the recurrent loop to be doing work.

It also means `ρ` cannot be set by citation alone. Adler et al. 2013 measured
striatal **output** correlation, and the presynaptic pool of the missing-GABA
streams consists of striatal neurons of exactly the kind being simulated — so `ρ`
is a **fixed point**, not a free input: the correlation assumed for the surround
must equal the correlation the simulated neurons produce. That is the same
self-consistency condition `TODO.md` §4 states for the firing rates, and it is
currently unchecked in both directions.

The intended procedure is therefore to **measure rather than assume**: scan the
input correlation against the simulated SPN output correlation and the simulated
BOLD amplitude, and choose from that scan, with Adler's 0.004 as a validation
target on the output. Note the warning in Baker et al. 2019
(*Phys Rev E* 99:052414): with *correlated* feedforward input, a recurrent network
produces **much larger** correlations than the asynchronous-state results would
suggest. Bernacchia & Wang predict zero-lag correlations of order `K^(−1/2)` and
longer-timescale ones of order `K^(−1)`; at `K ≈ 2640` afferents that is 0.019 and
3.8·10⁻⁴, which brackets Adler's 0.004.

---

## 6. Open

| item | where |
|---|---|
| `N_cortical_inputs_dict` = 7000 / 2800 has no written derivation | `TODO.md` |
| A larger, sparser cube would make `f(d)` vary meaningfully | `TODO.md` |
| Open-loop compensation forecloses active decorrelation | `TODO.md` |
| `f_FS_SPN` starts at 0.014 but is a lower bound | `TODO.md` |
| No number for corticosubthalamic / corticothalamic overlap | `TODO.md` |
| `ρ` and `τ_c` values await the scan described in §5 | `TODO.md` |

---

## Changing any of this

Every quantity in §2 is baked into the v07 input caches, and the cache state file
records the ones it was built with. A mismatch raises rather than silently
rebuilding. Changing any of them means rebuilding every cache — currently ~3.8 h
per DBS condition at full length.

## Sources

Read in full: none of the below — all statements above are taken from abstracts
and, for Cohen & Kohn, from Table 1 of the PDF in `../cortical_correlations/`.
Anything used for a numeric value in the model must be checked against the full
text before it is relied on.

- Kincaid AE, Zheng T, Wilson CJ (1998). Connectivity and convergence of single
  corticostriatal axons. *J Neurosci* 18:4722–4731.
- Cohen MR, Kohn A (2011). Measuring and interpreting neuronal correlations.
  *Nat Neurosci* 14:811–819. doi:10.1038/nn.2842
- Tetzlaff T, Helias M, Einevoll GT, Diesmann M (2012). Decorrelation of
  neural-network activity by inhibitory feedback. *PLoS Comput Biol* 8:e1002596.
- Tetzlaff T, Helias M, Einevoll GT, Diesmann M (2010). Decorrelation of
  low-frequency neural activity by inhibitory feedback. *BMC Neurosci* 11(S1):P57.
- Helias M, Tetzlaff T, Diesmann M (2013). The correlation structure of local
  neuronal networks intrinsically results from recurrent dynamics.
  *PLoS Comput Biol* 10:e1003428.
- Bernacchia A, Wang XJ (2011). Decorrelation by recurrent inhibition in
  heterogeneous neural circuits. *Neural Comput* 23:1732–1761.
- Baker C, Ebsch C, Lampl I, Rosenbaum R (2019). Correlated states in balanced
  neuronal networks. *Phys Rev E* 99:052414.
- van Albada SJ, Helias M, Diesmann M (2015). Scalability of asynchronous networks
  is limited by one-to-one mapping between effective connectivity and
  correlations. *PLoS Comput Biol* 11:e1004490.
- Ramanathan S, Hanley JJ, Deniau JM, Bolam JP (2002). Synaptic convergence of
  motor and somatosensory cortical afferents onto GABAergic interneurons in the
  rat striatum. *J Neurosci* 22:8158–8169.
- Choi K, Holly EN, Davatolhagh MF, Beier KT, Fuccillo MV (2018). Integrated
  anatomical and physiological mapping of striatal afferent projections.
  *Eur J Neurosci* 48:2833–2842.
- Raz A, Frechter-Mazar V, Feingold A, Abeles M, Vaadia E, Bergman H (2001).
  Activity of pallidal and striatal tonically active neurons is correlated in
  MPTP-treated monkeys but not in normal monkeys. *J Neurosci* 21:RC128.
- Deffains M, Iskhakova L, Katabi S, Haber SN, Israel Z, Bergman H (2016).
  Subthalamic, not striatal, activity correlates with basal ganglia downstream
  activity in normal and parkinsonian monkeys. *eLife* 5:e16443.

**Cited elsewhere in this project from memory and not yet verified** — do not rely
on these until checked: Renart et al. 2010 *Science* 327:587 (existence confirmed
via Tetzlaff 2010's reference list), Shadlen & Newsome 1998, de la Rocha et al.
2007, Ingham et al. 1996, Oorschot 1996.
