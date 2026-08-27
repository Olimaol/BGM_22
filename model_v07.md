# How the v07 model is created

Step-by-step account of what happens between `setup()` and `compile()` when
`get_loss.py --model-version v07` builds the model, and of where every number in
the compiled network comes from. Read alongside `CLAUDE.md` (orientation),
`TODO.md` and `DBS.md` (the DBS mechanism, which this document only
places in the sequence). `model_v08.md` describes the reduced model as a delta
against this one.

Standard ANNarchy usage — declaring a `Neuron`, a `Population`, a `Projection` —
is not explained here. What is explained is everything this project does on top
of that: the split between the `BGM` class and `parameters.csv`, the striatal
`Microcircuit`, the `CorticalInputs` drive, the two-loop naming scheme, and the
several places where a value is set *outside* the ANNarchy objects entirely.

---

## 1. Orientation

**Both BG loops live in one ANNarchy network and are compiled exactly once.**
`get_loss.py` constructs a separate `BGM` object per loop in a plain
`for loop in LOOPS:`, but ANNarchy has a single global network, so the second
construction simply adds more populations next to the first. The single
`model_dict["caudate"].compile()` at the end compiles everything, the putamen
loop included. `caudate` and `putamen` share no projection; they meet only when
their populations are pooled into the shared GPi/GPe/STN BOLD monitors, and only
the putamen loop is stimulated by DBS.

**Two naming schemes coexist, and the reason is which code created the object.**
Everything created by `BGM_v07` is named plainly (`stn`, `snr__thal`) and then
gets `:caudate` / `:putamen` appended afterwards by `BGM._add_name_appendix()`.
Everything created by `Microcircuit` or `CorticalInputs` already carries the loop
name inside it (`caudate_dSPN`, `TimedInput_dlPFC_dSPN_caudate`,
`Proj_FS_dSPN_caudate`) because those classes are constructed with `name=loop`,
and is deliberately **excluded** from the renaming pass. `get_loss.py →
population_name()` is the helper that hides the difference from calling code:
striatal compartments in v07 resolve to `f"{loop}_{dSPN|iSPN|FS}"`, everything
else to `f"{compartment}:{loop}"`.

The consequence worth remembering: the logical labels `str_d1`, `str_d2`,
`str_fsi` still exist in v07 — but only as *projection name fragments*. There is
no population called `str_d1:caudate` in v07; the projection `str_d1__snr:caudate`
runs from the population `caudate_dSPN`.

---

## 2. Prerequisites

Everything the model needs before a single ANNarchy object exists.

| artefact | consumed by | contents | produced by |
|---|---|---|---|
| `CompNeuroPy/.../bgm_22/parameters.csv`, column `BGM_v07_p01` | `BGM._get_params()`, at `BGM.__init__` time | population sizes, Izhikevich parameters, noise, and per-projection connectivity/number/weights/delays | hand-maintained |
| `BOLD_optimization/parameters.py` | `get_loss.py` throughout | `dt`, seed, durations, `update_time`, lattice size, cache dir, the two data paths, and every stream parameter baked into the caches (the full list is the §3.3 table) | hand-maintained |
| `striatal_microcircuit_requirements/connectivity_parameters/connectivity_fit_data/fitted_params.json` | `Microcircuit.__init__` | 7 pre→post pairs, each an amplitude and a σ in µm | `connectivity_fit_run.py` (skopt `gp_minimize`), BGM_22 commit `a59a5eb` |
| `striatal_microcircuit_requirements/cortical_firing_rates/cortical_firing_rates_data/firing_rates_matlab_condition-{on,off}.npz` | `Microcircuit`, `CorticalInputs`, and `get_loss.infer_max_sim_time_ms` | 9 regions × `{_time, _rate}` (310 values each) plus the `cortical_proportions_json` record | `cortical_drive_by_bold_run.py` (SPM HRF deconvolution, needs MATLAB), BGM_22 commit `1fe78dc` (regenerated 2026-08-06 for the current proportions) |
| `<cache-dir>/{mc,ci}_<loop>_cache_<dbs>/` | `Microcircuit`, `CorticalInputs` | pickled state + raw memmaps of precomputed spike counts | `build_input_caches.py` |

Both data directories carry a `__data_raw_meta__` file recording the producing
script, the BGM_22 commit and the full conda environment. The generating
pipelines are not described here — see the scripts.

### `fitted_params.json`

The complete table, since it fixes the entire intrinsic striatal connectivity:

| pre → post | amplitude `P0` | σ (µm) |
|---|---|---|
| dSPN → dSPN | 0.10838 | 400.33 |
| dSPN → iSPN | 0.10321 | 400.91 |
| iSPN → dSPN | 0.14061 | 255.35 |
| iSPN → iSPN | 0.13847 | 313.68 |
| FS → dSPN | 0.59877 | 394.22 |
| FS → iSPN | 0.91820 | 139.99 |
| FS → FS | 0.15242 | 189.16 |

**There is no `dSPN → FS` or `iSPN → FS` entry, so SPNs never project onto FS
neurons.** The pair set of this file is the only place the local connection types
are declared: the local projection list, the weight matrices and the compensation
streams are all built by iterating these keys, and the cache refuses to load if
the key set changed. No count is hardcoded — add a row and everything downstream
follows.

### The cortical rate file

Keys are `<region>_time` and `<region>_rate` for `M1, PMd, PMv, preSMA, SMA, S1,
dlPFC, caudate, putamen`, each 310 float64 values. `_time` is in **seconds** and
spaced 2.31 s apart — one TR. `_rate` is in Hz. A 19th array,
`cortical_proportions_json`, records the proportion weights the striatal mixes
were built with; `get_loss.infer_max_sim_time_ms` compares it against
`parameters.py` and **raises** on a mismatch (older files that lack the key only
warn). The same folder holds a `firing_rates_scipy_*` variant from a second
deconvolution route; nothing uses it.

The seven cortical series drive the model. The `caudate_rate` / `putamen_rate`
series are the anatomically mixed striatal drive; v07 does not use them for
input at all, only `get_loss.infer_max_sim_time_ms()` reads `caudate_rate` to
derive the run length (`310 × 2310 ms = 716 100 ms`, or `n_trs × 2310 ms` under
`--n-trs`).

**The drive is one value per TR, far coarser than `dt`.** Both `Microcircuit` and
`CorticalInputs` handle this by inferring the drive's own `dt` from
`<region>_time[1] - <region>_time[0]`, requiring it to be an integer multiple of
the model `dt`, and `np.repeat`-ing each value that many times (23 100× at
`dt = 0.1 ms`). A drive finer than `dt`, or one whose spacing is not an integer
multiple, is rejected.

---

## 3. The creation sequence

All of this is the `if __name__ == "__main__":` block of
`BOLD_optimization/get_loss.py`, in order.

### 3.1 `setup(dt, seed)`

`dt = 0.1 ms`, `seed = 42` from `parameters.py`. **Nothing may exist before
this** — ANNarchy's timestep and RNG seed are global and are baked into every
object created afterwards.

### 3.2 Infer the run duration

`infer_max_sim_time_ms()` opens the condition's rate `.npz`, takes the length of
`caudate_rate`, and returns `n_TRs × 2310 ms` plus the mixed striatal rates.
`--n-trs` truncates. The resulting `paramsS["t.duration"]` is handed to
`Microcircuit` and `CorticalInputs` as `T`, and **this is the number that has to
match the cache**: both classes compute `n_steps = int(T/dt)` and refuse a cache
whose stored `n_steps` differs by even one step.

### 3.3 Build the model creation kwargs

`get_loss.v07_model_creation_kwargs()` assembles the dict that `BGM_v07` reads.
It exists so that `build_input_caches.py` can import the *same* function and
build a cache that an evaluation is guaranteed to accept.

| key | value | note |
|---|---|---|
| `build_mc`, `build_ci` | `False` for evaluations | `True` only in `build_input_caches.py` |
| `mc.name` | `caudate` / `putamen` | names every mc/ci object; validated against `("caudate", "putamen")`. The proportion mix arrives separately, below |
| `mc.nx`, `mc.b` | 10, 10 | 1000 lattice sites |
| `dbs` | `on` / `off` | the rate file and cache dirs are resolved from `dbs_condition` into the separate keys below; this key is forwarded only to be validated against the cache state — see `DBS.md` |
| `timestep` | 0.1 | |
| `t.duration` | inferred above | |
| `update_time` | 110.0 | the chunk size the inputs are streamed in — see §7.7 |
| `mc.storage_dir`, `ci.storage_dir` | `<cache-dir>/{mc,ci}_<loop>_cache_<dbs>` | |
| `mc.seed`, `ci.seed` | 42 | numpy RNG, independent of ANNarchy's |
| `mc.fitted_params_path`, `mc.cortical_rate_path` | the two data files | |
| `mc.cortical_proportions_dict` | the loop's column of `cortical_proportions_dict` | the only physical difference between the loops (§7.5) |
| `mc.firing_rate_dict` | `{FS: 10.5, dSPN: 25.0, iSPN: 33.0}` | surround rates, baked into the caches (§7.3) |
| `mc.shared_fraction` | 0.014 | Kincaid; fixes the axon pool size (§7.5) |
| `mc.correlation_dict`, `mc.cortical_correlation` | all 0.0 | pending the TODO §25 scan (§7.4) |
| `mc.correlation_window_ms`, `mc.correlation_timescale_ms` | `None`, 0.0 | required as soon as any correlation is non-zero |
| `mc.source_multiplicity` | 10 | virtual-source coarsening for the missing-GABA pools (§7.4 1b) |
| `ci.shared_fraction_dict` | all 0.0 | per-population overlap for the BG streams (§8) |
| `ci.n_thal` / `n_gpe_arky` / `n_gpe_cp` / `n_stn` | 1000 / 500 / 500 / 500 | cortical afferents per receiver neuron |

One number the table does **not** contain: `N_cortical_inputs_dict`
(`{FS: 2800, dSPN: 7000, iSPN: 7000}`) is a `Microcircuit` class default, not a
`parameters.py` entry — `v07_model_creation_kwargs` never passes it, so the
default is what runs (derivation still unwritten, `TODO.md` §23).

### 3.4 `BGM(...)` per loop

```python
BGM(name="BGM_v07_p01", model_creation_kwargs=..., seed=42,
    compile_folder_name=..., name_appendix=loop,
    do_create=True, do_compile=False)
```

`BGM.__init__` validates the `BGM_v*_p*` name form, derives
`_model_version_name = "BGM_v07"`, and **reads `parameters.csv` immediately** —
before any ANNarchy object exists. `do_create=True` then runs `BGM.create()`,
which is the whole of §4 and §5. `do_compile=False` is essential: parameters
must be in place *before* compile, because a parameter set afterwards does not
survive `reset()` (see §11).

### 3.5 DBS retrofit

`add_dbs_mechanisms(populations=…, projections=…)` rewrites the equations of the
6 putamen BG populations and 6 putamen projections in place, and
`DBSstimulator(auto_implement=False)` creates the `pulse()` function and the
`dbs_pulse_*` constants. In the **on condition only** (`if dbs_condition ==
"on"`), `dbs_stimulator.on()` runs here, **before** compile, so the on-state
becomes the compile-time state that every later `reset()` restores.

Both conditions run the retrofit and construct the stimulator; only the guard on
`on()` differs. `DBS.md` has the complete account, including why
`auto_implement=True` cannot be used, why the stimulator must exist in the off
condition too, and what the added terms are. Measured consequence, from the
reports of §10: off and on compile the *same network*, differing only in the
values of `dbs_on`, `dbs_depolarization`, `antidromic`, `antidromic_prob`,
`prob_axon_spike` and the projections' `p_axon_spike_trans`.

### 3.6 BOLD monitors

Seven `BoldMonitor`s, one per experimental ROI, each pooling one or more
populations with `mapping={"I_CBF": input_var}` and `normalize_input=2000`:

| ROI | pooled populations | scale factors |
|---|---|---|
| GPi | `snr` × both loops | — |
| GPe | `gpe_proto`, `gpe_arky`, `gpe_cp` × both loops | 0.5 / 0.17 / 0.10, normalized |
| STN | `stn` × both loops | — |
| Cau | `caudate_dSPN`, `caudate_iSPN`, `caudate_FS` | — (v08 only, see below) |
| Put | `putamen_dSPN`, `putamen_iSPN`, `putamen_FS` | same |
| MD | `thal:caudate` | — |
| VAp | `thal:putamen` | — |

A `—` means no `scale_factor` is passed, so `BoldMonitor` falls back to weighting
each population by its share of the pooled neuron count (the size-share fallback
in `BoldMonitor.__init__`).

**The two loops therefore enter GPi, GPe and STN at 50/50.** For GPi and STN
that is the size-share fallback on two 100-neuron populations; for GPe the
passed factors are the same cell-type abundances in both loops, so the loop
split is 50/50 there too. Nothing chose that ratio — it follows from
`stn.size = 100`. The subject's own territory volumes make the motor
territory ≈ 37 % of the STN, so the DBS-carrying loop is over-weighted by
≈ 1.34, a factor the fitted DBS parameters would absorb. `TODO.md` §38 carries
the accepted remedy and the arithmetic, including the 8.8 % associative VTA
overlap that the caudate-loop exclusion sets to zero.

**No striatal scale factors in v07.** `Microcircuit` already sizes dSPN/iSPN/FS by
the del Rey et al. (2022) proportions (`microcircuit.py → props_delRey`), so the
size-proportional default *is* the del Rey weighting and passing it explicitly
would only restate it. The explicit `props_delRey` factors are therefore applied
in v08 only, where all three striatal populations have 100 neurons and the
default would weight them equally. The two are not bit-identical: the default
uses the integer cell counts after `int()` truncation and the remainder fix-up
(the type assignment in `Microcircuit.__init__`), which at `nx = b = 10` gives
486 / 485 / 29 → 0.486 / 0.485 / 0.029 against the exact
0.485327 / 0.485327 / 0.029345.

**The GPe factors are GPe cell-type abundances** — identified by the
community-conventions review of 2026-08-13
(`community_review/round1/review_seat7_experimentalist.md` point 7.1 and
`community_review/round1/synthesis.md` F6), against Courtney, Pamukcu & Chan 2023
(*Nat Neurosci* 26:1147, `10.1038/s41593-023-01368-7`): PV⁺ ≈ 50 %,
NPAS1⁺FOXP2⁺ (arkypallidal) ≈ 18 %, NPAS1⁺NKX2.1⁺ (cortex-projecting) ≈ 12 %,
with ChAT⁺ ≈ 5 % and unclassified ≈ 15 % making up the missing 0.23 — exactly
the GPe cell types the model does not contain. That confirms the conjecture
this paragraph previously recorded. The numbers are mouse molecular work and
Wichmann 2019 states that their translation to the primate GPe is untested.

They still appear only as `gpe_proportions` in `get_loss.py` and in
`test_microcircuit_bgm.py`, both introduced whole in commit `76d3867` and
uncited *in the code* — the citation above has yet to be written there. These
do change the pooling, since `gpe_proto/arky/cp` are all 100 neurons. Note the
normalization runs over all **six** pooled populations at once (both loops,
factors summing to 1.54), so each `gpe_proto` effectively weighs ≈ 0.325 — not
0.5, and not 0.5/0.77.

The open question is now a different one: the model **weights the BOLD** by
these realistic abundances while **simulating** the three populations at 100
neurons each, so the network dynamics carry three times as many arkypallidal
neurons relative to prototypic ones as the abundances imply. See
`community_review/round2/synthesis.md` F22 for the proposal (the round-1
pointer was F6).

**`gpe_cp`'s BOLD weight and its projections come from two different
identities.** The 0.10 above is Courtney's NPAS1⁺NKX2.1⁺ abundance, a class
reported to project to midbrain, cortex and thalamic reticular nucleus and
*not* to the striatum — while this model's `gpe_cp` projects to all three
striatal populations. `TODO.md` §40 carries the audit of the GPe three-way
split and owes this section the statement of what `gpe_cp` denotes, which
experimental population it is meant to be, and the claim boundary that
follows from its namesake cortical efferent being structurally absent here.

**The input variable differs by population family.** BGM populations expose the
total current as `I`; the Humphries striatal populations expose it as `I_v`. Cau
and Put therefore map `I_CBF` to `I_v`, everything else to `I`. One subtlety: for
the three GPe populations the quantity entering `dv/dt` is `f(I, nonlin)` (§6.1),
but the monitor maps the **raw** `I` — the BOLD signal sees the uncompressed
input current. That is deliberate and follows the same split as `I_base` below:
`nonlin` belongs to the *neuron model*, which is everything from input current
to spikes, while the neurovascular drive is meant to represent the *synaptic
input* that produced the current. ANNarchy defines both in one equation block,
but they are separate things.

**This mapping is scheduled to change** (`TODO.md` §42, decided 2026-08-27).
`I` is the signed *net* current, so inhibitory input enters negatively and
cancels against excitation, whereas the intended drive is non-negative synaptic
input. The drive is to be taken from the synaptic conductances instead — with
the raw-versus-stabilized-factor choice, the treatment of `g_nmda`, and the
excitatory/inhibitory weighting still open there. Under that mapping the
exclusions described here and below follow by construction rather than from
what `I` happens to contain.

**`I_base` is deliberately outside `I`.** The BGM populations add
`I_base = base_mean + offset_base` on the `dv/dt` line, not inside `I`, so the
BOLD monitor never sees it. That is the intended split: `I_base` belongs to the
neuron model *without* synaptic input, and the monitor should only see what the
synaptic input contributes. It does mean the fitted baseline currents of `snr` and
`gpe_proto` — the only two populations whose entire drive is `base_mean`, since
they receive no cortical input — reach the BOLD signal only through their effect
on the network. The DBS somatic term sits on the same line and is likewise unseen
(`DBS.md`).

### 3.7 `compile()`

One call, both loops. See §10.

---

## 4. `BGM.create()`: from `parameters.csv` to ANNarchy objects

`BGM_v07` itself sets **no weight and no connectivity**, and almost no parameter:
it instantiates bare `Population` and `Projection` objects with names, reading
from the CSV only the six `*.size` values it needs to construct the populations,
and forwarding the `model_creation_kwargs` of §3.3 to `Microcircuit` and
`CorticalInputs`. Everything else numeric arrives afterwards, from the CSV. The sequence in `bgm.py → BGM.create()`
is:

1. `CompNeuroModel.create(do_compile=False)` → runs `BGM_v07` (§5), then diffs
   the global ANNarchy network before and after to discover which populations and
   projections belong to this model.
2. `BGM._add_name_appendix()` → appends `:caudate` / `:putamen` to every
   population name, projection name **and parameter key**, skipping anything in
   `_components_created_by_mc_ci` and anything under `general.`.
3. `BGM._set_params()` → population attributes.
4. `BGM._set_noise_values()` → the `*_noise` attributes, handled separately.
5. `BGM._set_connections()` → projection connectivity, then remaining projection
   attributes.
6. `compile()` if asked — here it is not.

### The CSV key syntax

Keys are `<compartment>.<attribute>`. The compartment part is matched
**literally** against `self.populations` / `self.projections`, which is why
projection `name=` strings in `model_creation_functions.py` must be exactly
`pre__post`: `str_d1__snr.weights` only finds its projection because that
projection is named `str_d1__snr`.

Value parsing in `BGM._get_params()`: rows whose first cell contains `###`
(a substring test, not a prefix) are section headers and skipped; **empty cells
are skipped**, which is how the v07 column omits `str_d1.size` and the
intra-striatal projection rows; a value is an `int`
if integral, else a `float`; a value wrapped in `$…$` is `eval`'d as a formula;
anything else stays a string.

### `_set_params()`

Walks every parameter key. Skips compartments created by mc/ci. Skips any
attribute whose name ends in `_noise`. **Strips a trailing `_init`**, so
`stn.v_init = -69.42` sets `v`. Distribution strings (`Uniform(...)`,
`Normal(...)`, …) are rewritten to `ann_Random.*` and evaluated into an array of
the population's shape. Setting `base_mean` additionally sets `I_base` to the
same value.

The guard that matters: it only sets the attribute **if `param_name` is in the
population's `attributes`**. This is deliberate — assigning an attribute a
Population does not have is not an error in ANNarchy, it silently becomes a plain
Python attribute and the intended effect vanishes.

### `_set_connections()`

Two passes.

The first pass reads `<proj>.connectivity`, which is a **method name as a
string** — one of `connect_fixed_number_pre`, `connect_all_to_all`,
`connect_one_to_one`, `connect_fixed_probability`. It then inspects that method's
signature and collects every `<proj>.<argname>` key that matches a parameter of
it, so `str_d1__snr.number`, `.weights` and `.delays` become the call
`proj.connect_fixed_number_pre(number=10, weights=0.06, delays=4)`. A projection
with no `.connectivity` row aborts the run with a message. Projections created by
mc/ci and projections whose name starts with `TimedInput` are skipped.

The second pass sets any remaining projection attribute that was not already
consumed as a connectivity argument.

### Why mc/ci components are excluded from all of it

`BGM._model_creation_function()` collects
`mc.get_model_component_names()` and `ci.get_model_component_names()` into
`_components_created_by_mc_ci`. Every one of the four passes above checks that
list first. Without it, the renaming pass would turn `caudate_dSPN` into
`caudate_dSPN:caudate`, and `_set_connections` would demand a `.connectivity` row
for `Proj_FS_dSPN_caudate` and quit.

---

## 5. Structure: what `BGM_v07` creates

In order (`model_creation_functions.py → BGM_v07`):

1. `Microcircuit(...)` is constructed and `mc.create_model()` is called — so the
   three striatal populations and all their machinery exist **first**.
2. The three returned populations are bound to the local names `str_d1`,
   `str_d2`, `str_fsi` (they are `caudate_dSPN`, `caudate_iSPN`, `caudate_FS`).
3. The six BG populations are created.
4. The 28 projections are created — bare, no connectivity.
5. `CorticalInputs(...)` is constructed over `[thal, gpe_arky, gpe_cp, stn]` and
   `ci.create_model()` is called. It comes **last** because it takes the
   population objects themselves as arguments, so they must already exist.
6. `return mc, ci`.

### Populations

| population | neuron model | size | source of size |
|---|---|---|---|
| `<loop>_dSPN` | `Izhikevich2007Humphries2009SPND1` | 486 | lattice type assignment |
| `<loop>_iSPN` | `Izhikevich2007Humphries2009SPND2` | 485 | lattice type assignment |
| `<loop>_FS` | `Izhikevich2007Humphries2009FSI` | 29 | lattice type assignment |
| `stn:<loop>` | `Izhikevich2003NoisyBaseNonlin(stabilize=True, use_nonlin=False)` | 100 | `stn.size` |
| `snr:<loop>` | same | 100 | `snr.size` |
| `gpe_proto:<loop>` | `Izhikevich2003NoisyBaseNonlin(stabilize=True, use_nonlin=True)` | 100 | `gpe_proto.size` |
| `gpe_arky:<loop>` | same | 100 | `gpe_arky.size` |
| `gpe_cp:<loop>` | same | 100 | `gpe_cp.size` |
| `thal:<loop>` | `Izhikevich2003NoisyBaseNonlin(stabilize=True, use_nonlin=False)` | 100 | `thal.size` |

**The striatal sizes are not parameters.** The `str_d1.size` / `str_d2.size` /
`str_fsi.size` rows exist in the CSV (v08 uses them), but their `BGM_v07_p01`
cells are deliberately empty and empty cells are skipped (§4); those counts fall
out of the lattice instead (§7.1).

`exp_input` is left at its default `0.0` for every v07 population, so **no v07
neuron carries the `Exponential(lambda)` drive term**. The cortical drive arrives
entirely through `CurrentInjection`s instead. This is the sharpest structural
difference from v08.

### The 28 projections

All created with `connect_fixed_number_pre(number=10)`; the table gives the
target, the CSV weight and the CSV delay in ms. The `w` column is later
multiplied by an optimized per-cluster factor (§11), and the cluster each
projection belongs to is given in the last column.

**Where the delays come from.** The community-conventions review of 2026-08-13
established that the seven distinct subcortical delays below match Kumaravelu,
Brocker & Grill 2016 (*J Comput Neurosci* 40:207, `10.1007/s10827-016-0593-9`)
Table 1 **exactly** — `str→snr` 4, `str→gpe` 5, `stn→snr` 1.5, `stn→gpe` 2,
`gpe→stn` 4, `gpe→snr` 3, `snr→thal` 5 — i.e. that lineage's **rat** delay set,
sourced there to Nakanishi 1987, Kita & Kitai 1991, Fujimoto & Kita 1993 and Xu
2008 (`community_review/round1/review_seat4_grill.md` point 4.1). Nothing in this
project recorded that. Two caveats the review attaches: they are rat values in a
human fit, and Liénard et al. 2024's macaque estimates disagree by up to a
factor of two on the striatofugal pathways
(`community_review/round1/review_seat3_girard_doya.md` point 3.2). Separately, the
cortical drive arrives with **no delay at all** — `CurrentInjection` injects
into the current timestep (§7.7) — so the hyperdirect route has no speed
advantage over the trans-striatal one.

| pre → post | target | weight | delay | opt. cluster |
|---|---|---|---|---|
| str_d1 → snr | gaba | 0.06 | 4 | `str_d1__bg` |
| str_d1 → gpe_cp | gaba | 0.005 | 5 | `str_d1__bg` |
| str_d2 → gpe_proto | gaba | 0.04 | 5 | `str_d2__bg` |
| str_d2 → gpe_arky | gaba | 0.08 | 5 | `str_d2__bg` |
| str_d2 → gpe_cp | gaba | 0.08 | 5 | `str_d2__bg` |
| stn → snr | ampa | 0.04 | 1.5 | `stn__snr` |
| stn → gpe_proto | ampa | 0.001 | 2 | `stn__gpe` |
| stn → gpe_arky | ampa | 0.001 | 2 | `stn__gpe` |
| stn → gpe_cp | ampa | 0.001 | 2 | `stn__gpe` |
| gpe_proto → stn | gaba | 0.001 | 4 | `gpe_proto__stn` |
| gpe_proto → snr | gaba | 0.015 | 3 | `gpe_proto__snr` |
| gpe_proto → gpe_arky | gaba | 0.025 | 4 | `gpe_laterals` |
| gpe_proto → gpe_cp | gaba | 0.025 | 4 | `gpe_laterals` |
| gpe_proto → str_fsi | gaba | 1.6 | 5 | `gpe_striatum` |
| gpe_arky → str_d1 | gaba | 3.25 | 5 | `gpe_striatum` |
| gpe_arky → str_d2 | gaba | 6 | 5 | `gpe_striatum` |
| gpe_arky → str_fsi | gaba | 6.4 | 5 | `gpe_striatum` |
| gpe_arky → gpe_proto | gaba | 0.008 | 4 | `gpe_laterals` |
| gpe_arky → gpe_cp | gaba | 0.008 | 5 | `gpe_laterals` |
| gpe_cp → str_d1 | gaba | 0.5 | 5 | `gpe_striatum` |
| gpe_cp → str_d2 | gaba | 0.5 | 5 | `gpe_striatum` |
| gpe_cp → str_fsi | gaba | 0.8 | 5 | `gpe_striatum` |
| gpe_cp → gpe_proto | gaba | 0.008 | 4 | `gpe_laterals` |
| gpe_cp → gpe_arky | gaba | 0.008 | 5 | `gpe_laterals` |
| snr → thal | gaba | 0.06 | 5 | `snr__thal` |
| thal → str_d1 | glut | 7 | 4 | `thal__striatum` |
| thal → str_d2 | glut | 6 | 4 | `thal__striatum` |
| thal → str_fsi | ampa | 9.6 | 4 | `thal__striatum` |

**`thal → str_fsi` targets `ampa` while `thal → str_d1/str_d2` target `glut`.**
That is not a slip: the FSI neuron model has no `g_glut` term at all (§6.3), so
`glut` would have nowhere to go.

**The seven intra-striatal projections are absent** — they live inside the
Microcircuit as distance-dependent, individually-weighted connections (§7.2),
which is exactly what v07 exists to do.

---

## 6. Neuron models and equations

Equation strings are given as they are assembled by the neuron-model classes.
Parameter tables give the **effective compiled value**; entries marked † are
overridden by `parameters.csv` after creation, all others are class defaults
interpolated into the parameter block of the neuron definition. (The only things
baked into the equation *string* itself are the fixed `-50` driving force, the
choice between `I` and `f(I, nonlin)`, and the presence or absence of the
`exp_input` line.) The two `*_noise` values take a separate path —
`_set_noise_values`, not `_set_params` — and setting `base_mean` also writes
`I_base` (§4); the † marks cover both routes.

### 6.1 `Izhikevich2003NoisyBaseNonlin` — the six BG populations

Instantiated with `stabilize=True` for all six, `use_nonlin=True` for the three
GPe populations only, `exp_input=0.0` for all. The class name is CompNeuroPy's;
in a `report()` this model appears under its ANNarchy name
`Izhikevich2003_noisy_I_nonlin`.

```
dg_ampa/dt  = -g_ampa/tau_ampa
dg_gaba/dt  = -g_gaba / tau_gaba
offset_base = ite(Uniform(0.0, 1.0) * 1000.0 / dt > rate_base_noise, offset_base, Normal(0, 1) * base_noise)
I_base      = base_mean + offset_base
I           = I_app - g_ampa*(-50) / (1 + g_ampa * dt) - g_gaba*(v - E_gaba) / (1 + g_gaba * dt)
dv/dt       = n2 * v * v + n1 * v + n0 - u + <ext_current> + I_base
du/dt       = a * (b * v - u)
```

with spike condition `v >= 30` and reset `v = c; u = u + d`.

`<ext_current>` is `f(I, nonlin)` for the three GPe populations and plain `I` for
`stn`, `snr`, `thal`, where

```
f(x,y) = ((abs(x))**(1/y)) / ((x + 1e-20)/(abs(x) + 1e-20))
```

is `sign(x) · |x|^(1/nonlin)` written without a `sign` function: the divisor is
`x/|x|`. With the fitted `nonlin ≈ 1.24` the exponent is ≈ 0.81, so the GPe
populations see a mildly compressed input current, and `nonlin = 1` reduces it to
the identity.

Three things `stabilize=True` does, all visible in the `I` line. It makes
excitation **current-based** with a fixed 50 mV driving force (`-g_ampa*(-50)`
rather than `-g_ampa*(v - E_ampa)`) — which also leaves `E_ampa` declared but
unused, like `lambda` below. It divides each conductance term by `1 + g*dt`; that
divisor is what bounds the effective conductance, and it is where the optimizer's
weight ceilings come from (§11). And it drops the `neg()`/`pos()` rectifiers the
non-stabilized line wraps around both terms
(`I = I_app - neg(g_ampa*(v - E_ampa)) - pos(g_gaba*(v - E_gaba))`) — so nothing
clips the GABA term any more, and at `v < E_gaba = -70` it turns depolarizing.

The `offset_base` line is a resampled current offset: at every step a uniform
draw decides, with probability `rate_base_noise * dt / 1000`, to redraw the
offset from `Normal(0, base_noise)`; otherwise it is held. At
`rate_base_noise = 100` Hz that is a fresh draw about every 10 ms.

| parameter | stn | snr | gpe_proto | gpe_arky | gpe_cp | thal |
|---|---|---|---|---|---|---|
| `a` † | 0.005 | 0.005 | 0.0048944 | 0.0078397 | 0.0048944 | 0.02 |
| `b` † | 0.265 | 0.585 | 0.2300634 | 0.2440538 | 0.2300634 | 0.2 |
| `c` † | −65 | −65 | −67.23776 | −63.44135 | −67.23776 | −65 |
| `d` † | 2 | 4 | 4.9173033 | 11.531786 | 4.9173033 | 6 |
| `n2` † | 0.04 | 0.04 | 0.0505032 | 0.0457251 | 0.0505032 | 0.04 |
| `n1` † | 5 | 5 | 3.3423141 | 3.4175567 | 3.3423141 | 5 |
| `n0` † | 140 | 140 | 63.594189 | 56.643875 | 63.594189 | 140 |
| `tau_ampa` † | 2 | 2 | 2 | 2 | 2 | 2 |
| `tau_gaba` † | 10 | 10 | 10 | 10 | 10 | 10 |
| `E_ampa` † | 0 | 0 | 0 | 0 | 0 | 0 |
| `E_gaba` † | −70 | −70 | −70 | −70 | −70 | −70 |
| `I_app` † | 0 | 0 | 0 | 0 | 0 | **5** |
| `base_mean` † | 0 | 0 | 0 | 0 | 0 | 0 |
| `base_noise` † | 5 | 5 | 5 | 5 | 5 | 5 |
| `rate_base_noise` † | 100 | 100 | 100 | 100 | 100 | 100 |
| `nonlin` | 1.0 | 1.0 | 1.2421189 † | 1.2537563 † | 1.2421189 † | 1.0 |
| `lambda` (`exp_input`) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `exp_input_weight` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| `v` init † | −69.42 | −69.9 | −67 | −63 | −67 | −73.89 |
| `u` init † | −14.34 | −14.08 | −14 | −15 | −14 | −6.1 |

`base_mean` is 0 here only because it is the optimizer's job: `snr` and
`gpe_proto` receive no cortical drive at all, so their excitation *is* their
baseline current, fitted as parameters 7 and 8 (§11).

**`lambda` and `exp_input_weight` are declared but unused.** The constructor
always writes both into the parameter block — `lambda` holds the `exp_input`
argument, `exp_input_weight` is hard-coded to 1.0 — but the line that consumes
them,

```
exp_input  = Exponential(lambda) * exp_input_weight * g_cor
dg_ampa/dt = -g_ampa/tau_ampa + exp_input / dt
```

is only emitted when `exp_input > 0.0`. With `exp_input = 0.0` for every v07
population (§5) that line is absent, which is why neither parameter appears in
the equations above; they sit in the compiled model as inert constants. It's used
for cortical input in **v08** (see `model_v08.md` §4). v07 delivers cortical input
through `CurrentInjection`s instead.

### 6.2 `Izhikevich2007Humphries2009SPND1` / `SPND2` — dSPN and iSPN

Both instantiated by `Microcircuit.create_populations_annarchy()` with
`current_based_excitation=True` and everything else at defaults — so
`exp_input = 0.0`, `params_for_pop = False`, `phi_1 = phi_2 = 0.0`.

Shared synapse block:

```
dg_ampa/dt = -g_ampa / tau_ampa + g_glut / dt
dg_nmda/dt = -g_nmda / tau_nmda + g_glut / dt
dg_gaba/dt = -g_gaba / tau_gaba
B_nmda     = 1 / (1 + 0.28 * exp(-0.062 * v))
```

`g_glut` is the reason both dSPN and iSPN take `glut` as their excitatory target:
one incoming target feeds AMPA and NMDA at once. `B_nmda` is the standard
magnesium block.

dSPN current and voltage:

```
I_v       = g_ampa * E_exc / (1 + g_ampa * dt / C)
          + g_nmda * B_nmda * E_exc * (1 + beta_1 * phi_1) / (1 + g_nmda * dt / C)
          + g_gaba * (E_gaba - v) / (1 + g_gaba * dt / C)
          + I_app
C * dv/dt = k*(v - v_r)*(v - v_t) - u + I_v + phi_1 * c_da * (v - E_da)
du/dt     = a*(b*(v - v_r) - u)
```

iSPN differs in three places — the AMPA term carries the D2 attenuation factor,
the NMDA term **lacks** the `(1 + beta_1 * phi_1)` factor (`beta_1` is not even
declared in `SPND2`), and the quadratic term is scaled:

```
I_v       = g_ampa * E_exc * (1 - beta_2 * phi_2) / (1 + g_ampa * dt / C)
          + g_nmda * B_nmda * E_exc / (1 + g_nmda * dt / C)
          + g_gaba * (E_gaba - v) / (1 + g_gaba * dt / C)
          + I_app
C * dv/dt = k * (1 - alpha * phi_2) * (v - v_r) * (v - v_t) - u + I_v
```

Both: spike `v >= v_peak`, reset `v = c; u = u + d`.

**With `phi_1 = phi_2 = 0` every dopamine term above is inert** — `phi_1*c_da*(…)`
is zero, `(1 + beta_1*phi_1)` and `(1 - beta_2*phi_2)` are one. The dopamine
machinery is present in the equations but not used by this project. Whether 0 is
the right value for a PD patient off medication is unchecked against the source
paper — `TODO.md` §30.

| parameter | dSPN | iSPN | note |
|---|---|---|---|
| `tau_ampa` / `tau_nmda` / `tau_gaba` | 6 / 160 / 4 | 6 / 160 / 4 | ms |
| `E_ampa` / `E_nmda` / `E_gaba` | 0 / 0 / −60 | 0 / 0 / −60 | mV; `E_ampa`/`E_nmda` inert — `current_based_excitation=True` puts `E_exc` in their place |
| `E_exc` | 50 | 50 | fixed driving force, from `current_based_excitation=True` |
| `C` | 50 | 50 | pF |
| `k` | 1.14 | 1.14 | |
| `v_r` / `v_t` / `v_peak` | −80 / −33.8 / 40 | −80 / −33.8 / 40 | mV |
| `a` / `b` / `c` / `d` | 0.05 / −20 / −55 / 377 | same | |
| `c_da` / `E_da` | 22.7 / −68.4 | — | |
| `phi_1` / `beta_1` | 0.0 / 3.75 | — | |
| `phi_2` / `alpha` / `beta_2` | — | 0.0 / 0.03 / 0.156 | |
| `I_app` | 0 | 0 | |
| `lambda` / `exp_input_weight` | 0.0 / 1.0 | 0.0 / 1.0 | inert in v07 |

**None of these come from `parameters.csv`.** The `str_d1.*` / `str_d2.*` /
`str_fsi.*` rows exist in the file, but their v07 cells are empty, and
`_set_params` skips Microcircuit components anyway; every value above is a
CompNeuroPy class default.

**Nor are any initial values.** `create_populations_annarchy` passes no `init`,
and nothing else assigns `v` or `u`, so all three striatal populations start at
ANNarchy's defaults `v = 0, u = 0` — 80 mV above `v_r`, and above `v_peak` for
the FSI. Every striatal neuron therefore fires one synchronous spike on the first
step, and since `reset()` restores exactly this state, the volley recurs at the
start of **every** evaluation. The BOLD run absorbs it in the 2310 ms ramp-up;
the 9900 ms firing-rate probe starts straight after reset, so the artefactual
spike sits inside the probed window (~0.1 Hz upward bias per neuron, small
against the 12.7–37.3 Hz bands, plus whatever the t = 0 volley does to the
recurrent circuit). Contrast the six BG populations, whose `v_init`/`u_init` the
CSV sets deliberately (§6.1). `TODO.md` §28.

### 6.3 `Izhikevich2007Humphries2009FSI`

```
dg_ampa/dt = -g_ampa/tau_ampa
dg_gaba/dt = -g_gaba/tau_gaba
I_v        = g_ampa * E_exc / (1 + g_ampa * dt / C)
           + g_gaba * (E_gaba - v) * (1 - epsilon * phi_2) / (1 + g_gaba * dt / C)
           + I_app
C * dv/dt  = k * (v - v_r * (1 - eta * phi_1)) * (v - v_t) - u + I_v
du/dt      = if v < v_b: -a * u   else: a * (b * (v - v_b)**3 - u)
```

Spike `v >= v_peak`, reset `v = c; u = u` — **`u` is not incremented**, and `d`
is 0 anyway.

**No NMDA and no `g_glut` term.** That is the whole reason every external input
to an FS neuron is routed to `ampa` rather than `glut`, both in
`Microcircuit._create_inputs_annarchy()` and in the `thal__str_fsi` projection.

| parameter | value |
|---|---|
| `tau_ampa` / `tau_gaba` | 6 / 4 ms |
| `E_ampa` / `E_gaba` / `E_exc` | 0 / −60 / 50 mV (`E_ampa` inert, as in §6.2) |
| `C` / `k` | 80 pF / 1.0 |
| `v_r` / `v_t` / `v_b` / `v_peak` | −70 / −50 / −55 / 25 mV |
| `a` / `b` / `c` / `d` | 0.2 / 0.025 / −60 / 0 |
| `eta` / `epsilon` | 0.1 / 0.625 |
| `phi_1` / `phi_2` | 0.0 / 0.0 |
| `I_app` | 0 |
| `lambda` / `exp_input_weight` | 0.0 / 1.0 — inert, as in §6.2 |

---

## 7. The striatal microcircuit

`CompNeuroPy/src/CompNeuroPy/striatal_microcircuit/microcircuit.py`.

Most of what follows **does not run during an evaluation**. `build_mc` and
`build_ci` are `False` there, so `Microcircuit.__init__` takes the load branch at
each of its three decision points and reads the results from disk. §7.1–§7.6
describe what was computed once, by `build_input_caches.py`; §7.7 describes what
an evaluation actually does.

The three decision points are `build_connectivity` (§7.2),
`build_missing_gaba_input` (§7.3) and `build_cortical_input` (§7.5). The last two
both end in the same place — a set of **spike-count streams** on disk — and they
get there through the same generator, which §7.4 describes once for both.

### 7.1 The lattice — always built

Never cached, because it is cheap and deterministic given `seed`, `nx`, `b`.

`n_total = nx · b · b = 1000` sites on a regular 3D grid with spacing
`d = (1/density)^(1/3)`. At the fixed `density = 84 900 neurons/mm³` that is
**22.76 µm**, so the cube is 227.6 µm on a side and holds 0.01178 mm³.

Cell types are assigned by the del Rey et al. (2022) proportions
`[FS, dSPN, iSPN] = [0.026, 0.43, 0.43]`, normalized to
`{FS: 0.02935, dSPN: 0.48533, iSPN: 0.48533}`. Rounding down leaves one site
over, which goes to the largest-proportion type, and the resulting type vector is
shuffled. The outcome is fixed: **486 dSPN, 485 iSPN, 29 FS**.

The cube is **periodic**: 27 shifted copies of the position array are stacked
into one `cKDTree`, neighbour queries take the result modulo `n_total`, and
distances use the minimum-image convention.

### 7.2 Connectivity — built once, cached

For every postsynaptic neuron, all presynaptic candidates within a radius are
enumerated and connected independently with probability

```
p(d) = P0 · exp(-d² / σ²)
```

using that pair's `P0` and `σ` from `fitted_params.json`. Despite the method name
`_p_exp`, this is a **Gaussian in distance**, not an exponential.

The radius is `min(3 · σ_max(post), L_max/2)` where `σ_max(post)` is the largest
σ over all pairs targeting that post type. With σ up to 400 µm, `3σ` is 1.2 mm
while `L_max/2` is only 113.8 µm — **so the radius is clamped to half the box for
all three post types.** The simulated cube is far smaller than the kernel's
reach, and that gap is precisely what §7.3 has to compensate for.

Each accepted connection draws its weight from a `CombinedSampler` (the class
lives in `extra_functions.py`; the mixture definitions in `get_weights.py`) over
a pair-specific mixture of literature distributions, each component carrying its
own weight in the mixture:

| pair | mixture |
|---|---|
| SPN → SPN | uniform 0.18–0.37 (w 87) + truncated Gaussian μ 0.42, σ 0.25 (w 23) + truncated Gaussian μ 0.75, σ 0.57 on [0.07, 1.81] (w 26) |
| FS → SPN | 41-bin empirical histogram over 0–58.57 (w 75) + truncated Gaussian μ 1.57, σ 2.68 (w 31) + truncated Gaussian μ 3.84, σ 3.04 on [0.64, 8.14] (w 9) |
| FS → FS | single truncated Gaussian μ 1.1, σ 1.5 on [0, 10.1] |

Note the scale: **FS→SPN weights are an order of magnitude larger than SPN→SPN
weights** (mixture means 6.06 against 0.41, a factor of ~15; only the tails reach
two orders, 58.6 against 1.9), which is what makes the 29 FS neurons matter at
all. Weights land in a `lil_matrix` per pair, indexed by **type-local** indices.

Cached in `connectivity/connectivity_state.pkl` plus seven
`connectivity/weights_<pre>_<post>.npz`. On load, `n_total`, `nx` and the full
type vector must match, all seven weight files must exist, and **the numpy RNG
bit-generator state is restored** so that a load run continues on the same random
stream a build run would have been on. One hole: `b`, `density` and the fitted
kernel parameters are *stored* in the state but never compared on load, so a
connectivity cache built at a different density or from a different
`fitted_params.json` would load silently as long as the type vector matches.

### 7.3 Missing-GABA compensation — built once, cached

The simulated cube is 227.6 µm across; the connection kernel reaches
`Rout = 3σ`, which is 568 µm for FS and ~1.2 mm for both SPN types. So each
neuron is wired only out to `Rin = 113.8 µm` (§7.2) and everything beyond that is
real striatum that is not in the simulation. It is replaced by synthetic
spike-count streams, in two steps (`Microcircuit._missing_local_input()`).

**Step 1: how much is missing** (`_define_expected_input_counts`). For each pair,
integrate the kernel over the outer shell:

```
E_outer = 4π ρ ∫_{Rin}^{Rout} p(r) r² dr
```

with `ρ = props[pre] · density`. The same integral over `[0, Rin]` gives
`E_inner`, a sanity check against the connections actually made.

| pair | `E_inner` | `E_outer` | pre rate | spikes/s delivered |
|---|---|---|---|---|
| FS → FS | 1.9 | 12.4 | 10.5 Hz | 130 |
| FS → dSPN | 8.8 | 500.0 | 10.5 Hz | 5 250 |
| FS → iSPN | 9.6 | 25.3 | 10.5 Hz | 266 |
| dSPN → dSPN | 26.2 | 1568.5 | 25.0 Hz | 39 212 |
| dSPN → iSPN | 25.0 | 1500.3 | 25.0 Hz | 37 507 |
| iSPN → dSPN | 31.8 | 505.4 | 33.0 Hz | 16 678 |
| iSPN → iSPN | 32.5 | 948.1 | 33.0 Hz | 31 286 |

Read the first two columns together: **a dSPN gets ~67 simulated GABAergic
afferents against ~2573 synthetic ones, so the microcircuit is about 2 % circuit
and 98 % open-loop stream.** Note also that `fitted_params.json` has no
`dSPN→FS` or `iSPN→FS` pair, so **FS neurons receive GABA only from other FS
neurons** — 130 spikes/s, against 61 139 for a dSPN and 69 059 for an iSPN. What
this forecloses is set out in `experimental_data/input_streams/README.md` §4.1.

**Step 2: realise the pool** (`_simulate_distance_dependent_spike_counts`,
which builds the cloud with `build_geometric_source_pools` and draws the counts
with `simulate_receiver_counts_geometric_to_memmap`; §7.4 (1b) and (3b) state
the same steps as formulas). The missing striatum is not approximated by a rate
or a correlation matrix — it is *built*, as a cloud of virtual sources around
the lattice, and the stream is simply what that cloud delivers.

A **virtual source** is a fixed point in space standing for `k_mult` real
out-of-cube neurons of the presynaptic type — nothing more. It has no neuron
model, no membrane, no spike train of its own; once built it is just (a) a
position and (b) the list of receivers it connects to. Spikes are attributed to
it bin by bin, in step 2c. `k_mult` is `source_multiplicity` (10), capped per
pair so that no receiver ends up with fewer than 50 sources: at `k_mult = 10`
the 12-afferent FS→FS pair would have barely one source per receiver, and both
its degree and its shared fraction would be rounded away by the coarse grain.

**(2a) Place.** Scatter `S = ρ · V / k_mult` sources uniformly in the box that
extends `Rout` beyond the lattice on every axis, so no receiver sits near an
edge. This construction is **not periodic** — the surrounding box does for the
outer shell what the periodicity of §7.1 does for the in-cube wiring.

Worked through for dSPN→dSPN, the dominant pair (one realisation at the shipped
geometry; the numbers move by ~2 % between realisations): the 227.6 µm cube
padded by `Rout = 1.2 mm` per side gives a 2.61 mm box of 17.7 mm³. At
`ρ = 0.485 · 84 900 ≈ 41 200 dSPN/mm³` and `k_mult = 10` that is **S ≈ 73 000
sources standing for ~730 000 real dSPNs** — surrounding the 486 simulated
dSPN receivers.

**(2b) Wire.** Each receiver connects to each source at distance
`d ∈ [Rin, Rout]` with probability `p(d)` — the same kernel §7.2 fitted for
this pair. The out-of-cube world is wired by the same rule as the in-cube one;
the only difference is that its neurons are points rather than Izhikevich
models. For dSPN→dSPN this yields ~77 000 receiver–source connections: a mean
degree of ~159 sources per receiver, i.e. ~1 590 stood-for afferents.

That degree is the one property **checked at build time**, because it is a
prediction rather than an input: `mean_degree · k_mult` must reproduce the
`E_outer` integral of step 1 (here 1 593 against 1 568.5 — off by 1.5 %), and a
deviation beyond 20 % raises. The pool is one random realisation of the source
cloud, so that tolerance is about 4σ of a spread measured at 1.6 % (dSPN→dSPN)
to 4.8 % (FS→dSPN) with a bias below 1 %; anything that actually breaks the
construction — a wrong kernel, radius or density — is off by a factor, not by
20 %.

**(2c) Draw, per 0.1 ms bin.** One Binomial for the whole cloud, then scatter
the spikes onto the receivers:

```
n_events(t) ~ Binomial(S · k_mult, p)         p = rate · dt / 1000
source of each event ~ Uniform{0 … S−1}
c_i(t)      = events whose source connects to receiver i
```

The first line asks *how many of the ~730 000 real dSPNs fired in this bin* —
at 25 Hz, `p = 0.0025`, about 1 825 of them. Each of those spikes is assigned
to a uniform-random source and delivered to **every receiver on that source's
list**. A receiver wired to 159 of the 73 000 sources therefore catches on
average `1 825 · 159 / 73 000 ≈ 4` spikes per bin, and its marginal count is
exactly `Binomial(1 590, 0.0025)` — as if its ~1 590 missing afferents were
individually simulated.

**Sharing is never computed — it happens.** Take two nearby receivers A and B
and five sources: A is wired to {s1, s2, s3}, B to {s2, s3, s4}. In some bin
two spikes are drawn and land on s1 and s3. A's count is 2, B's is 1 — and the
spike on s3 entered **both** counts, because both receivers are wired to the
same point. That doubly-delivered spike *is* the shared input. Over many bins
the correlation between `c_A` and `c_B` equals the fraction of their pools they
share, `n_shared / √(deg_A · deg_B)`; receivers close together overlap in many
sources, distant ones in few, so the shared fraction falls with distance —
`f(d)`, reproduced by the geometry without ever being evaluated. That is what
changed on 2026-08-07: the old code computed `f(d)` by double quadrature and
*imposed* it on the draw through a Gaussian copula; realising the pool makes
the overlap emerge instead, exact at any lattice size or density — which is
what keeps `TODO.md` §24 (a larger, sparser cube) reachable without another
rewrite.

Note `f(0)` is **not 1**: two receivers at the same point each connect to a
given distant source only with probability `p(r)`, independently, so even
coincident receivers share just the `p`-weighted mean of `p` — ≈ 0.02–0.21
depending on the pair (dSPN→dSPN: ~0.034 averaged over the lattice).

**Two things could be called "shared input"; only one is active.** The overlap
above is shared *membership* — one presynaptic spike arriving at several
receivers. On top of that, the machinery can also make the sources themselves
co-fluctuate: a pairwise spike-count correlation `r_sc` among the unsimulated
neurons, realised as a shared modulation of `p(t)` (§7.4 step 2b). For the
missing-GABA streams `correlation_dict` sets `r_sc = 0` for all three types, so
that layer is **switched off** and step 2b is skipped entirely: the only
correlation between two receivers' missing-GABA streams is the pool overlap
`f(d)`.

The realised shared fractions are checked against the analytic `f(d)` **at
every build** since 2026-08-13 (`TODO.md` §27, resolved):
`Microcircuit._check_realised_shared_fractions` re-derives
`f(d) = E_shared(d) / E_outer` by the double quadrature the geometric
construction replaced (`_expected_shared_for_d`, restored from git history) and
compares it with the mean realised shared fraction around three probe
distances, raising beyond a 25 % relative tolerance (probes where the analytic
value is < 1e-3 are recorded but not judged); the probe table is stored in
`stream_statistics` under `f_d_check`. The original one-off manual
verification at the current geometry (commit `148aec2`):

| pair | `f(0)` analytic | `f(dmax)` analytic | realised (near / mid / far) |
|---|---|---|---|
| dSPN → dSPN | 0.0372 | 0.0318 | 0.0364 / 0.0344 / 0.0314 |

Finally the **mean** of 10 000 samples from each pair's weight sampler is stored
as `mean_weights_by_type[(pre, post)]`. That scalar is what the streamed counts
are multiplied by at simulation time (§7.7): the stream says *how many spikes
arrived*, the scalar says *what one is worth*. So the whole outer shell collapses
onto one mean weight, where the inner shell keeps individually sampled ones.

The rates come from `parameters.py: mc.firing_rate_dict`, threaded through
`v07_model_creation_kwargs`: `{FS: 10.5, dSPN: 25.0, iSPN: 33.0}` Hz, the
parkinsonian **medication-off** state of Liang et al. 2008 (see
`experimental_data/activity_striatum/README.md`). Alongside them
`mc.correlation_dict`, the `r_sc` layer disentangled above —
**`{FS: 0, dSPN: 0, iSPN: 0}`**.

Cached in `inputs/missing_input_state.pkl`. On load, the pair-key set, `dt`,
**`n_steps` exactly**, `firing_rate_dict`, `correlation_dict`,
`correlation_window_ms`, `correlation_timescale_ms` and `source_multiplicity`
must all match, every `.dat` must exist, and the RNG state is restored. A state
file written before any of those were recorded is refused rather than trusted.

### 7.4 How one spike-count stream is drawn

Everything cached in §7.3 and §7.5 is the same kind of object, produced by
`striatal_microcircuit/spike_input_cortex.py`. Understanding it once covers the
missing-GABA streams, the striatal cortical streams and all of `CorticalInputs`.
A visual walkthrough of this section on drawable toy examples — with real-code
statistics checks and real-scale benchmarks — lives in `docs/stream_demo/`
(dated snapshot 2026-08-11, not a living document; regenerate with its
`make_figures.py` if it drifts).

**What a stream is.** One `(R, n_steps)` matrix for one `(pre, post)` pair. Row
`i` is receiver `i`; column `t` is one `dt = 0.1 ms` step. The entry is a
**count**: how many of that pair's presynaptic neurons spiked into receiver `i`
in that bin. The presynaptic neurons are never represented individually — no
spike trains, no individual synapses, no per-synapse weights. What turns a count
into a current is a single mean weight applied later (§7.7).

**What it must reproduce.** A stream stands in for `N_eff` presynaptic neurons
firing at a stated rate, whose pools overlap between receivers by `f_ij`, and
which may themselves be pairwise correlated at `ρ`. Three things then follow with
no modelling freedom:

```
mean = N · p(t)                       p(t) = rate(t) · dt / 1000
Fano = (1 − p) · ((1 − ρ) + N·ρ)
corr = (f·(1 − ρ) + N·ρ) / ((1 − ρ) + N·ρ)
```

**These are the target statistics**, and the generator is checked against them at
build time. `experimental_data/input_streams/README.md` is the full contract —
what each input quantity is, where it comes from, and what this whole approach
deliberately cannot represent.

The checker (`stream_target_statistics`) states the same law in an equivalent
parameterisation: instead of ρ it works with the shared-modulation variance σ²,

```
var  = N·p·(1 − p·(1 + σ²)) + (N·p)²·σ²          Fano = var / (N·p)
cov  = f·N·p·(1 − p·(1 + σ²)) + (N·p)²·σ²        corr = cov / var
```

with `ρ = p·σ²/(1 − p)`; `solve_modulation_amplitude` maps a measured `r_sc` at
window `T_meas` onto σ² (step 2b below). Substituting recovers the forms above
exactly, and at ρ = σ = 0 — the shipped configuration — both reduce to
`Fano = 1 − p`, `corr = f`.

**The principle.** Every construction realises the presynaptic pool explicitly
and lets overlaps produce the correlations, rather than computing a correlation
and imposing it. That is the difference from the pre-2026-08-07 code, which drew
correlated uniforms from a Gaussian copula at the shared-fraction matrix, pushed
them through a Beta inverse-CDF and then a Binomial inverse-CDF, and got a
correlation of 0.00009 where 0.014 was intended and a Fano factor of 1922 where
1 was.

---

#### The procedure, step by step

Four steps. Step 1 runs once per stream — except in variant 1a, where it runs
once per cortical region, covering all of the region's receiver types at once;
steps 2 and 3 run once per time chunk; step 4 once at the end.

##### Step 1 — fix the presynaptic pool

Three variants, one per situation. This is the only step that differs between
them; steps 2–4 are shared.

**(1a) Cortical onto striatal neurons — one axon pool per region.**
`simulate_cortical_axon_pool_streams_to_memmap`, driven from
`Microcircuit._simulate_cor_input_spike_counts`.

```
M = round(N_eff(dSPN) / shared_fraction)          # axons in the region's pool
N_i = round(proportion(region) · N_cortical_inputs_dict[type_i])
```

`shared_fraction = 0.014` (Kincaid), `N_eff(dSPN)` is the SPN afferent count for
this region, and **all** receiver types of the region are drawn together from
this one `M`. No membership matrix is built: with a uniformly sampled pool the
hypergeometric draw of step 3a is exactly equivalent and needs no storage.

**(1b) Missing GABA — an explicit cloud of virtual sources.**
`build_geometric_source_pools`.

```
k_mult = max(1, min(source_multiplicity, floor(E_outer / 50)))
lo     = min_axis(receiver_positions) − Rout      # per axis
hi     = max_axis(receiver_positions) + Rout
S      = round(ρ_pre · ∏(hi − lo) / k_mult)       # number of virtual sources
```

Sources are placed uniformly in the box. Receiver `i` connects to source `s`
when

```
Rin ≤ |x_s − r_i| ≤ Rout    and    U(0,1) < p(|x_s − r_i|)
```

with `p` the pair's own kernel from `fitted_params.json`. One virtual source
stands for `k_mult` real neurons; the cap keeps at least 50 sources per receiver,
because at `source_multiplicity = 10` the 12-afferent FS→FS pair would have
barely one and both its degree and its shared fraction would be rounded away.
Connectivity is stored source-major (`src_indptr`, `src_receivers`) because step
3b scatters from sources to receivers.

The box extends `Rout` beyond the receiver bounding box in every direction, so no
receiver sees an edge. This construction is **not periodic**, unlike the lattice
of §7.1 — the surrounding box does the job the periodicity did.

One thing is then checked, because it is a prediction rather than an input:

```
|mean_degree · k_mult − E_outer|  ≤  0.20 · E_outer
```

The 20 % is about 4σ of a spread measured at 1.6 % (dSPN→dSPN) to 4.8 %
(FS→dSPN) across source clouds, with a bias below 1 %. The realised shared
fractions (`realised_shared_fractions`) are checked too, against the analytic
`f(d)` at three probe distances (`_check_realised_shared_fractions`, 25 %
tolerance, raising; §7.3) — since 2026-08-13; before that only their mean fed
the step-4 statistics check.

**(1c) `CorticalInputs` — a flat split.**
`simulate_receiver_counts_homogeneous_to_memmap`.

```
n_shared  = round(f · N)          # f = ci.shared_fraction_dict[population]
n_private = N − n_shared
```

One population per stream, so there is no cross-type structure to represent.

##### Step 2 — the per-bin probability `p(t)`

Shared by every receiver of the stream. `make_global_p_trace`.

**(2a) The drive.**

```
p_drive(t) = rate(t) · dt / 1000
```

`rate` is a scalar for the missing-GABA streams (`firing_rate_dict[pre]`) and a
per-step array for the cortical ones, expanded from one value per TR by repeating
each `TR/dt = 23 100` times.

**If `r_sc = 0`, `p(t) = p_drive(t)` and step 2b is skipped entirely.** That is
what currently ships.

**(2b) The shared rate modulation.** `r_sc` is the pairwise spike-count
correlation among the presynaptic neurons themselves, and it is meaningless
without the window `T_meas` it was measured at — it grows with the window and
saturates past the correlation timescale `τ_c`. So:

```
a    = exp(−dt / τ_c)                             # AR(1) coefficient, 0 if τ_c = 0
m    = round(T_meas / dt)                         # window in bins
W(m) = m + 2·[ m·g − w ]                          # Var of a sum of m AR(1) samples
       g = a(1 − a^(m−1)) / (1 − a)
       w = a(1 − m·a^(m−1) + (m−1)·a^m) / (1 − a)²
       W(m) = m when a = 0
σ²   = r_sc · m / ((1 − r_sc) · p̄ · W(m))         # p̄ = mean(p_drive) over the chunk
```

Then a unit-variance AR(1) Gaussian trace, continuous across chunk boundaries:

```
ξ ~ N(0,1)^n
z = lfilter([√(1 − a²)], [1, −a], ξ) + carry · a^(1..n)
```

`carry` is the last `z` of the previous chunk, and on the **first** chunk it is
drawn from `N(0,1)` rather than left at 0 — `lfilter` starts from a zero state,
which would leave the leading samples with reduced variance, and those are
exactly the samples step 4 measures.

Finally the modulation itself, with a Gamma marginal so it is strictly positive
at any amplitude and leaves the mean untouched:

```
u   = clip(Φ(z), 1e−12, 1 − 1e−12)                # Φ = standard normal CDF
Mod = Gamma.ppf(u, shape = 1/σ², scale = σ²)      # mean 1, variance σ²
p(t) = clip(p_drive(t) · Mod(t), 0, 1)
```

The clip on `u` is because `Φ` saturates to exactly 1 past `|z| ≈ 8.3` and
`Gamma.ppf(1)` is infinite. Cost is `O(n_steps)`, not `O(R · n_steps)` — one
trace per stream, not one per receiver.

##### Step 3 — draw the counts

**(3a) Cortical axon pool.** Per bin, decide how many axons of the shared pool
fire, then how many of those each receiver happens to own:

```
k(t)      ~ Binomial(M, p(t))                     # once per bin, shared
c_i(t)    ~ Hypergeometric(ngood = N_i, nbad = M − N_i, nsample = k(t))
```

This is exact in every respect. Marginals are `Binomial(N_i, p)`, and the
correlation between **any** two receivers is `√(N_i·N_j)/M` whether they are the
same type or not — so the cross-type shared fractions are derived from `M`
rather than being free parameters. For the reference type (the SPNs, whose count
sets `M`) it reduces to `N/M = f`; the FS within-type overlap is smaller,
`N_FS/M = f·N_FS/N_SPN = 0.0056` (§7.5).

**(3b) Geometric source pool.** Per bin, draw how many of the real neurons fired,
assign each spike to a source, and add it to every receiver that source feeds:

```
n_events(t) ~ Binomial(S · k_mult, p(t))
source of each event ~ Uniform{0 … S−1}
c_i(t) = number of events whose source connects to receiver i
```

The last line is a `bincount` over a flattened `(receiver, bin)` index, which is
what makes this affordable. Unlike 3a, this draw is not exact: the uniform
assignment is with replacement and uncapped, so the marginal is
`Binomial(S · k_mult, deg_i · p / S)` rather than the `Binomial(deg_i · k_mult, p)`
of a real pool — same mean, Fano `1 − deg_i·p/S` instead of `1 − p`, and
strictly a source can be assigned more spikes in a bin than the `k_mult`
neurons it stands for. All of these differences are O(p) at the rates in force
(see `experimental_data/input_streams/README.md` §4.3). The correlation between
two receivers is their pool overlap — i.e. `f(d)`, never computed.

**(3c) Flat split.**

```
S(t)   ~ Binomial(n_shared,  p(t))                # once per bin, shared
P_i(t) ~ Binomial(n_private, p(t))                # per receiver
c_i(t) = S(t) + P_i(t)
```

Exactly `Binomial(N, p)` per receiver and exactly `corr = f`, because that is
literally what a shared sub-pool plus a private sub-pool means.

##### Step 4 — write, then check

Counts go straight into the `(R, n_steps)` memmap; the full array is never
materialised. The time axis is chunked so the working set stays near 128 MB
(32 MB for the geometric missing-GABA streams, which carry more per-bin state),
and the AR(1) `carry` crosses the boundaries so chunking is invisible in the
output.

The first `min(chunk, 20 000)` bins of the first chunk are kept as the check
sample. The drive is constant across a TR (23 100 bins), so a sample that size
sits inside one TR and its measured Fano factor is comparable with the
single-bin target. Measured mean, Fano factor and mean pairwise correlation are
compared with the closed forms at the top of this section, and a mismatch
**raises**; the measured values are written into the cache state so a cache can
be audited without regenerating it.

Two subtleties are baked into the tolerances, both found by the check firing on
good draws during development:

- the mean tolerance is `max(2 %, 4 · relSE)` with
  `relSE = √(Fano · (1 + (R−1)·corr) / (mean · R · n))`, because a sparse stream
  like M1→FS (29 receivers, 56 afferents, 0.02 counts per bin) has a 1 %
  standard error and a flat 2 % would fire on a perfectly good draw;
- the 150 receivers entering the correlation are drawn at **random** (fixed seed),
  not taken from the front: receiver index order follows the x-major lattice, so
  the leading rows are spatially clustered, which biased the measured FS→iSPN
  correlation up by 25 % on a stream whose `f(d)` spans 0.20 to 0.067.

Fano is checked at 10 % relative, the correlation at `max(0.02, 20 %)`.

---

**What is *not* correlated: time.** Bins are drawn independently. Nothing
produces autocorrelation, burst structure or refractory effects except the rate
series itself and the shared modulation of step 2b.

**All three of these knobs are currently 0** — the two presynaptic correlations
`mc.correlation_dict` and `mc.cortical_correlation`, and the overlap fraction
`ci.shared_fraction_dict` — so step 2b is skipped and
`f` is the only source of receiver correlation. That is deliberate, not an
oversight: the input correlation is the dominant determinant of the simulated
BOLD amplitude (`Var(Σ I_i) = N·v·(1 + (N−1)r)`, a 481× swing at `N = 486`
between `r = 0` and `r = 0.99`), and the model cannot decorrelate the way a real
striatum does. `TODO.md` §25 has the reasoning and the scan meant to set them.

| argument | missing GABA (§7.3) | cortical, striatum (§7.5) | cortical, BG (§8) |
|---|---|---|---|
| pool | geometric, from the kernel (1b) | `M = N_eff/f` axons (1a) | flat `f` (1c) |
| `N_eff` | `E_outer`, 12–1568 | `round(proportion · N_total)` | `round(proportion · N_total)` |
| `rate` | scalar, `firing_rate_dict[pre]` | per-step array from the drive | per-step array |
| `f` | emerges, ≈ 0.02–0.21 | 0.014 SPN↔SPN (Kincaid); cross-type from the pool (§7.5) | `ci.shared_fraction_dict`, 0 |
| `r_sc` | `correlation_dict[pre]`, 0 | `cortical_correlation`, 0 | `cortical_correlation`, 0 |

### 7.5 Cortical input — built once, cached

`Microcircuit._simulate_cor_input_spike_counts()` turns the BOLD-derived cortical
drive into streams, in three steps.

**Step 1: the rate series.** The `.npz` holds one `<region>_rate` array of 310
values, one per TR, per cortical region (§2). The step is inferred from the
matching `<region>_time` array — 2.31 s — and each value is repeated
`TR/dt = 23 100` times, then truncated to `n_steps`. A drive finer than `dt`, or
one whose spacing is not an integer multiple of it, raises. Every region's series
averages exactly 5 Hz.

**Step 2: how many presynaptic neurons each region supplies.**

```
N_eff = round(cortical_proportions_dict[region] · N_cortical_inputs_dict[receiver])
```

with `N_cortical_inputs_dict = {FS: 2800, dSPN: 7000, iSPN: 7000}` — whose
derivation is **not yet written down**, see `TODO.md` §23. A caudate dSPN's 7000
afferents split 3850 dlPFC, 1260 PMd, 1050 preSMA, 420 SMA, 280 PMv, 140 M1, 0
S1. A region contributing nothing gets no stream (`Microcircuit` skips on the
proportion being ≤ 0, `CorticalInputs` on `N_eff = 0` — same outcome at the
current numbers), which is the only reason the two loops differ in stream count.
The counts here — caudate 25, putamen 28 — are the **Microcircuit's own** streams
(7 compensation + 18/21 cortical); `CorticalInputs` adds its own on top (§7.7).

**Step 3: the draw.** All three receiver types of a region are drawn **together**,
sampling one pool of

```
M = round(N_eff(dSPN) / shared_fraction)
```

axons. `shared_fraction = 0.014` comes from Kincaid et al. 1998 (*J Neurosci*
18:4722): one corticostriatal axon contacts ≤1.4 % of the cells in its
arborization, and the shared fraction between two SPNs equals that same figure.
For dlPFC that gives `M = 3850/0.014 = 275 000` axons — the same order as the
~380 000 cortical axons Kincaid reports innervating one spiny cell's dendritic
volume, which is mild corroboration of both numbers.

Drawing the region's types together is what fixes the cross-type sharing. Drawn
separately — as they were before 2026-08-07 — a dSPN and an iSPN sitting in the
same tissue and sampling the same axons would share **nothing at all**. With one
pool the shared fractions follow from `M`:

| pair | shared fraction | how |
|---|---|---|
| dSPN ↔ dSPN, iSPN ↔ iSPN, dSPN ↔ iSPN | 0.0140 | `N_SPN/M`, i.e. Kincaid |
| FS ↔ SPN | 0.00885 | `√(N_FS·N_SPN)/M = f·√(N_FS/N_SPN)` |
| FS ↔ FS | 0.0056 | `N_FS/M = f·N_FS/N_SPN` |

Note FS↔FS overlap is *smaller* than FS↔SPN — 15.7 axons against 39.2 — because
an SPN samples more of the pool. No nested block construction can represent that,
which is why the pool is realised explicitly. Ramanathan et al. 2002 and Choi et
al. 2018 both report higher cortical convergence onto FS interneurons than onto
SPNs, which this derivation does not capture; see `TODO.md` §26.

Regions, by contrast, are drawn **apart**: each has its own axon pool, so within
a bin the streams of different regions are independent. Their co-fluctuation
comes only through the correlated BOLD-derived rate series each pool is driven
by.

**FS cortical input is drawn like the SPNs'.** It used to be *derived* — the
weighted sum of the cortical counts of the SPNs each FS projects onto, rescaled
and Poisson-resampled. That asserted an FS↔SPN input correlation of about 1,
which nothing measures; it broke at larger cube sizes (FS→iSPN falls below one
connection per iSPN at 800 µm); and an FS with no outgoing connections got
scaling factor 0 and hence **no cortical input at all, silently**. All three go
away with a drawn stream.

The cortical proportions are **the only physical difference between the two
loops**:

| region | caudate | putamen |
|---|---|---|
| dlPFC | 0.55 | 0.10 |
| preSMA | 0.15 | 0.05 |
| PMd | 0.18 | 0.11 |
| PMv | 0.04 | 0.18 |
| SMA | 0.06 | 0.15 |
| M1 | 0.02 | 0.28 |
| S1 | 0.00 | 0.13 |

They are defined in exactly one place, `BOLD_optimization/parameters.py` under
`cortical_proportions_dict`, and reach the model as
`mc.cortical_proportions_dict` (§3.3), which `model_creation_functions.py` hands
to both `Microcircuit` and `CorticalInputs`. Neither class has a default; both
call `spike_input_cortex.validate_cortical_proportions()`. The *same* numbers
weight the mix producing `caudate_rate`/`putamen_rate` inside the rate `.npz`, so
**changing them means regenerating that file**. The derivation is in
`experimental_data/cortical_proportions/README.md` and `TODO.md` §21 (resolved
2026-08-06).

Cached in `inputs/cortical_input_state.pkl`, which also records
`shared_fraction`, `cortical_correlation`, `correlation_window_ms` and
`correlation_timescale_ms`; a mismatch on any of them raises.

### 7.6 What the cache actually is

Each stream is a **raw, headerless `np.memmap`** at
`<storage_dir>/inputs/receiver_counts_<pre>_<post>.dat`, of shape
`(R, n_steps)` and dtype `float64`. Shape and dtype live only in the pickled
state file, so a `.dat` of the wrong length would be silently misread — which is
why the load checks are strict.

The layout is row-per-receiver, column-per-`dt`, i.e. **time is the fast axis**,
which is the wrong way round for how the streams are read back: an evaluation
wants one 1100-step column block across all receivers at a time (§7.7). That, and
storing small integer counts in 8-byte floats, is what `TODO.md` §3 is about.

The checks that bite in practice:

- **`n_steps` must match exactly.** A cache is built for one duration and is
  usable at that duration only. This is why `mc_ci_cache_5tr` (115 500 steps) can
  only serve `--n-trs 5`.
- **`cortical_rate_path` is stored and compared as a resolved absolute path**
  (since 2026-08-13; `TODO.md` §13, resolved). `parameters.py` builds it from
  its own file location, so this check no longer ties builds and evaluations
  to a launch directory — but it does tie a cache to the machine (repo path)
  it was built on. Before, the verbatim string comparison forced both to run
  from `BOLD_optimization/`.
- `dbs_condition`, `dt`, the proportions dict, `N_cortical_inputs_dict`,
  `shared_fraction` and the correlation fields are compared too. The stored
  stream keys are compared against the pairs the **current configuration**
  would generate, and each stream's stored receiver count `R` against the
  population size (both since 2026-08-13 — the old "key set" check compared
  the pickle against itself and only caught corruption).
- Any **mismatch** raises; nothing is silently rebuilt. Since 2026-08-13 an
  *absent* field also raises, for every compared field in all three state
  files (`TODO.md` §29): a state file written by the current code always
  records all of them, so absence means a pre-format or corrupt file.

Sizes follow directly from the row counts. Across both loops the streams total
24 084 rows (caudate 11 342, putamen 12 742), so one DBS condition costs
`24 084 × n_steps × 8 B`:

| | `n_steps` | per DBS condition |
|---|---|---|
| `--n-trs 5` | 115 500 | 22.25 GB (20.7 GiB) derived — measured 22.28 GB, the ~30 MB gap being state pickles and connectivity files |
| full run, 310 TRs | 7 161 000 | 1 380 GB (**1.25 TiB**) |

A single full-length dSPN cortical stream is `486 × 7 161 000 × 8 B ≈ 27.8 GB` on
its own. Note that `mc_ci_cache_5tr/` holds **both** conditions and so measures
44.6 GB in total.

The `mc_ci_cache_dir` comment in `parameters.py` states both figures: ~1.25 TiB
per DBS condition in this current `float64`/per-region layout, ~120 GiB after
the planned `uint16`, pre-summed relayout (`TODO.md` §3; the history
of that number is `TODO.md` §19, resolved 2026-08-11).

### 7.7 What an evaluation does: `create_model()` and the streams

`Microcircuit.create_model()` runs four steps:

1. `create_populations_annarchy()` — the three populations, named
   `f"{self.name}_{cell_type}"`.
2. `create_local_projections_annarchy()` — one `Projection(target="gaba",
   name=f"Proj_{pre}_{post}_{self.name}")` per weight matrix, wired with
   `connect_from_sparse`. Seven of them, one per `fitted_params.json` pair.
   **These carry individual, distance-sampled weights** — not a
   `connect_fixed_number_pre` with a single scalar.
3. `_create_inputs_annarchy(local_input_memmap_dict)` — the compensation streams.
4. `_create_inputs_annarchy(cor_input_memmap_dict)` — the cortical streams.

Each stream becomes a `TimedArray` of shape `(update_time/dt, post_pop.size)` =
`(1100, size)` initialised to zeros, named `TimedInput_<pre>_<post>_<loop>`, plus
a `CurrentInjection` named `CurrentInjection_<pre>_<post>_<loop>` wired with
`connect_current()`. The `TimedArray` is a one-neuron-per-receiver rate-coded
population that simply reads out row `k` of its buffer at step `k` (default
schedule = one row per `dt`, so 1100 rows = 110 ms); `CurrentInjection` is a
one-to-one wiring that does `post.g_<target>[i] += pre.r[i]` every step. Nothing
else happens in between — no synapse model, no delay, no weight.

The target is decided here:

| stream | target |
|---|---|
| pre is a striatal cell type (compensation) | `gaba` |
| pre is a cortical region, post is FS | `ampa` |
| pre is a cortical region, post is dSPN/iSPN | `glut` |

**and the target decides what the injected number physically does**, because the
three conductances are treated differently by the neuron equations (§6.2, §6.3):

- `gaba` and `ampa` have their own ODE (`dg/dt = −g/tau`). The injected value
  lands as a jump on a conductance that then decays with `tau_gaba` / `tau_ampa`.
- `glut` has none. It appears only as `+ g_glut/dt` inside *both* `dg_ampa/dt`
  and `dg_nmda/dt`, and ANNarchy appends a `g_glut = 0.0` reset at the end of the
  neuron's update because no equation defines it
  (`ANNarchy/parser/AnalyseNeuron.py`, "Add a default reset behaviour for
  conductances"). So `g_glut` is a pure single-step accumulator: the `/dt`
  cancels the integration step, the value is added *once* to AMPA and *once* to
  NMDA, and it is cleared before the next step. One streamed cortical count to an
  SPN therefore drives two conductances; the same count to an FS neuron drives
  one, since the FSI model has no NMDA and no `g_glut` term.

Alongside each `TimedArray`, an `iter_memmap_spike_counts` iterator is opened on
the `.dat` with `chunk_size = update_time/dt`.

How many streams that is, per loop:

| streams | caudate | putamen |
|---|---|---|
| local compensation, one per `fitted_params.json` pair | 7 | 7 |
| cortical → dSPN / iSPN / FS, one per region with `N_eff > 0` | 18 (6 regions) | 21 (7 regions) |
| `CorticalInputs` → thal / gpe_arky / gpe_cp / stn | 24 (6 regions) | 28 (7 regions) |
| **total `TimedArray` + `CurrentInjection` pairs** | **49** | **56** |

The caudate has one region fewer everywhere because S1 contributes 0.00 to it
(§7.5) and zero-strength streams are skipped outright.

**The streaming itself.** `Microcircuit.update()` pulls the next `(R, 1100)`
block from every iterator, transposes it to `(1100, R)`, multiplies by that
stream's `mean_weights_by_type` scalar, and calls
`inp_population.update(rates=…, reset=True)`. The `reset=True` rewinds the
`TimedArray`'s internal timer so each chunk replays from block 0.
`Microcircuit.reset()` rebuilds every iterator from step 0.

That one multiplication is the entire count → current conversion, and **the two
stream families take their scalar from completely different places**:

| stream | `mean_weights_by_type` entry is | set when |
|---|---|---|
| compensation, `(pre_type, post_type)` | the mean of 10 000 draws from that pair's literature weight sampler (§7.3) | at cache build, restored from the state file |
| cortical, `(region, post_type)` | **a fitted parameter** — one per postsynaptic type, params 0–2 for the striatum and 3–6 for the BG populations (§11) | on every evaluation, by `set_opt_params_v07` |

So the compensation weight is fixed physiology and the cortical weight is what
the optimizer moves. The constructor seeds all cortical entries with a
placeholder `0.001`; if `set_opt_params_v07` did not overwrite them the model
would run at that placeholder rather than fail. Note also that one weight covers
all regions of a type, which is what makes the pre-summed cache layout in
`TODO.md` §3 lossless today.

**Where in the stream a given simulation sits.** The cache is consumed strictly
front-to-back, and `get_loss.rewind_inputs()` rewinds every iterator to step 0 at
the start of each evaluation. So the two run types read different parts of it:

- the firing-rate probe (`Spikes10s`) rewinds and then simulates 9900 ms, i.e. it
  always replays the **first 9900 ms** of the cache. This is why a cache must
  cover the probe even when `--n-trs` is small.
- a BOLD run (`get_BOLD_full`) rewinds, spends the 2310 ms ramp-up on the **first
  TR** of the cache, starts the BOLD monitors, and runs the remaining
  `duration − 2310 ms`. The BOLD signal therefore begins at cache TR 1, not TR 0.

This is also why **v07 can only be simulated in whole `update_time` chunks**.
`get_loss.simulate_model()` refuses a duration that is not a multiple of 110 ms,
calling every stream's `update(run_simulation=False)` and letting the last call
run `simulate(update_time)`. `update_time = 110.0` was chosen because it divides
the TR (2310 ms, 21 chunks), the full run, and the 9900 ms firing-rate probe; the
earlier 100 ms divided the full run and the probe but **not the TR**, which is
why the 2310 ms ramp-up would have failed on the first call.

An iterator that runs off the end raises `StopIteration` rather than looping or
zero-padding, so simulating past `n_steps` fails loudly. That is the same
`n_steps` the cache load checks enforce (§7.6).

---

## 8. `CorticalInputs`

`cortical_inputs.py`. The same cortical machinery of §7.5, applied to the four BG
populations that receive cortical drive: `thal`, `gpe_arky`, `gpe_cp`, `stn`,
with 1000 / 500 / 500 / 500 afferents each. There is no lattice, no connectivity,
no compensation — only the drive. Each of the four populations has 100 neurons
(§5), so a stream here is a `(100, n_steps)` matrix.

Receiver type labels are inferred by lower-cased matching of the population name
against the keys of `N_cortical_inputs_dict` (`CorticalInputs._infer_type()`),
which is why `BGM_v07` builds that dict from `pop.name` rather than from
literals.

Three differences from `Microcircuit` matter:

- **The streams use the flat split of §7.4 (1c), with every overlap at zero.**
  `shared_fraction_dict` is a *required* per-population argument (no default),
  and `parameters.py` sets all four entries to 0.0, so
  `simulate_receiver_counts_homogeneous_to_memmap` splits each receiver's `N`
  afferents into `round(f·N) = 0` shared and `N` private — receivers of the same
  cortical region share no presynaptic neurons at all, where the striatal
  streams share 1.4 %. This does *not* make the inputs independent over time —
  every receiver of a region is still driven by the same `p(t)`, so the slow,
  BOLD-derived co-fluctuation is fully present. What is removed is only the
  extra, within-bin correlation on top of it. (`TODO.md` §26: zero is certainly
  wrong physically; there is simply no measurement to put there.)
- **The target is always `ampa`**, with none of the per-type dispatch of §7.7 —
  correct, since all four are `Izhikevich2003NoisyBaseNonlin` populations with a
  single excitatory conductance and no `g_glut` term, so the streamed value lands
  on the decaying `g_ampa` directly.
- **Each population is drawn alone.** `Microcircuit` draws all three striatal
  receiver types of a region jointly from one shared axon pool (§7.5), which is
  what fixes their cross-type overlaps; here each of the four populations gets
  its own independent generator call per region, so there is no cross-population
  structure at all — consistent with every overlap being 0 anyway.

Everything else — the per-TR expansion, `N_eff = round(proportion · N_total)`
with zero-skip, the shared-modulation parameters `r_sc` / `tau_c_ms` /
`t_meas_ms` wired from `mc.cortical_correlation` / `mc.correlation_timescale_ms`
/ `mc.correlation_window_ms` (all 0 / 0 / `None`), the float64 `(R, n_steps)`
memmaps, the `TimedArray` + `CurrentInjection` pair per stream, the fitted
`mean_weights_by_type` multiplication in `update()`, and `reset()` — is the
same.

**The cache validation mirrors the Microcircuit's** since 2026-08-13
(`TODO.md` §29, resolved; it used to be strictly weaker).
`CorticalInputs._save_cortical_input_state` records `shared_fraction_dict`,
`cortical_correlation`, `correlation_window_ms`, `correlation_timescale_ms`
and the per-stream statistics, and `_load_cortical_input_state` compares them
— plus the stored `name` against the loop, `dt`, `n_steps`, `dbs_condition`,
the resolved rate path, the proportions dict, `N_cortical_inputs_dict`, the
stream keys against what the current configuration would generate, and each
stream's receiver count against the population size. Any absent field raises.

The smallest stream is M1 → gpe/stn in the caudate: `round(0.02 · 500) = 10`
presynaptic neurons. It survives the zero-skip, but a bin can then only take the
values 0–10, so that stream is far coarser than the 3150-source striatal ones.

`snr` and `gpe_proto` are deliberately **not** in the list. They receive no
cortical drive; their excitation is a fitted baseline current instead (§11).

---

## 9. Where DBS enters

Between model construction and the BOLD monitors. `add_dbs_mechanisms` rewrites
the equations of six putamen populations and six putamen projections;
`DBSstimulator(auto_implement=False)` supplies the pulse train; `on()` is called
pre-compile in the on condition.

Not repeated here — `DBS.md` covers the added equation terms, the pulse
definition, what `on()` writes, why `excluded_populations_list` is load-bearing,
and the known limitations. Two facts from that document are worth restating
because they shape this one:

- **`model_creation_kwargs["dbs"]` is not the stimulation.** It selects the
  cortical rate file and the cache directory. Nothing about the equations,
  weights or connectivity depends on it.
- **The caudate loop and the entire striatal microcircuit carry no DBS terms.**

---

## 10. Compile

```python
model_dict["caudate"].compile()
```

One call for the whole network. What it compiles, counted from a `report()` of
the result:

| | caudate | putamen | shared | total |
|---|---|---|---|---|
| striatal populations | 3 (486 + 485 + 29) | 3 (486 + 485 + 29) | — | 6 |
| BG populations | 6 × 100 | 6 × 100 | — | 12 |
| `TimedArray` input populations | 49 | 56 | — | 105 |
| BOLD model populations | — | — | 7 | 7 |
| **populations** | | | | **130** |
| BGM projections | 28 | 28 | — | 56 |
| microcircuit local projections | 7 | 7 | — | 14 |
| `CurrentInjection` input projections | 49 | 56 | — | 105 |
| projections into the BOLD models | — | — | 18 | 18 |
| **projections** | | | | **193** |

Two global `Constant`s exist, `dbs_pulse_frequency_Hz = 125` and
`dbs_pulse_width_us = 100`, together with the single global function `pulse()` —
all three from `DBSstimulator`, in both conditions.

The compile folder name is `bgm_v07_<dbs>[_<appendix>]`, passed at `BGM(...)`
construction; `--compile-appendix` gives each optimizer individual its own so
parallel jobs do not race. The on-disk path is
`annarchy_folders/bgm_v07_<dbs>[_<appendix>]`: CompNeuroPy's `compile_in_folder`
prepends `annarchy_folders/` by plain string concatenation
(`model_functions.py`), and ANNarchy then resolves the result relative to the
cwd — so the name must be relative and the process must `chdir` first; an
absolute name is concatenated into a nonsense path and fails.

To dump what was actually built:

```bash
python get_loss.py --dbs off --model-version v07 --n-trs 5 \
  --cache-dir <abs path>/mc_ci_cache_5tr \
  --compile-appendix report --report /tmp/report_v07_off.md
```

`--report` compiles and then calls ANNarchy's `report()` before anything is
simulated, so the output describes the real network including the DBS retrofit.
It writes `.md` or `.tex` only, and the Markdown carries LaTeX math intended for
pandoc rather than for GitHub.

### The off and on networks are the same network

`DBS.md` asserts this; two generated reports measured it once. (The report files
themselves were not retained — only the `annarchy_folders/bgm_v07_*_report_*`
compile folders survive — and regenerating them requires a cache, which none
currently exists.) Running `--report` for both conditions gave files of
**identical size**, with identical populations, projections, targets,
connectivity patterns, monitors, neuron models and synapse models. Everything
that differed is a parameter *value*, and every one of them is a DBS parameter.

Where those parameters actually live, verifiable from `dbs.py` today:

| parameter | set by `on()` on |
|---|---|
| `dbs_on` | all 6 putamen BG populations (an array on `stn`, `1` elsewhere) |
| `dbs_depolarization` | `stn` only — explicitly zeroed on every other population |
| `antidromic`, `antidromic_prob`, `prob_axon_spike` | `stn` (efferent branch), `gpe_proto` (afferent branch), `snr` (passing-fibre pre) |
| `p_axon_spike_trans` | all 6 putamen projections — the four `stn` efferents and `gpe_proto__stn` at 1, `snr__thal` at `passing_fibres_strength` |

The reports showed `p_axon_spike_trans` differing on only **5** projections
because they were generated with all three fitted DBS strengths at 0 (`--report`
needs no parameter vector): at `passing_fibres_strength = 0` the sixth,
`snr__thal`, is indistinguishable from off. For the same reason
`dbs_depolarization` and `prob_axon_spike` read 0 in both files; with a real
vector more values would differ, but still only values.

The reports also confirmed the footprint independently: of the 18 non-input BG
and striatal populations, **exactly six carry any `dbs_*` parameter at all** —
`stn`, `snr`, `gpe_proto`, `gpe_arky`, `gpe_cp` and `thal`, all `:putamen`. No
caudate population and no microcircuit population has one.

One thing this does **not** show: "identical" here means structure, equations
and parameters — **not simulation output**. Retrofitting DBS adds random
variables, and ANNarchy's RNG is one global stream, so the numbers every
population receives shift, including in the caudate loop that carries no DBS
terms at all (`CLAUDE.md`).

---

## 11. What is still not set at compile time

Compiling does not finish the model. The BOLD sampling period is fixed once
straight after `compile()`; the optimized parameters are written again at the
start of **every evaluation**, and the last part of this section explains why
they have to be.

**The BOLD sampling period.** `BoldMonitor` records every `dt` by default —
`report()`, which runs before the change, duly shows all seven monitors at period
0.1 ms. `bold_monitor._monitor.period = TR_S * 1000.0` is set immediately after
compile. Without it a full run stores 7.16 million samples per region, several GB
per process, of which only every 23 100th is used.

**The optimized parameters**, applied by `get_loss.apply_opt_params()` at the
start of every experiment run. v07 takes **19**: 9 drive parameters and 10 weight
cluster scalings.

| index | meaning | written to |
|---|---|---|
| 0–2 | cortical drive weight for dSPN, iSPN, FS | `mc.mean_weights_by_type[(region, type)]` for every region |
| 3–6 | cortical drive weight for thal, gpe_arky, gpe_cp, stn | `ci.mean_weights_by_type[…]` likewise |
| 7–8 | baseline current for snr, gpe_proto | `base_mean` on the population |
| 9–18 | one scaling factor per projection cluster | `proj.w` = CSV weight × factor |

**Parameters 0–6 never touch an ANNarchy object.** They are entries in plain
Python dicts that `Microcircuit.update()` and `CorticalInputs.update()` multiply
the streamed spike counts by on each chunk. That is also why they cannot be read
back from a compiled network or from a `report()`.

Under `--dbs on` the vector grows: base + 3 DBS parameters, or the staged layout
base + one putamen-only scaling per cluster + 3 DBS. **The DBS parameters are
always the last three** — they used to be read at fixed indices 21–23, which only
lined up with v08's 21.

**The reset trap.** `Population.__setattr__` writes to `pop.init` while the
population is uninitialized and to the C++ instance afterwards, and
`Population.reset()` is `self.set(self.init)`. So anything assigned post-compile
silently reverts at the next reset — and `CompNeuroExp.reset()` defaults to
`parameters=True`. This is exactly how DBS-on evaluations once ran with DBS
switched off, with no error and a plausible loss file. `Spikes10s.run()` is
ordered accordingly: `reset()` → `rewind_inputs()` → `apply_opt_params()` →
`simulate_model()`, so the optimized values are re-applied after every reset.
`get_loss.assert_dbs_state()` is the guard for the DBS half of the same problem.
