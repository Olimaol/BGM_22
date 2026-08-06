# How the v07 model is created

Step-by-step account of what happens between `setup()` and `compile()` when
`get_loss.py --model-version v07` builds the model, and of where every number in
the compiled network comes from. Read alongside `CLAUDE.md` (orientation),
`PLAN.md`, `TODO.md` and `DBS.md` (the DBS mechanism, which this document only
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
| `BOLD_optimization/parameters.py` | `get_loss.py` throughout | `dt`, seed, durations, `update_time`, lattice size, cache dir, the two data paths | hand-maintained |
| `striatal_microcircuit_requirements/connectivity_parameters/connectivity_fit_data/fitted_params.json` | `Microcircuit.__init__` | 7 pre→post pairs, each an amplitude and a σ in µm | `connectivity_fit_run.py` (skopt `gp_minimize`), BGM_22 commit `a59a5eb` |
| `striatal_microcircuit_requirements/cortical_firing_rates/cortical_firing_rates_data/firing_rates_matlab_condition-{on,off}.npz` | `Microcircuit`, `CorticalInputs`, and `get_loss.infer_max_sim_time_ms` | 9 regions × `{_time, _rate}`, each 310 values | `cortical_drive_by_bold_run.py` (SPM HRF deconvolution, needs MATLAB), BGM_22 commit `76d3867` |
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
spaced 2.31 s apart — one TR. `_rate` is in Hz.

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
| `mc.name` | `caudate` / `putamen` | selects the cortical proportion mix and names every mc object |
| `mc.nx`, `mc.b` | 10, 10 | 1000 lattice sites |
| `dbs` | `on` / `off` | **selects the rate file and cache dir only** — see `DBS.md` |
| `timestep` | 0.1 | |
| `t.duration` | inferred above | |
| `update_time` | 110.0 | the chunk size the inputs are streamed in — see §7.7 |
| `mc.storage_dir`, `ci.storage_dir` | `<cache-dir>/{mc,ci}_<loop>_cache_<dbs>` | |
| `mc.seed`, `ci.seed` | 42 | numpy RNG, independent of ANNarchy's |
| `mc.fitted_params_path`, `mc.cortical_rate_path` | the two data files | |
| `ci.n_thal` / `n_gpe_arky` / `n_gpe_cp` / `n_stn` | 1000 / 500 / 500 / 500 | cortical afferents per receiver neuron |

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
`dbs_pulse_*` constants. `dbs_stimulator.on()` runs here, **before** compile, so
the on-state becomes the compile-time state that every later `reset()` restores.

Both conditions run this block. `DBS.md` has the complete account, including why
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
each population by its share of the pooled neuron count
(`BoldMonitor.py:113-122`).

**No striatal scale factors in v07.** `Microcircuit` already sizes dSPN/iSPN/FS by
the del Rey et al. (2022) proportions (`microcircuit.py:180`), so the size-
proportional default *is* the del Rey weighting and passing it explicitly would
only restate it. The explicit `props_delRey` factors are therefore applied in v08
only, where all three striatal populations have 100 neurons and the default would
weight them equally. The two are not bit-identical: the default uses the integer
cell counts after `int()` truncation and the remainder fix-up
(`microcircuit.py:237-242`), which at `nx = b = 10` gives 486 / 485 / 29 →
0.486 / 0.485 / 0.029 against the exact 0.485327 / 0.485327 / 0.029345.

**The GPe factors have no recorded source.** They appear only in
`get_loss.py:1235` and `test_microcircuit_bgm.py:472`, both introduced whole in
commit `76d3867`, uncited; they sum to 0.77, not 1, which would fit fractions of
all GPe cells with the rest belonging to types the model does not have. These do
change the pooling, since `gpe_proto/arky/cp` are all 100 neurons. The source
still has to be found.

**The input variable differs by population family.** BGM populations expose the
total current as `I`; the Humphries striatal populations expose it as `I_v`. Cau
and Put therefore map `I_CBF` to `I_v`, everything else to `I`.

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

`BGM_v07` itself sets **no parameter, no weight, no connectivity**. It only
instantiates bare `Population` and `Projection` objects with names. Everything
numeric arrives afterwards, from the CSV. The sequence in `bgm.py → BGM.create()`
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

Value parsing in `BGM._get_params()`: rows starting with `###` are section
headers and skipped; **empty cells are skipped**, which is how the v07 column
omits `str_d1.size` and the intra-striatal projection rows; a value is an `int`
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

**The striatal sizes are not parameters.** The v07 CSV column deliberately has no
`str_d1.size` / `str_d2.size` / `str_fsi.size` rows; those counts fall out of the
lattice (§7.1).

`exp_input` is left at its default `0.0` for every v07 population, so **no v07
neuron carries the `Exponential(lambda)` drive term**. The cortical drive arrives
entirely through `CurrentInjection`s instead. This is the sharpest structural
difference from v08.

### The 28 projections

All created with `connect_fixed_number_pre(number=10)`; the table gives the
target, the CSV weight and the CSV delay in ms. The `w` column is later
multiplied by an optimized per-cluster factor (§11), and the cluster each
projection belongs to is given in the last column.

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
baked into the equation string.

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

Two things `stabilize=True` does, both visible in the `I` line: it makes
excitation **current-based** with a fixed 50 mV driving force (`-g_ampa*(-50)`
rather than `-g_ampa*(v - E_ampa)`), and it divides each conductance term by
`1 + g*dt`. That divisor is what bounds the effective conductance, and it is
where the optimizer's weight ceilings come from (§11).

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

iSPN differs in two places — the AMPA term carries the D2 attenuation factor and
the quadratic term is scaled:

```
I_v       = g_ampa * E_exc * (1 - beta_2 * phi_2) / (1 + g_ampa * dt / C) + …
C * dv/dt = k * (1 - alpha * phi_2) * (v - v_r) * (v - v_t) - u + I_v
```

Both: spike `v >= v_peak`, reset `v = c; u = u + d`.

**With `phi_1 = phi_2 = 0` every dopamine term above is inert** — `phi_1*c_da*(…)`
is zero, `(1 + beta_1*phi_1)` and `(1 - beta_2*phi_2)` are one. The dopamine
machinery is present in the equations but not used by this project.

| parameter | dSPN | iSPN | note |
|---|---|---|---|
| `tau_ampa` / `tau_nmda` / `tau_gaba` | 6 / 160 / 4 | 6 / 160 / 4 | ms |
| `E_ampa` / `E_nmda` / `E_gaba` | 0 / 0 / −60 | 0 / 0 / −60 | mV |
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

**None of these are in `parameters.csv`.** The striatal populations belong to the
Microcircuit, so `_set_params` skips them entirely; every value above is a
CompNeuroPy class default.

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
| `E_gaba` / `E_exc` | −60 / 50 mV |
| `C` / `k` | 80 pF / 1.0 |
| `v_r` / `v_t` / `v_b` / `v_peak` | −70 / −50 / −55 / 25 mV |
| `a` / `b` / `c` / `d` | 0.2 / 0.025 / −60 / 0 |
| `eta` / `epsilon` | 0.1 / 0.625 |
| `phi_1` / `phi_2` | 0.0 / 0.0 |

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

Each accepted connection draws its weight from a `CombinedSampler`
(`get_weights.py`) over a pair-specific mixture of literature distributions, each
component carrying its own weight in the mixture:

| pair | mixture |
|---|---|
| SPN → SPN | uniform 0.18–0.37 (w 87) + truncated Gaussian μ 0.42, σ 0.25 (w 23) + truncated Gaussian μ 0.75, σ 0.57 on [0.07, 1.81] (w 26) |
| FS → SPN | 41-bin empirical histogram over 0–58.57 (w 75) + truncated Gaussian μ 1.57, σ 2.68 (w 31) + truncated Gaussian μ 3.84, σ 3.04 on [0.64, 8.14] (w 9) |
| FS → FS | single truncated Gaussian μ 1.1, σ 1.5 on [0, 10.1] |

Note the scale: **FS→SPN weights are one to two orders of magnitude larger than
SPN→SPN weights**, which is what makes the 29 FS neurons matter at all. Weights
land in a `lil_matrix` per pair, indexed by **type-local** indices.

Cached in `connectivity/connectivity_state.pkl` plus seven
`connectivity/weights_<pre>_<post>.npz`. On load, `n_total`, `nx` and the full
type vector must match, all seven weight files must exist, and **the numpy RNG
bit-generator state is restored** so that a load run continues on the same random
stream a build run would have been on.

### 7.3 Missing-GABA compensation — built once, cached

The simulated cube contains only the inner sphere of each neuron's potential
presynaptic partners. Everything outside is replaced by synthetic input, in three
steps (`Microcircuit._missing_local_input()`):

1. **How much is missing.** For each pair, integrate the kernel over the outer
   shell from the neighbourhood radius `Rin` to `Rout = 3σ_max`:

   ```
   E_outer = 4π ρ ∫_{Rin}^{Rout} p(r) r² dr
   ```

   with `ρ = props[pre] · density`. The same integral over `[0, Rin]` gives
   `E_inner`, used only as a sanity check against the connections actually made.

   `Rin` is the neighbourhood radius, clamped to 113.8 µm for all three post
   types (§7.2); `Rout = 3σ_max(post)`, which is 568 µm for FS and ~1.2 mm for
   both SPN types. The gap is large, and so is what falls into it:

   | pair | `E_inner` | `E_outer` | `N_eff` | pre rate | mean counts per receiver per dt |
   |---|---|---|---|---|---|
   | FS → FS | 1.9 | 12.4 | 12 | 10.5 Hz | 0.013 |
   | FS → dSPN | 8.8 | 500.0 | 500 | 10.5 Hz | 0.525 |
   | FS → iSPN | 9.6 | 25.3 | 25 | 10.5 Hz | 0.026 |
   | dSPN → dSPN | 26.2 | 1568.5 | 1568 | 25.0 Hz | 3.920 |
   | dSPN → iSPN | 25.0 | 1500.3 | 1500 | 25.0 Hz | 3.750 |
   | iSPN → dSPN | 31.8 | 505.4 | 505 | 33.0 Hz | 1.667 |
   | iSPN → iSPN | 32.5 | 948.1 | 948 | 33.0 Hz | 3.128 |

   Read the first two columns together: **a dSPN neuron gets ~26 simulated dSPN
   afferents and ~1568 synthetic ones.** The lattice is a small window on the
   striatum and most of each neuron's GABAergic input is the compensation, not
   the simulated circuit. That is worth keeping in mind whenever a striatal rate
   comes out wrong — `TODO.md` §4 is about exactly this loop being untested.

2. **How much of it is shared between two receivers.** Two nearby neurons see
   largely the same distant presynaptic pool. For a receiver pair separated by
   `d`:

   ```
   E_shared(d) = 2π ρ ∫_{Rin}^{Rout} r² ∫_0^π p(r) p(r_B) sin θ dθ dr,
                 r_B = √(r² + d² − 2 r d cos θ)
   ```

   evaluated at 50 distances up to the half space-diagonal and stored as a cubic
   interpolant of `f(d) = E_shared(d) / E_outer`. It is 0 for `d ≥ 2·Rout`.
   Applied to the actual receiver positions this yields, per pair, an `(R, R)`
   float32 matrix of shared fractions with unit diagonal, clipped to [0, 1].

3. **Simulate it.** Per pair, `N_eff = int(round(E_outer))` presynaptic sources
   firing at `firing_rate_dict[pre]` are drawn into a memmap, correlated between
   receivers according to the shared-fraction matrix of step 2 — the mechanics
   are §7.4. `correlation_dict[pre]` enters as that call's `rho`: the pairwise
   spike correlation reported for those cells is reproduced by letting the whole
   outer pool's rate fluctuate together, redrawn every step from a Beta with the
   right mean and a variance of `rho·p·(1−p)`. A pair whose `N_eff` rounds to 0
   is skipped and gets no stream at all; with the current parameters all seven
   survive.

Finally the **mean** of 10 000 samples from each pair's weight sampler is stored
as `mean_weights_by_type[(pre, post)]`. That scalar is what the streamed spike
counts get multiplied by at simulation time (§7.7): the stream says *how many
spikes arrived*, the scalar says *what one of them is worth*. So the whole outer
shell — up to 1568 neurons — collapses onto one mean synaptic weight, where the
inner shell keeps its individually sampled weights (§7.2). The compensation is
right on average and carries none of the weight heterogeneity.

The rates driving this come from `parameters.py: mc.firing_rate_dict`, threaded
through `v07_model_creation_kwargs` so evaluation and cache building cannot
disagree: `{FS: 10.5, dSPN: 25.0, iSPN: 33.0}` Hz. dSPN/iSPN are the parkinsonian
**medication-off** state of Liang et al. 2008, measured in MPTP monkeys. FS has no
parkinsonian-primate measurement at all: it is the normal-primate level times a
chronic dopamine-depletion factor of 1.0 taken from rodents. The derivations, the
assumptions they rest on and the alternatives rejected are in
`experimental_data/activity_striatum/README.md`. Alongside them,
`correlation_dict = {FS: 0.06, dSPN: 0.004, iSPN: 0.004}` (Adler et al. 2013).

Cached in `inputs/missing_input_state.pkl`. On load, the pair-key set, `dt`,
**`n_steps` exactly**, `firing_rate_dict` and `correlation_dict` must match, every
`.dat` must exist, and the RNG state is restored. A state file written before the
last two were recorded is refused rather than trusted.

### 7.4 How one spike-count stream is drawn

Everything cached in §7.3 and §7.5 is the same kind of object, produced by the
same code — `striatal_microcircuit/spike_input_cortex.py`. Understanding it once
covers the missing-GABA streams, the striatal cortical streams and all of
`CorticalInputs` (§8).

**What a stream is.** One `(R, n_steps)` matrix for one `(pre, post)` pair. Row
`i` is receiver neuron `i` of the postsynaptic population; column `t` is one
simulation step of `dt = 0.1 ms`. The entry is a **count**: how many of that
pair's presynaptic neurons spiked into receiver `i` during that 0.1 ms bin. It is
not a rate, not a current, and not a spike train — the presynaptic neurons are
never represented individually, only their per-bin total. What turns a count into
a current is a single mean weight, applied later at simulation time (§7.7).

**What it is standing in for.** `N_eff` presynaptic neurons firing at some rate.
Everything below is a way of drawing `Binomial(N_eff, p)` per receiver per bin
while imposing the right amount of *sharing* — two receivers do not each own a
private pool of `N_eff` neurons, the pools overlap, and that overlap is what
makes their inputs correlated. `shared_input` is that overlap: 1.0 would mean two
receivers see the identical presynaptic population, 0.0 that they see disjoint
ones.

**The three stages** (`ReceiverSimulator`, called via
`simulate_receiver_counts_*_to_memmap`):

1. `get_global_p(dt, rate, rho)` → one probability per time step, shared by all
   receivers. This is where a firing rate becomes a spike probability:
   `p(t) = rate(t) · dt / 1000`. Two branches, and which one runs depends only on
   whether `rate` is an array or a scalar:
   - **`rate` is a time series** (all cortical streams): `p(t) = rate(t)·dt/1000`
     exactly, and **`rho` is ignored entirely** — the function returns before it
     is read. The cortical callers pass `rho = 0.0` anyway, so this is harmless
     today, but a non-zero value there would be silently discarded rather than
     applied.
   - **`rate` is a scalar** (all missing-GABA streams): with `rho = 0` a flat
     `p`; with `rho > 0`, `p(t)` is redrawn every step from a Beta with mean `p`
     and variance `rho·p·(1−p)`. This is the `correlation_dict` entry — a
     population-wide rate fluctuation shared by every receiver of the pair.
2. `generate_p_matrix(global_p, shared_input, concentration)` → an `(R, n_steps)`
   matrix of *per-receiver* probabilities. Correlated uniforms come from a
   Gaussian copula whose correlation matrix **is the shared-fraction matrix**,
   and each uniform is pushed through `Beta.ppf(·, p·c, (1−p)·c)`.
3. `simulate(p_matrix, shared_input)` → the counts, as
   `Binomial.ppf(u, N_eff, p_matrix)` with a **second, independently drawn** set
   of copula uniforms at the same correlation matrix.

So sharing is injected twice — once into the probabilities, once into the draw —
and the shared fraction is used directly as a Gaussian correlation coefficient.
That is an approximation: the correlation you get out in the counts is not the
number you put in. It is in the right direction and monotone, nothing more.

**What is *not* correlated: time.** Every bin is drawn independently of every
other. `shared_input` and `rho` both act *across receivers within one bin*.
Nothing in the generator produces autocorrelation, burst structure or refractory
effects — the only structure along the time axis is whatever the rate series
itself carries, which for the cortical streams is one value per 2.31 s TR.

| argument | missing-GABA (§7.3) | cortical, striatum (§7.5) | cortical, BG (§8) |
|---|---|---|---|
| `N_eff` | `round(E_outer)`, 12–1568 | `round(proportion · N_total)` | `round(proportion · N_total)` |
| `rate` | scalar, `firing_rate_dict[pre]` | per-step array from the drive | per-step array from the drive |
| `rho` | `correlation_dict[pre]` | 0.0 (ignored) | 0.0 (ignored) |
| `shared_input` | `(R, R)` matrix, `f(d)` | scalar 0.014 | scalar 0.0 |
| `concentration` | default 1.0 | default 1.0 | default 1.0 |

**`concentration` is the one to watch.** It is the Beta dispersion in stage 2, it
is never passed by any caller, and in both call sites an explicit
`concentration=1000.0` sits commented out one line below. At the default of 1.0,
`Beta(p·1, (1−p)·1)` with `p ≈ 5·10⁻⁴` has the correct mean but a variance of
`p(1−p)/2` — a standard deviation ~30× its own mean. The per-receiver
probability is therefore not "p with jitter"; it is almost always ≈ 0 and rarely
≈ 1. Measured on a cortical stream (`N_eff = 3150`, 5 Hz, `dt = 0.1 ms`, so 1.575
expected counts per bin):

| | expected | mean | SD | max | bins that are exactly 0 | Fano | realized corr (target 0.014) |
|---|---|---|---|---|---|---|---|
| `concentration = 1.0` | 1.575 | 1.540 | 48.65 | 3131 | 99.6 % | 1537 | 0.0002 |
| `concentration = 1000.0` | 1.575 | 1.574 | 2.55 | 42 | 49.1 % | 4.2 | 0.0083 |

and on a missing-GABA stream (`N_eff = 1568`, 25 Hz, `rho = 0.004`, shared
fraction 0.5 for the illustration):

| | expected | mean | SD | max | bins that are exactly 0 | Fano | realized corr (target 0.5) |
|---|---|---|---|---|---|---|---|
| `concentration = 1.0` | 3.920 | 3.793 | 54.44 | 1567 | 98.1 % | 781 | 0.15 |
| `concentration = 1000.0` | 3.920 | 3.907 | 5.82 | 75 | 37.6 % | 8.7 | 0.85 |

Read the `max` column: at the default, single 0.1 ms bins in which **the entire
presynaptic pool fires at once** are routine. The drive is delivered as rare
enormous conductance jumps rather than a dense stream, and the shared fractions
of §7.3 are largely washed out on the way. The mean is right throughout, which is
why nothing downstream ever complained. Nothing here has been validated against a
target input statistic — see `TODO.md` §22.

**How it is written.** `simulate_receiver_counts_*_to_memmap` never materialises
the full `(R, n_steps)` array. It slices the time axis into chunks sized so the
four internal float64 `(R, chunk)` arrays stay near 128 MB, builds a fresh
`ReceiverSimulator` per chunk on the **same** `rng`, and writes each result
straight into the memmap. Chunking is therefore invisible in the output: the
random stream continues across boundaries, and the boundaries are not aligned to
anything the simulation later does.

### 7.5 Cortical input — built once, cached

`Microcircuit._simulate_cor_input_spike_counts()` turns the BOLD-derived cortical
drive into streams, in three steps.

**Step 1: the rate series.** The `.npz` holds one `<region>_rate` array of 310
values, one per TR, per cortical region (§2). The step is inferred from the
matching `<region>_time` array — 2.31 s — and each value is repeated
`TR/dt = 23 100` times, then truncated to `n_steps`. A drive finer than `dt`, or
one whose spacing is not an integer multiple of it, raises. Every region's series
averages exactly 5 Hz — they are normalised to it — and spans roughly
0.14–119 Hz across regions, so the drive is a slow, strongly modulated envelope
around a common mean, not a stationary rate.

**Step 2: how many presynaptic neurons each region supplies.** Every receiver
type is assigned a total afferent count, and each region gets its share:

```
N_eff = round(cortical_proportions_dict[region] · N_cortical_inputs_dict[receiver])
```

with `N_cortical_inputs_dict = {FS: 2800, dSPN: 7000, iSPN: 7000}`. The
proportions sum to 1, so the per-region counts sum back to the total — a caudate
dSPN's 7000 cortical afferents are split 3850 dlPFC, 1260 PMd, 1050 preSMA, 420
SMA, 280 PMv, 140 M1, 0 S1. A region with `N_eff = 0` is skipped and gets no
stream, which is the only reason the two loops differ in stream count.

**Step 3: the draw**, for **dSPN and iSPN only**, one stream per region per
receiver type, through §7.4 with `shared_input = shared_fraction = 0.014`
(Kincaid et al. 1998) and `rho = 0.0`. Concretely for dlPFC → caudate dSPN:
`N_eff = 3850` sources at a mean 5 Hz give `p = 5·10⁻⁴` and 1.925 expected counts
per 0.1 ms bin per receiver — see §7.4 for what the realised distribution around
that mean actually looks like.

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

S1 contributes nothing to the caudate, so the caudate loop has one cortical
stream fewer per receiver type than the putamen loop.

**They are defined in exactly one place**, `BOLD_optimization/parameters.py`
under `cortical_proportions_dict`, and reach the model as
`mc.cortical_proportions_dict` in the creation kwargs (§3.3), which
`model_creation_functions.py` hands to both `Microcircuit` and `CorticalInputs`.
Neither class has a default any more — both call
`spike_input_cortex.validate_cortical_proportions()`, which raises if the mapping
is missing, negative, or does not sum to 1. The reason for the strictness is that
the *same* numbers weight the mix producing `caudate_rate`/`putamen_rate` inside
the cortical rate `.npz` (`cortical_drive_by_bold.py`, which imports them from
`parameters.py` too), so a second copy could silently disagree with the file the
model is driven by. **Changing them means regenerating that `.npz`** —
`cortical_drive_by_bold_run.py`, which needs MATLAB and records the proportions
in `__data_raw_meta__`.

The values come from quantitative macaque retrograde tracing — Borra et al. 2022
for the caudate/putamen region groups, Borra et al. 2021 for the per-area split
of the motor putamen — nudged toward the two ratios that survive the coarse
parcellation of the one human study reporting per-pathway percentages (Cacciola
et al. 2017). `TODO.md` §21 has the derivation, the alternatives considered, and
the sensitivity measurements; the short version is that the mixed drive is nearly
invariant to the choice (`corr` 0.98–0.995 against the previous hand-set numbers)
while the per-region stream sizes are not, and that putamen PMv is the one entry
worth arguing about (plausible range 0.10–0.24).

**FS cortical input is not drawn — it is derived**
(`Microcircuit._derive_fs_cortical_inputs()`). The FS streams never go through
§7.4 at all. Instead, per cortical region, each FS neuron's cortical input is the
weighted sum of the cortical counts of the SPNs *it projects onto*,

```
input_FS = W_{FS→dSPN} · counts_dSPN + W_{FS→iSPN} · counts_iSPN
```

rescaled per FS neuron by `N_FS / ((Σw_dSPN + Σw_iSPN) · N_SPN)`, then
Poisson-resampled and written chunkwise. The rationale is anatomical: an FS
neuron and the SPNs it inhibits sit in the same place and sample the same
cortical territory.

The rescaling looks region-blind but is not, and it is worth following once. For
region `r`, each SPN count has mean `N_eff(r)·p(t) = proportion(r)·7000·p(t)`, so
per FS neuron `i` the weighted sum has mean `(Σ_j w_ij)·proportion(r)·7000·p(t)`.
Multiplying by `2800/((Σ_j w_ij)·7000)` cancels both the weight sum and the 7000,
leaving `proportion(r)·2800·p(t)`. **Each region therefore delivers exactly its
own share of the 2800 FS target, with one scaling factor that does not depend on
the region** — and summed over regions the FS neuron sees 2800 afferents' worth
of drive, as intended.

Two consequences to be aware of. FS drive is not an independent quantity: it is a
deterministic linear function of the SPN streams plus independent Poisson noise,
so an FS neuron and its SPN targets share cortical fluctuation by construction.
And an FS neuron with **no** outgoing connections gets scaling factor 0 and hence
no cortical input at all, silently — the code masks the division rather than
raising. With 29 FS neurons on a periodic lattice this has not been observed, but
nothing checks it.

Cached in `inputs/cortical_input_state.pkl`.

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
- **`cortical_rate_path` is compared as a verbatim string.** `parameters.py`
  stores it relative (`../striatal_microcircuit_requirements/...`), so build and
  evaluation must be launched from the same working directory —
  `BOLD_optimization/`. See `TODO.md` §13.
- `dbs_condition`, `dt`, the proportions dict, `N_cortical_inputs_dict`,
  `shared_fraction` and the key set are all compared too.
- Any mismatch **raises**; nothing is silently rebuilt.

Sizes follow directly from the row counts. Across both loops the streams total
24 084 rows (caudate 11 342, putamen 12 742), so one DBS condition costs
`24 084 × n_steps × 8 B`:

| | `n_steps` | per DBS condition |
|---|---|---|
| `--n-trs 5` | 115 500 | 22.3 GB (20.7 GiB) — measured: 22.28 GB |
| full run, 310 TRs | 7 161 000 | 1 380 GB (**1.26 TiB**) |

A single full-length dSPN cortical stream is `486 × 7 161 000 × 8 B ≈ 27.8 GB` on
its own. Note that `mc_ci_cache_5tr/` holds **both** conditions and so measures
44.6 GB in total.

The `mc_ci_cache_dir` comment in `parameters.py` says "~138 GiB per DBS
condition". That is the estimate for the *planned* `uint16`, pre-summed layout
(`PLAN.md`, `TODO.md` §3), not for the caches that exist — see `TODO.md` §19.

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
earlier 100 ms divided none of them.

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

- **`shared_input = 0.0`.** The copula correlation matrix of §7.4 becomes the
  identity, so receivers of the same cortical region share no presynaptic
  neurons at all, where the striatal streams share 1.4 %. This does *not* make
  the inputs independent over time — every receiver of a region is still driven
  by the same `p(t)`, so the slow, BOLD-derived co-fluctuation is fully present.
  What is removed is only the extra, within-bin correlation on top of it.
- **The target is always `ampa`**, with none of the per-type dispatch of §7.7 —
  correct, since all four are `Izhikevich2003NoisyBaseNonlin` populations with a
  single excitatory conductance and no `g_glut` term, so the streamed value lands
  on the decaying `g_ampa` directly.
- **Nothing is derived.** Every one of the four gets its own drawn stream per
  region; there is no analogue of the FS derivation.

Everything else — the per-TR expansion, `N_eff = round(proportion · N_total)`
with zero-skip, `rho = 0.0` and `concentration` left at its default, the float64
`(R, n_steps)` memmaps, the `TimedArray` + `CurrentInjection` pair per stream,
the fitted `mean_weights_by_type` multiplication in `update()`, and `reset()` —
is the same, and so is the cache validation, plus one extra field: the stored
`name` must match the loop.

The smallest stream is M1 → gpe/stn in the caudate: `round(0.01 · 500) = 5`
presynaptic neurons. It survives the zero-skip, but a bin can then only take the
values 0–5, so that stream is far coarser than the 3150-source striatal ones.

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

The compile folder is `bgm_v07_<dbs>[_<appendix>]`, passed at `BGM(...)`
construction; `--compile-appendix` gives each optimizer individual its own so
parallel jobs do not race. **ANNarchy resolves
`compile(directory=…)` relative to the cwd**, so the name must be relative and
the process must `chdir` first — passing an absolute path joins it onto the cwd
and fails.

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

`DBS.md` asserts this; the two reports measure it. Running `--report` for both
conditions gives files of **identical size**, with identical populations,
projections, targets, connectivity patterns, monitors, neuron models and synapse
models. Everything that differs is a parameter *value*, and every one of them is
a DBS parameter:

| what differs | where |
|---|---|
| `dbs_on`, `dbs_depolarization`, `antidromic`, `antidromic_prob`, `prob_axon_spike` | the 6 putamen BG populations |
| `p_axon_spike_trans` | 5 putamen projections |

The report also confirms the footprint independently: of the 19 non-input
populations, **exactly six carry any `dbs_*` parameter at all** — `stn`, `snr`,
`gpe_proto`, `gpe_arky`, `gpe_cp` and `thal`, all `:putamen`. No caudate
population and no microcircuit population has one.

Two things this does **not** show. First, the reports were generated with all
three fitted DBS strengths at 0 (`--report` needs no parameter vector), so
`dbs_depolarization` and `prob_axon_spike` read 0 in both files and only
`dbs_on`, `antidromic` and `p_axon_spike_trans` visibly flip; with a real vector
more values would differ, but still only values. Second, "identical" here means
structure, equations and parameters — **not simulation output**. Retrofitting DBS
adds random variables, and ANNarchy's RNG is one global stream, so the numbers
every population receives shift, including in the caudate loop that carries no
DBS terms at all (`CLAUDE.md`).

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
