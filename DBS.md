# What DBS changes in the model

Reference for the DBS mechanism as it is implemented in CompNeuroPy and as it is
configured in this project. Written 2026-08-04 while making the DBS-on path
actually work. Read alongside `CLAUDE.md`, `PLAN.md` and `TODO.md`.

Everything cited here was checked against the code, not remembered. Line numbers
refer to `CompNeuroPy/src/CompNeuroPy/dbs.py` unless stated otherwise.

## Where DBS lives

The entire mechanism is `CompNeuroPy/src/CompNeuroPy/dbs.py` (1849 lines),
exported from `CompNeuroPy/__init__.py`. There are **no DBS neuron models,
synapse models, populations or projections anywhere** — `neuron_models/` and
`synapse_models/` contain no occurrence of `dbs`, `axon` or `antidromic`.

DBS is implemented by **rewriting the equation strings of existing neuron and
synapse models**, plus four primitives that live in ANNarchy itself and are
upstream, not part of our `timedarray-fastbuffer` patch: `axon_spike`,
`axon_reset`, `pre_axon_spike`, `axon_transmission`.

`ANNarchy_compneuro` contains nothing DBS-specific. Its two local commits
(`2a11e858` fast `TimedArray` buffer, `f215694e` rounding of schedule/period)
are unrelated.

Note the *other*, unrelated `dbs` flag: `model_creation_kwargs["dbs"]`, passed to
`Microcircuit(dbs_condition=...)` and `CorticalInputs(dbs_condition=...)`. It
changes no equation, weight or connectivity — it only selects the cortical
firing-rate file `firing_rates_matlab_condition-{on,off}.npz` and is written into
the cache state file for a consistency check.

## The two-stage structure

**Stage A — the mechanisms must be in the model before `compile()`.** They are
extra parameters, extra variables, and extra terms in the membrane equation.
Adding them is a source-level change to the neuron and synapse models.

**Stage B — `on()` / `off()` only write parameter values.** No structure changes
at run time. `on()` calls `_set_dbs_on`, `_set_depolarization`, `_set_axon_spikes`;
`off()` zeroes them.

There are two ways to accomplish stage A:

- `DBSstimulator(auto_implement=True)` — clears the whole network
  (`dbs.py:73` `cnp_clear`) and recreates every population and projection from
  introspected `__init__` kwargs. **This project does not use it**, because
  recreation rebuilds every population as a plain `ann.Population` and every
  projection through `_connector_methods_dict` (`dbs.py:8-21`), which has no
  `"Specific"` key — so any `TimedArray` or `CurrentInjection` either raises a
  `KeyError` or silently loses its specific class. v07 is full of both.
- `add_dbs_mechanisms(populations, projections)` — retrofits the terms onto the
  already-built objects before `compile()`, by swapping `neuron_type` /
  `synapse_type` and recomputing the five fields ANNarchy derives from them
  (`parameters`, `variables`/`attributes`, `functions`, `init`;
  `ANNarchy/core/Population.py:74-113`, `Projection.py:78-128`). Connectivity,
  object identity and `_specific_template` all survive, so `TimedArray`,
  `CurrentInjection` and every handle captured by `Microcircuit` /
  `CorticalInputs` stay valid. **This is what this project uses.**

## Exactly what is added to a spiking neuron model

`add_DBS_to_spiking_neuron_model` (`dbs.py:474-550`).

Five new parameters:

```
dbs_depolarization = 0 : population
dbs_on = 0                            # LOCAL — per neuron, this is how VTA coverage is expressed
antidromic = 0 : population
antidromic_prob = 0 : population
prob_axon_spike = 0 : population
```

Two new RNG variables, prepended so they are drawn once per neuron per timestep:

```
unif_var_dbs2 = Uniform(0.0, 1.0)
unif_var_dbs1 = Uniform(0.0, 1.0)
```

A term appended to **every** line containing `dv/dt` on the left:

```
 + pulse(t)*dbs_on*dbs_depolarization*neg(-90 - v)
```

**This term hyperpolarizes, despite the parameter being called
`dbs_depolarization`.** `neg(x)` is `#define negative(x) (x<0.0? x : 0.0)`
(`ANNarchy/generator/Template/BaseTemplate.py:1470`), so for `v > -90` the term is
`dbs_depolarization*(-90 - v) < 0`: a shunting pull toward −90 mV that does
nothing below −90 mV. `get_loss.py` says so inline. Bear this in mind when
reading a fitted `dbs_depolarization` value.

An axon-spike condition:

```
axon_spike = pulse(t)*dbs_on*unif_var_dbs1 > 1-prob_axon_spike
```

And an axon reset — the antidromic soma invasion:

```
v += ite(unif_var_dbs2 < antidromic_prob, dbs_on*antidromic*(-v + c), 0)
u += ite(unif_var_dbs2 < antidromic_prob, dbs_on*antidromic*d, 0)
```

i.e. a full Izhikevich spike reset (`v → c`, `u += d`) applied *without* the
neuron having crossed threshold and *without* entering refractoriness. This is
why the neuron model needs a `d` parameter at all.

`add_term_to_eq_line` (`dbs.py:552-578`) inserts before any `:` flag, so
`: init=...` and `: population` survive.

Rate-coded models get a parallel treatment (`dbs.py:620-683`): `axon_rate_amp`
instead of `prob_axon_spike`, a new `axon_rate = axon_rate_amp*dbs_on` variable,
and the same term on `dmp/dt` against −1 instead of −90. **It raises
`ValueError("No line with dmp/dt found…")` on any rate model without `mp`** —
which is exactly what a `TimedArray` is (`equations=" r = 0.0"`), and one of the
two reasons the input machinery must be kept out of the DBS footprint.

## Exactly what is added to a synapse model

Spiking (`dbs.py:763-809`):

```
p_axon_spike_trans = 0 : projection
unif_var_dbs = Uniform(0., 1.)
pre_axon_spike = g_target += ite(unif_var_dbs<p_axon_spike_trans, w*post.dbs_on, 0)
```

Three gates stack on an axon-spike-driven transmission: the per-synapse Bernoulli
`p_axon_spike_trans`, the normal weight `w`, and **`post.dbs_on`** — the
*postsynaptic* neuron's flag, so an afferent volley only reaches the neurons
inside the VTA.

Rate-coded (`dbs.py:811-858`): `pre.r` is replaced by `pre_rate` everywhere and

```
pre_rate = pre.r + p_axon_spike_trans*pre.axon_rate*post.dbs_on
```

## The pulse train

`_set_constants` (`dbs.py:1196-1213`), called from `DBSstimulator.__init__`:

```python
ann.Constant("dbs_pulse_frequency_Hz", dbs_pulse_frequency_Hz)
ann.Constant("dbs_pulse_width_us", self.dbs_pulse_width_us)
ann.add_function(
    "pulse(time_ms) = ite(modulo(time_ms*1000, 1000000./dbs_pulse_frequency_Hz) < dbs_pulse_width_us, 1., 0.)"
)
```

`modulo` is integer (`long(a) % long(b)`) and `t` parses to `(double(t)*dt)` in
ms. At this project's 125 Hz / 100 µs / dt = 0.1 ms the period is 8000 µs, so
**`pulse` is 1 for exactly one timestep every 8 ms**.

Because the rewritten equations reference `pulse(t)`, the stimulator must be
constructed in the **off** condition too — otherwise the function does not exist
and compilation fails. `DBSstimulator` is no longer an on-only object.

`axon_spikes_per_pulse` is converted to a per-timestep probability by
`np.clip(axon_spikes_per_pulse * 1000 * dt / dbs_pulse_width_us, 0, 1)`
(`dbs.py:1217-1232`). With this project's numbers that is the identity, so
`axon_spikes_per_pulse` *is* the per-pulse spike probability.

## What `on()` writes, and where

`_set_dbs_on` (`dbs.py:1755-1799`) — runs first:
- excluded populations → `dbs_on = 0`
- stimulated population → the 0/1 `dbs_on_array`
- every other population → `dbs_on = 1`

The array (`_create_dbs_on_array`, `dbs.py:1156-1196`) has
`rng.choice([ceil, floor])` of `proportion * N` ones, shuffled with
`np.random.default_rng(seed)`. **It is not `round(proportion*N)`** — assert
against `dbs_stimulator.dbs_on_array`, not against a recomputed count.

`_set_depolarization` (`dbs.py:1234-1259`) — `dbs_depolarization` on the
stimulated population only, 0 everywhere else. The somatic effect is STN-only.

`_set_axon_spikes` (`dbs.py:1261-1388`) — first `_deactivate_axon_DBS()` zeroes
everything, then:

*Orthodromic* (`dbs.py:1430-1532`):
- `efferents` → for each `ann.projections(pre=stim_pop)`: `axon_transmission = 1`,
  `p_axon_spike_trans = 1`, and `proj.pre.prob_axon_spike = prob`
- `afferents` → for each `ann.projections(post=stim_pop)`: the same, but
  `proj.pre` is now each afferent population, so those populations start emitting
  axon spikes
- `passing_fibres` → per projection in `passing_fibres_list`:
  `p_axon_spike_trans = passing_fibres_strength[i]`

*Antidromic* (`dbs.py:1534-1659`):
- `efferents` → `stim_pop.antidromic = 1`, `antidromic_prob = 1`
- `afferents` → each presynaptic population: `antidromic = 1`,
  `antidromic_prob = np.mean(stim_pop.dbs_on)`, i.e. the VTA coverage fraction
- `passing_fibres` → `antidromic_prob` = the summed branch strengths when
  `sum_branches=True`, with a hard `ValueError` if the sum exceeds 1

### `excluded_populations_list` is load-bearing

Most setters guard with `hasattr` (`_set_dbs_on:1787-1792`,
`_set_depolarization:1250-1255`, `_deactivate_axon_DBS:1394-1428`), so a
population without the mechanisms is inert there. But the **efferent and afferent
branches of `_set_orthodromic` and `_set_antidromic` are unguarded** — they set
`proj.axon_transmission`, `proj.p_axon_spike_trans`, `proj.pre.prob_axon_spike`
and `proj.pre.axon_rate_amp` directly, skipping only via
`excluded_populations_list`. Only the `passing_fibres` branch uses `hasattr`.

Two consequences with a partial (selective) implementation:

1. The cortical input populations **must** be in `excluded_populations_list`.
   `ann.projections(post=stn:putamen)` includes the `CurrentInjection`
   projections from the `TimedArray`s, and without the exclusion the afferent
   branch would set `axon_transmission = 1` on them.
2. A population that *should* carry the mechanisms but does not gets a plain
   Python attribute instead (`Population.__setattr__:331-332`) — no error, and
   the DBS effect is silently dropped. This is why `add_dbs_mechanisms` ships
   with a validation pass that raises when the stimulator would write a DBS
   parameter to something that lacks it.

## How ANNarchy executes it

- `axonal` is a separate event container from `spiked`
  (`Population/SingleThreadTemplates.py:392-408`), cleared each step.
- An axon spike is **suppressed on any timestep the neuron spikes naturally**,
  and **ignores refractoriness** (`Population/SingleThreadGenerator.py:806-816`).
- Because the DBS `pre_axon_spike` string differs from the synapse's `pre_spike`,
  ANNarchy takes a separate-loop branch reading `pop.axonal` directly
  (`Projection/SingleThreadGenerator.py:983-1013`, commented "quite hacky").
  **Axon spikes therefore bypass the synaptic delay line entirely** — regular
  spikes read `_delayed_spike[delay-1]`, axon spikes read the current step.
- Learning rules are disabled on such synapses by design (`core/Synapse.py:24`).

## This project's configuration

`BOLD_optimization/get_loss.py`:

| setting | value | source |
|---|---|---|
| stimulated population | `stn:putamen` | — |
| `population_proportion` | `(35+23)/(70+75)` = 0.4 | VTA, Berlin subject 1 |
| `dbs_pulse_frequency_Hz` | 125 | Berlin subject 1 |
| `dbs_pulse_width_us` | 100 | data says 60 µs; raised to a multiple of dt |
| passing fibre | `snr__thal:putamen` | actually GPi→thal, after Miocinovic et al. 2006 |
| orthodromic / antidromic / efferents / afferents / passing_fibres | all `True` | — |

Three parameters are fitted, always the **last three** of the CMA-ES vector:

| parameter | bounds | meaning |
|---|---|---|
| `dbs_depolarization` | [0, 10] | somatic term; hyperpolarizing, see above |
| `passing_fibres_strength` | [0, 1] | how strongly DBS drives `snr__thal:putamen` |
| `axon_spikes_per_pulse` | [0, 1] | mean axon spikes per pulse (max 1) |

All three start at 0, so generation 0 of a DBS-on fit reproduces the DBS-off fit
apart from the stimulation itself.

### The DBS footprint in this model

Mechanisms are retrofitted onto **six populations of the putamen loop** — `stn`
(stimulated), its efferent targets `snr`, `gpe_proto`, `gpe_arky`, `gpe_cp`, and
`thal` as the passing-fibre target — and the **six projections DBS can reach**:
`stn__snr`, `stn__gpe_proto`, `stn__gpe_arky`, `stn__gpe_cp`, `gpe_proto__stn`,
`snr__thal`, all `:putamen`.

Everything else is excluded:

- **The whole caudate loop.** It is excluded from every DBS effect by design and
  shares no projection with putamen, so terms there would be permanently dead.
  This is the free control: caudate's on-vs-off BOLD change must be explained
  entirely by its cortical drive.
- **The striatal microcircuit** (1000 neurons per loop). Neither afferent nor
  efferent to STN and carrying no passing fibre, so its terms would always
  evaluate to zero while costing two RNG draws per neuron per timestep on the
  largest population in the model.
- **All `TimedArray` / `CurrentInjection` input machinery.** The rate-coded
  rewriter raises on a `TimedArray`, and see the exclusion note above.

So the complete DBS-on delta is: 40 % of `stn:putamen` shunted toward −90 mV for
one timestep every 8 ms; STN axon spikes at probability `axon_spikes_per_pulse`
sent orthodromically down all four STN efferents at zero delay and antidromically
resetting the STN somata; `gpe_proto:putamen` emitting axon spikes that travel
orthodromically into the stimulated 40 % of STN and antidromically reset 40 % of
its own neurons; and `snr__thal:putamen` activated at `passing_fibres_strength`
in both directions.

### Both conditions compile the same network

The mechanisms are retrofitted in the **off** condition too, with every DBS
parameter left at 0. The two conditions therefore differ only in parameter
values, which is precisely the claim the inference rests on.

The cost is that off-condition numerics changed when this landed: ANNarchy's RNG
is a single global `std::vector<std::mt19937>` indexed `rng[0]` for every
population (`Template/BaseTemplate.py:138`,
`Population/SingleThreadTemplates.py:338`), so two extra `Uniform` draws anywhere
shift the stream everywhere — including in the caudate loop, which carries no DBS
terms at all. This was taken deliberately, before any real fit had been run. See
`PLAN.md` for the recorded before/after values.

## Known limitations

1. **Axon spikes bypass synaptic delays.** For `stn__snr` and `stn__gpe_*`, which
   carry multi-ms delays, the DBS-evoked volley arrives at a different time than
   the natural one. This is ANNarchy's separate-loop transmission, not something
   configured here.
2. **The hyperdirect cortical afferent to STN cannot be activated.** It is a
   `TimedArray` → `CurrentInjection`, which has no soma and is excluded from the
   DBS footprint, so `afferents=True` in practice means `gpe_proto→stn` only.
   Cortical fibre activation — a real and much-discussed DBS effect — is not
   represented in this model.
3. **The DBS constants are unvalidated single-subject values.** The 0.4 VTA
   proportion, 125 Hz, and the choice of `snr__thal` as the one passing fibre
   come from Berlin subject 1 and one 2006 paper. The pulse width is 100 µs
   against 60 µs in the data, because 60 µs is below dt.
4. **`dbs_depolarization` scales with `C` in Izhikevich-2007 models.** Those have
   `C * dv/dt = ...`, so an appended term is implicitly divided by `C`, unlike the
   Izhikevich-2003 models where `dv/dt = ...`. It does not bite here — every
   population in the footprint is `Izhikevich2003NoisyBaseNonlin` — but it would
   the moment a striatal population entered the footprint.
5. **The firing-rate gate bands are condition-independent.** `get_firing_rate_loss`
   has no DBS switch, and the rate probe runs with DBS active. See `TODO.md` §10.
