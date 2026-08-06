# How the v08 model is created

v08 is the reduced fallback model: plain 100-neuron populations, random
connectivity, and the entire cortical drive collapsed into one `TimedArray` per
loop. It was written under time pressure in Dec 2025 and is kept **only** as a
fast, cache-free end-to-end smoke test of the optimization pipeline — it needs no
input caches and runs in minutes.

This document is a delta against `model_v07.md`, which is the reference: the
creation sequence, the `BGM` / `parameters.csv` machinery, the shared neuron and
projection tables, and the prerequisites are described there and are not
repeated. What follows is what v08 does differently. Read alongside `CLAUDE.md`,
`PLAN.md`, `TODO.md` and `DBS.md`.

---

## 1. v07 vs v08 at a glance

| | v07 | v08 |
|---|---|---|
| striatum | `Microcircuit`: 3D lattice, 486 dSPN / 485 iSPN / 29 FS, distance-dependent connectivity fitted from data | three plain 100-neuron populations, `connect_fixed_number_pre` |
| striatal population names | `caudate_dSPN`, `putamen_FS`, … | `str_d1:caudate`, `str_fsi:putamen`, … |
| intra-striatal connections | inside the microcircuit, 7 sparse projections with individually sampled weights | 7 ordinary projections in the CSV |
| cortical drive | precomputed spike counts streamed from disk through many `TimedArray`s | one `TimedArray` per loop, per-TR, scaled by `exp_input_weight` |
| drive mechanism | `CurrentInjection` into `glut` / `ampa` (and `gaba` for the compensation streams) | a random `Exponential(lambda)` term inside the neuron equations |
| missing-GABA compensation | yes | none |
| projections per loop | 28 | 35 |
| optimized parameters | 19 (9 drive + 10 clusters) | 21 (9 drive + 12 clusters) |
| input caches | required, ~138 GiB per DBS condition | none |
| simulation granularity | whole `update_time` chunks (110 ms) | unconstrained, plain `simulate()` |

Everything else is shared: the same six `Izhikevich2003NoisyBaseNonlin` BG
populations, the same 28 core projections with the same weights and delays, the
same `parameters.csv` rows outside the striatal block, the same DBS retrofit, the
same seven BOLD monitors, the same single compile for both loops.

One detail of the BOLD monitors is v08-only: Cau and Put are pooled with explicit
`props_delRey` scale factors here, because v08's three striatal populations all
have 100 neurons and `BoldMonitor`'s size-proportional default would weight them
equally. v07 passes none and lets the default reproduce the proportions its
microcircuit was built with — `model_v07.md` §3.6.

---

## 2. Creation sequence

The `get_loss.py` main block, same steps in the same order as `model_v07.md` §3.
Steps 3, 4, 5 and 7 are where v08 departs.

1. `setup(dt, seed)` — identical.
2. Duration inferred from the rate `.npz` — identical, but v08 also **keeps the
   mixed striatal rate series**: `mixed_rates[loop]["rate"]` becomes the model's
   entire cortical drive.
3. **Model creation kwargs are three keys instead of nineteen:**

   | key | value |
   |---|---|
   | `input.rates` | `mixed_rates[loop]["rate"]` — the anatomically mixed `caudate_rate` / `putamen_rate` series, 310 values |
   | `input.schedule` | `TR_S * 1000.0` = 2310 ms |
   | `timestep` | 0.1 |

   There is no cache directory, no `fitted_params.json`, no per-region
   proportions. The seven cortical `*_rate` series that v07 uses are not read at
   all; v08 uses only the two striatal ones.
4. `BGM(name="BGM_v08_p01", …, do_create=True, do_compile=False)` per loop —
   identical call shape. `BGM_v08` returns `None`, so `self.mc` and `self.ci`
   stay `None` and `_components_created_by_mc_ci` is empty: **every** population
   and projection goes through the name-appendix pass and the CSV parameter
   passes.
5. **`add_TimedInputs()` per loop** — §5. Unique to v08, and it runs here, before
   the DBS block, so that every population exists before the DBS footprint is
   resolved.
6. DBS retrofit, `DBSstimulator(auto_implement=False)`, `on()` pre-compile —
   identical.
7. Seven `BoldMonitor`s — identical, except that Cau and Put pool
   `str_d1:<loop>`, `str_d2:<loop>`, `str_fsi:<loop>` instead of the
   `<loop>_dSPN` family. They still map `I_CBF` to `I_v` rather than `I`: the
   choice is made per ROI, not per model version, and v08's striatal populations
   are Humphries models too, so `I_v` is the variable they expose.
8. `model_dict["caudate"].compile()` — identical.

---

## 3. Populations

Nine per loop, all size 100, all from `parameters.csv` column `BGM_v08_p01` —
including `str_d1.size`, `str_d2.size` and `str_fsi.size`, the three rows the
v07 column leaves blank.

| population | neuron model | constructor arguments |
|---|---|---|
| `str_d1:<loop>` | `Izhikevich2007Humphries2009SPND1` | `current_based_excitation=True, exp_input=1/0.7, params_for_pop=True` |
| `str_d2:<loop>` | `Izhikevich2007Humphries2009SPND2` | `current_based_excitation=True, exp_input=1/0.7, params_for_pop=True` |
| `str_fsi:<loop>` | `Izhikevich2007Humphries2009FSI` | `current_based_excitation=True, exp_input=1/0.28, params_for_pop=True` |
| `stn:<loop>` | `Izhikevich2003NoisyBaseNonlin` | `stabilize=True, use_nonlin=False, exp_input=1/0.075` |
| `snr:<loop>` | `Izhikevich2003NoisyBaseNonlin` | `stabilize=True, use_nonlin=False` |
| `gpe_proto:<loop>` | `Izhikevich2003NoisyBaseNonlin` | `stabilize=True, use_nonlin=True` |
| `gpe_arky:<loop>` | `Izhikevich2003NoisyBaseNonlin` | `stabilize=True, use_nonlin=True, exp_input=1/0.03` |
| `gpe_cp:<loop>` | `Izhikevich2003NoisyBaseNonlin` | `stabilize=True, use_nonlin=True, exp_input=1/0.03` |
| `thal:<loop>` | `Izhikevich2003NoisyBaseNonlin` | `stabilize=True, use_nonlin=False, exp_input=1/0.12` |

The neuron classes are the same four as v07 and all their parameter values are
unchanged — see `model_v07.md` §6 for the equations and tables. The v07 CSV rows
for the six BG populations are byte-identical to the v08 ones, so the parameter
tables there apply here verbatim.

The two constructor arguments that are *not* v07 defaults are what §4 is about.

---

## 4. The `exp_input` drive

This is the one place where v08's **equations** differ from v07's, not just its
values.

`exp_input` is a float defaulting to `0.0`. When it is greater than zero, the
neuron model inserts an extra line and an extra term into the conductance ODEs:

```
exp_input   = Exponential(lambda) * exp_input_weight * g_cor
dg_ampa/dt  = -g_ampa/tau_ampa + g_glut/dt + exp_input/dt
dg_nmda/dt  = -g_nmda/tau_nmda + g_glut/dt + exp_input/dt
```

for the two SPN models,

```
exp_input   = Exponential(lambda) * exp_input_weight * g_cor
dg_ampa/dt  = -g_ampa/tau_ampa + exp_input/dt
```

for the FSI model (no NMDA), and

```
exp_input   = Exponential(lambda) * exp_input_weight * g_cor
dg_ampa/dt  = -g_ampa/tau_ampa + exp_input/dt
```

for `Izhikevich2003NoisyBaseNonlin`. In v07 all four are instantiated with
`exp_input = 0.0`, so the `exp_input` line is absent entirely and `dg_ampa/dt`
carries no such term.

Three things drive that term:

- **`g_cor`** is the conductance of a synaptic target called `cor`. It is fed by
  the `CurrentInjection` projections of §5, so it holds the current cortical
  drive value for this loop.
- **`lambda`** is the constructor's `exp_input` value, and it is the *rate* of
  the exponential distribution — so the mean draw is `1/lambda`. The
  constructors pass `1/x`, which makes `x` the mean directly:

  | population | `exp_input` argument | `lambda` | mean draw |
  |---|---|---|---|
  | `str_d1`, `str_d2` | `1/0.7` | 1.4285714 | 0.7 |
  | `str_fsi` | `1/0.28` | 3.5714286 | 0.28 |
  | `stn` | `1/0.075` | 13.333333 | 0.075 |
  | `gpe_arky`, `gpe_cp` | `1/0.03` | 33.333333 | 0.03 |
  | `thal` | `1/0.12` | 8.3333333 | 0.12 |
  | `snr`, `gpe_proto` | — | 0.0 | term absent |

- **`exp_input_weight`** starts at 1.0 and is the fitted parameter: seven of the
  21 optimized values are `exp_input_weight` on the seven populations above.

So the drive is a Poisson-like train of random conductance kicks whose *size* is
set per population and whose *rate* is modulated by the shared cortical series.

`snr` and `gpe_proto` get no `exp_input`, exactly as in v07 — their excitation is
a fitted `base_mean` baseline current instead.

**`params_for_pop=True`** (striatal populations only) adds the `: population`
flag to the parameters that carry it conditionally — `tau_ampa`, `tau_nmda`,
`tau_gaba`, `lambda`, `exp_input_weight` and the dopamine `phi_1` / `phi_2`. They
become single global values instead of per-neuron arrays. In v07 the striatal
populations are built by `Microcircuit` with the default
`params_for_pop=False`, so the same parameters are per-neuron there. `I_app`
is per-neuron in both; everything else in these models is unconditionally
`: population`.

**Caveat.** The `exp_input_weight` bounds used by the optimizer are orders of
magnitude too wide in v08 — see `TODO.md` §1. This is one reason v08 is a
pipeline test rather than a model to fit.

---

## 5. `add_TimedInputs`: the collapsed cortical drive

`get_loss.add_TimedInputs()`, run once per loop right after the two `BGM`
objects exist. Where v07 builds dozens of streams inside `Microcircuit` and
`CorticalInputs`, v08 builds one:

1. Take `input.rates` — the 310-value mixed striatal series for this loop.
2. `np.repeat` it across neurons into shape `(310, 100)`, so every neuron of
   every target population sees the same value.
3. Create `TimedArray(rates=…, schedule=2310.0, name=f"TimedInput_cortex:{loop}")`.
   **The `schedule` argument does the per-TR expansion**, so unlike v07 there is
   no `np.repeat` to `dt` resolution and no chunked replay: ANNarchy holds each
   row for a whole TR by itself.
4. Wire it to seven populations with `CurrentInjection(target="cor")`, named
   `TimedInput_cortex:<loop>__<pop>:<loop>`, connected one-to-one with weight 1.0
   and delay 0: `str_d1`, `str_d2`, `str_fsi`, `thal`, `gpe_arky`, `gpe_cp`,
   `stn`.

Those seven projections are what fills `g_cor` in §4. `snr` and `gpe_proto` are
not connected.

Consequences worth naming:

- **There is no chunking constraint.** `get_loss.simulate_model()` short-circuits
  to a plain `simulate(duration_ms)` for anything that is not v07, and
  `rewind_inputs()` just calls `TimedArray.reset()` on the two input populations
  rather than rebuilding memmap iterators.
- **All seven populations receive the identical drive**, differing only in their
  `lambda` and fitted `exp_input_weight`. v07's per-region proportion mixes, the
  0.014 shared fraction, the correlation structure and the derived FS input all
  have no counterpart here.
- `_set_connections()` skips these projections by name — it ignores anything
  starting with `TimedInput`.

---

## 6. The 35 projections

The 28 core projections are identical to v07 in pre, post, target, connectivity
method, `number`, weights and delays — see `model_v07.md` §5. v08 adds seven
intra-striatal projections, all `target="gaba"` and all
`connect_fixed_number_pre(number=10)`:

| pre → post | weight | delay | opt. cluster |
|---|---|---|---|
| str_d1 → str_d1 | 0.5 | 3 | `str_laterals` |
| str_d1 → str_d2 | 0.75 | 3 | `str_laterals` |
| str_d2 → str_d1 | 0.75 | 3 | `str_laterals` |
| str_d2 → str_d2 | 0.5 | 3 | `str_laterals` |
| str_fsi → str_d1 | 0.5 | 3 | `str_fsi__striatum` |
| str_fsi → str_d2 | 0.5 | 3 | `str_fsi__striatum` |
| str_fsi → str_fsi | 0.8 | 3 | `str_fsi__striatum` |

These are the connections that v07 keeps inside the microcircuit as
distance-dependent, individually sampled synapses. Here they are seven scalar
weights over random fixed-in-degree connectivity — and the two extra optimizer
clusters they introduce are the whole difference between 19 and 21 parameters.

Note the asymmetry the CSV encodes: SPN → *other* type is weighted 0.75 while
SPN → *same* type is 0.5, and `str_fsi__str_fsi` is the strongest at 0.8. There
is **no `str_d1 → str_fsi` or `str_d2 → str_fsi`**, matching v07, where
`fitted_params.json` has no SPN→FS pair either.

---

## 7. What is still not set at compile time

As in `model_v07.md` §11: the `BoldMonitor` period is set immediately after
compile, and the optimized parameters are applied at the start of every
experiment run — after every `reset()`, because a post-compile parameter write
does not survive one.

v08 takes **21** parameters (`get_loss.n_opt_params("v08")`):

| index | meaning | written to |
|---|---|---|
| 0–6 | `exp_input_weight` for str_d1, str_d2, str_fsi, thal, gpe_arky, gpe_cp, stn | the population parameter |
| 7–8 | baseline current for snr, gpe_proto | `base_mean` |
| 9–20 | one scaling factor per projection cluster (10 common + `str_fsi__striatum` + `str_laterals`) | `proj.w` = CSV weight × factor |

**Unlike v07, every one of these is a real ANNarchy parameter.** v07's drive
parameters 0–6 live in plain Python dicts inside `Microcircuit` and
`CorticalInputs`; here they are `exp_input_weight` on the populations
themselves, so they are visible in a `report()` and readable back from the
compiled model.

The upper bounds are derived from the numerical stabilization factor
`g_eff = g / (1 + g·dt/C)` documented in `get_loss.set_opt_params_v08()`: with
`dt = 0.1`, `C = 50` for the two SPN populations gives a ceiling around 500,
`C = 80` for FSI around 800, and `C = 1` for the remaining populations around 10.
The baseline currents scale by the 50 mV fixed driving force, giving ~500 for
`snr` and `gpe_proto`.

Under `--dbs on` the vector is base + 3 DBS parameters, or the staged base + one
putamen-only scaling per cluster + 3 DBS. The DBS parameters are always the
**last three**; the old fixed indices 21–23 happened to line up with v08 and with
nothing else.

---

## 8. The compiled network, in numbers

Counted from a `report()` of the result, against v07's 130 populations and 193
projections:

| | caudate | putamen | shared | total |
|---|---|---|---|---|
| model populations | 9 × 100 | 9 × 100 | — | 18 |
| `TimedArray` input populations | 1 | 1 | — | 2 |
| BOLD model populations | — | — | 7 | 7 |
| **populations** | | | | **27** |
| BGM projections | 35 | 35 | — | 70 |
| `CurrentInjection` input projections | 7 | 7 | — | 14 |
| projections into the BOLD models | — | — | 18 | 18 |
| **projections** | | | | **102** |

To regenerate:

```bash
python get_loss.py --dbs off --model-version v08 --n-trs 5 \
  --compile-appendix report --report /tmp/report_v08_off.md
```

The off and on conditions produce **identical** population, projection, monitor,
neuron-model and synapse-model tables — the two files even have the same byte
size. They differ only in the values of `dbs_on`, `dbs_depolarization`,
`antidromic`, `antidromic_prob` and `prob_axon_spike` on the six putamen DBS
populations, and `p_axon_spike_trans` on five putamen projections. This is the
same result v07 gives, in the same shape; `model_v07.md` §10 states it in full,
including what it does and does not prove.
