### parameters for test_microcircuit
parameters_test_microcircuit = {}
### general
parameters_test_microcircuit["timestep"] = 0.1
parameters_test_microcircuit["seed"] = 42
### simulation time
parameters_test_microcircuit["t.duration"] = None  # whole data time
parameters_test_microcircuit["t.rampup"] = 2310  # I match this to TR of BOLD
# short run used only to score the firing rates; must be a multiple of
# update_time because v07 hands its inputs to ANNarchy one chunk at a time
parameters_test_microcircuit["t.firing_rate_sim"] = 9900
# The rate probe gates the BOLD run: above this firing-rate loss the BOLD run is
# skipped and charged 1.0. Both losses are in [0, 1]; the rate loss is
# 1 - mean(logistic goodness) over 18 populations, which is ~0.02 when every
# population sits at its band centre and 0.5 when every population sits exactly
# on a band edge. 0.5 therefore means "roughly band-edge plausible or better".
# NOT yet calibrated against real v07 rate losses -- see TODO.md.
parameters_test_microcircuit["firing_rate_gate"] = 0.5

### v07 model (Microcircuit + CorticalInputs)
parameters_test_microcircuit["mc.nx"] = 10
parameters_test_microcircuit["mc.b"] = 10
# v07 hands its cached inputs to ANNarchy one update_time chunk at a time, so
# every simulated stretch has to be a whole number of chunks. The stretches are
# the ramp-up (2310 ms = 1 TR), the rest of the run ((n_trs - 1) * 2310 ms) and
# the firing-rate probe. 110 ms divides 2310 (21 chunks per TR) and 9900;
# the previous 100 ms divided none of them.
parameters_test_microcircuit["update_time"] = 110.0
# Rates assumed for the striatal neurons that surround the simulated lattice but
# are not themselves simulated; the missing-GABA spike counts are drawn at these,
# so they are baked into the input caches. dSPN/iSPN are the parkinsonian Off
# state (levodopa withdrawn) of Liang et al. 2008; FS is the normal-primate level
# times a chronic dopamine-depletion factor of 1.0, since no parkinsonian-primate
# FSI recording exists -- see ../experimental_data/activity_striatum/README.md for
# both derivations, and keep these in sync with the plausible bands in
# get_loss.get_firing_rate_loss.
parameters_test_microcircuit["mc.firing_rate_dict"] = {
    "FS": 10.5,
    "dSPN": 25.0,
    "iSPN": 33.0,
}
# Fraction of cortical afferents two SPNs have in common. Kincaid et al. 1998
# (J Neurosci 18:4722): one corticostriatal axon contacts <=1.4 % of the cells in
# its arborization, and the shared fraction between two SPNs equals that same
# figure. It fixes the size of the cortical axon pool, M = N_eff / f, and every
# other cortical shared fraction -- FS to FS, FS to SPN -- follows from that pool
# rather than being a free parameter. See
# ../experimental_data/input_streams/README.md section 2.
parameters_test_microcircuit["mc.shared_fraction"] = 0.014
# Pairwise spike-count correlation among the unsimulated striatal neurons feeding
# the missing-GABA streams (Adler et al. 2013), and among the cortical neurons
# feeding the cortical streams (Cohen & Kohn 2011 Table 1).
#
# BOTH ARE 0 PENDING THE SCAN described in input_streams/README.md section 5.
# These values set the simulated BOLD amplitude directly -- BOLD is driven by the
# summed synaptic input current, whose variance goes as 1 + (N-1)*r, so at
# N = 486 receivers a correlation of 0.99 rather than 0 is a 481x change. The
# model also cannot decorrelate (98 % of its GABA input is an open-loop stream),
# so a literature value injected here would not be cancelled the way a real
# striatum cancels it. The intended procedure is to scan input correlation
# against simulated SPN output correlation and BOLD amplitude and choose from
# that, with Adler's 0.004 as a validation target on the OUTPUT.
parameters_test_microcircuit["mc.correlation_dict"] = {
    "FS": 0.0,
    "dSPN": 0.0,
    "iSPN": 0.0,
}
parameters_test_microcircuit["mc.cortical_correlation"] = 0.0
# The window a non-zero correlation above was measured at, and the timescale it
# is realised with. A spike-count correlation is not defined without its window:
# it grows with the window and saturates once the window exceeds the timescale.
# Everything in Cohen & Kohn Table 1 is measured at 66-3000 ms; nothing at dt.
# Required as soon as any correlation above is non-zero.
parameters_test_microcircuit["mc.correlation_window_ms"] = None
parameters_test_microcircuit["mc.correlation_timescale_ms"] = 0.0
# How many real neurons one virtual source stands for when the missing-GABA pools
# are realised geometrically. Larger is cheaper and coarsens the granularity of
# the pool overlap without biasing it.
parameters_test_microcircuit["mc.source_multiplicity"] = 10
# Fraction of cortical afferents two neurons of the same BG population have in
# common. Zero is certainly wrong physically -- neighbouring STN neurons do share
# corticosubthalamic axons -- but there is no measurement of that overlap to put
# here. Kept as a parameter rather than a hardcoded constant so the assumption is
# visible; see input_streams/README.md section 4.6.
parameters_test_microcircuit["ci.shared_fraction_dict"] = {
    "thal": 0.0,
    "gpe_arky": 0.0,
    "gpe_cp": 0.0,
    "stn": 0.0,
}
# Share of a striatal neuron's cortical afferents coming from each of the seven
# cortical ROIs the Berlin data provides, per loop. This is the ONLY physical
# difference between the caudate and the putamen loop, and it is used in two
# places that must agree: Microcircuit/CorticalInputs turn it into
# N_eff = round(p * N_cortical_inputs) per region (a region at 0 gets no stream
# at all), and cortical_drive_by_bold.py uses it to mix the seven per-region rate
# series into caudate_rate/putamen_rate. Both read it from here -- do not copy
# the numbers anywhere else, and regenerate the rate .npz after changing them.
#
# Anchored on quantitative macaque retrograde tracing, which is the only kind of
# source reporting corticostriatal input as a fraction rather than a topography:
# Borra et al. 2022 (J Neurosci 42:7060) for the caudate/putamen region groups
# and Borra et al. 2021 (J Neurosci 41:1455) for the per-area split of the motor
# putamen, with Takada et al. 1998, Inase et al. 1999 and Calzavara et al. 2007
# for the bins those tables leave grouped. Nudged toward the two ratios that
# survive the coarse Desikan-Killiany parcellation of the one human study with
# per-pathway percentages (Cacciola et al. 2017, Front Neuroanat 11:85): more
# prefrontal in the caudate, more S1 in the putamen. Full derivation, the
# sensitivity measurements and the uncertainty on putamen PMv (the least certain
# entry, plausible range 0.10-0.24) are in TODO.md section 21.
#
# Each column must sum to 1: the seven rate series are each normalised to mean
# 5 Hz, so a mix summing to 1 leaves the mean drive unchanged and moves only the
# variance and timing.
parameters_test_microcircuit["cortical_proportions_dict"] = {
    "caudate": {
        "dlPFC": 0.55,
        "preSMA": 0.15,
        "PMd": 0.18,
        "PMv": 0.04,
        "SMA": 0.06,
        "M1": 0.02,
        "S1": 0.00,
    },
    "putamen": {
        "dlPFC": 0.10,
        "preSMA": 0.05,
        "PMd": 0.11,
        "PMv": 0.18,
        "SMA": 0.15,
        "M1": 0.28,
        "S1": 0.13,
    },
}
# where the precomputed input spike counts live; on the workstations point this
# at /scratch/olmai/... , the caches are ~138 GiB per DBS condition
parameters_test_microcircuit["mc_ci_cache_dir"] = "../mc_ci_cache"
# number of cortical input neurons per receiver TODO use lit motivated values
parameters_test_microcircuit["ci.n_thal"] = 1000
parameters_test_microcircuit["ci.n_gpe_arky"] = 500
parameters_test_microcircuit["ci.n_gpe_cp"] = 500
parameters_test_microcircuit["ci.n_stn"] = 500
# paths
parameters_test_microcircuit["data_folder"] = "data_BOLD_optimization"
parameters_test_microcircuit["mc.fitted_params_path"] = (
    "../striatal_microcircuit_requirements/connectivity_parameters/connectivity_fit_data/fitted_params.json"
)
parameters_test_microcircuit["mc.cortical_rate_path"] = {
    "on": "../striatal_microcircuit_requirements/cortical_firing_rates/cortical_firing_rates_data/firing_rates_matlab_condition-on.npz",
    "off": "../striatal_microcircuit_requirements/cortical_firing_rates/cortical_firing_rates_data/firing_rates_matlab_condition-off.npz",
}
# deap cma parameters
parameters_test_microcircuit["deap_cma.run.max_evals"] = (
    2000  # should be approx 14 days
)
parameters_test_microcircuit["deap_cma.lambda"] = 12
