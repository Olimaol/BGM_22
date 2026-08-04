"""Regression test for the DBS-on path.

Two defects made DBS-on unusable and neither announced itself:

1. `DBSstimulator(auto_implement=True)` cleared and recreated the network, which
   cannot survive v07's TimedArray/CurrentInjection inputs.
2. `on()` ran after `compile()`, so the first `reset()` restored `dbs_on = 0` and
   the evaluation ran without DBS while still recording `"dbs": "on"`.

Defect 2 is the dangerous one: nothing crashed, nothing looked wrong, and the
loss file was indistinguishable from a real DBS run. So the checks below are
mostly about proving the stimulation is *present and effective*, not just that
the script exits 0.

Usage:

    python test_dbs_on.py                # mechanism checks only, ~1 min
    python test_dbs_on.py --integration  # plus three v08 evaluations, ~10 min

Run the integration part from BOLD_optimization/ or leave the default; it drives
get_loss.py as a subprocess the same way deap_cma_opt.py does.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parent
OPT_DIR = REPO_DIR / "BOLD_optimization"

### v08 base vector: 7 drive weights, 2 base currents, 12 cluster scalings
V08_BASE = ["0.002"] * 7 + ["100", "100"] + ["1"] * 12

failures: list[str] = []


def check(condition: bool, description: str):
    """Record a check instead of aborting, so one run reports every failure."""
    print(f"  {'PASS' if condition else 'FAIL'}  {description}")
    if not condition:
        failures.append(description)


def test_mechanisms():
    """add_dbs_mechanisms, on() before compile, and survival of a reset."""
    from CompNeuroPy import ann, DBSstimulator, add_dbs_mechanisms

    print("\n[1] DBS mechanisms on a synthetic network")
    ann.setup(dt=0.1, seed=42)

    stim = ann.Population(20, ann.Izhikevich, name="stim")
    post = ann.Population(20, ann.Izhikevich, name="post")
    pre = ann.Population(20, ann.Izhikevich, name="pre")
    outside = ann.Population(20, ann.Izhikevich, name="outside")

    proj_eff = ann.Projection(pre=stim, post=post, target="exc", name="stim__post")
    proj_eff.connect_all_to_all(weights=0.1)
    proj_aff = ann.Projection(pre=pre, post=stim, target="exc", name="pre__stim")
    proj_aff.connect_all_to_all(weights=0.1)

    add_dbs_mechanisms(
        populations=[stim, post, pre], projections=[proj_eff, proj_aff]
    )

    check("dbs_on" in stim.attributes, "retrofit adds dbs_on to the stimulated pop")
    check(
        "p_axon_spike_trans" in proj_eff.attributes,
        "retrofit adds p_axon_spike_trans to a projection",
    )
    check(
        "dbs_on" not in outside.attributes,
        "a population outside the footprint is left alone",
    )
    check(
        proj_eff._connection_method is not None,
        "connectivity survives the retrofit (this is what recreation destroyed)",
    )

    stimulator = DBSstimulator(
        stimulated_population=stim,
        population_proportion=0.4,
        excluded_populations_list=[outside],
        dbs_depolarization=5.0,
        orthodromic=True,
        antidromic=True,
        efferents=True,
        afferents=True,
        passing_fibres=False,
        dbs_pulse_frequency_Hz=125,
        dbs_pulse_width_us=100,
        axon_spikes_per_pulse=0.5,
        seed=42,
        auto_implement=False,
    )
    ### before compile on purpose: this is what makes the state survive reset()
    stimulator.on()

    ann.compile(directory="annarchy_test_dbs_on")

    expected = stimulator.dbs_on_array.flatten()
    check(
        np.array_equal(np.asarray(stim.dbs_on).flatten(), expected),
        "after compile, dbs_on matches the stimulator's array",
    )
    check(
        np.isclose(stim.dbs_depolarization, 5.0),
        "after compile, dbs_depolarization is set on the stimulated pop",
    )
    check(bool(proj_eff.axon_transmission), "after compile, axon_transmission is on")

    ann.simulate(50.0)
    ann.reset()

    ### the defect-2 check
    check(
        np.array_equal(np.asarray(stim.dbs_on).flatten(), expected),
        "dbs_on SURVIVES reset() (this is the bug that made DBS-on silently inert)",
    )
    check(
        np.isclose(stim.dbs_depolarization, 5.0),
        "dbs_depolarization survives reset()",
    )
    check(bool(proj_eff.axon_transmission), "axon_transmission survives reset()")


def test_validation_rejects_partial_implementation():
    """on() must refuse to write DBS to a population that cannot hold it.

    ANNarchy turns `pop.dbs_on = 1` on a population without that parameter into a
    plain Python attribute, so the effect is dropped in silence. With a selective
    footprint that would turn one missing name into a run that looks stimulated
    and is not.
    """
    print("\n[2] validation pass rejects an incomplete footprint")
    ### a fresh interpreter, because the check above already compiled a network
    code = (
        "from CompNeuroPy import ann, DBSstimulator, add_dbs_mechanisms\n"
        "ann.setup(dt=0.1, seed=42)\n"
        "a = ann.Population(10, ann.Izhikevich, name='a')\n"
        "b = ann.Population(10, ann.Izhikevich, name='b')\n"
        "p = ann.Projection(pre=a, post=b, target='exc', name='a__b')\n"
        "p.connect_all_to_all(weights=0.1)\n"
        "add_dbs_mechanisms(populations=[a], projections=[p])  # b deliberately missing\n"
        "s = DBSstimulator(stimulated_population=a, population_proportion=0.4,\n"
        "    excluded_populations_list=[], dbs_depolarization=5.0, orthodromic=True,\n"
        "    antidromic=True, efferents=True, afferents=False, passing_fibres=False,\n"
        "    axon_spikes_per_pulse=0.5, seed=42, auto_implement=False)\n"
        "s.on()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_DIR
    )
    check(result.returncode != 0, "on() fails when a reachable population lacks DBS")
    check(
        "do not carry the DBS mechanisms" in result.stderr,
        "the error names the actual problem",
    )


def run_evaluation(dbs_condition: str, dbs_params: list[str], appendix: str):
    """Run one get_loss.py evaluation and return its loss file contents."""
    cmd = [
        sys.executable,
        "get_loss.py",
        "--dbs",
        dbs_condition,
        "--model-version",
        "v08",
        "--n-trs",
        "5",
        "--gate-threshold",
        "1.0",
        "--compile-appendix",
        appendix,
        *V08_BASE,
        *dbs_params,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=OPT_DIR)
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        raise RuntimeError(f"get_loss.py failed for dbs={dbs_condition}")
    loss_file = OPT_DIR / "data_BOLD_optimization" / f"loss_{appendix}.json"
    return json.loads(loss_file.read_text())


def test_integration():
    """The stimulation must move the putamen loop and leave caudate untouched."""
    print("\n[3] v08 end-to-end: DBS changes what it should and nothing else")

    ### same cortical drive in both, so the only difference is the stimulation
    inert = run_evaluation("on", ["0.0", "0.0", "0.0"], "dbstest_on_inert")
    active = run_evaluation("on", ["3.0", "0.5", "0.5"], "dbstest_on_active")

    check(inert["dbs"] == "on" and active["dbs"] == "on", "both runs record dbs=on")

    stn_inert = inert["firing_rates_hz"]["stn:putamen"]
    stn_active = active["firing_rates_hz"]["stn:putamen"]
    check(
        stn_active < stn_inert - 1.0,
        f"DBS lowers stn:putamen ({stn_inert:.2f} -> {stn_active:.2f} Hz); "
        "the somatic term pulls toward -90 mV despite being called a depolarization",
    )

    for pop_name, rate in inert["firing_rates_hz"].items():
        if not pop_name.endswith(":caudate"):
            continue
        check(
            np.isclose(rate, active["firing_rates_hz"][pop_name], atol=1e-6),
            f"free control: {pop_name} is unchanged by the DBS parameters",
        )

    moved = [
        name
        for name, rate in inert["firing_rates_hz"].items()
        if name.endswith(":putamen")
        and not np.isclose(rate, active["firing_rates_hz"][name], atol=1e-6)
    ]
    check(len(moved) > 0, f"DBS moves the putamen loop ({len(moved)} populations)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--integration",
        action="store_true",
        help="also run three v08 evaluations (~10 min) instead of the mechanism "
        "checks alone",
    )
    args = parser.parse_args()

    test_mechanisms()
    test_validation_rejects_partial_implementation()
    if args.integration:
        test_integration()
    else:
        print("\n[3] skipped, pass --integration to run the v08 evaluations")

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for description in failures:
            print(f"  - {description}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
