"""Build the precomputed input caches the v07 model streams during simulation.

v07 does not generate its cortical drive or its missing-GABA compensation on the
fly -- both are sampled once into memmaps under

    <cache-dir>/mc_<loop>_cache_<dbs>/inputs/receiver_counts_<pre>_<post>.dat
    <cache-dir>/ci_<loop>_cache_<dbs>/inputs/receiver_counts_<pre>_<post>.dat

and replayed through TimedArrays. A cache is only accepted by Microcircuit and
CorticalInputs when its stored n_steps equals int(duration / dt) exactly, so it
has to be built for the duration it will be simulated at. That duration must
cover the longest single run, which is the BOLD run (n_trs * TR) rather than the
firing-rate probe -- check both, because a short smoke test can invert the two.

The model creation kwargs come from get_loss.v07_model_creation_kwargs, so the
caches are built with exactly the settings the evaluation will later demand.
Note that the cortical rate path is stored in the cache as the *string* it was
passed, and reloading compares it verbatim: run this script from the same
working directory as get_loss.py (BOLD_optimization/).

Examples
--------
    # short cache for the end-to-end smoke test (5 TRs)
    python build_input_caches.py --dbs off --n-trs 5

    # full DBS-off cache, on a workstation, to an absolute scratch path
    python build_input_caches.py --dbs off --cache-dir /scratch/olmai/mc_ci_cache
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

from ANNarchy import setup
from CompNeuroPy.full_models import BGM

### local
from get_loss import LOOPS, TR_S, infer_max_sim_time_ms, v07_model_creation_kwargs
from parameters import parameters_test_microcircuit as paramsS


def build_caches(
    dbs_condition: str,
    duration_ms: float,
    cache_dir: str,
    loops=LOOPS,
    overwrite: bool = False,
):
    """Create one BGM_v07 per loop with building enabled, which writes the caches."""
    for loop in loops:
        kwargs = v07_model_creation_kwargs(
            loop=loop,
            dbs_condition=dbs_condition,
            duration_ms=duration_ms,
            cache_dir=cache_dir,
            build_caches=True,
        )

        for key in ("mc.storage_dir", "ci.storage_dir"):
            storage_dir = Path(kwargs[key])
            if storage_dir.exists():
                if not overwrite:
                    raise FileExistsError(
                        f"{storage_dir} already exists. Building into it would leave "
                        f"the memmaps of the old duration next to the new state file. "
                        f"Pass --overwrite to replace it."
                    )
                print(f"removing existing cache {storage_dir}")
                shutil.rmtree(storage_dir)

        print(f"\n=== building inputs for the {loop} loop, DBS {dbs_condition} ===")
        start = time.time()
        # do_create builds the model, which runs Microcircuit and CorticalInputs
        # with build_connectivity/build_*_input enabled; no compile is needed
        # because nothing is simulated here.
        BGM(
            name="BGM_v07_p01",
            model_creation_kwargs=kwargs,
            seed=paramsS["seed"],
            name_appendix=loop,
            do_create=True,
            do_compile=False,
        )
        print(f"=== {loop} done in {(time.time() - start) / 60:.1f} min ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dbs",
        type=str,
        required=True,
        choices=["on", "off"],
        help="DBS condition to build the caches for. Required for the same reason "
        "as in get_loss.py: a silent default builds the wrong condition.",
    )
    parser.add_argument(
        "--n-trs",
        type=int,
        default=None,
        help="Build for the first N TRs of the cortical drive instead of all 310. "
        "For smoke tests; the resulting cache is only usable with the same --n-trs.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Where the mc_*/ci_* cache folders go. Defaults to mc_ci_cache_dir "
        "from parameters.py. Prefer an absolute path.",
    )
    parser.add_argument(
        "--loops",
        type=str,
        nargs="+",
        default=list(LOOPS),
        choices=list(LOOPS),
        help="Which BG loops to build. Both by default.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing cache folders before building.",
    )
    args = parser.parse_args()

    setup(dt=paramsS["timestep"], seed=paramsS["seed"])

    duration_ms, mixed_rates = infer_max_sim_time_ms(
        dbs_condition=args.dbs,
        cortical_rate_path=paramsS["mc.cortical_rate_path"],
        dt_ms=paramsS["timestep"],
        n_trs=args.n_trs,
    )

    # The cache has to cover every run the model will do, and the firing-rate
    # probe is fixed-length: at full length it is far shorter than the BOLD run,
    # but at a few TRs it is the longer of the two.
    firing_rate_sim_ms = paramsS["t.firing_rate_sim"]
    if firing_rate_sim_ms > duration_ms:
        print(
            f"warning: the firing-rate probe is {firing_rate_sim_ms} ms but the "
            f"cache would cover only {duration_ms} ms "
            f"({len(mixed_rates['caudate']['rate'])} TRs). The probe would run off "
            f"the end of the cache. Use at least "
            f"--n-trs {int(-(-firing_rate_sim_ms // (TR_S * 1000.0)))}.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Every simulated stretch is handed over in update_time chunks, so all three
    # have to be whole numbers of chunks (see parameters.py).
    update_time = paramsS["update_time"]
    for label, stretch_ms in (
        ("ramp-up", paramsS["t.rampup"]),
        ("post-ramp-up", duration_ms - paramsS["t.rampup"]),
        ("firing-rate probe", firing_rate_sim_ms),
    ):
        if abs(round(stretch_ms / update_time) - stretch_ms / update_time) > 1e-9:
            print(
                f"warning: the {label} stretch ({stretch_ms} ms) is not a multiple "
                f"of update_time ({update_time} ms); simulate_model would refuse it.",
                file=sys.stderr,
            )
            sys.exit(1)

    cache_dir = args.cache_dir or paramsS["mc_ci_cache_dir"]
    n_steps = int(duration_ms / paramsS["timestep"])
    print(
        f"building input caches: DBS {args.dbs}, "
        f"{len(mixed_rates['caudate']['rate'])} TRs = {duration_ms / 1000.0:.1f} s "
        f"= {n_steps} steps at dt = {paramsS['timestep']} ms\n"
        f"loops: {', '.join(args.loops)}\n"
        f"cache dir: {Path(cache_dir).resolve()}"
    )

    build_caches(
        dbs_condition=args.dbs,
        duration_ms=duration_ms,
        cache_dir=cache_dir,
        loops=args.loops,
        overwrite=args.overwrite,
    )
