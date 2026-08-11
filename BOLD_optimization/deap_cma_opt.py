"""CMA-ES fit of the BGM model to one subject's resting-state BOLD.

Each individual is evaluated by running get_loss.py as a separate process, one
per parameter vector, because ANNarchy compiles a model per process and cannot
hold several of them. A generation is therefore a batch of lambda subprocesses.

Three things this script is deliberately careful about, all of them lessons from
the December 2025 run that died silently:

* **Every individual gets its own log file and its exit code is recorded.** The
  previous version used CompNeuroPy's run_script_parallel, which turns any
  non-zero child exit into a bare exit(1) with no message, and sent the
  children's output to a console that run_optimization.sh had backgrounded.
* **A dead individual is penalised, not fatal.** It gets FAILED_LOSS, which is
  above any loss a real evaluation can produce, so CMA-ES simply ranks it last.
  Only losing more than half a generation aborts the run -- at that point
  something systematic is wrong and continuing would just burn days.
* **The CMA-ES state is checkpointed every generation**, so a run killed by the
  OOM killer or a reboot resumes instead of starting over.

Examples
--------
    # DBS-off fit of the full model
    python deap_cma_opt.py --dbs off --model-version v07 --optimization-run 1

    # five-generation smoke run of the pipeline on the reduced model
    python deap_cma_opt.py --dbs off --model-version v08 --max-evals 5 --n-trs 20

    # resume whatever the checkpoint holds
    python deap_cma_opt.py --dbs off --model-version v07 --optimization-run 1 --resume
"""

import argparse
import json
import pickle
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from CompNeuroPy import DeapCma, save_variables, load_variables

### local
from get_loss import MODEL_VERSIONS, n_opt_params, proj_clusters
from parameters import parameters_test_microcircuit as paramsS

### An evaluation returns firing_rate_loss + bold_loss, both in [0, 1], so no
### real evaluation can reach 10. CMA-ES ranks rather than uses the magnitude,
### so this only has to sort last -- and it stands out in the logbook.
FAILED_LOSS = 10.0

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / paramsS["data_folder"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dbs",
        type=str,
        required=True,
        choices=["on", "off"],
        help="DBS condition passed to get_loss.py. Required: silently defaulting "
        "here means a whole optimization can run against the wrong condition.",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="v07",
        choices=list(MODEL_VERSIONS),
        help="Which BGM model to fit. v07 is the full model, v08 the reduced one "
        "used to smoke-test this pipeline.",
    )
    parser.add_argument(
        "--optimization-run",
        type=int,
        default=0,
        help="Run identifier, used in the compile appendix, the checkpoint name "
        "and the result name so parallel runs do not overwrite each other.",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=None,
        help="Number of generations. Defaults to deap_cma.run.max_evals.",
    )
    parser.add_argument(
        "--lambda",
        dest="lambda_",
        type=int,
        default=None,
        help="Individuals per generation, i.e. concurrent processes. Defaults to "
        "deap_cma.lambda. Keep it at or below the physical core count.",
    )
    parser.add_argument(
        "--n-trs",
        type=int,
        default=None,
        help="Shorten every evaluation to the first N TRs. For mini-runs only; "
        "for v07 the input cache must have been built for the same N.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Where the v07 input caches live. Defaults to mc_ci_cache_dir.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the checkpoint of this run instead of starting over.",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Assume the per-individual compile folders already exist.",
    )
    return parser.parse_args()


def load_best_off_fit(model_version: str, n_base: int):
    """Return the best base parameter vector from the finished DBS-off fits.

    The DBS-on fit does not re-search the base parameters: they describe the
    model without stimulation and are carried over unchanged.
    """
    result_folder = DATA_DIR / "deap_cma_result"
    pattern = f"deap_cma_result_{model_version}_off_run_*"
    result_files = sorted(p.stem for p in result_folder.glob(f"{pattern}.pkl"))
    if not result_files:
        raise FileNotFoundError(
            f"No DBS-off result matching {pattern} in {result_folder}. Fit the off "
            f"condition before the on condition."
        )

    loaded_dict = load_variables(
        name_list=result_files, path=str(result_folder)
    )

    best_loss = float("inf")
    best_result = None
    best_name = None
    for name, deap_cma_result in loaded_dict.items():
        if deap_cma_result["best_fitness"] < best_loss:
            best_loss = deap_cma_result["best_fitness"]
            best_result = deap_cma_result
            best_name = name
    print(f"carrying over the base parameters of {best_name} (loss {best_loss:.4f})")
    return [best_result[f"param{i}"] for i in range(n_base)], best_name


def search_space(model_version: str, dbs_condition: str):
    """Lower bounds, upper bounds, p0 and the fixed base vector for a condition.

    DBS off searches the base parameters. DBS on keeps them at their off-fit
    values and searches only the putamen weight scalings plus the 3 DBS
    parameters -- caudate is excluded from every DBS effect and shares no
    projection with putamen, so refitting caudate would assert a mechanism the
    model does not contain (and would leave the free control that caudate's
    on-vs-off BOLD change must come entirely from its cortical drive).
    """
    n_base = n_opt_params(model_version)
    n_clusters = len(proj_clusters(model_version))

    if model_version == "v08":
        # The v08 drive enters as exp_input_weight; the upper bounds come from
        # where the conductance stabilisation g/(1 + g*dt/C) saturates.
        # NOTE these are far too wide -- the rate saturates by ~0.2 and the loss
        # is flat across the whole range. See TODO.md section 1.
        base_lower = np.zeros(n_base)
        base_upper = np.array([500, 500, 800, 10, 10, 10, 10, 500, 500] + [5] * n_clusters)
        base_p0 = np.array([0] * 9 + [1] * n_clusters)
    else:
        # v07 drives the striatum through Microcircuit.mean_weights_by_type and
        # the rest through CorticalInputs.mean_weights_by_type, both of which
        # default to 0.001. p0 is that default, i.e. the model as its author
        # configured it, and the bound is an order of magnitude above it.
        # PROVISIONAL: nobody has measured the useful range for v07 either.
        # TODO.md section 1 must settle this before a real fit.
        base_lower = np.zeros(n_base)
        base_upper = np.array([0.01] * 7 + [500, 500] + [5] * n_clusters)
        base_p0 = np.array([0.001] * 7 + [0, 0] + [1] * n_clusters)

    if dbs_condition == "off":
        return base_lower, base_upper, base_p0, None, None

    fixed_base, off_fit_source = load_best_off_fit(model_version, n_base)
    # free: one scaling per weight cluster for putamen, then the 3 DBS parameters
    lower = np.array([0.0] * n_clusters + [0.0, 0.0, 0.0])
    upper = np.array(
        base_upper[n_base - n_clusters :].tolist()
        # dbs_depolarization like a conductance in stn; the other two are
        # fractions: passing-fibre activation and spikes per pulse
        + [10.0, 1.0, 1.0]
    )
    # Start from the off-fit weights with the three DBS parameters at zero, so
    # generation 0 is the off fit's network with the stimulation switched off.
    # It does NOT reproduce the off fit's output: the on condition also swaps the
    # cortical drive for the DBS-on recording, and that alone moves every
    # population in both loops. Measured on v08 at 5 TRs, DBS (0,0,0) vs the off
    # run: str_d1 -28 Hz, snr +3 Hz, and the same shifts in caudate, which carries
    # no DBS mechanisms at all. That drive change is the intended free control
    # (see DBS.md), not an artefact - but p0 is a neutral starting point, not a
    # reproduction of the off result.
    p0 = np.array(list(fixed_base[n_base - n_clusters :]) + [0.0, 0.0, 0.0])
    return lower, upper, p0, fixed_base, off_fit_source


def get_loss_args(
    individual, index, args, fixed_base, n_base
):
    """Command line for one individual's get_loss.py process."""
    if fixed_base is None:
        params = list(individual)
    else:
        # base (fixed) + putamen weight scalings + 3 DBS parameters
        params = list(fixed_base) + list(individual)

    cmd = [
        "--dbs",
        args.dbs,
        "--model-version",
        args.model_version,
        "--compile-appendix",
        # the DBS condition belongs in the name: an off and an on run with the same
        # --optimization-run would otherwise share compile folders and loss files
        f"{args.model_version}_{args.dbs}_run_{args.optimization_run}_ind{index}",
    ]
    if args.n_trs is not None:
        cmd += ["--n-trs", str(args.n_trs)]
    if args.cache_dir is not None:
        cmd += ["--cache-dir", args.cache_dir]
    return cmd + [str(p) for p in params]


def run_individuals(args_list, log_paths, n_jobs):
    """Run get_loss.py once per argument list, returning the exit codes.

    Unlike CompNeuroPy's run_script_parallel this neither kills its siblings on
    the first failure nor swallows the output: each child writes to its own log
    and its exit code comes back to the caller. sys.executable is used so the
    children run in this interpreter rather than whatever "python" resolves to.
    """

    def run_one(job):
        job_args, log_path = job
        with open(log_path, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                [sys.executable, str(SCRIPT_DIR / "get_loss.py")] + job_args,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(SCRIPT_DIR),
            )
            return process.wait()

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        return list(executor.map(run_one, zip(args_list, log_paths)))


def read_loss(loss_file: Path):
    """Total loss from an individual's loss file, or None if it is unusable."""
    if not loss_file.exists():
        return None
    try:
        with open(loss_file, "r", encoding="ascii") as f:
            payload = json.load(f)
        return float(payload["total_loss"])
    except (json.JSONDecodeError, KeyError, ValueError, OSError):
        return None


def save_checkpoint(path: Path, strategy, generation: int, history: list, best: dict):
    """Persist enough CMA-ES state to regenerate the next population."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generation": generation,
        "history": history,
        "best": best,
        "numpy_random_state": np.random.get_state(),
        "strategy": {
            name: getattr(strategy, name)
            for name in (
                "centroid",
                "sigma",
                "pc",
                "ps",
                "C",
                "diagD",
                "B",
                "BD",
                "cond",
                "update_count",
            )
        },
    }
    # write beside the target and rename, so a crash mid-write cannot destroy
    # the checkpoint that is already there
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f)
    tmp_path.replace(path)


def load_checkpoint(path: Path, strategy):
    """Restore a checkpoint onto a freshly prepared strategy."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    for name, value in payload["strategy"].items():
        setattr(strategy, name, value)
    np.random.set_state(payload["numpy_random_state"])
    return payload["generation"], payload["history"], payload["best"]


if __name__ == "__main__":
    args = parse_args()

    n_base = n_opt_params(args.model_version)
    n_clusters = len(proj_clusters(args.model_version))
    lower, upper, p0, fixed_base, off_fit_source = search_space(
        args.model_version, args.dbs
    )

    max_evals = (
        args.max_evals if args.max_evals is not None else paramsS["deap_cma.run.max_evals"]
    )
    lambda_ = args.lambda_ if args.lambda_ is not None else paramsS["deap_cma.lambda"]

    run_tag = f"{args.model_version}_{args.dbs}_run_{args.optimization_run}"
    log_dir = DATA_DIR / "individual_logs" / run_tag
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = DATA_DIR / "checkpoints" / f"cma_{run_tag}.pkl"

    print(
        f"fitting {args.model_version}, DBS {args.dbs}: {len(lower)} free parameters, "
        f"lambda {lambda_}, up to {max_evals} generations\n"
        f"logs: {log_dir}\ncheckpoint: {checkpoint_file}"
    )

    ### the generation counter, loss history and best individual survive restarts.
    ### DeapCma's own hall of fame does not: it is rebuilt when the class is
    ### constructed, so after a resume it only knows the generations of that
    ### segment and would report a best worse than one already found.
    state = {
        "generation": 0,
        "history": [],
        "best": {"loss": float("inf"), "params": None, "generation": None},
    }

    def evaluate_function(population):
        """Evaluate a whole generation, one subprocess per individual."""
        generation = state["generation"]
        args_list = [
            get_loss_args(individual, index, args, fixed_base, n_base)
            for index, individual in enumerate(population)
        ]
        loss_files = [
            DATA_DIR
            / f"loss_{args.model_version}_{args.dbs}_run_{args.optimization_run}"
            f"_ind{index}.json"
            for index in range(len(population))
        ]
        log_paths = [
            log_dir / f"gen{generation:04d}_ind{index}.log"
            for index in range(len(population))
        ]

        # A crashed individual leaves last generation's loss file behind, which
        # would be read as if it were this generation's result.
        for loss_file in loss_files:
            loss_file.unlink(missing_ok=True)

        start = time.time()
        exit_codes = run_individuals(args_list, log_paths, n_jobs=lambda_)

        loss_list = []
        failed = []
        for index, (exit_code, loss_file) in enumerate(zip(exit_codes, loss_files)):
            loss = read_loss(loss_file) if exit_code == 0 else None
            if loss is None:
                failed.append((index, exit_code))
                loss = FAILED_LOSS
            loss_list.append((loss,))

        finite_losses = [loss for (loss,) in loss_list if loss < FAILED_LOSS]
        print(
            f"generation {generation}: {len(finite_losses)}/{len(population)} evaluated "
            f"in {(time.time() - start) / 60:.1f} min"
            + (
                f", best {min(finite_losses):.4f}"
                if finite_losses
                else ", no individual survived"
            )
        )
        for index, exit_code in failed:
            print(
                f"  individual {index} failed (exit {exit_code}); "
                f"see {log_paths[index]}"
            )

        if len(failed) * 2 > len(population):
            raise RuntimeError(
                f"{len(failed)} of {len(population)} individuals failed in "
                f"generation {generation}. That is not bad luck -- fix the cause "
                f"before burning more days. The logs are in {log_dir}."
            )

        best_index = int(np.argmin([loss for (loss,) in loss_list]))
        if loss_list[best_index][0] < state["best"]["loss"]:
            state["best"] = {
                "loss": loss_list[best_index][0],
                "params": [float(p) for p in population[best_index]],
                "generation": generation,
            }

        state["history"].append(
            {
                "generation": generation,
                "losses": [loss for (loss,) in loss_list],
                "n_failed": len(failed),
                "exit_codes": list(exit_codes),
            }
        )
        state["generation"] = generation + 1
        save_checkpoint(
            checkpoint_file,
            deap_cma.deap_dict["strategy"],
            state["generation"],
            state["history"],
            state["best"],
        )
        return loss_list

    ### create the CMA-ES instance
    deap_cma = DeapCma(
        lower=lower,
        upper=upper,
        evaluate_function=evaluate_function,
        max_evals=max_evals,
        p0=p0,
        hard_bounds=True,
        plot_file=f"{paramsS['data_folder']}/deap_cma_plot_{run_tag}.png",
        cma_params_dict={"lambda_": lambda_},
    )

    remaining_evals = max_evals
    if args.resume:
        if not checkpoint_file.exists():
            raise FileNotFoundError(
                f"--resume given but no checkpoint at {checkpoint_file}."
            )
        done, history, best = load_checkpoint(
            checkpoint_file, deap_cma.deap_dict["strategy"]
        )
        state["generation"] = done
        state["history"] = history
        state["best"] = best
        remaining_evals = max_evals - done
        print(f"resuming after generation {done}; {remaining_evals} generations left")
        if remaining_evals <= 0:
            print("nothing left to do")
            sys.exit(0)

    ### compile one model per individual, in parallel, before the search starts
    number_of_individuals = deap_cma.deap_dict["strategy"].lambda_
    if not args.skip_compile:
        compile_args = [
            get_loss_args(
                [0.0] * len(lower), index, args, fixed_base, n_base
            )
            + ["--compile"]
            for index in range(number_of_individuals)
        ]
        compile_logs = [
            log_dir / f"compile_ind{index}.log" for index in range(number_of_individuals)
        ]
        print(f"compiling {number_of_individuals} models...")
        compile_codes = run_individuals(
            compile_args, compile_logs, n_jobs=number_of_individuals
        )
        if any(code != 0 for code in compile_codes):
            broken = [i for i, code in enumerate(compile_codes) if code != 0]
            raise RuntimeError(
                f"compilation failed for individuals {broken}; see {log_dir}"
            )

    ### run the optimization
    deap_cma_result = deap_cma.run(max_evals=remaining_evals)
    deap_cma_result["history"] = state["history"]

    # DeapCma reports the best of the segment it just ran; state["best"] spans
    # every generation of this run including the ones before a resume.
    if state["best"]["params"] is not None and (
        state["best"]["loss"] < deap_cma_result["best_fitness"]
    ):
        print(
            f"best of this segment {deap_cma_result['best_fitness']:.4f} is worse "
            f"than the best of generation {state['best']['generation']} "
            f"({state['best']['loss']:.4f}); reporting the latter"
        )
        deap_cma_result["best_fitness"] = state["best"]["loss"]
        for index, value in enumerate(state["best"]["params"]):
            deap_cma_result[f"param{index}"] = value

    # An on-run searches only the putamen scalings and the 3 DBS parameters, so
    # param0..paramN are half a vector on their own. Record the fixed base it was
    # conditioned on and where that came from, otherwise the one artefact meant to
    # answer "which parameters had to change" cannot be read without replaying
    # load_best_off_fit's tie-break by hand.
    deap_cma_result["dbs"] = args.dbs
    deap_cma_result["model_version"] = args.model_version
    deap_cma_result["fixed_base"] = (
        list(fixed_base) if fixed_base is not None else None
    )
    deap_cma_result["fixed_base_source"] = off_fit_source

    save_variables(
        variable_list=[deap_cma_result],
        name_list=[f"deap_cma_result_{run_tag}"],
        path=f"{paramsS['data_folder']}/deap_cma_result",
    )
    print(f"best loss {deap_cma_result['best_fitness']:.4f}")
