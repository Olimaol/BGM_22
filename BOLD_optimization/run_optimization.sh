#!/usr/bin/env bash
# Launch the DBS-off fits, then the DBS-on fits that carry their result over.
#
# Run from BOLD_optimization/: the cache state files store the cortical rate path
# verbatim, so it has to be spelled the same way it was at build time (TODO.md 13).
set -euo pipefail

# There is no `python` on PATH in the compneuro environment; name the interpreter.
PYTHON=${PYTHON:-/home/oliver/miniforge3/envs/compneuro/bin/python}
MODEL_VERSION=${MODEL_VERSION:-v07}

# numpy's OpenBLAS honours OMP_NUM_THREADS even though ANNarchy does not, so
# without this every one of the lambda processes grabs its own thread pool.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# create data folder
"$PYTHON" create_data_folder.py

# Run multiple deap cma optimizations which by themselves run multiple
# simulations in parallel (lambda) -> n*lambda cores needed
"$PYTHON" deap_cma_opt.py --dbs off --model-version "$MODEL_VERSION" --optimization-run 1 &
"$PYTHON" deap_cma_opt.py --dbs off --model-version "$MODEL_VERSION" --optimization-run 2 &
wait

# The DBS-on runs seed from the best DBS-off result via load_best_off_fit, so
# this wait is a real dependency, not just politeness.
"$PYTHON" deap_cma_opt.py --dbs on --model-version "$MODEL_VERSION" --optimization-run 1 &
"$PYTHON" deap_cma_opt.py --dbs on --model-version "$MODEL_VERSION" --optimization-run 2 &
wait
