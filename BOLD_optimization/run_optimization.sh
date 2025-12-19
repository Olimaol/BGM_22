# create data folder
python create_data_folder.py
# Run multiple deap cma optimizations which by themselves run multiple simulations in parallel (13) -> n*13 cores needed
python deap_cma_opt.py --dbs off --optimization-run 1
python deap_cma_opt.py --dbs off --optimization-run 2
# After dbs off optimizations run dbs on optimizations
python deap_cma_opt.py --dbs on --optimization-run 1
python deap_cma_opt.py --dbs on --optimization-run 2