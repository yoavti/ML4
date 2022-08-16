# Python Scripts

## Experiment Scripts

* `aug_experiments.py`: Runs the augmentation experiments (Part d). Could either run for single configuration (`run_args`) or for all configurations given in `best.csv` (see `find_best` in `results_processing.py`) (`run_best`). Run with `-h` argument for explanation of arguments.
* `run_experiments.py`: Runs the main experiments of Parts b and c. Executes an experiment for a single configuration of dataset and k, comparing different fs methods and classifiers. Run with `-h` argument for explanation of arguments.
* `toy.py`: Runs simple toy experiments. Simply add printing of results and run.

## Post-Processing Scripts

All files in this category require no arguments. Simply execute them for them to take their effect.

* `calc_stat.py`: Executes statistical tests for Part e.
* `clean.py`: Removes empty files and folders.
* `compare.py`: Compares both aug experiments to original corresponding experiments, and our improvement to the original algorithm.
* `missing.py`: Finds any missing configurations not covered by results.
* `preprocessed_results.py`: Copies results of fs preprocessing to separate folder.
* `results_processing_pipeline.py`: Combines all post-processing steps, as well as the aug experiments.

# SBATCH Scripts

These files are used to run the experiments on the cluster.

* `run.sbatch`: Allows running experiments across all selected datasets (see `datasets.txt`) for a single k value.
* `run_missing.sbatch`: Allows running missing configurations as given in `missing.csv` (see `missing.py`).
* `run_missing_dataset.sbatch`: Allows running experiments across all available k values for a selected dataset.