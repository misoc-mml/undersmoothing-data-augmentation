# Undersmoothing Causal Estimators with Generative Trees

This code accompanies the paper *Undersmoothing Causal Estimators with Generative Trees* [1].

The experimental setup is based on the [CATE benchmark](https://github.com/misoc-mml/cate-benchmark) and extended to test our proposed DeGeTs framework.

## Installation
The easiest way to replicate the running environment is through [Anaconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/download.html#anaconda-or-miniconda). Once installed, follow the steps below.

1. Download the repo.
2. Enter the directory (i.e. `cd cate-benchmark`).
3. Run the following command to recreate the 'cate-bench' conda environment:

`conda env create -f environment.yml`

4. Download datasets from [here](https://essexuniversity.box.com/s/69hiufo5cejvjux7a6zrsie5v7fd5s8o). Once downloaded, extract them to 'datasets' directory.

## Important Files

- `main.py` - main script to run the experiments.
- `models/tree_balancing.py` - our proposed DeGeTs framework (DeGeDTs and DeGeF).
- `utils.py` - helper functions.

## Experiments
Head to `experiments' folder to run the main script in a more automated and convenient way.

experiments/full_run.sh - Replicates 100% of the experiments as in the paper. Note this full setup takes days/weeks to complete, even on fairly strong machines.

experiments/demo.sh - A small experimental run for demonstration purposes. It involves the IHDP data set (10/1000 iterations) and only selected estimators.

Note the intermediate results printed to the console while running the scripts show mean +- standard error. The final results in the paper were computed separately from outputted CSV files in order to show mean +- 95% confidence intervals.

## More Info
For more details about how to use the framework and analyse the results, please visit the [CATE benchmark](https://github.com/misoc-mml/cate-benchmark) website.

## References
[1] D. Machlanski, S. Samothrakis, and P. Clarke, ‘Undersmoothing Causal Estimators with Generative Trees’, arXiv:2203.08570 [cs, stat], Mar. 2022, doi: 10.48550/arXiv.2203.08570.