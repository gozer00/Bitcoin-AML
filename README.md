# Leakage-Free and Direction-Preserving Graph Learning for Illicit-Transaction Detection in Bitcoin

Code, configurations and results for a master's thesis at Ulm University. The study
detects illicit transactions in the [Elliptic Bitcoin dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
(Weber et al., 2019) under chronological, leakage-free evaluation, and asks three questions:

1. How much do features computed after the sample period inflate reported performance?
2. Does message passing over incoming payment flows add information to transaction features?
3. Does preserving the direction of those flows matter?

## Main results

All models are selected on timesteps 30-34, refitted on timesteps 1-34 and evaluated once on
timesteps 35-49. Values are mean average precision (AP) over five seeds, with cross-seed
standard deviations, on the 21 transaction-local features (`local_21`).

| Model | AP | SD |
| --- | ---: | ---: |
| Predecessor GNN, two hops, incoming edges | 0.547 | 0.015 |
| Logistic regression, Yeo-Johnson scaling | 0.533 | 0.000 |
| GraphSAGE, sampled | 0.528 | 0.017 |
| GraphSAGE, full batch | 0.527 | 0.031 |
| Random forest | 0.519 | 0.062 |
| XGBoost | 0.368 | 0.025 |
| Logistic regression, standard scaling | 0.321 | 0.000 |
| MLP | 0.260 | 0.094 |
| GCN | 0.243 | 0.014 |

- The predecessor GNN has the highest AP, but its lead over logistic regression with
  Yeo-Johnson scaling is **not significant** (+0.014, 95% CI [-0.014, 0.040]).
- Its lead over random forest is +0.028 (95% CI [0.001, 0.055]).
- Address-level features computed after the sample period inflate AP by 0.13 to 0.48 for every
  model family except logistic regression, which barely changes, and they reorder the ranking:
  tree ensembles lead on `local_93` and `all_165`.
- The ranking of edge directions is unstable across hop depths and scaling.
- AP collapses for all models from timestep 43 onward (best AP 0.119 at timestep 43), with a
  partial recovery at timestep 46.

Complete tables and figures are in [`results/`](results/).

## Quick start

Requires Python 3.12. A CUDA GPU is recommended, and 4 GB of GPU memory is enough.

```bash
git clone https://github.com/gozer00/Bitcoin-AML.git
cd Bitcoin-AML
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python -m pytest
```

Run all commands from the repository root. Tests that need the real dataset are skipped until it
has been prepared.

Download the [Elliptic dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set), put
its three CSV files into `data/raw/`, and prepare it:

```bash
python scripts/prepare_data.py
python scripts/run_benchmark.py --smoke
```

The smoke test trains two models at one seed in about a minute.

## Reproducing the results

[`docs/reproducing.md`](docs/reproducing.md) is the complete guide. There are three levels of effort:

| Goal | Command | Time |
| --- | --- | --- |
| Check that everything works | `python scripts/run_benchmark.py --smoke` | About 1 minute |
| Rebuild all tables from archived runs | `bash scripts/reproduce_results.sh` | About 20 minutes |
| Rerun all experiments | `python scripts/run_benchmark.py --skip-existing` | About 24 hours |

## Repository structure

```text
configs/         Experiment, model and protocol definitions
data/            Local dataset location, not tracked by git
docs/            Reproduction guide, protocols, architecture, data contract
notebooks/       Exploratory data analysis
outputs/         Generated runs, tables and figures, not tracked by git
results/         Published tables and figures
scripts/         Command-line entry points
src/bitcoin_aml/ Library code
tests/           Unit, integration and regression tests
```

## Documentation

- [Reproducing the results](docs/reproducing.md)
- [Experiment protocols](docs/experiment_protocols.md): splits, leakage arguments, models, caveats
- [Data contract](docs/data_contract.md): raw data, canonical arrays, feature semantics
- [Architecture](docs/architecture.md): module structure and dependency rules

## Citation

Please cite this work using [`CITATION.cff`](CITATION.cff). The feature semantics behind `local_21`
come from Šafář et al. (2026), <https://github.com/nesfit/DeanonymizedEllipticBitcoinDataset>.

## License

The code is released under the [MIT License](LICENSE). The license does not cover the Elliptic
dataset, which is not included here and remains subject to its own terms.

## Author

Erich Gozebina, Master of Science in Mathematics and Business, Ulm University
