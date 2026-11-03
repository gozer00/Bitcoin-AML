# Reproducing the results

Run all commands from the repository root, inside the environment created in the
[README quick start](../README.md#quick-start).

| Level | Needs | Time | Result |
| --- | --- | --- | --- |
| [1. Smoke test](#1-smoke-test) | Dataset | About 1 minute | Confirms the pipeline runs |
| [2. Rebuild from archived runs](#2-rebuild-from-archived-runs) | Results archive | About 20 minutes | Every table and figure, exact |
| [3. Full rerun](#3-full-rerun) | Dataset, GPU recommended | About 24 hours | All 538 runs from scratch |

## Environment

The published results were produced with:

| Component | Version |
| --- | --- |
| Operating system | Ubuntu 24.04 on WSL2 |
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu130 (CUDA 13.0) |
| GPU | NVIDIA GeForce GTX 1650 Ti, 4 GB |
| CPU | 12 logical cores |
| Memory | 6 GB RAM, 32 GB swap |

`requirements.txt` pins every direct dependency. If `pip` installs a PyTorch build that does not
suit your GPU driver, install a matching build from <https://pytorch.org/get-started/locally/>
before installing the requirements. Models fall back to the CPU when no GPU is available, which
is slower for the GNNs and has not been benchmarked.

## Prepare the data

Download the [Elliptic dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) and
place these unmodified files in `data/raw/`:

- `elliptic_txs_features.csv`
- `elliptic_txs_classes.csv`
- `elliptic_txs_edgelist.csv`

Then run:

```bash
python scripts/prepare_data.py
diff --strip-trailing-cr data/processed/elliptic_v1/manifest.json docs/elliptic_v1_manifest.json
```

An empty `diff` confirms that your input and prepared arrays match the published ones byte for byte.

## 1. Smoke test

```bash
python scripts/run_benchmark.py --smoke
```

This writes two runs to `outputs/runs/smoke/`. The logistic-regression run should report
`test_average_precision=0.5331` exactly.

## 2. Rebuild from archived runs

This level fits no model. The archive holds the `manifest.json` and `predictions.npz` of every run.

```bash
curl -L -o results-archive.tar.gz https://github.com/gozer00/Bitcoin-AML/releases/download/v1.0.0/results-archive.tar.gz
echo "7abdf82415e591dda10a63a33776893b6874f389750b89a2a47ea608fda288b3  results-archive.tar.gz" | sha256sum --check
tar -xzf results-archive.tar.gz
bash scripts/reproduce_results.sh
python scripts/verify_results.py --mode archive
```

The dataset is optional here. Without it, the descriptive tables are taken from `results/tables/`.

`verify_results.py` compares every rebuilt CSV with `results/tables/`, and all of them should match.

## 3. Full rerun

```bash
python scripts/run_benchmark.py --skip-existing
bash scripts/reproduce_results.sh
python scripts/verify_results.py --mode rerun
```

Delete or move any archived runs first if you want a fresh run. Otherwise `--skip-existing` skips
them. To run only one part, add `--group local21`, `--group reference` or `--group evolvegcn`.

`--skip-existing` makes the run resumable: after an interruption, the same command continues with
the missing config and seed pairs. If a run was interrupted while saving, the command names the
incomplete directory to delete.

To run a single experiment:

```bash
python scripts/run_experiment.py --config configs/experiments/local21/lr_local21_strict.yaml
```

### Expected variation

| Models | Behaviour on the same machine |
| --- | --- |
| Logistic regression, XGBoost, MLP | Identical scores in our sample rerun |
| Random forest | Same AP, with scores differing by floating-point rounding only |
| GCN, GraphSAGE, predecessor GNN | Differ between runs, even at the same seed |

GNN runs differ between repeats although the seed is fixed. A sample rerun of `local_21` and
`local_93` configs at seed 42 mostly changed AP by less than 0.01. The noise can also change which
hyperparameters win selection, which moves the result further: for
`gcn_local93_bidirectional_strict`, AP moved from 0.489 to about 0.52.

Different hardware, drivers or library versions may change results for every model.

`verify_results.py --mode rerun` therefore applies two rules to mean AP:

- Tabular models fail if they differ from the published value by more than 1e-6.
- GNNs are flagged, without failing, if they differ by more than twice the published cross-seed
  standard deviation, or by more than 0.01 if that is larger.

## From scripts to results

`scripts/reproduce_results.sh` runs these steps in order:

| Step | Script | Output |
| --- | --- | --- |
| Cross-seed summaries | `build_results.py` | `outputs/tables/{local21,reference,evolvegcn}/` |
| Peak memory | `build_resource_usage_summary.py` | `outputs/tables/local21/resource_usage_summary.csv` |
| Dataset statistics | `build_descriptive_stats.py` | `outputs/tables/descriptive/`, `outputs/figures/sample_trend.png` |
| Calibration, alert budgets, temporal bootstrap | `build_post_hoc_analysis.py` | `outputs/tables/post_hoc/`, `outputs/figures/post_hoc/` |
| Paired bootstrap comparisons | `build_paired_comparisons.py` | `outputs/tables/paired_comparisons/` |
| Figures | `build_figures.py` | `outputs/figures/` |

The published copies are in `results/`.

| Thesis content | File in `results/` |
| --- | --- |
| Summary statistics and chronological splits | `tables/descriptive/summary_counts.csv` |
| Per-timestep statistics | `tables/descriptive/descriptives_per_timestep.csv` |
| Sample trend | `figures/sample_trend.png` |
| Main `local_21` results | `tables/local21/results_summary.csv`, `tables/local21/resource_usage_summary.csv` |
| Feature-set comparison | `tables/local21/results_summary.csv`, `tables/reference/results_summary.csv` |
| Direction and hop depth | `tables/local21/results_summary.csv` |
| Operational metrics | `tables/post_hoc/alert_budget_summary.csv`, `tables/post_hoc/calibration_summary.csv` |
| Paired differences | `tables/paired_comparisons/paired_ap_differences.csv` |
| EvolveGCN appendix | `tables/evolvegcn/results_summary.csv` |
| AP per timestep | `figures/fig_per_timestep_ap_local21.pdf` |
| Precision at alert budgets | `figures/fig_precision_budget_local21.pdf` |
| Reliability diagrams | `figures/fig_reliability_local21.pdf` |

## Run files

Each run directory is named `{experiment}_{config hash}_seed{seed}_{timestamp}` and contains:

- `manifest.json`: resolved configuration and hash, dataset hashes, selected hyperparameters,
  overall and per-timestep metrics, peak memory.
- `predictions.npz`: `tx_id`, `timestep`, `y_true` and `scores` for every test transaction.

`config_hash` covers everything that defines an experiment except the seeds.
