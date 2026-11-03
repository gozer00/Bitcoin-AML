# Architecture

Dependencies point from orchestration toward domain code:

```text
scripts → experiments → data, splits, features, graphs, models, evaluation, artifacts
```

Domain modules never import `scripts` or `experiments`. Concrete models implement the
contracts in `models/base.py` and `models/predictor.py`, and evaluation depends on those
contracts, never on a concrete model.

## Modules

| Package under `src/bitcoin_aml/` | Responsibility |
| --- | --- |
| `data/` | Raw-to-canonical preparation, integrity checks, dataset loading |
| `splits/` | Chronological splits and temporal leakage guards |
| `features/` | Feature-set selection and fold-scoped preprocessing |
| `graphs/` | Graph views, access policies, predecessor extraction and batching |
| `models/` | Model registry and adapters: `tabular/`, `graph/` |
| `evaluation/` | Metrics, per-timestep results, calibration, alert budgets, bootstrap |
| `experiments/` | YAML resolution, model selection, the end-to-end runner |
| `artifacts/` | Run manifests, prediction files, cross-seed aggregation |
| `utils/` | Hashing, seeding, logging, device and memory tracking |

## Experiment flow

`ExperimentRunner` resolves one validated configuration from dataset, protocol and model
YAML files and hashes it. For each seed it selects hyperparameters, refits on timesteps
1-34, evaluates once on timesteps 35-49, and persists `manifest.json` and
`predictions.npz` in a run directory named after the experiment, hash, seed and timestamp.

## Data views

`EllipticDataset` is the immutable canonical dataset. Experiments do not create separate
train, validation or test datasets. They build index-based views over one canonical node
ordering.

Three concepts are deliberately separate:

1. Temporal membership determines which timestep a node belongs to.
2. Label availability determines whether a node can be a supervised target.
3. Graph availability determines whether a node can take part in message passing.

Unknown-label nodes therefore stay available to graph construction when the protocol
permits it, although they never contribute to supervised loss or evaluation.

Feature sets are column views over the canonical feature matrix:

- `local_21`: columns 0-20
- `local_93`: columns 0-92
- `aggregate_72`: columns 93-164
- `all_165`: columns 0-164

Feature selection performs no scaling. Preprocessing is fitted only through explicit fit
indices, and under the strict protocol only on labeled fit observations.
