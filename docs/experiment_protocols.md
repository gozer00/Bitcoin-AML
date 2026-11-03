# Experiment protocols

This document defines how every published result is produced. For the commands
that run them, see [`reproducing.md`](reproducing.md).

## Strict protocol

Protocol `thesis_strict_v1` (`configs/protocols/thesis_strict_v1.yaml`) uses
chronological splits:

| Split | Timesteps |
| --- | --- |
| Training | 1-29 |
| Validation | 30-34 |
| Test | 35-49 |

### Model selection

1. Select the requested feature set.
2. Fit preprocessing on labeled training nodes from timesteps 1-29.
3. Transform training and validation observations with this preprocessing.
4. Fit each candidate model on labeled training nodes.
5. Evaluate candidates on labeled validation nodes.
6. Select the configuration with the highest validation average precision (AP).

No validation or test observation influences preprocessing during selection.

### Final refit

1. Discard the preprocessing object and model from selection.
2. Combine labeled training and validation targets from timesteps 1-34.
3. Fit new preprocessing on these observations.
4. Fit a new model with the selected hyperparameters on timesteps 1-34.

### Final test

Timesteps 35-49 are transformed with the final preprocessing and evaluated once.
Test data influence no preprocessing, selection or fitting decision. Decisions
use a fixed threshold of 0.5.

AP is the primary metric because of the class imbalance. It selects the
hyperparameters and is reported overall and per test timestep, together with
precision, recall and F1.

## Feature sets

Column meanings and their temporal status are in
[`data_contract.md`](data_contract.md), section 2.1.

| Feature set | Columns | Role |
| --- | --- | --- |
| `local_21` | 0-20 | Main benchmark, free of future address-level information |
| `local_93` | 0-92 | Reference, includes address statistics computed after the sample period |
| `all_165` | 0-164 | Reference, also includes neighbour aggregates |

Šafář et al. (2026) show that columns 21-92 were computed at block height 575,059,
after the sample period, so `local_93` and `all_165` serve only to measure how much
that information inflates results.

## Scaling

Experiment configs select one of two strategies:

- `standard`: standardization of every column.
- `yeo_johnson_skewed` (configs ending in `_yeojohnson`): a Yeo-Johnson transform of
  columns with absolute skewness above 1.0 on the fit rows. All other columns stay
  unchanged.

Both are fitted only on admissible fit rows. The column selection is statistical
because the Elliptic columns have no published names.

## Graph access

Graph-capable models receive an `edge_index` governed by `protocol.graph_access`
(`graphs/policies.py`), resolved per phase by `resolve_phase_edge_index`:

- `strict_rolling` (`thesis_strict_v1`): selection and refit see the graph masked to
  the last validation timestep, so a model is fit as if test-timestep nodes did not
  exist. Evaluation uses the full graph, because a test node's only possible
  neighbours are other nodes of its own timestep.
- `full_transductive` (`thesis_fulltransductive_v1`): every phase sees the full
  graph. Test labels are still never used before the final evaluation.

### Why the masking is safe, and why it changes little

Elliptic has no cross-timestep edges, which is enforced during data preparation.
Graph adapters use per-node operations only (linear layers, activations, dropout)
and no cross-node normalization such as BatchNorm. A visible node's forward pass
therefore never depended on future nodes.

This gives a structural claim: `strict_rolling` and `full_transductive` should agree
up to noise. A real divergence would indicate a bug. Published conclusions rest on
this structural argument, so no `local_21` transductive configs exist.

The constraint is enforced by `tests/regression/test_gcn_no_future_leakage.py`,
`test_graphsage_no_future_leakage.py` and `test_runner_graph_access_policy.py`.

## Model families

| Family | Registered names | Graph view |
| --- | --- | --- |
| Tabular | `logistic_regression`, `random_forest`, `xgboost`, `mlp` | None |
| Static GNN | `gcn`, `graphsage_full_batch`, `graphsage_sampled` | `bidirectional` |
| Predecessor GNN | `predecessor_gnn_{one,two,three}_hop` | `incoming`, `reversed`, `bidirectional` |
| Temporal GNN | `evolvegcn_h`, `evolvegcn_o` | `bidirectional` |

Hyperparameter grids are in `configs/models/`. `mlp` shares its search-space shape
with `gcn`, so the two are directly comparable.

`graphsage_full_batch` and `graphsage_sampled` are separate names because an
experiment selects a model by name only and the two modes need different
hyperparameters. The sampled variant uses a stateless, adapter-local neighbour hash.

### Predecessor-transaction models

The predecessor GNN (called Dir-GraphSAGE in the thesis) combines a transaction's
own features with a summary of the transactions feeding into it. Each layer computes
`concat(self, mean(local predecessors))`.

`fit` and `predict_scores` each build an `IncomingCSR` from the `edge_index` they
receive, then a `PredecessorBatch` covering the exact k-hop predecessor neighbourhoods
of all targets. They run `hops` rounds of message passing on that renumbered
subgraph, with one full batch per call.

The depth is fixed per registered name through the `_hops` class attribute. `graph.hops`
in an experiment YAML documents that depth and is not passed to the model, so the two
must be kept in sync.

Aggregation direction matters here. `PredecessorBatch.local_edge_index` is directed,
so `_local_mean_predecessor_adjacency` uses `rows = destination`. The symmetric
convention in `graphsage.py` is only safe for already bidirectional edges.
`tests/unit/test_predecessor_gnn_adapter.py::test_hop_depth_controls_receptive_field`
covers this.

Leakage safety rests on three points:

- (a) Layers use per-node operations only.
- (b) Under the `incoming` view, edges are a subset of the bidirectional edge set of
  the static GNNs.
- (c) Fit-phase masking is inherited, because the batch is built from the `edge_index`
  the runner already masked.

Point (b) does not hold for the ablations below. Points (a) and (c) are independent
of direction, and `tests/regression/test_predecessor_gnn_no_future_leakage.py` checks
all views and hop depths.

### Direction ablations

The same predecessor adapters run against other views, with no change to model code:

- `reversed` swaps source and target. It tests whether genuine payment direction
  matters or whether any local neighbourhood would do.
- `bidirectional` tests whether restricting to predecessors costs predictive power.

### Temporal models

`evolvegcn_h` and `evolvegcn_o` follow Pareja et al. (AAAI 2020):

- EvolveGCN-H: `W_t = GRU(H_t, W_{t-1})`. The GRU input is the layer's node embeddings
  of that timestep, summarized by differentiable top-k pooling.
- EvolveGCN-O: `W_t = LSTM(W_{t-1})`, with no external input and a persistent cell state.

Recurrence adds a coupling over time, so the leakage question becomes whether a later
timestep can reach an earlier one. Both adapters replay the recurrence strictly forward
from the earliest timestep up to a per-call `t_end`, and each timestep's outputs are
final before the next is processed. For H, the GRU input is computed only from the
local subgraph of timestep `t`.

`t_end` comes from `max(timestep[fit_idx])`, not from the graph, which makes fit-phase
behaviour identical under both graph-access policies. This is tested by
`tests/regression/test_evolvegcn_{h,o}_no_future_leakage.py`.

## Published benchmark

| Group | Configs | Seeds | Content |
| --- | --- | --- | --- |
| `local21` | 32 | 42-46 | All model families except EvolveGCN, both scalers, all predecessor views and depths |
| `reference` | 74 | 42-46 | `local_93` and `all_165` counterparts, plus 14 `full_transductive` configs |
| `evolvegcn` | 8 | 42 | H and O, `local_93` and `all_165`, both graph-access policies |

## Post-hoc analyses

All analyses read persisted `predictions.npz` files and fit no model. Each file holds
aligned `tx_id`, `timestep`, `y_true` and `scores` arrays for the test set.

- **Calibration:** reliability diagrams, Brier score and expected calibration error
  over 10 bins. No recalibration is applied.
- **Alert budgets:** precision at 0.5%, 1%, 2%, 5% and 10% of scored transactions.
- **Temporal bootstrap:** per-timestep confidence intervals on AP and F1, with 2,000
  resamples and 95% coverage.
- **Paired comparisons:** a paired bootstrap of seed-averaged AP differences for 31
  predefined contrasts, stratified by timestep. The same resample applies to all
  seeds of both models. The contrasts are listed in
  `scripts/build_paired_comparisons.py`.

The bootstrap uses `average_precision_metric` and `f1_metric` from
`evaluation/bootstrap.py`, which are fast numpy versions verified against scikit-learn
in `tests/unit/test_bootstrap.py`. They differ from scikit-learn only under exact score
ties. All other tables use `evaluation/metrics.py`.

## Known caveats

- **Seed variance of logistic regression is exactly zero.** `liblinear` solves a convex
  problem with a unique optimum, so the seed only changes the shuffle order. This is
  expected behaviour.
- **Random forest selection is bimodal on `local_21`.** Validation AP is nearly tied
  across seeds, but seeds 42, 44 and 45 select `max_depth=8` (test AP 0.56-0.58) while
  seeds 43 and 46 select `max_depth=32` (test AP 0.44-0.45). This explains the cross-seed
  standard deviation of about 0.06.
- **F1 of `predecessor_gnn_one_hop_local21_incoming_strict` is about 0.002.** Its scores
  vary, but very few reach the fixed 0.5 threshold. AP (0.111) is the meaningful figure.
- **GNN results are not bit-reproducible.** See
  [`reproducing.md`](reproducing.md#expected-variation).
- **EvolveGCN-O can collapse to a constant score.** With an earlier, wider search space
  (`hidden_dim` 4, 8, 16 and `dropout` 0, 0.2), `evolvegcn_o_local93_bidirectional_strict`
  scored every test node exactly 0.5. The root cause was not investigated. None of the 8
  published EvolveGCN runs does this: each has roughly 15,000 distinct test scores.
  EvolveGCN results are single-seed and should be read as preliminary.
