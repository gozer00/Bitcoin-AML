# Elliptic Data Contract

## 1. Source

The project uses the original Elliptic Bitcoin transaction dataset.

The raw dataset must lie under `data/raw/` and consists of:

- `elliptic_txs_features.csv`
- `elliptic_txs_classes.csv`
- `elliptic_txs_edgelist.csv`

Raw files are treated as immutable.

## 2. Transaction features

The feature file contains:

- transaction ID
- timestep
- 165 numerical features
- no column names

Features 1-93 are transaction-local features.
Features 94-165 are aggregate neighborhood features.

The canonical zero-based feature ranges are therefore:

- local: columns 0-92
- aggregate: columns 93-164

### 2.1 Feature semantics and the `local_21` subset

Šafář et al. (2026) reverse-engineered the feature construction; the
column meanings are published at
<https://github.com/nesfit/DeanonymizedEllipticBitcoinDataset>. That
repository numbers the timestep as feature 1, so its features 2-166 map to
canonical columns 0-164 here (canonical = repository index - 2).

| Canonical columns | Repository features | Content | Temporal status |
| --- | --- | --- | --- |
| 0-20 | 2-22 | transaction-local: volume, fee, input/output counts, unique input/output addresses, input/output value statistics | computed at transaction time |
| 21-92 | 23-94 | address-level statistics of input/output addresses (received, sent, available, in/out transaction counts, lifetime; min/max/std/mean/Pearson/Spearman) | computed at block height 575,059 (2019-05-08), after the sample period |
| 93-164 | 95-166 | aggregates over input and output neighbour transactions (value, block height, fee, input/output counts) | include successor transactions |

Selectable feature sets (`features/specification.py::FeatureSetName`):

- `local_21`: columns 0-20, the only columns free of future address-level
  information, and the main feature set of this study
- `local_93`: columns 0-92
- `aggregate_72`: columns 93-164
- `all_165`: columns 0-164

`data/prepare.py` records only the original `local_93`/`aggregate_72`
groups in the processed-dataset metadata; `local_21` is a selector-level
view and deliberately does not alter prepared data.

No scaling or imputation is performed during dataset preparation.

## 3. Labels

Raw Elliptic labels are converted as follows:

| Raw | Meaning | Canonical |
| --- | --- | ---: |
| `1` | illicit | `1` |
| `2` | licit | `0` |
| `unknown` | unknown | `-1` |

Unknown transactions remain in the graph but will not be used as
supervised targets.

## 4. Timesteps

The dataset contains 49 ordered timesteps.

Every transaction belongs to exactly one timestep.

The original Elliptic graph contains no edges between different
timesteps. Dataset preparation uses this invariant for verification.

## 5. Edge direction

The raw edge list is interpreted as:

`txId1 -> txId2`

This direction is preserved during canonical conversion.

The preprocessing pipeline does not add reverse edges.

## 6. Canonical node indexing

Transactions are mapped to contiguous zero-based node indices.

The node index preserves feature-file row order.

`tx_id.npy` stores the corresponding original transaction identifier.

## 7. Canonical arrays

| File | Meaning | Dtype |
| --- | --- | --- |
| `tx_id.npy` | original transaction ID | `int64` |
| `timestep.npy` | Elliptic timestep | `int16` |
| `features.npy` | 165 raw features | `float32` |
| `labels.npy` | canonical class | `int8` |
| `edge_index.npy` | directed graph indices | `int64` |

## 8. Integrity requirements

Dataset preparation fails if:

- transaction IDs are duplicated
- feature values are non-numeric or non-finite
- class entries are missing or duplicated
- unexpected class values occur
- edge endpoints do not exist
- directed edges are duplicated
- self-edges occur
- an edge connects different timesteps

No invalid observation is silently dropped or corrected.

## 9. Versioning and hashes

Each raw and processed file receives an SHA-256 digest.

`manifest.json` records the data schema, hashes, shapes, dtypes,
feature-group definitions, label mapping, and observation counts.

`docs/elliptic_v1_manifest.json` is a reference copy of the manifest used for
the published results. Comparing a locally prepared `manifest.json` with it
confirms that the input data are identical.

A dataset version must not silently change its underlying raw input.