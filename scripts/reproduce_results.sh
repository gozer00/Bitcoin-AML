#!/usr/bin/env bash

## Rebuilds every result table and figure from the run directories under
## outputs/runs/{local21,reference,evolvegcn}, which come either from the
## release archive or from scripts/run_benchmark.py. No model is fit.
## Run from the repository root.

set -euo pipefail

for group in local21 reference evolvegcn; do
  if [ -z "$(ls -A "outputs/runs/$group" 2>/dev/null)" ]; then
    echo "missing run directories: outputs/runs/$group" >&2
    echo "see docs/reproducing.md" >&2
    exit 1
  fi

  python scripts/build_results.py \
    --runs-root "outputs/runs/$group" \
    --output-dir "outputs/tables/$group"
done

python scripts/build_resource_usage_summary.py

## Descriptive statistics read the prepared dataset, which the archive does not
## contain. Without it, the committed tables are reused.
if [ -f data/processed/elliptic_v1/features.npy ]; then
  python scripts/build_descriptive_stats.py
else
  echo "prepared dataset not found, reusing results/tables/descriptive"
  mkdir -p outputs/tables
  cp -r results/tables/descriptive outputs/tables/
fi

python scripts/build_post_hoc_analysis.py
python scripts/build_paired_comparisons.py
python scripts/build_figures.py
