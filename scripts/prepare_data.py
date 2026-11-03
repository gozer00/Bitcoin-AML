from __future__ import annotations

import argparse
import logging
from pathlib import Path

from bitcoin_aml.data.prepare import prepare_elliptic_dataset

# -------------------------------------------------------
# main executable script for data preparation steps
#
# Arguments for calling the script:
#   * --raw-dir: directory with raw Elliptic data files
#   * --output-dir: directory the program writes processed files to
#   * --dataset-version: determines the version of the prepared dataset
#           -> only elliptic_v1 supported yet
#   * --force: forces an overwrite of already existing prepared data
#           -> otherwise the program checks for existence and omits recomputation
# -------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare canonical Elliptic dataset."
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw")
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/elliptic_v1")
    )

    parser.add_argument(
        "--dataset-version",
        default="elliptic_v1",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing processed directory"
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    args = parse_args()

    prepare_elliptic_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        dataset_version=args.dataset_version,
        force=args.force,
    )


if __name__ == "__main__":
    main()
