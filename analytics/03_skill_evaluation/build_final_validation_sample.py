"""Build a one-time sealed final-test QA sample.

This script must be used only after the skill extractor is frozen.

Design:
- Draw only from skill_final_test_pool.csv.
- Verify the final-test pool has zero overlap with the development pool.
- Draw 144 rows by default:
    * 72 stress rows: 3 description-length bins x 3 skill-signal bins x 8 rows
    * 72 pure-random rows from the remaining final-test population
- Export only record IDs and stratification metadata; never export descriptions.

If this final QA sample is inspected and the extractor is changed afterward,
all IDs in the sample become exposed. A later final QA must exclude them by
passing the old sample with --exclude-manifest.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LENGTH_LABELS = [
    "short",
    "medium",
    "long",
]

SIGNAL_LABELS = [
    "low",
    "medium",
    "high",
]

DEFAULT_STRESS_PER_STRATUM = 8
DEFAULT_RANDOM_ROWS = 72
DEFAULT_RANDOM_SEED = 20260825
DEFAULT_EXPECTED_FINAL_POOL_SIZE = 12857


def read_ids(
    path: Path,
) -> set[str]:
    """Read unique record IDs from a manifest."""
    frame = pd.read_csv(
        path,
        usecols=["record_id"],
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    frame["record_id"] = (
        frame["record_id"]
        .astype(str)
        .str.strip()
    )

    if frame["record_id"].eq("").any():
        raise ValueError(
            f"Blank record_id found in {path}"
        )

    if not frame["record_id"].is_unique:
        raise ValueError(
            f"Duplicate record_id values found in {path}"
        )

    return set(
        frame["record_id"].tolist()
    )


def build_final_validation_sample(
    final_test_path: Path,
    development_path: Path,
    exclude_paths: list[Path],
    stress_per_stratum: int,
    random_rows: int,
    random_seed: int,
    expected_final_pool_size: int,
) -> pd.DataFrame:
    """Build a sealed QA sample from the untouched final-test pool."""
    final_test = pd.read_csv(
        final_test_path,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    required_columns = {
        "record_id",
        "description_length_bin",
        "skill_signal_bin",
    }

    missing_columns = (
        required_columns
        - set(final_test.columns)
    )

    if missing_columns:
        raise ValueError(
            "Final-test manifest is missing required columns: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    final_test["record_id"] = (
        final_test["record_id"]
        .astype(str)
        .str.strip()
    )

    if final_test["record_id"].eq("").any():
        raise ValueError(
            "Final-test manifest contains blank record_id values."
        )

    if not final_test["record_id"].is_unique:
        raise ValueError(
            "Final-test record_id values must be unique."
        )

    if (
        expected_final_pool_size > 0
        and len(final_test)
        != expected_final_pool_size
    ):
        raise RuntimeError(
            "Unexpected final-test pool size: "
            f"expected {expected_final_pool_size:,}, "
            f"found {len(final_test):,}."
        )

    final_ids = set(
        final_test["record_id"].tolist()
    )

    development_ids = read_ids(
        development_path
    )

    overlap = (
        final_ids
        & development_ids
    )

    if overlap:
        example = ", ".join(
            sorted(
                overlap
            )[:5]
        )

        raise RuntimeError(
            "Leakage detected: final-test and development "
            "pools overlap. "
            f"Example IDs: {example}"
        )

    excluded_ids: set[str] = set()

    for path in exclude_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Exclude manifest not found: {path}"
            )

        ids = read_ids(
            path
        )

        outside_final = (
            ids
            - final_ids
        )

        if outside_final:
            example = ", ".join(
                sorted(
                    outside_final
                )[:5]
            )

            raise RuntimeError(
                "A final-QA exclude manifest contains IDs "
                "outside the final-test pool. "
                f"Example IDs: {example}"
            )

        excluded_ids.update(
            ids
        )

    available = final_test.loc[
        ~final_test["record_id"].isin(
            excluded_ids
        )
    ].copy()

    stress_parts = []

    stratum_index = 0

    for length_bin in LENGTH_LABELS:
        for signal_bin in SIGNAL_LABELS:
            group = available.loc[
                (
                    available[
                        "description_length_bin"
                    ]
                    == length_bin
                )
                & (
                    available[
                        "skill_signal_bin"
                    ]
                    == signal_bin
                )
            ].copy()

            if len(group) < stress_per_stratum:
                raise RuntimeError(
                    "Not enough untouched final-test rows "
                    "in stratum "
                    f"{length_bin}/{signal_bin}: "
                    f"need {stress_per_stratum:,}, "
                    f"found {len(group):,}."
                )

            sampled = group.sample(
                n=stress_per_stratum,
                random_state=(
                    random_seed
                    + stratum_index
                ),
            ).copy()

            sampled[
                "validation_component"
            ] = "stress"

            sampled[
                "validation_group"
            ] = (
                f"stress_{length_bin}_{signal_bin}"
            )

            stress_parts.append(
                sampled
            )

            stratum_index += 1

    stress = pd.concat(
        stress_parts,
        ignore_index=True,
    )

    stress_ids = set(
        stress["record_id"].tolist()
    )

    random_candidates = available.loc[
        ~available["record_id"].isin(
            stress_ids
        )
    ].copy()

    if len(random_candidates) < random_rows:
        raise RuntimeError(
            "Not enough untouched final-test rows "
            "for the pure-random component."
        )

    random_sample = random_candidates.sample(
        n=random_rows,
        random_state=(
            random_seed
            + 1000
        ),
    ).copy()

    random_sample[
        "validation_component"
    ] = "random"

    random_sample[
        "validation_group"
    ] = "random"

    validation = pd.concat(
        [
            stress,
            random_sample,
        ],
        ignore_index=True,
    )

    if not validation["record_id"].is_unique:
        raise RuntimeError(
            "Final validation sample IDs are not unique."
        )

    if set(
        validation["record_id"]
    ) & development_ids:
        raise RuntimeError(
            "Leakage detected: development IDs entered "
            "the final validation sample."
        )

    if set(
        validation["record_id"]
    ) & excluded_ids:
        raise RuntimeError(
            "Leakage detected: previously evaluated final "
            "IDs entered the new sample."
        )

    validation.insert(
        0,
        "pilot_order",
        range(
            1,
            len(validation) + 1,
        ),
    )

    return validation[
        [
            "pilot_order",
            "record_id",
            "validation_component",
            "validation_group",
            "description_length_bin",
            "skill_signal_bin",
        ]
    ]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a leakage-safe sealed final-test QA sample."
        )
    )

    parser.add_argument(
        "final_test_pool_csv",
        type=Path,
    )
    parser.add_argument(
        "development_pool_csv",
        type=Path,
    )
    parser.add_argument(
        "output_csv",
        type=Path,
    )
    parser.add_argument(
        "--exclude-manifest",
        action="append",
        type=Path,
        default=[],
        help=(
            "Previously evaluated final-QA manifest. "
            "Repeat this flag if needed."
        ),
    )
    parser.add_argument(
        "--stress-per-stratum",
        type=int,
        default=DEFAULT_STRESS_PER_STRATUM,
    )
    parser.add_argument(
        "--random-rows",
        type=int,
        default=DEFAULT_RANDOM_ROWS,
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
    )
    parser.add_argument(
        "--expected-final-pool-size",
        type=int,
        default=DEFAULT_EXPECTED_FINAL_POOL_SIZE,
        help=(
            "Safety check for the current permanent final-test pool. "
            "Set to 0 only if intentionally disabling the size check."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Build, save, and validate the sealed final-test QA sample."""
    args = parse_args()

    for path in [
        args.final_test_pool_csv,
        args.development_pool_csv,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    if args.stress_per_stratum <= 0:
        raise ValueError(
            "stress-per-stratum must be positive."
        )

    if args.random_rows <= 0:
        raise ValueError(
            "random-rows must be positive."
        )

    validation = build_final_validation_sample(
        final_test_path=(
            args.final_test_pool_csv
        ),
        development_path=(
            args.development_pool_csv
        ),
        exclude_paths=(
            args.exclude_manifest
        ),
        stress_per_stratum=(
            args.stress_per_stratum
        ),
        random_rows=(
            args.random_rows
        ),
        random_seed=(
            args.random_seed
        ),
        expected_final_pool_size=(
            args.expected_final_pool_size
        ),
    )

    args.output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation.to_csv(
        args.output_csv,
        index=False,
    )

    stress_rows = int(
        (
            validation[
                "validation_component"
            ]
            == "stress"
        ).sum()
    )

    random_rows = int(
        (
            validation[
                "validation_component"
            ]
            == "random"
        ).sum()
    )

    print(
        "Sealed final-test QA sample complete."
    )
    print(
        f"Rows: {len(validation):,}"
    )
    print(
        "Unique record IDs: "
        f"{validation['record_id'].nunique():,}"
    )
    print(
        f"Stress rows: {stress_rows:,}"
    )
    print(
        f"Pure-random rows: {random_rows:,}"
    )
    print(
        "Development overlap: 0"
    )
    print(
        "Previously evaluated final IDs included: 0"
    )
    print(
        f"Output: {args.output_csv}"
    )


if __name__ == "__main__":
    main()
