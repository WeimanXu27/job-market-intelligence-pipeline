"""Build a fresh blind development-validation sample.

This sample is drawn only from the permanent development pool created by
build_train_test_split.py. It never touches the final 20% test pool.

Design:
    - 72 total records
    - 36 stratified stress rows:
        3 description-length bins x 3 skill-signal bins x 4 rows
    - 36 pure-random rows from the remaining unseen development population
    - all previously exposed development rows are excluded
    - no semantic posting columns are exported

Any sample produced by this script is development data. Once inspected or used
for tuning, it must remain permanently excluded from the final test pool.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_STRESS_PER_STRATUM = 4
DEFAULT_RANDOM_ROWS = 36
DEFAULT_RANDOM_SEED = 20260821

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


def _parse_bool(value: object) -> bool:
    """Parse a CSV boolean value conservatively."""
    normalized = str(value).strip().lower()

    if normalized in {
        "true",
        "1",
        "yes",
    }:
        return True

    if normalized in {
        "false",
        "0",
        "no",
        "",
    }:
        return False

    raise ValueError(
        "Unexpected previously_exposed value: "
        f"{value!r}"
    )


def _read_excluded_ids(
    paths: list[Path],
) -> set[str]:
    """Read extra development IDs that have already been sampled or inspected."""
    excluded_ids: set[str] = set()

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Exclude manifest not found: {path}"
            )

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
                f"Blank record_id found in exclude manifest: {path}"
            )

        excluded_ids.update(
            frame["record_id"].tolist()
        )

    return excluded_ids


def build_validation_sample(
    development_path: Path,
    exclude_paths: list[Path],
    stress_per_stratum: int,
    random_rows: int,
    random_seed: int,
) -> pd.DataFrame:
    """Build a fresh blind sample from never-exposed development rows."""
    development = pd.read_csv(
        development_path,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    required_columns = {
        "record_id",
        "description_length_bin",
        "skill_signal_bin",
        "previously_exposed",
    }

    missing_columns = (
        required_columns
        - set(development.columns)
    )

    if missing_columns:
        raise ValueError(
            "Development manifest is missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    development["record_id"] = (
        development["record_id"]
        .astype(str)
        .str.strip()
    )

    if development["record_id"].eq("").any():
        raise ValueError(
            "Development manifest contains blank record_id values."
        )

    if not development["record_id"].is_unique:
        raise ValueError(
            "Development record_id values must be unique."
        )

    development["previously_exposed"] = (
        development["previously_exposed"]
        .map(_parse_bool)
    )

    extra_excluded_ids = _read_excluded_ids(
        exclude_paths
    )

    development_ids = set(
        development["record_id"].tolist()
    )

    unknown_excluded = sorted(
        extra_excluded_ids
        - development_ids
    )

    if unknown_excluded:
        example = ", ".join(
            unknown_excluded[:5]
        )
        raise RuntimeError(
            "An exclude manifest contains IDs outside "
            "the development pool. "
            f"Example IDs: {example}"
        )

    unseen = development.loc[
        (
            ~development["previously_exposed"]
        )
        & (
            ~development["record_id"].isin(
                extra_excluded_ids
            )
        )
    ].copy()

    if unseen.empty:
        raise RuntimeError(
            "No never-exposed development rows are available."
        )

    stress_parts = []

    stratum_index = 0

    for length_bin in LENGTH_LABELS:
        for signal_bin in SIGNAL_LABELS:
            group = unseen.loc[
                (
                    unseen["description_length_bin"]
                    == length_bin
                )
                & (
                    unseen["skill_signal_bin"]
                    == signal_bin
                )
            ].copy()

            if len(group) < stress_per_stratum:
                raise RuntimeError(
                    "Not enough rows in development stratum "
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

            sampled["validation_component"] = "stress"
            sampled["validation_group"] = (
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

    random_candidates = unseen.loc[
        ~unseen["record_id"].isin(
            stress_ids
        )
    ].copy()

    if len(random_candidates) < random_rows:
        raise RuntimeError(
            "Not enough remaining unseen development rows "
            "for the pure-random component."
        )

    random_sample = random_candidates.sample(
        n=random_rows,
        random_state=random_seed + 1000,
    ).copy()

    random_sample["validation_component"] = "random"
    random_sample["validation_group"] = "random"

    validation = pd.concat(
        [
            stress,
            random_sample,
        ],
        ignore_index=True,
    )

    if not validation["record_id"].is_unique:
        raise RuntimeError(
            "Validation sample record IDs are not unique."
        )

    if validation["previously_exposed"].any():
        raise RuntimeError(
            "Leakage detected: previously exposed development "
            "rows entered the validation sample."
        )

    validation.insert(
        0,
        "pilot_order",
        range(
            1,
            len(validation) + 1,
        ),
    )

    output_columns = [
        "pilot_order",
        "record_id",
        "validation_component",
        "validation_group",
        "description_length_bin",
        "skill_signal_bin",
    ]

    return validation[
        output_columns
    ]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a fresh blind development-validation sample "
            "without touching the permanent final test pool."
        )
    )

    parser.add_argument(
        "development_pool_csv",
        type=Path,
        help=(
            "Path to skill_development_pool.csv."
        ),
    )
    parser.add_argument(
        "output_csv",
        type=Path,
        help=(
            "Path for the blind development-validation manifest."
        ),
    )
    parser.add_argument(
        "--exclude-manifest",
        action="append",
        type=Path,
        default=[],
        help=(
            "Optional CSV of additional development record IDs "
            "already sampled or inspected. Repeat as needed."
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

    return parser.parse_args()


def main() -> None:
    """Build, save, and validate the fresh development-validation sample."""
    args = parse_args()

    if not args.development_pool_csv.exists():
        raise FileNotFoundError(
            f"Development pool not found: "
            f"{args.development_pool_csv}"
        )

    for exclude_path in args.exclude_manifest:
        if not exclude_path.exists():
            raise FileNotFoundError(
                f"Exclude manifest not found: "
                f"{exclude_path}"
            )

    if args.stress_per_stratum <= 0:
        raise ValueError(
            "stress-per-stratum must be positive."
        )

    if args.random_rows <= 0:
        raise ValueError(
            "random-rows must be positive."
        )

    validation = build_validation_sample(
        development_path=args.development_pool_csv,
        exclude_paths=args.exclude_manifest,
        stress_per_stratum=args.stress_per_stratum,
        random_rows=args.random_rows,
        random_seed=args.random_seed,
    )

    args.output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation.to_csv(
        args.output_csv,
        index=False,
    )

    expected_stress = (
        len(LENGTH_LABELS)
        * len(SIGNAL_LABELS)
        * args.stress_per_stratum
    )

    expected_total = (
        expected_stress
        + args.random_rows
    )

    if len(validation) != expected_total:
        raise RuntimeError(
            "Validation row-count mismatch: "
            f"expected {expected_total:,}, "
            f"found {len(validation):,}."
        )

    print(
        "Blind development-validation sample complete."
    )
    print(
        f"Rows: {len(validation):,}"
    )
    print(
        "Unique record IDs: "
        f"{validation['record_id'].nunique():,}"
    )
    print(
        "Stress rows: "
        f"{(validation['validation_component'] == 'stress').sum():,}"
    )
    print(
        "Pure-random rows: "
        f"{(validation['validation_component'] == 'random').sum():,}"
    )

    print(
        "\nStress rows by length/signal stratum:"
    )
    print(
        validation.loc[
            validation["validation_component"]
            == "stress"
        ]
        .groupby(
            [
                "description_length_bin",
                "skill_signal_bin",
            ],
            observed=True,
        )
        .size()
        .to_string()
    )

    print(
        "\nLeakage checks:"
    )
    print(
        "  Previously exposed rows included: 0"
    )
    print(
        "  Final-test rows included: 0 "
        "(sample source is development pool only)"
    )
    print(
        "\nBlindness check: no description, title, "
        "company, or search-category columns are exported."
    )
    print(
        f"Output: {args.output_csv}"
    )


if __name__ == "__main__":
    main()
