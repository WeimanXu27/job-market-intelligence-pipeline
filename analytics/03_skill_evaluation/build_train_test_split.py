"""Create a permanent 80/20 development/test split for skill extraction.

This split is designed to prevent evaluation leakage during iterative NLP
development.

Rules:
    - Only postings with nonblank descriptions are eligible.
    - Every record previously inspected in Batch A or Batch B is forced into
      the development pool.
    - The final test pool is sampled only from never-inspected records.
    - Development and final-test IDs are strictly disjoint.
    - The final test pool is approximately 20% of all eligible postings.
    - Sampling is stratified by description length x explicit skill-signal bin.
    - Output manifests intentionally exclude job titles, companies, search
      categories, descriptions, and previews.

The final test pool must remain sealed until the extractor is frozen.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd


DEFAULT_TEST_FRACTION = 0.20
DEFAULT_RANDOM_SEED = 20260820

INVALID_DESCRIPTION_VALUES = {
    "",
    "null",
    "none",
    "nan",
    "n/a",
    "na",
    "not available",
    "not applicable",
    "-",
    "--",
    ".",
    ":",
    "unknown",
}

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

SKILL_SIGNAL_PATTERN = re.compile(
    r"\b("
    r"require(?:d|ment|ments)?|"
    r"qualif(?:y|ied|ication|ications)?|"
    r"skill(?:s)?|"
    r"experience(?:d)?|"
    r"proficien(?:t|cy)|"
    r"knowledge|"
    r"familiar(?:ity)?|"
    r"expertise|"
    r"competenc(?:y|ies)|"
    r"certif(?:ied|ication|ications)?|"
    r"licen[cs](?:e|ed|ing)|"
    r"degree|"
    r"preferred|"
    r"must|"
    r"responsibilit(?:y|ies)|"
    r"ability"
    r")\b",
    flags=re.IGNORECASE,
)



def _usable_description_mask(
    descriptions: pd.Series,
) -> pd.Series:
    """Return True only for substantive description strings.

    Historical exports may contain sentinel strings such as "NULL" rather than
    an actual empty CSV field. Those values are not usable NLP source text and
    must not enter development or final-test sampling.
    """
    normalized = (
        descriptions
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return ~normalized.isin(
        INVALID_DESCRIPTION_VALUES
    )

def _assign_quantile_bins(
    values: pd.Series,
    labels: list[str],
) -> pd.Series:
    """Assign stable quantile bins, falling back to rank-based bins."""
    try:
        return pd.qcut(
            values,
            q=len(labels),
            labels=labels,
            duplicates="drop",
        ).astype(str)
    except ValueError:
        ranked = values.rank(
            method="first",
            pct=True,
        )
        return pd.cut(
            ranked,
            bins=[
                0.0,
                1.0 / 3.0,
                2.0 / 3.0,
                1.0,
            ],
            labels=labels,
            include_lowest=True,
        ).astype(str)


def _skill_signal_score(text: str) -> int:
    """Count explicit requirement/skill markers in one description."""
    if not text:
        return 0

    return len(
        SKILL_SIGNAL_PATTERN.findall(text)
    )


def read_exposed_ids(
    paths: list[Path],
) -> set[str]:
    """Read every record ID that has already been inspected or tuned against."""
    exposed_ids: set[str] = set()

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Exposed manifest not found: {path}"
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
                f"Blank record_id found in: {path}"
            )

        exposed_ids.update(
            frame["record_id"].tolist()
        )

    return exposed_ids


def allocate_test_counts(
    unseen: pd.DataFrame,
    test_count: int,
) -> dict[tuple[str, str], int]:
    """Allocate an exact test-row count proportionally across nine strata."""
    strata_columns = [
        "description_length_bin",
        "skill_signal_bin",
    ]

    counts = (
        unseen.groupby(
            strata_columns,
            observed=True,
        )
        .size()
        .rename("available")
        .reset_index()
    )

    counts["raw_quota"] = (
        counts["available"]
        / counts["available"].sum()
        * test_count
    )

    counts["quota"] = (
        counts["raw_quota"]
        .apply(math.floor)
        .astype(int)
    )

    remainder = (
        test_count
        - int(counts["quota"].sum())
    )

    if remainder > 0:
        counts["fractional"] = (
            counts["raw_quota"]
            - counts["quota"]
        )

        add_rows = (
            counts.sort_values(
                [
                    "fractional",
                    "available",
                    "description_length_bin",
                    "skill_signal_bin",
                ],
                ascending=[
                    False,
                    False,
                    True,
                    True,
                ],
                kind="stable",
            )
            .head(remainder)
            .index
        )

        counts.loc[
            add_rows,
            "quota",
        ] += 1

    if (
        counts["quota"]
        > counts["available"]
    ).any():
        raise RuntimeError(
            "A requested test stratum is larger than "
            "the available unseen records."
        )

    return {
        (
            row["description_length_bin"],
            row["skill_signal_bin"],
        ): int(row["quota"])
        for _, row in counts.iterrows()
    }


def build_split(
    recovered_path: Path,
    analysis_path: Path,
    exposed_paths: list[Path],
    test_fraction: float,
    random_seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Build disjoint development and final-test manifests."""
    recovered = pd.read_csv(
        recovered_path,
        usecols=[
            "record_id",
            "description",
        ],
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    analysis = pd.read_csv(
        analysis_path,
        usecols=[
            "record_id",
        ],
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    if recovered.empty:
        raise ValueError(
            "Recovered dataset is empty."
        )

    if analysis.empty:
        raise ValueError(
            "Analysis dataset is empty."
        )

    if not recovered["record_id"].is_unique:
        raise ValueError(
            "Recovered record_id values must be unique."
        )

    if not analysis["record_id"].is_unique:
        raise ValueError(
            "Analysis record_id values must be unique."
        )

    data = recovered.merge(
        analysis,
        on="record_id",
        how="inner",
        validate="one_to_one",
    )

    if len(data) != len(recovered):
        raise RuntimeError(
            "Recovered and analysis datasets do not have "
            "a complete one-to-one record_id match."
        )

    data["record_id"] = (
        data["record_id"]
        .astype(str)
        .str.strip()
    )

    data["description"] = (
        data["description"]
        .astype(str)
        .str.strip()
    )

    usable_mask = _usable_description_mask(
        data["description"]
    )

    eligible = data.loc[
        usable_mask
    ].copy()

    excluded_source_rows = data.loc[
        ~usable_mask,
        [
            "record_id",
            "description",
        ],
    ].copy()

    eligible["description_chars"] = (
        eligible["description"]
        .str.len()
    )

    eligible["skill_signal_score"] = (
        eligible["description"]
        .map(_skill_signal_score)
    )

    eligible["description_length_bin"] = (
        _assign_quantile_bins(
            eligible["description_chars"],
            LENGTH_LABELS,
        )
    )

    eligible["skill_signal_bin"] = (
        _assign_quantile_bins(
            eligible["skill_signal_score"],
            SIGNAL_LABELS,
        )
    )

    exposed_ids = read_exposed_ids(
        exposed_paths
    )

    eligible_ids = set(
        eligible["record_id"].tolist()
    )

    ineligible_exposed_ids = (
        exposed_ids
        - eligible_ids
    )

    eligible_exposed_ids = (
        exposed_ids
        & eligible_ids
    )

    total_count = len(eligible)

    test_count = math.ceil(
        total_count
        * test_fraction
    )

    unseen = eligible.loc[
        ~eligible["record_id"].isin(
            eligible_exposed_ids
        )
    ].copy()

    if test_count >= len(unseen):
        raise RuntimeError(
            "Not enough never-inspected rows to create "
            "the requested final test pool."
        )

    quotas = allocate_test_counts(
        unseen=unseen,
        test_count=test_count,
    )

    sampled_test_parts = []

    stratum_index = 0

    for length_bin in LENGTH_LABELS:
        for signal_bin in SIGNAL_LABELS:
            quota = quotas.get(
                (
                    length_bin,
                    signal_bin,
                ),
                0,
            )

            if quota <= 0:
                continue

            group = unseen.loc[
                (
                    unseen[
                        "description_length_bin"
                    ]
                    == length_bin
                )
                & (
                    unseen[
                        "skill_signal_bin"
                    ]
                    == signal_bin
                )
            ]

            sampled = group.sample(
                n=quota,
                random_state=(
                    random_seed
                    + stratum_index
                ),
            )

            sampled_test_parts.append(
                sampled
            )

            stratum_index += 1

    final_test = pd.concat(
        sampled_test_parts,
        ignore_index=True,
    )

    if len(final_test) != test_count:
        raise RuntimeError(
            "Final test row-count mismatch: "
            f"expected {test_count:,}, "
            f"found {len(final_test):,}."
        )

    final_test_ids = set(
        final_test["record_id"].tolist()
    )

    if final_test_ids & eligible_exposed_ids:
        raise RuntimeError(
            "Leakage detected: previously exposed IDs "
            "entered the final test pool."
        )

    development = eligible.loc[
        ~eligible["record_id"].isin(
            final_test_ids
        )
    ].copy()

    development[
        "previously_exposed"
    ] = development[
        "record_id"
    ].isin(
        eligible_exposed_ids
    )

    development_ids = set(
        development["record_id"].tolist()
    )

    if development_ids & final_test_ids:
        raise RuntimeError(
            "Leakage detected: development and final "
            "test pools overlap."
        )

    if (
        development_ids
        | final_test_ids
    ) != eligible_ids:
        raise RuntimeError(
            "Split coverage failed: development + final "
            "test does not equal the eligible population."
        )

    if not eligible_exposed_ids.issubset(
        development_ids
    ):
        raise RuntimeError(
            "Previously exposed IDs were not fully "
            "quarantined into development."
        )

    development_manifest = (
        development[
            [
                "record_id",
                "description_length_bin",
                "skill_signal_bin",
                "previously_exposed",
            ]
        ]
        .sort_values(
            "record_id",
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    final_test_manifest = (
        final_test[
            [
                "record_id",
                "description_length_bin",
                "skill_signal_bin",
            ]
        ]
        .sort_values(
            "record_id",
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    summary_rows = []

    for split_name, frame in [
        (
            "development",
            development,
        ),
        (
            "final_test",
            final_test,
        ),
    ]:
        counts = (
            frame.groupby(
                [
                    "description_length_bin",
                    "skill_signal_bin",
                ],
                observed=True,
            )
            .size()
            .rename("rows")
            .reset_index()
        )

        counts.insert(
            0,
            "split",
            split_name,
        )

        summary_rows.append(
            counts
        )

    summary = pd.concat(
        summary_rows,
        ignore_index=True,
    )

    source_audit = {
        "total_source_rows": int(len(data)),
        "eligible_rows": int(len(eligible)),
        "excluded_source_rows": int(len(excluded_source_rows)),
        "eligible_exposed_rows": int(len(eligible_exposed_ids)),
        "ineligible_exposed_rows": int(len(ineligible_exposed_ids)),
    }

    return (
        development_manifest,
        final_test_manifest,
        summary,
        source_audit,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Create a leakage-safe 80/20 development/"
            "final-test split for skill extraction."
        )
    )

    parser.add_argument(
        "recovered_csv",
        type=Path,
    )
    parser.add_argument(
        "analysis_csv",
        type=Path,
    )
    parser.add_argument(
        "development_output_csv",
        type=Path,
    )
    parser.add_argument(
        "final_test_output_csv",
        type=Path,
    )
    parser.add_argument(
        "summary_output_csv",
        type=Path,
    )
    parser.add_argument(
        "--exposed-manifest",
        action="append",
        type=Path,
        required=True,
        help=(
            "CSV containing record_id values already "
            "seen during development. Repeat this flag "
            "for Batch A, Batch B, or any other exposed set."
        ),
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
    )

    return parser.parse_args()


def main() -> None:
    """Build, save, and validate the permanent split."""
    args = parse_args()

    if not 0.0 < args.test_fraction < 1.0:
        raise ValueError(
            "test-fraction must be between 0 and 1."
        )

    for input_path in [
        args.recovered_csv,
        args.analysis_csv,
        *args.exposed_manifest,
    ]:
        if not input_path.exists():
            raise FileNotFoundError(
                f"Input file not found: {input_path}"
            )

    (
        development,
        final_test,
        summary,
        source_audit,
    ) = build_split(
        recovered_path=args.recovered_csv,
        analysis_path=args.analysis_csv,
        exposed_paths=args.exposed_manifest,
        test_fraction=args.test_fraction,
        random_seed=args.random_seed,
    )

    for output_path in [
        args.development_output_csv,
        args.final_test_output_csv,
        args.summary_output_csv,
    ]:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    development.to_csv(
        args.development_output_csv,
        index=False,
    )

    final_test.to_csv(
        args.final_test_output_csv,
        index=False,
    )

    summary.to_csv(
        args.summary_output_csv,
        index=False,
    )

    total = (
        len(development)
        + len(final_test)
    )

    exposed_count = int(
        development[
            "previously_exposed"
        ].sum()
    )

    print(
        "NLP source eligibility audit:"
    )
    print(
        f"  Total recovered rows: "
        f"{source_audit['total_source_rows']:,}"
    )
    print(
        f"  Usable descriptions: "
        f"{source_audit['eligible_rows']:,}"
    )
    print(
        f"  Excluded blank/placeholder descriptions: "
        f"{source_audit['excluded_source_rows']:,}"
    )
    print(
        f"  Exposed but ineligible rows excluded from split: "
        f"{source_audit['ineligible_exposed_rows']:,}"
    )


    print(
        "Skill-extraction split complete."
    )
    print(
        f"Eligible nonblank descriptions: {total:,}"
    )
    print(
        f"Development rows: {len(development):,} "
        f"({len(development) / total:.2%})"
    )
    print(
        f"Final test rows: {len(final_test):,} "
        f"({len(final_test) / total:.2%})"
    )
    print(
        f"Previously exposed rows quarantined "
        f"to development: {exposed_count:,}"
    )
    print(
        "Development/final-test overlap: 0"
    )
    print(
        "Previously exposed/final-test overlap: 0"
    )
    print(
        "\nFinal test remains blind:"
    )
    print(
        "  No description, title, company, or "
        "search-category columns are exported."
    )
    print(
        f"\nDevelopment manifest: "
        f"{args.development_output_csv}"
    )
    print(
        f"Final test manifest: "
        f"{args.final_test_output_csv}"
    )
    print(
        f"Split summary: "
        f"{args.summary_output_csv}"
    )


if __name__ == "__main__":
    main()
