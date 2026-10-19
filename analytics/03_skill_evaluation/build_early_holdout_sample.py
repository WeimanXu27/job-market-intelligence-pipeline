"""Build a blind holdout sample for skill-extraction generalization testing.

Batch B contains:
    - 18 stratified stress-test rows:
      2 rows from each description-length x skill-signal stratum
    - 18 pure-random rows from the remaining eligible postings

The script excludes every record_id listed in a prior pilot manifest and writes
only non-semantic sampling metadata. It intentionally does NOT export job titles,
companies, search categories, descriptions, or previews so the holdout can be
run before anyone inspects its content.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


DEFAULT_STRESS_PER_STRATUM = 2
DEFAULT_RANDOM_COUNT = 18
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
        cut_points = [
            0.0,
            1.0 / 3.0,
            2.0 / 3.0,
            1.0,
        ]
        return pd.cut(
            ranked,
            bins=cut_points,
            labels=labels,
            include_lowest=True,
        ).astype(str)


def _skill_signal_score(text: str) -> int:
    """Count explicit requirement/skill markers in a job description."""
    if not text:
        return 0

    return len(
        SKILL_SIGNAL_PATTERN.findall(text)
    )


def _sample_diverse_categories(
    group: pd.DataFrame,
    n: int,
    random_seed: int,
) -> pd.DataFrame:
    """Sample stress rows while preferring distinct search categories."""
    shuffled = group.sample(
        frac=1.0,
        random_state=random_seed,
    )

    diverse = shuffled.drop_duplicates(
        subset=["search_query_title"],
        keep="first",
    ).head(n)

    if len(diverse) >= n:
        return diverse

    remaining = shuffled.loc[
        ~shuffled["record_id"].isin(
            diverse["record_id"]
        )
    ].head(
        n - len(diverse)
    )

    return pd.concat(
        [
            diverse,
            remaining,
        ],
        ignore_index=True,
    )


def read_excluded_ids(
    exclude_path: Path,
) -> set[str]:
    """Read record IDs that must never enter the new holdout."""
    excluded = pd.read_csv(
        exclude_path,
        usecols=["record_id"],
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    excluded["record_id"] = (
        excluded["record_id"]
        .astype(str)
        .str.strip()
    )

    if excluded["record_id"].eq("").any():
        raise ValueError(
            "Exclude CSV contains blank record_id values."
        )

    if not excluded["record_id"].is_unique:
        raise ValueError(
            "Exclude CSV record_id values must be unique."
        )

    return set(
        excluded["record_id"].tolist()
    )


def build_blind_holdout(
    recovered_path: Path,
    analysis_path: Path,
    exclude_path: Path,
    stress_per_stratum: int,
    random_count: int,
    random_seed: int,
) -> pd.DataFrame:
    """Create Batch B without exposing semantic posting content."""
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
            "search_query_title",
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

    data = data.loc[
        data["description"].ne("")
    ].copy()

    data["description_chars"] = (
        data["description"]
        .str.len()
    )

    data["description_words"] = (
        data["description"]
        .str.split()
        .str.len()
    )

    data["skill_signal_score"] = (
        data["description"]
        .map(_skill_signal_score)
    )

    # Compute strata on the full nonblank source so Batch B uses the same
    # population-based binning logic as the original pilot.
    data["description_length_bin"] = (
        _assign_quantile_bins(
            data["description_chars"],
            LENGTH_LABELS,
        )
    )

    data["skill_signal_bin"] = (
        _assign_quantile_bins(
            data["skill_signal_score"],
            SIGNAL_LABELS,
        )
    )

    excluded_ids = read_excluded_ids(
        exclude_path
    )

    missing_excluded_ids = sorted(
        excluded_ids
        - set(data["record_id"])
    )

    if missing_excluded_ids:
        example = ", ".join(
            missing_excluded_ids[:5]
        )
        raise RuntimeError(
            "Some excluded record IDs are not present among "
            "nonblank source descriptions. "
            f"Example IDs: {example}"
        )

    eligible = data.loc[
        ~data["record_id"].isin(
            excluded_ids
        )
    ].copy()

    stress_groups = []

    for group_index, length_bin in enumerate(
        LENGTH_LABELS
    ):
        for signal_index, signal_bin in enumerate(
            SIGNAL_LABELS
        ):
            group = eligible.loc[
                (
                    eligible["description_length_bin"]
                    == length_bin
                )
                & (
                    eligible["skill_signal_bin"]
                    == signal_bin
                )
            ].copy()

            if len(group) < stress_per_stratum:
                raise RuntimeError(
                    "Not enough rows in holdout stratum: "
                    f"{length_bin}/{signal_bin}. "
                    f"Found {len(group):,}, "
                    f"need {stress_per_stratum:,}."
                )

            sampled = _sample_diverse_categories(
                group=group,
                n=stress_per_stratum,
                random_seed=(
                    random_seed
                    + group_index * 10
                    + signal_index
                ),
            )

            sampled["holdout_component"] = (
                "stress"
            )
            sampled["pilot_group"] = (
                f"stress_{length_bin}_{signal_bin}"
            )

            stress_groups.append(
                sampled
            )

    stress = pd.concat(
        stress_groups,
        ignore_index=True,
    )

    remaining = eligible.loc[
        ~eligible["record_id"].isin(
            stress["record_id"]
        )
    ].copy()

    if len(remaining) < random_count:
        raise RuntimeError(
            "Not enough remaining rows for the pure-random holdout component."
        )

    random_sample = remaining.sample(
        n=random_count,
        random_state=(
            random_seed + 1000
        ),
        replace=False,
    ).copy()

    random_sample["holdout_component"] = (
        "random"
    )
    random_sample["pilot_group"] = (
        "random"
    )

    holdout = pd.concat(
        [
            stress,
            random_sample,
        ],
        ignore_index=True,
    )

    holdout = holdout.sample(
        frac=1.0,
        random_state=(
            random_seed + 2000
        ),
    ).reset_index(
        drop=True
    )

    holdout.insert(
        0,
        "pilot_order",
        range(
            1,
            len(holdout) + 1,
        ),
    )

    output_columns = [
        "pilot_order",
        "record_id",
        "holdout_component",
        "pilot_group",
        "description_length_bin",
        "skill_signal_bin",
        "description_chars",
        "description_words",
        "skill_signal_score",
    ]

    result = holdout[
        output_columns
    ].copy()

    forbidden_columns = {
        "description",
        "description_preview",
        "job_title",
        "company",
        "search_query_title",
        "job_city",
        "job_county",
    }

    leaked_columns = (
        forbidden_columns
        & set(result.columns)
    )

    if leaked_columns:
        raise RuntimeError(
            "Blind holdout contains semantic columns: "
            + ", ".join(
                sorted(leaked_columns)
            )
        )

    return result


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a 36-row blind holdout for "
            "skill-extraction generalization testing."
        )
    )

    parser.add_argument(
        "recovered_csv",
        type=Path,
        help=(
            "Path to recovered_job_postings.csv."
        ),
    )
    parser.add_argument(
        "analysis_csv",
        type=Path,
        help=(
            "Path to analysis_job_postings.csv."
        ),
    )
    parser.add_argument(
        "exclude_csv",
        type=Path,
        help=(
            "Prior pilot manifest whose record IDs "
            "must be excluded."
        ),
    )
    parser.add_argument(
        "output_csv",
        type=Path,
        help=(
            "Path for the blind holdout manifest."
        ),
    )
    parser.add_argument(
        "--stress-per-stratum",
        type=int,
        default=DEFAULT_STRESS_PER_STRATUM,
        help=(
            "Stress rows per length/signal stratum. "
            "Default: 2 (18 stress rows total)."
        ),
    )
    parser.add_argument(
        "--random-count",
        type=int,
        default=DEFAULT_RANDOM_COUNT,
        help=(
            "Pure-random rows from the remaining pool. "
            "Default: 18."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
    )

    return parser.parse_args()


def main() -> None:
    """Build and save the blind holdout manifest."""
    args = parse_args()

    for path in (
        args.recovered_csv,
        args.analysis_csv,
        args.exclude_csv,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Input file not found: {path}"
            )

    if args.stress_per_stratum <= 0:
        raise ValueError(
            "stress-per-stratum must be positive."
        )

    if args.random_count <= 0:
        raise ValueError(
            "random-count must be positive."
        )

    holdout = build_blind_holdout(
        recovered_path=args.recovered_csv,
        analysis_path=args.analysis_csv,
        exclude_path=args.exclude_csv,
        stress_per_stratum=args.stress_per_stratum,
        random_count=args.random_count,
        random_seed=args.random_seed,
    )

    args.output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    holdout.to_csv(
        args.output_csv,
        index=False,
    )

    expected_stress_rows = (
        len(LENGTH_LABELS)
        * len(SIGNAL_LABELS)
        * args.stress_per_stratum
    )
    expected_rows = (
        expected_stress_rows
        + args.random_count
    )

    if len(holdout) != expected_rows:
        raise RuntimeError(
            "Holdout row count mismatch: "
            f"expected {expected_rows:,}, "
            f"found {len(holdout):,}."
        )

    if not holdout["record_id"].is_unique:
        raise RuntimeError(
            "Holdout record_id values are not unique."
        )

    excluded_ids = read_excluded_ids(
        args.exclude_csv
    )

    overlap = set(
        holdout["record_id"]
    ).intersection(
        excluded_ids
    )

    if overlap:
        raise RuntimeError(
            "Holdout overlaps the excluded development pilot."
        )

    stress = holdout.loc[
        holdout["holdout_component"]
        == "stress"
    ]
    random_sample = holdout.loc[
        holdout["holdout_component"]
        == "random"
    ]

    stress_counts = (
        stress.groupby(
            [
                "description_length_bin",
                "skill_signal_bin",
            ],
            observed=False,
        )
        .size()
    )

    if not (
        stress_counts
        == args.stress_per_stratum
    ).all():
        raise RuntimeError(
            "Stress-component stratum counts are not balanced."
        )

    print(
        "Blind holdout sample complete."
    )
    print(
        f"Rows: {len(holdout):,}"
    )
    print(
        "Unique record IDs: "
        f"{holdout['record_id'].nunique():,}"
    )
    print(
        "Overlap with development pilot: 0"
    )
    print(
        f"Stress rows: {len(stress):,}"
    )
    print(
        f"Pure-random rows: {len(random_sample):,}"
    )
    print(
        "\nStress rows by length/signal stratum:"
    )
    print(
        stress_counts.to_string()
    )
    print(
        "\nPure-random component distribution:"
    )
    print(
        random_sample.groupby(
            [
                "description_length_bin",
                "skill_signal_bin",
            ],
            observed=False,
        )
        .size()
        .to_string()
    )
    print(
        "\nBlindness check: semantic posting columns are not exported."
    )
    print(
        f"Output: {args.output_csv}"
    )


if __name__ == "__main__":
    main()
