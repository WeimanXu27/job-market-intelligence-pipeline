"""Build a deterministic representative pilot sample for skill extraction.

The pilot is selected from recovered job postings with nonblank descriptions.
It balances:
    - short / medium / long descriptions
    - low / medium / high explicit skill-signal density

Within each stratum, the script prefers different search categories so the
pilot covers a broad range of job-market queries rather than repeatedly
sampling the same category.

This script does not call an API and does not modify MySQL.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


DEFAULT_SAMPLE_PER_STRATUM = 4
DEFAULT_RANDOM_SEED = 42

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
    """Sample rows while preferring distinct search categories."""
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


def build_pilot_sample(
    recovered_path: Path,
    analysis_path: Path,
    sample_per_stratum: int,
    random_seed: int,
) -> pd.DataFrame:
    """Create the representative skill-extraction pilot manifest."""
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
            "company",
            "job_title",
            "search_query_title",
            "job_city",
            "job_county",
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

    selected_groups = []

    for length_bin in LENGTH_LABELS:
        for signal_bin in SIGNAL_LABELS:
            group = data.loc[
                (
                    data["description_length_bin"]
                    == length_bin
                )
                & (
                    data["skill_signal_bin"]
                    == signal_bin
                )
            ].copy()

            if group.empty:
                raise RuntimeError(
                    "Pilot stratum is empty: "
                    f"{length_bin}/{signal_bin}"
                )

            sampled = _sample_diverse_categories(
                group=group,
                n=sample_per_stratum,
                random_seed=(
                    random_seed
                    + len(selected_groups)
                ),
            )

            sampled["pilot_group"] = (
                f"{length_bin}_{signal_bin}"
            )

            selected_groups.append(
                sampled
            )

    pilot = pd.concat(
        selected_groups,
        ignore_index=True,
    )

    pilot = pilot.sort_values(
        [
            "description_length_bin",
            "skill_signal_bin",
            "search_query_title",
            "record_id",
        ],
        kind="stable",
    ).reset_index(drop=True)

    pilot.insert(
        0,
        "pilot_order",
        range(
            1,
            len(pilot) + 1,
        ),
    )

    pilot["description_preview"] = (
        pilot["description"]
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
        .str.slice(
            0,
            300,
        )
    )

    output_columns = [
        "pilot_order",
        "record_id",
        "pilot_group",
        "description_length_bin",
        "skill_signal_bin",
        "description_chars",
        "description_words",
        "skill_signal_score",
        "search_query_title",
        "job_title",
        "company",
        "job_city",
        "job_county",
        "description_preview",
        "description",
    ]

    return pilot[
        output_columns
    ]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a representative pilot sample "
            "for skill extraction QA."
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
        "output_csv",
        type=Path,
        help=(
            "Path for the pilot sample CSV."
        ),
    )
    parser.add_argument(
        "--sample-per-stratum",
        type=int,
        default=DEFAULT_SAMPLE_PER_STRATUM,
        help=(
            "Rows per length/signal stratum. "
            "Default: 4 (36 rows total)."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
    )

    return parser.parse_args()


def main() -> None:
    """Build and save the representative pilot sample."""
    args = parse_args()

    if not args.recovered_csv.exists():
        raise FileNotFoundError(
            f"Recovered CSV not found: "
            f"{args.recovered_csv}"
        )

    if not args.analysis_csv.exists():
        raise FileNotFoundError(
            f"Analysis CSV not found: "
            f"{args.analysis_csv}"
        )

    if args.sample_per_stratum <= 0:
        raise ValueError(
            "sample-per-stratum must be positive."
        )

    pilot = build_pilot_sample(
        recovered_path=args.recovered_csv,
        analysis_path=args.analysis_csv,
        sample_per_stratum=args.sample_per_stratum,
        random_seed=args.random_seed,
    )

    args.output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pilot.to_csv(
        args.output_csv,
        index=False,
    )

    expected_rows = (
        len(LENGTH_LABELS)
        * len(SIGNAL_LABELS)
        * args.sample_per_stratum
    )

    if len(pilot) != expected_rows:
        raise RuntimeError(
            "Pilot row count mismatch: "
            f"expected {expected_rows:,}, "
            f"found {len(pilot):,}."
        )

    if not pilot["record_id"].is_unique:
        raise RuntimeError(
            "Pilot record_id values are not unique."
        )

    print(
        "Skill-extraction pilot sample complete."
    )
    print(
        f"Rows: {len(pilot):,}"
    )
    print(
        "Unique record IDs: "
        f"{pilot['record_id'].nunique():,}"
    )
    print(
        "Unique search categories: "
        f"{pilot['search_query_title'].nunique():,}"
    )
    print(
        "\nRows by pilot group:"
    )
    print(
        pilot["pilot_group"]
        .value_counts()
        .sort_index()
        .to_string()
    )
    print(
        "\nDescription length range: "
        f"{pilot['description_chars'].min():,}"
        " - "
        f"{pilot['description_chars'].max():,}"
        " characters"
    )
    print(
        "Skill-signal score range: "
        f"{pilot['skill_signal_score'].min():,}"
        " - "
        f"{pilot['skill_signal_score'].max():,}"
    )
    print(
        f"Output: {args.output_csv}"
    )


if __name__ == "__main__":
    main()
