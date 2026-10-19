"""Build an analytics-ready job-posting dataset from recovered scrape data.

This step sits downstream of:
    raw historical CSV
        -> recover_scraped_jobs.py
        -> recovered_job_postings.csv
        -> audit_recovered_jobs.py
        -> this script

The analysis dataset keeps all recovered job postings, gives historically
ambiguous fields clearer analytical names, and adds conservative salary
quality flags. It does not classify job families, run NLP skill extraction,
or treat the search-query O*NET code as a verified classification of each
returned job posting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


INPUT_COLUMNS = [
    "record_id",
    "date",
    "company",
    "title",
    "search_title",
    "location",
    "search_location",
    "salary_job_type",
    "hourly_salary",
    "url",
    "onet_code",
    "city",
    "county",
    "min_salary",
    "max_salary",
    "salary_unit",
]

RENAME_MAP = {
    "date": "posting_date",
    "title": "job_title",
    "search_title": "search_query_title",
    "location": "location_raw",
    "search_location": "search_market",
    "onet_code": "search_onet_code",
    "city": "job_city",
    "county": "job_county",
    "salary_job_type": "salary_job_type_raw",
    "hourly_salary": "hourly_salary_raw",
    "min_salary": "min_salary_raw",
    "max_salary": "max_salary_raw",
}

STRING_COLUMNS = [
    "record_id",
    "company",
    "job_title",
    "search_query_title",
    "location_raw",
    "search_market",
    "url",
    "search_onet_code",
    "job_city",
    "job_county",
    "salary_unit",
]

NUMERIC_SALARY_COLUMNS = [
    "salary_job_type_raw",
    "hourly_salary_raw",
    "min_salary_raw",
    "max_salary_raw",
]

# These bounds are screening rules for the portfolio analysis layer, not claims
# that every value outside the range is impossible. Raw salary values are
# retained for traceability.
HOURLY_SALARY_LOWER_BOUND = 10.0
HOURLY_SALARY_UPPER_BOUND = 300.0
ANNUAL_HOURS = 2080.0

STRUCTURED_TO_HOURLY_FACTOR = {
    "hr": 1.0,
    "wk": 1.0 / 40.0,
    "mo": 12.0 / ANNUAL_HOURS,
    "yr": 1.0 / ANNUAL_HOURS,
}


def _clean_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim string columns without changing their substantive content."""
    for column in STRING_COLUMNS:
        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.replace("\u200b", "", regex=False)
            .str.strip()
        )

    df["salary_unit"] = df["salary_unit"].str.lower()

    return df


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Create validated posting-date fields."""
    parsed = pd.to_datetime(
        df["posting_date"],
        format="%Y-%m-%d",
        errors="coerce",
    )

    if parsed.isna().any():
        bad_count = int(parsed.isna().sum())
        raise ValueError(
            f"Found {bad_count} invalid posting_date values."
        )

    df["posting_date"] = parsed.dt.strftime("%Y-%m-%d")
    df["posting_month"] = parsed.dt.strftime("%Y-%m")

    return df


def _parse_salary_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """Convert historical salary fields to numeric analysis columns."""
    for column in NUMERIC_SALARY_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return df


def _add_structured_salary_checks(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Derive a structured salary midpoint and hourly equivalent.

    These fields are used as quality checks. They do not replace the historical
    hourly_salary field because the audit showed schema evolution and occasional
    unit inconsistencies in the structured salary fields.
    """
    min_salary = df["min_salary_raw"]
    max_salary = df["max_salary_raw"]

    midpoint = pd.Series(
        np.nan,
        index=df.index,
        dtype="float64",
    )

    both_bounds = min_salary.notna() & max_salary.notna()
    midpoint.loc[both_bounds] = (
        min_salary.loc[both_bounds]
        + max_salary.loc[both_bounds]
    ) / 2.0

    min_only = min_salary.notna() & max_salary.isna()
    midpoint.loc[min_only] = min_salary.loc[min_only]

    max_only = min_salary.isna() & max_salary.notna()
    midpoint.loc[max_only] = max_salary.loc[max_only]

    df["structured_salary_midpoint_raw"] = midpoint

    unit_factor = df["salary_unit"].map(
        STRUCTURED_TO_HOURLY_FACTOR
    )

    df["structured_hourly_equivalent"] = (
        df["structured_salary_midpoint_raw"]
        * unit_factor
    )

    comparison_mask = (
        df["hourly_salary_raw"].notna()
        & (df["hourly_salary_raw"] > 0)
        & df["structured_hourly_equivalent"].notna()
        & (df["structured_hourly_equivalent"] > 0)
    )

    df["structured_salary_consistency"] = "not_available"

    if comparison_mask.any():
        matches = np.isclose(
            df.loc[
                comparison_mask,
                "hourly_salary_raw",
            ],
            df.loc[
                comparison_mask,
                "structured_hourly_equivalent",
            ],
            rtol=0.03,
            atol=0.5,
        )

        df.loc[
            comparison_mask,
            "structured_salary_consistency",
        ] = np.where(
            matches,
            "match",
            "mismatch",
        )

    return df


def _add_salary_quality_fields(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add conservative salary screening fields for analysis."""
    salary = df["hourly_salary_raw"]

    conditions = [
        salary.isna(),
        salary.notna() & (salary <= 0),
        salary.notna()
        & (salary > 0)
        & (salary < HOURLY_SALARY_LOWER_BOUND),
        salary.notna()
        & (salary >= HOURLY_SALARY_LOWER_BOUND)
        & (salary <= HOURLY_SALARY_UPPER_BOUND),
        salary.notna()
        & (salary > HOURLY_SALARY_UPPER_BOUND),
    ]

    labels = [
        "missing",
        "non_positive",
        "below_screening_range",
        "within_screening_range",
        "above_screening_range",
    ]

    df["salary_quality_flag"] = np.select(
        conditions,
        labels,
        default="unclassified",
    )

    df["salary_usable_for_analysis"] = (
        df["salary_quality_flag"]
        == "within_screening_range"
    )

    df["hourly_salary_clean"] = salary.where(
        df["salary_usable_for_analysis"]
    )

    df["annual_salary_equivalent"] = (
        df["hourly_salary_clean"]
        * ANNUAL_HOURS
    )

    return df


def _add_duplicate_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Flag potential duplicate business records without deleting them."""
    duplicate_key = [
        "posting_date",
        "company",
        "job_title",
        "search_query_title",
        "location_raw",
        "url",
    ]

    df["is_potential_duplicate"] = df.duplicated(
        subset=duplicate_key,
        keep=False,
    )

    return df


def build_analysis_dataset(
    input_path: Path,
) -> pd.DataFrame:
    """Create the first analytics-ready job-posting table."""
    df = pd.read_csv(
        input_path,
        usecols=INPUT_COLUMNS,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    if df.empty:
        raise ValueError("Recovered dataset is empty.")

    if not df["record_id"].is_unique:
        raise ValueError(
            "record_id must be unique before building the analysis dataset."
        )

    df = df.rename(columns=RENAME_MAP)
    df = _clean_string_columns(df)
    df = _parse_dates(df)
    df = _parse_salary_numbers(df)
    df = _add_structured_salary_checks(df)
    df = _add_salary_quality_fields(df)
    df = _add_duplicate_flag(df)

    output_columns = [
        "record_id",
        "posting_date",
        "posting_month",
        "company",
        "job_title",
        "search_query_title",
        "search_onet_code",
        "location_raw",
        "search_market",
        "job_city",
        "job_county",
        "salary_job_type_raw",
        "hourly_salary_raw",
        "min_salary_raw",
        "max_salary_raw",
        "salary_unit",
        "structured_salary_midpoint_raw",
        "structured_hourly_equivalent",
        "structured_salary_consistency",
        "salary_quality_flag",
        "salary_usable_for_analysis",
        "hourly_salary_clean",
        "annual_salary_equivalent",
        "is_potential_duplicate",
        "url",
    ]

    return df[output_columns]


def build_report(df: pd.DataFrame) -> Dict:
    """Create a compact validation report for the analysis dataset."""
    salary_counts = (
        df["salary_quality_flag"]
        .value_counts(dropna=False)
        .to_dict()
    )

    consistency_counts = (
        df["structured_salary_consistency"]
        .value_counts(dropna=False)
        .to_dict()
    )

    usable_salary = df["hourly_salary_clean"].dropna()

    return {
        "rows": int(len(df)),
        "record_id_unique": bool(df["record_id"].is_unique),
        "columns": list(df.columns),
        "salary_quality_counts": {
            str(key): int(value)
            for key, value in salary_counts.items()
        },
        "salary_usable_rows": int(
            df["salary_usable_for_analysis"].sum()
        ),
        "salary_usable_rate": float(
            df["salary_usable_for_analysis"].mean()
        ),
        "usable_hourly_salary_summary": {
            "median": (
                float(usable_salary.median())
                if not usable_salary.empty
                else None
            ),
            "p95": (
                float(usable_salary.quantile(0.95))
                if not usable_salary.empty
                else None
            ),
            "min": (
                float(usable_salary.min())
                if not usable_salary.empty
                else None
            ),
            "max": (
                float(usable_salary.max())
                if not usable_salary.empty
                else None
            ),
        },
        "structured_salary_consistency_counts": {
            str(key): int(value)
            for key, value in consistency_counts.items()
        },
        "potential_duplicate_rows": int(
            df["is_potential_duplicate"].sum()
        ),
        "unique_search_query_titles": int(
            df["search_query_title"].nunique()
        ),
        "unique_search_onet_codes": int(
            df["search_onet_code"].nunique()
        ),
        "unique_job_cities": int(
            df["job_city"].replace("", np.nan).nunique()
        ),
        "unique_job_counties": int(
            df["job_county"].replace("", np.nan).nunique()
        ),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build the analytics-ready AllSumJobs job-posting dataset."
        )
    )

    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to recovered_job_postings.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/analysis_job_postings.csv"
        ),
        help="Analytics-ready CSV output path.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "data/processed/analysis_dataset_report.json"
        ),
        help="Validation-report output path.",
    )

    return parser.parse_args()


def main() -> None:
    """Build and validate the analytics-ready dataset."""
    args = parse_args()

    analysis_df = build_analysis_dataset(
        args.input_csv
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    analysis_df.to_csv(
        args.output,
        index=False,
        encoding="utf-8",
    )

    report = build_report(analysis_df)

    args.report.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("Analysis dataset build complete.")
    print(f"Rows: {report['rows']}")
    print(
        "Usable salary rows: "
        f"{report['salary_usable_rows']} "
        f"({report['salary_usable_rate']:.2%})"
    )
    print(
        "Potential duplicate rows: "
        f"{report['potential_duplicate_rows']}"
    )
    print(
        "Search query titles / O*NET codes: "
        f"{report['unique_search_query_titles']} / "
        f"{report['unique_search_onet_codes']}"
    )


if __name__ == "__main__":
    main()
