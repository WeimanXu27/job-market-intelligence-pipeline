"""Audit the recovered AllSumJobs job-posting dataset.

This script inspects field coverage, schema evolution, salary consistency,
occupation mappings, duplicate indicators, and text-quality flags before
feature engineering begins.

It does not delete records or apply modeling assumptions. The goal is to make
the downstream analysis-dataset design evidence-driven and reproducible.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


AUDIT_COLUMNS = [
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

NUMERIC_SALARY_COLUMNS = [
    "salary_job_type",
    "hourly_salary",
    "min_salary",
    "max_salary",
]


def _nonempty_count(series: pd.Series) -> int:
    """Count non-empty string values."""
    return int(series.fillna("").astype(str).str.strip().ne("").sum())


def _unique_nonempty_count(series: pd.Series) -> int:
    """Count distinct non-empty string values."""
    cleaned = series.fillna("").astype(str).str.strip()
    return int(cleaned[cleaned.ne("")].nunique())


def _safe_float(value) -> float | None:
    """Convert a scalar to a JSON-safe float."""
    if pd.isna(value):
        return None
    return float(value)


def audit_description_text(input_path: Path) -> Dict:
    """Stream the description column to avoid loading all long text into memory."""
    nonempty = 0
    lengths: List[int] = []
    unusual_2028 = 0
    unusual_2029 = 0

    with input_path.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as source:
        reader = csv.DictReader(source)

        for row in reader:
            description = row.get("description", "") or ""

            if description.strip():
                nonempty += 1
                lengths.append(len(description))

            unusual_2028 += description.count("\u2028")
            unusual_2029 += description.count("\u2029")

    length_series = pd.Series(lengths, dtype="float64")

    return {
        "nonempty_descriptions": nonempty,
        "description_length": {
            "median": _safe_float(length_series.median()),
            "p95": _safe_float(length_series.quantile(0.95)),
            "max": _safe_float(length_series.max()),
        },
        "unicode_line_separator_count": unusual_2028,
        "unicode_paragraph_separator_count": unusual_2029,
    }


def build_salary_audit(df: pd.DataFrame) -> Dict:
    """Audit salary fields without deciding final cleaning rules."""
    salary_df = df.copy()

    for column in NUMERIC_SALARY_COLUMNS:
        salary_df[f"{column}_num"] = pd.to_numeric(
            salary_df[column],
            errors="coerce",
        )

    hourly = salary_df["hourly_salary_num"].dropna()

    hourly_flags = {
        "non_null": int(hourly.count()),
        "non_positive": int((hourly <= 0).sum()),
        "positive_below_10": int(
            ((hourly > 0) & (hourly < 10)).sum()
        ),
        "between_10_and_300": int(
            ((hourly >= 10) & (hourly <= 300)).sum()
        ),
        "above_300": int((hourly > 300).sum()),
        "median": _safe_float(hourly.median()),
        "p95": _safe_float(hourly.quantile(0.95)),
        "p99": _safe_float(hourly.quantile(0.99)),
        "max": _safe_float(hourly.max()),
    }

    # Historical salary_job_type behaves like an annualized representation
    # for most rows where it overlaps with hourly_salary.
    legacy_mask = (
        salary_df["salary_job_type_num"].notna()
        & salary_df["hourly_salary_num"].notna()
        & (salary_df["hourly_salary_num"] > 0)
    )

    legacy_overlap = salary_df.loc[legacy_mask].copy()

    if not legacy_overlap.empty:
        expected_annual = (
            legacy_overlap["hourly_salary_num"] * 2080
        )
        legacy_match = np.isclose(
            legacy_overlap["salary_job_type_num"],
            expected_annual,
            rtol=0.03,
            atol=0.5,
        )
        legacy_match_rate = float(legacy_match.mean())
    else:
        legacy_match_rate = None

    range_df = salary_df[
        salary_df["salary_unit"].astype(str).str.strip().ne("")
    ].copy()

    range_df["base_salary"] = range_df["min_salary_num"].astype(float)

    both_bounds = (
        range_df["min_salary_num"].notna()
        & range_df["max_salary_num"].notna()
    )

    range_df.loc[both_bounds, "base_salary"] = (
        range_df.loc[both_bounds, "min_salary_num"]
        + range_df.loc[both_bounds, "max_salary_num"]
    ) / 2.0

    conversion_factor = {
        "hr": 1.0,
        "wk": 1.0 / 40.0,
        "mo": 12.0 / 2080.0,
        "yr": 1.0 / 2080.0,
    }

    range_df["expected_hourly"] = (
        range_df["base_salary"]
        * range_df["salary_unit"].map(conversion_factor)
    )

    comparison_mask = (
        range_df["expected_hourly"].notna()
        & (range_df["expected_hourly"] > 0)
        & range_df["hourly_salary_num"].notna()
    )

    comparison_df = range_df.loc[comparison_mask]

    if not comparison_df.empty:
        range_match = np.isclose(
            comparison_df["hourly_salary_num"],
            comparison_df["expected_hourly"],
            rtol=0.03,
            atol=0.5,
        )
        range_match_rate = float(range_match.mean())
    else:
        range_match_rate = None

    return {
        "hourly_salary": hourly_flags,
        "salary_unit_counts": (
            salary_df.loc[
                salary_df["salary_unit"].astype(str).str.strip().ne(""),
                "salary_unit",
            ]
            .value_counts()
            .to_dict()
        ),
        "legacy_salary_overlap": {
            "rows_compared": int(legacy_mask.sum()),
            "approx_matches_hourly_x_2080": legacy_match_rate,
        },
        "range_salary_overlap": {
            "rows_compared": int(comparison_mask.sum()),
            "approx_matches_recalculated_hourly": range_match_rate,
        },
    }


def build_monthly_schema_report(df: pd.DataFrame) -> List[Dict]:
    """Show how salary-field availability changed over time."""
    working = df.copy()
    working["date_parsed"] = pd.to_datetime(
        working["date"],
        errors="coerce",
    )
    working["month"] = working["date_parsed"].dt.to_period("M").astype(str)

    rows = []

    for month, group in working.groupby("month", dropna=False):
        rows.append(
            {
                "month": month,
                "rows": int(len(group)),
                "hourly_salary_nonempty": _nonempty_count(
                    group["hourly_salary"]
                ),
                "salary_job_type_nonempty": _nonempty_count(
                    group["salary_job_type"]
                ),
                "min_salary_nonempty": _nonempty_count(
                    group["min_salary"]
                ),
                "max_salary_nonempty": _nonempty_count(
                    group["max_salary"]
                ),
                "salary_unit_nonempty": _nonempty_count(
                    group["salary_unit"]
                ),
            }
        )

    return rows


def audit_dataset(input_path: Path) -> Dict:
    """Run the recovered-dataset field audit."""
    df = pd.read_csv(
        input_path,
        usecols=AUDIT_COLUMNS,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    dates = pd.to_datetime(df["date"], errors="coerce")

    coverage = {
        column: {
            "nonempty": _nonempty_count(df[column]),
            "coverage_rate": (
                _nonempty_count(df[column]) / len(df)
                if len(df)
                else 0.0
            ),
            "unique_nonempty": _unique_nonempty_count(df[column]),
        }
        for column in AUDIT_COLUMNS
    }

    search_title_to_onet = (
        df.groupby("search_title")["onet_code"]
        .nunique()
    )

    duplicate_rows = int(
        df.duplicated(
            subset=[
                "date",
                "company",
                "title",
                "search_title",
                "location",
                "url",
            ]
        ).sum()
    )

    report = {
        "input_file": input_path.name,
        "rows": int(len(df)),
        "record_id_unique": bool(df["record_id"].is_unique),
        "date_range": {
            "min": (
                dates.min().strftime("%Y-%m-%d")
                if dates.notna().any()
                else None
            ),
            "max": (
                dates.max().strftime("%Y-%m-%d")
                if dates.notna().any()
                else None
            ),
        },
        "field_coverage": coverage,
        "occupation_structure": {
            "unique_search_titles": _unique_nonempty_count(
                df["search_title"]
            ),
            "unique_onet_codes": _unique_nonempty_count(
                df["onet_code"]
            ),
            "max_onet_codes_per_search_title": int(
                search_title_to_onet.max()
                if not search_title_to_onet.empty
                else 0
            ),
            "top_search_titles": (
                df["search_title"]
                .value_counts()
                .head(20)
                .to_dict()
            ),
        },
        "location_structure": {
            "unique_search_locations": _unique_nonempty_count(
                df["search_location"]
            ),
            "unique_cities": _unique_nonempty_count(df["city"]),
            "unique_counties": _unique_nonempty_count(df["county"]),
        },
        "duplicate_indicators": {
            "exact_business_key_duplicates": duplicate_rows,
            "note": (
                "This is an audit flag only. Do not deduplicate on URL alone; "
                "historical URLs are not guaranteed to be unique job IDs."
            ),
        },
        "salary_audit": build_salary_audit(df),
        "monthly_schema_evolution": build_monthly_schema_report(df),
        "description_audit": audit_description_text(input_path),
    }

    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Audit the recovered AllSumJobs job-posting dataset."
    )

    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to recovered_job_postings.csv.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "data/processed/field_audit_report.json"
        ),
        help="JSON audit-report output path.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the recovered-data audit."""
    args = parse_args()

    report = audit_dataset(args.input_csv)

    args.report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.report.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("Recovered dataset audit complete.")
    print(f"Rows: {report['rows']}")
    print(
        "Unique search titles: "
        f"{report['occupation_structure']['unique_search_titles']}"
    )
    print(
        "Unique O*NET codes: "
        f"{report['occupation_structure']['unique_onet_codes']}"
    )
    print(
        "Hourly salary coverage: "
        f"{report['field_coverage']['hourly_salary']['coverage_rate']:.2%}"
    )
    print(
        "Salary rows between $10 and $300/hour: "
        f"{report['salary_audit']['hourly_salary']['between_10_and_300']}"
    )
    print(
        "Unicode LS/PS remaining: "
        f"{report['description_audit']['unicode_line_separator_count']} / "
        f"{report['description_audit']['unicode_paragraph_separator_count']}"
    )


if __name__ == "__main__":
    main()
