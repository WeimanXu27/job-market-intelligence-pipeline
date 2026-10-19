"""Explore the analytics-ready AllSumJobs job-posting dataset.

This script performs reproducible descriptive analysis after the recovered
dataset has been audited and transformed by build_analysis_dataset.py.

It creates compact aggregate tables for business analysis and future Tableau
work. It does not modify source records, classify job families, run NLP skill
extraction, or infer that search-query O*NET codes are verified occupations
for individual job postings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


MIN_SALARY_SAMPLE_SIZE = 50
TOP_COMPANIES_LIMIT = 100
TOP_CITIES_LIMIT = 100
TOP_ACTUAL_TITLES_PER_QUERY = 5


def _coerce_bool(series: pd.Series) -> pd.Series:
    """Convert common CSV boolean representations to bool."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    normalized = (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return normalized.isin(
        {"true", "1", "yes", "y"}
    )


def _nonempty_nunique(series: pd.Series) -> int:
    """Count unique non-empty string values."""
    cleaned = (
        series.fillna("")
        .astype(str)
        .str.strip()
    )
    return int(
        cleaned[cleaned.ne("")].nunique()
    )


def load_analysis_dataset(input_path: Path) -> pd.DataFrame:
    """Load and validate the analytics-ready job-posting table."""
    df = pd.read_csv(
        input_path,
        low_memory=False,
    )

    required_columns = {
        "record_id",
        "posting_date",
        "posting_month",
        "company",
        "job_title",
        "search_query_title",
        "search_onet_code",
        "search_market",
        "job_city",
        "job_county",
        "salary_usable_for_analysis",
        "hourly_salary_clean",
        "annual_salary_equivalent",
        "is_potential_duplicate",
    }

    missing_columns = sorted(
        required_columns.difference(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Analysis dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    if df.empty:
        raise ValueError(
            "Analysis dataset is empty."
        )

    if not df["record_id"].is_unique:
        raise ValueError(
            "record_id must remain unique in the analysis dataset."
        )

    df["salary_usable_for_analysis"] = _coerce_bool(
        df["salary_usable_for_analysis"]
    )
    df["is_potential_duplicate"] = _coerce_bool(
        df["is_potential_duplicate"]
    )

    df["posting_date"] = pd.to_datetime(
        df["posting_date"],
        errors="coerce",
    )

    if df["posting_date"].isna().any():
        raise ValueError(
            "Invalid posting_date values were found."
        )

    return df


def build_postings_by_month(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize posting volume and usable salary coverage by month."""
    monthly = (
        df.groupby(
            "posting_month",
            as_index=False,
        )
        .agg(
            postings=("record_id", "size"),
            unique_companies=(
                "company",
                _nonempty_nunique,
            ),
            unique_job_titles=(
                "job_title",
                _nonempty_nunique,
            ),
            usable_salary_rows=(
                "salary_usable_for_analysis",
                "sum",
            ),
            median_hourly_salary=(
                "hourly_salary_clean",
                "median",
            ),
            median_annual_salary=(
                "annual_salary_equivalent",
                "median",
            ),
        )
        .sort_values("posting_month")
    )

    monthly["usable_salary_rate"] = (
        monthly["usable_salary_rows"]
        / monthly["postings"]
    )

    return monthly


def build_search_query_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize each historical search query and its returned job titles."""
    rows = []

    for search_query_title, group in df.groupby(
        "search_query_title",
        dropna=False,
    ):
        job_title_counts = (
            group["job_title"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        job_title_counts = (
            job_title_counts[
                job_title_counts.ne("")
            ]
            .value_counts()
        )

        if job_title_counts.empty:
            top_actual_title = ""
            top_actual_title_count = 0
            top_actual_title_share = np.nan
        else:
            top_actual_title = str(
                job_title_counts.index[0]
            )
            top_actual_title_count = int(
                job_title_counts.iloc[0]
            )
            top_actual_title_share = (
                top_actual_title_count
                / len(group)
            )

        onet_codes = (
            group["search_onet_code"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        onet_codes = sorted(
            onet_codes[onet_codes.ne("")].unique()
        )

        usable_salary_rows = int(
            group[
                "salary_usable_for_analysis"
            ].sum()
        )

        rows.append(
            {
                "search_query_title": search_query_title,
                "search_onet_code": (
                    onet_codes[0]
                    if len(onet_codes) == 1
                    else "|".join(onet_codes)
                ),
                "search_onet_code_count": len(
                    onet_codes
                ),
                "postings": int(len(group)),
                "unique_job_titles": (
                    _nonempty_nunique(
                        group["job_title"]
                    )
                ),
                "job_title_diversity_rate": (
                    _nonempty_nunique(
                        group["job_title"]
                    )
                    / len(group)
                ),
                "top_actual_job_title": (
                    top_actual_title
                ),
                "top_actual_job_title_count": (
                    top_actual_title_count
                ),
                "top_actual_job_title_share": (
                    top_actual_title_share
                ),
                "unique_companies": (
                    _nonempty_nunique(
                        group["company"]
                    )
                ),
                "unique_job_cities": (
                    _nonempty_nunique(
                        group["job_city"]
                    )
                ),
                "usable_salary_rows": (
                    usable_salary_rows
                ),
                "usable_salary_rate": (
                    usable_salary_rows
                    / len(group)
                ),
                "median_hourly_salary": (
                    group["hourly_salary_clean"]
                    .median()
                ),
                "median_annual_salary": (
                    group[
                        "annual_salary_equivalent"
                    ]
                    .median()
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "postings",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def build_search_query_title_examples(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return the most common actual job titles within each search query."""
    rows = []

    for search_query_title, group in df.groupby(
        "search_query_title",
        dropna=False,
    ):
        title_counts = (
            group["job_title"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        title_counts = (
            title_counts[
                title_counts.ne("")
            ]
            .value_counts()
            .head(
                TOP_ACTUAL_TITLES_PER_QUERY
            )
        )

        for rank, (
            job_title,
            count,
        ) in enumerate(
            title_counts.items(),
            start=1,
        ):
            rows.append(
                {
                    "search_query_title": (
                        search_query_title
                    ),
                    "rank": rank,
                    "job_title": job_title,
                    "postings": int(count),
                    "share_within_search_query": (
                        count / len(group)
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_search_market_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize historical search markets and returned job geography."""
    grouped = (
        df.groupby(
            "search_market",
            as_index=False,
        )
        .agg(
            postings=("record_id", "size"),
            unique_job_cities=(
                "job_city",
                _nonempty_nunique,
            ),
            unique_job_counties=(
                "job_county",
                _nonempty_nunique,
            ),
            usable_salary_rows=(
                "salary_usable_for_analysis",
                "sum",
            ),
            median_hourly_salary=(
                "hourly_salary_clean",
                "median",
            ),
            median_annual_salary=(
                "annual_salary_equivalent",
                "median",
            ),
        )
        .sort_values(
            "postings",
            ascending=False,
        )
    )

    grouped["usable_salary_rate"] = (
        grouped["usable_salary_rows"]
        / grouped["postings"]
    )

    return grouped


def _build_location_summary(
    df: pd.DataFrame,
    location_column: str,
    output_name: str,
    limit: int | None = None,
) -> pd.DataFrame:
    """Build a posting and salary summary for one geography dimension."""
    working = df.copy()

    working[location_column] = (
        working[location_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    working = working[
        working[location_column].ne("")
    ]

    grouped = (
        working.groupby(
            location_column,
            as_index=False,
        )
        .agg(
            postings=("record_id", "size"),
            unique_companies=(
                "company",
                _nonempty_nunique,
            ),
            unique_job_titles=(
                "job_title",
                _nonempty_nunique,
            ),
            usable_salary_rows=(
                "salary_usable_for_analysis",
                "sum",
            ),
            median_hourly_salary=(
                "hourly_salary_clean",
                "median",
            ),
            median_annual_salary=(
                "annual_salary_equivalent",
                "median",
            ),
        )
    )

    grouped["usable_salary_rate"] = (
        grouped["usable_salary_rows"]
        / grouped["postings"]
    )

    grouped["salary_sample_sufficient"] = (
        grouped["usable_salary_rows"]
        >= MIN_SALARY_SAMPLE_SIZE
    )

    grouped = grouped.sort_values(
        "postings",
        ascending=False,
    )

    if limit is not None:
        grouped = grouped.head(limit)

    return grouped.rename(
        columns={
            location_column: output_name,
        }
    )


def build_company_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize the largest employers in the observed dataset."""
    working = df.copy()

    working["company"] = (
        working["company"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    working = working[
        working["company"].ne("")
    ]

    grouped = (
        working.groupby(
            "company",
            as_index=False,
        )
        .agg(
            postings=("record_id", "size"),
            unique_search_queries=(
                "search_query_title",
                _nonempty_nunique,
            ),
            unique_job_titles=(
                "job_title",
                _nonempty_nunique,
            ),
            unique_job_cities=(
                "job_city",
                _nonempty_nunique,
            ),
            usable_salary_rows=(
                "salary_usable_for_analysis",
                "sum",
            ),
            median_hourly_salary=(
                "hourly_salary_clean",
                "median",
            ),
        )
    )

    grouped["usable_salary_rate"] = (
        grouped["usable_salary_rows"]
        / grouped["postings"]
    )

    return (
        grouped.sort_values(
            "postings",
            ascending=False,
        )
        .head(TOP_COMPANIES_LIMIT)
    )


def build_overview_report(
    df: pd.DataFrame,
    search_query_summary: pd.DataFrame,
) -> Dict:
    """Build a compact JSON summary for validation and narrative review."""
    usable_salary = df.loc[
        df["salary_usable_for_analysis"],
        "hourly_salary_clean",
    ]

    annual_salary = df.loc[
        df["salary_usable_for_analysis"],
        "annual_salary_equivalent",
    ]

    low_concentration = (
        search_query_summary.sort_values(
            [
                "top_actual_job_title_share",
                "postings",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .head(10)
    )

    salary_queries = (
        search_query_summary[
            search_query_summary[
                "usable_salary_rows"
            ]
            >= MIN_SALARY_SAMPLE_SIZE
        ]
        .sort_values(
            "median_hourly_salary",
            ascending=False,
        )
        .head(10)
    )

    return {
        "rows": int(len(df)),
        "date_range": {
            "min": (
                df["posting_date"]
                .min()
                .strftime("%Y-%m-%d")
            ),
            "max": (
                df["posting_date"]
                .max()
                .strftime("%Y-%m-%d")
            ),
        },
        "coverage": {
            "unique_companies": (
                _nonempty_nunique(
                    df["company"]
                )
            ),
            "unique_actual_job_titles": (
                _nonempty_nunique(
                    df["job_title"]
                )
            ),
            "unique_search_query_titles": (
                _nonempty_nunique(
                    df[
                        "search_query_title"
                    ]
                )
            ),
            "unique_search_markets": (
                _nonempty_nunique(
                    df["search_market"]
                )
            ),
            "unique_job_cities": (
                _nonempty_nunique(
                    df["job_city"]
                )
            ),
            "unique_job_counties": (
                _nonempty_nunique(
                    df["job_county"]
                )
            ),
        },
        "salary": {
            "usable_rows": int(
                df[
                    "salary_usable_for_analysis"
                ].sum()
            ),
            "usable_rate": float(
                df[
                    "salary_usable_for_analysis"
                ].mean()
            ),
            "median_hourly": (
                float(usable_salary.median())
                if not usable_salary.empty
                else None
            ),
            "median_annual_equivalent": (
                float(annual_salary.median())
                if not annual_salary.empty
                else None
            ),
        },
        "potential_duplicate_rows": int(
            df["is_potential_duplicate"].sum()
        ),
        "search_query_title_dispersion_examples": [
            {
                "search_query_title": row[
                    "search_query_title"
                ],
                "postings": int(
                    row["postings"]
                ),
                "unique_job_titles": int(
                    row[
                        "unique_job_titles"
                    ]
                ),
                "top_actual_job_title": row[
                    "top_actual_job_title"
                ],
                "top_actual_job_title_share": (
                    float(
                        row[
                            "top_actual_job_title_share"
                        ]
                    )
                ),
            }
            for _, row in low_concentration.iterrows()
        ],
        "highest_median_salary_search_queries": [
            {
                "search_query_title": row[
                    "search_query_title"
                ],
                "usable_salary_rows": int(
                    row[
                        "usable_salary_rows"
                    ]
                ),
                "median_hourly_salary": (
                    float(
                        row[
                            "median_hourly_salary"
                        ]
                    )
                ),
            }
            for _, row in salary_queries.iterrows()
        ],
    }


def write_outputs(
    df: pd.DataFrame,
    output_dir: Path,
) -> Dict:
    """Create reusable exploration tables and the overview report."""
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    postings_by_month = build_postings_by_month(
        df
    )
    search_query_summary = build_search_query_summary(
        df
    )
    search_query_title_examples = (
        build_search_query_title_examples(
            df
        )
    )
    search_market_summary = (
        build_search_market_summary(
            df
        )
    )
    county_summary = _build_location_summary(
        df,
        location_column="job_county",
        output_name="job_county",
    )
    city_summary = _build_location_summary(
        df,
        location_column="job_city",
        output_name="job_city",
        limit=TOP_CITIES_LIMIT,
    )
    company_summary = build_company_summary(
        df
    )

    tables = {
        "postings_by_month.csv": (
            postings_by_month
        ),
        "search_query_summary.csv": (
            search_query_summary
        ),
        "search_query_title_examples.csv": (
            search_query_title_examples
        ),
        "search_market_summary.csv": (
            search_market_summary
        ),
        "county_summary.csv": (
            county_summary
        ),
        "city_summary_top100.csv": (
            city_summary
        ),
        "company_summary_top100.csv": (
            company_summary
        ),
    }

    for filename, table in tables.items():
        table.to_csv(
            output_dir / filename,
            index=False,
            encoding="utf-8",
        )

    report = build_overview_report(
        df,
        search_query_summary,
    )

    report_path = (
        output_dir
        / "exploration_report.json"
    )
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Explore the analytics-ready "
            "AllSumJobs job-posting dataset."
        )
    )

    parser.add_argument(
        "input_csv",
        type=Path,
        help=(
            "Path to "
            "analysis_job_postings.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/processed/exploration"
        ),
        help=(
            "Directory for aggregate CSV "
            "tables and exploration report."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the descriptive job-market exploration."""
    args = parse_args()

    df = load_analysis_dataset(
        args.input_csv
    )
    report = write_outputs(
        df,
        args.output_dir,
    )

    print(
        "Job-market exploration complete."
    )
    print(
        f"Rows: {report['rows']}"
    )
    print(
        "Companies / actual titles: "
        f"{report['coverage']['unique_companies']} / "
        f"{report['coverage']['unique_actual_job_titles']}"
    )
    print(
        "Search queries / markets: "
        f"{report['coverage']['unique_search_query_titles']} / "
        f"{report['coverage']['unique_search_markets']}"
    )
    print(
        "Usable salary rows: "
        f"{report['salary']['usable_rows']} "
        f"({report['salary']['usable_rate']:.2%})"
    )
    print(
        "Median usable hourly / annual equivalent: "
        f"${report['salary']['median_hourly']:.2f} / "
        f"${report['salary']['median_annual_equivalent']:,.0f}"
    )


if __name__ == "__main__":
    main()
