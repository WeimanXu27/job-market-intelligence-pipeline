from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

RECOVERED_PATH = DATA_DIR / "recovered_job_postings.csv"
PATCH28_PATH = DATA_DIR / "skill_patch28_full_output.csv"

SUMMARY_PATH = DATA_DIR / "skill_patch28_county_gap_audit_summary.csv"
CA_CITY_SUMMARY_PATH = DATA_DIR / "skill_patch28_ca_missing_county_city_summary.csv"
CA_POSTINGS_PATH = DATA_DIR / "skill_patch28_ca_missing_county_postings.csv"

EXPECTED_PATCH28_ROWS = 64_283

CA_RE = re.compile(r"(?:,\s*CA(?:\s*[,•]|$)|\bCalifornia\b)", re.IGNORECASE)
AZ_RE = re.compile(r"(?:,\s*AZ(?:\s*[,•]|$)|\bArizona\b)", re.IGNORECASE)
CANADA_RE = re.compile(
    r"(?:,\s*(?:ON|BC|AB|QC|MB|SK|NS|NB|NL|PE|YT|NT|NU)"
    r"(?:\s*[,•]|$)|\bCanada\b|"
    r"\bOntario\b|\bBritish Columbia\b|\bAlberta\b|\bQuebec\b)",
    re.IGNORECASE,
)


def normalize_string(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.replace("\u200b", "", regex=False)
        .str.strip()
    )


def classify_missing_county_location(location: str) -> str:
    value = str(location).strip()

    if CA_RE.search(value):
        return "california"
    if AZ_RE.search(value):
        return "arizona"
    if CANADA_RE.search(value):
        return "canada"

    if not value:
        return "location_missing"

    if value.casefold() in {
        "on-site",
        "onsite",
        "hybrid",
        "remote",
    }:
        return "work_arrangement_only"

    return "other_or_unresolved"


def main() -> None:
    for path in (RECOVERED_PATH, PATCH28_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    patch28 = pd.read_csv(
        PATCH28_PATH,
        usecols=["record_id"],
        dtype=str,
    )

    if len(patch28) != EXPECTED_PATCH28_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_PATCH28_ROWS:,} Patch 28 rows, "
            f"found {len(patch28):,}"
        )

    if patch28["record_id"].nunique() != EXPECTED_PATCH28_ROWS:
        raise RuntimeError("Patch 28 record_id uniqueness failed")

    recovered = pd.read_csv(
        RECOVERED_PATH,
        usecols=[
            "record_id",
            "location",
            "search_location",
            "city",
            "county",
        ],
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    for column in (
        "record_id",
        "location",
        "search_location",
        "city",
        "county",
    ):
        recovered[column] = normalize_string(recovered[column])

    scoped = patch28.merge(
        recovered,
        on="record_id",
        how="left",
        validate="one_to_one",
    )

    if scoped["location"].isna().any():
        raise RuntimeError("Unexpected missing join to recovered_job_postings.csv")

    missing = scoped.loc[scoped["county"].eq("")].copy()
    missing["missing_county_group"] = missing["location"].map(
        classify_missing_county_location
    )

    ca_missing = missing.loc[
        missing["missing_county_group"].eq("california")
    ].copy()

    # City-level audit. Keep this descriptive only; no county is assigned yet.
    def join_top_values(series: pd.Series, limit: int = 5) -> str:
        values = (
            series.fillna("")
            .astype(str)
            .str.strip()
        )
        values = values[values.ne("")]
        if values.empty:
            return ""

        counts = values.value_counts()
        return " | ".join(
            f"{value} ({count})"
            for value, count in counts.head(limit).items()
        )

    city_summary = (
        ca_missing.assign(
            city_clean=ca_missing["city"].replace("", "(blank city)")
        )
        .groupby("city_clean", dropna=False)
        .agg(
            posting_count=("record_id", "nunique"),
            location_examples=("location", join_top_values),
            search_market_examples=("search_location", join_top_values),
        )
        .reset_index()
        .rename(columns={"city_clean": "city"})
        .sort_values(
            ["posting_count", "city"],
            ascending=[False, True],
        )
    )

    summary_rows = [
        ("patch28_processed_postings", len(scoped)),
        ("postings_with_existing_county", int(scoped["county"].ne("").sum())),
        ("postings_missing_county", len(missing)),
        (
            "missing_county_california_postings",
            int((missing["missing_county_group"] == "california").sum()),
        ),
        (
            "missing_county_arizona_postings",
            int((missing["missing_county_group"] == "arizona").sum()),
        ),
        (
            "missing_county_canada_postings",
            int((missing["missing_county_group"] == "canada").sum()),
        ),
        (
            "missing_county_location_missing",
            int((missing["missing_county_group"] == "location_missing").sum()),
        ),
        (
            "missing_county_work_arrangement_only",
            int(
                (
                    missing["missing_county_group"]
                    == "work_arrangement_only"
                ).sum()
            ),
        ),
        (
            "missing_county_other_or_unresolved",
            int(
                (
                    missing["missing_county_group"]
                    == "other_or_unresolved"
                ).sum()
            ),
        ),
        ("unique_ca_places_missing_county", int(city_summary["city"].nunique())),
        (
            "ca_missing_county_blank_city_postings",
            int(ca_missing["city"].eq("").sum()),
        ),
        (
            "audit_note",
            "This audit does not assign counties. It isolates California "
            "postings whose historical county field is blank so the mapping "
            "can be built and reviewed separately.",
        ),
    ]

    summary = pd.DataFrame(summary_rows, columns=["metric", "value"])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    city_summary.to_csv(CA_CITY_SUMMARY_PATH, index=False)
    ca_missing[
        [
            "record_id",
            "location",
            "search_location",
            "city",
            "county",
        ]
    ].to_csv(CA_POSTINGS_PATH, index=False)

    print("Patch 28 California county-gap audit complete.")
    print()
    print(f"Patch 28 processed postings: {len(scoped):,}")
    print(f"Postings with existing county: {scoped['county'].ne('').sum():,}")
    print(f"Postings missing county: {len(missing):,}")
    print()
    print("Missing-county groups:")
    print(
        missing["missing_county_group"]
        .value_counts()
        .to_string()
    )
    print()
    print(f"California missing-county postings: {len(ca_missing):,}")
    print(f"Unique California city/place labels: {len(city_summary):,}")
    print(f"Blank city among CA missing-county rows: {ca_missing['city'].eq('').sum():,}")
    print()
    print("Top California city/place labels missing county:")
    print(city_summary.head(30).to_string(index=False))
    print()
    print("Wrote:")
    print(f"  {SUMMARY_PATH}")
    print(f"  {CA_CITY_SUMMARY_PATH}")
    print(f"  {CA_POSTINGS_PATH}")
    print()
    print(
        "IMPORTANT: No county values were filled in. "
        "This is the audit input for the next mapping step."
    )


if __name__ == "__main__":
    main()
