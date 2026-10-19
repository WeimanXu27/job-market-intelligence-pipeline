from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"

BASE_PATH = DATA_DIR / "skill_patch28_postings_base_final.csv"
MAPPING_PATH = REFERENCE_DIR / "ca_place_to_county_mapping.csv"

OUTPUT_PATH = DATA_DIR / "skill_patch28_postings_base_geo_final.csv"
SUMMARY_PATH = DATA_DIR / "skill_patch28_postings_base_geo_final_summary.csv"
COUNTY_SIZE_PATH = DATA_DIR / "skill_patch28_county_sizes_geo_final.csv"

EXPECTED_POSTINGS = 64_283
EXPECTED_EXISTING_COUNTY = 55_110
EXPECTED_CA_MISSING_COUNTY = 5_098
EXPECTED_CA_REFERENCE_RESOLVED = 5_096
EXPECTED_CA_REFERENCE_UNRESOLVED = 2
EXPECTED_FINAL_COUNTY_RESOLVED = 60_206
EXPECTED_FINAL_UNIQUE_COUNTIES = 37
EXPECTED_ELIGIBLE_COUNTIES_100 = 29

EXPECTED_MAPPING_SHA256 = (
    "81ad75bd413714bdb7c61d063c39562bac0a6e75ff68603e52b205866c9b9240"
)

CA_RE = re.compile(
    r"(?:,\s*CA(?:\s*[,•]|$)|\bCalifornia\b)",
    flags=re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.replace("\u200b", "", regex=False)
        .str.strip()
    )


def main() -> None:
    for path in (BASE_PATH, MAPPING_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    mapping_sha = sha256_file(MAPPING_PATH)
    if mapping_sha != EXPECTED_MAPPING_SHA256:
        raise RuntimeError(
            "Frozen California mapping SHA256 mismatch.\n"
            f"Expected: {EXPECTED_MAPPING_SHA256}\n"
            f"Actual:   {mapping_sha}"
        )

    base = pd.read_csv(BASE_PATH, low_memory=False)
    mapping = pd.read_csv(MAPPING_PATH, dtype=str, keep_default_na=False)

    if len(base) != EXPECTED_POSTINGS:
        raise RuntimeError(
            f"Expected {EXPECTED_POSTINGS:,} postings, found {len(base):,}"
        )

    if base["record_id"].nunique() != EXPECTED_POSTINGS:
        raise RuntimeError("record_id uniqueness failed")

    required_base_columns = {
        "record_id",
        "location_raw",
        "job_city",
        "job_county",
    }
    missing_columns = required_base_columns - set(base.columns)
    if missing_columns:
        raise RuntimeError(
            f"Base table missing required columns: {sorted(missing_columns)}"
        )

    if mapping["source_place"].duplicated().any():
        raise RuntimeError("Duplicate source_place values in mapping")

    resolved_mapping = mapping[
        mapping["mapping_status"].isin(["resolved_auto", "resolved_manual"])
    ]
    noncanonical = resolved_mapping.loc[
        ~resolved_mapping["mapped_county"].str.endswith(" County")
    ]
    if not noncanonical.empty:
        raise RuntimeError(
            "Reference mapping contains non-canonical county labels: "
            + ", ".join(noncanonical["mapped_county"].unique())
        )

    base["job_county_original"] = clean_text(base["job_county"])
    base["job_city"] = clean_text(base["job_city"])
    base["location_raw"] = clean_text(base["location_raw"])

    existing_county_mask = base["job_county_original"].ne("")
    if int(existing_county_mask.sum()) != EXPECTED_EXISTING_COUNTY:
        raise RuntimeError("Unexpected existing county count")

    ca_missing_mask = (
        ~existing_county_mask
        & base["location_raw"].str.contains(CA_RE, na=False)
    )

    if int(ca_missing_mask.sum()) != EXPECTED_CA_MISSING_COUNTY:
        raise RuntimeError("Unexpected California missing-county count")

    ca_places = set(base.loc[ca_missing_mask, "job_city"])
    reference_places = set(mapping["source_place"])
    missing_reference_places = sorted(ca_places - reference_places)
    if missing_reference_places:
        raise RuntimeError(
            "California place labels missing from reference: "
            f"{missing_reference_places}"
        )

    ref = mapping[
        [
            "source_place",
            "mapped_county",
            "mapping_status",
            "resolution_method",
        ]
    ].rename(
        columns={
            "mapped_county": "reference_mapped_county",
            "mapping_status": "county_mapping_status",
            "resolution_method": "county_mapping_method",
        }
    )

    out = base.merge(
        ref,
        left_on="job_city",
        right_on="source_place",
        how="left",
        validate="many_to_one",
    )

    out["job_county_final"] = out["job_county_original"]
    out["county_enrichment_status"] = "existing_county"

    missing_non_ca_mask = (
        out["job_county_original"].eq("")
        & ~out["location_raw"].str.contains(CA_RE, na=False)
    )
    out.loc[
        missing_non_ca_mask,
        "county_enrichment_status",
    ] = "missing_outside_ca_scope"

    ca_missing_out = (
        out["job_county_original"].eq("")
        & out["location_raw"].str.contains(CA_RE, na=False)
    )

    ca_resolved_mask = (
        ca_missing_out
        & out["county_mapping_status"].isin(
            ["resolved_auto", "resolved_manual"]
        )
        & out["reference_mapped_county"].fillna("").ne("")
    )

    ca_unresolved_mask = (
        ca_missing_out
        & out["county_mapping_status"].eq("unresolved")
    )

    out.loc[
        ca_resolved_mask,
        "job_county_final",
    ] = out.loc[
        ca_resolved_mask,
        "reference_mapped_county",
    ]

    out.loc[
        ca_resolved_mask
        & out["county_mapping_status"].eq("resolved_auto"),
        "county_enrichment_status",
    ] = "ca_reference_resolved_auto"

    out.loc[
        ca_resolved_mask
        & out["county_mapping_status"].eq("resolved_manual"),
        "county_enrichment_status",
    ] = "ca_reference_resolved_manual"

    out.loc[
        ca_unresolved_mask,
        "county_enrichment_status",
    ] = "ca_reference_unresolved"

    resolved_ca_count = int(ca_resolved_mask.sum())
    unresolved_ca_count = int(ca_unresolved_mask.sum())

    if resolved_ca_count != EXPECTED_CA_REFERENCE_RESOLVED:
        raise RuntimeError("Resolved California posting count mismatch")

    if unresolved_ca_count != EXPECTED_CA_REFERENCE_UNRESOLVED:
        raise RuntimeError("Unresolved California posting count mismatch")

    final_county_resolved = int(out["job_county_final"].ne("").sum())
    if final_county_resolved != EXPECTED_FINAL_COUNTY_RESOLVED:
        raise RuntimeError("Final county-resolved posting count mismatch")

    out["job_county"] = out["job_county_final"]

    # After enrichment, every resolved California county uses the same naming
    # convention as the historical county field.
    resolved_final = out.loc[out["job_county"].ne(""), "job_county"]
    bad_names = sorted(
        resolved_final.loc[~resolved_final.str.endswith(" County")].unique()
    )
    if bad_names:
        raise RuntimeError(
            "Non-canonical county labels remain after enrichment: "
            f"{bad_names}"
        )

    out = out.drop(columns=["source_place", "job_county_final"])

    county_sizes = (
        out.loc[out["job_county"].ne("")]
        .groupby("job_county", as_index=False)
        .agg(processed_postings=("record_id", "nunique"))
        .sort_values(
            ["processed_postings", "job_county"],
            ascending=[False, True],
        )
    )

    unique_counties = len(county_sizes)
    eligible_100 = int((county_sizes["processed_postings"] >= 100).sum())

    if unique_counties != EXPECTED_FINAL_UNIQUE_COUNTIES:
        raise RuntimeError(
            f"Expected {EXPECTED_FINAL_UNIQUE_COUNTIES} unique counties, "
            f"found {unique_counties}"
        )

    if eligible_100 != EXPECTED_ELIGIBLE_COUNTIES_100:
        raise RuntimeError(
            f"Expected {EXPECTED_ELIGIBLE_COUNTIES_100} counties with >=100 postings, "
            f"found {eligible_100}"
        )

    status_counts = out["county_enrichment_status"].value_counts().to_dict()

    summary_rows = [
        ("processed_postings", len(out)),
        ("existing_county_postings", EXPECTED_EXISTING_COUNTY),
        ("ca_missing_county_input_postings", EXPECTED_CA_MISSING_COUNTY),
        ("ca_reference_resolved_postings", resolved_ca_count),
        ("ca_reference_unresolved_postings", unresolved_ca_count),
        ("final_county_resolved_postings", final_county_resolved),
        (
            "final_county_resolved_pct",
            round(final_county_resolved / EXPECTED_POSTINGS * 100, 4),
        ),
        (
            "final_county_missing_postings",
            EXPECTED_POSTINGS - final_county_resolved,
        ),
        ("final_unique_counties", unique_counties),
        ("counties_with_at_least_100_postings", eligible_100),
        ("county_naming_convention", "<County Name> County"),
        ("mapping_sha256", mapping_sha),
        ("existing_county_status_rows", status_counts.get("existing_county", 0)),
        (
            "ca_reference_resolved_auto_rows",
            status_counts.get("ca_reference_resolved_auto", 0),
        ),
        (
            "ca_reference_resolved_manual_rows",
            status_counts.get("ca_reference_resolved_manual", 0),
        ),
        (
            "ca_reference_unresolved_rows",
            status_counts.get("ca_reference_unresolved", 0),
        ),
        (
            "missing_outside_ca_scope_rows",
            status_counts.get("missing_outside_ca_scope", 0),
        ),
    ]

    summary = pd.DataFrame(summary_rows, columns=["metric", "value"])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    county_sizes.to_csv(COUNTY_SIZE_PATH, index=False)

    print("Page 4 county enrichment complete.")
    print()
    print(f"Processed postings:             {len(out):,}")
    print(f"Final county-resolved postings: {final_county_resolved:,}")
    print(
        "Final county-resolved pct:      "
        f"{final_county_resolved / EXPECTED_POSTINGS * 100:.2f}%"
    )
    print(f"Unique final counties:          {unique_counties:,}")
    print(f"Counties with >=100 postings:   {eligible_100:,}")
    print(f"County naming:                  <County Name> County")
    print(f"Mapping SHA256:                 {mapping_sha}")
    print()
    print("Wrote:")
    print(f"  {OUTPUT_PATH}")
    print(f"  {SUMMARY_PATH}")
    print(f"  {COUNTY_SIZE_PATH}")


if __name__ == "__main__":
    main()
