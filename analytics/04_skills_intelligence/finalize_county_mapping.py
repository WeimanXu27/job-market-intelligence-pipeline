from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv
import hashlib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"

DRAFT_PATH = REFERENCE_DIR / "ca_place_to_county_mapping_draft.csv"
BATCH1_PATH = REFERENCE_DIR / "ca_place_to_county_manual_review_batch1.csv"
BATCH2_PATH = REFERENCE_DIR / "ca_place_to_county_manual_review_batch2.csv"

FINAL_PATH = REFERENCE_DIR / "ca_place_to_county_mapping.csv"
SUMMARY_PATH = REFERENCE_DIR / "ca_place_to_county_mapping_summary.csv"

EXPECTED_PLACES = 346
EXPECTED_POSTINGS = 5_098
EXPECTED_AUTO_PLACES = 313
EXPECTED_MANUAL_PLACES = 31
EXPECTED_UNRESOLVED_PLACES = 2
EXPECTED_RESOLVED_POSTINGS = 5_096
EXPECTED_UNRESOLVED_POSTINGS = 2
EXPECTED_FINAL_SHA256 = "81ad75bd413714bdb7c61d063c39562bac0a6e75ff68603e52b205866c9b9240"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_county_name(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    return value if value.endswith(" County") else f"{value} County"


def main() -> None:
    draft = read_csv(DRAFT_PATH)
    batch1 = read_csv(BATCH1_PATH)
    batch2 = read_csv(BATCH2_PATH)

    if len(draft) != EXPECTED_PLACES:
        raise RuntimeError(
            f"Expected {EXPECTED_PLACES} draft rows, found {len(draft)}"
        )

    if sum(int(row["posting_count"]) for row in draft) != EXPECTED_POSTINGS:
        raise RuntimeError("Draft posting total mismatch")

    draft_place_counts = Counter(row["source_place"] for row in draft)
    duplicates = sorted(
        place for place, count in draft_place_counts.items()
        if count != 1
    )
    if duplicates:
        raise RuntimeError(f"Duplicate draft place labels: {duplicates}")

    manual_rows = batch1 + batch2
    manual_place_counts = Counter(row["source_place"] for row in manual_rows)
    duplicate_manual = sorted(
        place for place, count in manual_place_counts.items()
        if count != 1
    )
    if duplicate_manual:
        raise RuntimeError(
            f"Duplicate manual-review place labels: {duplicate_manual}"
        )

    review_places = {
        row["source_place"]
        for row in draft
        if row["mapping_status"] == "review_required"
    }
    manual_places = set(manual_place_counts)

    if review_places != manual_places:
        missing_reviews = sorted(review_places - manual_places)
        unexpected_reviews = sorted(manual_places - review_places)
        raise RuntimeError(
            "Manual-review set mismatch.\n"
            f"Missing manual reviews: {missing_reviews}\n"
            f"Unexpected manual reviews: {unexpected_reviews}"
        )

    manual_by_place = {row["source_place"]: row for row in manual_rows}

    final_rows: list[dict[str, str]] = []

    for row in draft:
        out = dict(row)
        place = row["source_place"]

        if row["mapping_status"] == "review_required":
            reviewed = manual_by_place[place]
            out["mapped_county"] = reviewed["mapped_county"]
            out["mapping_status"] = reviewed["mapping_status"]
            out["resolution_method"] = reviewed["resolution_method"]
            out["evidence_note"] = reviewed["evidence_note"]
            out["evidence_source_1"] = reviewed["evidence_source_1"]
            out["evidence_source_2"] = reviewed["evidence_source_2"]
        else:
            out["evidence_source_1"] = ""
            out["evidence_source_2"] = ""

        # Canonical production naming: match historical job_county values.
        out["mapped_county"] = canonical_county_name(out["mapped_county"])

        final_rows.append(out)

    status_counts = Counter(row["mapping_status"] for row in final_rows)

    expected_status_counts = {
        "resolved_auto": EXPECTED_AUTO_PLACES,
        "resolved_manual": EXPECTED_MANUAL_PLACES,
        "unresolved": EXPECTED_UNRESOLVED_PLACES,
    }

    if status_counts != expected_status_counts:
        raise RuntimeError(
            "Final status counts mismatch.\n"
            f"Expected: {expected_status_counts}\n"
            f"Actual:   {dict(status_counts)}"
        )

    resolved_statuses = {"resolved_auto", "resolved_manual"}

    resolved_postings = sum(
        int(row["posting_count"])
        for row in final_rows
        if row["mapping_status"] in resolved_statuses
    )
    unresolved_postings = sum(
        int(row["posting_count"])
        for row in final_rows
        if row["mapping_status"] == "unresolved"
    )

    if resolved_postings != EXPECTED_RESOLVED_POSTINGS:
        raise RuntimeError(
            "Resolved posting count mismatch: "
            f"{resolved_postings:,}"
        )

    if unresolved_postings != EXPECTED_UNRESOLVED_POSTINGS:
        raise RuntimeError(
            "Unresolved posting count mismatch: "
            f"{unresolved_postings:,}"
        )

    for row in final_rows:
        status = row["mapping_status"]
        county = row["mapped_county"].strip()

        if status in resolved_statuses:
            if not county:
                raise RuntimeError(
                    f"Resolved place has blank county: {row['source_place']}"
                )
            if not county.endswith(" County"):
                raise RuntimeError(
                    f"Non-canonical county label: {row['source_place']} -> {county}"
                )

        if status == "unresolved" and county:
            raise RuntimeError(
                f"Unresolved place unexpectedly has county: {row['source_place']}"
            )

    fieldnames = [
        "source_place",
        "mapped_county",
        "mapping_status",
        "resolution_method",
        "posting_count",
        "geocoder_score",
        "official_place_name",
        "evidence_note",
        "evidence_source_1",
        "evidence_source_2",
        "location_examples",
        "search_market_examples",
    ]

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(FINAL_PATH, final_rows, fieldnames)

    final_sha = sha256_file(FINAL_PATH)
    if final_sha != EXPECTED_FINAL_SHA256:
        raise RuntimeError(
            "Final mapping SHA256 mismatch.\n"
            f"Expected: {EXPECTED_FINAL_SHA256}\n"
            f"Actual:   {final_sha}"
        )

    unresolved_labels = " | ".join(
        row["source_place"]
        for row in final_rows
        if row["mapping_status"] == "unresolved"
    )

    summary_rows = [
        {"metric": "input_ca_places", "value": str(EXPECTED_PLACES)},
        {"metric": "input_ca_postings", "value": str(EXPECTED_POSTINGS)},
        {"metric": "resolved_auto_places", "value": str(EXPECTED_AUTO_PLACES)},
        {"metric": "resolved_manual_places", "value": str(EXPECTED_MANUAL_PLACES)},
        {"metric": "unresolved_places", "value": str(EXPECTED_UNRESOLVED_PLACES)},
        {"metric": "resolved_postings", "value": str(resolved_postings)},
        {"metric": "unresolved_postings", "value": str(unresolved_postings)},
        {
            "metric": "resolved_posting_pct",
            "value": f"{resolved_postings / EXPECTED_POSTINGS * 100:.4f}",
        },
        {"metric": "unresolved_labels", "value": unresolved_labels},
        {"metric": "county_naming_convention", "value": "<County Name> County"},
        {"metric": "final_mapping_sha256", "value": final_sha},
        {
            "metric": "production_note",
            "value": (
                "Production geography enrichment should consume this frozen "
                "reference table; it should not rerun external geocoding."
            ),
        },
    ]

    write_csv(SUMMARY_PATH, summary_rows, ["metric", "value"])

    print("California place-to-county reference mapping finalized.")
    print()
    print(f"Places:                 {EXPECTED_PLACES:,}")
    print(f"Postings:               {EXPECTED_POSTINGS:,}")
    print(f"Resolved auto places:   {EXPECTED_AUTO_PLACES:,}")
    print(f"Resolved manual places: {EXPECTED_MANUAL_PLACES:,}")
    print(f"Unresolved places:      {EXPECTED_UNRESOLVED_PLACES:,}")
    print(f"Resolved postings:      {resolved_postings:,}")
    print(f"Unresolved postings:    {unresolved_postings:,}")
    print(f"County naming:          <County Name> County")
    print(f"Final SHA256:           {final_sha}")
    print()
    print("Wrote:")
    print(f"  {FINAL_PATH}")
    print(f"  {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
