from __future__ import annotations

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"

INPUT_PATH = DATA_DIR / "skill_patch28_ca_county_geocode_match_preview.csv"
DRAFT_PATH = REFERENCE_DIR / "ca_place_to_county_mapping_draft.csv"
REVIEW_PATH = REFERENCE_DIR / "ca_place_to_county_review_queue.csv"

EXPECTED_PLACES = 346
EXPECTED_POSTINGS = 5_098
EXPECTED_AUTO_RESOLVED = 313
EXPECTED_REVIEW_REQUIRED = 33


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing required input: {INPUT_PATH}")

    source = pd.read_csv(INPUT_PATH)

    if len(source) != EXPECTED_PLACES:
        raise RuntimeError(
            f"Expected {EXPECTED_PLACES} place labels, found {len(source)}"
        )

    if int(source["posting_count"].sum()) != EXPECTED_POSTINGS:
        raise RuntimeError("California posting total mismatch")

    draft = pd.DataFrame(
        {
            "source_place": source["city"],
            "mapped_county": source["final_mapped_county"].fillna(""),
            "mapping_status": source["final_match_status"].map(
                {
                    "auto_matched": "resolved_auto",
                    "manual_review": "review_required",
                    "unresolved": "review_required",
                }
            ),
            "resolution_method": source["final_match_method"].fillna(""),
            "geocoder_score": source["final_match_score"],
            "posting_count": source["posting_count"],
            "official_place_name": source["official_place_name"].fillna(""),
            "evidence_note": source["final_match_detail"].fillna(""),
            "location_examples": source["location_examples"].fillna(""),
            "search_market_examples": source["search_market_examples"].fillna(""),
        }
    )

    if draft["mapping_status"].isna().any():
        bad = source.loc[draft["mapping_status"].isna(), "final_match_status"].unique()
        raise RuntimeError(f"Unexpected final_match_status values: {bad}")

    # Do not let provisional geocoder candidates leak into unresolved rows.
    review_mask = draft["mapping_status"].eq("review_required")
    draft.loc[review_mask, "mapped_county"] = ""

    auto_count = int(draft["mapping_status"].eq("resolved_auto").sum())
    review_count = int(review_mask.sum())

    if auto_count != EXPECTED_AUTO_RESOLVED:
        raise RuntimeError(
            f"Expected {EXPECTED_AUTO_RESOLVED} auto-resolved places, "
            f"found {auto_count}"
        )

    if review_count != EXPECTED_REVIEW_REQUIRED:
        raise RuntimeError(
            f"Expected {EXPECTED_REVIEW_REQUIRED} review-required places, "
            f"found {review_count}"
        )

    review = (
        draft.loc[review_mask]
        .copy()
        .sort_values(
            ["posting_count", "source_place"],
            ascending=[False, True],
        )
    )

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    draft.to_csv(DRAFT_PATH, index=False)
    review.to_csv(REVIEW_PATH, index=False)

    print("California county reference-table draft created.")
    print()
    print(f"All place labels:       {len(draft):,}")
    print(f"All postings:           {int(draft['posting_count'].sum()):,}")
    print(f"Resolved automatically: {auto_count:,} places")
    print(f"Review required:        {review_count:,} places")
    print()
    print("Resolution methods among auto-resolved rows:")
    print(
        draft.loc[draft["mapping_status"].eq("resolved_auto"), "resolution_method"]
        .value_counts()
        .to_string()
    )
    print()
    print("Wrote:")
    print(f"  {DRAFT_PATH}")
    print(f"  {REVIEW_PATH}")
    print()
    print(
        "IMPORTANT: This is a draft reference table. "
        "The 33 review-required rows intentionally have blank mapped_county values."
    )


if __name__ == "__main__":
    main()
