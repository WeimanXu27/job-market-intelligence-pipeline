from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"

CAPABILITIES_INPUT = DATA_DIR / "skill_patch28_capabilities_long_final.csv"
BASE_INPUT = DATA_DIR / "skill_patch28_postings_base_geo_final.csv"
REVIEW_PATH = REFERENCE_DIR / "page4_residual_qualification_review.csv"

CAPABILITIES_OUTPUT = DATA_DIR / "skill_patch28_capabilities_long_page4_final.csv"
BASE_OUTPUT = DATA_DIR / "skill_patch28_postings_base_page4_final.csv"
REMOVED_OUTPUT = DATA_DIR / "skill_patch28_residual_qualification_removed.csv"
SUMMARY_OUTPUT = DATA_DIR / "skill_patch28_page4_production_final_summary.csv"
SUPPORT_OUTPUT = DATA_DIR / "skill_patch28_capability_support_page4_final.csv"

EXPECTED_INPUT_CAPABILITY_ROWS = 986_643
EXPECTED_INPUT_UNIQUE_KEYS = 255_421
EXPECTED_REVIEW_KEYS = 655
EXPECTED_EXCLUDE_KEYS = 648
EXPECTED_KEEP_KEYS = 7
EXPECTED_REMOVED_ROWS = 2_384
EXPECTED_OUTPUT_CAPABILITY_ROWS = 984_259
EXPECTED_OUTPUT_UNIQUE_KEYS = 254_773
EXPECTED_POSTINGS = 64_283


def main() -> None:
    for path in (CAPABILITIES_INPUT, BASE_INPUT, REVIEW_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    capabilities = pd.read_csv(CAPABILITIES_INPUT, low_memory=False)
    base = pd.read_csv(BASE_INPUT, low_memory=False)
    review = pd.read_csv(REVIEW_PATH, low_memory=False)

    if len(capabilities) != EXPECTED_INPUT_CAPABILITY_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_INPUT_CAPABILITY_ROWS:,} input capability rows, "
            f"found {len(capabilities):,}"
        )

    if capabilities["final_skill_key"].nunique() != EXPECTED_INPUT_UNIQUE_KEYS:
        raise RuntimeError("Unexpected input unique capability-key count")

    if capabilities.duplicated(["record_id", "final_skill_key"]).any():
        raise RuntimeError("Duplicate posting-capability rows found in input")

    if len(base) != EXPECTED_POSTINGS or base["record_id"].nunique() != EXPECTED_POSTINGS:
        raise RuntimeError("Posting-base row/ID count mismatch")

    if review["final_skill_key"].duplicated().any():
        raise RuntimeError("Duplicate final_skill_key in review table")

    if len(review) != EXPECTED_REVIEW_KEYS:
        raise RuntimeError(
            f"Expected {EXPECTED_REVIEW_KEYS} reviewed keys, found {len(review)}"
        )

    allowed_decisions = {"exclude_from_capability", "keep_capability"}
    unexpected_decisions = set(review["decision"]) - allowed_decisions
    if unexpected_decisions:
        raise RuntimeError(
            f"Unexpected review decisions: {sorted(unexpected_decisions)}"
        )

    exclude_keys = set(
        review.loc[
            review["decision"].eq("exclude_from_capability"),
            "final_skill_key",
        ]
    )
    keep_keys = set(
        review.loc[
            review["decision"].eq("keep_capability"),
            "final_skill_key",
        ]
    )

    if len(exclude_keys) != EXPECTED_EXCLUDE_KEYS:
        raise RuntimeError("Unexpected excluded-key count")
    if len(keep_keys) != EXPECTED_KEEP_KEYS:
        raise RuntimeError("Unexpected kept-key count")
    if exclude_keys & keep_keys:
        raise RuntimeError("A reviewed key cannot be both excluded and kept")

    missing_review_keys = (exclude_keys | keep_keys) - set(capabilities["final_skill_key"])
    if missing_review_keys:
        raise RuntimeError(
            "Reviewed keys missing from input capability table: "
            f"{sorted(missing_review_keys)[:20]}"
        )

    removed = capabilities.loc[
        capabilities["final_skill_key"].isin(exclude_keys)
    ].copy()

    output = capabilities.loc[
        ~capabilities["final_skill_key"].isin(exclude_keys)
    ].copy()

    if len(removed) != EXPECTED_REMOVED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_REMOVED_ROWS:,} removed rows, found {len(removed):,}"
        )

    if len(output) != EXPECTED_OUTPUT_CAPABILITY_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_OUTPUT_CAPABILITY_ROWS:,} output rows, "
            f"found {len(output):,}"
        )

    if output["final_skill_key"].nunique() != EXPECTED_OUTPUT_UNIQUE_KEYS:
        raise RuntimeError("Unexpected output unique capability-key count")

    if output["final_skill_key"].isin(exclude_keys).any():
        raise RuntimeError("Excluded qualification key survived cleanup")

    if not keep_keys.issubset(set(output["final_skill_key"])):
        raise RuntimeError("One or more reviewed keep-capability keys were removed")

    # Update posting-level capability counts while preserving all 64,283 postings.
    capability_counts = (
        output.groupby("record_id")
        .size()
        .rename("capability_count_page4_final")
    )

    base_out = base.copy()
    base_out["capability_count_page4_final"] = (
        base_out["record_id"]
        .map(capability_counts)
        .fillna(0)
        .astype(int)
    )
    base_out["has_capability_page4_final"] = (
        base_out["capability_count_page4_final"] > 0
    )
    base_out["capability_processing_status_page4_final"] = (
        base_out["has_capability_page4_final"]
        .map(
            {
                True: "has_capability",
                False: "no_included_capability",
            }
        )
    )

    postings_with_capability = int(
        base_out["has_capability_page4_final"].sum()
    )
    postings_without_capability = EXPECTED_POSTINGS - postings_with_capability

    # Final support table used for selector-threshold review and Tableau QA.
    support = (
        output.groupby(["final_skill_key", "skill_display"], as_index=False)
        .agg(posting_count=("record_id", "nunique"))
    )

    if support["final_skill_key"].duplicated().any():
        # Stable fallback if one normalized key has multiple displays.
        display_freq = (
            output.groupby(["final_skill_key", "skill_display"])
            .size()
            .rename("rows")
            .reset_index()
            .sort_values(
                ["final_skill_key", "rows", "skill_display"],
                ascending=[True, False, True],
            )
            .drop_duplicates("final_skill_key")
            [["final_skill_key", "skill_display"]]
        )
        support = (
            output.groupby("final_skill_key", as_index=False)
            .agg(posting_count=("record_id", "nunique"))
            .merge(
                display_freq,
                on="final_skill_key",
                how="left",
                validate="one_to_one",
            )
        )

    support["overall_prevalence_pct"] = (
        support["posting_count"] / EXPECTED_POSTINGS * 100
    )
    support = support.sort_values(
        ["posting_count", "final_skill_key"],
        ascending=[False, True],
    )

    selector_counts = {
        threshold: int((support["posting_count"] >= threshold).sum())
        for threshold in (100, 250, 500, 1000)
    }

    summary_rows = [
        ("processed_postings", EXPECTED_POSTINGS),
        ("input_capability_rows", len(capabilities)),
        ("removed_residual_qualification_rows", len(removed)),
        ("final_capability_rows", len(output)),
        ("final_unique_capability_keys", output["final_skill_key"].nunique()),
        ("postings_with_at_least_one_final_capability", postings_with_capability),
        ("postings_with_no_final_capability", postings_without_capability),
        (
            "final_capability_coverage_pct",
            round(postings_with_capability / EXPECTED_POSTINGS * 100, 4),
        ),
        ("capabilities_with_support_ge_100", selector_counts[100]),
        ("capabilities_with_support_ge_250", selector_counts[250]),
        ("capabilities_with_support_ge_500", selector_counts[500]),
        ("capabilities_with_support_ge_1000", selector_counts[1000]),
        (
            "production_scope_note",
            "This is the final Page 4 production capability scope after geography enrichment and residual qualification cleanup. Explicit qualification phrases are excluded according to the version-controlled review table; reviewed work-process/domain exceptions are retained.",
        ),
    ]

    summary = pd.DataFrame(summary_rows, columns=["metric", "value"])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(CAPABILITIES_OUTPUT, index=False)
    base_out.to_csv(BASE_OUTPUT, index=False)
    removed.to_csv(REMOVED_OUTPUT, index=False)
    support.to_csv(SUPPORT_OUTPUT, index=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False)

    print("Final Page 4 production cleanup complete.")
    print()
    print(f"Removed residual qualification rows: {len(removed):,}")
    print(f"Final capability rows:                {len(output):,}")
    print(
        "Final unique capability keys:         "
        f"{output['final_skill_key'].nunique():,}"
    )
    print(
        "Postings with >=1 final capability:   "
        f"{postings_with_capability:,}"
    )
    print(
        "Postings with no final capability:    "
        f"{postings_without_capability:,}"
    )
    print(
        "Final capability coverage:            "
        f"{postings_with_capability / EXPECTED_POSTINGS * 100:.2f}%"
    )
    print()
    print("Selector support counts:")
    for threshold in (100, 250, 500, 1000):
        print(
            f"  >= {threshold:>4,} postings: "
            f"{selector_counts[threshold]:,} capabilities"
        )
    print()
    print("Wrote:")
    print(f"  {CAPABILITIES_OUTPUT}")
    print(f"  {BASE_OUTPUT}")
    print(f"  {REMOVED_OUTPUT}")
    print(f"  {SUPPORT_OUTPUT}")
    print(f"  {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()
