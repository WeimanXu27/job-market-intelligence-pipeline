from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import hashlib
import re
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYTICS_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

FULL_OUTPUT_PATH = DATA_DIR / "skill_patch28_full_output.csv"
REVIEW_FLAGS_PATH = DATA_DIR / "skill_patch28_review_flags.csv"
ANALYSIS_POSTINGS_PATH = DATA_DIR / "analysis_job_postings.csv"

POSTINGS_BASE_PATH = DATA_DIR / "skill_patch28_postings_base.csv"
SKILLS_LONG_PATH = DATA_DIR / "skill_patch28_skills_long.csv"
BUILD_SUMMARY_PATH = DATA_DIR / "skill_patch28_analytics_tables_summary.csv"

EXPECTED_FULL_OUTPUT_SHA256 = (
    "ecf749007540fdf8e4c20109125fdbb1aae189956c297912c2600df818b8fe53"
)
EXPECTED_ANALYSIS_POSTINGS_SHA256 = (
    "bb80e310c866641f512469e6101eeb1a143c1b32db12bb5547c1a4304d1ccbfd"
)

EXPECTED_ANALYSIS_ROWS = 65_247
EXPECTED_SKILL_PROCESSED_ROWS = 64_283
EXPECTED_NON_SKILL_SOURCE_ROWS = 964
EXPECTED_RAW_SKILL_INSTANCES = 1_098_031
EXPECTED_ZERO_SKILL_ROWS = 1_730

EXPECTED_FINAL_POSTING_SKILL_ROWS_ALL = 1_097_790
EXPECTED_PAGE4_LONG_ROWS = 1_096_311
EXPECTED_EXCLUDED_POSTING_SKILL_ROWS = 1_479

EXPECTED_FINAL_UNIQUE_KEYS_ALL = 285_415
EXPECTED_PAGE4_UNIQUE_KEYS = 284_443
EXPECTED_CAPABILITY_KEYS = 269_978
EXPECTED_QUALIFICATION_KEYS = 14_465

EXPECTED_CAPABILITY_ROWS = 1_027_026
EXPECTED_QUALIFICATION_ROWS = 69_285

NUMBERED_LINE_RE = re.compile(r"^\s*(\d+)\.\s+(.*\S)\s*$")


# Import the already-reviewed frozen policy rather than maintaining a second
# independent copy of normalization/exclusion rules.
sys.path.insert(0, str(ANALYTICS_DIR))
try:
    import finalize_skill_policy as frozen_policy
except ImportError as exc:
    raise ImportError(
        "Could not import analytics/04_skills_intelligence/finalize_skill_policy.py. "
        "Keep the reviewed policy script in analytics/04_skills_intelligence/."
    ) from exc


DISPLAY_OVERRIDES = {
    "problem solving": "Problem Solving",
    "multitasking": "Multitasking",
    "preventive maintenance": "Preventive Maintenance",
    "microsoft excel": "Microsoft Excel",
    "microsoft word": "Microsoft Word",
    "microsoft powerpoint": "Microsoft PowerPoint",
    "microsoft outlook": "Microsoft Outlook",
    "record keeping": "Record Keeping",
    "time management": "Time Management",
    "bachelor's degree": "Bachelor's Degree",
    "high school diploma or ged": "High School Diploma or GED",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def main() -> None:
    for path in (
        FULL_OUTPUT_PATH,
        REVIEW_FLAGS_PATH,
        ANALYSIS_POSTINGS_PATH,
    ):
        require_file(path)

    full_sha = sha256_file(FULL_OUTPUT_PATH)
    if full_sha != EXPECTED_FULL_OUTPUT_SHA256:
        raise RuntimeError(
            "Patch 28 full-output SHA256 mismatch.\n"
            f"Expected: {EXPECTED_FULL_OUTPUT_SHA256}\n"
            f"Actual:   {full_sha}"
        )

    analysis_sha = sha256_file(ANALYSIS_POSTINGS_PATH)
    if analysis_sha != EXPECTED_ANALYSIS_POSTINGS_SHA256:
        raise RuntimeError(
            "analysis_job_postings.csv SHA256 mismatch.\n"
            "This script was validated against the frozen Page 1-3 analytics "
            "dataset. Inspect the file before proceeding.\n"
            f"Expected: {EXPECTED_ANALYSIS_POSTINGS_SHA256}\n"
            f"Actual:   {analysis_sha}"
        )

    full = pd.read_csv(FULL_OUTPUT_PATH)
    review_flags = pd.read_csv(REVIEW_FLAGS_PATH)
    analysis = pd.read_csv(ANALYSIS_POSTINGS_PATH, low_memory=False)

    if len(full) != EXPECTED_SKILL_PROCESSED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_SKILL_PROCESSED_ROWS:,} Patch 28 rows, "
            f"found {len(full):,}"
        )
    if full["record_id"].nunique() != EXPECTED_SKILL_PROCESSED_ROWS:
        raise RuntimeError("Patch 28 record_id uniqueness check failed")

    if len(analysis) != EXPECTED_ANALYSIS_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ANALYSIS_ROWS:,} analysis rows, "
            f"found {len(analysis):,}"
        )
    if analysis["record_id"].nunique() != EXPECTED_ANALYSIS_ROWS:
        raise RuntimeError("analysis_job_postings record_id uniqueness check failed")

    missing_analysis_ids = set(full["record_id"]) - set(analysis["record_id"])
    if missing_analysis_ids:
        raise RuntimeError(
            f"{len(missing_analysis_ids):,} Patch 28 IDs are missing from "
            "analysis_job_postings.csv"
        )

    non_skill_source_rows = len(analysis) - len(full)
    if non_skill_source_rows != EXPECTED_NON_SKILL_SOURCE_ROWS:
        raise RuntimeError(
            "Unexpected count of source postings outside the Patch 28 skill "
            f"universe: {non_skill_source_rows:,}"
        )

    review_flag_map = frozen_policy.load_review_flag_map(review_flags)

    flags_by_final_key: dict[str, set[str]] = defaultdict(set)
    for audit_key, flags in review_flag_map.items():
        base = frozen_policy.base_key(audit_key)
        final_key = frozen_policy.final_key_from_base(base)
        flags_by_final_key[final_key].update(flags)

    raw_rows = []
    for row in full.itertuples(index=False):
        if pd.isna(row.extracted_skills) or str(row.extracted_skills) == "":
            continue

        for line in str(row.extracted_skills).splitlines():
            match = NUMBERED_LINE_RE.match(line)
            if match is None:
                raise RuntimeError(
                    f"Malformed numbered skill line for {row.record_id}: {line!r}"
                )

            skill_raw = match.group(2).strip()
            base_key = frozen_policy.base_key(skill_raw)
            final_key = frozen_policy.final_key_from_base(base_key)
            policy_bucket = frozen_policy.classify_policy(
                final_key,
                flags_by_final_key.get(final_key, set()),
            )

            raw_rows.append(
                (
                    row.record_id,
                    skill_raw,
                    final_key,
                    policy_bucket,
                )
            )

    skills_raw = pd.DataFrame(
        raw_rows,
        columns=[
            "record_id",
            "skill_raw",
            "final_skill_key",
            "policy_bucket",
        ],
    )

    if len(skills_raw) != EXPECTED_RAW_SKILL_INSTANCES:
        raise RuntimeError(
            "Raw skill-instance count mismatch: "
            f"expected={EXPECTED_RAW_SKILL_INSTANCES:,}; "
            f"actual={len(skills_raw):,}"
        )

    # One analytical occurrence = one final normalized concept per posting.
    skills_all = skills_raw.drop_duplicates(
        subset=["record_id", "final_skill_key"],
        keep="first",
    ).copy()

    if len(skills_all) != EXPECTED_FINAL_POSTING_SKILL_ROWS_ALL:
        raise RuntimeError(
            "Final posting-skill row count mismatch: "
            f"expected={EXPECTED_FINAL_POSTING_SKILL_ROWS_ALL:,}; "
            f"actual={len(skills_all):,}"
        )

    if skills_all["final_skill_key"].nunique() != EXPECTED_FINAL_UNIQUE_KEYS_ALL:
        raise RuntimeError("Final unique-key count mismatch")

    # Canonical display is the most frequent raw wording for each final key,
    # with a small set of readability overrides already reviewed earlier.
    raw_counts = (
        skills_raw.groupby(["final_skill_key", "skill_raw"])
        .size()
        .rename("raw_instances")
        .reset_index()
    )

    display = (
        raw_counts.sort_values(
            ["final_skill_key", "raw_instances", "skill_raw"],
            ascending=[True, False, True],
        )
        .drop_duplicates("final_skill_key")
        [["final_skill_key", "skill_raw"]]
        .rename(columns={"skill_raw": "skill_display"})
    )

    display["skill_display"] = display.apply(
        lambda row: DISPLAY_OVERRIDES.get(
            row["final_skill_key"],
            row["skill_display"],
        ),
        axis=1,
    )

    skills_all = skills_all.merge(
        display,
        on="final_skill_key",
        how="left",
        validate="many_to_one",
    )

    # Page 4 production table contains only the frozen included concepts.
    # Excluded leakage remains preserved in the raw Patch 28 export and in
    # skill_patch28_policy_exclusions.csv, but cannot accidentally enter Tableau.
    skills_long = skills_all[
        skills_all["policy_bucket"] != "excluded"
    ][
        [
            "record_id",
            "final_skill_key",
            "skill_display",
            "policy_bucket",
        ]
    ].copy()

    skills_long = skills_long.rename(
        columns={"policy_bucket": "skill_type"}
    )
    skills_long["skill_posting_flag"] = 1

    if len(skills_long) != EXPECTED_PAGE4_LONG_ROWS:
        raise RuntimeError(
            "Page 4 long-table row count mismatch: "
            f"expected={EXPECTED_PAGE4_LONG_ROWS:,}; "
            f"actual={len(skills_long):,}"
        )

    if skills_long["final_skill_key"].nunique() != EXPECTED_PAGE4_UNIQUE_KEYS:
        raise RuntimeError("Page 4 unique-skill count mismatch")

    capability_mask = skills_long["skill_type"] == "capability"
    qualification_mask = skills_long["skill_type"] == "qualification"

    if int(capability_mask.sum()) != EXPECTED_CAPABILITY_ROWS:
        raise RuntimeError("Capability posting-skill row count mismatch")
    if int(qualification_mask.sum()) != EXPECTED_QUALIFICATION_ROWS:
        raise RuntimeError("Qualification posting-skill row count mismatch")

    if (
        skills_long.loc[capability_mask, "final_skill_key"].nunique()
        != EXPECTED_CAPABILITY_KEYS
    ):
        raise RuntimeError("Capability unique-key count mismatch")

    if (
        skills_long.loc[qualification_mask, "final_skill_key"].nunique()
        != EXPECTED_QUALIFICATION_KEYS
    ):
        raise RuntimeError("Qualification unique-key count mismatch")

    # Build posting-level denominator table from the exact 64,283 Patch 28
    # universe, then attach the existing Page 1-3 posting attributes.
    base_columns = [
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
        "salary_usable_for_analysis",
        "hourly_salary_clean",
        "annual_salary_equivalent",
        "is_potential_duplicate",
    ]

    postings_base = (
        full[
            [
                "record_id",
                "pool",
                "extracted_skill_count",
                "has_extracted_skill",
            ]
        ]
        .merge(
            analysis[base_columns],
            on="record_id",
            how="left",
            validate="one_to_one",
        )
        .rename(
            columns={
                "pool": "skill_run_pool",
                "extracted_skill_count": "raw_extracted_skill_count",
                "has_extracted_skill": "has_raw_extracted_skill",
            }
        )
    )

    if postings_base[base_columns[1:]].isna().all(axis=1).any():
        raise RuntimeError("Unexpected completely missing posting metadata after join")

    # Posting-level counts after normalization and policy filtering.
    page4_counts = (
        skills_long.groupby("record_id")
        .size()
        .rename("page4_skill_count")
    )
    capability_counts = (
        skills_long.loc[capability_mask]
        .groupby("record_id")
        .size()
        .rename("capability_count")
    )
    qualification_counts = (
        skills_long.loc[qualification_mask]
        .groupby("record_id")
        .size()
        .rename("qualification_count")
    )

    postings_base = postings_base.join(
        page4_counts,
        on="record_id",
    ).join(
        capability_counts,
        on="record_id",
    ).join(
        qualification_counts,
        on="record_id",
    )

    for column in (
        "page4_skill_count",
        "capability_count",
        "qualification_count",
    ):
        postings_base[column] = postings_base[column].fillna(0).astype(int)

    postings_base["has_page4_skill"] = (
        postings_base["page4_skill_count"] > 0
    )
    postings_base["skill_processing_status"] = postings_base[
        "has_page4_skill"
    ].map(
        {
            True: "has_included_skill",
            False: "zero_skill",
        }
    )

    raw_zero_rows = int(
        (postings_base["raw_extracted_skill_count"] == 0).sum()
    )
    if raw_zero_rows != EXPECTED_ZERO_SKILL_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ZERO_SKILL_ROWS:,} raw zero-skill postings, "
            f"found {raw_zero_rows:,}"
        )

    page4_zero_rows = int((postings_base["page4_skill_count"] == 0).sum())
    if page4_zero_rows != EXPECTED_ZERO_SKILL_ROWS:
        raise RuntimeError(
            "The frozen policy unexpectedly created excluded-only postings. "
            f"Expected {EXPECTED_ZERO_SKILL_ROWS:,} Page 4 zero-skill rows, "
            f"found {page4_zero_rows:,}"
        )

    if len(postings_base) != EXPECTED_SKILL_PROCESSED_ROWS:
        raise RuntimeError("Posting-base row count mismatch")
    if postings_base["record_id"].nunique() != EXPECTED_SKILL_PROCESSED_ROWS:
        raise RuntimeError("Posting-base record_id uniqueness failed")

    # Keep stable, human-readable column order.
    postings_base = postings_base[
        [
            "record_id",
            "skill_run_pool",
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
            "salary_usable_for_analysis",
            "hourly_salary_clean",
            "annual_salary_equivalent",
            "is_potential_duplicate",
            "raw_extracted_skill_count",
            "has_raw_extracted_skill",
            "page4_skill_count",
            "capability_count",
            "qualification_count",
            "has_page4_skill",
            "skill_processing_status",
        ]
    ]

    # Deterministic output order.
    run_order = full[["record_id", "run_order"]]
    postings_base = postings_base.merge(
        run_order,
        on="record_id",
        how="left",
        validate="one_to_one",
    ).sort_values("run_order")
    postings_base = postings_base.drop(columns="run_order")

    skills_long = skills_long.merge(
        run_order,
        on="record_id",
        how="left",
        validate="many_to_one",
    ).sort_values(
        ["run_order", "final_skill_key"],
        kind="stable",
    )
    skills_long = skills_long.drop(columns="run_order")

    excluded_rows = len(skills_all) - len(skills_long)

    summary_rows = [
        ("patch28_full_output_sha256", full_sha),
        ("analysis_job_postings_sha256", analysis_sha),
        ("analysis_source_rows", len(analysis)),
        ("skill_processed_postings", len(postings_base)),
        ("source_rows_outside_skill_universe", non_skill_source_rows),
        ("missing_record_id_joins", len(missing_analysis_ids)),
        ("raw_zero_skill_postings", raw_zero_rows),
        (
            "postings_with_at_least_one_page4_skill",
            int(postings_base["has_page4_skill"].sum()),
        ),
        (
            "postings_with_at_least_one_capability",
            int((postings_base["capability_count"] > 0).sum()),
        ),
        (
            "postings_with_at_least_one_qualification",
            int((postings_base["qualification_count"] > 0).sum()),
        ),
        ("raw_skill_instances", len(skills_raw)),
        ("final_posting_skill_rows_all", len(skills_all)),
        ("excluded_posting_skill_rows", excluded_rows),
        ("page4_skills_long_rows", len(skills_long)),
        ("page4_unique_skill_keys", skills_long["final_skill_key"].nunique()),
        (
            "page4_capability_rows",
            int((skills_long["skill_type"] == "capability").sum()),
        ),
        (
            "page4_qualification_rows",
            int((skills_long["skill_type"] == "qualification").sum()),
        ),
        (
            "page4_capability_unique_keys",
            skills_long.loc[
                skills_long["skill_type"] == "capability",
                "final_skill_key",
            ].nunique(),
        ),
        (
            "page4_qualification_unique_keys",
            skills_long.loc[
                skills_long["skill_type"] == "qualification",
                "final_skill_key",
            ].nunique(),
        ),
        (
            "tableau_relationship_note",
            "Relate skill_patch28_postings_base to skill_patch28_skills_long "
            "on record_id. Do not physically inner-join them; the base table "
            "preserves the 1,730 processed zero-skill postings and the exact "
            "64,283-posting denominator.",
        ),
        (
            "classification_note",
            "search_query_title and search_onet_code are search-derived fields, "
            "not verified job-family labels for each returned posting.",
        ),
    ]

    summary = pd.DataFrame(summary_rows, columns=["metric", "value"])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    postings_base.to_csv(POSTINGS_BASE_PATH, index=False)
    skills_long.to_csv(SKILLS_LONG_PATH, index=False)
    summary.to_csv(BUILD_SUMMARY_PATH, index=False)

    print("Patch 28 analytics tables build complete.")
    print()
    print(f"Posting base rows: {len(postings_base):,}")
    print(f"  zero-skill postings: {raw_zero_rows:,}")
    print(
        "  postings with >=1 Page 4 skill: "
        f"{int(postings_base['has_page4_skill'].sum()):,}"
    )
    print()
    print(f"Skills long rows: {len(skills_long):,}")
    print(
        "  capability rows: "
        f"{int((skills_long['skill_type'] == 'capability').sum()):,}"
    )
    print(
        "  qualification rows: "
        f"{int((skills_long['skill_type'] == 'qualification').sum()):,}"
    )
    print(f"  unique included skill keys: {skills_long['final_skill_key'].nunique():,}")
    print(f"Excluded posting-skill rows kept out of Page 4 table: {excluded_rows:,}")
    print()
    print("Wrote:")
    print(f"  {POSTINGS_BASE_PATH}")
    print(f"  {SKILLS_LONG_PATH}")
    print(f"  {BUILD_SUMMARY_PATH}")
    print()
    print(
        "IMPORTANT: In Tableau, use a logical relationship on record_id "
        "between the posting base and skills long tables. Do not use an "
        "inner physical join."
    )


if __name__ == "__main__":
    main()
