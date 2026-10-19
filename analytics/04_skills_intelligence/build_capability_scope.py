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

POSTINGS_BASE_PATH = DATA_DIR / "skill_patch28_postings_base_final.csv"
CAPABILITIES_LONG_PATH = DATA_DIR / "skill_patch28_capabilities_long_final.csv"
NONCAP_AUDIT_PATH = DATA_DIR / "skill_patch28_noncapability_audit_final.csv"
SUMMARY_PATH = DATA_DIR / "skill_patch28_page4_final_summary.csv"

EXPECTED_FULL_OUTPUT_SHA256 = (
    "ecf749007540fdf8e4c20109125fdbb1aae189956c297912c2600df818b8fe53"
)
EXPECTED_ANALYSIS_POSTINGS_SHA256 = (
    "bb80e310c866641f512469e6101eeb1a143c1b32db12bb5547c1a4304d1ccbfd"
)

EXPECTED_PROCESSED_POSTINGS = 64_283
EXPECTED_ANALYSIS_ROWS = 65_247
EXPECTED_RAW_SKILL_INSTANCES = 1_098_031
EXPECTED_FINAL_POSTING_SKILL_ROWS = 1_097_790
EXPECTED_RAW_ZERO_SKILL_POSTINGS = 1_730

# Expected results from the reviewed conservative Page 4 capability scope.
EXPECTED_CAPABILITY_ROWS = 986_643
EXPECTED_CAPABILITY_KEYS = 255_421
EXPECTED_POSTINGS_WITH_CAPABILITY = 61_966
EXPECTED_ZERO_CAPABILITY_POSTINGS = 2_317

EXPECTED_BASELINE_QUALIFICATION_ROWS = 69_285
EXPECTED_NEW_CREDENTIAL_MENTION_ROWS = 1_560
EXPECTED_EXPERIENCE_REQUIREMENT_ROWS = 38_823
EXPECTED_BASELINE_EXCLUDED_ROWS = 1_479
EXPECTED_NONCAP_AUDIT_ROWS = 111_147

NUMBERED_LINE_RE = re.compile(r"^\s*(\d+)\.\s+(.*\S)\s*$")
EXPERIENCE_RE = re.compile(r"\bexperience\b", flags=re.IGNORECASE)
CREDENTIAL_END_RE = re.compile(r"\bcredentials?$", flags=re.IGNORECASE)

DOMAIN_EXPERIENCE_EXACT = {
    "customer experience",
    "user experience",
    "user experience (ux)",
    "employee experience",
    "patient experience",
    "guest experience",
    "digital experience",
    "client experience",
    "member experience",
    "consumer experience",
    "developer experience",
    "candidate experience",
    "brand experience",
    "product experience",
    "learning experience",
}

DOMAIN_EXPERIENCE_RE = re.compile(
    r"^(?:customer|user|employee|patient|guest|digital|client|member|consumer|"
    r"developer|candidate|brand|product|learning)\s+experience"
    r"(?:\s+(?:management|design|research|strategy|optimization|delivery|"
    r"enhancement|improvement|analytics|analysis|measurement|operations|"
    r"platform|program|journey|mapping))?"
    r"(?:\s*\([^)]+\))?$",
    flags=re.IGNORECASE,
)

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
}


sys.path.insert(0, str(ANALYTICS_DIR))
try:
    import finalize_skill_policy as frozen_policy
except ImportError as exc:
    raise ImportError(
        "Could not import analytics/04_skills_intelligence/finalize_skill_policy.py. "
        "Keep the reviewed normalization/alias policy script in analytics/04_skills_intelligence/."
    ) from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_domain_experience_capability(final_key: str) -> bool:
    key = str(final_key).strip().lower()
    return (
        key in DOMAIN_EXPERIENCE_EXACT
        or DOMAIN_EXPERIENCE_RE.fullmatch(key) is not None
    )


def classify_page4_scope(
    final_key: str,
    baseline_bucket: str,
) -> str:
    """
    Precision-first Page 4 classification.

    The dashboard uses capability rows only.

    Qualification-like terms are retained for audit but are NOT interpreted
    as current/entry qualifications, because the source descriptions mix
    required, preferred, current, and post-hire credential conditions.

    Candidate-background experience phrases are also retained for audit but
    excluded from capability ranking. Clear domain concepts such as
    customer experience and user experience remain capabilities.
    """
    if baseline_bucket == "excluded":
        return "excluded_non_skill"

    if baseline_bucket == "qualification":
        return "qualification_like_mention"

    if CREDENTIAL_END_RE.search(final_key):
        return "credential_mention"

    if EXPERIENCE_RE.search(final_key):
        if is_domain_experience_capability(final_key):
            return "capability"
        return "experience_requirement"

    return "capability"


def main() -> None:
    for path in (
        FULL_OUTPUT_PATH,
        REVIEW_FLAGS_PATH,
        ANALYSIS_POSTINGS_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

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
            f"Expected: {EXPECTED_ANALYSIS_POSTINGS_SHA256}\n"
            f"Actual:   {analysis_sha}"
        )

    full = pd.read_csv(FULL_OUTPUT_PATH)
    review_flags = pd.read_csv(REVIEW_FLAGS_PATH)
    analysis = pd.read_csv(ANALYSIS_POSTINGS_PATH, low_memory=False)

    if len(full) != EXPECTED_PROCESSED_POSTINGS:
        raise RuntimeError("Unexpected Patch 28 posting count")
    if full["record_id"].nunique() != EXPECTED_PROCESSED_POSTINGS:
        raise RuntimeError("Patch 28 record_id uniqueness failed")

    if len(analysis) != EXPECTED_ANALYSIS_ROWS:
        raise RuntimeError("Unexpected analysis_job_postings row count")
    if analysis["record_id"].nunique() != EXPECTED_ANALYSIS_ROWS:
        raise RuntimeError("analysis_job_postings record_id uniqueness failed")

    missing_ids = set(full["record_id"]) - set(analysis["record_id"])
    if missing_ids:
        raise RuntimeError(
            f"{len(missing_ids):,} Patch 28 IDs missing from analysis data"
        )

    review_flag_map = frozen_policy.load_review_flag_map(review_flags)

    flags_by_final_key: dict[str, set[str]] = defaultdict(set)
    for audit_key, flags in review_flag_map.items():
        base_key = frozen_policy.base_key(audit_key)
        final_key = frozen_policy.final_key_from_base(base_key)
        flags_by_final_key[final_key].update(flags)

    raw_rows = []

    for row in full.itertuples(index=False):
        if pd.isna(row.extracted_skills) or str(row.extracted_skills) == "":
            continue

        for line in str(row.extracted_skills).splitlines():
            match = NUMBERED_LINE_RE.match(line)
            if match is None:
                raise RuntimeError(
                    f"Malformed skill line for {row.record_id}: {line!r}"
                )

            skill_raw = match.group(2).strip()
            base_key = frozen_policy.base_key(skill_raw)
            final_key = frozen_policy.final_key_from_base(base_key)

            baseline_bucket = frozen_policy.classify_policy(
                final_key,
                flags_by_final_key.get(final_key, set()),
            )

            final_type = classify_page4_scope(
                final_key,
                baseline_bucket,
            )

            raw_rows.append(
                {
                    "record_id": row.record_id,
                    "skill_raw": skill_raw,
                    "final_skill_key": final_key,
                    "baseline_bucket": baseline_bucket,
                    "final_type": final_type,
                }
            )

    raw_skills = pd.DataFrame(raw_rows)

    if len(raw_skills) != EXPECTED_RAW_SKILL_INSTANCES:
        raise RuntimeError(
            "Raw skill-instance mismatch: "
            f"{len(raw_skills):,}"
        )

    # One analytical occurrence per posting + normalized concept.
    all_terms = raw_skills.drop_duplicates(
        subset=["record_id", "final_skill_key"],
        keep="first",
    ).copy()

    if len(all_terms) != EXPECTED_FINAL_POSTING_SKILL_ROWS:
        raise RuntimeError(
            "Posting-skill dedup count mismatch: "
            f"{len(all_terms):,}"
        )

    # Most common raw wording becomes display label.
    raw_counts = (
        raw_skills.groupby(["final_skill_key", "skill_raw"])
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

    all_terms = all_terms.merge(
        display,
        on="final_skill_key",
        how="left",
        validate="many_to_one",
    )

    capabilities = all_terms.loc[
        all_terms["final_type"].eq("capability"),
        [
            "record_id",
            "final_skill_key",
            "skill_display",
        ],
    ].copy()

    capabilities["skill_posting_flag"] = 1

    noncap = all_terms.loc[
        ~all_terms["final_type"].eq("capability"),
        [
            "record_id",
            "final_skill_key",
            "skill_display",
            "skill_raw",
            "baseline_bucket",
            "final_type",
        ],
    ].copy()

    type_counts = all_terms["final_type"].value_counts()

    expected_type_counts = {
        "capability": EXPECTED_CAPABILITY_ROWS,
        "qualification_like_mention": EXPECTED_BASELINE_QUALIFICATION_ROWS,
        "credential_mention": EXPECTED_NEW_CREDENTIAL_MENTION_ROWS,
        "experience_requirement": EXPECTED_EXPERIENCE_REQUIREMENT_ROWS,
        "excluded_non_skill": EXPECTED_BASELINE_EXCLUDED_ROWS,
    }

    for type_name, expected_count in expected_type_counts.items():
        actual_count = int(type_counts.get(type_name, 0))
        if actual_count != expected_count:
            raise RuntimeError(
                f"{type_name} count mismatch: "
                f"expected={expected_count:,}; actual={actual_count:,}"
            )

    if len(capabilities) != EXPECTED_CAPABILITY_ROWS:
        raise RuntimeError("Final capability row count mismatch")
    if capabilities["final_skill_key"].nunique() != EXPECTED_CAPABILITY_KEYS:
        raise RuntimeError("Final capability unique-key count mismatch")
    if capabilities["record_id"].nunique() != EXPECTED_POSTINGS_WITH_CAPABILITY:
        raise RuntimeError("Final capability posting coverage mismatch")

    if len(noncap) != EXPECTED_NONCAP_AUDIT_ROWS:
        raise RuntimeError("Non-capability audit row count mismatch")

    # Build the final denominator-preserving posting table.
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

    capability_counts = (
        capabilities.groupby("record_id")
        .size()
        .rename("capability_count")
    )

    postings_base = postings_base.join(
        capability_counts,
        on="record_id",
    )
    postings_base["capability_count"] = (
        postings_base["capability_count"].fillna(0).astype(int)
    )
    postings_base["has_capability"] = postings_base["capability_count"] > 0
    postings_base["capability_processing_status"] = postings_base[
        "has_capability"
    ].map(
        {
            True: "has_capability",
            False: "no_included_capability",
        }
    )

    raw_zero = int(
        (postings_base["raw_extracted_skill_count"] == 0).sum()
    )
    zero_capability = int(
        (~postings_base["has_capability"]).sum()
    )

    if raw_zero != EXPECTED_RAW_ZERO_SKILL_POSTINGS:
        raise RuntimeError("Raw zero-skill count mismatch")
    if zero_capability != EXPECTED_ZERO_CAPABILITY_POSTINGS:
        raise RuntimeError("Zero-capability posting count mismatch")

    run_order = full[["record_id", "run_order"]]

    postings_base = (
        postings_base.merge(
            run_order,
            on="record_id",
            how="left",
            validate="one_to_one",
        )
        .sort_values("run_order")
        .drop(columns="run_order")
    )

    capabilities = (
        capabilities.merge(
            run_order,
            on="record_id",
            how="left",
            validate="many_to_one",
        )
        .sort_values(
            ["run_order", "final_skill_key"],
            kind="stable",
        )
        .drop(columns="run_order")
    )

    noncap = (
        noncap.merge(
            run_order,
            on="record_id",
            how="left",
            validate="many_to_one",
        )
        .sort_values(
            ["run_order", "final_type", "final_skill_key"],
            kind="stable",
        )
        .drop(columns="run_order")
    )

    capability_coverage = (
        EXPECTED_POSTINGS_WITH_CAPABILITY / EXPECTED_PROCESSED_POSTINGS * 100
    )

    summary_rows = [
        ("patch28_full_output_sha256", full_sha),
        ("analysis_job_postings_sha256", analysis_sha),
        ("processed_postings", len(postings_base)),
        ("raw_zero_skill_postings", raw_zero),
        ("postings_with_at_least_one_final_capability",
         int(postings_base["has_capability"].sum())),
        ("postings_with_no_final_capability", zero_capability),
        ("final_capability_coverage_pct", round(capability_coverage, 4)),
        ("final_capability_rows", len(capabilities)),
        ("final_capability_unique_keys",
         capabilities["final_skill_key"].nunique()),
        ("qualification_like_mention_rows",
         int(type_counts["qualification_like_mention"])),
        ("new_credential_mention_rows",
         int(type_counts["credential_mention"])),
        ("experience_requirement_rows",
         int(type_counts["experience_requirement"])),
        ("excluded_non_skill_rows",
         int(type_counts["excluded_non_skill"])),
        ("noncapability_audit_rows", len(noncap)),
        (
            "dashboard_scope_note",
            "Page 4 production visuals use capability rows only. Formal "
            "qualification/credential mentions are retained for audit but are "
            "not interpreted as current, preferred, or post-hire requirements.",
        ),
        (
            "experience_policy_note",
            "Candidate-background X experience phrases are excluded from "
            "capability ranking. Clear domain concepts such as customer "
            "experience, user experience, and guest experience management "
            "remain capabilities.",
        ),
        (
            "denominator_note",
            "Capability prevalence denominator remains all 64,283 Patch 28 "
            "processed postings in the active filter context.",
        ),
    ]

    summary = pd.DataFrame(summary_rows, columns=["metric", "value"])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    postings_base.to_csv(POSTINGS_BASE_PATH, index=False)
    capabilities.to_csv(CAPABILITIES_LONG_PATH, index=False)
    noncap.to_csv(NONCAP_AUDIT_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("Final Page 4 capability-scope build complete.")
    print()
    print(f"Posting base rows: {len(postings_base):,}")
    print(
        "Postings with >=1 final capability: "
        f"{int(postings_base['has_capability'].sum()):,}"
    )
    print(f"Postings with no final capability: {zero_capability:,}")
    print(f"Capability coverage: {capability_coverage:.2f}%")
    print()
    print(f"Final capability rows: {len(capabilities):,}")
    print(
        "Final capability unique keys: "
        f"{capabilities['final_skill_key'].nunique():,}"
    )
    print()
    print("Non-capability audit types:")
    print(
        noncap["final_type"]
        .value_counts()
        .to_string()
    )
    print()
    print("Wrote:")
    print(f"  {POSTINGS_BASE_PATH}")
    print(f"  {CAPABILITIES_LONG_PATH}")
    print(f"  {NONCAP_AUDIT_PATH}")
    print(f"  {SUMMARY_PATH}")
    print()
    print(
        "IMPORTANT: These outputs do not claim that a qualification-like "
        "mention is current, preferred, or post-hire. The Tableau Page 4 "
        "capability visuals should use capabilities_long_final only."
    )


if __name__ == "__main__":
    main()
