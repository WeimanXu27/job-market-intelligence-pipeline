from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "skill_patch28_full_output.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

SUMMARY_PATH = OUTPUT_DIR / "skill_patch28_vocabulary_audit_summary.csv"
FREQUENCY_PATH = OUTPUT_DIR / "skill_patch28_vocabulary_frequency.csv"
CASE_VARIANTS_PATH = OUTPUT_DIR / "skill_patch28_case_variants.csv"
FORMAT_ISSUES_PATH = OUTPUT_DIR / "skill_patch28_format_issues.csv"
REVIEW_FLAGS_PATH = OUTPUT_DIR / "skill_patch28_review_flags.csv"

EXPECTED_ROWS = 64_283
EXPECTED_UNIQUE_RECORD_IDS = 64_283
EXPECTED_ZERO_SKILL = 1_730
EXPECTED_TOTAL_SKILL_INSTANCES = 1_098_031
EXPECTED_OUTPUT_SHA256 = "ecf749007540fdf8e4c20109125fdbb1aae189956c297912c2600df818b8fe53"

NUMBERED_LINE_RE = re.compile(r"^\s*(\d+)\.(\s+)(.*)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_numbered_skills(record_id: str, text: object, expected_count: int):
    if pd.isna(text):
        lines = []
    else:
        lines = str(text).splitlines()

    parsed = []
    issues = []

    for position, line in enumerate(lines, start=1):
        match = NUMBERED_LINE_RE.match(line)
        if match is None:
            issues.append(
                {
                    "record_id": record_id,
                    "issue_type": "malformed_numbered_line",
                    "line_position": position,
                    "detail": line,
                }
            )
            continue

        line_number = int(match.group(1))
        separator_whitespace = match.group(2)
        skill_raw_exact = match.group(3)
        skill_raw = skill_raw_exact.strip()

        if len(separator_whitespace) != 1 or separator_whitespace != " ":
            issues.append(
                {
                    "record_id": record_id,
                    "issue_type": "nonstandard_separator_whitespace",
                    "line_position": position,
                    "detail": repr(separator_whitespace),
                }
            )

        if skill_raw_exact != skill_raw_exact.rstrip():
            issues.append(
                {
                    "record_id": record_id,
                    "issue_type": "trailing_whitespace_in_skill",
                    "line_position": position,
                    "detail": repr(skill_raw_exact),
                }
            )

        if line_number != position:
            issues.append(
                {
                    "record_id": record_id,
                    "issue_type": "non_sequential_numbering",
                    "line_position": position,
                    "detail": f"found={line_number}; expected={position}",
                }
            )

        if not skill_raw:
            issues.append(
                {
                    "record_id": record_id,
                    "issue_type": "empty_skill_after_numbering",
                    "line_position": position,
                    "detail": line,
                }
            )
            continue

        parsed.append((line_number, skill_raw_exact, skill_raw))

    if len(parsed) != int(expected_count):
        issues.append(
            {
                "record_id": record_id,
                "issue_type": "skill_count_mismatch",
                "line_position": None,
                "detail": f"parsed={len(parsed)}; expected={int(expected_count)}",
            }
        )

    return parsed, issues


def add_review_flags(skill_key: str) -> list[str]:
    """
    Review-only heuristics.

    These flags do NOT remove, rewrite, or semantically merge any skill.
    They only identify vocabulary subsets worth inspecting before Page 4.
    """
    flags = []

    if re.search(r"https?://|www\.", skill_key):
        flags.append("possible_url_artifact")

    if re.search(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", skill_key):
        flags.append("possible_email_artifact")

    if re.search(r"[$€£]\s*\d", skill_key):
        flags.append("currency_or_pay_fragment")

    if re.search(r"\b(?:salary range|pay rate|base salary|hourly wage)\b", skill_key):
        flags.append("pay_phrase")

    if re.search(r"\b\d+\+?\s*(?:years?|yrs?)\b|\byears? of experience\b", skill_key):
        flags.append("experience_requirement")

    if re.search(
        r"\b(?:bachelor(?:'s|’s)?|master(?:'s|’s)?|associate(?:'s|’s)?|degree|diploma|ged)\b",
        skill_key,
    ):
        flags.append("education_or_degree")

    if re.search(
        r"\b(?:license|licensed|licensure|certification|certified|certificate)\b",
        skill_key,
    ):
        flags.append("license_or_certification")

    if re.search(
        r"\b(?:full[- ]time|part[- ]time|weekend availability|evening availability|"
        r"night shift|day shift|shift work|overtime availability)\b",
        skill_key,
    ):
        flags.append("schedule_or_availability")

    if re.search(
        r"\b(?:heavy lifting|standing for (?:long|extended) periods|"
        r"ability to lift|physical ability)\b",
        skill_key,
    ):
        flags.append("physical_requirement")

    return flags


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    actual_sha256 = sha256_file(INPUT_PATH)
    if actual_sha256 != EXPECTED_OUTPUT_SHA256:
        raise RuntimeError(
            "Input SHA256 does not match the frozen Patch 28 full export.\n"
            f"Expected: {EXPECTED_OUTPUT_SHA256}\n"
            f"Actual:   {actual_sha256}"
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "run_order",
        "record_id",
        "pool",
        "extracted_skills",
        "extracted_skill_count",
        "has_extracted_skill",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise RuntimeError(f"Missing required columns: {sorted(missing_columns)}")

    if len(df) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} rows, found {len(df)}")

    if df["record_id"].nunique() != EXPECTED_UNIQUE_RECORD_IDS:
        raise RuntimeError(
            "record_id uniqueness failed: "
            f"{df['record_id'].nunique()} unique IDs"
        )

    if df["record_id"].duplicated().any():
        raise RuntimeError("Duplicate record_id values detected")

    expected_run_order = list(range(1, EXPECTED_ROWS + 1))
    if df["run_order"].tolist() != expected_run_order:
        raise RuntimeError("run_order is not exactly 1..64,283 in order")

    zero_skill_count = int((df["extracted_skill_count"] == 0).sum())
    if zero_skill_count != EXPECTED_ZERO_SKILL:
        raise RuntimeError(
            f"Expected {EXPECTED_ZERO_SKILL} zero-skill records, found {zero_skill_count}"
        )

    long_rows = []
    format_issues = []
    within_post_casefold_duplicates = []

    for row in df.itertuples(index=False):
        parsed, issues = parse_numbered_skills(
            record_id=row.record_id,
            text=row.extracted_skills,
            expected_count=row.extracted_skill_count,
        )
        format_issues.extend(issues)

        keys_in_post = []
        for line_number, skill_raw_exact, skill_raw in parsed:
            skill_whitespace_normalized = re.sub(r"\s+", " ", skill_raw)
            skill_key = skill_whitespace_normalized.casefold()

            long_rows.append(
                {
                    "record_id": row.record_id,
                    "pool": row.pool,
                    "line_number": line_number,
                    "skill_raw_exact": skill_raw_exact,
                    "skill_raw": skill_raw,
                    "skill_whitespace_normalized": skill_whitespace_normalized,
                    "skill_key": skill_key,
                }
            )
            keys_in_post.append(skill_key)

        counts = Counter(keys_in_post)
        duplicates = sorted(key for key, count in counts.items() if count > 1)
        if duplicates:
            within_post_casefold_duplicates.append(
                {
                    "record_id": row.record_id,
                    "duplicate_skill_keys": json.dumps(
                        duplicates, ensure_ascii=False
                    ),
                }
            )

    skills = pd.DataFrame(long_rows)

    if len(skills) != EXPECTED_TOTAL_SKILL_INSTANCES:
        raise RuntimeError(
            "Total parsed skill instances mismatch: "
            f"expected={EXPECTED_TOTAL_SKILL_INSTANCES}; actual={len(skills)}"
        )

    format_df = pd.DataFrame(
        format_issues,
        columns=["record_id", "issue_type", "line_position", "detail"],
    )

    # Frequency table at the safe normalization level:
    # trim + collapse whitespace + casefold only.
    raw_variant_counts = (
        skills.groupby(["skill_key", "skill_raw"])
        .size()
        .rename("raw_variant_instances")
        .reset_index()
    )

    canonical = (
        raw_variant_counts.sort_values(
            ["skill_key", "raw_variant_instances", "skill_raw"],
            ascending=[True, False, True],
        )
        .drop_duplicates("skill_key")
        .rename(columns={"skill_raw": "canonical_display"})
        [["skill_key", "canonical_display"]]
    )

    frequency = (
        skills.groupby("skill_key")
        .agg(
            posting_count=("record_id", "nunique"),
            instance_count=("record_id", "size"),
            raw_variant_count=("skill_raw", "nunique"),
        )
        .reset_index()
        .merge(canonical, on="skill_key", how="left")
    )

    frequency["posting_prevalence_pct"] = (
        frequency["posting_count"] / EXPECTED_ROWS * 100
    )

    frequency = frequency[
        [
            "skill_key",
            "canonical_display",
            "posting_count",
            "posting_prevalence_pct",
            "instance_count",
            "raw_variant_count",
        ]
    ].sort_values(
        ["posting_count", "instance_count", "skill_key"],
        ascending=[False, False, True],
    )

    # One row per observed raw variant is intentionally faster and easier to audit
    # than packing all variants into JSON.
    case_variants = (
        raw_variant_counts.merge(
            frequency[
                ["skill_key", "canonical_display", "posting_count", "raw_variant_count"]
            ],
            on="skill_key",
            how="left",
        )
    )
    case_variants = case_variants[
        case_variants["raw_variant_count"] > 1
    ].sort_values(
        ["posting_count", "skill_key", "raw_variant_instances", "skill_raw"],
        ascending=[False, True, False, True],
    )

    # Review-only semantic / artifact flags. Nothing is automatically excluded.
    review = frequency.copy()
    review["review_flags"] = review["skill_key"].map(add_review_flags)
    review = review[review["review_flags"].map(bool)].copy()
    review["review_flags"] = review["review_flags"].map(
        lambda values: "|".join(values)
    )
    review = review.sort_values(
        ["posting_count", "skill_key"], ascending=[False, True]
    )

    # Additional audit statistics.
    leading_or_trailing_ws = int(
        (skills["skill_raw_exact"] != skills["skill_raw_exact"].str.rstrip()).sum()
        + (format_df["issue_type"].eq("nonstandard_separator_whitespace").sum() if not format_df.empty else 0)
    )
    repeated_internal_ws = int(
        skills["skill_raw"].str.contains(r"\s{2,}", regex=True).sum()
    )
    unique_raw = int(skills["skill_raw"].nunique())
    unique_safe_keys = int(skills["skill_key"].nunique())
    singleton_keys = int((frequency["posting_count"] == 1).sum())
    keys_le_2 = int((frequency["posting_count"] <= 2).sum())
    keys_le_5 = int((frequency["posting_count"] <= 5).sum())
    keys_le_10 = int((frequency["posting_count"] <= 10).sum())
    keys_ge_100 = int((frequency["posting_count"] >= 100).sum())
    keys_ge_1000 = int((frequency["posting_count"] >= 1000).sum())

    singleton_instances = int(
        frequency.loc[frequency["posting_count"] == 1, "instance_count"].sum()
    )

    summary_rows = [
        ("input_sha256", actual_sha256),
        ("processed_postings", len(df)),
        ("unique_record_ids", df["record_id"].nunique()),
        ("postings_with_at_least_one_skill", int((df["extracted_skill_count"] > 0).sum())),
        ("zero_skill_records", zero_skill_count),
        ("total_skill_instances", len(skills)),
        ("unique_raw_skill_strings", unique_raw),
        ("unique_safe_skill_keys", unique_safe_keys),
        ("safe_key_reduction_from_case_whitespace", unique_raw - unique_safe_keys),
        ("safe_key_reduction_pct", round((unique_raw - unique_safe_keys) / unique_raw * 100, 4)),
        ("case_variant_groups", int((frequency["raw_variant_count"] > 1).sum())),
        ("leading_or_trailing_whitespace_instances", leading_or_trailing_ws),
        ("repeated_internal_whitespace_instances", repeated_internal_ws),
        ("format_issue_rows", len(format_df)),
        ("within_post_casefold_duplicate_postings", len(within_post_casefold_duplicates)),
        ("singleton_skill_keys", singleton_keys),
        ("singleton_pct_of_vocabulary", round(singleton_keys / unique_safe_keys * 100, 4)),
        ("singleton_pct_of_instances", round(singleton_instances / len(skills) * 100, 4)),
        ("skill_keys_posting_count_le_2", keys_le_2),
        ("skill_keys_posting_count_le_5", keys_le_5),
        ("skill_keys_posting_count_le_10", keys_le_10),
        ("skill_keys_posting_count_ge_100", keys_ge_100),
        ("skill_keys_posting_count_ge_1000", keys_ge_1000),
        ("review_flagged_skill_keys", len(review)),
        (
            "normalization_policy_note",
            "Audit only. No semantic merge or removal is applied. "
            "Safe key = trim + whitespace collapse + casefold. Raw text is preserved for whitespace audit.",
        ),
        (
            "denominator_note",
            "posting_prevalence_pct uses all 64,283 Patch-28 processed postings.",
        ),
    ]

    summary = pd.DataFrame(summary_rows, columns=["metric", "value"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    frequency.to_csv(FREQUENCY_PATH, index=False)
    case_variants.to_csv(CASE_VARIANTS_PATH, index=False)
    format_df.to_csv(FORMAT_ISSUES_PATH, index=False)
    review.to_csv(REVIEW_FLAGS_PATH, index=False)

    print("Patch 28 skill vocabulary audit complete.")
    print()
    print(f"Input: {INPUT_PATH}")
    print(f"SHA256: {actual_sha256}")
    print()
    print(f"Processed postings: {len(df):,}")
    print(f"Postings with >=1 skill: {(df['extracted_skill_count'] > 0).sum():,}")
    print(f"Zero-skill postings: {zero_skill_count:,}")
    print(f"Total skill instances: {len(skills):,}")
    print(f"Unique raw skill strings: {unique_raw:,}")
    print(f"Unique safe skill keys: {unique_safe_keys:,}")
    print(f"Case-variant groups: {(frequency['raw_variant_count'] > 1).sum():,}")
    print(f"Format issues: {len(format_df):,}")
    print(f"Within-post casefold duplicates: {len(within_post_casefold_duplicates):,}")
    print(f"Singleton keys: {singleton_keys:,} ({singleton_keys / unique_safe_keys * 100:.2f}% of vocabulary)")
    print()
    print("Wrote:")
    print(f"  {SUMMARY_PATH}")
    print(f"  {FREQUENCY_PATH}")
    print(f"  {CASE_VARIANTS_PATH}")
    print(f"  {FORMAT_ISSUES_PATH}")
    print(f"  {REVIEW_FLAGS_PATH}")
    print()
    print("IMPORTANT: review flags are diagnostic only; no rows were removed or merged.")


if __name__ == "__main__":
    main()
