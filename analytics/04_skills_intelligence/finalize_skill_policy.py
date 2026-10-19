from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "skill_patch28_full_output.csv"
REVIEW_FLAGS_PATH = PROJECT_ROOT / "data" / "processed" / "skill_patch28_review_flags.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

SUMMARY_PATH = OUTPUT_DIR / "skill_patch28_policy_freeze_summary.csv"
TOP100_PATH = OUTPUT_DIR / "skill_patch28_policy_top100.csv"
ALIAS_REPORT_PATH = OUTPUT_DIR / "skill_patch28_alias_report.csv"
EXCLUSIONS_PATH = OUTPUT_DIR / "skill_patch28_policy_exclusions.csv"
MANUAL_RESOLUTION_PATH = OUTPUT_DIR / "skill_patch28_manual_review_resolution.csv"

EXPECTED_INPUT_SHA256 = "ecf749007540fdf8e4c20109125fdbb1aae189956c297912c2600df818b8fe53"
EXPECTED_ROWS = 64_283
EXPECTED_SKILL_INSTANCES = 1_098_031

NUMBERED_LINE_RE = re.compile(r"^\s*(\d+)\.\s+(.*\S)\s*$")

APOSTROPHE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201b": "'",
        "\u2032": "'",
        "\u02bc": "'",
        "`": "'",
        "\u00b4": "'",
    }
)
DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
)

QUALIFICATION_FLAGS = {
    "education_or_degree",
    "license_or_certification",
}

EXCLUDE_FLAGS = {
    "currency_or_pay_fragment",
    "experience_requirement",
    "schedule_or_availability",
    "physical_requirement",
}

# Explicitly reviewed high-impact aliases only.
# We intentionally do NOT apply broad fuzzy matching or semantic similarity.
ALIASES = {
    # Clear spelling / punctuation variants
    "bachelors degree": "bachelor's degree",
    "bachelors' degree": "bachelor's degree",
    "bi-lingual communication": "bilingual communication",
    "california drivers license": "california driver's license",
    "california drivers' license": "california driver's license",
    "cash-handling": "cash handling",
    "conflict-resolution": "conflict resolution",
    "critical-thinking": "critical thinking",
    "cross functional collaboration": "cross-functional collaboration",
    "first-aid certification": "first aid certification",
    "fork lift operation": "forklift operation",
    "fork-lift operation": "forklift operation",
    "high school diploma (or equivalent)": "high school diploma or equivalent",
    "high school diploma or g.e.d": "high school diploma or ged",
    "high school diploma or g.e.d.": "high school diploma or ged",
    "high-school diploma or ged": "high school diploma or ged",
    "inter-personal": "interpersonal",
    "multi-tasking": "multitasking",
    "patient-care": "patient care",
    "problem-solving": "problem solving",
    "record-keeping": "record keeping",
    "recordkeeping": "record keeping",
    "relationship-building": "relationship building",
    "self motivation": "self-motivation",
    "time-management": "time management",
    "timemanagement": "time management",
    "trouble-shooting": "troubleshooting",
    "trouble shooting": "troubleshooting",
    "valid drivers license": "valid driver's license",
    "valid drivers' license": "valid driver's license",
    "preventative maintenance": "preventive maintenance",

    # Clear Microsoft product aliases.
    # Spot-checking raw descriptions confirmed these labels are used as the
    # Microsoft applications, not as generic English words.
    "excel": "microsoft excel",
    "ms excel": "microsoft excel",
    "word": "microsoft word",
    "ms word": "microsoft word",
    "msword": "microsoft word",
    "powerpoint": "microsoft powerpoint",
    "ms powerpoint": "microsoft powerpoint",
    "outlook": "microsoft outlook",
    "ms outlook": "microsoft outlook",
}

# The prior review queue had six ambiguous regex hits.
# Resolve them explicitly instead of letting regex decide.
MANUAL_POLICY_OVERRIDES = {
    "pay rate monitoring": "capability",
    "pay rate negotiation": "capability",
    "pay rate review": "capability",
    "salary range design": "capability",
    "knowledge of www.space-track.org": "capability",
    "accounting processes.salary range informationbased on market data": "excluded",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_typography(skill: str) -> str:
    value = str(skill).translate(APOSTROPHE_TRANSLATION).translate(DASH_TRANSLATION)
    return re.sub(r"\s+", " ", value.strip())


def base_key(skill: str) -> str:
    """
    Frozen conservative base normalization.

    This reproduces Patch 28's own conservative dedup logic plus typography
    normalization. It does not use fuzzy matching.
    """
    value = normalize_typography(skill).lower()
    value = value.replace("&", "and")

    for suffix in (" skills", " skill", " proficiency"):
        if value.endswith(suffix):
            value = value[: -len(suffix)].strip()
            break

    patch28_aliases = {
        "ml": "machine learning",
        "ms office": "microsoft office",
        "microsoft office suite": "microsoft office",
    }
    return patch28_aliases.get(value, value)


def final_key_from_base(key: str) -> str:
    return ALIASES.get(key, key)


def load_review_flag_map(review_flags: pd.DataFrame) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for row in review_flags.itertuples(index=False):
        flags = {
            flag
            for flag in str(row.review_flags).split("|")
            if flag and flag != "nan"
        }
        result[str(row.skill_key)] = flags
    return result


def classify_policy(final_key: str, flags: set[str]) -> str:
    if final_key in MANUAL_POLICY_OVERRIDES:
        return MANUAL_POLICY_OVERRIDES[final_key]

    # Qualifications were explicitly valid under the Patch 28 extraction
    # contract; keep them as a separate analytical type.
    if flags & QUALIFICATION_FLAGS:
        return "qualification"

    # These are explicit non-skill requirement leakage for Page 4.
    if flags & EXCLUDE_FLAGS:
        return "excluded"

    return "capability"


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_PATH}")
    if not REVIEW_FLAGS_PATH.exists():
        raise FileNotFoundError(f"Missing review flags: {REVIEW_FLAGS_PATH}")

    actual_sha = sha256_file(INPUT_PATH)
    if actual_sha != EXPECTED_INPUT_SHA256:
        raise RuntimeError(
            "Patch 28 full-output SHA256 mismatch.\n"
            f"Expected: {EXPECTED_INPUT_SHA256}\n"
            f"Actual:   {actual_sha}"
        )

    full = pd.read_csv(INPUT_PATH)
    review_flags = pd.read_csv(REVIEW_FLAGS_PATH)

    if len(full) != EXPECTED_ROWS or full["record_id"].nunique() != EXPECTED_ROWS:
        raise RuntimeError("Patch 28 posting-universe validation failed")

    review_flag_map = load_review_flag_map(review_flags)

    rows = []
    for row in full.itertuples(index=False):
        if pd.isna(row.extracted_skills) or str(row.extracted_skills) == "":
            continue

        for line in str(row.extracted_skills).splitlines():
            match = NUMBERED_LINE_RE.match(line)
            if match is None:
                raise RuntimeError(
                    f"Malformed numbered skill line for {row.record_id}: {line!r}"
                )

            raw = match.group(2).strip()
            bkey = base_key(raw)
            fkey = final_key_from_base(bkey)

            rows.append(
                {
                    "record_id": row.record_id,
                    "skill_raw": raw,
                    "base_key": bkey,
                    "final_key": fkey,
                }
            )

    skills = pd.DataFrame(rows)

    if len(skills) != EXPECTED_SKILL_INSTANCES:
        raise RuntimeError(
            "Skill-instance count mismatch: "
            f"expected={EXPECTED_SKILL_INSTANCES:,}; actual={len(skills):,}"
        )

    # Propagate review flags from the original audit key to the final alias key.
    flags_by_final_key: dict[str, set[str]] = defaultdict(set)
    for audit_key, flags in review_flag_map.items():
        bkey = base_key(audit_key)
        fkey = final_key_from_base(bkey)
        flags_by_final_key[fkey].update(flags)

    # Exact posting-level prevalence requires one row per record_id + final_key.
    deduped = skills.drop_duplicates(
        subset=["record_id", "final_key"],
        keep="first",
    ).copy()

    # Choose the most frequent raw wording as the display label, except for
    # explicit canonical labels where readability matters.
    raw_counts = (
        skills.groupby(["final_key", "skill_raw"])
        .size()
        .rename("raw_instances")
        .reset_index()
    )

    canonical_display = (
        raw_counts.sort_values(
            ["final_key", "raw_instances", "skill_raw"],
            ascending=[True, False, True],
        )
        .drop_duplicates("final_key")
        [["final_key", "skill_raw"]]
        .rename(columns={"skill_raw": "canonical_display"})
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
        "bachelor's degree": "Bachelor's Degree",
        "high school diploma or ged": "High School Diploma or GED",
    }

    frequency = (
        deduped.groupby("final_key")
        .agg(posting_count=("record_id", "nunique"))
        .reset_index()
        .merge(canonical_display, on="final_key", how="left")
    )

    frequency["canonical_display"] = frequency.apply(
        lambda r: DISPLAY_OVERRIDES.get(
            r["final_key"], r["canonical_display"]
        ),
        axis=1,
    )

    frequency["source_review_flags"] = frequency["final_key"].map(
        lambda k: "|".join(sorted(flags_by_final_key.get(k, set())))
    )
    frequency["policy_bucket"] = frequency["final_key"].map(
        lambda k: classify_policy(k, flags_by_final_key.get(k, set()))
    )
    frequency["include_in_page4"] = (
        frequency["policy_bucket"] != "excluded"
    ).astype(int)
    frequency["posting_prevalence_pct"] = (
        frequency["posting_count"] / EXPECTED_ROWS * 100
    )

    frequency = frequency[
        [
            "final_key",
            "canonical_display",
            "posting_count",
            "posting_prevalence_pct",
            "policy_bucket",
            "include_in_page4",
            "source_review_flags",
        ]
    ].sort_values(
        ["posting_count", "final_key"],
        ascending=[False, True],
    )

    # Build exact alias impact report using posting-level sets.
    base_sets = (
        skills.groupby("base_key")["record_id"]
        .agg(lambda s: set(s))
        .to_dict()
    )
    final_sets = (
        deduped.groupby("final_key")["record_id"]
        .agg(lambda s: set(s))
        .to_dict()
    )

    alias_rows = []
    for source_key, target_key in sorted(ALIASES.items()):
        source_posts = base_sets.get(source_key, set())
        target_before_posts = base_sets.get(target_key, set())
        final_posts = final_sets.get(target_key, set())

        alias_rows.append(
            {
                "source_key": source_key,
                "target_key": target_key,
                "source_posting_count": len(source_posts),
                "target_posting_count_before_alias": len(target_before_posts),
                "source_target_cooccurrence_postings": len(
                    source_posts & target_before_posts
                ),
                "combined_posting_count_after_alias": len(final_posts),
            }
        )

    alias_report = pd.DataFrame(alias_rows)

    exclusions = frequency[
        frequency["policy_bucket"] == "excluded"
    ].copy()

    manual_rows = []
    for key, decision in MANUAL_POLICY_OVERRIDES.items():
        row = frequency[frequency["final_key"] == key]
        posting_count = int(row["posting_count"].iloc[0]) if not row.empty else 0
        manual_rows.append(
            {
                "final_key": key,
                "resolved_policy": decision,
                "posting_count": posting_count,
            }
        )
    manual_resolution = pd.DataFrame(manual_rows)

    included_top100 = frequency[
        frequency["include_in_page4"] == 1
    ].head(100)

    bucket_summary = (
        frequency.groupby("policy_bucket")
        .agg(
            unique_final_keys=("final_key", "size"),
            summed_posting_occurrences=("posting_count", "sum"),
        )
        .reset_index()
    )

    summary_rows = [
        ("input_sha256", actual_sha),
        ("processed_postings", len(full)),
        ("raw_skill_instances", len(skills)),
        ("base_unique_keys", skills["base_key"].nunique()),
        ("final_unique_keys", skills["final_key"].nunique()),
        (
            "unique_key_reduction_from_reviewed_aliases",
            skills["base_key"].nunique() - skills["final_key"].nunique(),
        ),
        (
            "extra_rows_removed_by_record_final_key_dedup",
            len(skills) - len(deduped),
        ),
        ("explicit_alias_rules", len(ALIASES)),
        ("manual_review_resolutions", len(MANUAL_POLICY_OVERRIDES)),
        (
            "policy_note",
            "Qualifications remain valid and separately typed. Explicit "
            "experience/schedule/physical/pay leakage is excluded from Page 4. "
            "Raw Patch 28 values remain unchanged in the source export.",
        ),
        (
            "normalization_note",
            "Conservative base normalization + explicit reviewed aliases only. "
            "No fuzzy matching, embeddings, or broad semantic merging.",
        ),
    ]
    summary = pd.DataFrame(summary_rows, columns=["metric", "value"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    included_top100.to_csv(TOP100_PATH, index=False)
    alias_report.to_csv(ALIAS_REPORT_PATH, index=False)
    exclusions.to_csv(EXCLUSIONS_PATH, index=False)
    manual_resolution.to_csv(MANUAL_RESOLUTION_PATH, index=False)

    print("Patch 28 skill policy freeze preview complete.")
    print()
    print(f"Processed postings: {len(full):,}")
    print(f"Raw skill instances: {len(skills):,}")
    print(f"Base unique keys: {skills['base_key'].nunique():,}")
    print(f"Final unique keys: {skills['final_key'].nunique():,}")
    print(
        "Alias reduction: "
        f"{skills['base_key'].nunique() - skills['final_key'].nunique():,} keys"
    )
    print(
        "Extra posting-skill rows removed after final alias dedup: "
        f"{len(skills) - len(deduped):,}"
    )
    print()
    print("Policy buckets:")
    print(bucket_summary.to_string(index=False))
    print()
    print("Selected high-impact aliases:")
    for key in (
        "problem solving",
        "multitasking",
        "microsoft excel",
        "microsoft word",
        "microsoft powerpoint",
        "microsoft outlook",
        "preventive maintenance",
    ):
        row = frequency[frequency["final_key"] == key]
        if not row.empty:
            print(
                f"  {key}: "
                f"{int(row['posting_count'].iloc[0]):,} postings "
                f"({float(row['posting_prevalence_pct'].iloc[0]):.2f}%)"
            )
    print()
    print("Wrote:")
    for path in (
        SUMMARY_PATH,
        TOP100_PATH,
        ALIAS_REPORT_PATH,
        EXCLUSIONS_PATH,
        MANUAL_RESOLUTION_PATH,
    ):
        print(f"  {path}")
    print()
    print(
        "IMPORTANT: this freezes the proposed analytics policy only. "
        "It still does not create the final skills_long table."
    )


if __name__ == "__main__":
    main()
