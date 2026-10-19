from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

CAPABILITIES_PATH = DATA_DIR / "skill_patch28_capabilities_long_final.csv"
SUPPORT_PATH = DATA_DIR / "skill_patch28_capability_support_final.csv"

CANDIDATES_PATH = DATA_DIR / "skill_patch28_remaining_qualification_candidates.csv"
SUMMARY_PATH = DATA_DIR / "skill_patch28_remaining_qualification_audit_summary.csv"

EXPECTED_CAPABILITY_ROWS = 986_643
EXPECTED_UNIQUE_CAPABILITIES = 255_421


DEGREE_PATTERNS = [
    r"\bjuris doctor(?:ate)?\b",
    r"^(?:j\.?d\.?|jd)$",
    r"\b(?:j\.?d\.?|jd)\s+from\b",
    r"\b(?:j\.?d\.?|jd)\s+or equivalent\b",
    r"\bll\.?b\.?\b",
    r"^(?:mba|m\.b\.a\.)$",
    r"\bmba\s+(?:degree|or equivalent)\b",
    r"\bbachelor(?:'?s)?\b",
    r"\bbachelors\b",
    r"\bmaster(?:'?s)?\s+(?:degree|in|of)\b",
    r"\bmasters\s+(?:degree|in|of)\b",
    r"\bdoctor of medicine\b",
    r"\bdoctor of osteopath(?:y|ic medicine)\b",
    r"\bdoctor of physical therapy\b",
    r"\bdoctor of veterinary medicine\b",
    r"\bdoctorate(?:\s+in\b|$)",
    r"^(?:md|m\.d\.|do|d\.o\.|dpt|d\.p\.t\.|dvm|d\.v\.m\.)$",
    r"\bmd or do\b",
    r"\bmd/do\b",
    r"\bmd/phd\b",
    r"^(?:phd|ph\.d\.|ph\.d)$",
    r"\bphd\s+(?:degree|in|or)\b",
    r"\b(?:ba|bs|bsc|bsn)\s+(?:in|degree)\b",
    r"\b(?:ba/bs|bs/ba|bsc/ba|bs/ms|aa/ba|aa/as)\b",
    r"\bassociate(?:'?s)? degree\b",
    r"\bhigh school diploma\b",
    r"\bged\b",
]

CERTIFICATION_PATTERNS = [
    r"\bcertification(?:s)?\s*(?:\([^)]+\))?$",
    r"\b(?:cpr|bls|acls|pals|ase|aws|cisco|hvac|iicrc|icc|oem|asq|pmp)"
    r"\s+certification(?:s)?\b",
]

LICENSE_PATTERNS = [
    r"\bvalid (?:driver'?s?|driving) licen[cs]e\b",
    r"\b(?:state|nursing|medical|bar|insurance|finra|professional|teaching|driver|drivers)"
    r"\s+licen[cs]e(?:s)?\b",
    r"\blicen[cs]e(?:s)?\s+required\b",
]

CREDENTIAL_PATTERNS = [
    r"\b(?:teaching|pps|intern|merchant mariner)\s+credential\b",
    r"\btransportation worker identification credential",
    r"\bcredential\s+in\s+",
    r"\bcredential eligibility\b",
]


def matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def classify_candidate(key: str) -> tuple[str, str] | None:
    text = str(key).strip().lower()

    # Guard against known software/product names that contain degree-like tokens.
    if "jd edwards" in text:
        return None

    if matches_any(text, DEGREE_PATTERNS):
        return (
            "clear_degree_qualification",
            "Explicit degree/academic-qualification wording",
        )

    if matches_any(text, CERTIFICATION_PATTERNS):
        return (
            "clear_certification_qualification",
            "Phrase denotes a certification/certifications rather than a capability",
        )

    if matches_any(text, LICENSE_PATTERNS):
        return (
            "clear_license_qualification",
            "Phrase denotes a license/licensure requirement or status",
        )

    if matches_any(text, CREDENTIAL_PATTERNS):
        return (
            "clear_credential_qualification",
            "Phrase denotes a formal credential rather than credentialing work",
        )

    return None


def main() -> None:
    for path in (CAPABILITIES_PATH, SUPPORT_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    capabilities = pd.read_csv(CAPABILITIES_PATH, low_memory=False)
    support = pd.read_csv(SUPPORT_PATH, low_memory=False)

    if len(capabilities) != EXPECTED_CAPABILITY_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_CAPABILITY_ROWS:,} capability rows, "
            f"found {len(capabilities):,}"
        )

    if support["final_skill_key"].nunique() != EXPECTED_UNIQUE_CAPABILITIES:
        raise RuntimeError(
            "Unexpected unique capability count in support table"
        )

    candidate_records = []

    for row in support.itertuples(index=False):
        classified = classify_candidate(row.final_skill_key)
        if classified is None:
            continue

        candidate_group, reason = classified

        candidate_records.append(
            {
                "final_skill_key": row.final_skill_key,
                "skill_display": row.skill_display,
                "posting_count": int(row.posting_count),
                "overall_prevalence_pct": float(row.overall_prevalence_pct),
                "candidate_group": candidate_group,
                "confidence": "high",
                "reason": reason,
                "selector_ge_100": int(row.posting_count) >= 100,
                "selector_ge_250": int(row.posting_count) >= 250,
                "selector_ge_500": int(row.posting_count) >= 500,
            }
        )

    candidates = pd.DataFrame(candidate_records)

    if candidates.empty:
        raise RuntimeError("No qualification-leakage candidates found")

    candidate_keys = set(candidates["final_skill_key"])

    affected_long = capabilities.loc[
        capabilities["final_skill_key"].isin(candidate_keys)
    ]

    distinct_postings_affected = affected_long["record_id"].nunique()
    candidate_posting_skill_rows = len(affected_long)

    # Exact category counts from the candidate-key table.
    by_group = (
        candidates.groupby("candidate_group", as_index=False)
        .agg(
            candidate_keys=("final_skill_key", "nunique"),
            summed_key_posting_support=("posting_count", "sum"),
        )
        .sort_values("candidate_keys", ascending=False)
    )

    summary_rows = [
        ("final_capability_rows_input", len(capabilities)),
        ("unique_final_capabilities_input", support["final_skill_key"].nunique()),
        ("high_confidence_candidate_keys", candidates["final_skill_key"].nunique()),
        ("candidate_posting_capability_rows", candidate_posting_skill_rows),
        ("distinct_postings_affected", distinct_postings_affected),
        ("candidate_keys_with_support_ge_100", int(candidates["selector_ge_100"].sum())),
        ("candidate_keys_with_support_ge_250", int(candidates["selector_ge_250"].sum())),
        ("candidate_keys_with_support_ge_500", int(candidates["selector_ge_500"].sum())),
        (
            "audit_scope_note",
            "This audit flags only high-confidence residual formal-qualification phrases "
            "still present in the capability table. It intentionally does not blanket-"
            "exclude words such as credentialing, credential management, software licenses, "
            "or certification process because those can describe actual work capabilities.",
        ),
        (
            "important_note",
            "No production file is modified by this audit.",
        ),
    ]

    for row in by_group.itertuples(index=False):
        summary_rows.append(
            (
                f"{row.candidate_group}_keys",
                int(row.candidate_keys),
            )
        )
        summary_rows.append(
            (
                f"{row.candidate_group}_summed_key_posting_support",
                int(row.summed_key_posting_support),
            )
        )

    summary = pd.DataFrame(summary_rows, columns=["metric", "value"])

    candidates = candidates.sort_values(
        ["posting_count", "candidate_group", "final_skill_key"],
        ascending=[False, True, True],
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(CANDIDATES_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("Residual qualification-leakage audit complete.")
    print()
    print(f"High-confidence candidate keys:      {len(candidates):,}")
    print(f"Candidate posting-capability rows:   {candidate_posting_skill_rows:,}")
    print(f"Distinct postings affected:          {distinct_postings_affected:,}")
    print()
    print("Candidate groups:")
    print(by_group.to_string(index=False))
    print()
    print("Candidates with selector-level support:")
    print(
        f"  >=100 postings: {int(candidates['selector_ge_100'].sum()):,}"
    )
    print(
        f"  >=250 postings: {int(candidates['selector_ge_250'].sum()):,}"
    )
    print(
        f"  >=500 postings: {int(candidates['selector_ge_500'].sum()):,}"
    )
    print()
    print("Top 30 candidates:")
    print(
        candidates.head(30)[
            [
                "skill_display",
                "posting_count",
                "candidate_group",
            ]
        ].to_string(index=False)
    )
    print()
    print("Wrote:")
    print(f"  {CANDIDATES_PATH}")
    print(f"  {SUMMARY_PATH}")
    print()
    print(
        "IMPORTANT: Audit only. Do not remove rows until the candidate list is reviewed."
    )


if __name__ == "__main__":
    main()
