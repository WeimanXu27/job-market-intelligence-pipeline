from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

CANDIDATES_PATH = DATA_DIR / "skill_patch28_remaining_qualification_candidates.csv"
CAPABILITIES_PATH = DATA_DIR / "skill_patch28_capabilities_long_final.csv"
RECOVERED_PATH = DATA_DIR / "recovered_job_postings.csv"

OUTPUT_CONTEXT_PATH = DATA_DIR / "skill_patch28_cert_license_context_review.csv"
OUTPUT_TERM_SUMMARY_PATH = DATA_DIR / "skill_patch28_cert_license_term_summary.csv"
OUTPUT_SUMMARY_PATH = DATA_DIR / "skill_patch28_cert_license_context_summary.csv"

EXPECTED_REVIEW_KEYS = 266
EXPECTED_REVIEW_ROWS = 607


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def split_local_units(text: str) -> list[str]:
    """
    Split a job description into reasonably local units.
    Keeps bullets / sentence-like chunks separate so nearby requirements
    do not contaminate each other as much as a broad character window.
    """
    text = str(text or "").replace("\r", "\n")
    rough = re.split(r"(?:\n+|[•●▪◦]\s*|(?<=[.!?;])\s+)", text)
    return [normalize_space(x) for x in rough if normalize_space(x)]


def token_pattern(phrase: str) -> re.Pattern | None:
    tokens = re.findall(r"[A-Za-z0-9]+", str(phrase))
    if not tokens:
        return None
    joined = r"[\W_]+".join(re.escape(tok) for tok in tokens)
    return re.compile(joined, flags=re.IGNORECASE)


def find_local_context(description: str, phrase: str) -> tuple[str, str]:
    units = split_local_units(description)

    phrase_clean = normalize_space(phrase)
    exact_re = re.compile(re.escape(phrase_clean), flags=re.IGNORECASE)

    for unit in units:
        if exact_re.search(unit):
            return unit, "exact_phrase"

    pattern = token_pattern(phrase_clean)
    if pattern is not None:
        for unit in units:
            if pattern.search(unit):
                return unit, "token_fallback"

    # Last resort: compact character window around first meaningful token.
    tokens = re.findall(r"[A-Za-z0-9]+", phrase_clean)
    if tokens:
        first = re.escape(tokens[0])
        match = re.search(first, description, flags=re.IGNORECASE)
        if match:
            start = max(0, match.start() - 180)
            end = min(len(description), match.end() + 280)
            return normalize_space(description[start:end]), "token_window"

    return "", "not_found"


def signal_label(context: str) -> str:
    """
    Review aid only. This is deliberately not a production classifier.
    """
    text = str(context or "").lower()

    candidate_signal = bool(
        re.search(
            r"\b(required|preferred|must have|must possess|possess(?:ion)? of|"
            r"current|active|valid|certified|licensed|licensure|eligible for|"
            r"qualification|qualifications)\b",
            text,
        )
    )

    future_signal = bool(
        re.search(
            r"\b(obtain|acquire|earn|secure)\b.{0,80}"
            r"\b(within|after hire|upon hire|by|months?|days?|years?)\b",
            text,
        )
        or re.search(
            r"\bwithin\s+\d+\s+(?:days?|months?|years?)\b",
            text,
        )
    )

    work_process_signal = bool(
        re.search(
            r"\b(manage|management|maintain|maintenance|renew|renewal|"
            r"process|processing|track|tracking|verify|verification|"
            r"reimburse|reimbursement|administer|administration|"
            r"support|coordinate|coordination|documentation|audit|auditing)\b",
            text,
        )
    )

    if candidate_signal and work_process_signal:
        return "mixed_candidate_and_work_process"
    if candidate_signal:
        return "candidate_qualification_signal"
    if future_signal and work_process_signal:
        return "mixed_future_and_work_process"
    if future_signal:
        return "future_or_posthire_signal"
    if work_process_signal:
        return "work_process_signal"
    return "no_clear_signal"


def main() -> None:
    for path in (CANDIDATES_PATH, CAPABILITIES_PATH, RECOVERED_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    candidates = pd.read_csv(CANDIDATES_PATH, low_memory=False)

    review_groups = {
        "clear_certification_qualification",
        "clear_license_qualification",
    }

    review_terms = candidates.loc[
        candidates["candidate_group"].isin(review_groups),
        [
            "final_skill_key",
            "skill_display",
            "posting_count",
            "candidate_group",
        ],
    ].copy()

    if review_terms["final_skill_key"].nunique() != EXPECTED_REVIEW_KEYS:
        raise RuntimeError(
            "Unexpected certification/license key count: "
            f"{review_terms['final_skill_key'].nunique():,}"
        )

    capabilities = pd.read_csv(
        CAPABILITIES_PATH,
        usecols=["record_id", "final_skill_key", "skill_display"],
        low_memory=False,
    )

    occurrences = capabilities.merge(
        review_terms[
            ["final_skill_key", "posting_count", "candidate_group"]
        ],
        on="final_skill_key",
        how="inner",
        validate="many_to_one",
    )

    if len(occurrences) != EXPECTED_REVIEW_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_REVIEW_ROWS:,} review occurrences, "
            f"found {len(occurrences):,}"
        )

    # Find description column without requiring the file to use one exact name.
    header = pd.read_csv(RECOVERED_PATH, nrows=0)
    description_candidates = [
        "description",
        "job_description",
        "Description",
        "jobDescription",
    ]
    description_col = next(
        (col for col in description_candidates if col in header.columns),
        None,
    )
    if description_col is None:
        raise RuntimeError(
            "Could not find a description column in recovered_job_postings.csv. "
            f"Available columns include: {list(header.columns)[:30]}"
        )

    recovered = pd.read_csv(
        RECOVERED_PATH,
        usecols=["record_id", description_col],
        low_memory=False,
    )

    if recovered["record_id"].duplicated().any():
        raise RuntimeError(
            "recovered_job_postings.csv contains duplicate record_id values"
        )

    review = occurrences.merge(
        recovered,
        on="record_id",
        how="left",
        validate="many_to_one",
    )

    missing_description = review[description_col].isna().sum()

    contexts = review.apply(
        lambda row: find_local_context(
            "" if pd.isna(row[description_col]) else str(row[description_col]),
            row["skill_display"],
        ),
        axis=1,
        result_type="expand",
    )
    contexts.columns = ["local_context", "match_method"]

    review = pd.concat([review, contexts], axis=1)
    review["review_signal"] = review["local_context"].map(signal_label)

    review = review[
        [
            "record_id",
            "final_skill_key",
            "skill_display",
            "candidate_group",
            "posting_count",
            "match_method",
            "review_signal",
            "local_context",
        ]
    ].sort_values(
        ["posting_count", "final_skill_key", "record_id"],
        ascending=[False, True, True],
    )

    term_summary = (
        review.groupby(
            [
                "final_skill_key",
                "skill_display",
                "candidate_group",
                "posting_count",
            ],
            as_index=False,
        )
        .agg(
            reviewed_occurrences=("record_id", "nunique"),
            exact_phrase_matches=(
                "match_method",
                lambda s: int((s == "exact_phrase").sum()),
            ),
            token_fallback_matches=(
                "match_method",
                lambda s: int((s == "token_fallback").sum()),
            ),
            token_window_matches=(
                "match_method",
                lambda s: int((s == "token_window").sum()),
            ),
            not_found=(
                "match_method",
                lambda s: int((s == "not_found").sum()),
            ),
            candidate_qualification_signal=(
                "review_signal",
                lambda s: int(
                    (s == "candidate_qualification_signal").sum()
                ),
            ),
            work_process_signal=(
                "review_signal",
                lambda s: int((s == "work_process_signal").sum()),
            ),
            mixed_signal=(
                "review_signal",
                lambda s: int(
                    s.isin(
                        [
                            "mixed_candidate_and_work_process",
                            "mixed_future_and_work_process",
                        ]
                    ).sum()
                ),
            ),
            future_or_posthire_signal=(
                "review_signal",
                lambda s: int(
                    (s == "future_or_posthire_signal").sum()
                ),
            ),
            no_clear_signal=(
                "review_signal",
                lambda s: int((s == "no_clear_signal").sum()),
            ),
        )
        .sort_values(
            ["posting_count", "final_skill_key"],
            ascending=[False, True],
        )
    )

    summary_rows = [
        ("review_groups", "certification + license candidates"),
        ("review_candidate_keys", review_terms["final_skill_key"].nunique()),
        ("review_posting_key_occurrences", len(review)),
        ("distinct_postings_in_review", review["record_id"].nunique()),
        ("missing_descriptions", int(missing_description)),
        ("exact_phrase_matches", int((review["match_method"] == "exact_phrase").sum())),
        ("token_fallback_matches", int((review["match_method"] == "token_fallback").sum())),
        ("token_window_matches", int((review["match_method"] == "token_window").sum())),
        ("not_found", int((review["match_method"] == "not_found").sum())),
        (
            "candidate_qualification_signal_rows",
            int(
                (
                    review["review_signal"]
                    == "candidate_qualification_signal"
                ).sum()
            ),
        ),
        (
            "work_process_signal_rows",
            int((review["review_signal"] == "work_process_signal").sum()),
        ),
        (
            "mixed_signal_rows",
            int(
                review["review_signal"].isin(
                    [
                        "mixed_candidate_and_work_process",
                        "mixed_future_and_work_process",
                    ]
                ).sum()
            ),
        ),
        (
            "future_or_posthire_signal_rows",
            int(
                (
                    review["review_signal"]
                    == "future_or_posthire_signal"
                ).sum()
            ),
        ),
        (
            "no_clear_signal_rows",
            int((review["review_signal"] == "no_clear_signal").sum()),
        ),
        (
            "important_note",
            "review_signal is only a review aid. This script does not decide "
            "whether a term is removed from the capability table.",
        ),
    ]

    summary = pd.DataFrame(summary_rows, columns=["metric", "value"])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    review.to_csv(OUTPUT_CONTEXT_PATH, index=False)
    term_summary.to_csv(OUTPUT_TERM_SUMMARY_PATH, index=False)
    summary.to_csv(OUTPUT_SUMMARY_PATH, index=False)

    print("Certification/license context audit complete.")
    print()
    print(f"Candidate keys reviewed:       {review_terms['final_skill_key'].nunique():,}")
    print(f"Posting-key occurrences:       {len(review):,}")
    print(f"Distinct postings:             {review['record_id'].nunique():,}")
    print(f"Missing descriptions:          {missing_description:,}")
    print()
    print("Context match methods:")
    print(review["match_method"].value_counts().to_string())
    print()
    print("Review signals:")
    print(review["review_signal"].value_counts().to_string())
    print()
    print("Wrote:")
    print(f"  {OUTPUT_CONTEXT_PATH}")
    print(f"  {OUTPUT_TERM_SUMMARY_PATH}")
    print(f"  {OUTPUT_SUMMARY_PATH}")
    print()
    print(
        "IMPORTANT: Audit only. Degree and formal-credential candidates are "
        "not re-reviewed here; this step focuses on the ambiguous certification/"
        "license groups before any final cleanup."
    )


if __name__ == "__main__":
    main()
