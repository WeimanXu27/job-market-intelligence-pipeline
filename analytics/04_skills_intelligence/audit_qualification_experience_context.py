from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

FULL_OUTPUT_PATH = DATA_DIR / "skill_patch28_full_output.csv"
RECOVERED_POSTINGS_PATH = DATA_DIR / "recovered_job_postings.csv"

OUTPUT_PATH = DATA_DIR / "skill_patch28_context_audit_v2_review_sample.csv"
SUMMARY_PATH = DATA_DIR / "skill_patch28_context_audit_v2_summary.csv"

EXPECTED_PATCH28_ROWS = 64_283

NUMBERED_LINE_RE = re.compile(r"^\s*(\d+)\.\s+(.*\S)\s*$")

QUALIFICATION_TERM_RE = re.compile(
    r"\b(?:credential|credentials|license|licenses|licensed|licensure|"
    r"certification|certifications|certified|certificate|certificates|"
    r"degree|degrees|diploma|diplomas|ged)\b",
    flags=re.IGNORECASE,
)

EXPERIENCE_TERM_RE = re.compile(r"\bexperience\b", flags=re.IGNORECASE)

# Split fairly aggressively around sentence / bullet boundaries so that a
# post-hire condition for one credential does not contaminate another nearby
# credential in the same job description.
CLAUSE_SPLIT_RE = re.compile(
    r"(?:[\r\n]+|[•·▪●◦]\s*|\s+\*\s+|(?<=[.!?;])\s+)"
)

POST_HIRE_RE = re.compile(
    r"\b(?:"
    r"ability to obtain|able to obtain|must obtain|must earn|must acquire|"
    r"eligible to obtain|"
    r"obtain(?:ed)?\s+within|earn(?:ed)?\s+within|acquire(?:d)?\s+within|"
    r"within\s+\d+\s*(?:days?|weeks?|months?|years?)\s+"
    r"(?:of|after|from)\s+(?:hire|hiring|employment|start|appointment)|"
    r"within\s+(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:days?|weeks?|months?|years?)\s+"
    r"(?:of|after|from)\s+(?:hire|hiring|employment|start|appointment)|"
    r"after\s+(?:hire|hiring|employment|start|appointment)|"
    r"post[- ]hire"
    r")\b",
    flags=re.IGNORECASE,
)

PREFERRED_RE = re.compile(
    r"\b(?:preferred|preferably|desired|desirable|a plus|nice to have)\b",
    flags=re.IGNORECASE,
)

ENTRY_RE = re.compile(
    r"\b(?:"
    r"current|currently|valid|active|unrestricted|"
    r"required|must have|must hold|must possess|possession of|"
    r"licensed in|certified in|certified as"
    r")\b",
    flags=re.IGNORECASE,
)

DOMAIN_EXPERIENCE_EXCEPTIONS = {
    "customer experience",
    "user experience",
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
    "customer experience management",
    "user experience design",
    "user experience research",
}

STOPWORDS = {
    "with", "from", "that", "this", "have", "must", "skills", "skill",
    "and", "the", "for", "experience", "degree", "license",
    "certification", "certificate", "credential", "valid", "current",
}


def normalize_text(value: object) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(value).replace("’", "'").strip(),
    ).casefold()


def parse_skills(value: object) -> list[str]:
    if pd.isna(value) or str(value) == "":
        return []

    result = []
    for line in str(value).splitlines():
        match = NUMBERED_LINE_RE.match(line)
        if match is None:
            raise RuntimeError(f"Malformed numbered skill line: {line!r}")
        result.append(match.group(2).strip())
    return result


def meaningful_tokens(term: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9+#.-]+", normalize_text(term))
    return [
        token
        for token in tokens
        if len(token) >= 3 and token not in STOPWORDS
    ]


def split_clauses(description: object) -> list[str]:
    if pd.isna(description):
        return []

    return [
        re.sub(r"\s+", " ", clause).strip(" \t-:")
        for clause in CLAUSE_SPLIT_RE.split(str(description))
        if clause and clause.strip()
    ]


def choose_local_clause(
    clauses: list[str],
    raw_term: str,
) -> tuple[int, str, str, float]:
    if not clauses:
        return -1, "", "not_found", 0.0

    normalized_term = normalize_text(raw_term)

    # Prefer exact phrase presence in one local clause.
    for index, clause in enumerate(clauses):
        if normalized_term in normalize_text(clause):
            return index, clause, "exact_phrase", 1.0

    # Fallback: select the clause matching the most meaningful term tokens.
    tokens = meaningful_tokens(raw_term)
    if not tokens:
        return -1, "", "not_found", 0.0

    best = None
    for index, clause in enumerate(clauses):
        normalized_clause = normalize_text(clause)
        hits = sum(token in normalized_clause for token in tokens)
        if hits == 0:
            continue

        score = hits / len(tokens)
        candidate = (score, hits, -len(clause), index, clause)

        if best is None or candidate[:3] > best[:3]:
            best = candidate

    if best is None:
        return -1, "", "not_found", 0.0

    return best[3], best[4], "token_fallback", float(best[0])


def surrounding_context(clauses: list[str], index: int) -> str:
    if index < 0:
        return ""

    start = max(0, index - 1)
    end = min(len(clauses), index + 2)
    return " | ".join(clauses[start:end])


def qualification_suggestion(local_clause: str) -> str:
    has_post_hire = bool(POST_HIRE_RE.search(local_clause))
    has_preferred = bool(PREFERRED_RE.search(local_clause))
    has_entry = bool(ENTRY_RE.search(local_clause))

    # Post-hire timing takes priority over a generic word such as "required".
    # Example: "CPR certification required within 90 days of hire."
    if has_post_hire:
        return "post_hire_or_conditional"

    if has_preferred and has_entry:
        return "mixed_required_preferred_review"

    if has_preferred:
        return "preferred_qualification"

    if has_entry:
        return "entry_or_current_qualification"

    return "ambiguous_qualification_review"


def experience_suggestion(raw_term: str, local_clause: str) -> str:
    normalized_term = normalize_text(raw_term)

    if normalized_term in DOMAIN_EXPERIENCE_EXCEPTIONS:
        return "domain_experience_capability"

    if re.search(
        r"\bexperience\s+"
        r"(?:design|management|strategy|research|platform|optimization|"
        r"delivery|operations)\b",
        normalized_term,
    ):
        return "domain_experience_capability"

    # "sales experience", "litigation experience", "teaching experience",
    # etc. describe candidate background, not an atomic capability.
    if normalized_term.endswith(" experience"):
        return "experience_requirement_candidate"

    # Everything else stays review-only.
    return "ambiguous_experience_review"


def main() -> None:
    if not FULL_OUTPUT_PATH.exists():
        raise FileNotFoundError(FULL_OUTPUT_PATH)
    if not RECOVERED_POSTINGS_PATH.exists():
        raise FileNotFoundError(RECOVERED_POSTINGS_PATH)

    full = pd.read_csv(FULL_OUTPUT_PATH, low_memory=False)

    if len(full) != EXPECTED_PATCH28_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_PATCH28_ROWS:,} Patch 28 rows, "
            f"found {len(full):,}"
        )

    # Build candidate instances without touching descriptions first.
    candidate_rows = []

    for row in full.itertuples(index=False):
        for raw_term in parse_skills(row.extracted_skills):
            groups = []

            if QUALIFICATION_TERM_RE.search(raw_term):
                groups.append("qualification_like")

            if EXPERIENCE_TERM_RE.search(raw_term):
                groups.append("experience_term")

            for group in groups:
                candidate_rows.append(
                    {
                        "record_id": row.record_id,
                        "raw_extracted_term": raw_term,
                        "candidate_group": group,
                        "normalized_term": normalize_text(raw_term),
                    }
                )

    candidates = pd.DataFrame(candidate_rows)

    # Frequency drives a deterministic review sample.
    term_counts = (
        candidates.groupby(["candidate_group", "normalized_term"])
        .size()
        .rename("posting_term_rows")
        .reset_index()
        .sort_values(
            ["candidate_group", "posting_term_rows", "normalized_term"],
            ascending=[True, False, True],
        )
    )

    selected_indices = set()

    # Top 40 terms per group, up to 5 posting examples each.
    for group in ("qualification_like", "experience_term"):
        top_terms = (
            term_counts.loc[term_counts["candidate_group"].eq(group)]
            .head(40)["normalized_term"]
            .tolist()
        )

        for term in top_terms:
            indices = candidates.index[
                candidates["candidate_group"].eq(group)
                & candidates["normalized_term"].eq(term)
            ].tolist()[:5]
            selected_indices.update(indices)

    # Add deterministic random rows so long-tail cases are represented.
    for group in ("qualification_like", "experience_term"):
        group_rows = candidates.loc[
            candidates["candidate_group"].eq(group)
        ]

        sample_n = min(150, len(group_rows))
        random_indices = (
            group_rows.sample(n=sample_n, random_state=28).index.tolist()
        )
        selected_indices.update(random_indices)

    review = candidates.loc[sorted(selected_indices)].copy()

    # Read descriptions in chunks and keep only required IDs.
    needed_ids = set(review["record_id"])
    metadata_parts = []

    for chunk in pd.read_csv(
        RECOVERED_POSTINGS_PATH,
        usecols=["record_id", "title", "search_title", "description"],
        low_memory=False,
        chunksize=5_000,
    ):
        matched = chunk.loc[chunk["record_id"].isin(needed_ids)]
        if not matched.empty:
            metadata_parts.append(matched)

    metadata = pd.concat(metadata_parts, ignore_index=True)

    if metadata["record_id"].nunique() != len(needed_ids):
        missing = needed_ids - set(metadata["record_id"])
        raise RuntimeError(
            f"Missing {len(missing)} sampled record_ids from recovered postings"
        )

    review = review.merge(
        metadata,
        on="record_id",
        how="left",
        validate="many_to_one",
    )

    output_rows = []

    # Split each sampled job description only once.
    for record_id, posting_rows in review.groupby("record_id", sort=False):
        first = posting_rows.iloc[0]
        clauses = split_clauses(first["description"])

        for row in posting_rows.itertuples(index=False):
            index, local_clause, match_method, match_score = choose_local_clause(
                clauses,
                row.raw_extracted_term,
            )

            if row.candidate_group == "qualification_like":
                suggested_class = qualification_suggestion(local_clause)
            else:
                suggested_class = experience_suggestion(
                    row.raw_extracted_term,
                    local_clause,
                )

            output_rows.append(
                {
                    "record_id": row.record_id,
                    "job_title": row.title,
                    "search_query_title": row.search_title,
                    "raw_extracted_term": row.raw_extracted_term,
                    "candidate_group": row.candidate_group,
                    "match_method": match_method,
                    "match_score": round(match_score, 4),
                    "suggested_class": suggested_class,
                    "matched_local_clause": local_clause,
                    "surrounding_context": surrounding_context(clauses, index),
                    "review_decision": "",
                    "review_notes": "",
                }
            )

    output = pd.DataFrame(output_rows)

    summary = (
        output.groupby(
            ["candidate_group", "suggested_class", "match_method"],
            dropna=False,
        )
        .size()
        .rename("review_sample_rows")
        .reset_index()
        .sort_values(
            ["candidate_group", "review_sample_rows"],
            ascending=[True, False],
        )
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("Patch 28 context audit v2 review sample complete.")
    print()
    print(f"All candidate posting-term rows: {len(candidates):,}")
    print(f"Review sample rows: {len(output):,}")
    print()
    print("Suggested-class summary (REVIEW ONLY):")
    print(
        output.groupby(
            ["candidate_group", "suggested_class"]
        ).size().to_string()
    )
    print()
    print("Match methods:")
    print(output["match_method"].value_counts().to_string())
    print()
    print("Wrote:")
    print(f"  {OUTPUT_PATH}")
    print(f"  {SUMMARY_PATH}")
    print()
    print(
        "IMPORTANT: suggested_class is only a review aid. "
        "This script does not modify skills_long or any frozen source data."
    )


if __name__ == "__main__":
    main()
