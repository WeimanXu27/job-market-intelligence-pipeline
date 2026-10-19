"""Read-only causal triage for the 1,730 final zero-skill records.

This script uses the FINAL zero audit plus skill_extraction.log to determine
which execution path produced each current zero result.

It does NOT:
- call OpenAI,
- modify MySQL,
- change any extraction result,
- rerun the extractor.

Why this exists
---------------
The previous QA identified 1,730 zero-skill records and 592 heuristic
rich-source candidates. Before manually reviewing many records, this script
uses the production log to separate final zeros into paths such as:

1. role_screened_non_substantive
   The role-content screen deliberately stored an empty result.

2. extraction_returned_zero
   The record went through extraction but ultimately produced zero skills.

3. unknown_zero_path
   The current zero exists, but the final completed log cycle could not be
   confidently mapped to one of the two expected zero paths.

The script also creates a stratified review sample so manual QA can focus on
the highest-risk zeros without inspecting all 1,730 records.

Important
---------
The causal labels describe HOW the zero was produced, not whether the zero is
correct. A rich job description screened as non-substantive can still be a
false zero and therefore remains a QA priority.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "processed"
LOG_PATH = ROOT_DIR / "skill_extraction.log"

ZERO_ALL = DATA_DIR / "skill_full_run_zero_audit_all.csv"

ZERO_WITH_CAUSE = DATA_DIR / "skill_full_run_zero_with_log_cause.csv"
CAUSE_SUMMARY = DATA_DIR / "skill_full_run_zero_cause_summary.csv"
PRIORITY_SAMPLE = DATA_DIR / "skill_full_run_zero_priority_review_sample.csv"
UNMAPPED = DATA_DIR / "skill_full_run_zero_unmapped_log_records.csv"

RANDOM_SEED = 20260905

# Final manual-QA checkpoint size:
# - 96 higher-risk rich-source candidates
# - 48 lower-risk zero controls
# = 144 total, matching the project's established fresh-QA checkpoint size.
RICH_SAMPLE_TARGET = 96
CONTROL_SAMPLE_TARGET = 48


RECORD_RE = re.compile(r"\bRecord\s+(job_\d+)\b")
TIMESTAMP_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3})"
)
NER_RE = re.compile(r"NER extracted\s+(\d+)\s+skills", re.IGNORECASE)
EVIDENCE_RE = re.compile(r":\s*(\d+)\s+evidence chunks", re.IGNORECASE)
FINAL_RE = re.compile(r":\s*(\d+)\s+final skills", re.IGNORECASE)


def empty_cycle():
    return {
        "last_timestamp": "",
        "screened_non_substantive": False,
        "no_skills_extracted": False,
        "ner_count": None,
        "evidence_chunks": None,
        "final_skill_count": None,
        "saw_api_failure": False,
        "last_relevant_line": "",
    }


def snapshot_cycle(cycle: dict, completion_timestamp: str) -> dict:
    if cycle["screened_non_substantive"]:
        cause = "role_screened_non_substantive"
    elif (
        cycle["no_skills_extracted"]
        or cycle["final_skill_count"] == 0
    ):
        cause = "extraction_returned_zero"
    else:
        cause = "unknown_zero_path"

    return {
        "final_log_cause": cause,
        "final_cycle_completion_timestamp": completion_timestamp,
        "final_cycle_last_event_timestamp": cycle["last_timestamp"],
        "final_cycle_ner_count": cycle["ner_count"],
        "final_cycle_evidence_chunks": cycle["evidence_chunks"],
        "final_cycle_skill_count": cycle["final_skill_count"],
        "final_cycle_saw_api_failure": cycle["saw_api_failure"],
        "final_cycle_last_relevant_line": cycle["last_relevant_line"],
    }


def allocate_stratified_sample(
    frame: pd.DataFrame,
    target: int,
    rng: np.random.Generator,
    high_risk: bool,
) -> pd.DataFrame:
    """Sample across final log causes without over-concentrating one path."""
    if target <= 0 or frame.empty:
        return frame.iloc[0:0].copy()

    work = frame.copy()

    if high_risk:
        # Prefer higher richness first, but randomize within similar risk bands.
        work["risk_band"] = pd.cut(
            work["richness_score"],
            bins=[-1, 3, 4, 5, 99],
            labels=["low", "moderate", "high", "very_high"],
        )
    else:
        work["risk_band"] = "control"

    groups = []
    group_cols = ["final_log_cause", "pool"]
    for keys, group in work.groupby(group_cols, dropna=False, sort=True):
        groups.append((keys, group.copy()))

    if not groups:
        return work.iloc[0:0].copy()

    # Equal-ish allocation across cause x pool cells, then fill any leftover
    # from the remaining population.
    base = max(1, target // len(groups))
    selected_parts = []
    selected_ids = set()

    for _, group in groups:
        n = min(base, len(group))
        if n <= 0:
            continue

        if high_risk:
            # Take half from the highest-ranked records in the cell and half
            # random from the rest to avoid reviewing only one description type.
            group = group.sort_values(
                [
                    "richness_score",
                    "richness_marker_count",
                    "description_chars",
                ],
                ascending=False,
            )
            n_top = max(1, n // 2)
            top = group.head(n_top)
            remain = group.loc[~group["record_id"].isin(top["record_id"])]
            n_random = n - len(top)

            if n_random > 0 and not remain.empty:
                random_idx = rng.choice(
                    remain.index.to_numpy(),
                    size=min(n_random, len(remain)),
                    replace=False,
                )
                chosen = pd.concat([top, remain.loc[random_idx]])
            else:
                chosen = top
        else:
            random_idx = rng.choice(
                group.index.to_numpy(),
                size=n,
                replace=False,
            )
            chosen = group.loc[random_idx]

        selected_parts.append(chosen)
        selected_ids.update(chosen["record_id"])

    selected = (
        pd.concat(selected_parts, ignore_index=False)
        if selected_parts
        else work.iloc[0:0].copy()
    )

    if len(selected) < target:
        remaining = work.loc[
            ~work["record_id"].isin(selected_ids)
        ].copy()
        need = min(target - len(selected), len(remaining))

        if need > 0:
            if high_risk:
                # Fill first from the highest remaining risk.
                remaining = remaining.sort_values(
                    [
                        "richness_score",
                        "richness_marker_count",
                        "description_chars",
                    ],
                    ascending=False,
                )
                fill = remaining.head(need)
            else:
                idx = rng.choice(
                    remaining.index.to_numpy(),
                    size=need,
                    replace=False,
                )
                fill = remaining.loc[idx]

            selected = pd.concat([selected, fill], ignore_index=False)

    return selected.head(target).copy()


def main():
    if not ZERO_ALL.exists():
        raise FileNotFoundError(f"Missing final zero audit: {ZERO_ALL}")
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Missing production log: {LOG_PATH}")

    zero = pd.read_csv(
        ZERO_ALL,
        dtype={"record_id": str},
        keep_default_na=False,
        low_memory=False,
    )

    required = {
        "record_id",
        "pool",
        "description_chars",
        "description_words",
        "richness_marker_count",
        "richness_score",
        "rich_source_candidate",
        "description",
    }
    missing = required - set(zero.columns)
    if missing:
        raise ValueError(
            f"Zero audit missing required columns: {sorted(missing)}"
        )

    zero["record_id"] = zero["record_id"].astype(str).str.strip()
    zero_ids = set(zero["record_id"])

    if len(zero) != 1730:
        raise RuntimeError(
            f"Expected 1,730 final zero records; found {len(zero):,}."
        )
    if zero["record_id"].nunique() != len(zero):
        raise RuntimeError("Final zero audit has duplicate record IDs.")

    # Pandas may read the bool column as bool or string depending on version.
    if zero["rich_source_candidate"].dtype != bool:
        zero["rich_source_candidate"] = (
            zero["rich_source_candidate"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False})
        )
        if zero["rich_source_candidate"].isna().any():
            raise ValueError(
                "Could not parse rich_source_candidate as boolean."
            )

    cycles = {rid: empty_cycle() for rid in zero_ids}
    final_cycles = {}
    completion_counts = {rid: 0 for rid in zero_ids}

    with LOG_PATH.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            match = RECORD_RE.search(raw_line)
            if not match:
                continue

            rid = match.group(1)
            if rid not in zero_ids:
                continue

            ts_match = TIMESTAMP_RE.search(raw_line)
            timestamp = ts_match.group("ts") if ts_match else ""

            cycle = cycles[rid]
            cycle["last_timestamp"] = timestamp
            cycle["last_relevant_line"] = raw_line.strip()[:1000]

            lowered = raw_line.lower()

            if "screened as non-substantive role content" in lowered:
                cycle["screened_non_substantive"] = True

            if "no skills were extracted; storing an empty result" in lowered:
                cycle["no_skills_extracted"] = True

            ner_match = NER_RE.search(raw_line)
            if ner_match:
                cycle["ner_count"] = int(ner_match.group(1))

            evidence_match = EVIDENCE_RE.search(raw_line)
            if evidence_match:
                cycle["evidence_chunks"] = int(evidence_match.group(1))

            final_match = FINAL_RE.search(raw_line)
            if final_match:
                cycle["final_skill_count"] = int(final_match.group(1))

            if (
                "segment-grounded llm call failed" in lowered
                or "ratelimiterror" in lowered
                or "rate_limit_exceeded" in lowered
            ):
                cycle["saw_api_failure"] = True

            if "skills updated successfully" in lowered:
                completion_counts[rid] += 1
                final_cycles[rid] = snapshot_cycle(
                    cycle,
                    completion_timestamp=timestamp,
                )
                cycles[rid] = empty_cycle()

    cause_rows = []
    for rid in zero["record_id"]:
        cycle = final_cycles.get(rid)

        if cycle is None:
            cause_rows.append(
                {
                    "record_id": rid,
                    "final_log_cause": "no_completed_log_cycle_found",
                    "final_cycle_completion_timestamp": "",
                    "final_cycle_last_event_timestamp": "",
                    "final_cycle_ner_count": "",
                    "final_cycle_evidence_chunks": "",
                    "final_cycle_skill_count": "",
                    "final_cycle_saw_api_failure": "",
                    "final_cycle_last_relevant_line": "",
                    "completed_log_cycles_found": completion_counts[rid],
                }
            )
        else:
            row = {"record_id": rid, **cycle}
            row["completed_log_cycles_found"] = completion_counts[rid]
            cause_rows.append(row)

    causes = pd.DataFrame(cause_rows)

    merged = zero.merge(
        causes,
        on="record_id",
        how="left",
        validate="one_to_one",
    )

    # Current FINAL zero rows should not have a final completed cycle whose
    # explicit final skill count is >0.
    contradiction = merged.loc[
        pd.to_numeric(
            merged["final_cycle_skill_count"],
            errors="coerce",
        ).fillna(0).gt(0)
    ]

    if len(contradiction):
        contradiction.to_csv(
            UNMAPPED,
            index=False,
            encoding="utf-8-sig",
        )
        raise RuntimeError(
            f"{len(contradiction):,} current zero records have a final "
            "completed log cycle with >0 final skills. Review before QA."
        )

    merged.to_csv(
        ZERO_WITH_CAUSE,
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # Cause summary
    # ------------------------------------------------------------------
    summary_parts = []

    cause_counts = (
        merged.groupby("final_log_cause", dropna=False)
        .size()
        .rename("records")
        .reset_index()
    )

    rich_counts = (
        merged.loc[merged["rich_source_candidate"]]
        .groupby("final_log_cause", dropna=False)
        .size()
        .rename("rich_candidates")
        .reset_index()
    )

    cause_summary = cause_counts.merge(
        rich_counts,
        on="final_log_cause",
        how="left",
    )
    cause_summary["rich_candidates"] = (
        cause_summary["rich_candidates"].fillna(0).astype(int)
    )
    cause_summary["pct_of_all_zeros"] = (
        cause_summary["records"] / len(merged) * 100
    ).round(4)
    cause_summary["rich_share_within_cause_pct"] = (
        cause_summary["rich_candidates"]
        / cause_summary["records"]
        * 100
    ).round(4)

    cause_summary.to_csv(
        CAUSE_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # 144-record manual QA sample
    # ------------------------------------------------------------------
    rng = np.random.default_rng(RANDOM_SEED)

    rich = merged.loc[
        merged["rich_source_candidate"]
    ].copy()
    controls = merged.loc[
        ~merged["rich_source_candidate"]
    ].copy()

    rich_sample = allocate_stratified_sample(
        rich,
        RICH_SAMPLE_TARGET,
        rng,
        high_risk=True,
    )
    rich_sample["qa_sample_group"] = "rich_zero_priority"

    control_sample = allocate_stratified_sample(
        controls,
        CONTROL_SAMPLE_TARGET,
        rng,
        high_risk=False,
    )
    control_sample["qa_sample_group"] = "zero_control"

    sample = pd.concat(
        [rich_sample, control_sample],
        ignore_index=True,
    )

    if len(sample) != RICH_SAMPLE_TARGET + CONTROL_SAMPLE_TARGET:
        raise RuntimeError(
            f"Expected 144 QA rows; generated {len(sample):,}."
        )
    if sample["record_id"].nunique() != len(sample):
        raise RuntimeError("QA sample contains duplicate record IDs.")

    sample["qa_label"] = ""
    sample["qa_reason"] = ""
    sample["reviewer_note"] = ""

    sample_columns = [
        "qa_sample_group",
        "record_id",
        "pool",
        "final_log_cause",
        "completed_log_cycles_found",
        "final_cycle_completion_timestamp",
        "final_cycle_ner_count",
        "final_cycle_evidence_chunks",
        "final_cycle_saw_api_failure",
        "description_chars",
        "description_words",
        "richness_marker_count",
        "richness_score",
        "rich_source_candidate",
        "description",
        "qa_label",
        "qa_reason",
        "reviewer_note",
    ]

    sample[sample_columns].to_csv(
        PRIORITY_SAMPLE,
        index=False,
        encoding="utf-8-sig",
    )

    print("Final zero-result causal triage complete.")
    print()
    print(f"Current final zeros: {len(merged):,}")
    print(
        f"Heuristic rich-source candidates: "
        f"{int(merged['rich_source_candidate'].sum()):,}"
    )
    print()
    print("Final zero paths:")
    for _, row in cause_summary.iterrows():
        print(
            f"  {row['final_log_cause']}: "
            f"{int(row['records']):,} zeros | "
            f"{int(row['rich_candidates']):,} rich candidates"
        )

    api_failure_cycles = int(
        merged["final_cycle_saw_api_failure"]
        .astype(str)
        .str.lower()
        .eq("true")
        .sum()
    )
    print()
    print(
        "Final completed zero cycles with recorded API failure: "
        f"{api_failure_cycles:,}"
    )
    print(
        "NOTE: this should ideally be 0 after the repaired / Tier-2 production run."
    )
    print()
    print(
        f"Manual QA sample: {len(sample):,} "
        f"({RICH_SAMPLE_TARGET} rich-priority + "
        f"{CONTROL_SAMPLE_TARGET} controls)"
    )
    print()
    print(f"Cause summary: {CAUSE_SUMMARY}")
    print(f"All zeros with cause: {ZERO_WITH_CAUSE}")
    print(f"Priority QA sample: {PRIORITY_SAMPLE}")
    print()
    print("No MySQL rows were modified. No OpenAI API calls were made.")


if __name__ == "__main__":
    main()
