"""Audit production skill extraction for final LLM call failures caused by rate limits.

READ-ONLY:
- Does not modify MySQL.
- Does not call OpenAI.
- Does not reset any extracted skills.

It parses skill_extraction.log, maps final "Segment-grounded LLM call failed"
events to completed production shard time windows in skill_full_run_progress.csv,
and conservatively identifies the corresponding record IDs from the original
production pending manifest.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = ROOT / "skill_extraction.log"
DATA_DIR = ROOT / "data" / "processed"
PROGRESS_PATH = DATA_DIR / "skill_full_run_progress.csv"
PENDING_MANIFEST_PATH = DATA_DIR / "skill_full_run_pending_manifest.csv"

EVENTS_OUT = DATA_DIR / "skill_rate_limit_failure_events.csv"
SHARDS_OUT = DATA_DIR / "skill_rate_limit_affected_shards.csv"
AT_RISK_OUT = DATA_DIR / "skill_rate_limit_at_risk_manifest.csv"

FAIL_PATTERN = "Segment-grounded LLM call failed"
TIMESTAMP_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
)


def parse_failure_events(log_path: Path) -> pd.DataFrame:
    rows = []
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            if FAIL_PATTERN not in line:
                continue
            match = TIMESTAMP_RE.search(line)
            if not match:
                continue
            rows.append(
                {
                    "log_line": line_no,
                    "failure_timestamp": pd.to_datetime(
                        match.group("ts"),
                        format="%Y-%m-%d %H:%M:%S,%f",
                    ),
                    "is_rpd_10000": (
                        "requests per day (RPD)" in line
                        and "Limit 10000" in line
                        and "Used 10000" in line
                    ),
                    "log_message": line.strip(),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    for path in (LOG_PATH, PROGRESS_PATH, PENDING_MANIFEST_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    progress = pd.read_csv(PROGRESS_PATH, low_memory=False)
    manifest = pd.read_csv(
        PENDING_MANIFEST_PATH,
        dtype={"record_id": str},
        keep_default_na=False,
        low_memory=False,
    )

    required_progress = {
        "timestamp_start",
        "timestamp_end",
        "manifest_ids_in_shard",
        "pending_before_processing",
        "extractor_total",
        "extractor_success",
        "extractor_failed",
        "persisted_after_processing",
        "still_pending_after_processing",
    }
    missing = required_progress.difference(progress.columns)
    if missing:
        raise ValueError(f"Progress CSV missing columns: {sorted(missing)}")
    if "record_id" not in manifest.columns:
        raise ValueError("Pending manifest is missing record_id.")

    progress = progress.copy()
    progress["progress_row"] = range(1, len(progress) + 1)
    progress["start_dt"] = pd.to_datetime(progress["timestamp_start"])
    progress["end_dt"] = pd.to_datetime(progress["timestamp_end"])

    # Reconstruct original manifest slice consumed by each completed progress row.
    # Before any repair/rerun, the runner has processed the initial pending
    # manifest sequentially. We use cumulative manifest_ids_in_shard rather
    # than shard_index_this_invocation because that index resets on resume.
    sizes = progress["manifest_ids_in_shard"].astype(int).tolist()
    starts = []
    ends = []
    cursor = 0
    for size in sizes:
        starts.append(cursor)
        cursor += size
        ends.append(cursor)
    progress["manifest_start_zero_based"] = starts
    progress["manifest_end_exclusive"] = ends

    events = parse_failure_events(LOG_PATH)
    if events.empty:
        print("No final Segment-grounded LLM call failures found in the log.")
        pd.DataFrame().to_csv(EVENTS_OUT, index=False)
        pd.DataFrame().to_csv(SHARDS_OUT, index=False)
        pd.DataFrame().to_csv(AT_RISK_OUT, index=False)
        return

    production_start = progress["start_dt"].min()
    events = events.loc[events["failure_timestamp"] >= production_start].copy()
    events = events.sort_values("failure_timestamp").reset_index(drop=True)
    events["event_id"] = range(1, len(events) + 1)

    # Map each failure timestamp to a completed progress-row interval.
    mapped_rows = []
    mapping_status = []
    for ts in events["failure_timestamp"]:
        hit = progress.loc[
            (progress["start_dt"] <= ts) & (ts <= progress["end_dt"])
        ]
        if len(hit) == 1:
            mapped_rows.append(int(hit.iloc[0]["progress_row"]))
            mapping_status.append("mapped_completed_shard")
        elif len(hit) > 1:
            mapped_rows.append(None)
            mapping_status.append("ambiguous_multiple_progress_rows")
        else:
            mapped_rows.append(None)
            if ts > progress["end_dt"].max():
                mapping_status.append("after_last_completed_shard")
            else:
                mapping_status.append("between_completed_shards")

    events["progress_row"] = mapped_rows
    events["mapping_status"] = mapping_status

    completed_affected_rows = sorted(
        {
            int(x)
            for x in events["progress_row"].dropna().tolist()
        }
    )

    shard_rows = []
    risk_rows = []

    for progress_row in completed_affected_rows:
        p = progress.loc[progress["progress_row"] == progress_row].iloc[0]
        event_count = int((events["progress_row"] == progress_row).sum())
        start_idx = int(p["manifest_start_zero_based"])
        end_idx = int(p["manifest_end_exclusive"])
        shard_manifest = manifest.iloc[start_idx:end_idx].copy()

        shard_rows.append(
            {
                "progress_row": progress_row,
                "timestamp_start": p["timestamp_start"],
                "timestamp_end": p["timestamp_end"],
                "failure_event_count": event_count,
                "manifest_start_one_based": start_idx + 1,
                "manifest_end_one_based": end_idx,
                "records_in_shard": len(shard_manifest),
                "extractor_success": int(p["extractor_success"]),
                "extractor_failed": int(p["extractor_failed"]),
                "persisted_after_processing": int(p["persisted_after_processing"]),
                "still_pending_after_processing": int(
                    p["still_pending_after_processing"]
                ),
                "risk_reason": "final_llm_call_failed_inside_completed_shard",
            }
        )

        for _, row in shard_manifest.iterrows():
            risk_rows.append(
                {
                    "record_id": str(row["record_id"]).strip(),
                    "progress_row": progress_row,
                    "risk_reason": "completed_shard_with_final_llm_failure",
                }
            )

    # If failures occurred after the last completed shard, conservatively flag
    # the next manifest block as the interrupted/current shard. This does not
    # assume those rows were persisted; it simply preserves them for review.
    after_last = events.loc[
        events["mapping_status"] == "after_last_completed_shard"
    ]
    if not after_last.empty:
        next_start = int(progress["manifest_end_exclusive"].max())
        if next_start < len(manifest):
            default_next_size = int(progress["manifest_ids_in_shard"].iloc[-1])
            next_end = min(next_start + default_next_size, len(manifest))
            next_manifest = manifest.iloc[next_start:next_end].copy()

            shard_rows.append(
                {
                    "progress_row": "",
                    "timestamp_start": progress["timestamp_end"].max(),
                    "timestamp_end": "",
                    "failure_event_count": len(after_last),
                    "manifest_start_one_based": next_start + 1,
                    "manifest_end_one_based": next_end,
                    "records_in_shard": len(next_manifest),
                    "extractor_success": "",
                    "extractor_failed": "",
                    "persisted_after_processing": "",
                    "still_pending_after_processing": "",
                    "risk_reason": "possible_current_interrupted_shard_after_last_progress_row",
                }
            )

            for _, row in next_manifest.iterrows():
                risk_rows.append(
                    {
                        "record_id": str(row["record_id"]).strip(),
                        "progress_row": "",
                        "risk_reason": "possible_current_interrupted_shard",
                    }
                )

    events_export = events[
        [
            "event_id",
            "failure_timestamp",
            "is_rpd_10000",
            "mapping_status",
            "progress_row",
            "log_line",
            "log_message",
        ]
    ].copy()

    shards_export = pd.DataFrame(shard_rows)
    risk_export = pd.DataFrame(risk_rows).drop_duplicates(
        subset=["record_id"], keep="first"
    )

    events_export.to_csv(EVENTS_OUT, index=False, encoding="utf-8-sig")
    shards_export.to_csv(SHARDS_OUT, index=False, encoding="utf-8-sig")
    risk_export.to_csv(AT_RISK_OUT, index=False, encoding="utf-8-sig")

    print("Rate-limit failure audit complete.")
    print(f"Production start: {production_start}")
    print(f"Completed progress rows: {len(progress):,}")
    print(f"Completed manifest records represented: {int(progress['manifest_ids_in_shard'].sum()):,}")
    print(f"Final LLM call failures since production start: {len(events):,}")
    print(f"  RPD 10000/10000 failures: {int(events['is_rpd_10000'].sum()):,}")
    print(
        "  Mapped to completed shards: "
        f"{int((events['mapping_status'] == 'mapped_completed_shard').sum()):,}"
    )
    print(
        "  After last completed shard: "
        f"{int((events['mapping_status'] == 'after_last_completed_shard').sum()):,}"
    )
    print(
        "  Between completed shards: "
        f"{int((events['mapping_status'] == 'between_completed_shards').sum()):,}"
    )
    print(f"Unique completed shards affected: {len(completed_affected_rows):,}")
    print(f"Conservative at-risk record IDs exported: {len(risk_export):,}")
    print()
    print(f"Events: {EVENTS_OUT}")
    print(f"Affected shards: {SHARDS_OUT}")
    print(f"At-risk manifest: {AT_RISK_OUT}")
    print()
    print("READ-ONLY audit: no MySQL rows were modified and no OpenAI calls were made.")


if __name__ == "__main__":
    main()
