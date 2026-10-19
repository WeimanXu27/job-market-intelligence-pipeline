"""Prepare a read-only backup and preflight for rate-limit repair.

This script DOES NOT modify MySQL and DOES NOT call OpenAI.

It validates the conservative at-risk manifest created from completed shards
that contained final LLM-call rate-limit failures, then exports the CURRENT
persisted MySQL values for those records before any reset/rerun is allowed.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import URL, create_engine, text

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data" / "processed"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from settings import DATABASE_CONFIG, DATA_CONFIG  # noqa: E402

AT_RISK_MANIFEST = DATA_DIR / "skill_rate_limit_at_risk_manifest.csv"
FULL_PENDING_MANIFEST = DATA_DIR / "skill_full_run_pending_manifest.csv"

EXPECTED_AT_RISK = 4100
EXPECTED_PROGRESS_ROWS = 41
EXPECTED_PER_PROGRESS_ROW = 100


def build_engine():
    url = URL.create(
        "mysql+pymysql",
        username=DATABASE_CONFIG["username"],
        password=DATABASE_CONFIG["password"],
        host=DATABASE_CONFIG["host"],
        port=int(DATABASE_CONFIG["port"]),
        database=DATABASE_CONFIG["database"],
        query={"charset": DATABASE_CONFIG.get("charset", "utf8mb4")},
    )
    return create_engine(url, pool_pre_ping=True)


def count_skills(value) -> int:
    if value is None:
        return 0
    text_value = str(value).strip()
    if not text_value:
        return 0
    lines = [line.strip() for line in text_value.splitlines() if line.strip()]
    numbered = [line for line in lines if "." in line and line.split(".", 1)[0].isdigit()]
    return len(numbered) if numbered else len(lines)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_database_rows(engine, record_ids: list[str]) -> pd.DataFrame:
    table = DATA_CONFIG["table_name"]
    id_col = DATA_CONFIG["id_column"]
    desc_col = DATA_CONFIG["description_column"]
    skills_col = DATA_CONFIG["skills_column"]

    frames = []
    chunk_size = 800

    with engine.connect() as conn:
        for start in range(0, len(record_ids), chunk_size):
            chunk = record_ids[start:start + chunk_size]
            placeholders = ", ".join(f":id_{i}" for i in range(len(chunk)))
            params = {f"id_{i}": rid for i, rid in enumerate(chunk)}
            sql = text(
                f"SELECT {id_col} AS record_id, "
                f"{desc_col} AS description, "
                f"{skills_col} AS old_extracted_skills "
                f"FROM {table} "
                f"WHERE {id_col} IN ({placeholders})"
            )
            rows = conn.execute(sql, params).mappings().all()
            if rows:
                frames.append(pd.DataFrame(rows))

    if not frames:
        return pd.DataFrame(columns=["record_id", "description", "old_extracted_skills"])

    result = pd.concat(frames, ignore_index=True)
    result["record_id"] = result["record_id"].astype(str).str.strip()
    return result


def add_anomaly(anomalies, category, record_id="", detail=""):
    anomalies.append({"category": category, "record_id": record_id, "detail": detail})


def main():
    for path in (AT_RISK_MANIFEST, FULL_PENDING_MANIFEST):
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")

    at_risk = pd.read_csv(
        AT_RISK_MANIFEST,
        dtype={"record_id": str},
        keep_default_na=False,
        low_memory=False,
    )
    full_manifest = pd.read_csv(
        FULL_PENDING_MANIFEST,
        dtype={"record_id": str},
        keep_default_na=False,
        low_memory=False,
    )

    required = {"record_id", "progress_row", "risk_reason"}
    missing_cols = required.difference(at_risk.columns)
    if missing_cols:
        raise ValueError(f"At-risk manifest missing columns: {sorted(missing_cols)}")
    if "record_id" not in full_manifest.columns:
        raise ValueError("Full pending manifest missing record_id.")

    at_risk = at_risk.copy()
    at_risk["record_id"] = at_risk["record_id"].astype(str).str.strip()
    full_manifest["record_id"] = full_manifest["record_id"].astype(str).str.strip()

    anomalies = []

    at_risk_rows = len(at_risk)
    at_risk_unique = at_risk["record_id"].nunique()
    progress_unique = at_risk["progress_row"].nunique()

    if at_risk_rows != EXPECTED_AT_RISK:
        add_anomaly(anomalies, "unexpected_at_risk_row_count",
                    detail=f"expected={EXPECTED_AT_RISK}; actual={at_risk_rows}")

    if at_risk_unique != at_risk_rows:
        dupes = at_risk.loc[at_risk["record_id"].duplicated(keep=False), "record_id"].unique()
        for rid in dupes:
            add_anomaly(anomalies, "duplicate_at_risk_record_id", rid)

    if progress_unique != EXPECTED_PROGRESS_ROWS:
        add_anomaly(anomalies, "unexpected_progress_row_count",
                    detail=f"expected={EXPECTED_PROGRESS_ROWS}; actual={progress_unique}")

    group_sizes = at_risk.groupby("progress_row", dropna=False).size()
    for progress_row, size in group_sizes.loc[group_sizes != EXPECTED_PER_PROGRESS_ROW].items():
        add_anomaly(
            anomalies,
            "unexpected_records_per_progress_row",
            detail=f"progress_row={progress_row}; actual={size}",
        )

    full_ids = set(full_manifest["record_id"])
    for rid in at_risk.loc[~at_risk["record_id"].isin(full_ids), "record_id"]:
        add_anomaly(anomalies, "at_risk_id_not_in_full_pending_manifest", rid)

    order_map = pd.Series(
        range(1, len(full_manifest) + 1), index=full_manifest["record_id"]
    ).to_dict()
    at_risk["production_order"] = at_risk["record_id"].map(order_map)

    engine = build_engine()
    db_rows = fetch_database_rows(engine, at_risk["record_id"].tolist())

    found_ids = set(db_rows["record_id"])
    for rid in at_risk.loc[~at_risk["record_id"].isin(found_ids), "record_id"]:
        add_anomaly(anomalies, "at_risk_id_missing_from_mysql", rid)

    for rid in db_rows.loc[db_rows["record_id"].duplicated(keep=False), "record_id"].unique():
        add_anomaly(anomalies, "duplicate_mysql_record_id", rid)

    backup = at_risk.merge(
        db_rows, on="record_id", how="left", validate="one_to_one"
    ).sort_values(["production_order", "record_id"])

    persisted_nonnull = int(backup["old_extracted_skills"].notna().sum())
    persisted_null = int(backup["old_extracted_skills"].isna().sum())
    empty_string = int(
        backup["old_extracted_skills"].fillna("").astype(str).str.strip().eq("").sum()
    )

    if persisted_null:
        for rid in backup.loc[backup["old_extracted_skills"].isna(), "record_id"]:
            add_anomaly(
                anomalies,
                "at_risk_record_currently_null",
                rid,
                "Expected completed-shard record to have a persisted value.",
            )

    backup["old_extracted_skill_count"] = backup["old_extracted_skills"].map(count_skills)
    backup["old_result_is_empty_string"] = (
        backup["old_extracted_skills"].fillna("").astype(str).str.strip().eq("")
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = DATA_DIR / f"skill_rate_limit_repair_backup_{timestamp}.csv"
    summary_path = DATA_DIR / "skill_rate_limit_repair_preflight_summary.csv"
    anomalies_path = DATA_DIR / "skill_rate_limit_repair_preflight_anomalies.csv"
    checksum_path = DATA_DIR / f"skill_rate_limit_repair_backup_{timestamp}.sha256.txt"

    backup_columns = [
        "production_order",
        "record_id",
        "progress_row",
        "risk_reason",
        "description",
        "old_extracted_skill_count",
        "old_result_is_empty_string",
        "old_extracted_skills",
    ]

    backup[backup_columns].to_csv(
        backup_path,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL,
        lineterminator="\n",
    )

    backup_sha = sha256_file(backup_path)
    checksum_path.write_text(
        f"{backup_sha}  {backup_path.name}\n",
        encoding="utf-8",
    )

    anomalies_df = pd.DataFrame(
        anomalies, columns=["category", "record_id", "detail"]
    )
    anomalies_df.to_csv(anomalies_path, index=False, encoding="utf-8-sig")

    preflight_pass = len(anomalies_df) == 0

    summary = pd.DataFrame(
        [
            ("preflight_pass", preflight_pass),
            ("at_risk_rows", at_risk_rows),
            ("at_risk_unique_record_ids", at_risk_unique),
            ("affected_progress_rows", progress_unique),
            ("mysql_rows_found", len(db_rows)),
            ("mysql_persisted_nonnull", persisted_nonnull),
            ("mysql_persisted_null", persisted_null),
            ("old_empty_string_results", empty_string),
            ("full_pending_manifest_rows", len(full_manifest)),
            ("backup_rows", len(backup)),
            ("backup_sha256", backup_sha),
            ("backup_file", backup_path.name),
            ("anomaly_count", len(anomalies_df)),
        ],
        columns=["metric", "value"],
    )
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("Rate-limit repair backup/preflight complete.")
    print(f"At-risk rows: {at_risk_rows:,}")
    print(f"Unique at-risk record IDs: {at_risk_unique:,}")
    print(f"Affected progress rows: {progress_unique:,}")
    print(f"MySQL rows found: {len(db_rows):,}")
    print(f"Current persisted non-NULL: {persisted_nonnull:,}")
    print(f"Current NULL: {persisted_null:,}")
    print(f"Current empty-string results: {empty_string:,}")
    print(f"Backup rows: {len(backup):,}")
    print(f"Backup SHA-256: {backup_sha}")
    print(f"Backup: {backup_path}")
    print(f"Summary: {summary_path}")
    print(f"Anomalies: {anomalies_path}")
    print()

    if preflight_pass:
        print("REPAIR PRE-FLIGHT PASS")
        print("No MySQL rows were modified. No OpenAI API calls were made.")
        print("DO NOT reset or rerun the 4,100 records yet.")
    else:
        print("REPAIR PRE-FLIGHT FAILED")
        print("Do not reset any records. Review anomalies first.")


if __name__ == "__main__":
    main()
