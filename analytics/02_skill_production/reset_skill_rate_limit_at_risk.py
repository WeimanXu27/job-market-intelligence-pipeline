"""Safely reset only the rate-limit at-risk skill records for repair.

Default mode is DRY-RUN and performs no database writes.

Before any reset, this script verifies:
- repair preflight passed
- anomaly_count == 0
- at-risk manifest has exactly 4,100 unique IDs
- backup file named in the preflight summary exists
- backup SHA-256 matches the preflight summary
- backup contains exactly the same 4,100 record IDs
- all 4,100 rows currently exist in MySQL
- all 4,100 currently have non-NULL extracted_skills

Only with --execute will it set extracted_skills = NULL for exactly those
4,100 record IDs. It then verifies that all 4,100 are NULL and commits.

No OpenAI API calls are made.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
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
SUMMARY_PATH = DATA_DIR / "skill_rate_limit_repair_preflight_summary.csv"
ANOMALIES_PATH = DATA_DIR / "skill_rate_limit_repair_preflight_anomalies.csv"

EXPECTED_AT_RISK = 4100


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_summary() -> dict[str, str]:
    df = pd.read_csv(SUMMARY_PATH, dtype=str, keep_default_na=False)
    if set(df.columns) != {"metric", "value"}:
        raise ValueError("Unexpected repair preflight summary schema.")
    return dict(zip(df["metric"], df["value"]))


def fetch_states(conn, ids: list[str]) -> pd.DataFrame:
    table = DATA_CONFIG["table_name"]
    id_col = DATA_CONFIG["id_column"]
    skills_col = DATA_CONFIG["skills_column"]

    frames = []
    chunk_size = 800
    for start in range(0, len(ids), chunk_size):
        chunk = ids[start:start + chunk_size]
        placeholders = ", ".join(f":id_{i}" for i in range(len(chunk)))
        params = {f"id_{i}": rid for i, rid in enumerate(chunk)}
        sql = text(
            f"SELECT {id_col} AS record_id, "
            f"{skills_col} AS extracted_skills "
            f"FROM {table} WHERE {id_col} IN ({placeholders})"
        )
        rows = conn.execute(sql, params).mappings().all()
        if rows:
            frames.append(pd.DataFrame(rows))

    if not frames:
        return pd.DataFrame(columns=["record_id", "extracted_skills"])

    result = pd.concat(frames, ignore_index=True)
    result["record_id"] = result["record_id"].astype(str).str.strip()
    return result


def reset_ids(conn, ids: list[str]) -> int:
    table = DATA_CONFIG["table_name"]
    id_col = DATA_CONFIG["id_column"]
    skills_col = DATA_CONFIG["skills_column"]

    total_updated = 0
    chunk_size = 500

    for start in range(0, len(ids), chunk_size):
        chunk = ids[start:start + chunk_size]
        placeholders = ", ".join(f":id_{i}" for i in range(len(chunk)))
        params = {f"id_{i}": rid for i, rid in enumerate(chunk)}
        sql = text(
            f"UPDATE {table} "
            f"SET {skills_col} = NULL "
            f"WHERE {id_col} IN ({placeholders})"
        )
        result = conn.execute(sql, params)
        total_updated += int(result.rowcount or 0)

    return total_updated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually reset the validated 4,100 records. Default is dry-run.",
    )
    args = parser.parse_args()

    for path in (AT_RISK_MANIFEST, SUMMARY_PATH, ANOMALIES_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    summary = read_summary()

    if summary.get("preflight_pass", "").lower() != "true":
        raise RuntimeError("Repair preflight did not pass.")
    if int(summary.get("anomaly_count", "-1")) != 0:
        raise RuntimeError("Repair preflight anomaly_count is not zero.")

    anomalies = pd.read_csv(ANOMALIES_PATH, keep_default_na=False)
    if len(anomalies) != 0:
        raise RuntimeError("Repair anomalies CSV is not empty.")

    at_risk = pd.read_csv(
        AT_RISK_MANIFEST,
        dtype={"record_id": str},
        keep_default_na=False,
        low_memory=False,
    )
    at_risk["record_id"] = at_risk["record_id"].astype(str).str.strip()

    if len(at_risk) != EXPECTED_AT_RISK:
        raise RuntimeError(
            f"Expected {EXPECTED_AT_RISK:,} at-risk rows; got {len(at_risk):,}."
        )
    if at_risk["record_id"].nunique() != EXPECTED_AT_RISK:
        raise RuntimeError("At-risk record IDs are not exactly 4,100 unique IDs.")

    ids = at_risk["record_id"].tolist()
    id_set = set(ids)

    backup_name = summary.get("backup_file", "").strip()
    backup_sha_expected = summary.get("backup_sha256", "").strip().lower()

    if not backup_name or not backup_sha_expected:
        raise RuntimeError("Backup file/hash missing from preflight summary.")

    backup_path = DATA_DIR / backup_name
    if not backup_path.exists():
        raise FileNotFoundError(f"Backup file not found: {backup_path}")

    backup_sha_actual = sha256_file(backup_path).lower()
    if backup_sha_actual != backup_sha_expected:
        raise RuntimeError(
            "Backup SHA-256 mismatch. Refusing to reset.\n"
            f"Expected: {backup_sha_expected}\n"
            f"Actual:   {backup_sha_actual}"
        )

    backup = pd.read_csv(
        backup_path,
        dtype={"record_id": str},
        keep_default_na=False,
        low_memory=False,
    )
    if "record_id" not in backup.columns:
        raise RuntimeError("Backup is missing record_id.")
    backup["record_id"] = backup["record_id"].astype(str).str.strip()

    if len(backup) != EXPECTED_AT_RISK:
        raise RuntimeError(
            f"Backup must contain {EXPECTED_AT_RISK:,} rows; got {len(backup):,}."
        )
    if backup["record_id"].nunique() != EXPECTED_AT_RISK:
        raise RuntimeError("Backup does not contain exactly 4,100 unique IDs.")
    if set(backup["record_id"]) != id_set:
        raise RuntimeError("Backup record IDs do not exactly match at-risk manifest.")

    engine = build_engine()

    with engine.begin() as conn:
        before = fetch_states(conn, ids)

        found = before["record_id"].nunique()
        current_nonnull = int(before["extracted_skills"].notna().sum())
        current_null = int(before["extracted_skills"].isna().sum())

        print("Rate-limit repair reset validation complete.")
        print(f"At-risk IDs: {len(ids):,}")
        print(f"Backup: {backup_path.name}")
        print(f"Backup SHA-256 PASS: {backup_sha_actual}")
        print(f"MySQL rows found: {found:,}")
        print(f"Current non-NULL: {current_nonnull:,}")
        print(f"Current NULL: {current_null:,}")

        if found != EXPECTED_AT_RISK:
            raise RuntimeError(
                f"MySQL should contain all {EXPECTED_AT_RISK:,} records."
            )
        if current_null != 0 or current_nonnull != EXPECTED_AT_RISK:
            raise RuntimeError(
                "Expected all 4,100 at-risk records to still have persisted "
                "values before reset. Refusing to continue."
            )

        if not args.execute:
            print()
            print("DRY-RUN PASS")
            print("No MySQL rows were modified.")
            print(
                "Run the same script with --execute only after this dry-run "
                "has been reviewed."
            )
            # engine.begin() will commit, but there were no writes.
            return

        updated = reset_ids(conn, ids)
        after = fetch_states(conn, ids)

        after_null = int(after["extracted_skills"].isna().sum())
        after_nonnull = int(after["extracted_skills"].notna().sum())

        print()
        print(f"UPDATE rowcount reported: {updated:,}")
        print(f"After reset NULL: {after_null:,}")
        print(f"After reset non-NULL: {after_nonnull:,}")

        if after_null != EXPECTED_AT_RISK or after_nonnull != 0:
            raise RuntimeError(
                "Post-reset verification failed. Transaction will roll back."
            )

        print()
        print("RESET PASS")
        print("Exactly 4,100 validated at-risk records were reset to NULL.")
        print("Backup remains unchanged on disk.")
        print("No OpenAI API calls were made.")


if __name__ == "__main__":
    main()
