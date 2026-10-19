"""Read-only preflight for the full skill extraction run.

This script validates the exact 64,283-record production universe, confirms that
only known frozen-Patch-28 records already have persisted skill results, and
writes deterministic full/pending manifests for the production run.

It does not modify MySQL.
"""

from __future__ import annotations

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


FROZEN_EXTRACTOR_SHA256 = (
    "e07f0fcf11e4d5ce0c9028590bb550e668ae722feeb058739e7b51f38fb05dc4"
)

EXPECTED_DEVELOPMENT_ROWS = 51_426
EXPECTED_FINAL_ROWS = 12_857
EXPECTED_UNIVERSE_ROWS = 64_283
EXPECTED_FROZEN_PROCESSED_ROWS = 935
EXPECTED_PENDING_ROWS = 63_348

DEVELOPMENT_POOL = DATA_DIR / "skill_development_pool.csv"
FINAL_POOL = DATA_DIR / "skill_final_test_pool.csv"
EXPOSED_719 = DATA_DIR / "skill_exposed_regression_719.csv"
FINAL_V4_144 = DATA_DIR / "skill_final_validation_144_v4.csv"
FINAL_V5_72 = DATA_DIR / "skill_final_validation_72_v5.csv"

FULL_MANIFEST_OUTPUT = DATA_DIR / "skill_full_run_manifest_64283.csv"
PENDING_MANIFEST_OUTPUT = DATA_DIR / "skill_full_run_pending_manifest.csv"
SUMMARY_OUTPUT = DATA_DIR / "skill_full_run_preflight_summary.csv"
ANOMALIES_OUTPUT = DATA_DIR / "skill_full_run_preflight_anomalies.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_ids(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file for {label}: {path}")

    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    if "record_id" not in df.columns:
        raise ValueError(f"{label} is missing record_id: {path}")

    result = pd.DataFrame(
        {
            "record_id": df["record_id"].astype(str).str.strip(),
        }
    )

    if result["record_id"].eq("").any():
        raise ValueError(f"{label} contains blank record_id values.")

    if not result["record_id"].is_unique:
        duplicates = (
            result.loc[result["record_id"].duplicated(False), "record_id"]
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise ValueError(
            f"{label} contains duplicate record_id values. Examples: {duplicates}"
        )

    return result


def validate_runtime_config() -> None:
    required = {
        "host": DATABASE_CONFIG.get("host"),
        "port": DATABASE_CONFIG.get("port"),
        "database": DATABASE_CONFIG.get("database"),
        "username": DATABASE_CONFIG.get("username"),
        "password": DATABASE_CONFIG.get("password"),
    }
    missing = [key for key, value in required.items() if value in (None, "")]
    if missing:
        raise ValueError(
            "Missing MySQL configuration in repo-root .env: " + ", ".join(missing)
        )

    if DATABASE_CONFIG["database"] != "job_market_portfolio":
        raise ValueError(
            "DATABASE_NAME must be job_market_portfolio for this production run."
        )

    if DATA_CONFIG["table_name"] != "job_postings":
        raise ValueError("Expected DATA_CONFIG['table_name'] = 'job_postings'.")


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


def fetch_database_status(engine) -> pd.DataFrame:
    table = DATA_CONFIG["table_name"]
    id_col = DATA_CONFIG["id_column"]
    desc_col = DATA_CONFIG["description_column"]
    skills_col = DATA_CONFIG["skills_column"]

    query = text(
        f"SELECT {id_col} AS record_id, "
        f"CASE WHEN {skills_col} IS NULL THEN 0 ELSE 1 END AS has_persisted_result, "
        f"CASE WHEN {skills_col} IS NOT NULL AND TRIM({skills_col}) <> '' "
        f"THEN 1 ELSE 0 END AS has_nonempty_result, "
        f"CASE WHEN {desc_col} IS NULL OR TRIM({desc_col}) = '' "
        f"THEN 1 ELSE 0 END AS description_blank, "
        f"CASE WHEN LOWER(TRIM({desc_col})) IN ('null', 'none', 'nan', 'n/a', 'na') "
        f"THEN 1 ELSE 0 END AS description_sentinel "
        f"FROM {table}"
    )

    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()

    db = pd.DataFrame(rows)
    if db.empty:
        raise RuntimeError("job_postings returned zero rows.")

    db["record_id"] = db["record_id"].astype(str).str.strip()

    if not db["record_id"].is_unique:
        duplicates = (
            db.loc[db["record_id"].duplicated(False), "record_id"]
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise RuntimeError(
            "MySQL job_postings contains duplicate record_id values. "
            f"Examples: {duplicates}"
        )

    for col in [
        "has_persisted_result",
        "has_nonempty_result",
        "description_blank",
        "description_sentinel",
    ]:
        db[col] = pd.to_numeric(db[col], errors="raise").astype(int)

    return db


def add_anomaly(
    frames: list[pd.DataFrame],
    record_ids,
    anomaly_type: str,
) -> None:
    ids = pd.Series(list(record_ids), dtype=str)
    if ids.empty:
        return
    frames.append(
        pd.DataFrame(
            {
                "record_id": ids,
                "anomaly_type": anomaly_type,
            }
        )
    )


def main() -> None:
    print("Full skill extraction preflight")
    print("Mode: READ ONLY (no MySQL updates)\n")

    extractor_path = SRC_DIR / "skill_extraction.py"
    if not extractor_path.exists():
        raise FileNotFoundError(f"Missing frozen extractor: {extractor_path}")

    extractor_sha = sha256_file(extractor_path)
    print(f"Frozen extractor SHA-256: {extractor_sha}")
    if extractor_sha != FROZEN_EXTRACTOR_SHA256:
        raise RuntimeError(
            "src/skill_extraction.py does not match the frozen Patch 28 SHA-256."
        )
    print("  PASS: extractor matches frozen Patch 28\n")

    development = read_ids(DEVELOPMENT_POOL, "development pool")
    final_test = read_ids(FINAL_POOL, "final-test pool")

    if len(development) != EXPECTED_DEVELOPMENT_ROWS:
        raise RuntimeError(
            f"Development pool count changed: {len(development):,} "
            f"!= {EXPECTED_DEVELOPMENT_ROWS:,}"
        )
    if len(final_test) != EXPECTED_FINAL_ROWS:
        raise RuntimeError(
            f"Final-test pool count changed: {len(final_test):,} "
            f"!= {EXPECTED_FINAL_ROWS:,}"
        )

    development_ids = set(development["record_id"])
    final_ids = set(final_test["record_id"])
    overlap = development_ids & final_ids
    if overlap:
        raise RuntimeError(
            f"Development/final-test overlap is not zero: {len(overlap):,}"
        )

    universe = pd.concat(
        [
            development.assign(pool="development"),
            final_test.assign(pool="final_test"),
        ],
        ignore_index=True,
    )
    universe = universe.sort_values("record_id", kind="stable").reset_index(drop=True)
    universe.insert(0, "run_order", range(1, len(universe) + 1))

    if len(universe) != EXPECTED_UNIVERSE_ROWS or not universe["record_id"].is_unique:
        raise RuntimeError(
            "Production universe is not the expected 64,283 unique records."
        )

    print("Production universe:")
    print(f"  Development: {len(development):,}")
    print(f"  Final-test:   {len(final_test):,}")
    print(f"  Overlap:      {len(overlap):,}")
    print(f"  Union:        {len(universe):,}")
    print("  PASS: exact 64,283 usable-record universe\n")

    exposed = read_ids(EXPOSED_719, "Patch-28 exposed regression")
    final_v4 = read_ids(FINAL_V4_144, "Patch-28 final v4")
    final_v5 = read_ids(FINAL_V5_72, "Patch-28 final v5")

    expected_counts = [
        ("exposed regression", exposed, 719),
        ("final v4", final_v4, 144),
        ("final v5", final_v5, 72),
    ]
    for label, frame, expected in expected_counts:
        if len(frame) != expected:
            raise RuntimeError(
                f"{label} count changed: {len(frame):,} != {expected:,}"
            )

    known_processed = pd.concat(
        [
            exposed.assign(source="exposed_719"),
            final_v4.assign(source="final_v4_144"),
            final_v5.assign(source="final_v5_72"),
        ],
        ignore_index=True,
    )

    duplicated_processed = known_processed[
        known_processed["record_id"].duplicated(False)
    ]
    if not duplicated_processed.empty:
        raise RuntimeError(
            "Known frozen-Patch-28 manifests overlap unexpectedly."
        )

    universe_ids = set(universe["record_id"])
    known_processed_ids = set(known_processed["record_id"])
    outside_universe = known_processed_ids - universe_ids
    if outside_universe:
        raise RuntimeError(
            "Known Patch-28 processed manifests contain IDs outside the 64,283 universe."
        )

    if len(known_processed_ids) != EXPECTED_FROZEN_PROCESSED_ROWS:
        raise RuntimeError(
            f"Known frozen Patch-28 union changed: {len(known_processed_ids):,} "
            f"!= {EXPECTED_FROZEN_PROCESSED_ROWS:,}"
        )

    print("Known frozen Patch-28 results:")
    print(f"  Exposed regression: {len(exposed):,}")
    print(f"  Final v4:           {len(final_v4):,}")
    print(f"  Final v5:           {len(final_v5):,}")
    print(f"  Unique union:       {len(known_processed_ids):,}")
    print("  PASS: expected processed-manifest union = 935\n")

    validate_runtime_config()
    engine = build_engine()
    try:
        db = fetch_database_status(engine)
    finally:
        engine.dispose()

    print(f"MySQL job_postings rows: {len(db):,}\n")

    merged = universe.merge(
        db,
        on="record_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    anomaly_frames: list[pd.DataFrame] = []

    missing_from_db = merged.loc[merged["_merge"] != "both", "record_id"]
    add_anomaly(anomaly_frames, missing_from_db, "usable_id_missing_from_mysql")

    present = merged[merged["_merge"] == "both"].copy()
    invalid_description = present.loc[
        (present["description_blank"] == 1)
        | (present["description_sentinel"] == 1),
        "record_id",
    ]
    add_anomaly(
        anomaly_frames,
        invalid_description,
        "usable_id_has_invalid_description_in_mysql",
    )

    completed_ids = set(
        present.loc[present["has_persisted_result"] == 1, "record_id"]
    )
    pending_ids = set(
        present.loc[present["has_persisted_result"] == 0, "record_id"]
    )

    expected_but_not_completed = known_processed_ids - completed_ids
    unexpected_completed = completed_ids - known_processed_ids

    add_anomaly(
        anomaly_frames,
        sorted(expected_but_not_completed),
        "known_patch28_result_not_persisted",
    )
    add_anomaly(
        anomaly_frames,
        sorted(unexpected_completed),
        "unexpected_preexisting_usable_output",
    )

    db_ids = set(db["record_id"])
    nonusable_ids = db_ids - universe_ids
    nonusable_with_outputs = set(
        db.loc[
            db["record_id"].isin(nonusable_ids)
            & (db["has_persisted_result"] == 1),
            "record_id",
        ]
    )

    persisted_empty_count = int(
        present.loc[
            present["has_persisted_result"] == 1,
            "has_nonempty_result",
        ].eq(0).sum()
    )

    metrics = {
        "extractor_sha256": extractor_sha,
        "development_rows": len(development),
        "final_test_rows": len(final_test),
        "development_final_overlap": len(overlap),
        "usable_universe_rows": len(universe),
        "mysql_total_rows": len(db),
        "usable_ids_missing_from_mysql": len(missing_from_db),
        "usable_ids_invalid_description": len(invalid_description),
        "known_patch28_manifest_rows": len(known_processed_ids),
        "usable_completed_rows": len(completed_ids),
        "usable_pending_rows": len(pending_ids),
        "known_patch28_not_completed": len(expected_but_not_completed),
        "unexpected_completed_usable_rows": len(unexpected_completed),
        "persisted_empty_usable_rows": persisted_empty_count,
        "nonusable_rows_in_mysql": len(nonusable_ids),
        "nonusable_rows_with_persisted_skills": len(nonusable_with_outputs),
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(SUMMARY_OUTPUT, index=False)

    anomalies = (
        pd.concat(anomaly_frames, ignore_index=True)
        if anomaly_frames
        else pd.DataFrame(columns=["record_id", "anomaly_type"])
    )
    anomalies.to_csv(ANOMALIES_OUTPUT, index=False)

    print("Database status inside exact 64,283 universe:")
    print(f"  Completed (non-NULL persisted result): {len(completed_ids):,}")
    print(f"  Pending (NULL extracted_skills):       {len(pending_ids):,}")
    print(f"  Persisted empty-string results:        {persisted_empty_count:,}")
    print(f"  Known Patch-28 IDs not completed:      {len(expected_but_not_completed):,}")
    print(f"  Unexpected completed usable IDs:       {len(unexpected_completed):,}")
    print(f"  Usable IDs missing from MySQL:         {len(missing_from_db):,}")
    print(f"  Usable IDs with invalid descriptions:  {len(invalid_description):,}\n")

    print("Outside exact universe (informational only):")
    print(f"  Non-usable MySQL rows:                 {len(nonusable_ids):,}")
    print(f"  Non-usable rows with persisted skills: {len(nonusable_with_outputs):,}\n")

    fatal_count = (
        len(missing_from_db)
        + len(invalid_description)
        + len(expected_but_not_completed)
        + len(unexpected_completed)
    )

    print(f"Summary CSV:   {SUMMARY_OUTPUT}")
    print(f"Anomalies CSV: {ANOMALIES_OUTPUT}")

    if fatal_count > 0:
        print("\nPRE-FLIGHT FAILED: do not start the full extraction run.")
        print("Upload the summary and anomalies CSV for review.")
        raise SystemExit(1)

    if len(completed_ids) != EXPECTED_FROZEN_PROCESSED_ROWS:
        raise RuntimeError(
            f"Clean preflight expected {EXPECTED_FROZEN_PROCESSED_ROWS:,} completed "
            f"usable rows, found {len(completed_ids):,}."
        )

    if len(pending_ids) != EXPECTED_PENDING_ROWS:
        raise RuntimeError(
            f"Clean preflight expected {EXPECTED_PENDING_ROWS:,} pending usable rows, "
            f"found {len(pending_ids):,}."
        )

    universe_status = universe.copy()
    universe_status["already_processed"] = (
        universe_status["record_id"].isin(completed_ids)
    )
    universe_status.to_csv(FULL_MANIFEST_OUTPUT, index=False)

    pending_manifest = universe_status.loc[
        ~universe_status["already_processed"],
        ["run_order", "record_id", "pool"],
    ].reset_index(drop=True)
    pending_manifest.to_csv(PENDING_MANIFEST_OUTPUT, index=False)

    print(f"Full manifest:  {FULL_MANIFEST_OUTPUT}")
    print(f"Pending manifest: {PENDING_MANIFEST_OUTPUT}")
    print("\nPRE-FLIGHT PASS")
    print(
        f"Exact production scope: {EXPECTED_UNIVERSE_ROWS:,}; "
        f"already processed with frozen Patch 28: {EXPECTED_FROZEN_PROCESSED_ROWS:,}; "
        f"remaining: {EXPECTED_PENDING_ROWS:,}."
    )
    print("No MySQL rows were modified.")


if __name__ == "__main__":
    main()
