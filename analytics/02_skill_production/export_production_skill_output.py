"""Export the frozen Patch-28 full skill output for downstream analytics.

This is a READ-ONLY export step for Tableau Page 4 / later skill analytics.

Source of truth
---------------
- Universe / ordering / pool: data/processed/skill_full_run_manifest_64283.csv
- Persisted skill values: MySQL job_postings.extracted_skills
- Extractor version: frozen Patch-28 src/skill_extraction.py SHA-256

This script does NOT:
- call OpenAI or any other model API,
- modify MySQL,
- rerun skill extraction,
- use Patch-29 outputs.

Outputs
-------
1. data/processed/skill_patch28_full_output.csv
2. data/processed/skill_patch28_full_output_summary.csv

The first output is intentionally a production-source snapshot, not a
normalization table. Skill parsing / normalization belongs to the next
analytics-readiness step.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import URL, bindparam, create_engine, text


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data" / "processed"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from settings import DATABASE_CONFIG, DATA_CONFIG  # noqa: E402


FROZEN_PATCH28_SHA256 = (
    "e07f0fcf11e4d5ce0c9028590bb550e668ae722feeb058739e7b51f38fb05dc4"
)

EXPECTED_ROWS = 64_283
EXPECTED_DEVELOPMENT_ROWS = 51_426
EXPECTED_FINAL_TEST_ROWS = 12_857
EXPECTED_ZERO_SKILL_ROWS = 1_730
EXPECTED_MAX_SKILL_COUNT = 152
EXPECTED_MEAN_SKILL_COUNT_4DP = 17.0812
EXPECTED_MEDIAN_SKILL_COUNT = 14.0
EXPECTED_P95_SKILL_COUNT = 40.0

FULL_MANIFEST = DATA_DIR / "skill_full_run_manifest_64283.csv"
OUTPUT_CSV = DATA_DIR / "skill_patch28_full_output.csv"
SUMMARY_CSV = DATA_DIR / "skill_patch28_full_output_summary.csv"

TABLE_NAME = DATA_CONFIG["table_name"]
ID_COLUMN = DATA_CONFIG["id_column"]
SKILLS_COLUMN = DATA_CONFIG["skills_column"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            "DATABASE_NAME must be job_market_portfolio for this frozen export."
        )

    if TABLE_NAME != "job_postings":
        raise ValueError("Expected DATA_CONFIG['table_name'] = 'job_postings'.")

    if ID_COLUMN != "record_id":
        raise ValueError("Expected DATA_CONFIG['id_column'] = 'record_id'.")

    if SKILLS_COLUMN != "extracted_skills":
        raise ValueError(
            "Expected DATA_CONFIG['skills_column'] = 'extracted_skills'."
        )


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
    return create_engine(url, pool_pre_ping=True, pool_recycle=3600)


def count_skills(value) -> int:
    """Mirror the frozen full-run QA counting rule."""
    if value is None:
        return 0

    text_value = str(value).strip()
    if not text_value:
        return 0

    lines = [line.strip() for line in text_value.splitlines() if line.strip()]
    numbered = [
        line
        for line in lines
        if re.match(r"^\s*\d+[\.\)]\s*\S+", line)
    ]

    return len(numbered) if numbered else len(lines)


def load_manifest() -> pd.DataFrame:
    if not FULL_MANIFEST.exists():
        raise FileNotFoundError(f"Missing full-run manifest: {FULL_MANIFEST}")

    manifest = pd.read_csv(
        FULL_MANIFEST,
        dtype={"record_id": str},
        keep_default_na=False,
        low_memory=False,
    )

    required = {"run_order", "record_id", "pool"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Full manifest missing columns: {sorted(missing)}")

    manifest = manifest[["run_order", "record_id", "pool"]].copy()
    manifest["record_id"] = manifest["record_id"].astype(str).str.strip()
    manifest["pool"] = manifest["pool"].astype(str).str.strip()
    manifest["run_order"] = pd.to_numeric(
        manifest["run_order"], errors="raise"
    ).astype(int)

    if len(manifest) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Manifest row count changed: {len(manifest):,} != {EXPECTED_ROWS:,}"
        )

    if manifest["record_id"].eq("").any():
        raise RuntimeError("Manifest contains blank record_id values.")

    if not manifest["record_id"].is_unique:
        duplicates = (
            manifest.loc[
                manifest["record_id"].duplicated(False), "record_id"
            ]
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise RuntimeError(
            f"Manifest contains duplicate record_id values. Examples: {duplicates}"
        )

    if not manifest["run_order"].is_unique:
        raise RuntimeError("Manifest run_order is not unique.")

    expected_orders = list(range(1, EXPECTED_ROWS + 1))
    if manifest.sort_values("run_order")["run_order"].tolist() != expected_orders:
        raise RuntimeError("Manifest run_order is not exactly 1..64,283.")

    pool_counts = manifest["pool"].value_counts().to_dict()
    if pool_counts.get("development", 0) != EXPECTED_DEVELOPMENT_ROWS:
        raise RuntimeError(
            "Development pool count changed: "
            f"{pool_counts.get('development', 0):,} "
            f"!= {EXPECTED_DEVELOPMENT_ROWS:,}"
        )
    if pool_counts.get("final_test", 0) != EXPECTED_FINAL_TEST_ROWS:
        raise RuntimeError(
            "Final-test pool count changed: "
            f"{pool_counts.get('final_test', 0):,} "
            f"!= {EXPECTED_FINAL_TEST_ROWS:,}"
        )

    unexpected_pools = set(pool_counts) - {"development", "final_test"}
    if unexpected_pools:
        raise RuntimeError(
            f"Unexpected pool values in manifest: {sorted(unexpected_pools)}"
        )

    return manifest.sort_values("run_order", kind="stable").reset_index(drop=True)


def fetch_exact_skill_rows(engine, record_ids: list[str]) -> pd.DataFrame:
    """Fetch only manifest-whitelisted IDs; never query by non-NULL skills alone."""
    query = text(
        f"SELECT {ID_COLUMN} AS record_id, "
        f"{SKILLS_COLUMN} AS extracted_skills "
        f"FROM {TABLE_NAME} "
        f"WHERE {ID_COLUMN} IN :record_ids"
    ).bindparams(bindparam("record_ids", expanding=True))

    frames: list[pd.DataFrame] = []
    chunk_size = 800

    with engine.connect() as conn:
        for start in range(0, len(record_ids), chunk_size):
            chunk = record_ids[start : start + chunk_size]
            rows = conn.execute(query, {"record_ids": chunk}).mappings().all()
            if rows:
                frames.append(pd.DataFrame(rows))

    if not frames:
        return pd.DataFrame(columns=["record_id", "extracted_skills"])

    result = pd.concat(frames, ignore_index=True)
    result["record_id"] = result["record_id"].astype(str).str.strip()
    return result


def validate_and_build_output(
    manifest: pd.DataFrame,
    db_rows: pd.DataFrame,
) -> pd.DataFrame:
    if len(db_rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"MySQL rows fetched: {len(db_rows):,} != {EXPECTED_ROWS:,}. "
            "The export will not continue."
        )

    if not db_rows["record_id"].is_unique:
        duplicates = (
            db_rows.loc[
                db_rows["record_id"].duplicated(False), "record_id"
            ]
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise RuntimeError(
            "MySQL returned duplicate record_id values. "
            f"Examples: {duplicates}"
        )

    manifest_ids = set(manifest["record_id"])
    db_ids = set(db_rows["record_id"])

    missing_ids = sorted(manifest_ids - db_ids)
    unexpected_ids = sorted(db_ids - manifest_ids)

    if missing_ids:
        raise RuntimeError(
            f"{len(missing_ids):,} manifest IDs are missing from MySQL. "
            f"Examples: {missing_ids[:10]}"
        )
    if unexpected_ids:
        raise RuntimeError(
            f"Fetched {len(unexpected_ids):,} IDs outside the manifest whitelist. "
            f"Examples: {unexpected_ids[:10]}"
        )

    output = manifest.merge(
        db_rows,
        on="record_id",
        how="left",
        validate="one_to_one",
    )

    null_mask = output["extracted_skills"].isna()
    if null_mask.any():
        examples = output.loc[null_mask, "record_id"].head(10).tolist()
        raise RuntimeError(
            f"Found {int(null_mask.sum()):,} NULL extracted_skills values. "
            f"Examples: {examples}"
        )

    output["extracted_skill_count"] = output["extracted_skills"].map(count_skills)
    output["has_extracted_skill"] = output["extracted_skill_count"] > 0

    counts = output["extracted_skill_count"].astype(int)
    zero_rows = int((counts == 0).sum())
    max_count = int(counts.max())
    mean_count_4dp = round(float(counts.mean()), 4)
    median_count = float(counts.median())
    p95_count = float(counts.quantile(0.95))

    if zero_rows != EXPECTED_ZERO_SKILL_ROWS:
        raise RuntimeError(
            f"Zero-skill row count changed: {zero_rows:,} "
            f"!= {EXPECTED_ZERO_SKILL_ROWS:,}"
        )
    if max_count != EXPECTED_MAX_SKILL_COUNT:
        raise RuntimeError(
            f"Max skill count changed: {max_count:,} "
            f"!= {EXPECTED_MAX_SKILL_COUNT:,}"
        )
    if mean_count_4dp != EXPECTED_MEAN_SKILL_COUNT_4DP:
        raise RuntimeError(
            f"Mean skill count changed: {mean_count_4dp:.4f} "
            f"!= {EXPECTED_MEAN_SKILL_COUNT_4DP:.4f}"
        )
    if median_count != EXPECTED_MEDIAN_SKILL_COUNT:
        raise RuntimeError(
            f"Median skill count changed: {median_count} "
            f"!= {EXPECTED_MEDIAN_SKILL_COUNT}"
        )
    if p95_count != EXPECTED_P95_SKILL_COUNT:
        raise RuntimeError(
            f"P95 skill count changed: {p95_count} "
            f"!= {EXPECTED_P95_SKILL_COUNT}"
        )

    return output[
        [
            "run_order",
            "record_id",
            "pool",
            "extracted_skills",
            "extracted_skill_count",
            "has_extracted_skill",
        ]
    ]


def build_summary(
    output: pd.DataFrame,
    extractor_sha: str,
    output_sha: str,
) -> pd.DataFrame:
    counts = output["extracted_skill_count"].astype(int)

    metrics = [
        ("structural_pass", True),
        ("frozen_patch28_sha256", extractor_sha),
        ("output_csv_sha256", output_sha),
        ("rows", len(output)),
        ("unique_record_ids", output["record_id"].nunique()),
        ("development_rows", int((output["pool"] == "development").sum())),
        ("final_test_rows", int((output["pool"] == "final_test").sum())),
        ("non_null_extracted_skills", int(output["extracted_skills"].notna().sum())),
        ("postings_with_at_least_one_skill", int((counts > 0).sum())),
        ("zero_skill_records", int((counts == 0).sum())),
        ("zero_skill_rate_pct", round(float((counts == 0).mean() * 100), 4)),
        ("total_extracted_skill_instances", int(counts.sum())),
        ("mean_skill_count", round(float(counts.mean()), 4)),
        ("median_skill_count", float(counts.median())),
        ("p95_skill_count", float(counts.quantile(0.95))),
        ("max_skill_count", int(counts.max())),
        (
            "denominator_note",
            "64,283 Patch-28 processed postings with usable descriptions; "
            "not all 65,247 recovered postings",
        ),
    ]

    return pd.DataFrame(metrics, columns=["metric", "value"])


def main() -> None:
    print("Patch 28 full-output export")
    print("Mode: READ ONLY (no MySQL updates; no model/API calls)\n")

    validate_runtime_config()

    extractor_path = SRC_DIR / "skill_extraction.py"
    if not extractor_path.exists():
        raise FileNotFoundError(f"Missing frozen extractor: {extractor_path}")

    extractor_sha = sha256_file(extractor_path)
    print(f"Frozen extractor SHA-256: {extractor_sha}")
    if extractor_sha != FROZEN_PATCH28_SHA256:
        raise RuntimeError(
            "src/skill_extraction.py does not match the frozen Patch-28 SHA-256."
        )
    print("  PASS: extractor matches frozen Patch 28")

    manifest = load_manifest()
    print(f"Manifest rows: {len(manifest):,}")
    print("  PASS: exact 64,283-record whitelist")

    engine = build_engine()
    try:
        db_rows = fetch_exact_skill_rows(engine, manifest["record_id"].tolist())
    finally:
        engine.dispose()

    print(f"MySQL rows fetched for whitelist: {len(db_rows):,}")

    output = validate_and_build_output(manifest, db_rows)

    # Write only after all source/metric validations pass.
    output.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    output_sha = sha256_file(OUTPUT_CSV)

    summary = build_summary(output, extractor_sha, output_sha)
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print("\nExport validation: PASS")
    print(f"Rows: {len(output):,}")
    print(f"Postings with >=1 skill: {int(output['has_extracted_skill'].sum()):,}")
    print(f"Zero-skill postings: {int((~output['has_extracted_skill']).sum()):,}")
    print(f"Mean skills/posting: {output['extracted_skill_count'].mean():.4f}")
    print(f"Median skills/posting: {output['extracted_skill_count'].median():.0f}")
    print(f"P95 skills/posting: {output['extracted_skill_count'].quantile(0.95):.0f}")
    print(f"Max skills/posting: {output['extracted_skill_count'].max():,}")
    print(f"Output SHA-256: {output_sha}")

    print("\nCreated:")
    print(f"  {OUTPUT_CSV}")
    print(f"  {SUMMARY_CSV}")
    print("\nNo MySQL rows were modified. No OpenAI API calls were made.")


if __name__ == "__main__":
    main()
