"""Final full-corpus skill extraction QA: distribution + empty-result audit.

This is a READ-ONLY QA script for the frozen Patch-28 extraction output.

It does NOT:
- call OpenAI,
- modify MySQL,
- reset or rerun any record,
- change src/skill_extraction.py.

Checkpoint covered by this script
---------------------------------
1. Verify the exact 64,283 usable-record universe and frozen Patch-28 SHA.
2. Compute full-corpus skill-count distribution.
3. Audit all empty-string outputs structurally and rank the most suspicious
   "rich-source false-zero candidates" for later review.
4. Export high-count outliers for the next QA checkpoint.

Important:
The zero-candidate heuristic is ONLY a prioritization tool. It is NOT the
final rich-source false-zero rate and does not automatically label any record
as Major/Minor/Pass.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import URL, bindparam, create_engine, text

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data" / "processed"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from settings import DATABASE_CONFIG, DATA_CONFIG  # noqa: E402


FROZEN_SHA256 = (
    "e07f0fcf11e4d5ce0c9028590bb550e668ae722feeb058739e7b51f38fb05dc4"
)

EXPECTED_USABLE_ROWS = 64_283

FULL_MANIFEST = DATA_DIR / "skill_full_run_manifest_64283.csv"

SUMMARY_CSV = DATA_DIR / "skill_full_run_final_qa_summary.csv"
DISTRIBUTION_CSV = DATA_DIR / "skill_full_run_skill_count_distribution.csv"
ZERO_ALL_CSV = DATA_DIR / "skill_full_run_zero_audit_all.csv"
ZERO_TOP_CSV = DATA_DIR / "skill_full_run_zero_candidates_top200.csv"
HIGH_COUNT_CSV = DATA_DIR / "skill_full_run_high_count_top100.csv"
ANOMALIES_CSV = DATA_DIR / "skill_full_run_final_qa_anomalies.csv"

TABLE_NAME = DATA_CONFIG["table_name"]
ID_COLUMN = DATA_CONFIG["id_column"]
DESCRIPTION_COLUMN = DATA_CONFIG["description_column"]
SKILLS_COLUMN = DATA_CONFIG["skills_column"]


RICHNESS_MARKERS = [
    "requirements",
    "qualifications",
    "responsibilities",
    "required qualifications",
    "preferred qualifications",
    "minimum qualifications",
    "experience",
    "skills",
    "proficiency",
    "proficient",
    "knowledge of",
    "ability to",
    "must have",
    "you will",
    "what you'll do",
    "what you will do",
    "what we're looking for",
    "what we are looking for",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_engine():
    db = DATABASE_CONFIG
    url = URL.create(
        "mysql+pymysql",
        username=db["username"],
        password=db["password"],
        host=db["host"],
        port=int(db["port"]),
        database=db["database"],
        query={"charset": db.get("charset", "utf8mb4")},
    )
    return create_engine(url, pool_pre_ping=True, pool_recycle=3600)


def normalize_text(value) -> str:
    if value is None:
        return ""
    return str(value)


def count_skills(value) -> int:
    text_value = normalize_text(value).strip()
    if not text_value:
        return 0

    lines = [line.strip() for line in text_value.splitlines() if line.strip()]
    numbered = [
        line
        for line in lines
        if re.match(r"^\s*\d+[\.\)]\s*\S+", line)
    ]

    if numbered:
        return len(numbered)

    # Fallback for any legacy / malformed non-numbered result.
    return len(lines)


def description_metrics(value) -> tuple[int, int, int]:
    text_value = normalize_text(value)
    lowered = text_value.lower()

    char_count = len(text_value)
    word_count = len(re.findall(r"\b[\w+#./-]+\b", text_value))

    marker_count = sum(1 for marker in RICHNESS_MARKERS if marker in lowered)

    return char_count, word_count, marker_count


def fetch_exact_rows(engine, record_ids: list[str]) -> pd.DataFrame:
    query = text(
        f"SELECT {ID_COLUMN} AS record_id, "
        f"{DESCRIPTION_COLUMN} AS description, "
        f"{SKILLS_COLUMN} AS extracted_skills "
        f"FROM {TABLE_NAME} "
        f"WHERE {ID_COLUMN} IN :record_ids"
    ).bindparams(bindparam("record_ids", expanding=True))

    frames = []
    chunk_size = 800

    with engine.connect() as conn:
        for start in range(0, len(record_ids), chunk_size):
            chunk = record_ids[start:start + chunk_size]
            rows = conn.execute(
                query, {"record_ids": chunk}
            ).mappings().all()
            if rows:
                frames.append(pd.DataFrame(rows))

    if not frames:
        return pd.DataFrame(
            columns=["record_id", "description", "extracted_skills"]
        )

    result = pd.concat(frames, ignore_index=True)
    result["record_id"] = result["record_id"].astype(str).str.strip()
    return result


def main():
    anomalies = []

    extractor_path = SRC_DIR / "skill_extraction.py"
    if not extractor_path.exists():
        raise FileNotFoundError(f"Missing frozen extractor: {extractor_path}")

    actual_sha = sha256_file(extractor_path)
    if actual_sha.lower() != FROZEN_SHA256:
        anomalies.append(
            {
                "category": "frozen_extractor_sha_mismatch",
                "record_id": "",
                "detail": f"expected={FROZEN_SHA256}; actual={actual_sha}",
            }
        )

    if not FULL_MANIFEST.exists():
        raise FileNotFoundError(f"Missing full manifest: {FULL_MANIFEST}")

    manifest = pd.read_csv(
        FULL_MANIFEST,
        dtype={"record_id": str},
        keep_default_na=False,
        low_memory=False,
    )

    required = {"record_id", "run_order", "pool"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Full manifest missing columns: {sorted(missing)}")

    manifest["record_id"] = manifest["record_id"].astype(str).str.strip()
    manifest["run_order"] = pd.to_numeric(
        manifest["run_order"], errors="raise"
    ).astype(int)
    manifest.sort_values("run_order", kind="stable", inplace=True)
    manifest.reset_index(drop=True, inplace=True)

    if len(manifest) != EXPECTED_USABLE_ROWS:
        anomalies.append(
            {
                "category": "unexpected_manifest_row_count",
                "record_id": "",
                "detail": (
                    f"expected={EXPECTED_USABLE_ROWS}; actual={len(manifest)}"
                ),
            }
        )

    if manifest["record_id"].nunique() != EXPECTED_USABLE_ROWS:
        anomalies.append(
            {
                "category": "manifest_record_ids_not_unique",
                "record_id": "",
                "detail": (
                    f"rows={len(manifest)}; "
                    f"unique={manifest['record_id'].nunique()}"
                ),
            }
        )

    engine = build_engine()
    try:
        db = fetch_exact_rows(engine, manifest["record_id"].tolist())
    finally:
        engine.dispose()

    db_unique = db.drop_duplicates(subset=["record_id"], keep="first").copy()
    db_ids = set(db_unique["record_id"])
    manifest_ids = set(manifest["record_id"])

    missing_db = sorted(manifest_ids - db_ids)
    for rid in missing_db[:100]:
        anomalies.append(
            {
                "category": "usable_id_missing_from_mysql",
                "record_id": rid,
                "detail": "",
            }
        )

    duplicate_db = db.loc[
        db["record_id"].duplicated(keep=False), "record_id"
    ].unique()
    for rid in duplicate_db[:100]:
        anomalies.append(
            {
                "category": "duplicate_usable_id_in_mysql",
                "record_id": rid,
                "detail": "",
            }
        )

    merged = manifest[
        ["run_order", "record_id", "pool"]
    ].merge(
        db_unique,
        on="record_id",
        how="left",
        validate="one_to_one",
    )

    null_mask = merged["extracted_skills"].isna()
    if null_mask.any():
        for rid in merged.loc[null_mask, "record_id"].head(100):
            anomalies.append(
                {
                    "category": "usable_extracted_skills_is_null",
                    "record_id": rid,
                    "detail": "",
                }
            )

    merged["skill_count"] = merged["extracted_skills"].map(count_skills)

    # ------------------------------------------------------------------
    # Full-corpus skill-count distribution
    # ------------------------------------------------------------------
    counts = merged["skill_count"].astype(int)

    quantiles = counts.quantile(
        [0.50, 0.75, 0.90, 0.95, 0.99]
    ).to_dict()

    distribution_metrics = [
        ("records", len(merged)),
        ("mean_skill_count", round(float(counts.mean()), 4)),
        ("median_skill_count", float(quantiles[0.50])),
        ("p75_skill_count", float(quantiles[0.75])),
        ("p90_skill_count", float(quantiles[0.90])),
        ("p95_skill_count", float(quantiles[0.95])),
        ("p99_skill_count", float(quantiles[0.99])),
        ("zero_count", int((counts == 0).sum())),
        ("zero_rate_pct", round(float((counts == 0).mean() * 100), 4)),
        ("count_gt_30", int((counts > 30).sum())),
        ("rate_gt_30_pct", round(float((counts > 30).mean() * 100), 4)),
        ("count_gt_40", int((counts > 40).sum())),
        ("rate_gt_40_pct", round(float((counts > 40).mean() * 100), 4)),
        ("count_gt_60", int((counts > 60).sum())),
        ("rate_gt_60_pct", round(float((counts > 60).mean() * 100), 4)),
        ("count_gt_80", int((counts > 80).sum())),
        ("rate_gt_80_pct", round(float((counts > 80).mean() * 100), 4)),
        ("max_skill_count", int(counts.max())),
    ]

    pd.DataFrame(
        distribution_metrics, columns=["metric", "value"]
    ).to_csv(
        DISTRIBUTION_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # Zero-result audit
    # ------------------------------------------------------------------
    zero = merged.loc[counts.eq(0)].copy()

    metrics = zero["description"].map(description_metrics)
    zero["description_chars"] = metrics.map(lambda x: x[0])
    zero["description_words"] = metrics.map(lambda x: x[1])
    zero["richness_marker_count"] = metrics.map(lambda x: x[2])

    # Heuristic ranking only — not a final QA label.
    # Score deliberately combines length and explicit role-content markers.
    zero["richness_score"] = (
        (zero["description_chars"] >= 1500).astype(int)
        + (zero["description_chars"] >= 3000).astype(int)
        + (zero["description_words"] >= 250).astype(int)
        + (zero["description_words"] >= 500).astype(int)
        + (zero["richness_marker_count"] >= 2).astype(int)
        + (zero["richness_marker_count"] >= 4).astype(int)
    )

    zero["rich_source_candidate"] = zero["richness_score"].ge(4)

    zero.sort_values(
        [
            "rich_source_candidate",
            "richness_score",
            "richness_marker_count",
            "description_chars",
        ],
        ascending=[False, False, False, False],
        inplace=True,
    )

    zero_columns = [
        "run_order",
        "record_id",
        "pool",
        "skill_count",
        "description_chars",
        "description_words",
        "richness_marker_count",
        "richness_score",
        "rich_source_candidate",
        "description",
    ]

    zero[zero_columns].to_csv(
        ZERO_ALL_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    zero.head(200)[zero_columns].to_csv(
        ZERO_TOP_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # High-count outliers
    # ------------------------------------------------------------------
    high = merged.sort_values(
        ["skill_count", "run_order"],
        ascending=[False, True],
    ).head(100).copy()

    high[
        [
            "run_order",
            "record_id",
            "pool",
            "skill_count",
            "description",
            "extracted_skills",
        ]
    ].to_csv(
        HIGH_COUNT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # Summary / anomalies
    # ------------------------------------------------------------------
    anomalies_df = pd.DataFrame(
        anomalies,
        columns=["category", "record_id", "detail"],
    )
    anomalies_df.to_csv(
        ANOMALIES_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    zero_rich_count = int(zero["rich_source_candidate"].sum())

    summary_rows = [
        ("qa_structural_pass", len(anomalies_df) == 0),
        ("anomaly_count", len(anomalies_df)),
        ("frozen_patch28_sha256", actual_sha),
        ("usable_records", len(merged)),
        ("mysql_records_found", len(db_unique)),
        ("null_extracted_skills", int(null_mask.sum())),
        ("zero_skill_records", len(zero)),
        ("zero_skill_rate_pct", round(len(zero) / len(merged) * 100, 4)),
        (
            "heuristic_rich_zero_candidates",
            zero_rich_count,
        ),
        (
            "heuristic_rich_zero_candidate_rate_among_zeros_pct",
            round(
                zero_rich_count / len(zero) * 100,
                4,
            )
            if len(zero)
            else 0,
        ),
        (
            "IMPORTANT",
            (
                "heuristic_rich_zero_candidates is prioritization only; "
                "it is NOT the final rich-source false-zero rate"
            ),
        ),
    ]

    for metric, value in distribution_metrics:
        summary_rows.append((metric, value))

    summary = pd.DataFrame(
        summary_rows, columns=["metric", "value"]
    )
    summary.to_csv(
        SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print("Final full-corpus extraction QA checkpoint complete.")
    print()
    print(f"Frozen Patch-28 SHA: {actual_sha}")
    print(f"Usable records: {len(merged):,}")
    print(f"MySQL records found: {len(db_unique):,}")
    print(f"NULL extracted_skills: {int(null_mask.sum()):,}")
    print()
    print("Skill-count distribution:")
    print(f"  Mean:   {counts.mean():.2f}")
    print(f"  Median: {quantiles[0.50]:.0f}")
    print(f"  P90:    {quantiles[0.90]:.0f}")
    print(f"  P95:    {quantiles[0.95]:.0f}")
    print(f"  P99:    {quantiles[0.99]:.0f}")
    print(f"  >30:    {int((counts > 30).sum()):,}")
    print(f"  >40:    {int((counts > 40).sum()):,}")
    print(f"  >60:    {int((counts > 60).sum()):,}")
    print(f"  >80:    {int((counts > 80).sum()):,}")
    print(f"  Max:    {int(counts.max()):,}")
    print()
    print(
        f"Zero-skill records: {len(zero):,} "
        f"({len(zero) / len(merged) * 100:.2f}%)"
    )
    print(
        f"Heuristic rich-zero candidates: {zero_rich_count:,} "
        f"({zero_rich_count / len(zero) * 100:.2f}% of zeros)"
        if len(zero)
        else "Heuristic rich-zero candidates: 0"
    )
    print(
        "NOTE: heuristic rich-zero candidates are NOT final false-zero labels."
    )
    print()

    if len(anomalies_df) == 0:
        print("STRUCTURAL QA PASS")
    else:
        print("STRUCTURAL QA FAILED — review anomalies before continuing.")

    print()
    print(f"Summary: {SUMMARY_CSV}")
    print(f"Distribution: {DISTRIBUTION_CSV}")
    print(f"All zero audit: {ZERO_ALL_CSV}")
    print(f"Top zero candidates: {ZERO_TOP_CSV}")
    print(f"High-count top 100: {HIGH_COUNT_CSV}")
    print(f"Anomalies: {ANOMALIES_CSV}")
    print()
    print("No MySQL rows were modified. No OpenAI API calls were made.")


if __name__ == "__main__":
    main()
