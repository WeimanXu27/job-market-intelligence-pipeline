"""First candidate exposed regression runner (READ-ONLY MySQL).

Purpose
-------
Run the Patch-29 v5 candidate through its normal production-like pipeline on the
935 records that were already exposed before the full production run, then
compare the candidate outputs with the persisted Patch-28 outputs.

This is DEVELOPMENT / REGRESSION testing. It MUST NOT touch the frozen
288-record Patch-29 blind holdout.

Safety
------
- Reads MySQL only.
- Does NOT update extracted_skills.
- Calls OpenAI through the candidate extractor.
- Resumable: completed candidate outputs are written after each shard.
- Verifies the exact Patch-29 v5 candidate SHA.
- Verifies all selected IDs are `already_processed=True` in the exact
  64,283-record manifest.
- Explicitly verifies zero overlap with the 288 blind holdout.

Recommended first checkpoint
----------------------------
python analytics/03_skill_evaluation/run_first_candidate_regression.py --records-per-shard 100 --max-shards 1

If that passes, rerun without --max-shards to finish all 935 exposed records.
Do not run the remaining shards until the first 100-record checkpoint has been reviewed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import re
import sys
import types
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sqlalchemy import URL, bindparam, create_engine, text

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data" / "processed"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from settings import DATABASE_CONFIG, DATA_CONFIG  # noqa: E402

CANDIDATE_PATH = SRC_DIR / "reliability_first_blind_candidate.py"
EXPECTED_CANDIDATE_SHA256 = (
    "932c80eaac297c9c295b719744a9e9a6df716c87481d505af9cbd9906608770d"
)

FULL_MANIFEST = DATA_DIR / "skill_full_run_manifest_64283.csv"
BLIND_MANIFEST = DATA_DIR / "skill_patch29_blind_holdout_manifest.csv"

OUTPUT_CSV = DATA_DIR / "skill_patch29_v5_exposed_regression_outputs.csv"
PROGRESS_CSV = DATA_DIR / "skill_patch29_v5_exposed_regression_progress.csv"
SUMMARY_CSV = DATA_DIR / "skill_patch29_v5_exposed_regression_summary.csv"
CHANGED_CSV = DATA_DIR / "skill_patch29_v5_exposed_regression_changed_top100.csv"

EXPECTED_EXPOSED = 935

TABLE_NAME = DATA_CONFIG["table_name"]
ID_COLUMN = DATA_CONFIG["id_column"]
DESCRIPTION_COLUMN = DATA_CONFIG["description_column"]
SKILLS_COLUMN = DATA_CONFIG["skills_column"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_engine():
    db = DATABASE_CONFIG
    url = URL.create(
        drivername="mysql+mysqlconnector",
        username=db["username"],
        password=db["password"],
        host=db["host"],
        port=int(db["port"]),
        database=db["database"],
        query={"charset": db.get("charset", "utf8mb4")},
    )
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False,
    )


def parse_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    normalized = series.astype(str).str.strip().str.lower()
    bad = sorted(set(normalized) - set(mapping))
    if bad:
        raise ValueError(
            "Unexpected already_processed values: " + ", ".join(bad[:10])
        )
    return normalized.map(mapping)


def load_candidate_module():
    if not CANDIDATE_PATH.exists():
        raise FileNotFoundError(
            f"Patch-29 candidate not found: {CANDIDATE_PATH}"
        )

    actual_sha = sha256_file(CANDIDATE_PATH)
    if actual_sha.lower() != EXPECTED_CANDIDATE_SHA256:
        raise RuntimeError(
            "Patch-29 candidate SHA mismatch.\n"
            f"Expected: {EXPECTED_CANDIDATE_SHA256}\n"
            f"Actual:   {actual_sha}\n"
            "Do not run regression against an untracked candidate."
        )

    spec = importlib.util.spec_from_file_location(
        "reliability_first_blind_candidate",
        CANDIDATE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Patch-29 candidate module.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, actual_sha


def fetch_exposed_rows(engine, record_ids: List[str]) -> pd.DataFrame:
    query = text(
        f"SELECT {ID_COLUMN} AS record_id, "
        f"{DESCRIPTION_COLUMN} AS description, "
        f"{SKILLS_COLUMN} AS patch28_skills "
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
            columns=["record_id", "description", "patch28_skills"]
        )

    result = pd.concat(frames, ignore_index=True)
    result["record_id"] = result["record_id"].astype(str).str.strip()
    return result.drop_duplicates("record_id", keep="first")


def skill_lines(value) -> List[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    if isinstance(value, list):
        lines = [str(x).strip() for x in value if str(x).strip()]
    else:
        text_value = str(value).strip()
        if not text_value:
            return []
        lines = [line.strip() for line in text_value.splitlines() if line.strip()]

    cleaned = []
    for line in lines:
        line = re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip()
        if line:
            cleaned.append(line)
    return cleaned


def skill_count(value) -> int:
    return len(skill_lines(value))


def skill_key(value: str) -> str:
    return re.sub(r"[^a-z0-9+#]+", " ", value.lower()).strip()


def jaccard(a, b) -> float:
    sa = {skill_key(x) for x in skill_lines(a) if skill_key(x)}
    sb = {skill_key(x) for x in skill_lines(b) if skill_key(x)}
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def load_existing_outputs() -> pd.DataFrame:
    if not OUTPUT_CSV.exists():
        return pd.DataFrame(
            columns=[
                "record_id",
                "patch29_skills",
                "patch29_skill_count",
            ]
        )

    df = pd.read_csv(
        OUTPUT_CSV,
        dtype={"record_id": str},
        keep_default_na=False,
        low_memory=False,
    )
    if "record_id" not in df.columns:
        raise ValueError("Existing regression output has no record_id column.")
    df["record_id"] = df["record_id"].astype(str).str.strip()
    return df.drop_duplicates("record_id", keep="last")


def save_outputs(output_map: Dict[str, List[str]]):
    existing = load_existing_outputs()

    new_rows = []
    for record_id, skills in output_map.items():
        skill_list = skill_lines(skills)
        numbered = "\n".join(
            f"{i}. {skill}" for i, skill in enumerate(skill_list, start=1)
        )
        new_rows.append(
            {
                "record_id": str(record_id),
                "patch29_skills": numbered,
                "patch29_skill_count": len(skill_list),
            }
        )

    new_df = pd.DataFrame(new_rows)

    if existing.empty:
        combined = new_df
    else:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates("record_id", keep="last")

    combined.sort_values("record_id", inplace=True)
    combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")


def append_progress(row: dict):
    frame = pd.DataFrame([row])
    header = not PROGRESS_CSV.exists()
    frame.to_csv(
        PROGRESS_CSV,
        mode="a",
        header=header,
        index=False,
        encoding="utf-8-sig",
    )


def build_comparison(
    exposed: pd.DataFrame,
    outputs: pd.DataFrame,
    candidate_sha: str,
):
    merged = exposed.merge(
        outputs[["record_id", "patch29_skills", "patch29_skill_count"]],
        on="record_id",
        how="inner",
        validate="one_to_one",
    ).copy()

    merged["patch28_skill_count"] = merged["patch28_skills"].map(skill_count)
    merged["patch29_skill_count"] = pd.to_numeric(
        merged["patch29_skill_count"], errors="raise"
    ).astype(int)
    merged["count_delta"] = (
        merged["patch29_skill_count"] - merged["patch28_skill_count"]
    )
    merged["patch28_zero"] = merged["patch28_skill_count"].eq(0)
    merged["patch29_zero"] = merged["patch29_skill_count"].eq(0)
    merged["nonzero_to_zero"] = (
        merged["patch28_skill_count"].gt(0) & merged["patch29_zero"]
    )
    merged["zero_to_nonzero"] = (
        merged["patch28_zero"] & merged["patch29_skill_count"].gt(0)
    )

    merged["large_drop"] = (
        merged["patch28_skill_count"].ge(5)
        & (merged["patch29_skill_count"] <= merged["patch28_skill_count"] * 0.5)
        & (merged["count_delta"] <= -5)
    )

    merged["large_growth"] = (
        merged["patch28_skill_count"].ge(1)
        & (merged["patch29_skill_count"] >= merged["patch28_skill_count"] * 2)
        & (merged["count_delta"] >= 10)
    )

    merged["skill_jaccard"] = [
        jaccard(p28, p29)
        for p28, p29 in zip(
            merged["patch28_skills"],
            merged["patch29_skills"],
        )
    ]

    complete_n = len(merged)

    metrics = [
        ("candidate_sha256", candidate_sha),
        ("comparison_records", complete_n),
        ("expected_exposed_records", EXPECTED_EXPOSED),
        ("patch28_zero_records", int(merged["patch28_zero"].sum())),
        ("patch29_zero_records", int(merged["patch29_zero"].sum())),
        ("zero_to_nonzero", int(merged["zero_to_nonzero"].sum())),
        ("nonzero_to_zero", int(merged["nonzero_to_zero"].sum())),
        ("large_drop_records", int(merged["large_drop"].sum())),
        ("large_growth_records", int(merged["large_growth"].sum())),
        (
            "patch28_mean_skill_count",
            round(float(merged["patch28_skill_count"].mean()), 4)
            if complete_n else 0,
        ),
        (
            "patch29_mean_skill_count",
            round(float(merged["patch29_skill_count"].mean()), 4)
            if complete_n else 0,
        ),
        (
            "patch28_median_skill_count",
            float(merged["patch28_skill_count"].median())
            if complete_n else 0,
        ),
        (
            "patch29_median_skill_count",
            float(merged["patch29_skill_count"].median())
            if complete_n else 0,
        ),
        (
            "patch28_p95_skill_count",
            float(merged["patch28_skill_count"].quantile(0.95))
            if complete_n else 0,
        ),
        (
            "patch29_p95_skill_count",
            float(merged["patch29_skill_count"].quantile(0.95))
            if complete_n else 0,
        ),
        (
            "patch28_max_skill_count",
            int(merged["patch28_skill_count"].max())
            if complete_n else 0,
        ),
        (
            "patch29_max_skill_count",
            int(merged["patch29_skill_count"].max())
            if complete_n else 0,
        ),
        (
            "median_skill_jaccard",
            round(float(merged["skill_jaccard"].median()), 4)
            if complete_n else 0,
        ),
    ]

    pd.DataFrame(metrics, columns=["metric", "value"]).to_csv(
        SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    changed = merged.loc[
        merged["nonzero_to_zero"]
        | merged["zero_to_nonzero"]
        | merged["large_drop"]
        | merged["large_growth"]
    ].copy()

    changed["review_priority"] = (
        changed["nonzero_to_zero"].astype(int) * 100
        + changed["large_drop"].astype(int) * 50
        + changed["large_growth"].astype(int) * 20
        + changed["zero_to_nonzero"].astype(int) * 10
        + changed["count_delta"].abs()
    )

    changed.sort_values(
        ["review_priority", "record_id"],
        ascending=[False, True],
        inplace=True,
    )

    changed[
        [
            "record_id",
            "pool",
            "patch28_skill_count",
            "patch29_skill_count",
            "count_delta",
            "skill_jaccard",
            "patch28_zero",
            "patch29_zero",
            "zero_to_nonzero",
            "nonzero_to_zero",
            "large_drop",
            "large_growth",
            "patch28_skills",
            "patch29_skills",
        ]
    ].head(100).to_csv(
        CHANGED_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    return merged, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--records-per-shard",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    if args.records_per_shard <= 0:
        raise ValueError("--records-per-shard must be > 0")
    if args.max_shards is not None and args.max_shards <= 0:
        raise ValueError("--max-shards must be > 0")

    candidate_module, candidate_sha = load_candidate_module()

    manifest = pd.read_csv(
        FULL_MANIFEST,
        dtype={"record_id": str},
        keep_default_na=False,
        low_memory=False,
    )
    manifest["record_id"] = manifest["record_id"].astype(str).str.strip()
    manifest["already_processed_bool"] = parse_bool(
        manifest["already_processed"]
    )

    exposed_manifest = manifest.loc[
        manifest["already_processed_bool"]
    ][["run_order", "record_id", "pool"]].copy()

    if len(exposed_manifest) != EXPECTED_EXPOSED:
        raise RuntimeError(
            f"Expected {EXPECTED_EXPOSED} exposed records; "
            f"found {len(exposed_manifest)}."
        )
    if exposed_manifest["record_id"].nunique() != EXPECTED_EXPOSED:
        raise RuntimeError("Exposed record IDs are not unique.")

    blind = pd.read_csv(
        BLIND_MANIFEST,
        dtype={"record_id": str},
        keep_default_na=False,
        low_memory=False,
    )
    blind_ids = set(blind["record_id"].astype(str).str.strip())
    overlap = set(exposed_manifest["record_id"]) & blind_ids
    if overlap:
        raise RuntimeError(
            f"FATAL: exposed regression overlaps blind holdout: {len(overlap)} IDs"
        )

    engine = build_engine()
    try:
        db = fetch_exposed_rows(
            engine,
            exposed_manifest["record_id"].tolist(),
        )
    finally:
        engine.dispose()

    if len(db) != EXPECTED_EXPOSED:
        raise RuntimeError(
            f"Expected {EXPECTED_EXPOSED} DB rows; found {len(db)}."
        )

    exposed = exposed_manifest.merge(
        db,
        on="record_id",
        how="left",
        validate="one_to_one",
    )
    exposed.sort_values("run_order", inplace=True)

    existing = load_existing_outputs()
    completed_ids = set(existing["record_id"]) if not existing.empty else set()

    pending = exposed.loc[
        ~exposed["record_id"].isin(completed_ids)
    ].copy()

    print("Patch-29 v5 exposed regression runner")
    print("Mode: READ-ONLY MySQL / OpenAI inference")
    print(f"Candidate SHA: {candidate_sha}")
    print(f"Exact exposed universe: {len(exposed):,}")
    print(f"Blind holdout overlap: 0")
    print(f"Already completed regression outputs: {len(completed_ids):,}")
    print(f"Still pending: {len(pending):,}")
    print(f"Shard size: {args.records_per_shard}")

    if pending.empty:
        print("No exposed records remain to run.")
        outputs = load_existing_outputs()
        merged, _ = build_comparison(exposed, outputs, candidate_sha)
        print(f"Comparison records: {len(merged):,}")
        print(f"Summary: {SUMMARY_CSV}")
        return

    shards = [
        pending.iloc[i:i + args.records_per_shard].copy()
        for i in range(0, len(pending), args.records_per_shard)
    ]

    if args.max_shards is not None:
        shards = shards[:args.max_shards]

    print(f"Shards this invocation: {len(shards):,}")
    print()

    extractor = candidate_module.SkillsExtractor()
    invocation_success = 0
    invocation_failed = 0

    for shard_index, shard in enumerate(shards, start=1):
        capture: Dict[str, List[str]] = {}

        shard_df = shard[["record_id", "description"]].copy()
        shard_df.rename(columns={"record_id": "id"}, inplace=True)

        def fake_get_all_records(self):
            return shard_df[["id", "description"]].copy(), ID_COLUMN

        def capture_update(self, record_id, skills, id_field="id"):
            capture[str(record_id)] = list(skills)
            return True

        extractor.get_all_records = types.MethodType(
            fake_get_all_records,
            extractor,
        )
        extractor.update_record_skills = types.MethodType(
            capture_update,
            extractor,
        )

        print("=" * 72)
        print(
            f"Regression shard {shard_index}/{len(shards)} | "
            f"records={len(shard)}"
        )

        stats = extractor.process_all_records()

        expected_ids = set(shard["record_id"].astype(str))
        captured_ids = set(capture)

        missing_capture = expected_ids - captured_ids
        unexpected_capture = captured_ids - expected_ids

        if missing_capture or unexpected_capture:
            raise RuntimeError(
                "Regression capture mismatch. "
                f"missing={len(missing_capture)}, "
                f"unexpected={len(unexpected_capture)}"
            )

        save_outputs(capture)

        append_progress(
            {
                "candidate_sha256": candidate_sha,
                "shard_records": len(shard),
                "captured_records": len(capture),
                "extractor_total": stats.get("total", ""),
                "extractor_success": stats.get("success", ""),
                "extractor_failed": stats.get("failed", ""),
                "status": (
                    "COMPLETE"
                    if (
                        len(capture) == len(shard)
                        and int(stats.get("failed", 0)) == 0
                    )
                    else "CHECK"
                ),
            }
        )

        invocation_success += len(capture)
        invocation_failed += int(stats.get("failed", 0))

        outputs = load_existing_outputs()
        merged, metrics = build_comparison(
            exposed,
            outputs,
            candidate_sha,
        )

        print(
            f"Shard captured: {len(capture)}/{len(shard)} | "
            f"extractor failed={stats.get('failed', 0)}"
        )
        print(
            f"Regression outputs complete overall: "
            f"{len(outputs):,}/{EXPECTED_EXPOSED:,}"
        )

        metric_map = dict(metrics)
        print(
            "Current comparison: "
            f"zero→nonzero={metric_map['zero_to_nonzero']}, "
            f"nonzero→zero={metric_map['nonzero_to_zero']}, "
            f"large_drop={metric_map['large_drop_records']}, "
            f"large_growth={metric_map['large_growth_records']}"
        )
        print()

    print("=" * 72)
    print("Invocation complete.")
    print(f"Captured this invocation: {invocation_success:,}")
    print(f"Extractor failures this invocation: {invocation_failed:,}")
    print(f"Outputs: {OUTPUT_CSV}")
    print(f"Progress: {PROGRESS_CSV}")
    print(f"Summary: {SUMMARY_CSV}")
    print(f"Changed top 100: {CHANGED_CSV}")
    print()
    print("No MySQL rows were modified.")
    print("The 288-record Patch-29 blind holdout was not touched.")


if __name__ == "__main__":
    main()
