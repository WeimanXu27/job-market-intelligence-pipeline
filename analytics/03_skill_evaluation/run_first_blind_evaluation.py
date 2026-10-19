"""First frozen blind evaluation runner (READ-ONLY MySQL).

This runner is for the frozen 288-record Patch-29 blind holdout only.
The candidate must already be frozen before this script is run.

Blindness / safety rules
------------------------
- Verifies the exact frozen candidate SHA-256.
- Verifies the exact frozen blind-manifest SHA-256 and 288 unique IDs.
- Verifies zero overlap with the canonical 935 pre-production exposed IDs.
- Queries ONLY record_id + description from MySQL. It does NOT read Patch-28 skills.
- Does NOT export descriptions.
- Does NOT update MySQL.
- Does NOT print extracted skills or descriptions to the console.
- Is resumable and saves only clean, complete shards.
- If the extractor logs an LLM-call exception or a shard is incomplete, that
  shard is NOT saved and must be rerun. This prevents infrastructure/API
  failures from silently becoming blind false-zero results.

IMPORTANT
---------
Run all 288 blind records before reviewing outputs. Do not inspect a partial
blind result and do not change the candidate after this run begins.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import logging
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

BLIND_MANIFEST = DATA_DIR / "skill_patch29_blind_holdout_manifest.csv"
EXPECTED_BLIND_MANIFEST_SHA256 = (
    "e456953b4eb7009b0026a363b214d9e22c114868e58cbb17648fabfc8b367d41"
)
FULL_MANIFEST = DATA_DIR / "skill_full_run_manifest_64283.csv"

OUTPUT_CSV = DATA_DIR / "skill_patch29_v5_blind_outputs.csv"
PROGRESS_CSV = DATA_DIR / "skill_patch29_v5_blind_progress.csv"
SUMMARY_CSV = DATA_DIR / "skill_patch29_v5_blind_inference_summary.csv"

EXPECTED_BLIND = 288
EXPECTED_EXPOSED = 935

TABLE_NAME = DATA_CONFIG["table_name"]
ID_COLUMN = DATA_CONFIG["id_column"]
DESCRIPTION_COLUMN = DATA_CONFIG["description_column"]


LLM_EXCEPTION_LOG_MARKERS = (
    "llm call failed",
    "llm validation call failed",
    "single-record refinement retry failed",
    "role-content screening failed",
    "grounded chunk retry failed",
    "segment-grounded llm call failed",
)


class LLMExceptionCapture(logging.Handler):
    """Capture extractor LLM-call exceptions without exposing source text."""

    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.events: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:
            message = ""
        lowered = message.lower()
        if any(marker in lowered for marker in LLM_EXCEPTION_LOG_MARKERS):
            # Keep only the first line to avoid traceback/source leakage.
            self.events.append(message.splitlines()[0][:500])


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
        raise FileNotFoundError(f"Frozen candidate not found: {CANDIDATE_PATH}")

    actual_sha = sha256_file(CANDIDATE_PATH)
    if actual_sha.lower() != EXPECTED_CANDIDATE_SHA256:
        raise RuntimeError(
            "FROZEN candidate SHA mismatch.\n"
            f"Expected: {EXPECTED_CANDIDATE_SHA256}\n"
            f"Actual:   {actual_sha}\n"
            "Do not run the blind with a modified candidate."
        )

    spec = importlib.util.spec_from_file_location(
        "reliability_first_blind_candidate",
        CANDIDATE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load frozen Patch-29 v5 candidate.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, actual_sha


def load_and_verify_blind_manifest() -> pd.DataFrame:
    if not BLIND_MANIFEST.exists():
        raise FileNotFoundError(f"Missing frozen blind manifest: {BLIND_MANIFEST}")

    actual_sha = sha256_file(BLIND_MANIFEST)
    if actual_sha.lower() != EXPECTED_BLIND_MANIFEST_SHA256:
        raise RuntimeError(
            "FROZEN blind manifest SHA mismatch.\n"
            f"Expected: {EXPECTED_BLIND_MANIFEST_SHA256}\n"
            f"Actual:   {actual_sha}"
        )

    blind = pd.read_csv(
        BLIND_MANIFEST,
        dtype={"record_id": str},
        keep_default_na=False,
        low_memory=False,
    )
    required = {"blind_order", "record_id", "blind_group", "pool"}
    missing = required - set(blind.columns)
    if missing:
        raise ValueError(f"Blind manifest missing columns: {sorted(missing)}")

    blind["record_id"] = blind["record_id"].astype(str).str.strip()
    blind["blind_order"] = pd.to_numeric(
        blind["blind_order"], errors="raise"
    ).astype(int)

    if len(blind) != EXPECTED_BLIND:
        raise RuntimeError(
            f"Expected {EXPECTED_BLIND} blind records; found {len(blind)}."
        )
    if blind["record_id"].nunique() != EXPECTED_BLIND:
        raise RuntimeError("Blind record IDs are not unique.")

    full = pd.read_csv(
        FULL_MANIFEST,
        dtype={"record_id": str},
        keep_default_na=False,
        low_memory=False,
    )
    full["record_id"] = full["record_id"].astype(str).str.strip()
    full["already_processed_bool"] = parse_bool(full["already_processed"])
    exposed_ids = set(
        full.loc[full["already_processed_bool"], "record_id"]
    )
    if len(exposed_ids) != EXPECTED_EXPOSED:
        raise RuntimeError(
            f"Expected {EXPECTED_EXPOSED} canonical exposed IDs; "
            f"found {len(exposed_ids)}."
        )
    overlap = exposed_ids & set(blind["record_id"])
    if overlap:
        raise RuntimeError(
            f"FATAL: blind manifest overlaps exposed set: {len(overlap)} IDs"
        )

    return blind.sort_values("blind_order").reset_index(drop=True)


def fetch_blind_descriptions(engine, record_ids: List[str]) -> pd.DataFrame:
    # Deliberately DO NOT select the Patch-28 skills column here.
    query = text(
        f"SELECT {ID_COLUMN} AS record_id, "
        f"{DESCRIPTION_COLUMN} AS description "
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
        return pd.DataFrame(columns=["record_id", "description"])

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


def load_existing_outputs(blind: pd.DataFrame) -> pd.DataFrame:
    if not OUTPUT_CSV.exists():
        return pd.DataFrame(
            columns=[
                "blind_order",
                "record_id",
                "blind_group",
                "pool",
                "patch29_skills",
                "patch29_skill_count",
            ]
        )

    if not PROGRESS_CSV.exists():
        raise RuntimeError(
            "Blind outputs exist but progress metadata is missing. "
            "Do not resume an untracked blind run."
        )

    progress = pd.read_csv(PROGRESS_CSV, keep_default_na=False)
    for col, expected in (
        ("candidate_sha256", EXPECTED_CANDIDATE_SHA256),
        ("blind_manifest_sha256", EXPECTED_BLIND_MANIFEST_SHA256),
    ):
        if col not in progress.columns:
            raise RuntimeError(f"Blind progress missing {col}.")
        observed = set(progress[col].astype(str).str.strip())
        if observed != {expected}:
            raise RuntimeError(
                f"Blind progress {col} does not match the frozen value."
            )

    bad_status = progress.loc[progress["status"].ne("COMPLETE")]
    if not bad_status.empty:
        raise RuntimeError(
            "Blind progress contains a non-COMPLETE shard. Resolve/re-run it "
            "before continuing."
        )

    df = pd.read_csv(
        OUTPUT_CSV,
        dtype={"record_id": str},
        keep_default_na=False,
        low_memory=False,
    )
    required = {
        "blind_order",
        "record_id",
        "blind_group",
        "pool",
        "patch29_skills",
        "patch29_skill_count",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Existing blind output missing columns: {sorted(missing)}")

    df["record_id"] = df["record_id"].astype(str).str.strip()
    if df["record_id"].duplicated().any():
        raise RuntimeError("Existing blind outputs contain duplicate record IDs.")

    blind_ids = set(blind["record_id"])
    extra = set(df["record_id"]) - blind_ids
    if extra:
        raise RuntimeError(
            f"Existing blind outputs contain {len(extra)} IDs outside the frozen blind."
        )
    return df


def save_clean_shard(
    capture: Dict[str, List[str]],
    blind: pd.DataFrame,
) -> None:
    existing = load_existing_outputs(blind)
    meta = blind.set_index("record_id")

    new_rows = []
    for record_id, skills in capture.items():
        skill_list = skill_lines(skills)
        numbered = "\n".join(
            f"{i}. {skill}" for i, skill in enumerate(skill_list, start=1)
        )
        m = meta.loc[str(record_id)]
        new_rows.append(
            {
                "blind_order": int(m["blind_order"]),
                "record_id": str(record_id),
                "blind_group": m["blind_group"],
                "pool": m["pool"],
                "patch29_skills": numbered,
                "patch29_skill_count": len(skill_list),
            }
        )

    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates("record_id", keep="last")
    combined.sort_values("blind_order", inplace=True)
    combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")


def append_progress(row: dict) -> None:
    frame = pd.DataFrame([row])
    header = not PROGRESS_CSV.exists()
    frame.to_csv(
        PROGRESS_CSV,
        mode="a",
        header=header,
        index=False,
        encoding="utf-8-sig",
    )


def write_final_summary(outputs: pd.DataFrame) -> None:
    counts = pd.to_numeric(outputs["patch29_skill_count"], errors="raise").astype(int)
    metrics = [
        ("candidate_sha256", EXPECTED_CANDIDATE_SHA256),
        ("blind_manifest_sha256", EXPECTED_BLIND_MANIFEST_SHA256),
        ("expected_blind_records", EXPECTED_BLIND),
        ("output_records", len(outputs)),
        ("unique_record_ids", outputs["record_id"].nunique()),
        ("zero_records", int(counts.eq(0).sum())),
        ("mean_skill_count", round(float(counts.mean()), 4)),
        ("median_skill_count", float(counts.median())),
        ("p95_skill_count", float(counts.quantile(0.95))),
        ("max_skill_count", int(counts.max())),
    ]
    pd.DataFrame(metrics, columns=["metric", "value"]).to_csv(
        SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-per-shard", type=int, default=48)
    args = parser.parse_args()
    if args.records_per_shard <= 0:
        raise ValueError("--records-per-shard must be > 0")

    candidate_module, candidate_sha = load_candidate_module()
    blind = load_and_verify_blind_manifest()
    existing = load_existing_outputs(blind)
    completed_ids = set(existing["record_id"]) if not existing.empty else set()
    pending = blind.loc[~blind["record_id"].isin(completed_ids)].copy()

    print("Patch-29 v5 FINAL blind inference runner")
    print("Mode: READ-ONLY MySQL / frozen candidate / no Patch-28 skill reads")
    print(f"Candidate SHA: {candidate_sha}")
    print(f"Blind manifest SHA: {EXPECTED_BLIND_MANIFEST_SHA256}")
    print(f"Frozen blind universe: {len(blind)}")
    print("Exposed overlap: 0")
    print(f"Already completed blind outputs: {len(completed_ids)}")
    print(f"Still pending: {len(pending)}")
    print(f"Shard size: {args.records_per_shard}")
    print("DO NOT inspect partial blind outputs or modify the frozen candidate.")

    if pending.empty:
        if len(existing) != EXPECTED_BLIND:
            raise RuntimeError(
                f"No pending rows but only {len(existing)} blind outputs exist."
            )
        write_final_summary(existing)
        print("Blind inference already complete: 288/288.")
        print(f"Outputs: {OUTPUT_CSV}")
        print(f"Summary: {SUMMARY_CSV}")
        return

    # Source text is fetched only after the candidate + manifests are verified.
    engine = build_engine()
    try:
        db = fetch_blind_descriptions(engine, blind["record_id"].tolist())
    finally:
        engine.dispose()

    if len(db) != EXPECTED_BLIND:
        raise RuntimeError(
            f"Expected {EXPECTED_BLIND} blind DB rows; found {len(db)}."
        )

    blind_work = blind.merge(db, on="record_id", how="left", validate="one_to_one")
    blind_work.sort_values("blind_order", inplace=True)
    pending = blind_work.loc[~blind_work["record_id"].isin(completed_ids)].copy()

    shards = [
        pending.iloc[i:i + args.records_per_shard].copy()
        for i in range(0, len(pending), args.records_per_shard)
    ]

    extractor = candidate_module.SkillsExtractor()
    error_capture = LLMExceptionCapture()
    candidate_module.logger.addHandler(error_capture)

    try:
        for shard_index, shard in enumerate(shards, start=1):
            capture: Dict[str, List[str]] = {}
            events_before = len(error_capture.events)

            shard_df = shard[["record_id", "description"]].copy()
            shard_df.rename(columns={"record_id": "id"}, inplace=True)

            def fake_get_all_records(self):
                return shard_df[["id", "description"]].copy(), ID_COLUMN

            def capture_update(self, record_id, skills, id_field="id"):
                capture[str(record_id)] = list(skills)
                return True

            extractor.get_all_records = types.MethodType(fake_get_all_records, extractor)
            extractor.update_record_skills = types.MethodType(capture_update, extractor)

            print("=" * 72)
            print(
                f"Blind shard {shard_index}/{len(shards)} | "
                f"records={len(shard)}"
            )

            try:
                stats = extractor.process_all_records()
            except Exception as exc:
                append_progress(
                    {
                        "candidate_sha256": candidate_sha,
                        "blind_manifest_sha256": EXPECTED_BLIND_MANIFEST_SHA256,
                        "shard_records": len(shard),
                        "captured_records": len(capture),
                        "extractor_total": "",
                        "extractor_success": "",
                        "extractor_failed": "",
                        "llm_exception_events": "",
                        "status": "DEFERRED_EXCEPTION",
                    }
                )
                raise RuntimeError(
                    "Blind shard raised an exception. No outputs from this shard "
                    "were saved. Re-run after resolving the infrastructure issue."
                ) from exc

            new_events = error_capture.events[events_before:]
            expected_ids = set(shard["record_id"].astype(str))
            captured_ids = set(capture)
            missing_ids = expected_ids - captured_ids
            extra_ids = captured_ids - expected_ids
            failed_count = int(stats.get("failed", 0))

            clean = (
                not new_events
                and not missing_ids
                and not extra_ids
                and len(capture) == len(shard)
                and failed_count == 0
            )

            if not clean:
                append_progress(
                    {
                        "candidate_sha256": candidate_sha,
                        "blind_manifest_sha256": EXPECTED_BLIND_MANIFEST_SHA256,
                        "shard_records": len(shard),
                        "captured_records": len(capture),
                        "extractor_total": stats.get("total", ""),
                        "extractor_success": stats.get("success", ""),
                        "extractor_failed": stats.get("failed", ""),
                        "llm_exception_events": len(new_events),
                        "status": "DEFERRED_LLM_OR_INCOMPLETE",
                    }
                )
                raise RuntimeError(
                    "Blind shard was not clean. No outputs from this shard were "
                    "saved. Do not inspect/tune on this partial blind result; "
                    "resolve the infrastructure issue and rerun the same frozen "
                    "candidate."
                )

            # Save only after the entire shard passes infrastructure/integrity checks.
            save_clean_shard(capture, blind)
            append_progress(
                {
                    "candidate_sha256": candidate_sha,
                    "blind_manifest_sha256": EXPECTED_BLIND_MANIFEST_SHA256,
                    "shard_records": len(shard),
                    "captured_records": len(capture),
                    "extractor_total": stats.get("total", ""),
                    "extractor_success": stats.get("success", ""),
                    "extractor_failed": stats.get("failed", ""),
                    "llm_exception_events": 0,
                    "status": "COMPLETE",
                }
            )
            print(f"Blind shard {shard_index} saved cleanly.")

    finally:
        candidate_module.logger.removeHandler(error_capture)

    outputs = load_existing_outputs(blind)
    if len(outputs) != EXPECTED_BLIND:
        raise RuntimeError(
            f"Blind invocation ended with {len(outputs)}/{EXPECTED_BLIND} outputs. "
            "Re-run the same command to resume; do not inspect partial outputs."
        )

    write_final_summary(outputs)
    print("=" * 72)
    print("FINAL BLIND INFERENCE COMPLETE: 288/288")
    print("Candidate remained frozen throughout this run.")
    print(f"Outputs: {OUTPUT_CSV}")
    print(f"Progress: {PROGRESS_CSV}")
    print(f"Inference summary: {SUMMARY_CSV}")
    print("Now, and only now, the blind may be opened for manual evaluation.")


if __name__ == "__main__":
    main()
