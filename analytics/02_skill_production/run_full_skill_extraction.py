"""Run the frozen Patch-28 skill extractor on the exact pending production manifest.

This is a production orchestrator around src/skill_extraction.py. It does not
modify the frozen extractor. It processes the exact preflight-approved pending
manifest in small record shards so the run can be resumed safely after an
interruption without reprocessing rows whose extracted_skills value is already
non-NULL (including legitimate empty-string results).

Recommended workflow:
    1) python analytics/02_skill_production/run_full_skill_extraction.py --dry-run
    2) python analytics/02_skill_production/run_full_skill_extraction.py --max-batches 1
    3) python analytics/02_skill_production/run_full_skill_extraction.py

The script uses the existing repo-root .env through src/settings.py.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import bindparam, text


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import skill_extraction as skill_module  # noqa: E402


FROZEN_SHA256 = "e07f0fcf11e4d5ce0c9028590bb550e668ae722feeb058739e7b51f38fb05dc4"
EXPECTED_UNIVERSE_ROWS = 64_283
EXPECTED_INITIAL_PENDING_ROWS = 63_348
EXPECTED_PRECOMPLETED_ROWS = 935

ID_COLUMN = skill_module.ID_COLUMN
DESCRIPTION_COLUMN = skill_module.DESCRIPTION_COLUMN
SKILLS_COLUMN = skill_module.SKILLS_COLUMN
TABLE_NAME = skill_module.TABLE_NAME

DEFAULT_MANIFEST = ROOT_DIR / "data" / "processed" / "skill_full_run_pending_manifest.csv"
DEFAULT_PREFLIGHT = ROOT_DIR / "data" / "processed" / "skill_full_run_preflight_summary.csv"
DEFAULT_PROGRESS = ROOT_DIR / "data" / "processed" / "skill_full_run_progress.csv"


class ProductionSkillsExtractor(skill_module.SkillsExtractor):
    """Run the frozen extractor on one explicit production shard at a time."""

    def __init__(self) -> None:
        self.current_ids: list[str] = []
        super().__init__()

    def set_current_ids(self, record_ids: list[str]) -> None:
        self.current_ids = list(record_ids)

    def get_all_records(self):
        """Fetch exactly the currently assigned pending IDs."""
        if not self.current_ids:
            return pd.DataFrame(columns=["id", "description"]), ID_COLUMN

        query = text(
            f"SELECT {ID_COLUMN}, {DESCRIPTION_COLUMN} "
            f"FROM {TABLE_NAME} "
            f"WHERE {SKILLS_COLUMN} IS NULL "
            f"AND {DESCRIPTION_COLUMN} IS NOT NULL "
            f"AND TRIM({DESCRIPTION_COLUMN}) <> '' "
            f"AND {ID_COLUMN} IN :record_ids"
        ).bindparams(bindparam("record_ids", expanding=True))

        with self.db_engine.connect() as conn:
            rows = conn.execute(query, {"record_ids": self.current_ids}).fetchall()

        data = pd.DataFrame(rows, columns=[ID_COLUMN, DESCRIPTION_COLUMN])
        fetched_ids = set(data[ID_COLUMN].astype(str).tolist())
        expected_ids = set(self.current_ids)

        missing = sorted(expected_ids - fetched_ids)
        unexpected = sorted(fetched_ids - expected_ids)

        if missing:
            raise RuntimeError(
                "Current production shard is no longer fully pending. "
                "Stop and rerun the orchestrator so it can refresh DB state. "
                f"Example IDs: {', '.join(missing[:5])}"
            )

        if unexpected:
            raise RuntimeError(
                "Unexpected IDs were fetched for the production shard. "
                f"Example IDs: {', '.join(unexpected[:5])}"
            )

        order_map = {record_id: i for i, record_id in enumerate(self.current_ids)}
        data["id"] = data[ID_COLUMN].astype(str)
        data["_run_order"] = data["id"].map(order_map)
        data = data.sort_values("_run_order", kind="stable")

        print(f"  Production records fetched: {len(data):,}")
        return data[["id", DESCRIPTION_COLUMN]].rename(
            columns={DESCRIPTION_COLUMN: "description"}
        ), ID_COLUMN


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_runtime_config() -> None:
    db = skill_module.DATABASE_CONFIG
    required = {
        "host": db.get("host"),
        "port": db.get("port"),
        "database": db.get("database"),
        "username": db.get("username"),
        "password": db.get("password"),
    }
    missing = [key for key, value in required.items() if value in (None, "")]
    if missing:
        raise ValueError("Missing MySQL configuration in .env: " + ", ".join(missing))
    if not skill_module.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is missing from the repo-root .env.")
    if db["database"] != "job_market_portfolio":
        raise ValueError("DATABASE_NAME must be job_market_portfolio.")
    if TABLE_NAME != "job_postings":
        raise ValueError("DATA_CONFIG['table_name'] must be job_postings.")


def validate_frozen_extractor() -> str:
    extractor_path = SRC_DIR / "skill_extraction.py"
    actual = sha256_file(extractor_path)
    if actual.lower() != FROZEN_SHA256:
        raise RuntimeError(
            "Frozen extractor SHA mismatch. Do not run production extraction.\n"
            f"Expected: {FROZEN_SHA256}\nActual:   {actual}"
        )
    return actual


def read_and_validate_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    required = {"run_order", "record_id", "pool"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError("Pending manifest missing columns: " + ", ".join(sorted(missing)))

    manifest["record_id"] = manifest["record_id"].astype(str).str.strip()
    manifest["run_order"] = pd.to_numeric(manifest["run_order"], errors="raise").astype(int)
    manifest = manifest.sort_values("run_order", kind="stable").reset_index(drop=True)

    if len(manifest) != EXPECTED_INITIAL_PENDING_ROWS:
        raise RuntimeError(
            f"Pending manifest must contain exactly {EXPECTED_INITIAL_PENDING_ROWS:,} rows; "
            f"found {len(manifest):,}."
        )
    if not manifest["record_id"].is_unique:
        raise RuntimeError("Pending manifest record_id values are not unique.")
    if manifest["record_id"].eq("").any():
        raise RuntimeError("Pending manifest contains blank record_id values.")
    if not set(manifest["pool"]).issubset({"development", "final_test"}):
        raise RuntimeError("Pending manifest contains an unexpected pool value.")

    return manifest


def validate_preflight_summary(path: Path) -> None:
    summary = pd.read_csv(path)
    if len(summary) != 1:
        raise RuntimeError("Preflight summary must contain exactly one row.")
    row = summary.iloc[0]

    checks = {
        "extractor_sha256": str(row["extractor_sha256"]).lower() == FROZEN_SHA256,
        "usable_universe_rows": int(row["usable_universe_rows"]) == EXPECTED_UNIVERSE_ROWS,
        "known_patch28_manifest_rows": int(row["known_patch28_manifest_rows"]) == EXPECTED_PRECOMPLETED_ROWS,
        "usable_completed_rows": int(row["usable_completed_rows"]) == EXPECTED_PRECOMPLETED_ROWS,
        "usable_pending_rows": int(row["usable_pending_rows"]) == EXPECTED_INITIAL_PENDING_ROWS,
        "known_patch28_not_completed": int(row["known_patch28_not_completed"]) == 0,
        "unexpected_completed_usable_rows": int(row["unexpected_completed_usable_rows"]) == 0,
        "usable_ids_missing_from_mysql": int(row["usable_ids_missing_from_mysql"]) == 0,
        "usable_ids_invalid_description": int(row["usable_ids_invalid_description"]) == 0,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(
            "Preflight summary no longer matches the approved production state: "
            + ", ".join(failed)
        )


def fetch_state(engine, record_ids: list[str]) -> pd.DataFrame:
    if not record_ids:
        return pd.DataFrame(columns=["record_id", "description", "extracted_skills"])

    query = text(
        f"SELECT {ID_COLUMN}, {DESCRIPTION_COLUMN}, {SKILLS_COLUMN} "
        f"FROM {TABLE_NAME} "
        f"WHERE {ID_COLUMN} IN :record_ids"
    ).bindparams(bindparam("record_ids", expanding=True))

    with engine.connect() as conn:
        rows = conn.execute(query, {"record_ids": record_ids}).fetchall()

    return pd.DataFrame(rows, columns=["record_id", "description", "extracted_skills"])


def classify_state(state: pd.DataFrame, expected_ids: list[str]) -> tuple[list[str], list[str]]:
    expected = set(expected_ids)
    actual = set(state["record_id"].astype(str).tolist())
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        raise RuntimeError("Manifest IDs missing from MySQL. Example: " + ", ".join(missing[:5]))
    if unexpected:
        raise RuntimeError("Unexpected MySQL IDs returned. Example: " + ", ".join(unexpected[:5]))

    invalid_desc = state[
        state["description"].isna() | state["description"].astype(str).str.strip().eq("")
    ]["record_id"].astype(str).tolist()
    if invalid_desc:
        raise RuntimeError(
            "Manifest IDs now have invalid descriptions. Example: " + ", ".join(invalid_desc[:5])
        )

    pending = state.loc[state["extracted_skills"].isna(), "record_id"].astype(str).tolist()
    completed = state.loc[state["extracted_skills"].notna(), "record_id"].astype(str).tolist()
    return pending, completed


def append_progress(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    if path.exists():
        frame.to_csv(path, mode="a", header=False, index=False)
    else:
        frame.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frozen Patch-28 extraction on the exact pending production manifest."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--preflight-summary", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--progress-csv", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument(
        "--records-per-batch",
        type=int,
        default=100,
        help="Number of manifest records per resumable production shard (default: 100).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Stop after this many shards; 0 means no limit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate frozen code, manifests, and live MySQL state without running extraction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.records_per_batch <= 0:
        raise ValueError("--records-per-batch must be positive.")
    if args.max_batches < 0:
        raise ValueError("--max-batches cannot be negative.")
    if not args.manifest.exists():
        raise FileNotFoundError(f"Pending manifest not found: {args.manifest}")
    if not args.preflight_summary.exists():
        raise FileNotFoundError(f"Preflight summary not found: {args.preflight_summary}")

    print("Full skill extraction production runner")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'WRITE / RESUMABLE'}")
    print(f"Started: {datetime.now().isoformat(timespec='seconds')}")

    actual_sha = validate_frozen_extractor()
    print(f"PASS: frozen Patch-28 SHA-256 = {actual_sha}")

    validate_runtime_config()
    validate_preflight_summary(args.preflight_summary)
    manifest = read_and_validate_manifest(args.manifest)
    print(f"PASS: initial pending manifest = {len(manifest):,} unique IDs")
    print(f"Record shard size: {args.records_per_batch:,}")

    extractor = None
    try:
        extractor = ProductionSkillsExtractor()
        if extractor.llm_client is None:
            raise RuntimeError("LLM client did not initialize; refusing production run.")

        # Refresh the live state in safe chunks so resume counts do not rely on stale CSV state.
        pending_now: list[str] = []
        completed_now = 0
        manifest_ids = manifest["record_id"].tolist()
        state_chunk_size = 500

        for start in range(0, len(manifest_ids), state_chunk_size):
            ids = manifest_ids[start : start + state_chunk_size]
            state = fetch_state(extractor.db_engine, ids)
            pending_ids, completed_ids = classify_state(state, ids)
            pending_now.extend(pending_ids)
            completed_now += len(completed_ids)

        pending_set = set(pending_now)
        ordered_pending = [record_id for record_id in manifest_ids if record_id in pending_set]

        print("Live MySQL state inside initial 63,348 pending manifest:")
        print(f"  Already completed since preflight / prior resume: {completed_now:,}")
        print(f"  Still pending now: {len(ordered_pending):,}")

        if completed_now + len(ordered_pending) != EXPECTED_INITIAL_PENDING_ROWS:
            raise RuntimeError("Live resume accounting does not sum to the initial pending manifest.")

        if args.dry_run:
            print("\nDRY-RUN PASS")
            print("No MySQL rows were modified and no extraction API calls were made.")
            return

        if not ordered_pending:
            print("\nNo pending production records remain. Full extraction is already complete.")
            return

        shards = [
            ordered_pending[i : i + args.records_per_batch]
            for i in range(0, len(ordered_pending), args.records_per_batch)
        ]
        planned = len(shards)
        if args.max_batches > 0:
            shards = shards[: args.max_batches]

        print(f"Production shards planned from current state: {planned:,}")
        if args.max_batches > 0:
            print(f"This invocation will run only: {len(shards):,} shard(s)")

        run_success = 0
        run_failed = 0
        invocation_start = datetime.now()

        for shard_index, shard_ids in enumerate(shards, start=1):
            global_label = f"{shard_index}/{planned}"
            print("\n" + "=" * 72)
            print(f"Production shard {global_label} | manifest IDs: {len(shard_ids):,}")

            # Recheck immediately before work so a restart or concurrent write cannot cause reruns.
            before = fetch_state(extractor.db_engine, shard_ids)
            shard_pending, shard_completed = classify_state(before, shard_ids)

            if shard_completed:
                print(f"  Resume skip before shard: {len(shard_completed):,}")
            if not shard_pending:
                print("  Nothing pending in this shard; skipping.")
                continue

            started = datetime.now()
            extractor.set_current_ids(shard_pending)
            stats = extractor.process_all_records()
            finished = datetime.now()

            after = fetch_state(extractor.db_engine, shard_pending)
            still_pending, now_completed = classify_state(after, shard_pending)

            append_progress(
                args.progress_csv,
                {
                    "timestamp_start": started.isoformat(timespec="seconds"),
                    "timestamp_end": finished.isoformat(timespec="seconds"),
                    "shard_index_this_invocation": shard_index,
                    "planned_shards_at_start": planned,
                    "manifest_ids_in_shard": len(shard_ids),
                    "pending_before_processing": len(shard_pending),
                    "extractor_total": stats["total"],
                    "extractor_success": stats["success"],
                    "extractor_failed": stats["failed"],
                    "persisted_after_processing": len(now_completed),
                    "still_pending_after_processing": len(still_pending),
                    "elapsed_seconds": int((finished - started).total_seconds()),
                },
            )

            run_success += stats["success"]
            run_failed += stats["failed"]

            print(
                f"  Shard result: total={stats['total']:,}, "
                f"success={stats['success']:,}, failed={stats['failed']:,}, "
                f"still_pending={len(still_pending):,}"
            )

            if stats["failed"] > 0 or still_pending:
                raise RuntimeError(
                    "Production shard did not finish cleanly. Stop here; do not continue until reviewed."
                )

        elapsed = datetime.now() - invocation_start
        print("\n" + "=" * 72)
        print("Invocation complete.")
        print(f"Successful updates this invocation: {run_success:,}")
        print(f"Failed updates this invocation: {run_failed:,}")
        print(f"Elapsed: {elapsed}")
        print(f"Progress CSV: {args.progress_csv}")
        print("Re-running this same command is safe: non-NULL rows are skipped.")

    finally:
        if extractor is not None and extractor.db_engine is not None:
            extractor.db_engine.dispose()


if __name__ == "__main__":
    main()
