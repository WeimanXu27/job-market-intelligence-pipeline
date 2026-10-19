"""Fast production runner that defers API-error shards instead of stopping.

Design goals
------------
- Keep frozen Patch-28 src/skill_extraction.py byte-for-byte unchanged.
- No artificial 10-second delay between normal OpenAI requests.
- If the SDK ultimately raises an API error, prevent the frozen extractor from
  converting it into an empty LLM result.
- Conservatively defer the ENTIRE current production shard, record its exact
  IDs in a durable CSV, and continue with later shards.
- On restart, previously deferred IDs are skipped so the main run can finish
  the rest of the corpus first.
- Final cleanup can reset/rerun only the deferred manifest.

Important
---------
A whole shard is deferred because one failed LLM call can contain chunks from
multiple records. This deliberately favors a slightly larger repair scope over
silent incomplete skill extraction.

Global billing/credit/spend-limit errors still stop the run because continuing
cannot succeed and would only defer every remaining shard.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from openai import RateLimitError

ROOT_DIR = Path(__file__).resolve().parents[2]
ANALYTICS_DIR = Path(__file__).resolve().parent

if str(ANALYTICS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYTICS_DIR))

import run_full_skill_extraction as base  # noqa: E402


DEFAULT_DEFERRED = (
    ROOT_DIR / "data" / "processed" / "skill_full_run_deferred_manifest.csv"
)
DEFAULT_PROGRESS = (
    ROOT_DIR / "data" / "processed" / "skill_full_run_fast_defer_progress.csv"
)

# The OpenAI Python SDK already performs its own retries. If a RateLimitError
# still escapes, do only one additional bounded retry before deferring the shard.
EXTRA_RATE_LIMIT_RETRIES = 1
DEFAULT_RETRY_SECONDS = 5.0
RETRY_BUFFER_SECONDS = 1.0


class DeferredLLMError(BaseException):
    """API failure that must bypass frozen extractor `except Exception` blocks."""


class GlobalProductionStop(BaseException):
    """Global account/billing failure for which later shards cannot succeed."""


def _parse_retry_seconds(message: str) -> float | None:
    patterns = [
        r"try again in\s+([0-9]*\.?[0-9]+)\s*s",
        r"retry after\s+([0-9]*\.?[0-9]+)\s*s",
    ]
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def _looks_like_global_billing_error(message: str) -> bool:
    lowered = message.lower()
    markers = [
        "insufficient_quota",
        "credit_balance_exhausted",
        "billing",
        "spend limit",
        "spend_limit",
        "organization_usage_limit_exceeded",
        "organization_spend_limit_exceeded",
    ]
    return any(marker in lowered for marker in markers)


class _FastDeferCompletions:
    def __init__(self, inner_completions) -> None:
        self._inner = inner_completions

    def create(self, *args, **kwargs):
        rate_limit_attempt = 0

        while True:
            try:
                return self._inner.create(*args, **kwargs)

            except RateLimitError as exc:
                message = str(exc)

                if _looks_like_global_billing_error(message):
                    raise GlobalProductionStop(
                        "Global billing/credit/spend-limit error. Continuing later "
                        "shards cannot succeed, so production is stopping safely.\n"
                        f"{message}"
                    ) from exc

                if rate_limit_attempt < EXTRA_RATE_LIMIT_RETRIES:
                    retry_seconds = _parse_retry_seconds(message)
                    wait_seconds = (
                        retry_seconds + RETRY_BUFFER_SECONDS
                        if retry_seconds is not None
                        else DEFAULT_RETRY_SECONDS
                    )
                    rate_limit_attempt += 1

                    print(
                        "\n[FAST-DEFER] Final SDK 429 escaped. "
                        f"Waiting {wait_seconds:.2f}s for one extra retry "
                        f"({rate_limit_attempt}/{EXTRA_RATE_LIMIT_RETRIES})."
                    )
                    base.skill_module.logger.warning(
                        "FAST-DEFER wrapper caught RateLimitError; "
                        "waiting %.2f seconds for bounded extra retry.",
                        wait_seconds,
                    )
                    time.sleep(max(wait_seconds, 1.0))
                    continue

                raise DeferredLLMError(
                    "OpenAI rate-limit error remained after bounded retries. "
                    "The current production shard must be deferred instead of "
                    "returning an empty LLM result.\n"
                    f"{message}"
                ) from exc

            except Exception as exc:
                raise DeferredLLMError(
                    "OpenAI API exception escaped the SDK retries. "
                    "The current production shard must be deferred instead of "
                    "returning an empty LLM result.\n"
                    f"{type(exc).__name__}: {exc}"
                ) from exc

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _FastDeferChat:
    def __init__(self, inner_chat) -> None:
        self._inner = inner_chat
        self.completions = _FastDeferCompletions(inner_chat.completions)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _FastDeferClient:
    def __init__(self, inner_client) -> None:
        self._inner = inner_client
        self.chat = _FastDeferChat(inner_client.chat)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class FastDeferProductionSkillsExtractor(base.ProductionSkillsExtractor):
    """Frozen Patch-28 extractor with non-silent, defer-on-API-error behavior."""

    def __init__(self) -> None:
        super().__init__()
        if self.llm_client is not None:
            self.llm_client = _FastDeferClient(self.llm_client)
            print(
                "[FAST-DEFER] No artificial request pacing. "
                "API failures are prevented from becoming empty results."
            )


base.ProductionSkillsExtractor = FastDeferProductionSkillsExtractor


def read_deferred_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    df = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    if "record_id" not in df.columns:
        raise RuntimeError(f"Deferred manifest missing record_id: {path}")

    ids = df["record_id"].astype(str).str.strip()
    return set(ids[ids.ne("")])


def append_deferred(
    path: Path,
    shard_ids: list[str],
    error_type: str,
    error_message: str,
    shard_index: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat(timespec="seconds")
    safe_message = " ".join(str(error_message).split())

    existing = read_deferred_ids(path)

    rows = [
        {
            "timestamp": timestamp,
            "record_id": record_id,
            "shard_index_this_invocation": shard_index,
            "error_type": error_type,
            "error_message": safe_message,
            "repair_status": "pending",
        }
        for record_id in shard_ids
        if record_id not in existing
    ]

    if not rows:
        return

    frame = pd.DataFrame(rows)
    if path.exists():
        frame.to_csv(path, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        frame.to_csv(path, index=False, encoding="utf-8-sig")


def append_fast_progress(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    if path.exists():
        frame.to_csv(path, mode="a", header=False, index=False)
    else:
        frame.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen Patch-28 production extraction quickly; defer API-error "
            "shards to a repair manifest instead of stopping."
        )
    )
    parser.add_argument("--manifest", type=Path, default=base.DEFAULT_MANIFEST)
    parser.add_argument(
        "--preflight-summary",
        type=Path,
        default=base.DEFAULT_PREFLIGHT,
    )
    parser.add_argument(
        "--progress-csv",
        type=Path,
        default=DEFAULT_PROGRESS,
    )
    parser.add_argument(
        "--deferred-csv",
        type=Path,
        default=DEFAULT_DEFERRED,
    )
    parser.add_argument(
        "--records-per-batch",
        type=int,
        default=100,
        help="Production shard size. Default: 100.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Stop after this many eligible shards; 0 means no limit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate state without API calls or DB writes.",
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
        raise FileNotFoundError(
            f"Preflight summary not found: {args.preflight_summary}"
        )

    print("Fast-defer full skill extraction runner")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'WRITE / RESUMABLE'}")
    print(f"Started: {datetime.now().isoformat(timespec='seconds')}")

    actual_sha = base.validate_frozen_extractor()
    print(f"PASS: frozen Patch-28 SHA-256 = {actual_sha}")

    base.validate_runtime_config()
    base.validate_preflight_summary(args.preflight_summary)
    manifest = base.read_and_validate_manifest(args.manifest)
    print(f"PASS: initial pending manifest = {len(manifest):,} unique IDs")
    print(f"Record shard size: {args.records_per_batch:,}")

    extractor = None

    try:
        extractor = base.ProductionSkillsExtractor()
        if extractor.llm_client is None:
            raise RuntimeError(
                "LLM client did not initialize; refusing production run."
            )

        manifest_ids = manifest["record_id"].tolist()

        pending_now: list[str] = []
        completed_now = 0
        state_chunk_size = 500

        for start in range(0, len(manifest_ids), state_chunk_size):
            ids = manifest_ids[start:start + state_chunk_size]
            state = base.fetch_state(extractor.db_engine, ids)
            pending_ids, completed_ids = base.classify_state(state, ids)
            pending_now.extend(pending_ids)
            completed_now += len(completed_ids)

        deferred_ids = read_deferred_ids(args.deferred_csv)
        manifest_id_set = set(manifest_ids)
        unexpected_deferred = deferred_ids - manifest_id_set
        if unexpected_deferred:
            raise RuntimeError(
                "Deferred manifest contains IDs outside the production manifest. "
                f"Example: {sorted(unexpected_deferred)[:5]}"
            )

        pending_set = set(pending_now)
        ordered_pending = [
            record_id
            for record_id in manifest_ids
            if record_id in pending_set
        ]
        eligible_pending = [
            record_id
            for record_id in ordered_pending
            if record_id not in deferred_ids
        ]
        pending_deferred = [
            record_id
            for record_id in ordered_pending
            if record_id in deferred_ids
        ]

        print("Live MySQL state inside initial 63,348 pending manifest:")
        print(f"  Already completed: {completed_now:,}")
        print(f"  Pending total: {len(ordered_pending):,}")
        print(f"  Pending already deferred for final repair: {len(pending_deferred):,}")
        print(f"  Eligible to process now: {len(eligible_pending):,}")

        if completed_now + len(ordered_pending) != base.EXPECTED_INITIAL_PENDING_ROWS:
            raise RuntimeError(
                "Live resume accounting does not sum to initial pending manifest."
            )

        if args.dry_run:
            print("\nDRY-RUN PASS")
            print("No MySQL rows were modified and no API calls were made.")
            return

        if not eligible_pending:
            print("\nNo non-deferred pending production records remain.")
            print(
                f"Deferred repair manifest contains "
                f"{len(pending_deferred):,} still-pending records."
            )
            return

        shards = [
            eligible_pending[i:i + args.records_per_batch]
            for i in range(0, len(eligible_pending), args.records_per_batch)
        ]
        planned = len(shards)

        if args.max_batches > 0:
            shards = shards[:args.max_batches]

        print(f"Eligible production shards planned: {planned:,}")
        if args.max_batches > 0:
            print(f"This invocation will run only: {len(shards):,} shard(s)")

        run_success = 0
        run_deferred = 0
        run_failed = 0
        invocation_start = datetime.now()

        for shard_index, shard_ids in enumerate(shards, start=1):
            print("\n" + "=" * 72)
            print(
                f"Production shard {shard_index}/{planned} | "
                f"manifest IDs: {len(shard_ids):,}"
            )

            before = base.fetch_state(extractor.db_engine, shard_ids)
            shard_pending, shard_completed = base.classify_state(
                before, shard_ids
            )

            if shard_completed:
                print(f"  Resume skip before shard: {len(shard_completed):,}")
            if not shard_pending:
                print("  Nothing pending in this shard; skipping.")
                continue

            started = datetime.now()
            extractor.set_current_ids(shard_pending)

            try:
                stats = extractor.process_all_records()

            except DeferredLLMError as exc:
                finished = datetime.now()

                # Conservative rule: defer the whole shard. Some rows may have
                # been persisted before the failed call; they remain included
                # so final repair can reset/rerun this exact scope.
                append_deferred(
                    args.deferred_csv,
                    shard_pending,
                    type(exc).__name__,
                    str(exc),
                    shard_index,
                )

                after = base.fetch_state(extractor.db_engine, shard_pending)
                still_pending, now_completed = base.classify_state(
                    after, shard_pending
                )

                run_deferred += len(shard_pending)

                append_fast_progress(
                    args.progress_csv,
                    {
                        "timestamp_start": started.isoformat(timespec="seconds"),
                        "timestamp_end": finished.isoformat(timespec="seconds"),
                        "shard_index_this_invocation": shard_index,
                        "planned_shards_at_start": planned,
                        "manifest_ids_in_shard": len(shard_ids),
                        "pending_before_processing": len(shard_pending),
                        "status": "DEFERRED_API_ERROR",
                        "persisted_before_defer": len(now_completed),
                        "still_pending_after_defer": len(still_pending),
                        "elapsed_seconds": int(
                            (finished - started).total_seconds()
                        ),
                        "error_type": type(exc).__name__,
                    },
                )

                print(
                    f"  [DEFERRED] API error. Conservatively logged "
                    f"{len(shard_pending):,} shard IDs for final repair."
                )
                print(
                    f"  Current DB state inside deferred shard: "
                    f"persisted={len(now_completed):,}, "
                    f"still_pending={len(still_pending):,}"
                )
                print("  Continuing to the next shard.")
                continue

            except GlobalProductionStop as exc:
                finished = datetime.now()

                append_deferred(
                    args.deferred_csv,
                    shard_pending,
                    type(exc).__name__,
                    str(exc),
                    shard_index,
                )

                append_fast_progress(
                    args.progress_csv,
                    {
                        "timestamp_start": started.isoformat(timespec="seconds"),
                        "timestamp_end": finished.isoformat(timespec="seconds"),
                        "shard_index_this_invocation": shard_index,
                        "planned_shards_at_start": planned,
                        "manifest_ids_in_shard": len(shard_ids),
                        "pending_before_processing": len(shard_pending),
                        "status": "GLOBAL_STOP",
                        "persisted_before_defer": "",
                        "still_pending_after_defer": "",
                        "elapsed_seconds": int(
                            (finished - started).total_seconds()
                        ),
                        "error_type": type(exc).__name__,
                    },
                )

                print("\n[GLOBAL STOP]")
                print(str(exc))
                print(
                    "Current shard IDs were logged for repair. "
                    "The run is stopping because later API calls cannot succeed."
                )
                break

            finished = datetime.now()

            after = base.fetch_state(extractor.db_engine, shard_pending)
            still_pending, now_completed = base.classify_state(
                after, shard_pending
            )

            append_fast_progress(
                args.progress_csv,
                {
                    "timestamp_start": started.isoformat(timespec="seconds"),
                    "timestamp_end": finished.isoformat(timespec="seconds"),
                    "shard_index_this_invocation": shard_index,
                    "planned_shards_at_start": planned,
                    "manifest_ids_in_shard": len(shard_ids),
                    "pending_before_processing": len(shard_pending),
                    "status": "COMPLETE",
                    "persisted_before_defer": len(now_completed),
                    "still_pending_after_defer": len(still_pending),
                    "elapsed_seconds": int(
                        (finished - started).total_seconds()
                    ),
                    "error_type": "",
                },
            )

            run_success += stats["success"]
            run_failed += stats["failed"]

            print(
                f"  Shard result: total={stats['total']:,}, "
                f"success={stats['success']:,}, failed={stats['failed']:,}, "
                f"still_pending={len(still_pending):,}"
            )

            # Non-API failures are not silently ignored. They remain visible
            # because they indicate an engineering/database problem rather than
            # the rate-limit condition this runner is designed to defer.
            if stats["failed"] > 0 or still_pending:
                raise RuntimeError(
                    "Shard had non-API extraction/database failures. "
                    "Stopping for review."
                )

        elapsed = datetime.now() - invocation_start
        current_deferred = read_deferred_ids(args.deferred_csv)

        print("\n" + "=" * 72)
        print("Invocation complete.")
        print(f"Successful updates this invocation: {run_success:,}")
        print(f"Deferred shard records this invocation: {run_deferred:,}")
        print(f"Non-API failed updates this invocation: {run_failed:,}")
        print(f"Deferred manifest unique IDs total: {len(current_deferred):,}")
        print(f"Elapsed: {elapsed}")
        print(f"Progress CSV: {args.progress_csv}")
        print(f"Deferred CSV: {args.deferred_csv}")
        print(
            "Re-running this command is safe: completed rows and previously "
            "deferred IDs are skipped."
        )

    finally:
        if extractor is not None and extractor.db_engine is not None:
            extractor.db_engine.dispose()


if __name__ == "__main__":
    main()
