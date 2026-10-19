"""Read-only diagnostic for exposed first-blind failures.

IMPORTANT STATE CHANGE
----------------------
The original 288-record Patch-29 blind holdout was opened after the v5
candidate was frozen and the blind evaluation failed. All 288 IDs are now
EXPOSED. This script is therefore allowed to inspect and re-run those records,
but it must never be used to describe them as a clean blind set again.

This diagnostic does NOT update MySQL. It reproduces the v5 grounded extraction
retry/selection path while capturing intermediate raw and validated responses so
we can determine whether failures arise from:

1. source/evidence parsing,
2. initial LLM extraction,
3. grounding/schema validation,
4. isolated confirmation / retry selection, or
5. residual stochastic partial under-recall.

The manifest contains only QA metadata. It is never passed into LLM prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import bindparam, text


SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "src").is_dir():
    PROJECT_ROOT = SCRIPT_DIR
elif (SCRIPT_DIR.parents[1] / "src").is_dir():
    PROJECT_ROOT = SCRIPT_DIR.parents[1]
else:
    raise RuntimeError(
        "Could not locate project root containing src/. Place this script in "
        "the project root or analytics/."
    )

SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CANDIDATE_SOURCE = SRC_DIR / "reliability_first_blind_candidate.py"
EXPECTED_V5_SHA256 = (
    "932c80eaac297c9c295b719744a9e9a6"
    "df716c87481d505af9cbd9906608770d"
)
EXPOSED_REGISTRY = DATA_DIR / "skill_patch29_v5_failed_blind_exposed_ids.csv"

if not CANDIDATE_SOURCE.exists():
    raise FileNotFoundError(f"Patch-29 v5 candidate not found: {CANDIDATE_SOURCE}")

_spec = importlib.util.spec_from_file_location(
    "reliability_first_blind_candidate",
    CANDIDATE_SOURCE,
)
if _spec is None or _spec.loader is None:
    raise RuntimeError("Could not load Patch-29 v5 candidate module.")
skill_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(skill_module)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_runtime_config() -> None:
    database_config = skill_module.DATABASE_CONFIG
    required_database_fields = {
        "host": database_config.get("host"),
        "port": database_config.get("port"),
        "database": database_config.get("database"),
        "username": database_config.get("username"),
        "password": database_config.get("password"),
    }
    missing = [key for key, value in required_database_fields.items() if value in (None, "")]
    if missing:
        raise ValueError(
            "Missing MySQL configuration in the repo-root .env: " + ", ".join(missing)
        )
    if not skill_module.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is missing from the repo-root .env.")
    if database_config.get("database") != "job_market_portfolio":
        raise ValueError("DATABASE_NAME must be job_market_portfolio for this diagnostic.")
    if skill_module.TABLE_NAME != "job_postings":
        raise ValueError("This diagnostic expects TABLE_NAME='job_postings'.")


def fetch_descriptions(extractor: Any, record_ids: list[str]) -> pd.DataFrame:
    query = text(
        f"SELECT {skill_module.ID_COLUMN}, {skill_module.DESCRIPTION_COLUMN} "
        f"FROM {skill_module.TABLE_NAME} "
        f"WHERE {skill_module.ID_COLUMN} IN :record_ids"
    ).bindparams(bindparam("record_ids", expanding=True))

    with extractor.db_engine.connect() as conn:
        rows = conn.execute(query, {"record_ids": record_ids}).fetchall()

    frame = pd.DataFrame(rows, columns=["record_id", "description"])
    frame["record_id"] = frame["record_id"].astype(str)
    missing = sorted(set(record_ids) - set(frame["record_id"]))
    if missing:
        raise RuntimeError("Diagnostic IDs missing from MySQL: " + ", ".join(missing))

    order = {record_id: i for i, record_id in enumerate(record_ids)}
    frame["_order"] = frame["record_id"].map(order)
    return frame.sort_values("_order", kind="stable").drop(columns="_order").reset_index(drop=True)


def scope_chunk_ids(record_id: str, chunks: list[str], ner_skills: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, chunk_text in enumerate(chunks, start=1):
        chunk_id = f"{record_id}__chunk_{index:03d}"
        scoped_text = re.sub(
            r"\[(S\d{4})\]",
            lambda match: f"[{chunk_id}__{match.group(1)}]",
            chunk_text,
        )
        chunk_low = scoped_text.lower()
        filtered_ner = [
            skill
            for skill in ner_skills
            if isinstance(skill, str)
            and skill.strip()
            and skill.lower().strip() in chunk_low
        ]
        records.append({"id": chunk_id, "text": scoped_text, "ner": filtered_ner})
    return records


def raw_items(raw_map: Any, chunk_id: str) -> list[dict[str, Any]]:
    if not isinstance(raw_map, dict):
        return []
    values = raw_map.get(chunk_id, [])
    if not isinstance(values, list):
        return []
    return [item for item in values if isinstance(item, dict)]


def raw_skill_names(raw_map: Any, chunk_id: str) -> list[str]:
    names: list[str] = []
    for item in raw_items(raw_map, chunk_id):
        skill = str(item.get("skill", "") or "").strip()
        if skill:
            names.append(skill)
    return names


def raw_schema_stats(raw_map: Any, chunk_id: str) -> dict[str, int]:
    items = raw_items(raw_map, chunk_id)
    missing_type = 0
    missing_evidence = 0
    missing_skill = 0
    for item in items:
        if not str(item.get("skill", "") or "").strip():
            missing_skill += 1
        if not str(item.get("skill_type", "") or "").strip():
            missing_type += 1
        evidence = item.get("evidence_ids", [])
        if not isinstance(evidence, list) or not evidence:
            missing_evidence += 1
    return {
        "raw_item_count": len(items),
        "raw_missing_skill_count": missing_skill,
        "raw_missing_skill_type_count": missing_type,
        "raw_missing_evidence_count": missing_evidence,
    }


def call_raw_once(extractor: Any, records: list[dict[str, Any]]) -> dict[str, Any]:
    messages, input_ids, prompt_len = extractor._build_grounded_chunk_prompt(records)
    response = extractor.llm_client.chat.completions.create(
        model=extractor.openai_model,
        messages=messages,
        temperature=0.0,
    )
    response_text = response.choices[0].message.content or ""
    if not response_text.strip():
        raw_map: dict[str, Any] = {}
    else:
        raw_map = extractor._repair_json_to_dict(response_text)

    validated, retry_ids = extractor._validate_grounded_chunk_map(raw_map, records)
    return {
        "input_ids": input_ids,
        "prompt_chars": prompt_len,
        "response_text": response_text,
        "raw_map": raw_map,
        "validated": validated,
        "retry_ids": set(retry_ids),
    }


def diagnostic_production_path(
    extractor: Any,
    records: list[dict[str, Any]],
) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Reproduce v5 retry/selection while retaining stage evidence.

    This intentionally mirrors batch_llm_extract_grounded_chunks, including
    prompt-size recursive splitting, one isolated retry for every retry_id, and
    one additional isolated confirmation for unresolved substantive zeros.
    """
    final_by_chunk: dict[str, list[str]] = {}
    stage_by_chunk: dict[str, dict[str, Any]] = {}
    calls: list[dict[str, Any]] = []

    def recurse(batch: list[dict[str, Any]]) -> None:
        _, _, prompt_len = extractor._build_grounded_chunk_prompt(batch)
        if prompt_len > extractor.max_prompt_chars and len(batch) > 1:
            midpoint = max(1, len(batch) // 2)
            recurse(batch[:midpoint])
            recurse(batch[midpoint:])
            return

        initial = call_raw_once(extractor, batch)
        calls.append({
            "call_type": "initial_batch",
            "input_ids": initial["input_ids"],
            "prompt_chars": initial["prompt_chars"],
            "response_text": initial["response_text"],
        })

        lookup = {str(record["id"]): record for record in batch}
        for chunk_id in initial["input_ids"]:
            initial_validated = list(initial["validated"].get(chunk_id, []))
            stage = {
                "initial_raw_skills": raw_skill_names(initial["raw_map"], chunk_id),
                "initial_validated_skills": initial_validated,
                "initial_retry_triggered": chunk_id in initial["retry_ids"],
                "initial_schema": raw_schema_stats(initial["raw_map"], chunk_id),
                "retry1_raw_skills": [],
                "retry1_validated_skills": [],
                "retry1_schema": {},
                "retry2_raw_skills": [],
                "retry2_validated_skills": [],
                "retry2_schema": {},
            }
            selected = initial_validated

            if chunk_id in initial["retry_ids"]:
                record = lookup[chunk_id]
                retry1 = call_raw_once(extractor, [record])
                calls.append({
                    "call_type": "isolated_retry_1",
                    "input_ids": retry1["input_ids"],
                    "prompt_chars": retry1["prompt_chars"],
                    "response_text": retry1["response_text"],
                })
                retry1_validated = list(retry1["validated"].get(chunk_id, []))
                stage["retry1_raw_skills"] = raw_skill_names(retry1["raw_map"], chunk_id)
                stage["retry1_validated_skills"] = retry1_validated
                stage["retry1_schema"] = raw_schema_stats(retry1["raw_map"], chunk_id)
                selected = extractor._select_more_complete_grounded_pass(
                    selected,
                    retry1_validated,
                )

                record_text = str(record.get("text", "") or "")
                has_substantive_signal = bool(
                    extractor._meaningful_role_evidence_ids(record_text)
                    or extractor._has_generic_candidate_role_signal(record_text)
                )
                if not selected and has_substantive_signal:
                    retry2 = call_raw_once(extractor, [record])
                    calls.append({
                        "call_type": "isolated_retry_2",
                        "input_ids": retry2["input_ids"],
                        "prompt_chars": retry2["prompt_chars"],
                        "response_text": retry2["response_text"],
                    })
                    retry2_validated = list(retry2["validated"].get(chunk_id, []))
                    stage["retry2_raw_skills"] = raw_skill_names(retry2["raw_map"], chunk_id)
                    stage["retry2_validated_skills"] = retry2_validated
                    stage["retry2_schema"] = raw_schema_stats(retry2["raw_map"], chunk_id)
                    selected = extractor._select_more_complete_grounded_pass(
                        selected,
                        retry2_validated,
                    )

            stage["selected_skills"] = list(selected)
            stage_by_chunk[chunk_id] = stage
            final_by_chunk[chunk_id] = list(selected)

    for start in range(0, len(records), extractor.batch_size):
        recurse(records[start : start + extractor.batch_size])

    return final_by_chunk, stage_by_chunk, calls


def infer_failure_stage(
    evidence_chunk_count: int,
    meaningful_role_id_count: int,
    meaningful_role_word_count: int,
    chunk_stages: list[dict[str, Any]],
    final_skill_count: int,
) -> str:
    if evidence_chunk_count == 0:
        return "no_evidence_chunks"
    if meaningful_role_id_count == 0 and meaningful_role_word_count == 0:
        return "no_meaningful_role_evidence"

    all_raw_counts = []
    all_validated_counts = []
    for stage in chunk_stages:
        for prefix in ("initial", "retry1", "retry2"):
            raw = stage.get(f"{prefix}_raw_skills", []) or []
            validated = stage.get(f"{prefix}_validated_skills", []) or []
            if raw or prefix == "initial":
                all_raw_counts.append(len(raw))
                all_validated_counts.append(len(validated))

    if final_skill_count == 0:
        if all_raw_counts and max(all_raw_counts) == 0:
            return "llm_zero_across_observed_passes"
        if all_raw_counts and max(all_raw_counts) > 0 and max(all_validated_counts or [0]) == 0:
            return "grounding_validation_collapse"
        return "unresolved_zero_after_confirmation"

    if meaningful_role_id_count >= 6 and final_skill_count <= 3:
        return "severe_partial_under_recall"
    return "recovered_or_acceptable"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose the now-exposed failed Patch-29 v5 blind risk set."
    )
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("output_prefix", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    actual_hash = sha256_file(CANDIDATE_SOURCE)
    print(f"Patch-29 v5 candidate SHA-256: {actual_hash}")
    if actual_hash.lower() != EXPECTED_V5_SHA256:
        raise RuntimeError(
            "Candidate SHA mismatch. Diagnostic aborted. Expected frozen v5: "
            + EXPECTED_V5_SHA256
        )

    if not EXPOSED_REGISTRY.exists():
        raise FileNotFoundError(
            f"Missing failed-blind exposure registry: {EXPOSED_REGISTRY}"
        )

    registry = pd.read_csv(EXPOSED_REGISTRY, dtype=str, keep_default_na=False)
    if len(registry) != 288 or registry["record_id"].nunique() != 288:
        raise RuntimeError("Exposure registry must contain exactly 288 unique failed-blind IDs.")
    exposed_ids = set(registry["record_id"].astype(str).str.strip())

    manifest = pd.read_csv(args.manifest_csv, dtype=str, keep_default_na=False)
    required = {"record_id", "case_group", "blind_group", "blind_v5_skill_count"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError("Manifest missing columns: " + ", ".join(sorted(missing)))
    manifest["record_id"] = manifest["record_id"].astype(str).str.strip()
    if manifest["record_id"].eq("").any() or not manifest["record_id"].is_unique:
        raise ValueError("Diagnostic record IDs must be nonblank and unique.")
    outside = set(manifest["record_id"]) - exposed_ids
    if outside:
        raise RuntimeError(
            "Diagnostic contains IDs outside the failed-blind exposed registry: "
            + ", ".join(sorted(outside))
        )

    print(f"Failed-blind exposed registry: 288/288 verified")
    print(f"Diagnostic records: {len(manifest)}")
    validate_runtime_config()

    extractor = None
    try:
        extractor = skill_module.SkillsExtractor()
        record_ids = manifest["record_id"].tolist()
        source = fetch_descriptions(extractor, record_ids)
        source_lookup = {row["record_id"]: row for _, row in source.iterrows()}
        manifest_lookup = {row["record_id"]: row for _, row in manifest.iterrows()}

        print("\nStep 1/3: NER + deterministic evidence construction...")
        record_stage: dict[str, dict[str, Any]] = {}
        all_chunks: list[dict[str, Any]] = []
        chunk_parent: dict[str, str] = {}
        chunk_literal_items: dict[str, list[str]] = {}

        for record_id in record_ids:
            description = str(source_lookup[record_id]["description"] or "")
            ner_skills = extractor.extract_skills_with_ner(description)
            chunks = extractor.build_skill_evidence_chunks(description, ner_skills)
            chunk_records = scope_chunk_ids(record_id, chunks, ner_skills)
            literal_items: list[str] = []
            meaningful_ids: set[str] = set()
            meaningful_words = 0

            for chunk in chunk_records:
                chunk_id = str(chunk["id"])
                all_chunks.append(chunk)
                chunk_parent[chunk_id] = record_id
                recovered = extractor._extract_literal_list_skills_from_evidence(str(chunk["text"]))
                chunk_literal_items[chunk_id] = recovered
                literal_items.extend(recovered)
                meaningful_ids |= extractor._meaningful_role_evidence_ids(str(chunk["text"]))
                meaningful_words += extractor._meaningful_role_evidence_word_count(str(chunk["text"]))

            literal_items = extractor.merge_and_deduplicate_skills([], literal_items)
            row = manifest_lookup[record_id]
            record_stage[record_id] = {
                "record_id": record_id,
                "case_group": str(row["case_group"]),
                "blind_group": str(row["blind_group"]),
                "blind_v5_skill_count": int(str(row["blind_v5_skill_count"]) or 0),
                "blind_v5_skills": str(row.get("blind_v5_skills", "")),
                "title": str(row.get("title", "")),
                "description": description,
                "ner_skills": ner_skills,
                "chunks": chunk_records,
                "literal_items": literal_items,
                "meaningful_role_id_count": len(meaningful_ids),
                "meaningful_role_word_count": meaningful_words,
            }

        print(f"Evidence chunks: {len(all_chunks)}")
        print("Step 2/3: exact v5 retry/selection path with stage capture...")
        final_by_chunk, stage_by_chunk, calls = diagnostic_production_path(extractor, all_chunks)

        print("Step 3/3: record-level failure classification + export...")
        json_records: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []

        for record_id in record_ids:
            stage = record_stage[record_id]
            chunk_ids = [str(chunk["id"]) for chunk in stage["chunks"]]
            selected_llm: list[str] = []
            initial_raw_count = 0
            initial_validated_count = 0
            retry1_raw_count = 0
            retry1_validated_count = 0
            retry2_raw_count = 0
            retry2_validated_count = 0
            missing_type_count = 0
            retry_trigger_count = 0
            chunk_stage_list: list[dict[str, Any]] = []

            for chunk_id in chunk_ids:
                selected_llm.extend(final_by_chunk.get(chunk_id, []))
                cstage = stage_by_chunk.get(chunk_id, {})
                chunk_stage_list.append({"chunk_id": chunk_id, **cstage})
                initial_raw_count += len(cstage.get("initial_raw_skills", []) or [])
                initial_validated_count += len(cstage.get("initial_validated_skills", []) or [])
                retry1_raw_count += len(cstage.get("retry1_raw_skills", []) or [])
                retry1_validated_count += len(cstage.get("retry1_validated_skills", []) or [])
                retry2_raw_count += len(cstage.get("retry2_raw_skills", []) or [])
                retry2_validated_count += len(cstage.get("retry2_validated_skills", []) or [])
                missing_type_count += int((cstage.get("initial_schema", {}) or {}).get("raw_missing_skill_type_count", 0))
                missing_type_count += int((cstage.get("retry1_schema", {}) or {}).get("raw_missing_skill_type_count", 0))
                missing_type_count += int((cstage.get("retry2_schema", {}) or {}).get("raw_missing_skill_type_count", 0))
                retry_trigger_count += int(bool(cstage.get("initial_retry_triggered", False)))

            final_skills = extractor.merge_and_deduplicate_skills(
                [],
                selected_llm + stage["literal_items"],
            )
            failure_stage = infer_failure_stage(
                len(chunk_ids),
                int(stage["meaningful_role_id_count"]),
                int(stage["meaningful_role_word_count"]),
                chunk_stage_list,
                len(final_skills),
            )

            json_records.append({
                **stage,
                "chunk_stages": chunk_stage_list,
                "diagnostic_final_skills": final_skills,
                "failure_stage": failure_stage,
            })
            summary_rows.append({
                "record_id": record_id,
                "case_group": stage["case_group"],
                "blind_group": stage["blind_group"],
                "title": stage["title"],
                "blind_v5_skill_count": stage["blind_v5_skill_count"],
                "evidence_chunk_count": len(chunk_ids),
                "meaningful_role_id_count": stage["meaningful_role_id_count"],
                "meaningful_role_word_count": stage["meaningful_role_word_count"],
                "ner_skill_count": len(stage["ner_skills"]),
                "literal_skill_count": len(stage["literal_items"]),
                "initial_raw_skill_count": initial_raw_count,
                "initial_validated_skill_count": initial_validated_count,
                "retry_trigger_count": retry_trigger_count,
                "retry1_raw_skill_count": retry1_raw_count,
                "retry1_validated_skill_count": retry1_validated_count,
                "retry2_raw_skill_count": retry2_raw_count,
                "retry2_validated_skill_count": retry2_validated_count,
                "raw_missing_skill_type_count": missing_type_count,
                "diagnostic_final_skill_count": len(final_skills),
                "failure_stage": failure_stage,
                "diagnostic_final_skills": " | ".join(final_skills),
                "blind_v5_skills": stage["blind_v5_skills"].replace("\n", " | "),
            })

        output_json = args.output_prefix.with_suffix(".json")
        output_csv = Path(str(args.output_prefix) + "_summary.csv")
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(
                {
                    "candidate_sha256": actual_hash,
                    "failed_blind_status": "EXPOSED_AFTER_FAILED_V5_BLIND",
                    "record_count": len(json_records),
                    "chunk_count": len(all_chunks),
                    "important_note": (
                        "These 288 blind IDs are no longer blind. QA manifest metadata was "
                        "not passed to any LLM prompt. No MySQL UPDATE was executed."
                    ),
                    "records": json_records,
                    "llm_calls": calls,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        pd.DataFrame(summary_rows).to_csv(output_csv, index=False, encoding="utf-8-sig")

        print("\nDiagnostic complete.")
        print(f"Records: {len(json_records)}")
        print(f"Chunks: {len(all_chunks)}")
        print(f"JSON: {output_json}")
        print(f"Summary CSV: {output_csv}")
        print("READ-ONLY CHECK: no database UPDATE statements were executed.")
        print("EXPOSURE CHECK: all diagnostic IDs belong to the failed 288-record blind.")

    finally:
        if extractor is not None and extractor.db_engine is not None:
            extractor.db_engine.dispose()


if __name__ == "__main__":
    main()
