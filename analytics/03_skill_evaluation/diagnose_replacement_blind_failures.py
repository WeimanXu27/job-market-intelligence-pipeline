"""Read-only diagnostic for exposed replacement-blind failures.

This script diagnoses a small fixed set of already-exposed records after the
Patch-29 v18 failed replacement-blind post-hoc review. It never updates MySQL and it must not
access the replacement v6/v7 blind holdout.
"""
from __future__ import annotations

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
    raise RuntimeError("Could not locate project root containing src/.")

SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CANDIDATE_SOURCE = SRC_DIR / "reliability_replacement_blind_candidate.py"
EXPECTED_SHA256 = "93d3187591b0a93c7fcea4c02ae673ba1778a49e9c9469e7bc6833f1de87e8a8"
REPLACEMENT_BLIND = DATA_DIR / "skill_patch29_v6_blind_holdout_manifest.csv"
EXPECTED_BLIND_SHA256 = "217a3fdfd770068292b2b41de19d6d81a11854ea38615031e71df4a732a76399"
MANIFEST = DATA_DIR / "skill_patch29_v18_failed_blind_diagnostic_manifest.csv"
OUTPUT_PREFIX = DATA_DIR / "skill_patch29_v18_failed_blind_stage_diagnostic"

_spec = importlib.util.spec_from_file_location("reliability_replacement_blind_candidate", CANDIDATE_SOURCE)
if _spec is None or _spec.loader is None:
    raise RuntimeError("Could not load Patch-29 v18 candidate module.")
skill_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(skill_module)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_runtime_config() -> None:
    db = skill_module.DATABASE_CONFIG
    required = {k: db.get(k) for k in ("host", "port", "database", "username", "password")}
    missing = [k for k, v in required.items() if v in (None, "")]
    if missing:
        raise ValueError("Missing MySQL configuration in repo-root .env: " + ", ".join(missing))
    if not skill_module.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is missing from repo-root .env.")
    if db.get("database") != "job_market_portfolio":
        raise ValueError("DATABASE_NAME must be job_market_portfolio.")
    if skill_module.TABLE_NAME != "job_postings":
        raise ValueError("Expected TABLE_NAME='job_postings'.")


def fetch_descriptions(extractor: Any, record_ids: list[str]) -> pd.DataFrame:
    query = text(
        f"SELECT {skill_module.ID_COLUMN}, {skill_module.DESCRIPTION_COLUMN} "
        f"FROM {skill_module.TABLE_NAME} WHERE {skill_module.ID_COLUMN} IN :record_ids"
    ).bindparams(bindparam("record_ids", expanding=True))
    with extractor.db_engine.connect() as conn:
        rows = conn.execute(query, {"record_ids": record_ids}).fetchall()
    frame = pd.DataFrame(rows, columns=["record_id", "description"])
    frame["record_id"] = frame["record_id"].astype(str)
    missing = sorted(set(record_ids) - set(frame["record_id"]))
    if missing:
        raise RuntimeError("Diagnostic IDs missing from MySQL: " + ", ".join(missing))
    order = {rid: i for i, rid in enumerate(record_ids)}
    frame["_order"] = frame["record_id"].map(order)
    return frame.sort_values("_order", kind="stable").drop(columns="_order").reset_index(drop=True)


def scope_chunk_ids(record_id: str, chunks: list[str], ner_skills: list[str]) -> list[dict[str, Any]]:
    out = []
    for idx, chunk_text in enumerate(chunks, start=1):
        chunk_id = f"{record_id}__chunk_{idx:03d}"
        scoped = re.sub(r"\[(S\d{4})\]", lambda m: f"[{chunk_id}__{m.group(1)}]", chunk_text)
        low = scoped.lower()
        filtered_ner = [s for s in ner_skills if isinstance(s, str) and s.strip() and s.lower().strip() in low]
        out.append({"id": chunk_id, "text": scoped, "ner": filtered_ner})
    return out


def raw_items(raw_map: Any, chunk_id: str) -> list[dict[str, Any]]:
    if not isinstance(raw_map, dict):
        return []
    values = raw_map.get(chunk_id, [])
    if not isinstance(values, list):
        return []
    return [item for item in values if isinstance(item, dict)]


def compact_raw_items(raw_map: Any, chunk_id: str) -> list[dict[str, Any]]:
    out = []
    for item in raw_items(raw_map, chunk_id):
        out.append({
            "skill": str(item.get("skill", "") or "").strip(),
            "skill_type": str(item.get("skill_type", "") or "").strip(),
            "evidence_ids": item.get("evidence_ids", []),
        })
    return out


def call_primary_once(extractor: Any, records: list[dict[str, Any]]) -> dict[str, Any]:
    messages, input_ids, prompt_len = extractor._build_grounded_chunk_prompt(records)
    response = extractor.llm_client.chat.completions.create(
        model=extractor.openai_model, messages=messages, temperature=0.0
    )
    response_text = response.choices[0].message.content or ""
    raw_map = extractor._repair_json_to_dict(response_text) if response_text.strip() else {}
    validated, retry_ids = extractor._validate_grounded_chunk_map(raw_map, records)
    return {
        "input_ids": input_ids,
        "prompt_chars": prompt_len,
        "response_text": response_text,
        "raw_map": raw_map,
        "validated": validated,
        "retry_ids": set(retry_ids),
    }


def call_recovery_once(extractor: Any, record: dict[str, Any]) -> dict[str, Any]:
    messages, chunk_id, prompt_len = extractor._build_grounded_zero_recovery_prompt(record)
    response = extractor.llm_client.chat.completions.create(
        model=extractor.openai_model, messages=messages, temperature=0.0
    )
    response_text = response.choices[0].message.content or ""
    raw_map = extractor._repair_json_to_dict(response_text) if response_text.strip() else {}
    validated, retry_ids = extractor._validate_grounded_chunk_map(raw_map, [record])
    return {
        "input_ids": [chunk_id],
        "prompt_chars": prompt_len,
        "response_text": response_text,
        "raw_map": raw_map,
        "validated": validated,
        "retry_ids": set(retry_ids),
    }


def segment_flags(extractor: Any, chunk_text: str) -> list[dict[str, Any]]:
    rows = []
    for evidence_id, segment in extractor._segment_lookup(chunk_text).items():
        rows.append({
            "evidence_id": evidence_id,
            "segment": segment,
            "is_company_context": bool(extractor._is_company_context_segment(segment)),
            "is_employer_offering": bool(extractor._is_employer_offering_segment(segment)),
            "is_application_process": bool(extractor._is_application_process_segment(segment)),
            "is_general": bool(extractor._is_general_segment(segment)),
            "is_candidate_relevant_general": bool(extractor._is_candidate_relevant_general_segment(segment)),
        })
    return rows


def diagnostic_production_path(extractor: Any, records: list[dict[str, Any]]):
    final_by_chunk: dict[str, list[str]] = {}
    stage_by_chunk: dict[str, dict[str, Any]] = {}
    calls: list[dict[str, Any]] = []

    def recurse(batch: list[dict[str, Any]]) -> None:
        _, _, prompt_len = extractor._build_grounded_chunk_prompt(batch)
        if prompt_len > extractor.max_prompt_chars and len(batch) > 1:
            midpoint = max(1, len(batch) // 2)
            recurse(batch[:midpoint]); recurse(batch[midpoint:]); return

        initial = call_primary_once(extractor, batch)
        calls.append({"call_type": "initial_batch", "input_ids": initial["input_ids"], "prompt_chars": initial["prompt_chars"], "response_text": initial["response_text"]})
        lookup = {str(r["id"]): r for r in batch}

        for chunk_id in initial["input_ids"]:
            selected = list(initial["validated"].get(chunk_id, []))
            stage = {
                "initial_raw_items": compact_raw_items(initial["raw_map"], chunk_id),
                "initial_validated_skills": list(selected),
                "initial_retry_triggered": chunk_id in initial["retry_ids"],
                "retry1_raw_items": [], "retry1_validated_skills": [],
                "recovery_raw_items": [], "recovery_validated_skills": [],
            }

            if chunk_id in initial["retry_ids"]:
                record = lookup[chunk_id]
                retry1 = call_primary_once(extractor, [record])
                calls.append({"call_type": "isolated_primary_retry", "input_ids": retry1["input_ids"], "prompt_chars": retry1["prompt_chars"], "response_text": retry1["response_text"]})
                retry1_validated = list(retry1["validated"].get(chunk_id, []))
                stage["retry1_raw_items"] = compact_raw_items(retry1["raw_map"], chunk_id)
                stage["retry1_validated_skills"] = retry1_validated
                selected = extractor._select_more_complete_grounded_pass(selected, retry1_validated)

                record_text = str(record.get("text", "") or "")
                has_signal = bool(
                    extractor._meaningful_role_evidence_ids(record_text)
                    or extractor._has_generic_candidate_role_signal(record_text)
                )
                if not selected and has_signal:
                    recovery = call_recovery_once(extractor, record)
                    calls.append({"call_type": "zero_recovery_audit", "input_ids": recovery["input_ids"], "prompt_chars": recovery["prompt_chars"], "response_text": recovery["response_text"]})
                    recovery_validated = list(recovery["validated"].get(chunk_id, []))
                    stage["recovery_raw_items"] = compact_raw_items(recovery["raw_map"], chunk_id)
                    stage["recovery_validated_skills"] = recovery_validated
                    selected = extractor._select_more_complete_grounded_pass(selected, recovery_validated)

            stage["selected_skills"] = list(selected)
            stage["segment_flags"] = segment_flags(extractor, str(lookup[chunk_id].get("text", "") or ""))
            stage_by_chunk[chunk_id] = stage
            final_by_chunk[chunk_id] = list(selected)

    for start in range(0, len(records), extractor.batch_size):
        recurse(records[start:start + extractor.batch_size])
    return final_by_chunk, stage_by_chunk, calls


def main() -> None:
    actual_sha = sha256_file(CANDIDATE_SOURCE)
    print(f"Patch-29 v18 candidate SHA-256: {actual_sha}")
    if actual_sha.lower() != EXPECTED_SHA256:
        raise RuntimeError("Candidate SHA mismatch; aborting diagnostic.")
    if not REPLACEMENT_BLIND.exists():
        raise FileNotFoundError(f"Missing replacement blind manifest: {REPLACEMENT_BLIND}")
    blind_sha = sha256_file(REPLACEMENT_BLIND)
    print(f"Replacement blind SHA-256: {blind_sha}")
    if blind_sha.lower() != EXPECTED_BLIND_SHA256:
        raise RuntimeError("Replacement blind SHA mismatch; aborting diagnostic.")
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Missing diagnostic manifest: {MANIFEST}")

    manifest = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False)
    manifest["record_id"] = manifest["record_id"].astype(str).str.strip()
    if len(manifest) != 12 or manifest["record_id"].nunique() != 12:
        raise RuntimeError("Diagnostic manifest must contain exactly 12 unique exposed IDs.")

    blind = pd.read_csv(REPLACEMENT_BLIND, dtype=str, keep_default_na=False)
    blind_ids = set(blind["record_id"].astype(str).str.strip())
    manifest_ids = set(manifest["record_id"])
    overlap = sorted(manifest_ids & blind_ids)
    print(f"Replacement blind overlap: {len(overlap)} (EXPECTED: 12; post-blind diagnostic)")
    if len(overlap) != 12 or set(overlap) != manifest_ids:
        missing_from_blind = sorted(manifest_ids - blind_ids)
        unexpected_overlap = sorted(set(overlap) - manifest_ids)
        raise RuntimeError(
            "Post-blind diagnostic manifest integrity check failed. "
            f"Expected all 12 diagnostic IDs to be members of the now-exposed replacement blind. "
            f"overlap={len(overlap)}, missing_from_blind={missing_from_blind}, "
            f"unexpected_overlap={unexpected_overlap}"
        )

    validate_runtime_config()
    extractor = None
    try:
        extractor = skill_module.SkillsExtractor()
        ids = manifest["record_id"].tolist()
        source = fetch_descriptions(extractor, ids)
        source_lookup = {row["record_id"]: row for _, row in source.iterrows()}

        all_chunks = []
        record_data = {}
        for rid in ids:
            desc = str(source_lookup[rid]["description"] or "")
            ner = extractor.extract_skills_with_ner(desc)
            chunks = extractor.build_skill_evidence_chunks(desc, ner)
            scoped = scope_chunk_ids(rid, chunks, ner)
            all_chunks.extend(scoped)
            record_data[rid] = {"description": desc, "ner_skills": ner, "chunks": scoped}

        print(f"Diagnostic records: {len(ids)}; evidence chunks: {len(all_chunks)}")
        final_by_chunk, stage_by_chunk, calls = diagnostic_production_path(extractor, all_chunks)

        json_records = []
        summary_rows = []
        for _, mrow in manifest.iterrows():
            rid = mrow["record_id"]
            chunks = record_data[rid]["chunks"]
            chunk_ids = [c["id"] for c in chunks]
            selected = []
            for cid in chunk_ids:
                selected.extend(final_by_chunk.get(cid, []))
            final_skills = extractor.merge_and_deduplicate_skills([], selected)
            stages = [{"chunk_id": cid, **stage_by_chunk.get(cid, {})} for cid in chunk_ids]
            json_records.append({
                "record_id": rid,
                "case_group": mrow.get("case_group", ""),
                "exposure_reason": mrow.get("exposure_reason", ""),
                "description": record_data[rid]["description"],
                "ner_skills": record_data[rid]["ner_skills"],
                "chunks": chunks,
                "chunk_stages": stages,
                "diagnostic_final_skills": final_skills,
            })
            summary_rows.append({
                "record_id": rid,
                "case_group": mrow.get("case_group", ""),
                "chunk_count": len(chunk_ids),
                "initial_raw_count": sum(len(s.get("initial_raw_items", [])) for s in stages),
                "initial_validated_count": sum(len(s.get("initial_validated_skills", [])) for s in stages),
                "retry1_raw_count": sum(len(s.get("retry1_raw_items", [])) for s in stages),
                "retry1_validated_count": sum(len(s.get("retry1_validated_skills", [])) for s in stages),
                "recovery_raw_count": sum(len(s.get("recovery_raw_items", [])) for s in stages),
                "recovery_validated_count": sum(len(s.get("recovery_validated_skills", [])) for s in stages),
                "diagnostic_final_count": len(final_skills),
                "diagnostic_final_skills": " | ".join(final_skills),
            })

        out_json = OUTPUT_PREFIX.with_suffix(".json")
        out_csv = Path(str(OUTPUT_PREFIX) + "_summary.csv")
        out_json.write_text(json.dumps({
            "candidate_sha256": actual_sha,
            "replacement_blind_sha256": blind_sha,
            "replacement_blind_overlap": len(overlap),
            "replacement_blind_overlap_status": "EXPECTED_POST_BLIND_EXPOSED_12_OF_12",
            "record_count": len(json_records),
            "records": json_records,
            "llm_calls": calls,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame(summary_rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
        print("Diagnostic complete.")
        print(f"JSON: {out_json}")
        print(f"Summary: {out_csv}")
        print("READ-ONLY CHECK: no MySQL UPDATE executed.")
    finally:
        if extractor is not None and extractor.db_engine is not None:
            extractor.db_engine.dispose()


if __name__ == "__main__":
    main()
