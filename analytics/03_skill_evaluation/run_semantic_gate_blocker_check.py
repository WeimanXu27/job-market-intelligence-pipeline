"""Run the semantic-gate candidate on ten already-exposed precision blockers.

READ-ONLY with respect to MySQL.

This stage validates the v29 architecture on already-exposed postings before any
broader regression. It captures final skills and the all-proposal semantic-gate
trace so we can verify that:
  1) known hiring/application contamination is removed;
  2) no structurally grounded proposal escapes via parser-section bypass;
  3) real qualifications / prior experience / job duties remain KEEP-capable;
  4) UNKNOWN / incoherent adjudications fail closed without retrying extraction.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from sqlalchemy import URL, bindparam, create_engine, text

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data" / "processed"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from settings import DATABASE_CONFIG, DATA_CONFIG  # noqa: E402

CANDIDATE_PATH = SRC_DIR / "reliability_semantic_gate_candidate.py"
EXPECTED_SHA = "f8682ba1f69e1bf838df24ba05f6620219d37c5e845fc1f19d4861e336f5f7c4"
BLIND_MANIFEST = DATA_DIR / "skill_patch29_v19_fresh_blind_holdout_manifest.csv"
EXPECTED_BLIND_SHA = "fcc15697a80ba9b222ae25f781d2d04de4c60ce9945e98e888f672fc738ef7b5"
MANIFEST = DATA_DIR / "skill_patch29_v29_independent_semantic_gate_blocker_manifest.csv"
OUT = DATA_DIR / "skill_patch29_v29_independent_semantic_gate_blocker_outputs.csv"
TRACE = DATA_DIR / "skill_patch29_v29_independent_semantic_gate_blocker_gate_trace.csv"
SUMMARY = DATA_DIR / "skill_patch29_v29_independent_semantic_gate_blocker_summary.csv"
INTEGRITY = DATA_DIR / "skill_patch29_v29_independent_semantic_gate_stage2_integrity.json"

TABLE_NAME = DATA_CONFIG["table_name"]
ID_COLUMN = DATA_CONFIG["id_column"]
DESCRIPTION_COLUMN = DATA_CONFIG["description_column"]
SKILLS_COLUMN = DATA_CONFIG["skills_column"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_skill_text(value: object) -> str:
    text_value = re.sub(r"\s+", " ", str(value or "")).strip()
    # Strip repeated upstream numbering: "23. 23. skill" -> "skill".
    for _ in range(6):
        text_value = re.sub(r"^\s*\d+\.\s*", "", text_value).strip()
    return text_value.lower()


def count_numbered(value: object) -> int:
    if value is None or not str(value).strip():
        return 0
    return len([x for x in str(value).splitlines() if x.strip()])


def record_from_chunk_id(chunk_id: str) -> str:
    value = str(chunk_id)
    if "__chunk_" in value:
        return value.split("__chunk_", 1)[0]
    match = re.match(r"^(job_\d+)", value, flags=re.IGNORECASE)
    return match.group(1) if match else value


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
    return create_engine(url, pool_pre_ping=True, pool_recycle=3600, echo=False)


def load_candidate():
    actual = sha256_file(CANDIDATE_PATH)
    if actual.lower() != EXPECTED_SHA:
        raise RuntimeError(
            f"v29 candidate SHA mismatch expected={EXPECTED_SHA} actual={actual}"
        )
    spec = importlib.util.spec_from_file_location(
        "reliability_semantic_gate_candidate",
        CANDIDATE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load v29 independent-semantic-gate candidate")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, actual


def fetch_rows(engine, ids):
    query = text(
        f"SELECT {ID_COLUMN} AS record_id, {DESCRIPTION_COLUMN} AS description, "
        f"{SKILLS_COLUMN} AS patch28_skills FROM {TABLE_NAME} "
        f"WHERE {ID_COLUMN} IN :record_ids"
    ).bindparams(bindparam("record_ids", expanding=True))
    with engine.connect() as conn:
        rows = conn.execute(query, {"record_ids": ids}).mappings().all()
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No MySQL rows returned")
    df["record_id"] = df["record_id"].astype(str).str.strip()
    return df.drop_duplicates("record_id", keep="first")


def main():
    manifest = pd.read_csv(MANIFEST, dtype={"record_id": str}, keep_default_na=False)
    manifest["record_id"] = manifest["record_id"].astype(str).str.strip()
    if len(manifest) != 10 or manifest.record_id.duplicated().any():
        raise RuntimeError(
            f"Expected ten unique exposed blocker records; found {len(manifest)}"
        )

    actual_blind_sha = sha256_file(BLIND_MANIFEST)
    if actual_blind_sha.lower() != EXPECTED_BLIND_SHA:
        raise RuntimeError(
            f"fresh blind SHA mismatch expected={EXPECTED_BLIND_SHA} actual={actual_blind_sha}"
        )
    blind = pd.read_csv(BLIND_MANIFEST, dtype={"record_id": str}, keep_default_na=False)
    blind_ids = set(blind.record_id.astype(str).str.strip())
    overlap = set(manifest.record_id) & blind_ids
    if overlap:
        raise RuntimeError(f"FATAL blocker/blind overlap: {sorted(overlap)}")

    candidate, actual_sha = load_candidate()
    engine = build_engine()
    try:
        db = fetch_rows(engine, manifest.record_id.tolist())
    finally:
        engine.dispose()
    if len(db) != 10:
        raise RuntimeError(
            f"Missing DB rows: {sorted(set(manifest.record_id) - set(db.record_id))}"
        )

    work = manifest.merge(db, on="record_id", how="left", validate="one_to_one")
    extractor = candidate.SkillsExtractor()

    capture: Dict[str, List[str]] = {}
    run_df = work[["record_id", "description"]].rename(columns={"record_id": "id"}).copy()

    def fake_get_all_records(self):
        return run_df[["id", "description"]].copy(), ID_COLUMN

    def capture_update(self, record_id, skills, id_field="id"):
        capture[str(record_id)] = list(skills)
        return True

    extractor.get_all_records = types.MethodType(fake_get_all_records, extractor)
    extractor.update_record_skills = types.MethodType(capture_update, extractor)

    # Capture every raw grounded proposal at validator time, including whether it
    # was structurally eligible, whether it was routed to the all-proposal gate,
    # the independent KEEP/DROP + semantic class, and final validator acceptance.
    original_validate = extractor._validate_grounded_chunk_map
    trace_rows: List[Dict[str, Any]] = []
    validate_call_index = 0

    def traced_validate(self, raw_map, records, adjudication_map=None):
        nonlocal validate_call_index
        validate_call_index += 1
        validated, retry_ids = original_validate(
            raw_map,
            records,
            adjudication_map=adjudication_map,
        )

        if isinstance(raw_map, dict):
            for record in records:
                chunk_id = str(record.get("id", ""))
                raw_items = raw_map.get(chunk_id, [])
                if not isinstance(raw_items, list):
                    continue

                record_text = str(record.get("text", "") or "")
                segment_lookup = self._segment_lookup(record_text)
                valid_segment_ids = {
                    evidence_id.upper()
                    for evidence_id in re.findall(
                        r"\[([A-Za-z0-9_]+__S\d{4})\]",
                        record_text,
                    )
                }
                expected_prefix = f"{chunk_id}__".upper()
                accepted_norm = {
                    normalize_skill_text(x)
                    for x in validated.get(chunk_id, [])
                }

                for item_index, item in enumerate(raw_items, start=1):
                    base = {
                        "validate_call": validate_call_index,
                        "record_id": record_from_chunk_id(chunk_id),
                        "chunk_id": chunk_id,
                        "item_index": item_index,
                    }
                    if not isinstance(item, dict):
                        trace_rows.append({
                            **base,
                            "candidate_id": "",
                            "skill": "",
                            "skill_type": "",
                            "evidence_ids": "",
                            "evidence_segments": "",
                            "structurally_grounded": False,
                            "routed_to_semantic_gate": False,
                            "adjudicator_decision": "<non_dict_item>",
                            "semantic_class": "<non_dict_item>",
                            "semantic_keep_decision": "",
                            "accepted_by_grounded_validator": False,
                        })
                        continue

                    skill = str(item.get("skill", "") or "").strip()
                    skill_type = self._normalize_grounded_skill_type_label(
                        str(item.get("skill_type", "") or "")
                    )
                    raw_evidence_ids = item.get("evidence_ids", [])
                    evidence_is_list = isinstance(raw_evidence_ids, list)
                    if evidence_is_list:
                        cleaned_evidence_ids = [
                            str(x or "").strip().strip("[]").upper()
                            for x in raw_evidence_ids
                            if str(x or "").strip()
                        ]
                    else:
                        cleaned_evidence_ids = []

                    structurally_grounded = bool(
                        skill
                        and evidence_is_list
                        and cleaned_evidence_ids
                        and all(
                            evidence_id.startswith(expected_prefix)
                            and evidence_id in valid_segment_ids
                            for evidence_id in cleaned_evidence_ids
                        )
                    )
                    cited_segments = [
                        segment_lookup.get(evidence_id, "")
                        for evidence_id in cleaned_evidence_ids
                    ] if structurally_grounded else []
                    routed = bool(
                        structurally_grounded
                        and self._should_independently_adjudicate(
                            skill_type,
                            cited_segments,
                        )
                    )
                    candidate_id = self._adjudication_candidate_id(chunk_id, item_index)
                    decision_obj = (adjudication_map or {}).get(candidate_id, {}) if routed else {}
                    decision_label = str(
                        decision_obj.get("decision", "") or ""
                    ).strip().upper()
                    semantic_class = str(
                        decision_obj.get("semantic_class", "") or ""
                    ).strip().lower()
                    keep_decision = (
                        self._adjudication_keep_decision(
                            decision_label,
                            semantic_class,
                        )
                        if routed
                        else None
                    )

                    trace_rows.append({
                        **base,
                        "candidate_id": candidate_id if routed else "",
                        "skill": skill,
                        "skill_type": skill_type,
                        "evidence_ids": ";".join(cleaned_evidence_ids),
                        "evidence_segments": " || ".join(cited_segments),
                        "structurally_grounded": structurally_grounded,
                        "routed_to_semantic_gate": routed,
                        "adjudicator_decision": decision_label if routed else "<not_routed>",
                        "semantic_class": semantic_class if routed else "<not_routed>",
                        "semantic_keep_decision": (
                            "KEEP" if keep_decision is True else
                            "DROP" if keep_decision is False else
                            "UNRESOLVED" if routed else
                            "NOT_ROUTED"
                        ),
                        "accepted_by_grounded_validator": normalize_skill_text(skill) in accepted_norm,
                    })

        return validated, retry_ids

    extractor._validate_grounded_chunk_map = types.MethodType(traced_validate, extractor)

    print("Patch-29 v29 independent semantic gate: 10-record exposed blocker recheck")
    print(f"Candidate SHA: {actual_sha}")
    print("Fresh blind overlap: 0")
    stats = extractor.process_all_records()

    expected = set(work.record_id)
    captured = set(capture)
    if captured != expected:
        raise RuntimeError(
            f"Capture mismatch missing={sorted(expected-captured)} unexpected={sorted(captured-expected)}"
        )

    output_rows = []
    for _, row in work.iterrows():
        rid = row.record_id
        skills = capture[rid]
        normalized_final = [normalize_skill_text(s) for s in skills]
        noise = [
            x.strip()
            for x in str(row.suspected_noise_terms).split(";")
            if x.strip()
        ]
        exact_hits = [
            term
            for term in noise
            if normalize_skill_text(term) in normalized_final
        ]
        substring_hits = [
            term
            for term in noise
            if normalize_skill_text(term)
            and any(
                normalize_skill_text(term) in skill
                or skill in normalize_skill_text(term)
                for skill in normalized_final
                if skill
            )
        ]
        output_rows.append({
            "record_id": rid,
            "case_group": row.case_group,
            "patch28_skill_count": count_numbered(row.patch28_skills),
            "patch29_v29_skill_count": len(skills),
            "count_delta_vs_patch28": len(skills) - count_numbered(row.patch28_skills),
            "patch28_skills": row.patch28_skills,
            "patch29_v29_skills": "\n".join(f"{i}. {s}" for i, s in enumerate(skills, 1)),
            "expected_terms": row.expected_terms,
            "suspected_noise_terms": row.suspected_noise_terms,
            "exact_noise_hits": ";".join(dict.fromkeys(exact_hits)),
            "substring_noise_hits": ";".join(dict.fromkeys(substring_hits)),
            "diagnostic_note": row.diagnostic_note,
        })

    output = pd.DataFrame(output_rows)
    output.to_csv(OUT, index=False, encoding="utf-8-sig")

    trace = pd.DataFrame(trace_rows)
    if trace.empty:
        trace = pd.DataFrame(columns=[
            "validate_call", "record_id", "chunk_id", "candidate_id", "item_index",
            "skill", "skill_type", "evidence_ids", "evidence_segments",
            "structurally_grounded", "routed_to_semantic_gate",
            "adjudicator_decision", "semantic_class", "semantic_keep_decision",
            "accepted_by_grounded_validator",
        ])
    trace.to_csv(TRACE, index=False, encoding="utf-8-sig")

    grounded = trace[trace["structurally_grounded"].astype(bool)].copy() if not trace.empty else trace.copy()
    routed = grounded[grounded["routed_to_semantic_gate"].astype(bool)].copy() if not grounded.empty else grounded.copy()
    parser_bypass_count = int((~grounded["routed_to_semantic_gate"].astype(bool)).sum()) if not grounded.empty else 0
    semantic_counts = Counter(routed["semantic_class"].astype(str)) if not routed.empty else Counter()
    decision_counts = Counter(routed["semantic_keep_decision"].astype(str)) if not routed.empty else Counter()

    summary_rows = [
        {"metric": "candidate_sha256", "value": actual_sha},
        {"metric": "records", "value": len(output)},
        {"metric": "extractor_total", "value": stats.get("total_records", "") if isinstance(stats, dict) else ""},
        {"metric": "exact_noise_hit_records", "value": int((output.exact_noise_hits != "").sum())},
        {"metric": "substring_noise_hit_records", "value": int((output.substring_noise_hits != "").sum())},
        {"metric": "raw_grounded_trace_items", "value": len(trace)},
        {"metric": "structurally_grounded_items", "value": len(grounded)},
        {"metric": "routed_to_semantic_gate_items", "value": len(routed)},
        {"metric": "parser_section_bypass_items", "value": parser_bypass_count},
        {"metric": "semantic_keep_items", "value": int(decision_counts.get("KEEP", 0))},
        {"metric": "semantic_drop_items", "value": int(decision_counts.get("DROP", 0))},
        {"metric": "semantic_unresolved_items", "value": int(decision_counts.get("UNRESOLVED", 0))},
        {"metric": "fresh_blind_overlap", "value": 0},
    ]
    for label, count in sorted(semantic_counts.items()):
        summary_rows.append({"metric": f"semantic_class__{label}", "value": int(count)})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY, index=False, encoding="utf-8-sig")

    integrity = {
        "candidate": str(CANDIDATE_PATH.relative_to(ROOT_DIR)).replace("\\", "/"),
        "candidate_sha256": actual_sha,
        "stage": "10-record exposed blocker end-to-end independent-semantic-gate test",
        "records": len(output),
        "fresh_blind_touched": False,
        "fresh_blind_overlap": 0,
        "fresh_blind_expected_sha256": EXPECTED_BLIND_SHA,
        "parser_section_bypass_expected": 0,
        "outputs": str(OUT.relative_to(ROOT_DIR)).replace("\\", "/"),
        "trace": str(TRACE.relative_to(ROOT_DIR)).replace("\\", "/"),
        "summary": str(SUMMARY.relative_to(ROOT_DIR)).replace("\\", "/"),
    }
    INTEGRITY.write_text(json.dumps(integrity, indent=2), encoding="utf-8")

    print(output[[
        "record_id", "patch29_v29_skill_count", "exact_noise_hits", "substring_noise_hits"
    ]].to_string(index=False))
    print("\nIndependent semantic-gate decisions:")
    for label in ["KEEP", "DROP", "UNRESOLVED"]:
        print(f"  {label}: {decision_counts.get(label, 0)}")
    print(f"structurally grounded items: {len(grounded)}")
    print(f"routed items: {len(routed)}")
    print(f"parser-section bypass items: {parser_bypass_count}")
    print(f"outputs: {OUT}")
    print(f"semantic-gate trace: {TRACE}")
    print(f"summary: {SUMMARY}")
    print(f"integrity: {INTEGRITY}")


if __name__ == "__main__":
    main()
