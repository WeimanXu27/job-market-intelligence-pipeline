from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "src" / "reliability_semantic_gate_candidate.py"
OUT = ROOT / "data" / "processed" / "skill_patch29_v29_independent_semantic_gate_schema_replay.csv"
INTEGRITY = ROOT / "data" / "processed" / "skill_patch29_v29_independent_semantic_gate_integrity.json"
EXPECTED_SHA = "f8682ba1f69e1bf838df24ba05f6620219d37c5e845fc1f19d4861e336f5f7c4"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def install_offline_stubs() -> None:
    settings = types.ModuleType("settings")
    settings.OPENAI_API_KEY = None
    settings.DATABASE_CONFIG = {
        "username": "offline", "password": "offline", "host": "offline",
        "port": 3306, "database": "offline", "charset": "utf8mb4",
    }
    settings.DATA_CONFIG = {
        "table_name": "job_postings", "id_column": "record_id",
        "description_column": "description", "skills_column": "extracted_skills",
    }
    settings.MODEL_CONFIG = {"llm_model": "gpt-4o-mini", "ner_model": "offline"}
    settings.PROCESSING_CONFIG = {
        "llm_batch_size": 10, "confidence_threshold": 0.5,
        "max_prompt_chars": 12000, "debug_llm_empty": False,
        "debug_llm_empty_max": 0, "record_limit": 0,
    }
    settings.LOGGING_CONFIG = {
        "log_level": "CRITICAL",
        "log_file": str(ROOT / "v29_offline_replay.log"),
    }
    sys.modules["settings"] = settings

    try:
        import openai  # noqa: F401
    except Exception:
        m = types.ModuleType("openai")
        class _OpenAI:
            def __init__(self, *args, **kwargs):
                pass
        m.OpenAI = _OpenAI
        sys.modules["openai"] = m

    try:
        import transformers  # noqa: F401
    except Exception:
        m = types.ModuleType("transformers")
        m.AutoTokenizer = object
        m.AutoModelForTokenClassification = object
        m.pipeline = lambda *args, **kwargs: None
        sys.modules["transformers"] = m


def load_candidate():
    install_offline_stubs()
    spec = importlib.util.spec_from_file_location("v29_candidate", CANDIDATE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load candidate module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    actual_sha = sha256(CANDIDATE)
    if actual_sha != EXPECTED_SHA:
        raise RuntimeError(f"candidate SHA mismatch: {actual_sha} != {EXPECTED_SHA}")

    mod = load_candidate()
    cls = mod.SkillsExtractor
    ex = cls.__new__(cls)
    ex.max_prompt_chars = 12000
    ex.llm_client = None
    ex.openai_model = "gpt-4o-mini"

    rows: list[dict[str, object]] = []

    def check(group: str, name: str, actual, expected) -> None:
        passed = actual == expected
        rows.append({
            "group": group,
            "test": name,
            "actual": repr(actual),
            "expected": repr(expected),
            "pass": passed,
        })
        if not passed:
            raise AssertionError(f"{group}/{name}: {actual!r} != {expected!r}")

    # 1) Decision/semantic-class contract. No actor-pair ambiguity remains.
    decision_cases = [
        ("KEEP", "role_capability_or_experience", True),
        ("KEEP", "required_credential", True),
        ("DROP", "application_or_proof_process", False),
        ("DROP", "employment_condition", False),
        ("DROP", "employer_context", False),
        ("KEEP", "employment_condition", None),
        ("DROP", "required_credential", None),
        ("KEEP", "unknown", None),
        ("UNKNOWN", "role_capability_or_experience", None),
        ("", "role_capability_or_experience", None),
    ]
    for i, (decision, semantic, expected) in enumerate(decision_cases, 1):
        check(
            "decision",
            f"case_{i:02d}_{decision}_{semantic}",
            cls._adjudication_keep_decision(decision, semantic),
            expected,
        )

    # 2) Routing: every grounded proposal is adjudicated, regardless of parser label.
    route_cases = [
        ("skill", ["[RESPONSIBILITY] Conduct patient assessments"], True),
        ("workflow", ["[RESPONSIBILITY] Complete all requested application information"], True),
        ("credential", ["[RESPONSIBILITY] Maintain RN license"], True),
        ("skill", ["[QUALIFICATION] Experience with Python required"], True),
        ("workflow", ["[GENERAL] Candidate will coordinate care"], True),
        ("skill", ["[RESPONSIBILITY] Analyze data", "[QUALIFICATION] SQL experience"], True),
        ("skill", [], False),
    ]
    for i, (skill_type, segments, expected) in enumerate(route_cases, 1):
        check(
            "routing",
            f"route_{i:02d}",
            cls._should_independently_adjudicate(skill_type, segments),
            expected,
        )

    # 3) Response schema: only coherent decision/class pairs survive.
    candidates = [
        {"candidate_id": "A", "skill": "screening duty", "evidence_segments": ["x"]},
        {"candidate_id": "B", "skill": "RN license", "evidence_segments": ["x"]},
        {"candidate_id": "C", "skill": "application upload", "evidence_segments": ["x"]},
        {"candidate_id": "D", "skill": "employment condition", "evidence_segments": ["x"]},
        {"candidate_id": "E", "skill": "employer benefit", "evidence_segments": ["x"]},
        {"candidate_id": "F", "skill": "inconsistent", "evidence_segments": ["x"]},
        {"candidate_id": "G", "skill": "unknown", "evidence_segments": ["x"]},
        {"candidate_id": "H", "skill": "missing", "evidence_segments": ["x"]},
    ]
    raw_response = {
        "A": {"decision": "KEEP", "semantic_class": "role_capability_or_experience"},
        "B": {"decision": "KEEP", "semantic_class": "required_credential"},
        "C": {"decision": "DROP", "semantic_class": "application_or_proof_process"},
        "D": {"decision": "DROP", "semantic_class": "employment_condition"},
        "E": {"decision": "DROP", "semantic_class": "employer_context"},
        "F": {"decision": "KEEP", "semantic_class": "employment_condition"},
        "G": {"decision": "DROP", "semantic_class": "unknown"},
    }
    decisions, unresolved = cls._validate_independent_adjudicator_response(raw_response, candidates)
    check("response_schema", "valid_decision_count", len(decisions), 5)
    check("response_schema", "unresolved_ids", [x["candidate_id"] for x in unresolved], ["F", "G", "H"])

    # 4) Candidate preparation no longer trusts RESPONSIBILITY as a bypass.
    record = {
        "id": "job_test__chunk_001",
        "text": "\n".join([
            "[job_test__chunk_001__S0001] [RESPONSIBILITY] Conduct background investigations for new hires",
            "[job_test__chunk_001__S0002] [QUALIFICATION] Must hold an active RN license",
            "[job_test__chunk_001__S0003] [GENERAL] Upload proof of degree with the application",
            "[job_test__chunk_001__S0004] [RESPONSIBILITY] Complete all requested information fully",
        ]),
    }
    raw_map = {
        "job_test__chunk_001": [
            {"skill": "background investigations", "evidence_ids": ["job_test__chunk_001__S0001"], "skill_type": "workflow"},
            {"skill": "RN license", "evidence_ids": ["job_test__chunk_001__S0002"], "skill_type": "credential"},
            {"skill": "proof of degree upload", "evidence_ids": ["job_test__chunk_001__S0003"], "skill_type": "workflow"},
            {"skill": "complete application information", "evidence_ids": ["job_test__chunk_001__S0004"], "skill_type": "workflow"},
            {"skill": "invalid", "evidence_ids": ["other__chunk_001__S0001"], "skill_type": "workflow"},
        ]
    }
    prepared = ex._prepare_independent_adjudication_candidates(raw_map, [record])
    check("preparation", "routed_count", len(prepared), 4)
    check(
        "preparation",
        "routed_skills",
        [x["skill"] for x in prepared],
        ["background investigations", "RN license", "proof of degree upload", "complete application information"],
    )

    # 5) Prompt contract: independent decision + semantic class, no parser-label trust.
    extraction_messages, _, _ = ex._build_grounded_chunk_prompt([{**record, "ner": []}])
    extraction_prompt = "\n".join(m["content"] for m in extraction_messages)
    check("prompt_contract", "extractor_omits_semantic_class", "semantic_class" in extraction_prompt, False)

    adj_messages, _ = ex._build_independent_adjudicator_prompt(prepared)
    adj_prompt = "\n".join(m["content"] for m in adj_messages)
    check("prompt_contract", "adjudicator_independent", "You did NOT generate these proposals" in adj_prompt, True)
    check("prompt_contract", "decision_present", "decision: exactly KEEP or DROP" in adj_prompt, True)
    check("prompt_contract", "semantic_class_present", "semantic_class" in adj_prompt, True)
    check("prompt_contract", "qualification_experience_keepable", "candidate qualification such as experience doing that work" in adj_prompt, True)
    check("prompt_contract", "credential_definition_narrow", "ONLY for the credential itself" in adj_prompt, True)
    check("prompt_contract", "employment_condition_defined", "legal employment eligibility or work-authorization status" in adj_prompt, True)
    check("prompt_contract", "parser_labels_not_truth", "Do not trust section labels" in adj_prompt, True)

    # 6) Deterministic validator behavior under the new semantic gate.
    def validate_one(
        name: str,
        section: str,
        evidence: str,
        skill: str,
        skill_type: str,
        decision: dict[str, str] | None,
        expected_skills: list[str],
        expect_retry: bool = False,
    ) -> None:
        cid = f"{name}__chunk_001"
        sid = f"{cid}__S0001"
        rec = {"id": cid, "text": f"[{sid}] [{section}] {evidence}"}
        raw = {cid: [{"skill": skill, "evidence_ids": [sid], "skill_type": skill_type}]}
        amap = {}
        if decision is not None:
            amap[f"{cid}__I0001"] = decision
        validated, retry_ids = ex._validate_grounded_chunk_map(raw, [rec], adjudication_map=amap)
        check("validator", f"{name}_skills", validated[cid], expected_skills)
        check("validator", f"{name}_retry", cid in retry_ids, expect_retry)

    validate_one(
        "qualification_experience_keep",
        "QUALIFICATION",
        "Experience conducting street outreach and clinical service delivery is required",
        "street outreach",
        "workflow",
        {"decision": "KEEP", "semantic_class": "role_capability_or_experience"},
        ["street outreach"],
    )
    validate_one(
        "rn_credential_keep",
        "QUALIFICATION",
        "Must hold an active RN license",
        "RN license",
        "credential",
        {"decision": "KEEP", "semantic_class": "required_credential"},
        ["RN license"],
    )
    validate_one(
        "proof_drop",
        "QUALIFICATION",
        "Upload proof of your RN license with the application",
        "RN license document upload",
        "workflow",
        {"decision": "DROP", "semantic_class": "application_or_proof_process"},
        [],
    )
    validate_one(
        "drug_drop",
        "QUALIFICATION",
        "Selected hire must satisfy a screening test before starting work",
        "pre-employment screening test",
        "workflow",
        {"decision": "DROP", "semantic_class": "employment_condition"},
        [],
    )
    validate_one(
        "workauth_drop",
        "QUALIFICATION",
        "Successful candidate must have legal authorization to work",
        "work authorization",
        "credential",
        {"decision": "DROP", "semantic_class": "employment_condition"},
        [],
    )
    validate_one(
        "generic_policy_drop",
        "QUALIFICATION",
        "As an employee, you must comply with all institutional policies",
        "institutional policy compliance",
        "skill",
        {"decision": "DROP", "semantic_class": "employment_condition"},
        [],
    )
    validate_one(
        "hr_screening_keep",
        "RESPONSIBILITY",
        "Conduct screening and verification for new hires",
        "screening and verification",
        "workflow",
        {"decision": "KEEP", "semantic_class": "role_capability_or_experience"},
        ["screening and verification"],
    )
    validate_one(
        "i9_keep",
        "RESPONSIBILITY",
        "Verify employment eligibility and complete I-9 documentation for new hires",
        "employment eligibility verification",
        "workflow",
        {"decision": "KEEP", "semantic_class": "role_capability_or_experience"},
        ["employment eligibility verification"],
    )
    validate_one(
        "mislabelled_responsibility_drop",
        "RESPONSIBILITY",
        "Complete all requested information fully on the employment application",
        "complete application information",
        "workflow",
        {"decision": "DROP", "semantic_class": "application_or_proof_process"},
        [],
    )
    validate_one(
        "mislabelled_responsibility_employment_condition_drop",
        "RESPONSIBILITY",
        "Selected hire must clear a generic onboarding condition before work begins",
        "onboarding condition",
        "workflow",
        {"decision": "DROP", "semantic_class": "employment_condition"},
        [],
    )
    validate_one(
        "employer_context_drop",
        "GENERAL",
        "The employer offers extensive professional development benefits",
        "professional development benefits",
        "skill",
        {"decision": "DROP", "semantic_class": "employer_context"},
        [],
    )
    validate_one(
        "missing_decision_fail_closed",
        "RESPONSIBILITY",
        "Analyze operational data",
        "data analysis",
        "skill",
        None,
        [],
        False,
    )
    validate_one(
        "inconsistent_decision_fail_closed",
        "QUALIFICATION",
        "Must satisfy an employment condition",
        "employment condition",
        "workflow",
        {"decision": "KEEP", "semantic_class": "employment_condition"},
        [],
        False,
    )

    # 7) Explicit anti-overfitting semantic taxonomy checks.
    taxonomy_cases = [
        ("experience_requirement", "KEEP", "role_capability_or_experience", True),
        ("formal_license", "KEEP", "required_credential", True),
        ("application_material", "DROP", "application_or_proof_process", False),
        ("credential_verification_process", "DROP", "application_or_proof_process", False),
        ("prehire_screening", "DROP", "employment_condition", False),
        ("legal_eligibility_status", "DROP", "employment_condition", False),
        ("generic_employee_policy", "DROP", "employment_condition", False),
        ("company_benefit", "DROP", "employer_context", False),
        ("hr_verification_duty", "KEEP", "role_capability_or_experience", True),
        ("compliance_investigation_duty", "KEEP", "role_capability_or_experience", True),
    ]
    for name, decision, semantic, expected in taxonomy_cases:
        check(
            "semantic_taxonomy",
            name,
            cls._adjudication_keep_decision(decision, semantic),
            expected,
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "test", "actual", "expected", "pass"])
        writer.writeheader()
        writer.writerows(rows)

    passed = sum(bool(r["pass"]) for r in rows)
    integrity = {
        "candidate": str(CANDIDATE.relative_to(ROOT)).replace("\\", "/"),
        "candidate_sha256": actual_sha,
        "stage": "independent semantic-gate schema/unit replay only",
        "local_replay": f"{passed}/{len(rows)} PASS",
        "live_records_touched": 0,
        "fresh_blind_touched": False,
        "architecture": "grounded extractor -> all-proposal batched independent semantic gate -> deterministic validator",
        "semantic_classes": [
            "role_capability_or_experience",
            "required_credential",
            "application_or_proof_process",
            "employment_condition",
            "employer_context",
        ],
        "parser_section_bypass": False,
        "unresolved_policy": "adjudicator retries once, then fail-closed item drop without whole-extractor retry",
    }
    INTEGRITY.write_text(json.dumps(integrity, indent=2), encoding="utf-8")

    print(f"candidate_sha: {actual_sha}")
    print(f"schema/unit replay: {passed}/{len(rows)} passed")
    print(f"wrote: {OUT}")
    print(f"wrote: {INTEGRITY}")


if __name__ == "__main__":
    main()
