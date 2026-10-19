"""Deterministic replay of rule-hardening precision tests.

Runs directly against the v26 source via AST. No database, model load, or API call
is required. It protects the new application-instruction/background-verification
family while preserving legitimate application-processing and credential duties.
"""
from __future__ import annotations

import ast, csv, hashlib, logging, re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "src" / "reliability_rule_hardened_candidate.py"
OUT = ROOT / "data" / "processed" / "skill_patch29_v26_policy_tail_precision_replay.csv"
EXPECTED_SHA = "baae047bc9be5304ccf6f2d91cb8d908f1472e20734c8d6becee088bca24372c"

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def load_class():
    actual=sha256_file(CANDIDATE)
    if actual.lower()!=EXPECTED_SHA: raise RuntimeError(f"v26 candidate SHA mismatch expected={EXPECTED_SHA} actual={actual}")
    tree=ast.parse(CANDIDATE.read_text(encoding='utf-8'), filename=str(CANDIDATE))
    class_node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SkillsExtractor')
    module=ast.Module(body=[class_node], type_ignores=[]); ast.fix_missing_locations(module)
    ns={'re':re,'logging':logging,'logger':logging.getLogger('v26_replay'),'pd':pd,'Any':Any,'Dict':Dict,'List':List,'Optional':Optional,'Tuple':Tuple,
        'MAX_CANDIDATE_TEXT_CHARS':8000,'MAX_EVIDENCE_CHUNK_CHARS':3200,'MAX_EVIDENCE_SEGMENT_CHARS':1600}
    exec(compile(module,str(CANDIDATE),'exec'),ns)
    cls=ns['SkillsExtractor']; return cls.__new__(cls), actual

SEGMENT_CASES = [
 ('marshal_application_heading','[GENERAL] APPLICATION AND SELECTION INFORMATION Carefully complete your application and supplemental questionnaire and submit by the application deadline.',True),
 ('marshal_list_titles','[GENERAL] If you held multiple positions with the same employer, please list each job title and duties separately.',True),
 ('marshal_employment_gaps','[GENERAL] Explain gaps between any employment periods by way of a separate attachment to the application.',True),
 ('marshal_military_volunteer','[GENERAL] Include U.S. Military Service, self-employment, and relevant unpaid volunteer work.',True),
 ('marshal_check_email','[GENERAL] You are responsible for checking your email account on a regular basis to find out the status of your application.',True),
 ('ucd_application_requirements','[GENERAL] Application Requirements Document requirements Curriculum Vitae - Your most recently updated C.V.',True),
 ('ucd_release_form','[GENERAL] Authorization to Release Information Form - This form is required for all applicants applying to academic recruitments.',True),
 ('ucd_download_upload_form','[GENERAL] Download, complete, sign, and upload the form as part of the application.',True),
 ('sjsu_application_procedure','[GENERAL] Application Procedure Click Apply Now to complete the SJSU Online Employment Application and attach the following documents.',True),
 ('sjsu_standard_background','[GENERAL] The standard background check includes: criminal check, employment and education verification.',True),
 ('sjsu_motor_vehicle_check','[GENERAL] Depending on the position, a motor vehicle and/or credit check may be required.',True),
 ('sjsu_degree_evidence','[GENERAL] Evidence of required degree(s) or certification(s) will be required at time of hire.',True),
 ('permit_real_duty','[RESPONSIBILITY] Prepare, submit, and track permit applications for infrastructure installation projects.',False),
 ('benefits_real_application_processing','[RESPONSIBILITY] Process applications and other paperwork including review for accuracy and follow-up on missing information.',False),
 ('admissions_real_application_evaluation','[RESPONSIBILITY] Evaluate and verify applications for admissibility.',False),
 ('hr_real_work_authorization','[RESPONSIBILITY] Verify work authorization for new hires and complete I-9 documentation.',False),
 ('legit_rn_license','[QUALIFICATION] Must possess an active RN license.',False),
 ('legit_degree','[QUALIFICATION] Bachelor\'s degree required.',False),
 ('legit_cpr','[QUALIFICATION] Current CPR certification required.',False),
 # prior blocker family protection
 ('prior_preemployment','[QUALIFICATION] The successful candidate will be required to complete DOJ clearance and pre-employment online training.',True),
 ('prior_drug_test','[GENERAL] Any person given a conditional offer of employment will be required to pass a drug test.',True),
 ('prior_work_auth','[GENERAL] At the time of appointment, the successful candidate must have authorization to work in the United States.',True),
 ('faculty_finalist_misconduct_disclosure','[GENERAL] Finalists for the position are required to disclose violations of professional conduct policies and complete the related disclosure form.',True),
 ('faculty_academic_recruitment_misconduct','[GENERAL] As part of the academic recruitment process, candidates must complete misconduct disclosure and authorization forms.',True),
 ('keep_misconduct_investigation_duty','[RESPONSIBILITY] Investigate allegations of employee misconduct and interpret applicable university conduct policies.',False),
]

VALIDATOR_CASES = [
 ('drop_complete_application_forms','[GENERAL] Carefully complete your application and supplemental questionnaire.','complete application forms','workflow',False),
 ('drop_explain_employment_gaps','[GENERAL] Explain gaps between any employment periods by way of a separate attachment to the application.','explain employment gaps','workflow',False),
 ('drop_military_volunteer_instruction','[GENERAL] Include U.S. Military Service and relevant unpaid volunteer work.','include military service and volunteer work','workflow',False),
 ('drop_application_documentation','[GENERAL] Application Requirements Document requirements Curriculum Vitae and Cover Letter.','application documentation','workflow',False),
 ('drop_completing_forms','[GENERAL] Download, complete, sign, and upload the Authorization to Release Information Form.','completing forms','workflow',False),
 ('drop_motor_vehicle_check','[GENERAL] Depending on the position, a motor vehicle and/or credit check may be required.','motor vehicle check','credential',False),
 ('drop_degree_verification','[GENERAL] The standard background check includes criminal check, employment and education verification.','degree verification','credential',False),
 ('drop_certification_verification','[GENERAL] Evidence of required degree(s) or certification(s) will be required at time of hire.','certification verification','credential',False),
 ('drop_provide_relevant_education_experience','[GENERAL] Please provide relevant education and experience in your application.','provide relevant education and experience','workflow',False),
 ('drop_meet_minimum_requirements','[GENERAL] Applicants must meet minimum requirements before moving forward in the selection process.','meet minimum requirements','workflow',False),
 ('keep_hr_minimum_qualification_screening','[RESPONSIBILITY] Screen applicants against minimum qualifications and document eligibility decisions.','minimum qualifications screening','workflow',True),
 ('keep_patient_education','[RESPONSIBILITY] Provide patient education and counseling about treatment plans.','patient education','workflow',True),
 ('keep_permit_application_submission','[RESPONSIBILITY] Prepare and submit permit applications to municipalities.','permit application submission','workflow',True),
 ('keep_benefit_application_processing','[RESPONSIBILITY] Process applications and other paperwork for employee benefit plans.','application processing','workflow',True),
 ('keep_admissions_application_evaluation','[RESPONSIBILITY] Evaluate applications for admissibility.','application evaluation','workflow',True),
 ('keep_hr_background_checks','[RESPONSIBILITY] Administer background checks for new hires.','background checks','workflow',True),
 ('keep_rn_license','[QUALIFICATION] Must possess an active RN license.','RN license','credential',True),
 ('keep_degree','[QUALIFICATION] Bachelor\'s degree required.','Bachelor\'s degree','credential',True),
 ('keep_cpr','[QUALIFICATION] Current CPR certification required.','current CPR certification','credential',True),
 # prior family protection
 ('drop_preemployment_training','[QUALIFICATION] The successful candidate will be required to complete DOJ clearance and pre-employment online training.','pre-employment online training','workflow',False),
 ('drop_drug_test','[GENERAL] Any person given a conditional offer of employment will be required to pass a drug test.','drug test compliance','workflow',False),
 ('drop_work_authorization','[GENERAL] At the time of appointment, the successful candidate must have authorization to work in the United States.','authorization to work in the United States','credential',False),
 ('keep_hr_i9','[RESPONSIBILITY] Verify employment eligibility and complete I-9 documentation for new hires.','employment eligibility verification','workflow',True),
 ('drop_understanding_misconduct_policy','[GENERAL] Finalists must review the university policy on professional misconduct and complete the required disclosure form.','understanding policies on misconduct','workflow',False),
 ('drop_generic_university_policy_adherence','[GENERAL] Applicants must comply with university policies as a condition of appointment.','complying with university policies','workflow',False),
 ('keep_misconduct_investigation','[RESPONSIBILITY] Investigate allegations of employee misconduct and prepare findings.','misconduct investigation','workflow',True),
 ('keep_title_ix_compliance','[RESPONSIBILITY] Advise departments on Title IX compliance and conduct policy investigations.','Title IX compliance','workflow',True),
 ('keep_policy_analysis','[RESPONSIBILITY] Analyze institutional policies and recommend revisions.','policy analysis','workflow',True),
]

def validate_single(extractor, cid, segment, skill, skill_type):
    chunk=f'{cid}__chunk_001'; evid=f'{chunk}__S0001'
    raw={chunk:[{'skill':skill,'evidence_ids':[evid],'skill_type':skill_type}]}
    validated,_=extractor._validate_grounded_chunk_map(raw,[{'id':chunk,'text':f'[{evid}] {segment}'}])
    return validated.get(chunk,[])

def main():
    ex,sha=load_class(); rows=[]; failures=[]
    for cid,text,expected in SEGMENT_CASES:
        actual=bool(ex._is_application_process_segment(text)); passed=actual==expected
        row={'test_layer':'segment_guard','case_id':cid,'expected':str(expected),'actual':str(actual),'pass':passed,'detail':text}; rows.append(row)
        if not passed: failures.append(row)
    for cid,seg,skill,typ,expected_kept in VALIDATOR_CASES:
        got=validate_single(ex,cid,seg,skill,typ); actual_kept=skill in got; passed=actual_kept==expected_kept
        row={'test_layer':'validator','case_id':cid,'expected':f'kept={expected_kept}','actual':f'kept={actual_kept}; skills={got}','pass':passed,'detail':f'{seg} || {skill}'}; rows.append(row)
        if not passed: failures.append(row)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with OUT.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f'Candidate SHA: {sha}')
    for r in rows: print(f"{'PASS' if r['pass'] else 'FAIL'}\t{r['test_layer']}\t{r['case_id']}")
    print(f'\nsummary: {len(rows)-len(failures)}/{len(rows)} passed')
    print(f'output: {OUT}')
    if failures: raise SystemExit(1)
if __name__=='__main__': main()
