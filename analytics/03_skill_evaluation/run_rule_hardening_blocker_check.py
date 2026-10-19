"""Run the rule-hardening candidate on ten already-exposed precision blockers.

READ-ONLY with respect to MySQL. The set includes the seven prior blockers plus
three v23 full-935 application/hiring-tail failures.
"""
from __future__ import annotations
import hashlib, importlib.util, sys, types, re
from pathlib import Path
import pandas as pd
from sqlalchemy import URL, bindparam, create_engine, text

ROOT_DIR=Path(__file__).resolve().parents[2]; SRC_DIR=ROOT_DIR/'src'; DATA_DIR=ROOT_DIR/'data'/'processed'
if str(SRC_DIR) not in sys.path: sys.path.insert(0,str(SRC_DIR))
from settings import DATABASE_CONFIG, DATA_CONFIG  # noqa: E402

CANDIDATE_PATH=SRC_DIR/'reliability_rule_hardened_candidate.py'
EXPECTED_SHA='baae047bc9be5304ccf6f2d91cb8d908f1472e20734c8d6becee088bca24372c'
BLIND_MANIFEST=DATA_DIR/'skill_patch29_v19_fresh_blind_holdout_manifest.csv'
EXPECTED_BLIND_SHA='fcc15697a80ba9b222ae25f781d2d04de4c60ce9945e98e888f672fc738ef7b5'
MANIFEST=DATA_DIR/'skill_patch29_v26_blocker_recheck_manifest.csv'
OUT=DATA_DIR/'skill_patch29_v26_blocker_recheck_outputs.csv'
SUMMARY=DATA_DIR/'skill_patch29_v26_blocker_recheck_summary.csv'
TABLE_NAME=DATA_CONFIG['table_name']; ID_COLUMN=DATA_CONFIG['id_column']; DESCRIPTION_COLUMN=DATA_CONFIG['description_column']; SKILLS_COLUMN=DATA_CONFIG['skills_column']

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def build_engine():
    db=DATABASE_CONFIG
    url=URL.create(drivername='mysql+mysqlconnector',username=db['username'],password=db['password'],host=db['host'],port=int(db['port']),database=db['database'],query={'charset':db.get('charset','utf8mb4')})
    return create_engine(url,pool_pre_ping=True,pool_recycle=3600,echo=False)

def load_candidate():
    actual=sha256_file(CANDIDATE_PATH)
    if actual.lower()!=EXPECTED_SHA: raise RuntimeError(f'v26 candidate SHA mismatch expected={EXPECTED_SHA} actual={actual}')
    spec=importlib.util.spec_from_file_location('reliability_rule_hardened_candidate',CANDIDATE_PATH)
    if spec is None or spec.loader is None: raise RuntimeError('Could not load v26 candidate')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module,actual

def fetch_rows(engine,ids):
    query=text(f'SELECT {ID_COLUMN} AS record_id, {DESCRIPTION_COLUMN} AS description, {SKILLS_COLUMN} AS patch28_skills FROM {TABLE_NAME} WHERE {ID_COLUMN} IN :record_ids').bindparams(bindparam('record_ids',expanding=True))
    with engine.connect() as conn: rows=conn.execute(query,{'record_ids':ids}).mappings().all()
    df=pd.DataFrame(rows)
    if df.empty: raise RuntimeError('No MySQL rows returned')
    df['record_id']=df['record_id'].astype(str).str.strip(); return df.drop_duplicates('record_id',keep='first')

def count_numbered(value):
    if value is None or not str(value).strip(): return 0
    return len([x for x in str(value).splitlines() if x.strip()])

def main():
    manifest=pd.read_csv(MANIFEST,dtype={'record_id':str},keep_default_na=False); manifest['record_id']=manifest.record_id.astype(str).str.strip()
    if len(manifest)!=10 or manifest.record_id.duplicated().any(): raise RuntimeError(f'Expected ten unique blocker records; found {len(manifest)}')
    actual_blind_sha=sha256_file(BLIND_MANIFEST)
    if actual_blind_sha.lower()!=EXPECTED_BLIND_SHA: raise RuntimeError(f'fresh blind SHA mismatch expected={EXPECTED_BLIND_SHA} actual={actual_blind_sha}')
    blind=pd.read_csv(BLIND_MANIFEST,dtype={'record_id':str},keep_default_na=False)
    overlap=set(manifest.record_id)&set(blind.record_id.astype(str).str.strip())
    if overlap: raise RuntimeError(f'FATAL blocker/blind overlap: {sorted(overlap)}')
    candidate,actual_sha=load_candidate(); engine=build_engine()
    try: db=fetch_rows(engine,manifest.record_id.tolist())
    finally: engine.dispose()
    if len(db)!=10: raise RuntimeError(f'Missing DB rows: {sorted(set(manifest.record_id)-set(db.record_id))}')
    work=manifest.merge(db,on='record_id',how='left',validate='one_to_one')
    extractor=candidate.SkillsExtractor(); capture={}; run_df=work[['record_id','description']].rename(columns={'record_id':'id'}).copy()
    def fake_get_all_records(self): return run_df[['id','description']].copy(),ID_COLUMN
    def capture_update(self,record_id,skills,id_field='id'): capture[str(record_id)]=list(skills); return True
    extractor.get_all_records=types.MethodType(fake_get_all_records,extractor); extractor.update_record_skills=types.MethodType(capture_update,extractor)
    print('Patch-29 v26 ten-record exposed blocker recheck'); print(f'Candidate SHA: {actual_sha}'); print('Fresh blind overlap: 0')
    stats=extractor.process_all_records(); expected=set(work.record_id); captured=set(capture)
    if captured!=expected: raise RuntimeError(f'Capture mismatch missing={sorted(expected-captured)} unexpected={sorted(captured-expected)}')
    rows=[]
    def normalize_skill_text(value):
        text_value=re.sub(r"\s+"," ",str(value or "")).strip()
        # Some extractor paths already return numbered strings. Strip repeated
        # leading enumeration before exact noise comparison so the QA runner
        # cannot report a false PASS merely because output formatting changed.
        for _ in range(3):
            text_value=re.sub(r"^\s*\d+\.\s*","",text_value).strip()
        return text_value.lower()
    for _,r in work.iterrows():
        rid=r.record_id; sk=capture[rid]; numbered='\n'.join(f'{i}. {s}' for i,s in enumerate(sk,1)); noise=[x.strip() for x in str(r.suspected_noise_terms).split(';') if x.strip()]; normalized=[normalize_skill_text(s) for s in sk]; hits=[t for t in noise if normalize_skill_text(t) in normalized]
        rows.append({'record_id':rid,'case_group':r.case_group,'patch28_skill_count':count_numbered(r.patch28_skills),'patch29_v26_skill_count':len(sk),'count_delta_vs_patch28':len(sk)-count_numbered(r.patch28_skills),'patch28_skills':r.patch28_skills,'patch29_v26_skills':numbered,'expected_terms':r.expected_terms,'suspected_noise_terms':r.suspected_noise_terms,'exact_noise_hits':';'.join(hits),'diagnostic_note':r.diagnostic_note})
    output=pd.DataFrame(rows); output.to_csv(OUT,index=False,encoding='utf-8-sig')
    summary=pd.DataFrame([{'metric':'candidate_sha256','value':actual_sha},{'metric':'records','value':len(output)},{'metric':'extractor_total','value':stats.get('total_records','') if isinstance(stats,dict) else ''},{'metric':'exact_noise_hit_records','value':int((output.exact_noise_hits!='').sum())},{'metric':'fresh_blind_overlap','value':0}]); summary.to_csv(SUMMARY,index=False,encoding='utf-8-sig')
    print(output[['record_id','patch29_v26_skill_count','exact_noise_hits']].to_string(index=False)); print(f'outputs: {OUT}'); print(f'summary: {SUMMARY}')
if __name__=='__main__': main()
