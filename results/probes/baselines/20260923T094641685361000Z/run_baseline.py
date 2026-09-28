#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'results'))
sys.path.insert(0,str(ROOT/'python'))
from compare_jev import DEFAULT_KEY, DEFAULT_URL, decode_questions, jev_questions, load_cases, native_response_dict, normalize_native, normalize_jev, post_jev, redact
FIXTURE=ROOT/'benchmarks/cases/probe-test-90.jsonl'
STORE=Path('/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/decengine-model-store')
LIB=ROOT/'target/release/libdecengine.dylib'
MODELS=['Qwen/Qwen3-Embedding-0.6B','microsoft/harrier-oss-v1-0.6b']
MANIFESTS={'Qwen/Qwen3-Embedding-0.6B':ROOT/'models/manifests/qwen3-embedding-0.6b.json','microsoft/harrier-oss-v1-0.6b':ROOT/'models/manifests/harrier-oss-v1-0.6b.json'}
OUT=Path(__file__).resolve().parent

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def store_dir(model): return STORE/'models'/model.replace('/','--')
def scalar(model, decisions, questions):
 owner=decisions.get('owner',{})
 urgent=decisions.get('urgent',{})
 impact=decisions.get('impact',{})
 value=urgent.get('value')
 if model=='jev':
  probs=impact.get('probabilities')
  labels={str(i):str(level['label']) for i,level in enumerate(questions['impact']['levels'])}
 else: probs=impact.get('distribution'); labels=None
 winners=[]
 if isinstance(probs,dict):
  vals=[(str(k),v) for k,v in probs.items() if isinstance(v,(int,float))]
  if vals:
   best=max(v for _,v in vals); winners=[k for k,v in vals if v==best]
 impact_label=(labels.get(winners[0],winners[0]) if labels else winners[0]) if len(winners)==1 else None
 return {'owner':owner.get('selected'), 'urgent':value>=0.5 if isinstance(value,(int,float)) else None,'impact':impact_label}

def main():
 cases=load_cases(FIXTURE)
 if len(cases)!=90: raise RuntimeError(f'expected 90 cases, got {len(cases)}')
 if not LIB.is_file(): raise RuntimeError('native library missing')
 key=DEFAULT_KEY.read_text(encoding='utf-8').strip()
 if not key: raise RuntimeError('JEV key empty')
 from decengine import Engine
 metadata={'run_id':OUT.name,'started_at':datetime.now(timezone.utc).isoformat(),'fixture':str(FIXTURE),'fixture_sha256':sha(FIXTURE),'case_count':90,'models':{},'jev':{'model':'jev-latest','url':DEFAULT_URL},'native_library':str(LIB),'native_library_sha256':sha(LIB),'model_store':str(STORE),'python':sys.version.split()[0],'platform':sys.platform,'calls_per_case':{m:1 for m in MODELS}|{'jev_api':1}}
 for model in MODELS:
  manifest_path=MANIFESTS[model]; install_path=store_dir(model)/'install.json'; weights=store_dir(model)/'model.safetensors'
  install=json.loads(install_path.read_text()); manifest=json.loads(manifest_path.read_text())
  if install.get('model_id')!=model: raise RuntimeError(f'installed model mismatch for {model}')
  metadata['models'][model]={'manifest_path':str(manifest_path),'manifest':manifest,'manifest_sha256':sha(manifest_path),'install_record':install,'install_record_sha256':sha(install_path),'checkpoint_path':str(weights),'checkpoint_sha256':sha(weights),'checkpoint_size_bytes':weights.stat().st_size}
 (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n')
  output=(OUT/'cases.jsonl').open('w',encoding='utf-8')
 failures={'models':{m:0 for m in MODELS},'jev':0}
 try:
  with __import__('contextlib').ExitStack() as stack:
   engines={m:stack.enter_context(Engine(m,native_library=LIB,options={'engine':'mlx','model_store':str(STORE)})) for m in MODELS}
   for index,case in enumerate(cases,1):
    req=case['request']; record={'case_id':str(case['id']),'input':case,'index':index,'run_id':OUT.name}
    for model in MODELS:
     start=time.perf_counter_ns()
     try:
      raw=native_response_dict(engines[model].decide(state=req['state'],questions=decode_questions(req['questions'])))
      record_key='qwen' if model.startswith('Qwen/') else 'harrier'
      record[record_key]={'model_id':model,'raw_response':raw,'normalized_decisions':normalize_native(raw),'timing_ms':(time.perf_counter_ns()-start)/1e6,'error':None}
     except Exception as e:
      failures['models'][model]+=1; record_key='qwen' if model.startswith('Qwen/') else 'harrier'
      record[record_key]={'model_id':model,'raw_response':None,'normalized_decisions':None,'timing_ms':(time.perf_counter_ns()-start)/1e6,'error':f'{type(e).__name__}: {str(e).replace(key,"[REDACTED]")}'}
    state=json.dumps(req['state'],ensure_ascii=False,sort_keys=True,separators=(',',':'))
    payload={'state':state,'model':'jev-latest','questions':jev_questions(req['questions'])}
    jstart=time.perf_counter_ns()
    try:
     raw,elapsed=post_jev(DEFAULT_URL,key,payload,120.0); body=raw['body']
     if not isinstance(body,dict): raise RuntimeError('JEV response must be object')
     body=body.get('result',body)
     if not isinstance(body,dict) or not isinstance(body.get('answers'),dict): raise RuntimeError('unexpected JEV response shape')
     record['jev']={'raw_response':raw,'normalized_decisions':normalize_jev(body,req['questions']),'timing_ms':elapsed,'error':None}
    except Exception as e:
     failures['jev']+=1; record['jev']={'raw_response':None,'normalized_decisions':None,'timing_ms':(time.perf_counter_ns()-jstart)/1e6,'error':f'{type(e).__name__}: {str(e).replace(key,"[REDACTED]")}'}
    record['jev_request_payload']=payload
     output.write(json.dumps(redact(record,key),ensure_ascii=False,allow_nan=False)+'\n'); output.flush()
    print(f'[{index}/90] {case["id"]}: Qwen={"err" if record["qwen"]["error"] else "ok"}, Harrier={"err" if record["harrier"]["error"] else "ok"}, JEV={"err" if record["jev"]["error"] else "ok"}',flush=True)
 finally:
   output.close()
 # Predictions are durably persisted above before any reference labels are read/scored.
 for model,name in [(MODELS[0],'qwen'),(MODELS[1],'harrier')]:
   rows=[json.loads(line) for line in (OUT/'cases.jsonl').read_text().splitlines()]
  with (OUT/f'{name}-predictions.jsonl').open('w') as out:
   for row in rows:
    out.write(json.dumps({'id':row['case_id'],**scalar(name,row[name]['normalized_decisions'] or {},row['input']['request']['questions'])},ensure_ascii=False)+'\n')
  rows=[json.loads(line) for line in (OUT/'cases.jsonl').read_text().splitlines()]
 with (OUT/'jev-predictions.jsonl').open('w') as out:
  for row in rows: out.write(json.dumps({'id':row['case_id'],**scalar('jev',row['jev']['normalized_decisions'] or {},row['input']['request']['questions'])},ensure_ascii=False)+'\n')
 # exact-match metrics only after raw and hard-label prediction files were persisted
 byid={str(c['id']):c for c in cases}; metrics={}
 for name in ('qwen','harrier','jev'):
  preds=[json.loads(x) for x in (OUT/f'{name}-predictions.jsonl').read_text().splitlines()]
  question={q:{'correct':0,'total':0} for q in ('owner','urgent','impact')}
  errors=0
  # use run records after predictions exist
   records=[json.loads(x) for x in (OUT/'cases.jsonl').read_text().splitlines()]
  for p,r in zip(preds,records):
   ref=byid[str(p['id'])]['reference']; errors += r[name]['error'] is not None
   for q in question:
    question[q]['total']+=1
    if p[q]==ref[q]: question[q]['correct']+=1
  metrics[name]={'successes':len(preds)-errors,'errors':errors,'per_question_exact_match':{q:{**v,'accuracy':v['correct']/v['total'] if v['total'] else None} for q,v in question.items()},'all_three_exact_match':sum(all(p[q]==byid[str(p['id'])]['reference'][q] for q in question) for p in preds),'case_count':len(preds)}
 (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
 summary={'run_id':OUT.name,'case_count':90,'failures':failures,'model_successes':{m:90-failures['models'][m] for m in MODELS},'jev_successes':90-failures['jev'],'sha256':{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ('summary.json','run_baseline.py')}}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps(summary,indent=2))
 return 0 if all(v==0 for v in failures['models'].values()) and failures['jev']==0 else 2
if __name__=='__main__': raise SystemExit(main())
