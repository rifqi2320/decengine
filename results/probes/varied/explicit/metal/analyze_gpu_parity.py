#!/usr/bin/env python3
"""Parity-only comparison of GPU outputs with archived NumPy prediction outputs."""
import hashlib,json,math,pathlib,re,statistics

ROOT=pathlib.Path(__file__).resolve().parents[5]
FIXTURE=ROOT/"benchmarks/cases/varied/multi/test-explicit-60.jsonl"
EXPECTED_FIXTURE="4cb002e1c29f5273ecc23fcdf742b093cb5433d661f5df0b50d209a534fb1409"
HERE=pathlib.Path(__file__).resolve().parent

def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def read(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def pct(xs,p):
 xs=sorted(xs); x=(len(xs)-1)*p; lo=int(x); hi=min(lo+1,len(xs)-1); return xs[lo]+(xs[hi]-xs[lo])*(x-lo)
if sha(FIXTURE)!=EXPECTED_FIXTURE:raise SystemExit("explicit fixture hash changed")
report={"scope":"all-MLX explicit60 predictions compared only to archived CPU prediction outputs; no labels/references opened",
 "fixture_sha256":sha(FIXTURE),"frozen_models":{},"models":{}}
for model in ("qwen","harrier"):
 gpu_dir=HERE/"runs"/model
 gpu=read(gpu_dir/"predictions.jsonl")
 cpu_path=ROOT/f"results/probes/varied/explicit/runs/{model}/cpu-reference/evaluation/predictions.jsonl"
 cpu=read(cpu_path)
 gm={(r["id"],r["question_name"]):r for r in gpu}; cm={(r["id"],r["question_name"]):r for r in cpu}
 if len(gm)!=180 or len(cm)!=180 or set(gm)!=set(cm):raise SystemExit(f"{model}: expected exactly 180 matching question keys")
 types={k:{"n":0,"exact_selected":0,"exact_label_or_boolean":0,"exact_score_argmax":0,"max_probability_abs_diff":0.0} for k in ("choice","noul","score")}
 max_ev=0.0; bad=[]; cases={}
 for key,c in cm.items():
  g=gm[key]; kind=c["question_type"]
  if set(c["candidate_keys"])!=set(g["candidate_keys"]):raise SystemExit(f"{model} {key}: candidate ID mismatch")
  t=types[kind];t["n"]+=1
  pd=max(abs(float(c["probabilities"][k])-float(g["probabilities"][k])) for k in c["candidate_keys"])
  t["max_probability_abs_diff"]=max(t["max_probability_abs_diff"],pd)
  t["exact_selected"]+=c["selected"]==g["selected"]
  t["exact_label_or_boolean"]+=c["label"]==g["label"] and c.get("value")==g.get("value")
  if kind=="score":
   t["exact_score_argmax"]+=c["selected_level"]==g["selected_level"]
   max_ev=max(max_ev,abs(float(c["expected_value"])-float(g["expected_value"])))
  if c["selected"]!=g["selected"] or c["label"]!=g["label"] or c.get("value")!=g.get("value") or (kind=="score" and c["selected_level"]!=g["selected_level"]):bad.append((key,c["selected"],g["selected"]))
  cases.setdefault(key[0],[]).append(c["selected"]==g["selected"] and c["label"]==g["label"] and c.get("value")==g.get("value") and (kind!="score" or c["selected_level"]==g["selected_level"]))
 if bad:raise SystemExit(f"{model}: prediction parity failure: {bad[:5]}")
 if len(cases)!=60 or any(len(v)!=3 for v in cases.values()):raise SystemExit(f"{model}: expected 60 cases with three questions")
 timings=read(gpu_dir/"predictions.jsonl.case-timings.jsonl")
 if len(timings)!=60:raise SystemExit(f"{model}: expected 60 synchronized case timings")
 case_ms=[r["case_total_ns"]/1e6 for r in timings]
 resource=(gpu_dir/"time-resource.txt").read_text()
 elapsed=re.search(r"([0-9.]+) real\s+([0-9.]+) user\s+([0-9.]+) sys",resource)
 rss=re.search(r"\n\s*(\d+)\s+maximum resident set size",resource)
 footprint=re.search(r"\n\s*(\d+)\s+peak memory footprint",resource)
 if not (elapsed and rss and footprint):raise SystemExit(f"{model}: incomplete process resource report")
 for t in types.values():
  for field in ("exact_selected","exact_label_or_boolean","exact_score_argmax"):
   t[field+"_count"]=t[field];del t[field]
  t["probability_parity_tolerance_claimed"]=False
 report["frozen_models"][model]=sha(ROOT/f"results/probes/varied/matched/frozen-{model}/models.json")
 report["models"][model]={"question_count":180,"per_type":types,"all_three_question_predictions_exact_cases":sum(all(v) for v in cases.values()),
   "all_three_case_parity":all(all(v) for v in cases.values()),"max_score_expected_value_abs_diff":max_ev,
   "max_probability_abs_diff_overall":max(t["max_probability_abs_diff"] for t in types.values()),
   "three_question_case_total_ms":{"p50":pct(case_ms,.5),"p95":pct(case_ms,.95)},
   "runtime":{"backend":"MLX/Metal Device(gpu, 0)","process_including_load_warmup":{"real_seconds":float(elapsed.group(1)),"user_seconds":float(elapsed.group(2)),"system_seconds":float(elapsed.group(3)),"maximum_resident_set_bytes":int(rss.group(1)),"peak_memory_footprint_bytes":int(footprint.group(1))},"conditions":(gpu_dir/"conditions.txt").read_text()},
   "timing_scope":"synchronized GPU embedding + GPU scorer over three questions; excludes model load and one per-type warmup",
   "sha256":{"gpu_predictions":sha(gpu_dir/"predictions.jsonl"),"cpu_predictions":sha(cpu_path),"fixture":sha(FIXTURE),
     "gpu_question_timings":sha(gpu_dir/"predictions.jsonl.timings.jsonl"),"gpu_case_timings":sha(gpu_dir/"predictions.jsonl.case-timings.jsonl"),
     "run_metadata":sha(gpu_dir/"time-resource.txt")}}
report_path=HERE/"gpu-parity-report.json"
report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
lines=["# Explicit60 all-MLX prediction parity", "", "Compared only against the archived CPU prediction outputs; no target/reference labels were read. Same frozen typed scorers, no fitting or tuning. Every question has exact candidate identifiers.","",
"| Model | Type | n | selected/boolean/score exact | max probability abs diff |", "|---|---|---:|---:|---:|"]
for m,r in report["models"].items():
 for kind,t in r["per_type"].items():
  exact=t["exact_selected_count"]==t["n"] and t["exact_label_or_boolean_count"]==t["n"] and (kind!="score" or t["exact_score_argmax_count"]==t["n"])
  lines.append(f"| {m} | {kind} | {t['n']} | {exact} ({t['exact_selected_count']}/{t['n']}) | {t['max_probability_abs_diff']:.9g} |")
 lines.append(f"| {m} | all-three case prediction parity | 60 | {r['all_three_question_predictions_exact_cases']}/60 | — |")
 lines.append(f"| {m} | 3-question GPU case time p50/p95 | 60 | — | {r['three_question_case_total_ms']['p50']:.2f}/{r['three_question_case_total_ms']['p95']:.2f} ms |")
(HERE/"GPU_PARITY_REPORT.md").write_text("\n".join(lines)+"\n")
print((HERE/"GPU_PARITY_REPORT.md").read_text())
