#!/usr/bin/env python3
"""Compare persisted all-MLX outputs to pre-frozen mixed-GPU/CPU prediction files."""
import hashlib,json,math,pathlib,re,statistics

ROOT=pathlib.Path(__file__).resolve().parents[5]
PROMPT=ROOT/"results/probes/varied/prompt-study"
STUDY=PROMPT/"runs/trainonly-clean300-20260924T002000Z"
HOLD=PROMPT/"runs/holdout-fresh60-prompt-study-20260924T000000Z"
RUN=PROMPT/"metal/runs/fresh60/shared-v2-baseline"
FIXTURE=ROOT/"benchmarks/cases/varied/multi/test-prompt-fresh-60.jsonl"
VARIANTS=("v2-baseline","readable-state","rule-in-candidate","joint-context-candidate")
MODELS={"qwen":"Qwen/Qwen3-Embedding-0.6B","harrier":"microsoft/harrier-oss-v1-0.6b"}

def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def sha_array(values):return hashlib.sha256(json.dumps(values,separators=(",",":")).encode()).hexdigest()
def read(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def pct(xs,p):
 xs=sorted(xs);pos=(len(xs)-1)*p;lo=int(pos);hi=min(lo+1,len(xs)-1)
 return xs[lo]+(xs[hi]-xs[lo])*(pos-lo)
def qstats(xs):return {"p50":pct(xs,.5),"p95":pct(xs,.95),"mean":statistics.mean(xs)}

meta=json.loads((HOLD/"run-metadata.json").read_text())
freeze=json.loads((HOLD/"PRE_REFERENCE_FREEZE.json").read_text())
if freeze.get("status")!="predictions_frozen_before_reference_scoring":raise SystemExit("CPU reference prediction freeze missing")
if sha(FIXTURE)!=meta["fixture_sha256"] or sha(FIXTURE)!=freeze["fixture_sha256"]:raise SystemExit("fresh fixture hash mismatch")
report={"protocol":"shared-v2-baseline-head across all four text variants; all-MLX Metal inference compared with frozen CPU-NumPy reference predictions",
 "reference_scope":"prediction-output-only parity; no holdout target/label/reference files were opened",
 "case_count":60,"question_count":180,"models":{},"execution_order":(RUN/"run-order.txt").read_text(),
 "timing_scope":"synchronized per-question embedding + GPU scorer, summed as three-question case total; excludes model load and one-per-type warmup",
 "fixture_sha256":sha(FIXTURE),"frozen_config_sha256":sha(STUDY/"frozen-config.json"),
 "head_protocol":{"qwen_C":1.0,"harrier_C":0.1,"same_frozen_weights_mean_scale_for_each_model_across_all_prompt_variants":True}}
for mk,model in MODELS.items():
 model_results={}
 for variant in VARIANTS:
  d=RUN/mk/variant
  gpu=read(d/"predictions.jsonl")
  cpu_path=HOLD/"predictions"/mk/f"{variant}.jsonl"
  cpu=read(cpu_path)
  cpu_by={(row["id"],qname):{**decision,"question_type":decision["question_type"],"model":row["model"],"variant":row["variant"]}
          for row in cpu for qname,decision in row["decisions"].items()}
  gpu_by={(p["id"],p["question_name"]):p for p in gpu}
  expected_hash=freeze["prediction_hashes"][f"{mk}/{variant}"]
  if sha(cpu_path)!=expected_hash or sha(cpu_path)!=meta["models"][mk]["variants"][variant]["predictions_sha256"]:
   raise SystemExit(f"{mk}/{variant}: CPU reference hash mismatch")
  if len(gpu_by)!=180 or len(cpu_by)!=180 or set(gpu_by)!=set(cpu_by):raise SystemExit(f"{mk}/{variant}: question-key set mismatch")
  by_type={k:{"n":0,"exact_selected":0,"exact_label_or_boolean":0,"exact_score_label":0,"max_probability_abs_diff":0.0} for k in ("choice","noul","score")}
  expected_value_diff=[];bad=[];case_parity={};all_prob_diffs=[]
  for key,c in cpu_by.items():
   g=gpu_by[key];kind=c["question_type"]
   if c["model"]!=model or c["variant"]!=variant or g["model"]!=model or g["variant"]!=variant:raise SystemExit(f"{mk}/{variant}/{key}: provenance mismatch")
   cp=c["probabilities"];gp=g["probabilities"]
   if set(cp)!=set(gp) or set(cp)!=set(g["candidate_keys"]):raise SystemExit(f"{mk}/{variant}/{key}: candidate key mismatch")
   dif=max(abs(float(cp[k])-float(gp[k])) for k in cp)
   by_type[kind]["n"]+=1;by_type[kind]["max_probability_abs_diff"]=max(by_type[kind]["max_probability_abs_diff"],dif);all_prob_diffs.append(dif)
   selected=(c["selected"]==g["selected"])
   value_parity=(c.get("expected_value")==g.get("value") if kind=="noul" else True)
   label=(c["label"]==g["label"] and value_parity)
   score=(kind!="score" or c.get("selected")==g.get("selected") and c.get("expected_value") is not None and g.get("expected_value") is not None)
   by_type[kind]["exact_selected"]+=selected;by_type[kind]["exact_label_or_boolean"]+=label
   if kind=="score":
    by_type[kind]["exact_score_label"]+=c["selected"]==g["selected"]
    expected_value_diff.append(abs(float(c["expected_value"])-float(g["expected_value"])))
   if not selected or not label or not score:bad.append((key,c.get("selected"),g.get("selected")))
   case_parity.setdefault(key[0],[]).append(selected and label and score)
  if len(case_parity)!=60 or any(len(v)!=3 for v in case_parity.values()):raise SystemExit(f"{mk}/{variant}: not 60 cases x three question types")
  if bad:raise SystemExit(f"{mk}/{variant}: selected/boolean/score-label mismatch {bad[:5]}")
  timing=read(d/"predictions.jsonl.timings.jsonl");cases=read(d/"predictions.jsonl.case-timings.jsonl")
  if len(timing)!=180 or len(cases)!=60:raise SystemExit(f"{mk}/{variant}: timing count mismatch")
  case_ms=[x["case_total_ns"]/1e6 for x in cases]
  emb_ms=[x["embedding_ns"]/1e6 for x in timing]
  gpu_ms=[x["gpu_scorer_ns"]/1e6 for x in timing]
  tokens=[sum(x["candidate_token_counts"])+x["query_token_count"] for x in timing]
  tokens_by_case={}
  for x in timing:tokens_by_case[x["id"]]=tokens_by_case.get(x["id"],0)+sum(x["candidate_token_counts"])+x["query_token_count"]
  resource=(d/"time-resource.txt").read_text()
  real=re.search(r"([0-9.]+) real\s+([0-9.]+) user\s+([0-9.]+) sys",resource)
  rss=re.search(r"\n\s*(\d+)\s+maximum resident set size",resource)
  footprint=re.search(r"\n\s*(\d+)\s+peak memory footprint",resource)
  if not(real and rss and footprint):raise SystemExit(f"{mk}/{variant}: missing process resource metrics")
  bundle=PROMPT/"metal/frozen-scorers/shared-v2-baseline"/mk/f"{variant}.json"
  model_json=STUDY/"frozen"/mk/"models.json"
  bundle_doc=json.loads(bundle.read_text())
  if bundle_doc["provenance"]["models_sha256"]!=sha(model_json):raise SystemExit("bundle is not bound to shared frozen model")
  array_hashes={kind:{field:sha_array(bundle_doc["scorers"][kind][field]) for field in ("weights","mean","scale")} for kind in ("choice","noul","score")}
  by_type_summary={}
  for k,x in by_type.items():
   by_type_summary[k]={"n":x["n"],"selected_exact":x["exact_selected"]==x["n"],"label_boolean_exact":x["exact_label_or_boolean"]==x["n"],
      "score_argmax_exact":(x["exact_score_label"]==x["n"] if k=="score" else None),"max_probability_abs_diff":x["max_probability_abs_diff"]}
  model_results[variant]={"head_mode":"shared-v2-baseline","C":bundle_doc["C"],"scorer_array_sha256":array_hashes,
   "typed_parity":by_type_summary,"all_three_cases_exact":sum(all(v) for v in case_parity.values()),
   "max_probability_abs_diff":max(all_prob_diffs),"max_score_expected_value_abs_diff":max(expected_value_diff),
   "timing_ms":{"three_question_case_total":qstats(case_ms),"embedding_per_question":qstats(emb_ms),"gpu_scorer_per_question":qstats(gpu_ms)},
   "tokens":{"query_plus_candidate_per_question":qstats(tokens),"query_plus_candidate_per_three_question_case":qstats(list(tokens_by_case.values())),
      "total_query_plus_candidate_tokens":sum(tokens),"estimated_dense_embedding_flops_2pt":2*600_000_000*sum(tokens)},
   "process_including_load_warmup":{"real_seconds":float(real.group(1)),"user_seconds":float(real.group(2)),"system_seconds":float(real.group(3)),
      "maximum_resident_set_bytes":int(rss.group(1)),"peak_memory_footprint_bytes":int(footprint.group(1))},
   "hashes":{"gpu_predictions":sha(d/"predictions.jsonl"),"gpu_timings":sha(d/"predictions.jsonl.timings.jsonl"),
      "gpu_case_timings":sha(d/"predictions.jsonl.case-timings.jsonl"),"bundle":sha(bundle),"frozen_models":sha(model_json),
      "cpu_reference_predictions":sha(cpu_path),"feature_manifest":sha(HOLD/"features"/mk/variant/"fresh60.features.jsonl.manifest.json"),
      "run_resource_log":sha(d/"time-resource.txt")},"conditions":(d/"conditions.txt").read_text()}
 report["models"][mk]=model_results
REPORT=RUN/"parity-timing-report.json";REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
lines=["# Fresh60 prompt variants: shared v2 head, all-MLX parity", "", "Inference mode: one unchanged v2-baseline frozen head (weights, means, scales, C) per model reused across all four prompt-only text variants. Compared with the pre-reference-frozen CPU NumPy prediction files; only predictions were read, no targets/labels.","",
"| Model | Prompt variant | all 180 selected/typed values exact | all-three cases | max probability abs diff | max score EV diff | case p50/p95 ms | embed p50/p95 ms/question | GPU scorer p50/p95 ms/question | tokens/case p50/p95 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for mk,variants in report["models"].items():
 for v,r in variants.items():
  exact=all(x["selected_exact"] and x["label_boolean_exact"] and (x["score_argmax_exact"] is not False) for x in r["typed_parity"].values())
  c=r["timing_ms"]["three_question_case_total"];e=r["timing_ms"]["embedding_per_question"];s=r["timing_ms"]["gpu_scorer_per_question"];t=r["tokens"]["query_plus_candidate_per_three_question_case"]
  lines.append(f"| {mk} | {v} | {exact} | {r['all_three_cases_exact']}/60 | {r['max_probability_abs_diff']:.9g} | {r['max_score_expected_value_abs_diff']:.9g} | {c['p50']:.2f}/{c['p95']:.2f} | {e['p50']:.2f}/{e['p95']:.2f} | {s['p50']:.3f}/{s['p95']:.3f} | {t['p50']:.0f}/{t['p95']:.0f} |")
(RUN/"PARITY_TIMING_REPORT.md").write_text("\n".join(lines)+"\n")
print((RUN/"PARITY_TIMING_REPORT.md").read_text())
