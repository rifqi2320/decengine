#!/usr/bin/env python3
"""Compare Metal results to the secondary CPU variant-head predictions only."""
import hashlib,json,pathlib,re,statistics

ROOT=pathlib.Path(__file__).resolve().parents[5]
PROMPT=ROOT/"results/probes/varied/prompt-study"
STUDY=PROMPT/"runs/trainonly-clean300-20260924T002000Z"
HOLD=PROMPT/"runs/holdout-fresh60-prompt-study-20260924T000000Z"
SECOND=HOLD/"secondary/variant-heads"
RUN=PROMPT/"metal/runs/fresh60/per-variant-heads"
FIXTURE=ROOT/"benchmarks/cases/varied/multi/test-prompt-fresh-60.jsonl"
VARIANTS=("v2-baseline","readable-state","rule-in-candidate","joint-context-candidate")
MODELS={"qwen":"Qwen/Qwen3-Embedding-0.6B","harrier":"microsoft/harrier-oss-v1-0.6b"}

def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def read(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def pct(xs,p):
 xs=sorted(xs);pos=(len(xs)-1)*p;lo=int(pos);hi=min(lo+1,len(xs)-1)
 return xs[lo]+(xs[hi]-xs[lo])*(pos-lo)
def stats(xs):return {"p50":pct(xs,.5),"p95":pct(xs,.95),"mean":statistics.mean(xs)}

handoff=json.loads((SECOND/"gpu-variant-head-handoff.json").read_text())
meta=json.loads((HOLD/"run-metadata.json").read_text())
if sha(FIXTURE)!=handoff["holdout_sha256"]:raise SystemExit("secondary holdout hash mismatch")
report={"protocol":"per-variant full-train heads vs secondary pre-frozen CPU variant-head predictions",
 "scope":"prediction-only parity; no fresh60 targets/labels read",
 "case_count":60,"question_count":180,"fixture_sha256":sha(FIXTURE),"run_order":(RUN/"run-order.txt").read_text(),"models":{}}
for mk,model in MODELS.items():
 by_variant={}
 for variant in VARIANTS:
  d=RUN/mk/variant; bundle_path=PROMPT/"metal/frozen-scorers/per-variant"/mk/f"{variant}.json"
  bundle=json.loads(bundle_path.read_text()); source_sha=handoff["models"][mk][variant]["head_sha256"]
  if bundle["provenance"]["source_bundle_sha256"]!=source_sha:raise SystemExit(f"{mk}/{variant}: wrapped head differs from reference source hash")
  if bundle["model"]!=model or bundle["C"]!=handoff["models"][mk][variant]["C"]:raise SystemExit(f"{mk}/{variant}: head model/C mismatch")
  cpu_path=SECOND/"predictions"/mk/f"{variant}.jsonl"
  cpu=read(cpu_path);gpu=read(d/"predictions.jsonl")
  rec=handoff["models"][mk][variant]
  if sha(cpu_path)!=rec["cpu_prediction_sha256"]:raise SystemExit(f"{mk}/{variant}: secondary reference prediction hash changed")
  cpu_by={(r["id"],name):{**decision,"model":r["model"],"variant":r["variant"],"scorer_head_sha256":r["scorer_head_sha256"]}
    for r in cpu for name,decision in r["decisions"].items()}
  gpu_by={(x["id"],x["question_name"]):x for x in gpu}
  if len(cpu_by)!=180 or len(gpu_by)!=180 or set(cpu_by)!=set(gpu_by):raise SystemExit(f"{mk}/{variant}: 180 question keys do not align")
  by_type={k:{"n":0,"selected_exact":0,"label_exact":0,"score_argmax_exact":0,"max_probability_abs_diff":0.0} for k in ("choice","noul","score")}
  score_diffs=[];all_diffs=[];bad=[];case_hits={}
  for key,c in cpu_by.items():
   g=gpu_by[key];kind=c["question_type"]
   if c["model"]!=model or c["variant"]!=variant or g["model"]!=model or g["variant"]!=variant:raise SystemExit(f"{mk}/{variant}/{key}: model/variant mismatch")
   if c["scorer_head_sha256"]!=source_sha or g["head_mode"]!="per-variant" or g["scorer_bundle_sha256"]!=sha(bundle_path):raise SystemExit(f"{mk}/{variant}/{key}: scorer head hash mismatch")
   cp=c["probabilities"];gp=g["probabilities"]
   if set(cp)!=set(gp) or set(cp)!=set(g["candidate_keys"]):raise SystemExit(f"{mk}/{variant}/{key}: candidate-key mismatch")
   diff=max(abs(float(cp[k])-float(gp[k])) for k in cp);all_diffs.append(diff)
   t=by_type[kind];t["n"]+=1;t["max_probability_abs_diff"]=max(t["max_probability_abs_diff"],diff)
   selected=c["selected"]==g["selected"]
   value_exact=(c.get("expected_value")==g.get("value") if kind=="noul" else True)
   label=(c["label"]==g["label"] and value_exact)
   t["selected_exact"]+=selected;t["label_exact"]+=label
   score_exact=True
   if kind=="score":
    score_exact=c["selected"]==g["selected"]
    t["score_argmax_exact"]+=score_exact
    score_diffs.append(abs(float(c["expected_value"])-float(g["expected_value"])))
   if not selected or not label or not score_exact:bad.append((key,c["selected"],g["selected"]))
   case_hits.setdefault(key[0],[]).append(selected and label and score_exact)
  if bad:raise SystemExit(f"{mk}/{variant}: parity mismatch {bad[:5]}")
  if len(case_hits)!=60 or any(len(x)!=3 for x in case_hits.values()):raise SystemExit(f"{mk}/{variant}: expected 60 three-question cases")
  timing=read(d/"predictions.jsonl.timings.jsonl");cases=read(d/"predictions.jsonl.case-timings.jsonl")
  if len(timing)!=180 or len(cases)!=60:raise SystemExit(f"{mk}/{variant}: timing row count mismatch")
  resource=(d/"time-resource.txt").read_text()
  real=re.search(r"([0-9.]+) real\s+([0-9.]+) user\s+([0-9.]+) sys",resource)
  rss=re.search(r"\n\s*(\d+)\s+maximum resident set size",resource)
  footprint=re.search(r"\n\s*(\d+)\s+peak memory footprint",resource)
  if not(real and rss and footprint):raise SystemExit(f"{mk}/{variant}: incomplete resource metrics")
  token_q=[sum(x["candidate_token_counts"])+x["query_token_count"] for x in timing]
  case_tokens={}
  for x,tok in zip(timing,token_q):case_tokens[x["id"]]=case_tokens.get(x["id"],0)+tok
  clean_type={k:{"n":v["n"],"selected_exact":v["selected_exact"]==v["n"],"label_or_boolean_exact":v["label_exact"]==v["n"],
      "score_argmax_exact":(v["score_argmax_exact"]==v["n"] if k=="score" else None),"max_probability_abs_diff":v["max_probability_abs_diff"]} for k,v in by_type.items()}
  by_variant[variant]={"head_mode":"per-variant","C":bundle["C"],"source_head_sha256":source_sha,"wrapped_gpu_bundle_sha256":sha(bundle_path),
   "scorer_array_sha256":rec["normalization_and_weights"],
   "per_type":clean_type,"all_three_cases_exact":sum(all(x) for x in case_hits.values()),"max_probability_abs_diff":max(all_diffs),
   "max_score_expected_value_abs_diff":max(score_diffs),
   "timing_ms":{"case_total":stats([x["case_total_ns"]/1e6 for x in cases]),"embedding_per_question":stats([x["embedding_ns"]/1e6 for x in timing]),
      "gpu_scorer_per_question":stats([x["gpu_scorer_ns"]/1e6 for x in timing])},
   "tokens":{"query_plus_candidate_per_question":stats(token_q),"query_plus_candidate_per_case":stats(list(case_tokens.values())),"total_tokens":sum(token_q),
      "estimated_dense_embedding_flops_2pt":2*600_000_000*sum(token_q)},
   "process_including_load_warmup":{"real_seconds":float(real.group(1)),"user_seconds":float(real.group(2)),"system_seconds":float(real.group(3)),"maximum_resident_set_bytes":int(rss.group(1)),"peak_memory_footprint_bytes":int(footprint.group(1))},
   "hashes":{"gpu_predictions":sha(d/"predictions.jsonl"),"gpu_question_timings":sha(d/"predictions.jsonl.timings.jsonl"),"gpu_case_timings":sha(d/"predictions.jsonl.case-timings.jsonl"),
    "cpu_predictions":sha(cpu_path),"source_frozen_head":source_sha,"wrapper_bundle":sha(bundle_path),"conditions":sha(d/"conditions.txt"),"resource_log":sha(d/"time-resource.txt")},
   "conditions":(d/"conditions.txt").read_text()}
 report["models"][mk]=by_variant
OUT=RUN/"PARITY_TIMING_REPORT.json";OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
lines=["# Fresh60 prompt variants: per-variant frozen heads, all-MLX parity","","Matched against the separate secondary CPU prediction artifacts. Each per-variant full-train scorer bundle/hash and C is bound to the secondary handoff; no holdout labels were read.","",
"| Model | Variant | head C | typed outputs exact (180/180) | all-three | max prob abs diff | max score EV diff | case p50/p95 ms | GPU scorer p50/p95 ms/question | tokens/case p50/p95 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for mk,variants in report["models"].items():
 for v,r in variants.items():
  exact=all(x["selected_exact"] and x["label_or_boolean_exact"] and (x["score_argmax_exact"] is not False) for x in r["per_type"].values())
  c=r["timing_ms"]["case_total"];s=r["timing_ms"]["gpu_scorer_per_question"];t=r["tokens"]["query_plus_candidate_per_case"]
  lines.append(f"| {mk} | {v} | {r['C']:g} | {exact} | {r['all_three_cases_exact']}/60 | {r['max_probability_abs_diff']:.9g} | {r['max_score_expected_value_abs_diff']:.9g} | {c['p50']:.2f}/{c['p95']:.2f} | {s['p50']:.3f}/{s['p95']:.3f} | {t['p50']:.0f}/{t['p95']:.0f} |")
(RUN/"PARITY_TIMING_REPORT.md").write_text("\n".join(lines)+"\n")
print((RUN/"PARITY_TIMING_REPORT.md").read_text())
