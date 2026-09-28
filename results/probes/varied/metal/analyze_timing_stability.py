#!/usr/bin/env python3
"""Summarize isolated matched60 repetitions; only outputs/timings/feature metadata are read."""
import hashlib, json, math, pathlib, re, statistics

ROOT = pathlib.Path(__file__).resolve().parents[4]
RUNS = ROOT / "results/probes/varied/metal/runs"
OUT = RUNS / "timing-stability"
OUT.mkdir(parents=True, exist_ok=True)

def read(path): return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def percentile(xs, p):
    xs=sorted(xs); pos=(len(xs)-1)*p; lo=int(pos); hi=min(lo+1,len(xs)-1)
    return xs[lo]+(xs[hi]-xs[lo])*(pos-lo)
def qstats(xs): return {"p50":percentile(xs,.50),"p95":percentile(xs,.95),"mean":statistics.mean(xs)}
def corr(x,y):
    mx,my=statistics.mean(x),statistics.mean(y)
    vx=sum((a-mx)**2 for a in x); vy=sum((b-my)**2 for b in y)
    return sum((a-mx)*(b-my) for a,b in zip(x,y))/math.sqrt(vx*vy) if vx and vy else None

report={"scope":"matched60 all-MLX frozen inference; no targets/references opened by this script",
        "protocol":{"additional_repetitions_per_model":2,"sequential":True,"order":["replicate-1: harrier then qwen","replicate-2: qwen then harrier"],
          "each_run":"fresh process, model loaded, one warmup per typed scorer excluded; every measured GPU stage synchronized",
          "power":"AC power for every additional run; exact battery/load snapshot in each run conditions.txt",
          "cache":"OS page cache was not flushed; model/load and three typed warmups repeated each invocation; no concurrent MLX model loads",
          "initial_run_conditions":"not captured; its time-resource.txt is retained"},"runs":{}}
for model in ("qwen","harrier"):
  model_report={}
  feature_path=ROOT/f"results/probes/varied/matched/runs/{model}/matched.features.jsonl"
  feature_rows=read(feature_path)
  token_by_key={}
  for row in feature_rows:
    for cand in row["candidates"]:
      token_by_key[(row["id"],row["question_key"],cand["candidate_id"])]=int(cand["token_length"])
    token_by_key[(row["id"],row["question_key"],"__query__")]=int(row["query_token_length"])
  for run_name in (model,"replicate-1/"+model,"replicate-2/"+model):
    d=RUNS/run_name
    cases=read(d/"predictions.jsonl.case-timings.jsonl")
    questions=read(d/"predictions.jsonl.timings.jsonl")
    case_ms=[r["case_total_ns"]/1e6 for r in cases]
    embed_ms=[r["embedding_ns"]/1e6 for r in questions]
    scorer_ms=[r["gpu_scorer_ns"]/1e6 for r in questions]
    # Original first-run sidecars predate explicit token-count columns. Recover exact token
    # counts from the matching verified exporter rows without opening reference fields.
    if "query_token_count" not in questions[0]:
      for q in questions:
        q["query_token_count"]=token_by_key[(q["id"],q["question_name"],"__query__")]
        q["candidate_token_counts"]=[v for (case,name,cand),v in token_by_key.items() if case==q["id"] and name==q["question_name"] and cand!="__query__"]
        q["total_candidate_tokens"]=sum(q["candidate_token_counts"])
    tok=[r["query_token_count"]+r["total_candidate_tokens"] for r in questions]
    resource=(d/"time-resource.txt").read_text()
    elapsed=re.search(r"([0-9.]+) real\s+([0-9.]+) user\s+([0-9.]+) sys",resource)
    rss=re.search(r"\n\s*(\d+)\s+maximum resident set size",resource)
    footprint=re.search(r"\n\s*(\d+)\s+peak memory footprint",resource)
    switches=re.search(r"\n\s*(\d+)\s+involuntary context switches",resource)
    instructions=re.search(r"\n\s*(\d+)\s+instructions retired",resource)
    cycles=re.search(r"\n\s*(\d+)\s+cycles elapsed",resource)
    conditions=(d/"conditions.txt").read_text() if (d/"conditions.txt").exists() else "initial-run environment snapshot unavailable"
    model_report[run_name]={"case_count":len(cases),"question_count":len(questions),
      "case_total_ms":qstats(case_ms),"question_embedding_ms":qstats(embed_ms),"question_gpu_scorer_ms":qstats(scorer_ms),
      "query_plus_candidate_tokens_per_question":qstats(tok),"embedding_ms_token_count_pearson":corr(tok,embed_ms),
      "embedding_ms_per_1000_tokens":1000*statistics.mean(embed_ms)/statistics.mean(tok),
      "process_including_load_warmup":({"real_seconds":float(elapsed.group(1)),"maximum_resident_set_bytes":int(rss.group(1)),"peak_memory_footprint_bytes":int(footprint.group(1)),
        "involuntary_context_switches":int(switches.group(1)) if switches else None,"instructions_retired":int(instructions.group(1)) if instructions else None,"cycles_elapsed":int(cycles.group(1)) if cycles else None} if elapsed and rss and footprint else None),
      "prediction_sha256":sha(d/"predictions.jsonl"),"conditions":conditions}
  model_report["replicate_pair_relative_spread"]={}
  for stage in ("case_total_ms","question_embedding_ms","query_plus_candidate_tokens_per_question"):
    a=model_report[f"replicate-1/{model}"][stage]["p50"]
    b=model_report[f"replicate-2/{model}"][stage]["p50"]
    model_report["replicate_pair_relative_spread"][stage+"_p50_pct"]=abs(a-b)/statistics.mean([a,b])*100
  report["runs"][model]=model_report

report["prediction_determinism"]={model:len({v["prediction_sha256"] for k,v in runs.items() if k==model or k.endswith("/"+model)})==1 for model,runs in report["runs"].items()}
report["preexisting_matched_encoder_flops_estimate"]={"source":"results/probes/varied/matched/compute-speed.md","per_three-question_case_p50_gflops":904.71,"per_three-question_case_p95_gflops":945.46,"scope_note":"Same matched60 fixture and query/candidate texts; encoder-only analytical estimate, not measured device FLOPs. The report's timed run was export + CPU scoring, so its latency is not compared to all-MLX timings."}
report["caveat"]="The first Qwen all-MLX run is substantially slower than both repeats; those repeats cluster tightly with Harrier, despite identical p50 query+candidate token count (391). The initial run lacks a contemporaneous host-load snapshot, so the cause cannot be attributed to model architecture, cache, thermal, or host contention. Repetitions were sequential and on AC but snapshots show variable background activity and nonzero involuntary context switches; OS page cache was left intact. CPU instructions/cycles do not establish GPU clock/state. Treat the initial 527 ms p50 as an outlier, not a model-level Qwen-vs-Harrier performance difference."
(OUT/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
lines=["# Matched60 timing stability", "", "No references/targets were opened. All repeated MLX invocations were sequential, with a fresh model load and one warmup per type; order alternated between repetitions. AC status, load snapshot and resource reports are stored per run. The OS page cache was not cleared.","",
"| Model/run | case p50/p95 ms | embedding p50/p95 ms/question | scorer p50/p95 ms/question | query+candidate tokens p50 | embed/token Pearson |", "|---|---:|---:|---:|---:|---:|"]
for model in ("qwen","harrier"):
  for run,rec in report["runs"][model].items():
    if run=="replicate_pair_relative_spread": continue
    c=rec["case_total_ms"]; e=rec["question_embedding_ms"]; s=rec["question_gpu_scorer_ms"]
    lines.append(f"| {run} | {c['p50']:.2f}/{c['p95']:.2f} | {e['p50']:.2f}/{e['p95']:.2f} | {s['p50']:.3f}/{s['p95']:.3f} | {rec['query_plus_candidate_tokens_per_question']['p50']:.0f} | {rec['embedding_ms_token_count_pearson']:.3f} |")
lines += ["",report["caveat"],"", "All measurements and exact process-condition snapshots are in `report.json`; initial run outputs were not rewritten."]
(OUT/"report.md").write_text("\n".join(lines)+"\n")
print((OUT/"report.md").read_text())
