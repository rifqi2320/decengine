#!/usr/bin/env python3
"""Recompute the archived 300-case accuracy/timing/FLOPs comparison (no inference)."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[3]
LAYA = ROOT / "results/laya/mlx/runs/20260923T190000Z"
PROBES = ROOT / "results/probes/comparisons/20260923T104208Z"
FIXTURE = ROOT / "benchmarks/cases/probe-test-300.jsonl"
STORE = Path("/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/decengine-model-store/models")
TASKS = ("owner", "urgent", "impact")


def jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def stats(values, nearest_rank=False):
    xs = sorted(map(float, values))
    def pct(p):
        if nearest_rank:
            return xs[math.ceil(p * len(xs)) - 1]
        pos = (len(xs) - 1) * p
        lo, hi = math.floor(pos), math.ceil(pos)
        return xs[lo] if lo == hi else xs[lo] * (hi - pos) + xs[hi] * (pos - lo)
    return {"n": len(xs), "mean_ms": statistics.fmean(xs), "p50_ms": pct(.5), "p95_ms": pct(.95)}


def acc(cases, preds):
    refs = {r["id"]: r["reference"] for r in cases}
    assert len(preds) == 300 and {p["id"] for p in preds} == set(refs)
    preds_byid = {p["id"]: p for p in preds}
    q = {}
    for task in TASKS:
        n = sum(preds_byid[p["id"]].get(task) == refs[p["id"]][task] for p in preds)
        q[task] = {"correct": n, "total": 300, "accuracy": n / 300}
    complete = sum(all(preds_byid[p["id"]].get(t) == refs[p["id"]][t] for t in TASKS) for p in preds)
    return {"per_question": q, "question_mean_accuracy": statistics.mean(x["accuracy"] for x in q.values()),
            "complete_case": {"correct": complete, "total": 300, "accuracy": complete / 300}}


def compiled_text(case, task, profile):
    req = case["request"]
    state = json.dumps(req["state"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    q = req["questions"][task]
    prompt = q["prompt"]
    kind = {"owner": "choose the single best-matching option from the reported issue and candidate scopes; do not infer facts absent from the case",
            "urgent": "decide whether same-day action is needed to prevent likely harm, material interruption, or a near-term missed deadline; an available safe workaround or no concrete near-term consequence favors no; emphatic wording alone is not evidence",
            "impact": "select the highest impact supported if unresolved; distinguish routine/no concrete near-term risk, localized disruption or safe workaround, and credible immediate or broad serious risk"}[task]
    instruction = profile.get("query_instruction_template", "{prompt}").replace("{prompt}", prompt).replace("{kind}", kind)
    raw = f"Decision: {prompt}\nState: {state}"
    query = profile["query_template"].replace("{instruction}", instruction).replace("{text}", raw)
    return query, state


def qwen_cost(n, c):
    h, i, l = c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"]
    qh, kvh = c["num_attention_heads"], c["num_key_value_heads"]
    d = h // qh
    # MACs: Q/K/V and output projections + gated MLP; causal QK and AV.
    linear = l * n * (2*h*(h + kvh*d) + 3*h*i)
    pairs = n*(n+1)//2
    attention = l * pairs * 2*h
    return {"linear_gflops": 2*linear/1e9, "attention_gflops": 2*attention/1e9,
            "total_gflops": 2*(linear+attention)/1e9}


def main():
    cases = jsonl(FIXTURE)
    assert len(cases) == 300
    # Ensure exact shared fixture provenance.
    lm = json.loads((LAYA / "run-metadata.json").read_text())
    pm = json.loads((PROBES / "run-metadata.json").read_text())
    assert lm["fixture_sha256"] == pm["fixture_sha256"]
    refs = {x["id"]: x["reference"] for x in cases}
    models = {}
    for name, file in (("Laya", LAYA / "laya-mlx-predictions.jsonl"),
                       ("Qwen", PROBES / "qwen-predictions-pooled300.jsonl"),
                       ("Harrier", PROBES / "harrier-predictions-pooled300.jsonl"),
                       ("JEV", PROBES / "jev-predictions-pooled300.jsonl"),
                       ("probe-Qwen", PROBES / "probe-qwen-predictions-all300.jsonl"),
                       ("probe-Harrier", PROBES / "probe-harrier-predictions-all300.jsonl")):
        rows = jsonl(file)
        if name.startswith("probe-"):
            rows = [{"id": r["id"], "owner": r["owner"]["label"],
                     "urgent": str(r["urgent"]["label"]).lower() == "true", "impact": r["impact"]["label"]} for r in rows]
        preds_byid = {p["id"]: p for p in rows}
        models[name] = acc(cases, rows)

    laya_raw = jsonl(LAYA / "raw-results.jsonl")
    laya_lat = stats([r["case_inference_ms_sum_of_three_calls"] for r in laya_raw])
    laya_perq = {t: stats([r["decisions"][t]["inference_ms"] for r in laya_raw]) for t in TASKS}
    probe_raw = jsonl(PROBES / "raw-new210.jsonl")
    probe_latency = {}
    for system in ("qwen", "harrier", "jev"):
        probe_latency[system] = stats([r[system]["timing_ms"] for r in probe_raw], nearest_rank=True)

    # Laya: actual tokenized input_ids stored for all 900 individual forwards.
    laya_tokens = {r["case_id"]: {t: int(r["decisions"][t]["input_tokens"]) for t in TASKS} for r in laya_raw}
    enc = json.loads((Path("/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/laya-root/checkpoint/encoder/config.json")).read_text())
    # Per layer dense encoder MACs = 4H^2 + 3HI. Effective pair count is global N^2,
    # or local |i-j| <= floor(local_attention/2), matching pinned MLX mask.
    h, inter, layers = enc["hidden_size"], enc["intermediate_size"], enc["num_hidden_layers"]
    local = enc["local_attention"] // 2
    layer_types = enc["layer_types"]
    def laya_cost(n, include_head=True):
        dense = layers * n * (4*h*h + 3*h*inter)
        pairs = 0
        for typ in layer_types:
            pairs += n*n if typ == "full_attention" else sum(min(n-1, i+local)-max(0, i-local)+1 for i in range(n))
        enc_att = pairs * 2*h
        # Laya decision head: 2 full-sequence transformer layers, FF width 4H,
        # plus scorer HxH and the tiny act MLP. MAC estimate; ignores norms/activations.
        head = 2*n*(4*h*h + 8*h*h) + 4*n*n*h + n*h*h + n*(h+4)*256 + 256*2
        total = dense + enc_att + (head if include_head else 0)
        return 2*total/1e9
    layaflops = [sum(laya_cost(laya_tokens[c["id"]][t]) for t in TASKS) for c in cases]

    profs = {"Qwen": json.loads((ROOT / "models/manifests/qwen3-embedding-0.6b.json").read_text()),
             "Harrier": json.loads((ROOT / "models/manifests/harrier-oss-v1-0.6b.json").read_text())}
    flops = {}
    token_detail = {}
    for name, modeldir in (("Qwen", "Qwen--Qwen3-Embedding-0.6B"), ("Harrier", "microsoft--harrier-oss-v1-0.6b")):
        base = STORE / modeldir
        tok = Tokenizer.from_file(str(base / "tokenizer.json"))
        config = json.loads((base / "config.json").read_text())
        qlens, slens = {}, {}
        query_costs, four_costs = [], []
        for c in cases:
            qtexts = [compiled_text(c, t, profs[name])[0] for t in TASKS]
            st = compiled_text(c, "owner", profs[name])[1]
            lens = [len(tok.encode(text).ids) for text in qtexts]
            slen = len(tok.encode(st).ids)
            qlens[c["id"]], slens[c["id"]] = dict(zip(TASKS, lens)), slen
            query_costs.append(sum(qwen_cost(n, config)["total_gflops"] for n in lens))
            four_costs.append(query_costs[-1] + qwen_cost(slen, config)["total_gflops"])
        flops[name] = {"three_task_queries": {"mean_gflops": statistics.mean(query_costs), "p50_gflops": statistics.median(query_costs)},
                       "queries_plus_state_fourth_embedding": {"mean_gflops": statistics.mean(four_costs), "p50_gflops": statistics.median(four_costs)}}
        token_detail[name] = {"three_query_tokens_total_mean": statistics.mean(sum(qlens[c["id"]].values()) for c in cases),
                              "state_tokens_mean": statistics.mean(slens.values()),
                              "three_query_tokens_per_task_mean": {t: statistics.mean(qlens[c["id"]][t] for c in cases) for t in TASKS}}

    out = {
      "case_set": {"count": 300, "ids": "test-001..test-300", "fixture_sha256": lm["fixture_sha256"], "identical_fixture_in_both_runs": True},
      "accuracy": models,
      "timing_ms": {"Laya_three_separate_question_calls_per_case": laya_lat, "Laya_per_question": laya_perq,
                    "probe_comparison_new210_only": probe_latency,
                    "sources": {"Laya timings": "results/laya/mlx/runs/20260923T190000Z/raw-results.jsonl: decisions.{owner,urgent,impact}.inference_ms and case_inference_ms_sum_of_three_calls",
                                "Qwen/Harrier/JEV": "results/probes/comparisons/20260923T104208Z/raw-new210.jsonl: {qwen,harrier,jev}.timing_ms; only test-091..test-300; 210 rows"}},
      "estimated_compute": {"convention": "1 MAC=2 FLOPs; GFLOPs are arithmetic estimates, no sparsity/quantization discount; excludes softmax/norm/nonlinear/reductions unless explicitly head affine ops; not measured hardware FLOPs.",
        "Laya": {"mean_gflops_per_three_question_case": statistics.mean(layaflops), "p50_gflops_per_three_question_case": statistics.median(layaflops),
          "source_tokens": "raw-results.jsonl decisions.*.input_tokens (actual tokenizer length), 3 independent calls/case",
          "config": {"hidden_size": h, "intermediate_size": inter, "layers": layers, "layer_types": layer_types, "local_attention": enc["local_attention"], "global_local_effective_pair_formula": "full layer N^2 pairs; sliding layer sum_i count(j: |i-j|<=local_attention/2); 2H FLOPs per pair"},
          "components_mean_gflops": {"modernbert_dense_projections": statistics.mean([sum(2*(layers*n*(4*h*h+3*h*inter))/1e9 for n in laya_tokens[c["id"]].values()) for c in cases]),
            "encoder_attention": statistics.mean([sum((2*sum((n*n if typ=='full_attention' else sum(min(n-1,i+local)-max(0,i-local)+1 for i in range(n))) for typ in layer_types)*h)/1e9 for n in laya_tokens[c["id"]].values()) for c in cases]),
            "decision_head_including_attention_and_output_head": statistics.mean([sum(laya_cost(laya_tokens[c["id"]][t])-laya_cost(laya_tokens[c["id"]][t],False) for t in TASKS) for c in cases])},
          "limitation": "Laya layer types/window from pinned encoder config; output head estimate uses two full-attention transformer blocks, scorer and act affine layers; does not pretend parameter-count-only dense lower bound is full cost."},
        "Qwen_Harrier": {"per_case": flops, "compiled_token_lengths": token_detail,
          "source": "fixture request.state/questions compiled with crates/decengine-compiler/src/lib.rs template logic and model manifests; tokenized using corresponding local tokenizer.json; full model config.json in model store.",
          "config_basis": "hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads and intermediate_size; full causal attention; gated MLP.",
          "limitation": "Three query embeddings represent task-specific probe query compute. Fourth state embedding counted separately only as optional/local-cosine context; probe classifier itself consumes only three task embeddings."}},
      "caveats": ["Probe comparator makes fresh local Qwen/Harrier (3 compiled query embeddings and local cosine baseline/context) and remote JEV HTTP calls only for 210 new cases; archived first 90 predictions have no fresh timing. Thus probe latency n=210 and cannot be directly matched to 300-case Laya latency.",
                   "Laya latency sums three separately timed inference forwards per case, excludes preprocessing; probe timing surrounds each full decision or remote request and is not comparable hardware/mode/work definition.",
                   "Accuracy counts recomputed against shared 300-case fixture references; includes probes plus local baseline Qwen/Harrier and JEV."]}
    out["summary_table"] = [
      {"method": "Laya MLX", "per_question_accuracy_mean": models["Laya"]["question_mean_accuracy"], "complete_case_accuracy": models["Laya"]["complete_case"]["accuracy"],
       "median_latency_ms": laya_lat["p50_ms"], "latency_n": 300, "estimated_gflops_per_case": statistics.mean(layaflops), "latency_scope": "3 separate question calls"},
      *[{"method": f"{name} local cosine", "per_question_accuracy_mean": models[name]["question_mean_accuracy"], "complete_case_accuracy": models[name]["complete_case"]["accuracy"],
         "median_latency_ms": probe_latency[name.lower()]["p50_ms"], "latency_n": 210,
         "estimated_gflops_per_case": flops[name]["three_task_queries"]["mean_gflops"], "latency_scope": "fresh test-091..300; embedding + local cosine/context"}
        for name in ("Qwen", "Harrier")],
      {"method": "JEV HTTP", "per_question_accuracy_mean": models["JEV"]["question_mean_accuracy"], "complete_case_accuracy": models["JEV"]["complete_case"]["accuracy"],
       "median_latency_ms": probe_latency["jev"]["p50_ms"], "latency_n": 210, "estimated_gflops_per_case": None, "latency_scope": "fresh remote HTTP test-091..300; compute not observable"},
      *[{"method": f"{name} frozen probe", "per_question_accuracy_mean": models[f"probe-{name}"]["question_mean_accuracy"],
         "complete_case_accuracy": models[f"probe-{name}"]["complete_case"]["accuracy"], "median_latency_ms": None, "latency_n": 0,
         "estimated_gflops_per_case": flops[name]["three_task_queries"]["mean_gflops"], "latency_scope": "no per-case predictor latency archived; encoder cost shown for feature generation only"}
        for name in ("Qwen", "Harrier")]
    ]
    (LAYA / "compute-comparison.json").write_text(json.dumps(out, indent=2, allow_nan=False)+"\n")
    print(LAYA / "compute-comparison.json")

if __name__ == "__main__":
    main()
