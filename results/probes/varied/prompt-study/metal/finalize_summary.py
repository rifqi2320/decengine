#!/usr/bin/env python3
"""Join shared-head and per-variant-head all-MLX parity reports without reading labels."""
import hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[5]
RUN=ROOT/"results/probes/varied/prompt-study/metal/runs/fresh60"
shared=json.loads((RUN/"shared-v2-baseline/parity-timing-report.json").read_text())
pervar=json.loads((RUN/"per-variant-heads/PARITY_TIMING_REPORT.json").read_text())
pre=json.loads((ROOT/"results/probes/varied/prompt-study/runs/holdout-fresh60-prompt-study-20260924T000000Z/PRE_REFERENCE_FREEZE.json").read_text())
shared_exact=all(
 all(x["all_three_cases_exact"]==60 and all(t["selected_exact"] and t["label_boolean_exact"] and (t["score_argmax_exact"] is not False) for t in x["typed_parity"].values()) for x in variants.values())
 for variants in shared["models"].values())
per_variant_exact=all(
 all(x["all_three_cases_exact"]==60 and all(t["selected_exact"] and t["label_or_boolean_exact"] and (t["score_argmax_exact"] is not False) for t in x["per_type"].values()) for x in variants.values())
 for variants in pervar["models"].values())
summary={"holdout_sha256":shared["fixture_sha256"],"holdout_question_count":180,"holdout_case_count":60,
 "backend":"MLX Device(gpu, 0) / Metal; no host embedding transfer; only final per-question probabilities are read back",
 "model_store":"/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/decengine-model-store",
 "target_access":"none; both analyzers only read pre-frozen prediction outputs and run metadata/hashes",
 "runs_sequential":True,"variant_protocols":{
 "shared-v2-baseline":{"description":"same exact train-frozen baseline weights, means, scales, C reused for all four prompt-only texts","report":"shared-v2-baseline/parity-timing-report.json","models":shared["models"]},
  "per-variant":{"description":"each variant's own full-train frozen train-only scorer head/C; separate CPU reference protocol","report":"per-variant-heads/PARITY_TIMING_REPORT.json","models":pervar["models"]}},
 "pre_reference_prediction_hashes":pre["prediction_hashes"],
 "all_shared_variant_model_predictions_exact":shared_exact,
 "all_per_variant_model_predictions_exact":per_variant_exact,
 "scorer_discrepancy_resolution":"Initial per-variant candidate-head bundles were not like-for-like with the primary shared-v2 CPU reference. The first GPU protocol used the exact shared baseline arrays. The secondary per-variant protocol separately used original bundle hashes confirmed by its handoff; no test-time fitting or weight changes.",
 "preflight_rejections":"A stale shared bundle fixture hash and two legacy per-variant wrapper metadata gaps were rejected before MLX model load. Bundles were regenerated/wrapped from the same source arrays with exact fixture/checkpoint/profile provenance; logs are retained under each protocol's preflight-failures directory. These attempts are excluded from all timing/resource summaries.",
 "run_order":{"shared-v2-baseline":(RUN/"shared-v2-baseline/run-order.txt").read_text(),"per-variant":(RUN/"per-variant-heads/run-order.txt").read_text()},
 "report_sha256":{"shared":hashlib.sha256((RUN/"shared-v2-baseline/parity-timing-report.json").read_bytes()).hexdigest(),
    "per_variant":hashlib.sha256((RUN/"per-variant-heads/PARITY_TIMING_REPORT.json").read_bytes()).hexdigest()}}
(RUN/"FINAL_SUMMARY.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
lines=["# Prompt-study fresh60 all-MLX results","",
"The fresh60 fixture remained unchanged. Both protocols ran as separate sequential, isolated MLX/Metal processes using the explicit model store; no holdout labels or targets were read. Raw GPU predictions were persisted before prediction-only CPU-reference comparison.","",
"## Protocol A — shared frozen v2 head for every text variant", "",
"CPU reference protocol: unchanged Qwen C=1.0 / Harrier C=0.1 train-frozen weights, means, and scales applied to all four prompt texts. Every variant/model has exact selected-label, boolean, and score-argmax parity on all 180 typed predictions and all 60 three-question cases.","",
"| Model | Variant | max probability abs diff | max score EV abs diff | case p50/p95 ms | scorer p50/p95 ms/question | query+candidate tokens/case p50/p95 |", "|---|---|---:|---:|---:|---:|---:|"]
for mk,variants in shared["models"].items():
 for v,x in variants.items():
  c=x["timing_ms"]["three_question_case_total"];s=x["timing_ms"]["gpu_scorer_per_question"];t=x["tokens"]["query_plus_candidate_per_three_question_case"]
  lines.append(f"| {mk} | {v} | {x['max_probability_abs_diff']:.9g} | {x['max_score_expected_value_abs_diff']:.9g} | {c['p50']:.2f}/{c['p95']:.2f} | {s['p50']:.3f}/{s['p95']:.3f} | {t['p50']:.0f}/{t['p95']:.0f} |")
lines += ["", "## Protocol B — own frozen train-only head for each variant", "",
"Compared against the distinct secondary CPU predictions with per-variant heads; original head bundle hashes/C/array fingerprints are verified in `secondary/variant-heads/gpu-variant-head-handoff.json`. No heldout fit/tuning. Each model/variant matches all 180 labels/booleans/score argmaxes and all 60 three-question cases.", "",
"| Model | Variant | C | max probability abs diff | max score EV abs diff | case p50/p95 ms | scorer p50/p95 ms/question | query+candidate tokens/case p50/p95 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
for mk,variants in pervar["models"].items():
 for v,x in variants.items():
  c=x["timing_ms"]["case_total"];s=x["timing_ms"]["gpu_scorer_per_question"];t=x["tokens"]["query_plus_candidate_per_case"]
  lines.append(f"| {mk} | {v} | {x['C']:g} | {x['max_probability_abs_diff']:.9g} | {x['max_score_expected_value_abs_diff']:.9g} | {c['p50']:.2f}/{c['p95']:.2f} | {s['p50']:.3f}/{s['p95']:.3f} | {t['p50']:.0f}/{t['p95']:.0f} |")
lines += ["", "Both case totals include synchronized query/candidate embedding, pair-feature construction, frozen scorer, and softmax over each case's three questions. Token totals and dense embedding FLOP estimates, process-level load/warmup footprint, conditions, model/profile/bundle/CPU-output hashes, and detailed stage percentiles are in the two JSON reports. The FLOP figure is the analytical `2*600M*tokens` estimate, not a measured device counter.", "", "Several metadata-only preflight launches were correctly rejected before the engine/model load; they are retained but excluded from GPU timing. Full separate reports: `shared-v2-baseline/PARITY_TIMING_REPORT.md` and `per-variant-heads/PARITY_TIMING_REPORT.md`. `HEAD_PROTOCOLS.md` records why the two head protocols must not be mixed."]
(RUN/"FINAL_SUMMARY.md").write_text("\n".join(lines)+"\n")
print((RUN/"FINAL_SUMMARY.md").read_text())
