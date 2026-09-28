#!/usr/bin/env python3
"""Evaluate existing, pre-holdout, train-only per-variant heads; no fitting here."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,sys,time
from datetime import datetime,timezone
from pathlib import Path
from typing import Any

import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
TRAIN_RUN=HERE/"runs/trainonly-clean300-20260924T002000Z"
PRIMARY=HERE/"runs/holdout-fresh60-prompt-study-20260924T000000Z"
FIXTURE=ROOT/"benchmarks/cases/varied/multi/test-prompt-fresh-60.jsonl"
HEAD_DIR=HERE/"metal/frozen-scorers"
MODELS={"qwen":"Qwen/Qwen3-Embedding-0.6B","harrier":"microsoft/harrier-oss-v1-0.6b"}
VARIANTS=("v2-baseline","readable-state","rule-in-candidate","joint-context-candidate")
TASKS=("choice","noul","score")
EXPECTED_FIXTURE_SHA="316b3c2e1d0c5cf50356d0e236a5f4adf27fdac9ee6d6c6b56a14f5354ef6a9f"

sys.path.insert(0,str(HERE))
import evaluate_prompt_fresh as common  # noqa: E402

def sha(p):return common.sha(Path(p))
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def dump(p,value):common.dump(Path(p),value)
def jsonl(p):return common.read_jsonl(Path(p))

def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def variant_head(model:str,variant:str,metrics_mtime:float)->tuple[dict[str,Any],dict[str,Any]]:
    headpath=HEAD_DIR/model/f"{variant}.json";sidecar=HEAD_DIR/model/f"{variant}.sha256"
    if not headpath.is_file() or not sidecar.is_file():raise FileNotFoundError(f"missing pre-frozen variant scorer {model}/{variant}")
    expected=sidecar.read_text(encoding="utf-8").split()[0]
    if sha(headpath)!=expected:raise ValueError(f"scorer bundle checksum mismatch {model}/{variant}")
    if headpath.stat().st_mtime>=metrics_mtime or sidecar.stat().st_mtime>=metrics_mtime:
        raise ValueError(f"scorer bundle was not frozen before first test-reference scoring: {model}/{variant}")
    doc=load(headpath);cv_cfg=load(TRAIN_RUN/"cv"/model/variant/"frozen-config.json")
    expected_c=float(cv_cfg["selected"]["C"])
    if doc.get("model")!=MODELS[model] or doc.get("model_key")!=model or doc.get("variant")!=variant or float(doc["C"])!=expected_c:
        raise ValueError(f"scorer bundle identity/C mismatch {model}/{variant}")
    if doc.get("training_scope")!="clean300 + multi120 only; targets read only from those train fixtures":raise ValueError("unexpected scorer training scope")
    provenance=doc["provenance"]
    expected_hashes={"models_sha256":sha(TRAIN_RUN/"frozen"/model/"models.json"),
      "selection_sha256":sha(TRAIN_RUN/"frozen"/model/"selection.json"),
      "variant_cv_config_sha256":sha(TRAIN_RUN/"cv"/model/variant/"frozen-config.json"),
      "train_case_sha256":sha(HERE/"data/train-clean-300.jsonl"),
      "multi_train_case_sha256":sha(ROOT/"benchmarks/cases/varied/multi/train-120.jsonl"),
      "single_features_sha256":sha(TRAIN_RUN/"features"/model/variant/"single.jsonl"),
      "single_manifest_sha256":sha(TRAIN_RUN/"features"/model/variant/"single.jsonl.manifest.json"),
      "multi_features_sha256":sha(TRAIN_RUN/"features"/model/variant/"multi.jsonl"),
      "multi_manifest_sha256":sha(TRAIN_RUN/"features"/model/variant/"multi.jsonl.manifest.json")}
    for k,v in expected_hashes.items():
        if provenance.get(k)!=v:raise ValueError(f"variant scorer train provenance mismatch {model}/{variant}/{k}")
    if len(doc.get("scorers",{}))!=3 or set(doc["scorers"])!={"choice","noul","score"}:raise ValueError("variant scorer bundle missing typed heads")
    norm={}
    for kind,scorer in doc["scorers"].items():
        if not scorer.get("fit_diagnostics",{}).get("converged"):raise ValueError(f"nonconverged frozen head {model}/{variant}/{kind}")
        if not all(np.isfinite(np.asarray(scorer[k],dtype=np.float64)).all() for k in ("weights","mean","scale")):raise ValueError("nonfinite frozen head")
        if len(scorer["weights"])!=4096 or len(scorer["mean"])!=4096 or len(scorer["scale"])!=4096:raise ValueError("frozen head vector dimension mismatch")
        if any(float(x)<=0 for x in scorer["scale"]):raise ValueError("invalid frozen scaler")
        norm[kind]={"mean_sha256":canonical_sha(scorer["mean"]),"scale_sha256":canonical_sha(scorer["scale"]),"weights_sha256":canonical_sha(scorer["weights"]),
                    "dimension":len(scorer["weights"]),"fit_diagnostics":scorer["fit_diagnostics"]}
    record={"head_path":str(headpath),"sha256":expected,"sidecar_path":str(sidecar),"C":float(doc["C"]),
        "mtime_utc":datetime.fromtimestamp(headpath.stat().st_mtime,timezone.utc).isoformat(),
        "sidecar_mtime_utc":datetime.fromtimestamp(sidecar.stat().st_mtime,timezone.utc).isoformat(),
        "precedes_first_reference_score":True,"baseline_bit_exact_to_selected_frozen_head":doc["baseline_bit_exact_to_published_frozen_models"],
        "provenance":provenance,"normalization_and_weights":norm}
    return doc,record

def prediction_stage(run:Path)->None:
    freeze=load(PRIMARY/"PRE_REFERENCE_FREEZE.json");primary_meta=load(PRIMARY/"run-metadata.json")
    if freeze["status"]!="predictions_frozen_before_reference_scoring" or sha(FIXTURE)!=EXPECTED_FIXTURE_SHA:raise ValueError("primary prediction freeze or fixture hash invalid")
    metric_mtime=(PRIMARY/"holdout-metrics.json").stat().st_mtime
    # Frozen scorer files and sidecars are verified to predate the first reference scoring timestamp.
    docs={};heads={}
    for m in MODELS:
        docs[m]={};heads[m]={}
        for v in VARIANTS:docs[m][v],heads[m][v]=variant_head(m,v,metric_mtime)
    if run.exists():raise FileExistsError(f"refusing overwrite: {run}")
    run.mkdir(parents=True)
    cases=common.read_prediction_safe(FIXTURE)
    ids=[str(c["id"]) for c in cases]
    if len(ids)!=60 or ids!=[f"mq-fresh-{i:03d}" for i in range(1,61)]:raise ValueError("fresh60 case ID order mismatch")
    summary={"status":"secondary per-variant heads frozen before secondary target scoring","scope":"pre-existing train-only scorer bundles only; no refit",
        "fixture_sha256":EXPECTED_FIXTURE_SHA,"primary_pre_reference_freeze_sha256":sha(PRIMARY/"PRE_REFERENCE_FREEZE.json"),
        "primary_prompt_only_report_sha256":sha(PRIMARY/"REPORT.md"),"first_reference_score_file_mtime_utc":datetime.fromtimestamp(metric_mtime,timezone.utc).isoformat(),
        "models":{},"prediction_paths":{},"prediction_hashes":{}}
    allfiles=[]
    for m,model_id in MODELS.items():
        summary["models"][m]={"model_id":model_id,"variants":{}}
        for v in VARIANTS:
            feature_path=PRIMARY/"features"/m/v/"fresh60.features.jsonl";feature_manifest=load(Path(str(feature_path)+".manifest.json"))
            features=jsonl(feature_path)
            if len(features)!=180:raise ValueError(f"feature count mismatch {m}/{v}")
            feat_map={(str(x["id"]),str(x["question_key"])):x for x in features}
            if len(feat_map)!=180:raise ValueError("duplicate feature keys")
            outputs=[];checks={"permutation_checked":0,"noul_polarity_checked":0}
            for case in cases:
                decisions={}
                for qname,qwire in case["request"]["questions"].items():
                    x=feat_map[str(case["id"]),str(qname)]
                    pred,c=common.score_question(x,docs[m][v]["scorers"]);decisions[str(qname)]=pred
                    for k,n in c.items():checks[k]+=n
                outputs.append({"id":str(case["id"]),"model":model_id,"variant":v,"scorer_head_sha256":heads[m][v]["sha256"],"decisions":decisions})
            path=run/"predictions"/m/f"{v}.jsonl";common.write_jsonl(path,outputs)
            if len(jsonl(path))!=60:raise ValueError("secondary predictions not complete")
            summary["models"][m]["variants"][v]={**heads[m][v],"feature_sha256":sha(feature_path),"feature_manifest_sha256":sha(Path(str(feature_path)+".manifest.json")),
                "embedding_calls_reused":feature_manifest["embedding_call_count"],"query_plus_candidate_tokens_reused":sum(int(x["query_token_length"])+sum(int(z["token_length"]) for z in x["candidates"]) for x in features),
                "permutation_checked":checks["permutation_checked"],"noul_polarity_checked":checks["noul_polarity_checked"],"prediction_path":str(path),"predictions_sha256":sha(path),"prediction_count":60}
            summary["prediction_paths"][f"{m}/{v}"]=str(path);summary["prediction_hashes"][f"{m}/{v}"]=sha(path);allfiles.append(path)
    marker={"status":"variant-head predictions frozen before secondary reference scoring","fixture_sha256":EXPECTED_FIXTURE_SHA,
        "primary_freeze_sha256":sha(PRIMARY/"PRE_REFERENCE_FREEZE.json"),"predictions":summary["prediction_hashes"],
        "scorer_bundle_hashes":{f"{m}/{v}":heads[m][v]["sha256"] for m in MODELS for v in VARIANTS},
        "all_variant_model_predictions_written":len(allfiles)==8,"reference_scoring":"not started"}
    dump(run/"variant-head-freeze.json",marker);dump(run/"variant-head-run-metadata.json",summary)
    print(json.dumps({"status":"secondary predictions frozen; no reference scoring run yet","prediction_files":len(allfiles),"run":str(run)},indent=2))

def target(ref,kind):
    t=ref["target"]
    if kind=="choice":return str(t["selected"])
    if kind=="noul":
        if type(t["value"]) is not bool:raise ValueError("noul reference target must be bool")
        return t["value"]
    return str(t["label"])

def score_stage(run:Path)->None:
    freeze=load(run/"variant-head-freeze.json");meta=load(run/"variant-head-run-metadata.json")
    if freeze.get("status")!="variant-head predictions frozen before secondary reference scoring" or not freeze.get("all_variant_model_predictions_written"):
        raise ValueError("secondary predictions not frozen")
    if sha(FIXTURE)!=EXPECTED_FIXTURE_SHA or sha(PRIMARY/"PRE_REFERENCE_FREEZE.json")!=freeze["primary_freeze_sha256"]:raise ValueError("primary artifact/fixture changed")
    preds={}
    for m in MODELS:
        for v in VARIANTS:
            p=Path(meta["prediction_paths"][f"{m}/{v}"])
            if sha(p)!=freeze["predictions"][f"{m}/{v}"]:raise ValueError(f"secondary prediction hash mismatch {m}/{v}")
            rows=jsonl(p)
            if len(rows)!=60:raise ValueError("secondary predictions incomplete")
            preds[m,v]={str(x["id"]):x for x in rows}
    # Only now, after the secondary prediction freeze, are references opened again.
    cases=jsonl(FIXTURE)
    primary_preds={(m,v):{str(x["id"]):x for x in jsonl(PRIMARY/"predictions"/m/f"{v}.jsonl")} for m in MODELS for v in VARIANTS}
    metrics={"fixture_sha256":EXPECTED_FIXTURE_SHA,"scorer_policy":"pre-existing full-train per-variant typed heads from metal/frozen-scorers; no fitting/tuning in this stage",
        "primary_run":"../holdout-fresh60-prompt-study-20260924T000000Z","models":{}}
    for m in MODELS:
        metrics["models"][m]={"variants":{},"paired_head_vs_fixed_v2_prompt_only":{}}
        for v in VARIANTS:
            types={k:{"n":0,"correct":0,"local_f1_sum":0.0,"score_mae":0.0,"score_nmae":0.0,"score_n":0} for k in TASKS}
            all3={};groups={"family":{},"domain":{}}
            for case in cases:
                cid=str(case["id"]);ref_case=case["reference"];questions=case["request"]["questions"];complete=True
                for qname,qwire in questions.items():
                    kind=str(qwire["type"]);ref=ref_case[qname];gold=target(ref,kind)
                    d=preds[m,v][cid]["decisions"][qname];hit=d["label"]==gold;complete &= hit
                    item=types[kind];item["n"]+=1;item["correct"]+=int(hit)
                    candidates=qwire.get("options",{}) if kind=="choice" else (qwire["levels"] if kind=="score" else {"supported":0,"unsupported":0})
                    item["local_f1_sum"]+=(1/len(candidates) if hit else 0)
                    if kind=="score":
                        level=next((x for x in qwire["levels"] if str(x["label"])==gold),None)
                        if level is None:raise ValueError("gold rubric label missing")
                        err=abs(float(d["expected_value"])-float(level["value"]))
                        span=max(float(x["value"]) for x in qwire["levels"])-min(float(x["value"]) for x in qwire["levels"])
                        item["score_mae"]+=err;item["score_nmae"]+=err/span if span>0 else 0;item["score_n"]+=1
                    fam=str(ref.get("question_family_id",case.get("question_family_id","unknown")))
                    metadata=case.get("metadata",{});metadata=metadata if isinstance(metadata,dict) else {}
                    dom=str(case.get("domain",metadata.get("domain","unknown")))
                    for gt,gid in (("family",fam),("domain",dom)):
                        g=groups[gt].setdefault(gid,{"n":0,"correct":0,"per_type":{}});g["n"]+=1;g["correct"]+=int(hit)
                        t=g["per_type"].setdefault(kind,{"n":0,"correct":0});t["n"]+=1;t["correct"]+=int(hit)
                all3[cid]=complete
            per_type={}
            for k,x in types.items():
                per_type[k]={"n":x["n"],"correct":x["correct"],"accuracy":x["correct"]/x["n"],"wilson_95_ci":common.wilson(x["correct"],x["n"]),
                    "candidate_local_macro_f1":x["local_f1_sum"]/x["n"]}
                if k=="score":per_type[k].update({"expected_value_mae":x["score_mae"]/x["score_n"],"normalized_expected_value_mae":x["score_nmae"]/x["score_n"]})
            def outgroups(groups):
                return {gid:{"n":g["n"],"correct":g["correct"],"accuracy":g["correct"]/g["n"],"per_type":{k:{"n":x["n"],"correct":x["correct"],"accuracy":x["correct"]/x["n"]} for k,x in g["per_type"].items()}} for gid,g in sorted(groups.items())}
            existing=primary_preds[m,v];new=preds[m,v];paired={}
            for k in TASKS:
                b=w=n=0
                for case in cases:
                    qmap=case["request"]["questions"]
                    for qname,qwire in qmap.items():
                        if qwire["type"]!=k:continue
                        gold=target(case["reference"][qname],k)
                        base=existing[str(case["id"])]["decisions"][qname]["label"]==gold
                        alt=new[str(case["id"])]["decisions"][qname]["label"]==gold
                        if not base and alt:b+=1
                        elif base and not alt:w+=1
                        n+=1
                paired[k]={"n":n,"variant_head_correct_only":b,"fixed_v2_head_correct_only":w,
                    "mcnemar_exact_two_sided_p_descriptive":common.exact_mcnemar(b,w),"paired_accuracy_delta":(b-w)/n,
                    "paired_accuracy_delta_normal_95_ci_descriptive":common.paired_delta_ci(b,w,n)}
            base_metrics=json.loads((PRIMARY/"holdout-metrics.json").read_text(encoding="utf-8"))["models"][m]["variants"][v]
            balanced=sum(x["accuracy"] for x in per_type.values())/3;macro_f1=sum(x["candidate_local_macro_f1"] for x in per_type.values())/3
            metrics["models"][m]["variants"][v]={"scorer_head_sha256":meta["models"][m]["variants"][v]["sha256"],"C":meta["models"][m]["variants"][v]["C"],
                "per_type":per_type,"type_balanced_accuracy":balanced,"type_macro_candidate_local_f1":macro_f1,
                "all_three_correct":{"correct":sum(all3.values()),"n":len(all3),"accuracy":sum(all3.values())/len(all3),
                                     "wilson_95_ci":common.wilson(sum(all3.values()),len(all3))},
                "per_family":outgroups(groups["family"]),"per_domain":outgroups(groups["domain"]),
                "prompt_only_fixed_v2_head_comparison":{"type_balanced_accuracy":base_metrics["type_balanced_accuracy"],
                    "type_macro_candidate_local_f1":base_metrics["type_macro_candidate_local_f1"],
                    "all_three_correct":base_metrics["all_three_correct"]}}
            metrics["models"][m]["paired_head_vs_fixed_v2_prompt_only"][v]=paired
    metrics["test_used_for_fit_or_selection"]=False
    metrics["secondary_vs_primary_warning"]="This is a secondary comparison on the same already-seen fresh60 test. It is not an independent test and was not used to alter the train-only selection."
    dump(run/"secondary-head-metrics.json",metrics);write_secondary_report(run,metrics,meta)
    print(json.dumps({m:{v:{"type_balanced_accuracy":x["type_balanced_accuracy"],"all_three_correct":x["all_three_correct"]} for v,x in d["variants"].items()} for m,d in metrics["models"].items()},indent=2))

def write_secondary_report(run,metrics,meta):
    lines=["# Secondary fresh60 evaluation: per-variant frozen heads","",
        "This is a secondary comparison on the **same already-seen fresh60 holdout**, not an independent test. The scorer bundles were already frozen from clean300+multi120 train-only features before the first reference scoring. This stage performed no fitting, tuning, or test-based selection. Embeddings/features were reused; only the scorer head changed. CPU NumPy scoring is the reference; this report is not all-GPU.","",
        "Predictions were written to `predictions/{model}/{variant}.jsonl` and frozen in `variant-head-freeze.json` before this metric stage read references. Full model/head/scaler hashes and per-family/per-domain results are in `variant-head-run-metadata.json` and `secondary-head-metrics.json`.","",
        "| Model | Variant | C | Head SHA-256 | Type-balanced accuracy | Type-macro local F1 | All-three | Fixed-v2 prompt-only accuracy | Delta |","|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for m,d in metrics["models"].items():
        for v,x in d["variants"].items():
            all3=x["all_three_correct"];base=x["prompt_only_fixed_v2_head_comparison"]["type_balanced_accuracy"]
            lines.append(f"| {m} | {v} | {x['C']:g} | `{x['scorer_head_sha256']}` | {x['type_balanced_accuracy']:.3f} | {x['type_macro_candidate_local_f1']:.3f} | {all3['correct']}/60 | {base:.3f} | {x['type_balanced_accuracy']-base:+.3f} |")
    lines += ["","## Secondary per-type metrics", "", "Wilson intervals are descriptive; score MAE is probability-weighted expected value versus the variable rubric's numeric target (raw / normalized by rubric span)."]
    for m,d in metrics["models"].items():
        lines += ["",f"### {m}","","| Variant | Type | Correct / n | Accuracy | Candidate-local F1 | Expected-value MAE raw / normalized |","|---|---|---:|---:|---:|---:|"]
        for v,x in d["variants"].items():
            for k,t in x["per_type"].items():
                mae=f"{t['expected_value_mae']:.3f} / {t['normalized_expected_value_mae']:.3f}" if k=="score" else "—"
                lines.append(f"| {v} | {k} | {t['correct']}/{t['n']} | {t['accuracy']:.3f} | {t['candidate_local_macro_f1']:.3f} | {mae} |")
    lines += ["","Detailed per-type accuracy/F1, score expected-value MAE, 95% Wilson intervals for all-three cases, per-family/domain accuracy, and paired McNemar/descriptive CIs are in `secondary-head-metrics.json`.","",
        "Embedding token counts, MLX export wall timings, feature hashes, and FLOP estimates are unchanged from `../run-metadata.json`; this secondary pass incurs CPU scorer execution only. The head files and all their mean/scale/weight array hashes, C values, per-variant CV config and train-feature provenance are in `variant-head-run-metadata.json`.","",
        "No test result changes the original v2-baseline train-only selection. In particular, this secondary exercise was prompted by a protocol correction after the prompt-only report and must not be presented as fresh confirmation.",""]
    (run/"SECONDARY-HEAD-REPORT.md").write_text("\n".join(lines),encoding="utf-8")

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--stage",choices=("predict","score"),required=True);p.add_argument("--run",type=Path,required=True);a=p.parse_args();run=a.run.resolve()
    if a.stage=="predict":prediction_stage(run)
    else:score_stage(run)
if __name__=="__main__":main()
