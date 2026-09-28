#!/usr/bin/env python3
"""Two-stage frozen prompt-variant evaluation on fresh60.

Stage ``predict`` exports MLX features and writes every model/variant prediction plus
PRE_REFERENCE_FREEZE.json before any reference field is accessed. Stage ``score`` is a
separate invocation, verifies the freeze hashes, and only then reads reference labels.
The probe is a fixed CPU NumPy application of the training-frozen typed weights; only
the feature export is MLX/Metal. No fitting, tuning, or test-driven selection occurs.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
FIXTURE=ROOT/"benchmarks/cases/varied/multi/test-prompt-fresh-60.jsonl"
EXPECTED_FIXTURE_SHA="316b3c2e1d0c5cf50356d0e236a5f4adf27fdac9ee6d6c6b56a14f5354ef6a9f"
TRAIN_RUN=HERE/"runs/trainonly-clean300-20260924T002000Z"
STORE=Path("/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/decengine-model-store")
BINARY=HERE/"target/release/decengine-prompt-study-exporter"
MODELS={"qwen":"Qwen/Qwen3-Embedding-0.6B","harrier":"microsoft/harrier-oss-v1-0.6b"}
VARIANTS=("v2-baseline","readable-state","rule-in-candidate","joint-context-candidate")
TASKS=("choice","noul","score")
PARAMETERS=600_000_000

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def dump(path:Path,value:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")

def write_jsonl(path:Path,values:list[dict[str,Any]])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False)+"\n" for x in values),encoding="utf-8")

def read_prediction_safe(path:Path)->list[dict[str,Any]]:
    """Parse without retaining any object named `reference` (including nested ones)."""
    def strip_refs(pairs): return {k:v for k,v in pairs if k!="reference"}
    out=[]
    for n,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip(): continue
        try: row=json.loads(line,object_pairs_hook=strip_refs)
        except json.JSONDecodeError as e: raise ValueError(f"{path}:{n}: invalid JSON: {e}") from e
        if "reference" in row: raise AssertionError("reference stripping failed")
        out.append(row)
    return out

def read_jsonl(path:Path)->list[dict[str,Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def wilson(hits:int,n:int)->list[float]|None:
    if n==0:return None
    z=1.959963984540054;p=hits/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,mid-half),min(1.,mid+half)]

def exact_mcnemar(better:int,worse:int)->float:
    n=better+worse
    if n==0:return 1.0
    tail=sum(math.comb(n,k) for k in range(min(better,worse)+1))/(2**n)
    return min(1.,2*tail)

def paired_delta_ci(better:int,worse:int,n:int)->list[float]|None:
    if n==0:return None
    d=(better-worse)/n
    variance=max(0.,((better+worse)/n-d*d)/n)
    half=1.959963984540054*math.sqrt(variance)
    return [max(-1.,d-half),min(1.,d+half)]

def pair_features(query:np.ndarray,candidate:np.ndarray)->np.ndarray:
    return np.concatenate((query,candidate,query*candidate,np.abs(query-candidate)))

def frozen_models()->dict[str,dict[str,Any]]:
    result={}
    frozen=json.loads((TRAIN_RUN/"frozen-config.json").read_text(encoding="utf-8"))
    if sha(TRAIN_RUN/"frozen-config.json")!="305a124523f12e28686bd037d8be96719717ee36cbfc86bfc7b3f8c41b7fd325":
        raise ValueError("frozen training config hash changed")
    for name in MODELS:
        if frozen["models"][name]["selected_variant"]!="v2-baseline":raise ValueError("unexpected train-time selected variant")
        artifact=json.loads((TRAIN_RUN/"frozen"/name/"models.json").read_text(encoding="utf-8"))
        if artifact["variant"]!="v2-baseline":raise ValueError("final fit variant mismatch")
        result[name]=artifact
    return result

def score_question(row:dict[str,Any],scorers:dict[str,Any])->tuple[dict[str,Any],dict[str,int]]:
    kind=str(row["question_type"]); model=scorers[kind]
    q=np.asarray(row["query_embedding"],dtype=np.float64)
    cands=row["candidates"]
    if q.shape!=(1024,) or not np.isfinite(q).all():raise ValueError("invalid query vector")
    ids=[str(c["candidate_id"]) for c in cands]
    if len(ids)<2 or len(set(ids))!=len(ids):raise ValueError("empty/duplicate candidate IDs")
    x=np.stack([pair_features(q,np.asarray(c["embedding"],dtype=np.float64)) for c in cands])
    if x.shape[1]!=len(model["weights"]):raise ValueError("frozen scorer feature width mismatch")
    weights=np.asarray(model["weights"],dtype=np.float64);mean=np.asarray(model["mean"],dtype=np.float64);scale=np.asarray(model["scale"],dtype=np.float64)
    norm=(x-mean)/scale;scores=norm@weights
    reversed_scores=((x[::-1]-mean)/scale)@weights
    if not np.allclose(scores,reversed_scores[::-1],rtol=0,atol=1e-10):raise AssertionError("candidate permutation changed scores")
    shifted=scores-float(scores.max());probs=np.exp(shifted);probs/=probs.sum()
    reverse_probs=np.exp(reversed_scores-float(reversed_scores.max()));reverse_probs/=reverse_probs.sum()
    if not np.allclose(probs,reverse_probs[::-1],rtol=0,atol=1e-12):raise AssertionError("candidate permutation changed probabilities")
    selected_ix=int(np.argmax(probs));selected=ids[selected_ix]
    if kind=="noul":
        if set(ids)!={"supported","unsupported"}:raise ValueError("noul candidates must preserve explicit binary polarity")
        # Reversal invariance by semantic ID verifies that polarity is not candidate-position encoded.
        by_id=dict(zip(ids,scores));reverse_by_id=dict(zip(ids[::-1],reversed_scores))
        if not np.allclose([by_id[k] for k in ids],[reverse_by_id[k] for k in ids],rtol=0,atol=1e-10):raise AssertionError("noul polarity changed under order swap")
        answer=selected=="supported"
        expected_value=float(answer)
    elif kind=="score":
        values={str(c["candidate_id"]):float(c["score_value"]) for c in cands}
        expected_value=float(sum(probs[i]*values[ids[i]] for i in range(len(ids))))
        answer=selected
    elif kind=="choice":
        expected_value=None;answer=selected
    else:raise ValueError(f"unknown question type {kind!r}")
    return ({"question_type":kind,"label":answer,"selected":selected,"probabilities":{ids[i]:float(probs[i]) for i in range(len(ids))},
             "confidence":float(probs[selected_ix]),"expected_value":expected_value},
            {"permutation_checked":1,"noul_polarity_checked":int(kind=="noul")})

def predict_stage(fixture:Path,run:Path,store:Path)->None:
    if not fixture.is_file():raise FileNotFoundError(fixture)
    digest=sha(fixture)
    if digest!=EXPECTED_FIXTURE_SHA:raise ValueError(f"fresh holdout SHA mismatch: {digest}")
    if not BINARY.is_file():raise FileNotFoundError(f"isolated MLX exporter binary missing: {BINARY}")
    cases=read_prediction_safe(fixture)
    ids=[str(c.get("id","")) for c in cases]
    if len(cases)!=60 or ids!=[f"mq-fresh-{i:03d}" for i in range(1,61)]:
        raise ValueError("fresh fixture must be ordered mq-fresh-001..060")
    for c in cases:
        qs=c.get("request",{}).get("questions")
        if not isinstance(qs,dict) or len(qs)!=3:raise ValueError(f"{c['id']}: expected 3 request questions")
        if "reference" in c:raise AssertionError("reference field was not excluded from prediction-stage objects")
    if run.exists():raise FileExistsError(f"refusing to overwrite evaluation run: {run}")
    run.mkdir(parents=True)
    models=frozen_models();metadata={"run_id":run.name,"fixture_sha256":digest,"case_count":60,"question_count":180,
        "scope":"fresh60 prediction stage; references discarded/not accessed","probe":"mixed MLX/Metal embeddings + frozen CPU NumPy scorer",
        "selected_training_variant":"v2-baseline scorer applied unchanged to every prompt text variant",
        "frozen_config_sha256":sha(TRAIN_RUN/"frozen-config.json"),"exporter_binary_sha256":sha(BINARY),"model_store":str(store),
        "models":{},"variants":list(VARIANTS),"started_at":datetime.now(timezone.utc).isoformat()}
    total_permutation=0;total_polarity=0
    for model_key,model_id in MODELS.items():
        metadata["models"][model_key]={"model_id":model_id,"final_fit_sha256":sha(TRAIN_RUN/"frozen"/model_key/"models.json"),"variants":{}}
        for variant in VARIANTS:
            feat=run/"features"/model_key/variant/"fresh60.features.jsonl";feat.parent.mkdir(parents=True,exist_ok=True)
            started=time.perf_counter()
            subprocess.run([str(BINARY),"--home",str(store),"--input",str(fixture),"--output",str(feat),"--model",model_id,"--variant",variant],cwd=ROOT,check=True)
            elapsed=time.perf_counter()-started
            manifest=json.loads(Path(str(feat)+".manifest.json").read_text(encoding="utf-8"))
            if manifest.get("input_sha256")!=digest or manifest.get("model")!=model_id or manifest.get("variant")!=variant:
                raise ValueError(f"exporter manifest mismatch {model_key}/{variant}")
            if manifest.get("output_sha256")!=sha(feat):raise ValueError("exporter feature output hash mismatch")
            if manifest.get("backend")!="MLX/Metal":raise ValueError("embeddings were not exported with MLX/Metal")
            if manifest.get("text_version")!=f"decengine-varied-pair-text-v2-generic-types/{variant}":raise ValueError("variant text version mismatch")
            baseline_manifest=json.loads(Path(str(TRAIN_RUN/"features"/model_key/"v2-baseline"/"single.jsonl")+".manifest.json").read_text(encoding="utf-8"))
            for key in ("profile_version","profile_sha256","tokenizer_sha256","checkpoint_files"):
                if manifest.get(key)!=baseline_manifest.get(key):raise ValueError(f"{key} differs from frozen training profile/checkpoint")
            features=read_jsonl(feat)
            if len(features)!=180:raise ValueError(f"{model_key}/{variant}: expected 180 feature questions, got {len(features)}")
            fkeys={(str(x["id"]),str(x["question_key"])) for x in features}
            request_keys={(str(c["id"]),str(k)) for c in cases for k in c["request"]["questions"]}
            if fkeys!=request_keys or len(fkeys)!=180:raise ValueError("feature question coverage/uniqueness mismatch")
            f_by_id={}
            for x in features:f_by_id.setdefault(str(x["id"]),{})[str(x["question_key"])]=x
            output=[];variant_counts={"permutation_checked":0,"noul_polarity_checked":0}
            for case in cases:
                decisions={}
                for qname,qwire in case["request"]["questions"].items():
                    x=f_by_id[str(case["id"])][str(qname)]
                    if x["question_type"]!=qwire["type"]:raise ValueError("question type changed in export")
                    request_candidates=set(qwire.get("options",{})) if qwire["type"]=="choice" else (
                        {"unsupported","supported"} if qwire["type"]=="noul" else {str(z["label"]) for z in qwire["levels"]})
                    if {str(z["candidate_id"]) for z in x["candidates"]}!=request_candidates:raise ValueError("candidate set mismatch")
                    pred,checks=score_question(x,models[model_key]["scorers"])
                    decisions[str(qname)]=pred
                    for k,v in checks.items():variant_counts[k]+=v
                output.append({"id":str(case["id"]),"model":model_id,"variant":variant,"decisions":decisions})
            predpath=run/"predictions"/model_key/f"{variant}.jsonl";write_jsonl(predpath,output)
            if len(read_jsonl(predpath))!=60:raise ValueError("prediction output count mismatch")
            total_permutation+=variant_counts["permutation_checked"];total_polarity+=variant_counts["noul_polarity_checked"]
            tok=sum(int(x["query_token_length"])+sum(int(c["token_length"]) for c in x["candidates"]) for x in features)
            metadata["models"][model_key]["variants"][variant]={"features_sha256":sha(feat),"manifest_sha256":sha(Path(str(feat)+".manifest.json")),
                "predictions_sha256":sha(predpath),"prediction_count":len(output),"question_count":len(features),
                "embedding_calls":manifest["embedding_call_count"],"query_plus_candidate_tokens":tok,
                "estimated_dense_embedding_flops_2pt":2*PARAMETERS*tok,"measured_export_wall_seconds":elapsed,
                "model_profile_version":manifest["profile_version"],"profile_sha256":manifest["profile_sha256"],
                "text_version":manifest["text_version"],"tokenizer_sha256":manifest["tokenizer_sha256"],
                "checkpoint_files":manifest["checkpoint_files"],**variant_counts}
    metadata["completed_at"]=datetime.now(timezone.utc).isoformat()
    metadata["integrity"]={"prediction_files":16,"total_prediction_rows":480,"permutation_checks":total_permutation,"noul_polarity_checks":total_polarity,
        "all_variants_models_predictions_written":True}
    freeze={"status":"predictions_frozen_before_reference_scoring","fixture_sha256":digest,"frozen_config_sha256":metadata["frozen_config_sha256"],
        "exporter_binary_sha256":metadata["exporter_binary_sha256"],"prediction_hashes":{
            f"{m}/{v}":metadata["models"][m]["variants"][v]["predictions_sha256"] for m in MODELS for v in VARIANTS},
        "feature_hashes":{f"{m}/{v}":metadata["models"][m]["variants"][v]["features_sha256"] for m in MODELS for v in VARIANTS},
        "metric_stage":"not run; no reference values read in prediction stage"}
    dump(run/"run-metadata.json",metadata);dump(run/"PRE_REFERENCE_FREEZE.json",freeze)
    print(json.dumps({"status":"predictions frozen; references still untouched","run":str(run),"prediction_files":16,
                      "rows":480,"permutation_checks":total_permutation,"noul_polarity_checks":total_polarity},indent=2))

def score_stage(run:Path)->None:
    freeze_path=run/"PRE_REFERENCE_FREEZE.json"
    if not freeze_path.is_file():raise FileNotFoundError("prediction freeze marker missing; cannot read references")
    freeze=json.loads(freeze_path.read_text(encoding="utf-8"));meta=json.loads((run/"run-metadata.json").read_text(encoding="utf-8"))
    if freeze.get("status")!="predictions_frozen_before_reference_scoring":raise ValueError("invalid pre-reference freeze status")
    if sha(FIXTURE)!=EXPECTED_FIXTURE_SHA or sha(FIXTURE)!=freeze["fixture_sha256"]:raise ValueError("holdout fixture digest mismatch")
    model_artifacts=frozen_models()
    for model_key in MODELS:
        if sha(TRAIN_RUN/"frozen"/model_key/"models.json")!=meta["models"][model_key]["final_fit_sha256"]:
            raise ValueError(f"frozen final weights changed: {model_key}")
    preds={}
    for m in MODELS:
        for v in VARIANTS:
            p=run/"predictions"/m/f"{v}.jsonl"
            if sha(p)!=freeze["prediction_hashes"][f"{m}/{v}"]:raise ValueError(f"prediction hash changed: {m}/{v}")
            rows=read_jsonl(p)
            if len(rows)!=60:raise ValueError("expected 60 predictions for each model/variant")
            preds[m,v]={str(x["id"]):x for x in rows}
    # Reference access begins only after all eight prediction hashes and the pre-reference marker validate.
    cases=read_jsonl(FIXTURE)
    if len(cases)!=60:raise ValueError("holdout case count changed")
    ids=[str(x["id"]) for x in cases]
    for m in MODELS:
        for v in VARIANTS:
            if list(preds[m,v])!=ids:raise ValueError(f"prediction ID order mismatch {m}/{v}")
    def target(ref:dict[str,Any],kind:str)->Any:
        t=ref["target"]
        if kind=="choice":return str(t["selected"])
        if kind=="noul":
            if type(t["value"]) is not bool:raise ValueError("noul target must be boolean")
            return t["value"]
        return str(t["label"])
    result={"run":str(run),"fixture_sha256":EXPECTED_FIXTURE_SHA,"metrics_stage":"references opened after PRE_REFERENCE_FREEZE hashes verified",
            "probe":"mixed MLX/Metal embedding export + fixed CPU NumPy scorer; not all-GPU",
            "test_used_for_fit_or_selection":False,"models":{}}
    for m in MODELS:
        train_cv={}
        for v in VARIANTS:
            cv=json.loads((TRAIN_RUN/"cv"/m/v/"selection.json").read_text(encoding="utf-8"))
            train_cv[v]={"selected_train_cv_scorer":cv["selected"],"selected_metrics":cv["selected_metrics"],
                         "fold_count":cv["folds"],"grouping":cv["grouping"]}
        result["models"][m]={"frozen_train_selection":json.loads((TRAIN_RUN/"frozen"/m/"selection.json").read_text()),
                              "train_cv_by_prompt_variant":train_cv,"variants":{}}
        family_domain={}
        for v in VARIANTS:
            by_type={k:{"n":0,"correct":0,"candidate_local_macro_f1_sum":0.0,"score_abs_error_sum":0.0,"score_norm_abs_error_sum":0.0,"score_n":0} for k in TASKS}
            correct_by_case={}; correct_question={}; groups={"family":{},"domain":{}}
            for case in cases:
                cid=str(case["id"]); refs=case["reference"]
                qmap=case["request"]["questions"]
                if set(refs)!=set(qmap):raise ValueError(f"{cid}: references/question key mismatch")
                allcorrect=True
                for qname,qwire in qmap.items():
                    kind=str(qwire["type"]); ref=refs[qname]; gold=target(ref,kind)
                    d=preds[m,v][cid]["decisions"].get(qname)
                    if d is None:raise ValueError(f"missing prediction {m}/{v}/{cid}/{qname}")
                    hit=d["label"]==gold
                    allcorrect=allcorrect and hit;correct_question[cid,qname]=hit
                    metric=by_type[kind];metric["n"]+=1;metric["correct"]+=int(hit)
                    candidates=(qwire["options"] if kind=="choice" else (qwire["levels"] if kind=="score" else {"unsupported":0,"supported":1}))
                    k=len(candidates);metric["candidate_local_macro_f1_sum"]+=(1/k if hit else 0.0)
                    if kind=="score":
                        level=next((x for x in qwire["levels"] if str(x["label"])==str(gold)),None)
                        if level is None:raise ValueError("gold score label absent from variable rubric")
                        gold_value=float(level["value"]);absolute=abs(float(d["expected_value"])-gold_value)
                        span=max(float(x["value"]) for x in qwire["levels"])-min(float(x["value"]) for x in qwire["levels"])
                        metric["score_abs_error_sum"]+=absolute;metric["score_norm_abs_error_sum"]+=absolute/span if span>0 else 0.0;metric["score_n"]+=1
                    fam=str(ref.get("question_family_id",case.get("question_family_id","unknown")))
                    md=case.get("metadata",{}); md=md if isinstance(md,dict) else {}
                    dom=str(case.get("domain",md.get("domain","unknown")))
                    for gname,gval in (("family",fam),("domain",dom)):
                        group=groups[gname].setdefault(gval,{"n":0,"correct":0,"types":{}})
                        group["n"]+=1;group["correct"]+=int(hit)
                        gt=group["types"].setdefault(kind,{"n":0,"correct":0});gt["n"]+=1;gt["correct"]+=int(hit)
                correct_by_case[cid]=allcorrect
            typed={}
            for kind,metric in by_type.items():
                if not metric["n"]:continue
                typed[kind]={"n":metric["n"],"correct":metric["correct"],"accuracy":metric["correct"]/metric["n"],
                    "accuracy_wilson_95_ci":wilson(metric["correct"],metric["n"]),
                    "candidate_local_macro_f1":metric["candidate_local_macro_f1_sum"]/metric["n"]}
                if kind=="score":typed[kind].update({"expected_value_mae":metric["score_abs_error_sum"]/metric["score_n"],
                    "normalized_expected_value_mae":metric["score_norm_abs_error_sum"]/metric["score_n"],"score_n":metric["score_n"]})
            ntypes=len(typed);acc=sum(x["accuracy"] for x in typed.values())/ntypes;f1=sum(x["candidate_local_macro_f1"] for x in typed.values())/ntypes
            allhits=sum(correct_by_case.values());alln=len(correct_by_case)
            def group_report(g):
                return {name:{"n":x["n"],"correct":x["correct"],"accuracy":x["correct"]/x["n"],"per_type":{
                    k:{"n":z["n"],"correct":z["correct"],"accuracy":z["correct"]/z["n"]} for k,z in x["types"].items()}} for name,x in sorted(g.items())}
            result["models"][m]["variants"][v]={"per_type":typed,"type_balanced_accuracy":acc,"type_macro_candidate_local_f1":f1,
                "all_three_correct":{"correct":allhits,"n":alln,"accuracy":allhits/alln,"wilson_95_ci":wilson(allhits,alln)},
                "per_family":group_report(groups["family"]),"per_domain":group_report(groups["domain"])}
        # Paired, descriptive comparisons against frozen v2 baseline, at question level.
        result["models"][m]["paired_vs_v2_baseline"]={}
        for v in VARIANTS:
            if v=="v2-baseline":continue
            comparisons={}
            for kind in TASKS:
                pairs=[]
                for case in cases:
                    for qname,qwire in case["request"]["questions"].items():
                        if qwire["type"]!=kind:continue
                        base=correct_question_for(preds,m,"v2-baseline",case,qname,qwire,target)
                        alt=correct_question_for(preds,m,v,case,qname,qwire,target)
                        pairs.append((base,alt))
                better=sum((not b) and a for b,a in pairs);worse=sum(b and (not a) for b,a in pairs);n=len(pairs)
                comparisons[kind]={"n":n,"baseline_correct_only":sum(b and not a for b,a in pairs),"variant_correct_only":better,
                    "mcnemar_exact_two_sided_p_descriptive":exact_mcnemar(better,worse),"paired_accuracy_delta_variant_minus_baseline":(better-worse)/n,
                    "paired_accuracy_delta_normal_95_ci_descriptive":paired_delta_ci(better,worse,n)}
            result["models"][m]["paired_vs_v2_baseline"][v]=comparisons
    result["non_selection_caveat"]="All challenger holdout results are descriptive only; v2 baseline was frozen from train-only CV and this report does not select a test winner."
    dump(run/"holdout-metrics.json",result)
    write_group_csv(run,result)
    write_report(run,result,meta)
    print(json.dumps({"status":"scored after prediction freeze","metrics_path":str(run/"holdout-metrics.json"),
        "models":{m:{v:{"type_balanced_accuracy":x["type_balanced_accuracy"],"type_macro_candidate_local_f1":x["type_macro_candidate_local_f1"],"all_three_correct":x["all_three_correct"]}
            for v,x in doc["variants"].items()} for m,doc in result["models"].items()}},indent=2))

def correct_question_for(preds:dict,model:str,variant:str,case:dict,qname:str,qwire:dict,targetfn)->bool:
    d=preds[model,variant][str(case["id"])]["decisions"][qname]
    return d["label"]==targetfn(case["reference"][qname],str(qwire["type"]))

def write_report(run:Path,result:dict,meta:dict)->None:
    lines=["# Fresh60 prompt-variant holdout evaluation","",
        "**Status:** scored only after all eight prediction files were saved and hash-frozen. No fitting, tuning, or test-driven selection occurred. The probe is **mixed GPU/CPU**: MLX/Metal generated embeddings; the final frozen typed scorer was applied with CPU NumPy. This is not an all-GPU result.","",
        f"Fixture SHA-256: `{result['fixture_sha256']}`. Frozen train config SHA-256: `{meta['frozen_config_sha256']}`.","","## Per-model, per-variant summary","",
        "| Model | Variant | Type-balanced accuracy | Type-macro candidate-local F1 | All three correct |","|---|---|---:|---:|---:|"]
    for m,doc in result["models"].items():
        for v,x in doc["variants"].items():
            a=x["all_three_correct"]
            lines.append(f"| {m} | {v} | {x['type_balanced_accuracy']:.3f} | {x['type_macro_candidate_local_f1']:.3f} | {a['correct']}/{a['n']} ({a['accuracy']:.3f}) |")
    lines.extend(["","## Typed metrics","","Accuracy CIs are Wilson 95% descriptive intervals. Score MAE compares the model's probability-weighted expected value with the numeric value of the reference rubric label; normalized MAE divides each error by that question's rubric span.",""])
    for m,doc in result["models"].items():
        lines.append(f"### {m}")
        lines.append("| Variant | Type | Correct / n | Accuracy (Wilson 95% CI) | Candidate-local F1 | Expected-value MAE / normalized |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for v,x in doc["variants"].items():
            for k,t in x["per_type"].items():
                ci=t["accuracy_wilson_95_ci"]
                mae=f"{t['expected_value_mae']:.3f} / {t['normalized_expected_value_mae']:.3f}" if k=="score" else "—"
                lines.append(f"| {v} | {k} | {t['correct']}/{t['n']} | {t['accuracy']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {t['candidate_local_macro_f1']:.3f} | {mae} |")
        lines.append("")
    lines.extend(["## Paired comparisons","","Exact two-sided McNemar tests and normal-approximation paired accuracy-difference intervals compare each challenger with v2 baseline within the same model. They are descriptive, not a test-set selection mechanism.",""])
    for m,doc in result["models"].items():
        lines.append(f"### {m}")
        lines.append("| Variant | Type | Variant-only correct | Baseline-only correct | Exact McNemar p | Paired delta [95% descriptive CI] |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for v,types in doc["paired_vs_v2_baseline"].items():
            for k,x in types.items():
                ci=x["paired_accuracy_delta_normal_95_ci_descriptive"]
                lines.append(f"| {v} | {k} | {x['variant_correct_only']} | {x['baseline_correct_only']} | {x['mcnemar_exact_two_sided_p_descriptive']:.4f} | {x['paired_accuracy_delta_variant_minus_baseline']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] |")
        lines.append("")
    lines.extend(["## Train-only CV versus fresh60","","The train column is the pre-existing five-fold OOF result for the same text variant (with the train runner's internally selected scorer). Fresh60 uses the single frozen v2-baseline scorer unchanged for every text variant, so this is descriptive cross-protocol comparison, not a test-time refit or selection.",""])
    for m,doc in result["models"].items():
        lines.append(f"### {m}")
        lines.append("| Prompt variant | Train-only OOF type-balanced accuracy | Fresh60 type-balanced accuracy | Fresh60 − train OOF |")
        lines.append("|---|---:|---:|---:|")
        for v in VARIANTS:
            train=doc["train_cv_by_prompt_variant"][v]["selected_metrics"]["balanced_accuracy_exact_match"]
            fresh=doc["variants"][v]["type_balanced_accuracy"]
            lines.append(f"| {v} | {train:.3f} | {fresh:.3f} | {fresh-train:+.3f} |")
        lines.append("")
        lines.append("| Prompt variant | Type | Train-only OOF accuracy | Fresh60 accuracy | Fresh60 − train OOF |")
        lines.append("|---|---|---:|---:|---:|")
        for v in VARIANTS:
            train_types=doc["train_cv_by_prompt_variant"][v]["selected_metrics"]["per_type"]
            fresh_types=doc["variants"][v]["per_type"]
            for kind in TASKS:
                train_acc=train_types[kind]["accuracy"];fresh_acc=fresh_types[kind]["accuracy"]
                lines.append(f"| {v} | {kind} | {train_acc:.3f} | {fresh_acc:.3f} | {fresh_acc-train_acc:+.3f} |")
        lines.append("")
    lines.extend(["## Provenance, cost, and limits","","Every variant has an MLX/Metal feature export, text/profile/tokenizer/checkpoint manifest, prediction JSONL, feature SHA-256, prediction SHA-256, measured export wall time, exact query-plus-candidate token count, and estimated dense embedding FLOPs in `run-metadata.json`. Candidate score/order permutation is asserted on every question; explicit supported/unsupported polarity order checks are asserted for every noul question. Checkpoint hashes and exact feature text/token counts are in the exporter manifests.","",
       "FLOPs use `2 × 600,000,000 parameters × sum(query and candidate tokens)`. This is a transparent arithmetic estimate, not measured hardware FLOPs, and omits attention/cache and implementation details. Export wall timing includes exporter process startup, model/checkpoint verification/load, embedding, and output writing; it is not scorer latency.","",
       "The prompt variants are evaluated with the **unchanged training-frozen v2 typed weights** to isolate text/embedding changes without test fitting. Differences from train-only grouped CV are reported but do not alter the frozen selection. Embedding plus linear scoring does not acquire language-model reasoning just from prompt repetition.","",
       "Full per-family and per-domain total and per-type supports/correct counts/accuracies are in `holdout-metrics.json` and the flat `family-domain-metrics.csv`. All predictions were frozen before reference scoring in `PRE_REFERENCE_FREEZE.json`. No JEV/API was called.","",
       "All-MLX GPU parity is pending a separate agent: the nested-agent tool hit a depth limit, and the existing Metal runner does not directly accept these four isolated text variants. See `GPU-PARITY-HANDOFF.md`; this report remains mixed GPU/CPU.",""])
    (run/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")

def write_group_csv(run:Path,result:dict)->None:
    path=run/"family-domain-metrics.csv"
    fields=["model","variant","group_type","group_id","n","correct","accuracy","choice_n","choice_correct","choice_accuracy",
            "noul_n","noul_correct","noul_accuracy","score_n","score_correct","score_accuracy"]
    with path.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for model,doc in result["models"].items():
            for variant,metrics in doc["variants"].items():
                for group_type in ("family","domain"):
                    for group_id,group in metrics[f"per_{group_type}"].items():
                        row={"model":model,"variant":variant,"group_type":group_type,"group_id":group_id,"n":group["n"],
                             "correct":group["correct"],"accuracy":group["accuracy"]}
                        for kind in TASKS:
                            item=group.get("per_type",{}).get(kind,{})
                            row[f"{kind}_n"]=item.get("n",0);row[f"{kind}_correct"]=item.get("correct",0)
                            row[f"{kind}_accuracy"]=item.get("accuracy")
                        writer.writerow(row)

def main()->int:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--stage",choices=("predict","score"),required=True)
    p.add_argument("--fixture",type=Path,default=FIXTURE);p.add_argument("--run",type=Path,help="required for score; optional new output path for predict")
    p.add_argument("--model-store",type=Path,default=STORE);a=p.parse_args()
    if a.stage=="predict":
        run=a.run or HERE/"runs"/"holdout-fresh60-20260924T000000Z"
        predict_stage(a.fixture,run.resolve(),a.model_store.resolve())
    else:
        if not a.run:raise ValueError("--run required for --stage score")
        score_stage(a.run.resolve())
    return 0

if __name__=="__main__":raise SystemExit(main())
