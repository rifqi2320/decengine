#!/usr/bin/env python3
"""Train-artifact-only audit for a completed prompt-study run; never accepts test data."""
from __future__ import annotations
import argparse, hashlib, json, math, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[3]
TRAIN=HERE/"data/train-clean-300.jsonl"; MULTI=ROOT/"benchmarks/cases/varied/multi/train-120.jsonl"
VARIANTS=("v2-baseline","readable-state","rule-in-candidate","joint-context-candidate")
STORE=Path("/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/decengine-model-store")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]

def metric(items):
    out={}
    for kind in ("choice","noul","score"):
        subset=[x for x in items if x["question_type"]==kind]
        if not subset: raise ValueError(f"missing type {kind}")
        hits=[x["selected"]==x["gold"] for x in subset]
        out[kind]={"n":len(subset),"correct":sum(hits),"accuracy":sum(hits)/len(hits),
                   "candidate_local_macro_f1":sum((1/len(x["candidate_keys"]) if h else 0) for x,h in zip(subset,hits))/len(subset)}
    out["macro_accuracy_by_type"]=sum(out[k]["accuracy"] for k in ("choice","noul","score"))/3
    out["macro_candidate_local_f1_by_type"]=sum(out[k]["candidate_local_macro_f1"] for k in ("choice","noul","score"))/3
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("run",type=Path); ap.add_argument("--model-store",type=Path,default=STORE); a=ap.parse_args()
    run=a.run.resolve(); frozen=load(run/"frozen-config.json")
    if sha(TRAIN)!=frozen["fixture_sha256"]["single"] or sha(MULTI)!=frozen["fixture_sha256"]["multi"]: raise ValueError("training fixture hashes changed")
    binary=HERE/"target/release/decengine-prompt-study-exporter"
    report={"run":str(run),"scope":"clean300 + multi120 only","frozen_config_sha256":sha(run/"frozen-config.json"),
            "exporter_binary_sha256":sha(binary) if binary.is_file() else None,"models":{}}
    for model in ("qwen","harrier"):
        selected=frozen["models"][model]; variant=selected["selected_variant"]; cvdir=run/"cv"/model/variant
        cv=load(cvdir/"selection.json"); oof=rows(cvdir/f"oof-{cv['selected']['variant']}-C{cv['selected']['C']:g}.jsonl")
        if len(oof)!=660: raise ValueError(f"{model}: expected 660 train OOF rows")
        for filename,digest in cv.get("oof_predictions_sha256",{}).items():
            if sha(cvdir/filename)!=digest: raise ValueError(f"OOF prediction hash mismatch: {model}/{filename}")
        foldmap={};
        for r in oof:
            if not isinstance(r.get("fold"),int) or not 0<=r["fold"]<5: raise ValueError("invalid fold id")
            for key in (("id",str(r["id"])),("family",str(r["family"]))):
                if key in foldmap and foldmap[key]!=r["fold"]: raise ValueError("case/family leakage across grouped folds")
                foldmap[key]=r["fold"]
        # OOF metrics are reconstructed from train-only gold annotations in OOF output.
        stats=metric(oof)
        for typ in ("choice","noul","score"):
            for key in ("accuracy","candidate_local_macro_f1"):
                if abs(stats[typ][key]-cv["selected_metrics"]["per_type"][typ][key])>1e-12: raise ValueError(f"OOF metric mismatch {model}/{typ}/{key}")
        stats["score"]["normalized_rubric_expected_value_mae"]=cv["selected_metrics"]["per_type"]["score"]["normalized_rubric_expected_value_mae"]
        baseline=load(run/"cv"/model/"v2-baseline"/"selection.json")
        def entry(doc): return next(x for x in doc["all_cv"] if x["variant"]==doc["selected"]["variant"] and float(x["C"])==float(doc["selected"]["C"]))
        bentry=entry(baseline); bmetrics=baseline["selected_metrics"]; passed=[]; robust_folds={}
        for challenger_name in (x for x in VARIANTS if x!="v2-baseline"):
            challenger=load(run/"cv"/model/challenger_name/"selection.json"); centry=entry(challenger); cmetrics=challenger["selected_metrics"]
            strict=[]
            for bf,cf in zip(bentry["fold_metrics"],centry["fold_metrics"]):
                strict.append(cf["macro_candidate_local_f1_by_type"]>bf["macro_candidate_local_f1_by_type"] and cf["balanced_accuracy_exact_match"]>bf["balanced_accuracy_exact_match"])
            robust_folds[challenger_name]=sum(strict)
            if (cmetrics["macro_candidate_local_f1_by_type"]>bmetrics["macro_candidate_local_f1_by_type"] and
                cmetrics["balanced_accuracy_exact_match"]>bmetrics["balanced_accuracy_exact_match"] and sum(strict)>=4): passed.append(challenger_name)
        if sorted(passed)!=sorted(selected["candidate_variants_that_passed"]): raise ValueError("robust-selection pass list mismatch")
        expected_winner=max(passed,key=lambda x:(load(run/"cv"/model/x/"selection.json")["selected_metrics"]["macro_candidate_local_f1_by_type"],
            load(run/"cv"/model/x/"selection.json")["selected_metrics"]["balanced_accuracy_exact_match"],x)) if passed else "v2-baseline"
        if variant!=expected_winner or bool(passed)!=bool(selected["robust_improvement"]): raise ValueError("frozen prompt variant violates robust selection policy")
        final=load(run/"frozen"/model/"models.json")
        if final.get("variant")!=variant or float(final.get("C",-1))!=float(cv["selected"]["C"]): raise ValueError("final fit does not match frozen variant/C")
        if set(final.get("scorers",{}))!={"choice","noul","score"}: raise ValueError("final typed fit is incomplete")
        for task,scorer in final["scorers"].items():
            if len(scorer["weights"])!=4096 or len(scorer["mean"])!=4096 or len(scorer["scale"])!=4096: raise ValueError("final scorer dimension mismatch")
            if not all(math.isfinite(float(x)) for key in ("weights","mean","scale") for x in scorer[key]): raise ValueError("nonfinite final coefficient")
            if not scorer.get("fit_diagnostics",{}).get("converged"): raise ValueError("final scorer did not report convergence")
        # Verify the final model's scoring rule is candidate-position invariant using
        # the selected training embeddings only. No heldout fixture is accepted here.
        sys.path.insert(0,str(HERE.parent)); import multi_train_pairwise as tr
        model_id=final["model"]; vec1,_=tr.vectors(run/"features"/model/variant/"single.jsonl",model_id); vec2,_=tr.vectors(run/"features"/model/variant/"multi.jsonl",model_id)
        c1,_=tr.flatten(tr.lines(TRAIN),vec1,False,"clean300 train"); c2,_=tr.flatten(tr.lines(MULTI),vec2,True,"multi120 train")
        checks={"pairwise_order_invariance":0,"noul_polarity_order_invariance":0}
        for case in c1+c2:
            sc=final["scorers"][case["type"]]; w=__import__("numpy").asarray(sc["weights"]); mean=__import__("numpy").asarray(sc["mean"]); scale=__import__("numpy").asarray(sc["scale"])
            x=(case["x"]-mean)/scale; score=x@w; reverse=((case["x"][::-1]-mean)/scale)@w
            if not __import__("numpy").allclose(score,reverse[::-1],rtol=0,atol=1e-10): raise ValueError("candidate permutation changed candidate scores")
            checks["pairwise_order_invariance"]+=1
            if case["type"]=="noul":
                label_to_score=dict(zip(case["keys"],score)); reversed_map=dict(zip(case["keys"][::-1],reverse))
                if not __import__("numpy").allclose([label_to_score[k] for k in label_to_score],[reversed_map[k] for k in label_to_score],rtol=0,atol=1e-10): raise ValueError("noul polarity changed under order swap")
                checks["noul_polarity_order_invariance"]+=1
        costs={}; profile_hashes=set(); tokenizer_hashes=set(); checked_checkpoints=set()
        for variant_name in VARIANTS:
            costs[variant_name]={}
            for split in ("single","multi"):
                fp=run/"features"/model/variant_name/f"{split}.jsonl"; man=load(str(fp)+".manifest.json")
                expected_input=TRAIN if split=="single" else MULTI
                if man.get("input_sha256")!=sha(expected_input): raise ValueError("feature exporter input hash mismatch")
                if man.get("output_sha256")!=sha(fp): raise ValueError("feature exporter output hash mismatch")
                mdir=a.model_store/"models"/("Qwen--Qwen3-Embedding-0.6B" if model=="qwen" else "microsoft--harrier-oss-v1-0.6b")
                for checkpoint in man["checkpoint_files"]:
                    path=mdir/checkpoint["path"]
                    if (str(path),checkpoint["sha256"]) not in checked_checkpoints:
                        if sha(path)!=checkpoint["sha256"] or path.stat().st_size!=checkpoint["size"]: raise ValueError("checkpoint artifact hash/size mismatch")
                        checked_checkpoints.add((str(path),checkpoint["sha256"]))
                records=rows(fp); tokens=sum(int(x["query_token_length"])+sum(int(c["token_length"]) for c in x["candidates"]) for x in records)
                if len(records)!=(300 if split=="single" else 360): raise ValueError("feature question count mismatch")
                if man.get("embedding_dimensions")!=1024 or man.get("normalization")!="L2": raise ValueError("model feature contract mismatch")
                profile_hashes.add(man["profile_sha256"]); tokenizer_hashes.add(man["tokenizer_sha256"])
                costs[variant_name][split]={"questions":len(records),"embedding_calls":man["embedding_call_count"],"tokens":tokens,"estimated_dense_flops_2pt":1_200_000_000*tokens,
                   "feature_sha256":sha(fp),"manifest_sha256":sha(str(fp)+".manifest.json"),"profile_sha256":man["profile_sha256"],"text_version":man["text_version"]}
        if len(profile_hashes)!=1 or len(tokenizer_hashes)!=1: raise ValueError("variant/profile/tokenizer mismatch within model")
        report["models"][model]={"selected_variant":variant,"robust_improvement":selected["robust_improvement"],"train_oof_metrics":stats,
            "fold_count":len(set(foldmap.values())),"case_family_membership_keys":len(foldmap),"permutation_checks":checks,"final_models_sha256":sha(run/"frozen"/model/"models.json"),"robust_challenger_fold_improvements":robust_folds,"passed_challengers":passed,"costs":costs}
    (run/"audit.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
