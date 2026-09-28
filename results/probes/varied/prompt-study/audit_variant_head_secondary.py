#!/usr/bin/env python3
"""Audit existing train-only variant scorer bundles and the secondary fresh60 outputs."""
from __future__ import annotations
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
TRAIN=HERE/"runs/trainonly-clean300-20260924T002000Z";FIXTURE=ROOT/"benchmarks/cases/varied/multi/test-prompt-fresh-60.jsonl"
MODELS=("qwen","harrier");VARIANTS=("v2-baseline","readable-state","rule-in-candidate","joint-context-candidate")

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def canonsha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("run",type=Path);a=p.parse_args();run=a.run.resolve();secondary=run/"secondary/variant-heads"
    meta=load(secondary/"variant-head-run-metadata.json");freeze=load(secondary/"variant-head-freeze.json");primary=load(run/"PRE_REFERENCE_FREEZE.json")
    first_score=(run/"holdout-metrics.json").stat().st_mtime
    if sha(FIXTURE)!=meta["fixture_sha256"]:raise ValueError("fresh fixture SHA changed")
    if freeze.get("primary_freeze_sha256")!=sha(run/"PRE_REFERENCE_FREEZE.json"):raise ValueError("primary reference-freeze hash changed")
    result={"secondary_prediction_files_verified":0,"variant_scorer_bundles_verified":0,"head_files_predate_first_reference_score":True,
        "per_model_variant":{},"first_reference_score_mtime_utc":datetime.fromtimestamp(first_score,timezone.utc).isoformat(),
        "primary_freeze_sha256":sha(run/"PRE_REFERENCE_FREEZE.json"),"secondary_freeze_sha256":sha(secondary/"variant-head-freeze.json")}
    for m in MODELS:
        result["per_model_variant"][m]={}
        for v in VARIANTS:
            rec=meta["models"][m]["variants"][v];head=Path(rec["head_path"]);sidecar=Path(rec["sidecar_path"])
            if head.stat().st_mtime>=first_score or sidecar.stat().st_mtime>=first_score:raise ValueError(f"head was not pre-frozen: {m}/{v}")
            hdoc=load(head);digest=sha(head)
            if digest!=rec["sha256"] or sidecar.read_text().split()[0]!=digest:raise ValueError("scorer bundle SHA/sidecar mismatch")
            cv=load(TRAIN/"cv"/m/v/"frozen-config.json");selected=cv["selected"]
            if selected["variant"]!="baseline" or selected["objective"]!="per_type" or selected["noul_balance"] or float(selected["C"])!=float(hdoc["C"]):raise ValueError("head CV selection/C mismatch")
            if hdoc["variant"]!=v or hdoc["model_key"]!=m or hdoc["training_scope"]!="clean300 + multi120 only; targets read only from those train fixtures":raise ValueError("head model/variant/train-scope mismatch")
            for kind,s in hdoc["scorers"].items():
                norm=rec["normalization_and_weights"][kind]
                for field,key in (("mean","mean_sha256"),("scale","scale_sha256"),("weights","weights_sha256")):
                    if canonsha(s[field])!=norm[key]:raise ValueError(f"normalizer/head hash mismatch {m}/{v}/{kind}/{field}")
                if not s["fit_diagnostics"]["converged"] or len(s["weights"])!=4096:raise ValueError("invalid frozen typed head")
            path=Path(rec["prediction_path"]);key=f"{m}/{v}"
            if sha(path)!=freeze["predictions"][key] or sha(path)!=rec["predictions_sha256"]:raise ValueError("secondary prediction checksum mismatch")
            preds=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
            if len(preds)!=60 or any(x.get("scorer_head_sha256")!=digest for x in preds):raise ValueError("secondary prediction shape/head binding mismatch")
            metrics=load(secondary/"secondary-head-metrics.json")["models"][m]["variants"][v]
            if any(metrics["per_type"][k]["n"]!=60 for k in ("choice","noul","score")) or metrics["all_three_correct"]["n"]!=60:raise ValueError("secondary metric support incomplete")
            result["secondary_prediction_files_verified"]+=1;result["variant_scorer_bundles_verified"]+=1
            result["per_model_variant"][m][v]={"head_sha256":digest,"C":hdoc["C"],"head_mtime_utc":rec["mtime_utc"],
                "norm_sha256":{k:{"mean":x["mean_sha256"],"scale":x["scale_sha256"],"weights":x["weights_sha256"]} for k,x in rec["normalization_and_weights"].items()},
                "prediction_path":str(path),"predictions_sha256":sha(path),"feature_sha256":rec["feature_sha256"]}
    if result["secondary_prediction_files_verified"]!=8 or result["variant_scorer_bundles_verified"]!=8:raise ValueError("incomplete secondary 8-way evaluation")
    parity=load(secondary/"pre-reference-baseline-parity.json")
    if any(x["exact_labels"]!=180 or x["max_probability_abs_diff"]!=0 for x in parity["models"].values()):raise ValueError("v2-baseline sidecar is not bit-exact to primary frozen scorer")
    result["v2_baseline_cross_path_parity"]=parity["models"]
    result["scoring_scope"]="pre-existing heads only; no fitting/tuning/selection; CPU NumPy reference"
    out=secondary/"secondary-integrity-audit.json";out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"heads_verified":result["variant_scorer_bundles_verified"],"predictions_verified":result["secondary_prediction_files_verified"],
        "all_heads_predate_first_score":result["head_files_predate_first_reference_score"],"audit":str(out)},indent=2))
if __name__=="__main__":main()
