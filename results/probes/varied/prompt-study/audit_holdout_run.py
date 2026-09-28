#!/usr/bin/env python3
"""Verify frozen fresh60 prompt-study outputs; never fits or selects."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
TRAIN=HERE/"runs/trainonly-clean300-20260924T002000Z"
FIXTURE=ROOT/"benchmarks/cases/varied/multi/test-prompt-fresh-60.jsonl"
STORE=Path("/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/decengine-model-store")
MODELS={"qwen":"Qwen/Qwen3-Embedding-0.6B","harrier":"microsoft/harrier-oss-v1-0.6b"}
VARIANTS=("v2-baseline","readable-state","rule-in-candidate","joint-context-candidate")

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("run",type=Path);p.add_argument("--model-store",type=Path,default=STORE);a=p.parse_args();run=a.run.resolve()
    meta=load(run/"run-metadata.json");freeze=load(run/"PRE_REFERENCE_FREEZE.json");metrics=load(run/"holdout-metrics.json")
    if sha(FIXTURE)!=meta["fixture_sha256"] or sha(FIXTURE)!="316b3c2e1d0c5cf50356d0e236a5f4adf27fdac9ee6d6c6b56a14f5354ef6a9f":raise ValueError("fixture digest mismatch")
    if freeze["status"]!="predictions_frozen_before_reference_scoring" or len(freeze["prediction_hashes"])!=8:raise ValueError("invalid/partial pre-reference freeze")
    if sha(TRAIN/"frozen-config.json")!=freeze["frozen_config_sha256"]:raise ValueError("train selection hash mismatch")
    if sha(HERE/"target/release/decengine-prompt-study-exporter")!=freeze["exporter_binary_sha256"]:raise ValueError("exporter binary hash mismatch")
    results={"run":str(run),"fixture_sha256":meta["fixture_sha256"],"frozen_config_sha256":freeze["frozen_config_sha256"],"prediction_files_verified":0,
             "features_verified":0,"checkpoint_files_verified":0,"question_records_verified":0,"model_variant_provenance":{}}
    checked_checkpoint_files=set()
    for m,mid in MODELS.items():
        train_manifest=load(TRAIN/"features"/m/"v2-baseline"/"single.jsonl.manifest.json")
        modelhash=sha(TRAIN/"frozen"/m/"models.json")
        if meta["models"][m]["final_fit_sha256"]!=modelhash:raise ValueError(f"final weights changed {m}")
        mdir=a.model_store/"models"/("Qwen--Qwen3-Embedding-0.6B" if m=="qwen" else "microsoft--harrier-oss-v1-0.6b")
        results["model_variant_provenance"][m]={"model_id":mid,"final_fit_sha256":modelhash,"variants":{}}
        for v in VARIANTS:
            rec=meta["models"][m]["variants"][v];prefix=f"{m}/{v}"
            fp=run/"features"/m/v/"fresh60.features.jsonl";manifest_path=Path(str(fp)+".manifest.json");pp=run/"predictions"/m/f"{v}.jsonl"
            if sha(pp)!=freeze["prediction_hashes"][prefix] or sha(pp)!=rec["predictions_sha256"]:raise ValueError(f"prediction hash mismatch {prefix}")
            if sha(fp)!=freeze["feature_hashes"][prefix] or sha(fp)!=rec["features_sha256"]:raise ValueError(f"feature hash mismatch {prefix}")
            manifest=load(manifest_path)
            if sha(manifest_path)!=rec["manifest_sha256"]:raise ValueError(f"feature manifest hash mismatch {prefix}")
            if manifest["input_sha256"]!=meta["fixture_sha256"] or manifest["output_sha256"]!=sha(fp):raise ValueError(f"feature manifest input/output hash mismatch {prefix}")
            if manifest["model"]!=mid or manifest["variant"]!=v or manifest["text_version"]!=f"decengine-varied-pair-text-v2-generic-types/{v}" or manifest["backend"]!="MLX/Metal":raise ValueError(f"model/text/backend mismatch {prefix}")
            for key in ("profile_version","profile_sha256","tokenizer_sha256","checkpoint_files"):
                if manifest[key]!=train_manifest[key]:raise ValueError(f"frozen profile/checkpoint mismatch {prefix}/{key}")
            for item in manifest["checkpoint_files"]:
                cp=mdir/item["path"]
                key=(str(cp),item["sha256"])
                if key not in checked_checkpoint_files:
                    if cp.stat().st_size!=item["size"] or sha(cp)!=item["sha256"]:raise ValueError(f"installed checkpoint mismatch {prefix}/{item['path']}")
                    checked_checkpoint_files.add(key);results["checkpoint_files_verified"]+=1
            predictions=[json.loads(x) for x in pp.read_text(encoding="utf-8").splitlines() if x.strip()]
            features=[json.loads(x) for x in fp.read_text(encoding="utf-8").splitlines() if x.strip()]
            if len(predictions)!=60 or len(features)!=180 or rec["prediction_count"]!=60 or rec["question_count"]!=180:raise ValueError(f"record count mismatch {prefix}")
            if rec["embedding_calls"]!=manifest["embedding_call_count"]:raise ValueError(f"embedding call mismatch {prefix}")
            tokens=sum(int(x["query_token_length"])+sum(int(c["token_length"]) for c in x["candidates"]) for x in features)
            if tokens!=rec["query_plus_candidate_tokens"] or 2*600_000_000*tokens!=rec["estimated_dense_embedding_flops_2pt"]:raise ValueError(f"token/FLOP account mismatch {prefix}")
            if rec["permutation_checked"]!=180 or rec["noul_polarity_checked"]!=60:raise ValueError(f"permutation/polarity assertion accounting mismatch {prefix}")
            resultvar=metrics["models"][m]["variants"][v]
            if sum(x["n"] for x in resultvar["per_type"].values())!=180 or resultvar["all_three_correct"]["n"]!=60:raise ValueError(f"metric support mismatch {prefix}")
            results["prediction_files_verified"]+=1;results["features_verified"]+=1;results["question_records_verified"]+=180
            results["model_variant_provenance"][m]["variants"][v]={"profile_sha256":manifest["profile_sha256"],"text_version":manifest["text_version"],
                "tokenizer_sha256":manifest["tokenizer_sha256"],"features_sha256":sha(fp),"predictions_sha256":sha(pp),
                "embedding_calls":rec["embedding_calls"],"tokens":tokens,"measured_export_wall_seconds":rec["measured_export_wall_seconds"]}
    if results["prediction_files_verified"]!=8 or results["features_verified"]!=8 or results["question_records_verified"]!=1440:raise ValueError("incomplete eight-way result")
    results["holdout_metrics_sha256"]=sha(run/"holdout-metrics.json");results["report_sha256"]=sha(run/"REPORT.md")
    results["family_domain_csv_sha256"]=sha(run/"family-domain-metrics.csv")
    results["evaluation_script_sha256_at_audit"]=sha(HERE/"evaluate_prompt_fresh.py")
    results["integrity_auditor_sha256"]=sha(Path(__file__))
    (run/"integrity-audit.json").write_text(json.dumps(results,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"verified_predictions":results["prediction_files_verified"],"verified_feature_exports":results["features_verified"],
        "question_feature_records":results["question_records_verified"],"checkpoint_files_verified":results["checkpoint_files_verified"],
        "audit":str(run/"integrity-audit.json")},indent=2))
if __name__=="__main__":main()
