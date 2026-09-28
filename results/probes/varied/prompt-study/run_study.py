#!/usr/bin/env python3
"""Export train-only variants, run grouped CV, freeze winner and train final fit."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TRAIN = HERE / "data/train-clean-300.jsonl"
MULTI = ROOT / "benchmarks/cases/varied/multi/train-120.jsonl"
STORE = Path("/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/decengine-model-store")
VARIANTS = ("v2-baseline", "readable-state", "rule-in-candidate", "joint-context-candidate")
MODELS = ("Qwen/Qwen3-Embedding-0.6B", "microsoft/harrier-oss-v1-0.6b")

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def jread(path): return json.loads(Path(path).read_text())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--model-store",type=Path,default=STORE)
    p.add_argument("--skip-export",action="store_true",help="use only if all variant/model/split feature files already exist")
    p.add_argument("--resume",action="store_true",help="resume this run directory; completed exports are hash-validated and reused")
    a=p.parse_args(); out=a.out.resolve()
    if a.resume:
        if not out.is_dir(): raise FileNotFoundError(f"cannot resume missing run directory {out}")
    else: out.mkdir(parents=True,exist_ok=False)
    if len([x for x in TRAIN.read_text().splitlines() if x.strip()]) != 300 or len([x for x in MULTI.read_text().splitlines() if x.strip()]) != 120:
        raise ValueError("expected clean300 and multi-train120 only")
    data = {"single":TRAIN,"multi":MULTI}
    feature = {}
    for model in MODELS:
        modelkey="qwen" if model.startswith("Qwen") else "harrier"
        for variant in VARIANTS:
            for split,inp in data.items():
                dest=out/"features"/modelkey/variant/f"{split}.jsonl"; dest.parent.mkdir(parents=True,exist_ok=True)
                if a.resume and dest.is_file() and Path(str(dest)+".manifest.json").is_file():
                    manifest=jread(str(dest)+".manifest.json")
                    if manifest.get("input_sha256")!=sha(inp) or manifest.get("variant")!=variant or manifest.get("model")!=model:
                        raise ValueError(f"refusing to resume mismatched export {dest}")
                    if manifest.get("output_sha256") and manifest["output_sha256"]!=sha(dest): raise ValueError(f"resume feature hash mismatch {dest}")
                elif a.skip_export:
                    if not dest.is_file() or not Path(str(dest)+".manifest.json").is_file(): raise FileNotFoundError(f"--skip-export requires complete feature pair: {dest}")
                else:
                    subprocess.run(["cargo","run","--release","--manifest-path",str(HERE/"Cargo.toml"),"--",
                        "--home",str(a.model_store),"--input",str(inp),"--output",str(dest),"--model",model,"--variant",variant],cwd=ROOT,check=True)
                feature[modelkey,variant,split]=dest
    results={}
    for model in MODELS:
        mk="qwen" if model.startswith("Qwen") else "harrier"; results[mk]={}
        for variant in VARIANTS:
            run=out/"cv"/mk/variant; run.mkdir(parents=True,exist_ok=True)
            subprocess.run([sys.executable,str(HERE/"train_cv.py"),"--train",str(TRAIN),"--train-embeddings",str(feature[mk,variant,"single"]),
                "--multi-train",str(MULTI),"--multi-train-embeddings",str(feature[mk,variant,"multi"]),"--model",model,"--out",str(run)],cwd=ROOT,check=True)
            results[mk][variant]=jread(run/"selection.json")
    frozen={"protocol_sha256":sha(HERE/"PROTOCOL.md"),"exporter_sha256":sha(HERE/"src/exporter_base.rs"),
        "runner_sha256":sha(HERE/"run_study.py"),"training_script_sha256":sha(HERE/"train_cv.py"),"fixture_sha256":{"single":sha(TRAIN),"multi":sha(MULTI)},"models":{}}
    for mk,variants in results.items():
        baseline=variants["v2-baseline"]
        bscore=baseline["selected_metrics"]
        bfold=baseline["robustness_comparison"]["folds"] if baseline["selected"]["kept_challenger"] else None
        # Compare each candidate's selected OOF result against baseline selected OOF.
        eligible=[]
        for name,doc in variants.items():
            if name=="v2-baseline": continue
            score=doc["selected_metrics"]
            base_oof=bscore; alt_oof=score
            # Robust fold policy uses the same five grouped folds; reconstruct per-fold selected model metrics.
            base_entry=next(x for x in doc["all_cv"] if x["variant"]==baseline["selected"]["variant"] and x["C"]==baseline["selected"]["C"])
            alt_entry=doc["all_cv"] and next(x for x in doc["all_cv"] if x["variant"]==doc["selected"]["variant"] and x["C"]==doc["selected"]["C"])
            folds=[]
            for bf,af in zip(base_entry["fold_metrics"],alt_entry["fold_metrics"]):
                folds.append(af["macro_candidate_local_f1_by_type"]>bf["macro_candidate_local_f1_by_type"] and af["balanced_accuracy_exact_match"]>bf["balanced_accuracy_exact_match"])
            if (alt_oof["macro_candidate_local_f1_by_type"]>base_oof["macro_candidate_local_f1_by_type"] and
                alt_oof["balanced_accuracy_exact_match"]>base_oof["balanced_accuracy_exact_match"] and sum(folds)>=4):
                eligible.append((alt_oof["macro_candidate_local_f1_by_type"],alt_oof["balanced_accuracy_exact_match"],name,doc))
        winner=max(eligible)[2:] if eligible else ("v2-baseline",baseline)
        frozen["models"][mk]={"selected_variant":winner[0],"robust_improvement":bool(eligible),"baseline_metrics":bscore,
            "selected_metrics":winner[1]["selected_metrics"],"candidate_variants_that_passed":sorted(x[2] for x in eligible)}
        (out/"frozen"/mk).mkdir(parents=True,exist_ok=True)
        (out/"frozen"/mk/"selection.json").write_text(json.dumps(frozen["models"][mk],indent=2)+"\n")
        # Final fit follows selection and sees only clean300 + multi120 labels.
        sys.path.insert(0,str(HERE.parent))
        import multi_train_pairwise as trainer
        import importlib.util
        spec=importlib.util.spec_from_file_location("prompt_study_cv",HERE/"train_cv.py")
        cvmod=importlib.util.module_from_spec(spec); spec.loader.exec_module(cvmod)
        chosen=winner[1]; chosen_variant=chosen["selected"]["variant"]; chosen_c=float(chosen["selected"]["C"])
        vecs1,_=trainer.vectors(feature[mk,winner[0],"single"],MODELS[0] if mk=="qwen" else MODELS[1])
        vecs2,_=trainer.vectors(feature[mk,winner[0],"multi"],MODELS[0] if mk=="qwen" else MODELS[1])
        one=trainer.lines(TRAIN); many=trainer.lines(MULTI)
        cases1,refs1=trainer.flatten(one,vecs1,False,"clean300 train"); cases2,refs2=trainer.flatten(many,vecs2,True,"multi120 train")
        cases=cases1+cases2; refs=refs1+refs2; targets=[]
        for case,ref in zip(cases,refs): targets.append(case["keys"].index(trainer.get_target(ref,case["type"])))
        models={}; selected_spec=next(x for x in chosen["all_cv"] if x["variant"]==chosen_variant and float(x["C"])==chosen_c)
        mode=next(x for x in chosen["variants"] if x["name"]==chosen_variant)
        for kind in ("choice","noul","score"):
            ix=[i for i,c in enumerate(cases) if c["type"]==kind]
            w,mean,scale,diag=cvmod.fit_variant([cases[i] for i in ix],[targets[i] for i in ix],chosen_c,
                noul_balance=mode["noul_balance"],score_smoothing=mode["score_smoothing"])
            models[kind]={"weights":w.tolist(),"mean":mean.tolist(),"scale":scale.tolist(),
                "candidate_keys":sorted({k for c in cases if c["type"]==kind for k in c["keys"]}),"fit_diagnostics":diag}
        (out/"frozen"/mk/"models.json").write_text(json.dumps({"model":MODELS[0] if mk=="qwen" else MODELS[1],"variant":winner[0],"C":chosen_c,"scorers":models},allow_nan=False)+"\n")
        # Cost accounting uses recorded exact tokenizer lengths; FLOPs are a transparent
        # dense 2*parameter_count*tokens estimate, not a measured device count.
        costs={}
        for tested_variant in VARIANTS:
            costs[tested_variant]={}
            for split in ("single","multi"):
                fp=feature[mk,tested_variant,split]; lines=[json.loads(x) for x in fp.read_text().splitlines() if x.strip()]
                tok=sum(int(row["query_token_length"])+sum(int(c["token_length"]) for c in row["candidates"]) for row in lines)
                man=jread(str(fp)+".manifest.json")
                costs[tested_variant][split]={"question_records":len(lines),"embedding_calls":man["embedding_call_count"],"query_plus_candidate_tokens":tok,
                    "estimated_dense_embedding_flops_2pt":int(2*600_000_000*tok),"feature_sha256":sha(fp),"manifest_sha256":sha(str(fp)+".manifest.json"),
                    "model":man["model"],"profile_version":man["profile_version"],"profile_sha256":man["profile_sha256"],
                    "tokenizer_sha256":man["tokenizer_sha256"],"checkpoint_files":man["checkpoint_files"],"text_version":man["text_version"]}
        frozen["models"][mk]["cost_accounting"]=costs
        (out/"frozen"/mk/"selection.json").write_text(json.dumps(frozen["models"][mk],indent=2)+"\n")
    (out/"frozen-config.json").write_text(json.dumps(frozen,indent=2)+"\n")
    # Every CV OOF row remains under cv/<model>/<variant>; all fits use train data only.
    print(json.dumps(frozen["models"],indent=2))

if __name__=="__main__": main()
