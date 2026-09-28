#!/usr/bin/env python3
"""Make deterministic final scorer bundles from prompt-study train-only features/configs.

This does not open fresh-holdout files. The v2-baseline refits are asserted bit-exact
against the already frozen final models. Challenger text scorers use their own recorded
train-only per-variant CV C; no parameters are selected from the holdout.
"""
from __future__ import annotations
import hashlib, importlib.util, json, pathlib, sys
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[5]
HERE=ROOT/"results/probes/varied/prompt-study"
RUN=HERE/"runs/trainonly-clean300-20260924T002000Z"
OUT=pathlib.Path(__file__).resolve().parent/"frozen-scorers"
TRAIN=HERE/"data/train-clean-300.jsonl"
MULTI=ROOT/"benchmarks/cases/varied/multi/train-120.jsonl"
VARIANTS=("v2-baseline","readable-state","rule-in-candidate","joint-context-candidate")
MODELS={"qwen":"Qwen/Qwen3-Embedding-0.6B","harrier":"microsoft/harrier-oss-v1-0.6b"}
sys.path.insert(0,str(HERE.parent))
import multi_train_pairwise as trainer  # noqa: E402
spec=importlib.util.spec_from_file_location("prompt_study_cv",HERE/"train_cv.py")
cv=importlib.util.module_from_spec(spec);spec.loader.exec_module(cv)

def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

single=trainer.lines(TRAIN);multi=trainer.lines(MULTI)
if len(single)!=300 or len(multi)!=120:raise SystemExit("frozen training split sizes changed")
for mk,model in MODELS.items():
 for variant in VARIANTS:
  features=RUN/"features"/mk/variant
  manifests=[pathlib.Path(str(features/x)+".manifest.json") for x in ("single.jsonl","multi.jsonl")]
  for p in manifests:
   if not p.is_file():raise SystemExit(f"missing train-only variant manifest {p}")
   m=json.loads(p.read_text())
   if m.get("variant")!=variant or m.get("model")!=model:raise SystemExit(f"variant/model identity mismatch: {p}")
  cvcfg=json.loads((RUN/"cv"/mk/variant/"frozen-config.json").read_text())
  selected=cvcfg["selected"]
  if selected.get("variant")!="baseline" or selected.get("objective")!="per_type" or selected.get("noul_balance") or selected.get("score_smoothing")!=0:
   raise SystemExit(f"unsupported frozen per-variant scorer config {mk}/{variant}: {selected}")
  C=float(selected["C"])
  vec1,mod1=trainer.vectors(features/"single.jsonl",model)
  vec2,mod2=trainer.vectors(features/"multi.jsonl",model)
  if mod1!=model or mod2!=model:raise SystemExit(f"train feature model mismatch {mk}/{variant}")
  cases1,refs1=trainer.flatten(single,vec1,False,"clean300 train")
  cases2,refs2=trainer.flatten(multi,vec2,True,"multi120 train")
  cases=cases1+cases2;refs=refs1+refs2
  targets=[c["keys"].index(trainer.get_target(r,c["type"])) for c,r in zip(cases,refs)]
  if len(cases)!=660:raise SystemExit(f"wrong train-only question count for {mk}/{variant}")
  scorers={}
  for kind in ("choice","noul","score"):
   ix=[i for i,c in enumerate(cases) if c["type"]==kind]
   w,mean,scale,diag=cv.fit_variant([cases[i] for i in ix],[targets[i] for i in ix],C)
   scorers[kind]={"weights":w.tolist(),"mean":mean.tolist(),"scale":scale.tolist(),"fit_diagnostics":diag}
  if variant=="v2-baseline":
   frozen=json.loads((RUN/"frozen"/mk/"models.json").read_text())
   if frozen["variant"]!=variant or float(frozen["C"])!=C:raise SystemExit(f"baseline frozen config mismatch {mk}")
   for kind in scorers:
    for field in ("weights","mean","scale"):
     if not np.array_equal(np.asarray(scorers[kind][field]),np.asarray(frozen["scorers"][kind][field])):
      raise SystemExit(f"baseline refit is not bit-exact for {mk}/{kind}/{field}")
  doc={"model":model,"model_key":mk,"variant":variant,"C":C,"scorers":scorers,
       "training_scope":"clean300 + multi120 only; targets read only from those train fixtures",
       "baseline_bit_exact_to_published_frozen_models":variant=="v2-baseline",
       "provenance":{"models_sha256":sha(RUN/"frozen"/mk/"models.json"),
        "selection_sha256":sha(RUN/"frozen"/mk/"selection.json"),"variant_cv_config_sha256":sha(RUN/"cv"/mk/variant/"frozen-config.json"),
        "train_case_sha256":sha(TRAIN),"multi_train_case_sha256":sha(MULTI),
        "single_features_sha256":sha(features/"single.jsonl"),"single_manifest_sha256":sha(manifests[0]),
        "multi_features_sha256":sha(features/"multi.jsonl"),"multi_manifest_sha256":sha(manifests[1]),
        "optimizer_script_sha256":sha(HERE/"train_cv.py"),"optimizer_protocol":"train_cv.fit_variant baseline/per_type deterministic bounded L-BFGS, fixed C from that variant's frozen train-only CV config"}}
  out=OUT/mk;out.mkdir(parents=True,exist_ok=True)
  p=out/f"{variant}.json";p.write_text(json.dumps(doc,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")
  digest=sha(p)
  (out/f"{variant}.sha256").write_text(f"{digest}  {p.name}\n")
  print(f"{mk}/{variant} C={C:g} scorer_bundle_sha256={digest} baseline_bit_exact={variant=='v2-baseline'}")
