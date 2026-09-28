#!/usr/bin/env python3
"""Package the exact frozen v2 head for each prompt text variant; no fitting."""
import hashlib,json,pathlib

ROOT=pathlib.Path(__file__).resolve().parents[5]
STUDY=ROOT/"results/probes/varied/prompt-study/runs/trainonly-clean300-20260924T002000Z"
HOLD=ROOT/"results/probes/varied/prompt-study/runs/holdout-fresh60-prompt-study-20260924T000000Z"
OUT=pathlib.Path(__file__).resolve().parent/"frozen-scorers/shared-v2-baseline"
VARIANTS=("v2-baseline","readable-state","rule-in-candidate","joint-context-candidate")
MODELS={"qwen":"Qwen/Qwen3-Embedding-0.6B","harrier":"microsoft/harrier-oss-v1-0.6b"}
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

meta=json.loads((HOLD/"run-metadata.json").read_text())
freeze=json.loads((HOLD/"PRE_REFERENCE_FREEZE.json").read_text())
if freeze.get("status")!="predictions_frozen_before_reference_scoring":raise SystemExit("holdout prediction freeze gate missing")
if sha(STUDY/"frozen-config.json")!=meta["frozen_config_sha256"]:raise SystemExit("frozen-config hash mismatch")
for mk,model in MODELS.items():
 source_path=STUDY/"frozen"/mk/"models.json"
 source=json.loads(source_path.read_text())
 selection_path=STUDY/"frozen"/mk/"selection.json"
 if source["model"]!=model or source["variant"]!="v2-baseline":raise SystemExit(f"wrong frozen model identity for {mk}")
 if source["C"]!=(1.0 if mk=="qwen" else 0.1):raise SystemExit(f"unexpected frozen v2 C for {mk}")
 if meta["models"][mk]["final_fit_sha256"]!=sha(source_path):raise SystemExit(f"CPU-reference final-fit hash differs for {mk}")
 if set(source["scorers"])!={"choice","noul","score"}:raise SystemExit(f"typed frozen scorers missing for {mk}")
 for variant in VARIANTS:
  key=f"{mk}/{variant}"
  result=meta["models"][mk]["variants"][variant]
  feature=HOLD/"features"/mk/variant/"fresh60.features.jsonl"
  manifest=pathlib.Path(str(feature)+".manifest.json")
  reference=HOLD/"predictions"/mk/f"{variant}.jsonl"
  if sha(feature)!=result["features_sha256"] or sha(manifest)!=result["manifest_sha256"] or sha(reference)!=result["predictions_sha256"]:
   raise SystemExit(f"fresh reference artifact hash mismatch at {key}")
  man=json.loads(manifest.read_text())
  if man["model"]!=model or man["variant"]!=variant or man["input_sha256"]!=meta["fixture_sha256"]:
   raise SystemExit(f"holdout feature identity mismatch at {key}")
  if sha(feature)!=freeze["feature_hashes"][key] or sha(reference)!=freeze["prediction_hashes"][key]:
   raise SystemExit(f"pre-reference freeze hash mismatch at {key}")
  # Preserve exact serialized frozen arrays; no cast, fit, scaler adjustment, or C override.
  scorers={kind:{"weights":source["scorers"][kind]["weights"],"mean":source["scorers"][kind]["mean"],"scale":source["scorers"][kind]["scale"]} for kind in ("choice","noul","score")}
  bundle={"model":model,"model_key":mk,"variant":variant,"head_mode":"shared-v2-baseline","C":source["C"],"scorers":scorers,
   "source_protocol":"prompt-only: identical train-frozen v2-baseline typed head and scaler reused for every text variant",
   "provenance":{"models_sha256":sha(source_path),"selection_sha256":sha(selection_path),"frozen_config_sha256":sha(STUDY/"frozen-config.json"),
     "holdout_run_metadata_sha256":sha(HOLD/"run-metadata.json"),"pre_reference_freeze_sha256":sha(HOLD/"PRE_REFERENCE_FREEZE.json"),
     "fixture_sha256":meta["fixture_sha256"],
     "variant_cv_config_sha256":sha(STUDY/"cv"/mk/variant/"frozen-config.json"),
     "fresh_feature_sha256":sha(feature),"fresh_manifest_sha256":sha(manifest),"cpu_reference_prediction_sha256":sha(reference),
     "profile_sha256":man["profile_sha256"],"tokenizer_sha256":man["tokenizer_sha256"],"checkpoint_files":man["checkpoint_files"]}}
  path=OUT/mk/f"{variant}.json";path.parent.mkdir(parents=True,exist_ok=True)
  path.write_text(json.dumps(bundle,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")
  h=sha(path);path.with_suffix(".sha256").write_text(f"{h}  {path.name}\n")
  print(f"{key}: shared C={source['C']} frozen_models_sha256={sha(source_path)} bundle_sha256={h}")
