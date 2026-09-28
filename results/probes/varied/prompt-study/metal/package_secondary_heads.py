#!/usr/bin/env python3
"""Add run-mode/fixture provenance wrappers without changing secondary frozen head arrays."""
import hashlib,json,pathlib

ROOT=pathlib.Path(__file__).resolve().parents[5]
BASE=ROOT/"results/probes/varied/prompt-study/metal/frozen-scorers"
HOLD=ROOT/"results/probes/varied/prompt-study/runs/holdout-fresh60-prompt-study-20260924T000000Z"
HAND=HOLD/"secondary/variant-heads/gpu-variant-head-handoff.json"
doc=json.loads(HAND.read_text())
run_meta=json.loads((HOLD/"run-metadata.json").read_text())
OUT=BASE/"per-variant"
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
for mk,variants in doc["models"].items():
 for variant,record in variants.items():
  source=BASE/mk/f"{variant}.json"
  source_sha=sha(source)
  if source_sha!=record["head_sha256"]:raise SystemExit(f"secondary head SHA mismatch {mk}/{variant}")
  original=json.loads(source.read_text())
  if original["model"]!=record["model_id"] or float(original["C"])!=float(record["C"]):raise SystemExit(f"secondary model/C mismatch {mk}/{variant}")
  for kind,fields in record["normalization_and_weights"].items():
   spec=original["scorers"][kind]
   if len(spec["weights"])!=fields["dimension"]:raise SystemExit(f"dimension mismatch {mk}/{variant}/{kind}")
   for field in ("weights","mean","scale"):
    if sha_bytes:=hashlib.sha256(json.dumps(spec[field],separators=(",",":")).encode()).hexdigest()!=fields[field+"_sha256"]:
     raise SystemExit(f"secondary {field} fingerprint mismatch {mk}/{variant}/{kind}")
  # A wrapper adds provenance only; it reuses the exact original bytes for all model arrays.
  wrapped=dict(original)
  wrapped["head_mode"]="per-variant"
  wrapped["provenance"]={**original.get("provenance",{}),"source_bundle_sha256":source_sha,
    "fixture_sha256":doc["holdout_sha256"],"secondary_handoff_sha256":sha(HAND),
    "secondary_cpu_prediction_sha256":record["cpu_prediction_sha256"],
    "checkpoint_files":run_meta["models"][mk]["variants"][variant]["checkpoint_files"],
    "profile_sha256":run_meta["models"][mk]["variants"][variant]["profile_sha256"],
    "tokenizer_sha256":run_meta["models"][mk]["variants"][variant]["tokenizer_sha256"]}
  out=OUT/mk/f"{variant}.json";out.parent.mkdir(parents=True,exist_ok=True)
  out.write_text(json.dumps(wrapped,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")
  out.with_suffix(".sha256").write_text(f"{sha(out)}  {out.name}\n")
  print(f"{mk}/{variant} exact original head={source_sha}; wrapper={sha(out)}; C={record['C']}")
