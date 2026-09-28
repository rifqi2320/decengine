#!/usr/bin/env python3
"""Matched-head train/dev experiment: frozen Qwen3 vs final-layer LoRA.

One process handles exactly one model and one arm. It accepts no heldout/test paths.
The only data passed to the differentiable encoders are profile-compiled train texts.
Cached vectors are stripped by the shared train-data reader and are never inputs to
either arm. The head architecture, seed, objective, optimizer, order, and split are
identical; the sole treatment is whether final-block q/v LoRA is present.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map

HERE = Path(__file__).resolve().parent
LORA_DIR = HERE.parent
sys.path.insert(0, str(LORA_DIR))
import train as shared  # noqa: E402

ROOT = shared.ROOT
SEED = 271828
HEAD_INIT_SEED = 314159
LORA_INIT_SEED = 161803
MAX_EPOCHS = 9
MAX_STEPS = 9 * 460
REPORT_EPOCHS = (1, 3, 6, 9)
MAX_TOKENS = 288
SEQUENCE_BUCKETS = (64, 128, 192, 288)
PROCESS_STOP_BYTES = 6 * 1024 ** 3
PROCESS_HARD_BYTES = 8 * 1024 ** 3
SYSTEM_SWAP_LIMIT_BYTES = 4 * 1024 ** 3
MEMORY_SAMPLE_EVERY_STEP = 1
MODELS = {
    "Qwen/Qwen3-Embedding-0.6B": "qwen",
    "microsoft/harrier-oss-v1-0.6b": "harrier",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def swap_used_bytes() -> int:
    text = subprocess.run(["sysctl", "vm.swapusage"], text=True, capture_output=True, check=True).stdout
    import re
    match = re.search(r"used\s*=\s*([0-9.]+)M", text)
    if not match:
        raise RuntimeError(f"cannot parse vm.swapusage: {text}")
    return int(float(match.group(1)) * 1024 ** 2)


def memory_snapshot():
    text = subprocess.run(["footprint", "--format", "bytes", "--noCategories", str(__import__("os").getpid())],
                          text=True, capture_output=True, check=True).stdout
    import re
    current = re.search(r"Footprint:\s+([0-9,]+)\s+B", text)
    peak = re.search(r"phys_footprint_peak:\s+([0-9,]+)\s+B", text)
    if current is None or peak is None:
        raise RuntimeError(f"cannot parse process footprint: {text[:300]}")
    return {"pid": __import__("os").getpid(),
            "process_current_bytes": int(current.group(1).replace(",", "")),
            "process_peak_bytes": int(peak.group(1).replace(",", "")),
            "mlx_active_bytes": int(mx.get_active_memory()),
            "mlx_peak_bytes": int(mx.get_peak_memory()),
            "system_swap_used_bytes": swap_used_bytes()}


def check_memory(out: Path, phase: str, step: int, swap_baseline: int, extra=None):
    mx.clear_cache()
    gc.collect()
    snap = memory_snapshot()
    row = {"phase": phase, "step": step, "swap_baseline_bytes": swap_baseline, **snap}
    if extra:
        row.update(extra)
    with (out / "memory.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    if snap["process_peak_bytes"] >= PROCESS_HARD_BYTES:
        raise MemoryError(f"absolute 8 GiB process cap reached: {row}")
    if snap["process_peak_bytes"] >= PROCESS_STOP_BYTES:
        raise MemoryError(f"6 GiB early-stop reached: {row}")
    if snap["system_swap_used_bytes"] > swap_baseline:
        raise MemoryError(f"system swap increased from run start: {row}")
    if snap["system_swap_used_bytes"] > SYSTEM_SWAP_LIMIT_BYTES:
        raise MemoryError(f"system swap exceeded 4 GiB cap: {row}")
    return row


def init_row_matches(reference_path: Path | None, row: dict):
    if reference_path is None:
        return
    ref = json.loads(reference_path.read_text(encoding="utf-8"))
    if ref["sample_key"] != row["sample_key"]:
        raise ValueError("matched-arm step-0 sample differs")
    if ref["head_init_sha256"] != row["head_init_sha256"]:
        raise ValueError("matched arms do not have byte-identical initial typed heads")
    left = np.asarray(ref["logits"], dtype=np.float64)
    right = np.asarray(row["logits"], dtype=np.float64)
    if left.shape != right.shape or not np.allclose(left, right, rtol=0, atol=1e-6):
        raise ValueError(f"matched step-0 logits differ: max_abs={float(np.max(np.abs(left-right)))}")
    row["reference_initial_logits_sha256"] = sha(reference_path)


def eval_dev(model, cases, indices, tokenizer, out, swap_baseline, first_step):
    stats = {kind: {"n": 0, "correct": 0, "losses": [], "confidence": [],
                    "predicted_candidate_index_hist": {}} for kind in ("choice", "noul", "score")}
    score_errors = []
    for local_step, index in enumerate(indices, 1):
        case = cases[index]
        tokens, lengths = shared.batch_inputs(case, tokenizer)
        logits = model(tokens, lengths, len(case["keys"]), case["kind"])
        loss = shared.preference_loss(logits, case)
        probs = mx.softmax(logits.astype(mx.float32), axis=0)
        mx.eval(loss, probs, logits)
        arr = np.asarray(logits, dtype=np.float64)
        prob = np.asarray(probs, dtype=np.float64)
        pred = int(np.argmax(arr))
        s = stats[case["kind"]]
        s["n"] += 1
        s["correct"] += int(pred == case["target"])
        s["losses"].append(float(np.asarray(loss)))
        s["confidence"].append(float(prob[pred]))
        rank = str(pred)
        s["predicted_candidate_index_hist"][rank] = s["predicted_candidate_index_hist"].get(rank, 0) + 1
        if case["kind"] == "score":
            values = np.asarray(case["values"], dtype=np.float64)
            span = float(values.max() - values.min())
            if span > 0:
                expected = float(np.dot(prob, values))
                score_errors.append(abs(expected - values[case["target"]]) / span)
        max_tokens = int(np.max(np.asarray(lengths)))
        batch_width = int(tokens.shape[1])
        del tokens, lengths, logits, loss, probs
        mem = check_memory(out, "dev", first_step + local_step, swap_baseline,
                           {"case": case["id"], "question": case["name"],
                            "candidate_count": len(case["keys"]), "max_text_tokens": max_tokens,
                            "batch_width": batch_width})
        del mem
    by_type = {}
    for kind, s in stats.items():
        if not s["n"]:
            raise ValueError(f"dev split is missing question type {kind}")
        by_type[kind] = {
            "n": s["n"], "exact_accuracy": s["correct"] / s["n"],
            "mean_objective_loss": float(np.mean(s["losses"])),
            "mean_selected_confidence": float(np.mean(s["confidence"])),
            "predicted_candidate_index_hist": s["predicted_candidate_index_hist"],
        }
    macro = float(np.mean([by_type[k]["exact_accuracy"] for k in ("choice", "noul", "score")]))
    mean_loss = float(np.mean([x for s in stats.values() for x in s["losses"]]))
    return {"n": len(indices), "per_type": by_type,
            "primary_macro_exact_accuracy": macro,
            "secondary_mean_objective_loss": mean_loss,
            "score_normalized_expected_value_mae": float(np.mean(score_errors)) if score_errors else None}


def load_adapter(adapter_path: Path):
    spec = importlib.util.spec_from_file_location("matched_head_adapter", adapter_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--arm", required=True, choices=("head", "lora"))
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--reference-init", type=Path,
                   help="step-0 JSON from the same model's head-arm smoke; required for the LoRA arm")
    args = p.parse_args()
    if args.smoke_only and args.resume:
        raise ValueError("--smoke-only cannot be combined with --resume")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()) and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.out}")
    if args.arm == "lora" and args.reference_init is None:
        raise ValueError("LoRA arm requires --reference-init from its matched frozen-head smoke")

    model_tag = MODELS[args.model]
    clean_path, multi_path = shared.DEFAULT_CLEAN, shared.DEFAULT_MULTI
    _, clean_features, multi_features = shared.MODEL_CONFIGS[args.model]
    # The only rows admitted are the exact existing clean300 + multi train120 set.
    _, cases = shared.train_rows(clean_path, multi_path, clean_features, multi_features)
    train_ix, dev_ix, split = shared.grouped_split(cases)
    if len(cases) != 660 or len(train_ix) != 460 or len(dev_ix) != 200:
        raise ValueError("expected exact grouped 460 train / 200 dev split")
    freeze = json.loads((LORA_DIR / "SELECTION-FREEZE.json").read_text(encoding="utf-8"))
    frozen = freeze["models"][model_tag]["grouped_split"]
    got_train = hashlib.sha256("\n".join(sorted(split["train_case_ids"])).encode()).hexdigest()
    got_dev = hashlib.sha256("\n".join(sorted(split["dev_case_ids"])).encode()).hexdigest()
    if got_train != frozen["train_case_id_set_sha256"] or got_dev != frozen["dev_case_id_set_sha256"]:
        raise ValueError("matched experiment case split differs from frozen LoRA split")
    profile_features = Path(clean_features)
    feature_manifest = json.loads(Path(str(profile_features) + ".manifest.json").read_text(encoding="utf-8"))
    for c in cases:
        if any(n > MAX_TOKENS for n in c["compiled_token_lengths"]):
            raise ValueError("profile-compiled train text exceeds reject-only 288 token cap")
    start_swap = swap_used_bytes()

    adapter_path = LORA_DIR / "checkpoint_adapter.py"
    adapter = load_adapter(adapter_path)
    config_dir = shared.MODEL_CONFIGS[args.model][0]
    model_dir = shared.DEFAULT_STORE / "models" / config_dir
    loaded = adapter.load_checkpoint(model_dir)
    parity = shared.compiled_text_parity(loaded, profile_features, args.model)
    if not parity.get("passed"):
        raise RuntimeError("compiled profile-text/Rust parity gate failed")
    backbone = loaded.model
    if loaded.checkpoint_dtype != mx.bfloat16:
        raise TypeError(f"expected BF16 base, got {loaded.checkpoint_dtype}")
    if any(v.dtype != mx.bfloat16 for _, v in tree_flatten(backbone.parameters())):
        raise TypeError("base decoder must remain BF16")

    # Seed the generic head independently of adapter initialization so both arms
    # receive exactly identical FP32 head tensors.
    mx.random.seed(HEAD_INIT_SEED)
    head = shared.TypedScorer()
    head.set_dtype(mx.float32)
    head_hash = hashlib.sha256()
    head_path = args.out / "initial-head.safetensors"
    shared.save_safetensors_atomic(head_path, head.parameters())
    head_hash_value = sha(head_path)
    if args.arm == "head":
        backbone.freeze()
        if sum(int(v.size) for _, v in tree_flatten(backbone.trainable_parameters())) != 0:
            raise AssertionError("head-only arm unexpectedly has trainable backbone parameters")
        lora_modules = []
    else:
        mx.random.seed(LORA_INIT_SEED)
        lora_modules = adapter.inject_upper_lora(loaded, layers=1, projections=("q_proj", "v_proj"), rank=2, alpha=4.0)
        if any("lora_" not in str(k) or v.dtype != mx.float32 for k, v in tree_flatten(backbone.trainable_parameters())):
            raise AssertionError("LoRA arm has non-LoRA or non-FP32 trainable backbone params")
    restore_checkpointing = shared.enable_upper_layer_checkpointing(backbone, 1) if args.arm == "lora" else (lambda: None)
    model = shared.PreferenceModel(backbone, head)
    if sum(int(v.size) for _, v in tree_flatten(model.trainable_parameters())) == 0:
        raise RuntimeError("no trainable head/LoRA parameters")

    sample_index = train_ix[0]
    sample = cases[sample_index]
    token_batch, lengths = shared.batch_inputs(sample, loaded.tokenizer)
    initial_logits = model(token_batch, lengths, len(sample["keys"]), sample["kind"])
    mx.eval(initial_logits)
    initial_record = {"arm": args.arm, "model": args.model, "sample_key": [sample["id"], sample["name"]],
                      "query_sha256": hashlib.sha256(sample["query"].encode()).hexdigest(),
                      "candidate_text_sha256": [hashlib.sha256(x.encode()).hexdigest() for x in sample["candidate_texts"]],
                      "head_init_sha256": head_hash_value, "logits": np.asarray(initial_logits).tolist(),
                      "parity_report": parity, "profile_version": feature_manifest["profile_version"]}
    init_row_matches(args.reference_init, initial_record)
    (args.out / "initial.json").write_text(json.dumps(initial_record, indent=2, sort_keys=True) + "\n")
    del token_batch, lengths, initial_logits

    config = {
        "schema_version": 1, "model": args.model, "arm": args.arm,
        "seed": SEED, "head_init_seed": HEAD_INIT_SEED, "lora_init_seed": LORA_INIT_SEED,
        "max_epochs": MAX_EPOCHS, "steps_per_epoch": 460, "max_steps": MAX_STEPS,
        "report_epochs": list(REPORT_EPOCHS), "train_questions": 460, "dev_questions": 200,
        "split": split, "selection": "maximize equal-weight macro per-type exact accuracy; tie-break lower dev mean objective loss, then lower score normalized expected-value MAE, then earlier epoch",
        "optimizer": "AdamW", "learning_rate": shared.LEARNING_RATE, "weight_decay": shared.WEIGHT_DECAY,
        "gradient_clip_norm": shared.GRAD_CLIP, "gradient_accumulation_steps": 1,
        "micro_batch_questions": 1, "max_tokens": MAX_TOKENS, "sequence_buckets": list(SEQUENCE_BUCKETS),
        "overflow_policy": "reject; no truncation", "loss": "same shared.preference_loss for both arms",
        "head": {"architecture": "shared 4096 -> 192 GELU -> typed scalar choice/noul/score", "dtype": "FP32",
                 "seed": HEAD_INIT_SEED, "sha256": head_hash_value},
        "lora": {"enabled": args.arm == "lora", "layers": 1 if args.arm == "lora" else 0,
                 "targets": lora_modules, "rank": 2, "alpha": 4.0,
                 "base_dtype": "BF16", "adapter_dtype": "FP32", "rematerialized_blocks": 1 if args.arm == "lora" else 0},
        "initial_logits_sha256": sha(args.out / "initial.json"),
        "reference_init_sha256": sha(args.reference_init) if args.reference_init else None,
        "profile_version": feature_manifest["profile_version"], "profile_sha256": feature_manifest["profile_sha256"],
        "tokenizer_sha256": feature_manifest["tokenizer_sha256"], "text_version": feature_manifest["text_version"],
        "source_sha256": {"trainer": sha(Path(__file__)), "shared_trainer": sha(LORA_DIR / "train.py"),
          "adapter": sha(adapter_path), "clean300": sha(clean_path), "multi_train120": sha(multi_path),
          "clean_train_features": sha(clean_features), "multi_train_features": sha(shared.MODEL_CONFIGS[args.model][2]),
          "base_checkpoint": sha(model_dir / "model.safetensors")},
        "resource": {"process_stop_bytes": PROCESS_STOP_BYTES, "absolute_process_cap_bytes": PROCESS_HARD_BYTES,
                     "system_swap_must_not_increase": True, "system_swap_limit_bytes": SYSTEM_SWAP_LIMIT_BYTES,
                     "memory_sample_every_step": True, "host_memory_bytes": 24*1024**3},
    }
    config_path = args.out / "config.json"
    if args.resume:
        if not config_path.is_file() or json.loads(config_path.read_text()) != config:
            raise ValueError("resume config mismatch; refusing to mix matched-run provenance")
    else:
        if any(p.name not in ("initial-head.safetensors", "initial.json") for p in args.out.iterdir()):
            raise FileExistsError("nonempty matched run output directory")
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    optimizer = optim.AdamW(learning_rate=shared.LEARNING_RATE, weight_decay=shared.WEIGHT_DECAY)
    logs = args.out / "steps.jsonl"
    epochs_file = args.out / "epochs.jsonl"
    latest_path = args.out / "latest.safetensors"
    latest_meta_path = args.out / "latest.json"
    best_path = args.out / "best.safetensors"
    best_score = None
    best_epoch = None
    best_metrics = None
    global_step = 0
    completed_epoch = 0
    if args.resume:
        latest_meta = json.loads(latest_meta_path.read_text())
        completed_epoch = int(latest_meta["completed_epoch"])
        global_step = int(latest_meta["step"])
        model.load_weights(list(mx.load(str(latest_path)).items()), strict=False)
        optimizer.state = shared.restore_optimizer_state(args.out / "optimizer.safetensors")
        # Keep only committed update/epoch records after restart.
        kept=[line for line in logs.read_text().splitlines() if line.strip() and int(json.loads(line)["step"])<=global_step]
        logs.write_text("\n".join(kept)+("\n" if kept else ""))
        erows=[json.loads(x) for x in epochs_file.read_text().splitlines() if x.strip() and int(json.loads(x)["epoch"])<=completed_epoch]
        epochs_file.write_text("\n".join(json.dumps(x,sort_keys=True) for x in erows)+("\n" if erows else ""))
        if (args.out/"best.json").is_file():
            best=json.loads((args.out/"best.json").read_text());best_score=best["selection_key"];best_epoch=best["epoch"];best_metrics=best["metrics"]
    else:
        shared.save_safetensors_atomic(latest_path, model.trainable_parameters())
        latest_meta_path.write_text(json.dumps({"completed_epoch":0,"step":0},indent=2)+"\n")
    mx.reset_peak_memory()
    start_memory = check_memory(args.out, "startup", global_step, start_swap,
                                {"model": args.model, "arm": args.arm})
    if not args.resume and start_memory["system_swap_used_bytes"] != start_swap:
        raise MemoryError(f"swap changed during startup: {start_memory}")

    rng_order=[]
    for epoch0 in range(completed_epoch, MAX_EPOCHS):
        epoch=epoch0+1
        order=list(train_ix);random.Random(SEED+epoch0).shuffle(order)
        train_losses={k:[] for k in ("choice","noul","score")}
        train_grad_head=[];train_grad_lora=[]
        for idx in order:
            case=cases[idx];tokens,lengths=shared.batch_inputs(case,loaded.tokenizer)
            batch_width=int(tokens.shape[1]); token_sum=int(np.asarray(mx.sum(lengths)))
            token_max=int(np.asarray(mx.max(lengths)))
            def loss_fn(m):return shared.preference_loss(m(tokens,lengths,len(case["keys"]),case["kind"]),case)
            value_grad=nn.value_and_grad(model,loss_fn)
            loss,grads=value_grad(model);mx.eval(loss,grads)
            lv=float(np.asarray(loss))
            if not math.isfinite(lv):raise FloatingPointError(f"nonfinite loss at step {global_step}")
            flat=tree_flatten(grads)
            hnorm=math.sqrt(sum(float(np.asarray(mx.sum(v.astype(mx.float32)**2))) for k,v in flat if str(k).startswith("head.")))
            lnorm=math.sqrt(sum(float(np.asarray(mx.sum(v.astype(mx.float32)**2))) for k,v in flat if "lora_" in str(k)))
            if hnorm<=0 or not math.isfinite(hnorm):raise RuntimeError("typed-head gradient is zero/nonfinite")
            if args.arm=="lora" and (lnorm<=0 or not math.isfinite(lnorm)):
                raise RuntimeError("LoRA gradient is zero/nonfinite")
            if args.arm=="head" and lnorm!=0:
                raise RuntimeError("head-only arm unexpectedly has LoRA gradient")
            # Match the shared trainer's global L2 norm clip over the complete
            # trainable pytree, rather than adding group norms (which would make
            # LoRA and head-only arms use different clipping geometry).
            total_norm=math.sqrt(hnorm*hnorm+lnorm*lnorm)
            scale=min(1.0,shared.GRAD_CLIP/max(total_norm,1e-12))
            if scale!=1:grads=tree_map(lambda x:x*scale,grads)
            optimizer.update(model,grads);mx.eval(model.parameters(),optimizer.state)
            if global_step==0:
                after=loss_fn(model);mx.eval(after);after_value=float(np.asarray(after))
                if not math.isfinite(after_value) or after_value>=lv:raise RuntimeError(f"first minibatch loss did not decrease: {lv}->{after_value}")
            else:after_value=None
            global_step+=1;train_losses[case["kind"]].append(lv);train_grad_head.append(hnorm);train_grad_lora.append(lnorm)
            with logs.open("a",encoding="utf8") as f:
                f.write(json.dumps({"step":global_step,"epoch":epoch,"arm":args.arm,"case":case["id"],"question":case["name"],
                 "type":case["kind"],"candidate_count":len(case["keys"]),"batch_rows":len(case["keys"])+1,
                 "batch_width":batch_width,"max_text_tokens":token_max,"sum_text_tokens":token_sum,
                 "loss":lv,"head_grad_norm":hnorm,"lora_grad_norm":lnorm,"post_update_loss":after_value,"clip_scale":scale},sort_keys=True)+"\n")
            del loss,grads,flat,value_grad,loss_fn,tokens,lengths
            if "after" in locals():del after
            mem=check_memory(args.out,"train",global_step,start_swap,{"epoch":epoch,"arm":args.arm,
             "candidate_count":len(case["keys"]),"batch_width":batch_width,"max_text_tokens":token_max})
            if args.smoke_only:
                shared.save_safetensors_atomic(args.out/"smoke.safetensors",model.trainable_parameters())
                smoke={"status":"gradient_smoke_passed","model":args.model,"arm":args.arm,"step":global_step,
                 "loss_before":lv,"loss_after":after_value,"head_grad_norm":hnorm,"lora_grad_norm":lnorm,
                 "head_init_sha256":head_hash_value,"initial_logits_sha256":sha(args.out/"initial.json"),
                 "process_peak_bytes":mem["process_peak_bytes"],"system_swap_used_bytes":mem["system_swap_used_bytes"]}
                (args.out/"smoke.json").write_text(json.dumps(smoke,indent=2,sort_keys=True)+"\n")
                restore_checkpointing()
                print(json.dumps(smoke,sort_keys=True))
                return
            if global_step % len(order)==0:
                break
        # Epoch completion is exactly 460 training questions; an interrupted partial
        # epoch never produces dev-selection/checkpoint output.
        if len(train_losses["choice"])+len(train_losses["noul"])+len(train_losses["score"])!=len(order):
            restore_checkpointing();return
        dev=eval_dev(model,cases,dev_ix,loaded.tokenizer,args.out,start_swap,global_step)
        train_by_type={k:{"n":len(v),"mean_loss":float(np.mean(v))} for k,v in train_losses.items()}
        epoch_row={"epoch":epoch,"global_step":global_step,"train_mean_objective_loss_by_type":train_by_type,
         "mean_head_grad_norm":float(np.mean(train_grad_head)),"mean_lora_grad_norm":float(np.mean(train_grad_lora)) if args.arm=="lora" else 0.0,
         "dev":dev}
        with epochs_file.open("a",encoding="utf8") as f:f.write(json.dumps(epoch_row,sort_keys=True)+"\n")
        # Immutable per-epoch weights preserve the full learning trajectory; the
        # separate best pointer is selected only by the predeclared dev macro rule.
        shared.save_safetensors_atomic(args.out/f"epoch-{epoch:02d}.safetensors",model.trainable_parameters())
        score_key=(dev["primary_macro_exact_accuracy"],-dev["secondary_mean_objective_loss"],
                   -(dev["score_normalized_expected_value_mae"] if dev["score_normalized_expected_value_mae"] is not None else math.inf),-epoch)
        if best_score is None or score_key>tuple(best_score):
            best_score=score_key;best_epoch=epoch;best_metrics=dev
            shared.save_safetensors_atomic(best_path,model.trainable_parameters())
            (args.out/"best.json").write_text(json.dumps({"epoch":epoch,"selection_key":list(score_key),"metrics":dev},indent=2,sort_keys=True)+"\n")
        shared.save_safetensors_atomic(latest_path,model.trainable_parameters())
        shared.save_safetensors_atomic(args.out/"optimizer.safetensors",dict(tree_flatten(optimizer.state)))
        latest_meta_path.write_text(json.dumps({"completed_epoch":epoch,"step":global_step},indent=2)+"\n")
        check_memory(args.out,"epoch_boundary",global_step,start_swap,{"epoch":epoch})
        if epoch in REPORT_EPOCHS:
            (args.out/f"report-epoch-{epoch}.json").write_text(json.dumps(epoch_row,indent=2,sort_keys=True)+"\n")
    restore_checkpointing()
    summary={"model":args.model,"arm":args.arm,"status":"complete","epochs":MAX_EPOCHS,"steps":global_step,
     "best_epoch":best_epoch,"best_metrics":best_metrics,"scope":"same grouped train/dev only; no heldout accessed"}
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print(json.dumps(summary,sort_keys=True))


if __name__=="__main__":
    main()
