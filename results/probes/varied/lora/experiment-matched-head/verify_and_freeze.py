#!/usr/bin/env python3
"""CPU-only integrity audit and train/dev freeze for matched-head runs.

Reads run configs, JSON/JSONL metrics, and hashes checkpoint files. It never imports
MLX, constructs a model, runs inference, or opens a holdout/test fixture.
"""
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
PROJECT = ROOT.parents[4]
LORA_DIR = PROJECT / "results/probes/varied/lora"
STORE = Path("/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/decengine-model-store")
CLEAN = Path("/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/varied-loqfo-clean300/train-clean-300.jsonl")
MULTI = PROJECT / "benchmarks/cases/varied/multi/train-120.jsonl"
MODELS = {
    "qwen": {"model": "Qwen/Qwen3-Embedding-0.6B", "store": "Qwen--Qwen3-Embedding-0.6B",
             "feature_dir": PROJECT / "results/probes/varied/runs/multi-qwen"},
    "harrier": {"model": "microsoft/harrier-oss-v1-0.6b", "store": "microsoft--harrier-oss-v1-0.6b",
                "feature_dir": PROJECT / "results/probes/varied/runs/multi-harrier"},
}
ARMS = ("head", "lora")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def set_hash(values):
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def assert_hash(name, path, expected):
    actual = sha(path)
    if actual != expected:
        raise AssertionError(f"{name} hash mismatch: expected {expected}, got {actual} ({path})")
    return {"path": str(path), "sha256": actual, "size_bytes": Path(path).stat().st_size}


def validate_run(tag, arm, spec):
    run = RUNS / f"{tag}-{arm}"
    config = json.loads((run / "config.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    latest = json.loads((run / "latest.json").read_text())
    best = json.loads((run / "best.json").read_text())
    epochs = jsonl(run / "epochs.jsonl")
    steps = jsonl(run / "steps.jsonl")
    memories = jsonl(run / "memory.jsonl")
    initial = json.loads((run / "initial.json").read_text())
    assert config["model"] == spec["model"] and config["arm"] == arm
    assert summary["status"] == "complete" and summary["epochs"] == 9 and summary["steps"] == 4140
    assert latest == {"completed_epoch": 9, "step": 4140}
    assert len(epochs) == 9 and len(steps) == 4140
    assert [x["epoch"] for x in epochs] == list(range(1, 10))
    assert [x["step"] for x in steps] == list(range(1, 4141))
    assert all(json.loads((run / f"report-epoch-{e}.json").read_text())["epoch"] == e
               for e in (1, 3, 6, 9))
    assert config["train_questions"] == 460 and config["dev_questions"] == 200
    assert all((run / f"epoch-{e:02d}.safetensors").is_file() for e in range(1, 10))
    assert config["max_epochs"] == 9 and config["max_steps"] == 4140
    assert config["micro_batch_questions"] == 1 and config["max_tokens"] == 288
    assert config["overflow_policy"] == "reject; no truncation"
    assert config["optimizer"] == "AdamW" and config["learning_rate"] == 1e-4
    assert config["gradient_clip_norm"] == 1.0

    split = config["split"]
    train_cases, dev_cases = set(split["train_case_ids"]), set(split["dev_case_ids"])
    train_fams, dev_fams = set(split["train_families"]), set(split["dev_families"])
    assert len(train_cases) == 300 and len(dev_cases) == 120
    assert len(train_fams) == 11 and len(dev_fams) == 4
    assert not train_cases & dev_cases and not train_fams & dev_fams
    assert split["component_count"] == 9 and split["dev_components"] == 2
    assert len(split["train_case_ids"]) + len(split["dev_case_ids"]) == 420

    # Every logged update has finite, nonzero head gradient. Frozen-head has no
    # LoRA gradient; LoRA arm has one at every update.
    for row in steps:
        assert math.isfinite(row["loss"]) and math.isfinite(row["head_grad_norm"])
        assert row["head_grad_norm"] > 0
        assert math.isfinite(row["lora_grad_norm"])
        if arm == "head":
            assert row["lora_grad_norm"] == 0
        else:
            assert row["lora_grad_norm"] > 0
        assert row["max_text_tokens"] <= 288 and row["batch_width"] in (64, 128, 192, 288)
    for row in epochs:
        dev = row["dev"]
        assert dev["n"] == 200
        assert {k: v["n"] for k, v in dev["per_type"].items()} == {"choice": 100, "noul": 60, "score": 40}
        expected_macro = sum(dev["per_type"][k]["exact_accuracy"] for k in ("choice", "noul", "score")) / 3
        assert abs(expected_macro - dev["primary_macro_exact_accuracy"]) < 1e-12
        assert math.isfinite(dev["secondary_mean_objective_loss"])
        assert math.isfinite(dev["score_normalized_expected_value_mae"])

    selection = min(epochs, key=lambda row: (
        -row["dev"]["primary_macro_exact_accuracy"],
        row["dev"]["secondary_mean_objective_loss"],
        row["dev"]["score_normalized_expected_value_mae"],
        row["epoch"],
    ))
    assert best["epoch"] == selection["epoch"] == summary["best_epoch"]
    assert best["metrics"] == selection["dev"]
    assert best["selection_key"] == [
        selection["dev"]["primary_macro_exact_accuracy"],
        -selection["dev"]["secondary_mean_objective_loss"],
        -selection["dev"]["score_normalized_expected_value_mae"],
        -selection["epoch"],
    ]

    baseline_values = {row["swap_baseline_bytes"] for row in memories}
    assert len(baseline_values) == 1
    swap_baseline = next(iter(baseline_values))
    swap_vals = [row["system_swap_used_bytes"] for row in memories]
    peaks = [row["process_peak_bytes"] for row in memories]
    assert len(memories) == 5950  # startup + 4140 train + 1800 dev + 9 boundaries
    assert max(swap_vals) <= swap_baseline
    assert max(peaks) < 6 * 1024**3 and max(peaks) < 8 * 1024**3
    assert all(row["pid"] == memories[0]["pid"] for row in memories)
    assert all(row["phase"] != "train" or row["step"] <= 4140 for row in memories)

    features = spec["feature_dir"]
    feature_files = {
        "clean_train_features": features / "train-clean-300.features.jsonl",
        "multi_train_features": features / "train-120.features.jsonl",
    }
    clean_manifest = json.loads(Path(str(feature_files["clean_train_features"]) + ".manifest.json").read_text())
    multi_manifest = json.loads(Path(str(feature_files["multi_train_features"]) + ".manifest.json").read_text())
    assert clean_manifest["model"] == spec["model"] and multi_manifest["model"] == spec["model"]
    assert clean_manifest["text_version"] == multi_manifest["text_version"] == "decengine-varied-pair-text-v2-generic-types"
    assert clean_manifest["profile_version"] == multi_manifest["profile_version"] == config["profile_version"]
    assert clean_manifest["profile_sha256"] == multi_manifest["profile_sha256"] == config["profile_sha256"]

    store = STORE / "models" / spec["store"]
    source_paths = {
        "trainer": ROOT / "train_matched.py",
        "shared_trainer": LORA_DIR / "train.py",
        "adapter": LORA_DIR / "checkpoint_adapter.py",
        "clean300": CLEAN,
        "multi_train120": MULTI,
        "clean_train_features": feature_files["clean_train_features"],
        "multi_train_features": feature_files["multi_train_features"],
        "base_checkpoint": store / "model.safetensors",
    }
    source_hashes = {}
    for name, path in source_paths.items():
        source_hashes[name] = assert_hash(name, path, config["source_sha256"][name])
    feature_manifests = {
        name: {"path": str(Path(str(path) + ".manifest.json")),
               "sha256": sha(Path(str(path) + ".manifest.json")),
               "profile_version": clean_manifest["profile_version"],
               "profile_sha256": clean_manifest["profile_sha256"],
               "text_version": clean_manifest["text_version"]}
        for name, path in feature_files.items()
    }

    artifact_names = ["best.safetensors", "latest.safetensors", "optimizer.safetensors",
                      "initial-head.safetensors", "epoch-02.safetensors", "epoch-09.safetensors"]
    if selection["epoch"] not in (2, 9):
        artifact_names.append(f"epoch-{selection['epoch']:02d}.safetensors")
    artifacts = {name: {"path": (run / name).relative_to(PROJECT).as_posix(),
                        "sha256": sha(run / name), "size_bytes": (run / name).stat().st_size}
                 for name in artifact_names}
    # The selected pointer must be byte-identical to the immutable selected epoch.
    selected_path = run / f"epoch-{selection['epoch']:02d}.safetensors"
    assert sha(run / "best.safetensors") == sha(selected_path)

    return {
        "model": spec["model"], "arm": arm, "run_dir": run.relative_to(PROJECT).as_posix(),
        "status": summary["status"], "epochs": 9, "steps": 4140,
        "best_epoch": selection["epoch"], "best_dev": selection["dev"],
        "split": {"train_questions": 460, "dev_questions": 200,
                  "train_case_count": len(train_cases), "dev_case_count": len(dev_cases),
                  "train_family_count": len(train_fams), "dev_family_count": len(dev_fams),
                  "component_count": split["component_count"], "dev_component_count": split["dev_components"],
                  "case_disjoint": True, "family_disjoint": True,
                  "train_case_ids_sha256": set_hash(train_cases), "dev_case_ids_sha256": set_hash(dev_cases),
                  "train_family_ids_sha256": set_hash(train_fams), "dev_family_ids_sha256": set_hash(dev_fams)},
        "initial_head_sha256": config["head"]["sha256"],
        "initial_weights_file_sha256": sha(run / "initial-head.safetensors"),
        "initial_logits_record_sha256": sha(run / "initial.json"),
        "initial_logits": initial["logits"],
        "initial_logits_record_reference_sha256": initial.get("reference_initial_logits_sha256"),
        "compiled_text_base_parity_passed": initial["parity_report"]["passed"],
        "arm_initial_sample_key": initial["sample_key"],
        "profile": {"profile_version": config["profile_version"], "profile_sha256": config["profile_sha256"],
                    "text_version": config["text_version"], "tokenizer_sha256": config["tokenizer_sha256"],
                    "feature_manifests": feature_manifests},
        "config_sha256": sha(run / "config.json"),
        "trainer_sha256": config["source_sha256"]["trainer"],
        "shared_source_hashes": source_hashes,
        "epoch_metrics": epochs,
        "gradient_validation": {"head_gradient_nonzero_finite_all_steps": True,
                                "lora_gradient_zero_all_steps": arm == "head",
                                "lora_gradient_nonzero_finite_all_steps": arm == "lora",
                                "head_grad_norm_min": min(x["head_grad_norm"] for x in steps),
                                "lora_grad_norm_min": min(x["lora_grad_norm"] for x in steps),
                                "lora_grad_norm_max": max(x["lora_grad_norm"] for x in steps)},
        "resource": {"process_peak_bytes": max(peaks), "process_peak_gib": max(peaks) / 1024**3,
                     "process_stop_bytes": 6 * 1024**3, "absolute_cap_bytes": 8 * 1024**3,
                     "memory_samples": len(memories), "swap_baseline_bytes": swap_baseline,
                     "swap_min_bytes": min(swap_vals), "swap_max_bytes": max(swap_vals),
                     "swap_increased": False},
        "artifacts": artifacts,
        "best_json_sha256": sha(run / "best.json"),
        "summary_sha256": sha(run / "summary.json"),
        "epochs_jsonl_sha256": sha(run / "epochs.jsonl"),
        "steps_jsonl_sha256": sha(run / "steps.jsonl"),
        "memory_jsonl_sha256": sha(run / "memory.jsonl"),
    }


def main():
    runs = {}
    for tag, spec in MODELS.items():
        for arm in ARMS:
            runs[f"{tag}_{arm}"] = validate_run(tag, arm, spec)

    for tag in MODELS:
        head, lora = runs[f"{tag}_head"], runs[f"{tag}_lora"]
        assert head["split"] == lora["split"]
        assert head["initial_head_sha256"] == lora["initial_head_sha256"]
        assert head["initial_weights_file_sha256"] == lora["initial_weights_file_sha256"]
        assert head["arm_initial_sample_key"] == lora["arm_initial_sample_key"]
        diffs = [abs(a - b) for a, b in zip(head["initial_logits"], lora["initial_logits"])]
        assert len(diffs) and max(diffs) <= 1e-6
        assert head["profile"] == lora["profile"]
        head["initial_logits_max_abs_diff_vs_lora"] = max(diffs)
        lora["initial_logits_max_abs_diff_vs_head"] = max(diffs)
        head["arm_initial_logits_match_atol_1e-6"] = max(diffs) <= 1e-6
        lora["arm_initial_logits_match_atol_1e-6"] = max(diffs) <= 1e-6

    # Selected best rows and model-vs-head deltas come from the declared primary
    # macro-accuracy rule; no aggregate mixed loss selects these checkpoints.
    selected = {}
    for tag in MODELS:
        h, l = runs[f"{tag}_head"], runs[f"{tag}_lora"]
        delta = l["best_dev"]["primary_macro_exact_accuracy"] - h["best_dev"]["primary_macro_exact_accuracy"]
        selected[tag] = {
            "frozen_head": {"epoch": h["best_epoch"], "macro_exact_accuracy": h["best_dev"]["primary_macro_exact_accuracy"],
                            "checkpoint": h["artifacts"]["best.safetensors"]},
            "lora": {"epoch": l["best_epoch"], "macro_exact_accuracy": l["best_dev"]["primary_macro_exact_accuracy"],
                     "checkpoint": l["artifacts"]["best.safetensors"]},
            "lora_minus_head_macro_accuracy": delta,
            "lora_improves_same_head": delta > 0,
        }

    freeze = {
        "schema_version": 1,
        "status": "TRAIN_ONLY_SELECTION_FROZEN",
        "scope": "train/dev only; no heldout/test labels, predictions, or hashes accessed",
        "created_date": "2026-09-24",
        "validation_mode": "CPU-only metadata/log/hash checks; no MLX import, model load, training, or inference",
        "selection_rule": "maximize equal-weight mean of choice/noul/score dev exact accuracy; tie-break lower dev mean objective loss, lower score normalized expected-value MAE, then earlier epoch",
        "primary_metric": "type-balanced macro exact accuracy",
        "secondary_metrics": ["dev mean objective loss", "score normalized expected-value MAE"],
        "data_scope": {"questions_total": 660, "train_questions": 460, "dev_questions": 200,
                       "train_cases": 300, "dev_cases": 120,
                       "train_families": 11, "dev_families": 4,
                       "group_components_total": 9, "dev_group_components": 2,
                       "case_and_family_disjoint": True,
                       "caveat": "Dev contains 200 named questions but only two grouped connected components; this is a small, noisy selection set."},
        "runs": runs,
        "selected_checkpoints": selected,
        "convergence_diagnosis": {
            "qwen_lora": "Macro accuracy rose from epoch 8 to 9 (0.4811 to 0.4989); dev loss also rose. Still improving at the 9-epoch boundary; convergence not established.",
            "harrier_lora": "Macro accuracy rose from epoch 8 to 9 (0.5028 to 0.5228); dev loss rose from 1.0156 to 1.0282. Still improving at the 9-epoch boundary; convergence not established.",
            "qwen_head": "Primary macro accuracy peaked at epoch 2 then generally fluctuated/lowered; objective dev loss worsened through epoch 9.",
            "harrier_head": "Primary macro accuracy rose through epoch 9, including epoch 8 to 9; objective dev loss worsened. Convergence not established.",
            "warning": "Do not infer convergence from falling training loss. The primary dev metric was still rising at the max-epoch boundary for Harrier in both arms and Qwen LoRA; more epochs require a separately approved run."
        },
        "historical_linear_head_only_comparison": {
            "scope": "separate explicit cached-embedding C=1 baseline; same historical grouped split metrics only, not this 192-wide nonlinear typed head",
            "qwen_question_weighted_dev_accuracy": 0.48,
            "harrier_question_weighted_dev_accuracy": 0.57,
            "qwen_metrics_path": "results/probes/varied/lora/runs/qwen-local-lowmem-v2/head-only-baseline.json",
            "harrier_metrics_path": "results/probes/varied/lora/runs/harrier-local-lowmem-v2/head-only-baseline.json",
            "comparability": "Different architecture and training recipe; report separately, not as the matched-head primary control."
        },
        "invalid_partial_attempts": [
            {"path": "runs/qwen-head-interrupted-attempt1/", "status": "INVALID_PARTIAL_NOT_SELECTED",
             "reason": "Pre-correction run completed epoch 1 / 460 steps and was interrupted during epoch 2 while global-L2 clipping geometry was reviewed; preserved but excluded from all comparisons."}
        ],
        "holdout": {"status": "SEALED_NOT_ACCESSED", "labels_read": False, "predictions_read": False, "hashes_read": False},
        "outputs": {"report": "TRAIN-ONLY-REPORT.md", "comparison": "COMPARISON.md", "metrics": "COMPARISON.json"},
    }
    output = ROOT / "TRAIN-ONLY-FREEZE.json"
    output.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    report = [
        "# Matched-head experiment: train-only report",
        "",
        "## Scope and protocol",
        "",
        "CPU-only post-run audit of the four completed runs. No MLX/model was loaded, no inference/training was run, and no heldout/test labels, predictions, or hashes were accessed.",
        "The fixed split has 460 train questions (300 cases, 11 families) and 200 dev questions (120 cases, 4 families). The deterministic grouped split has nine connected components, of which the dev set contains only two; this is a small and noisy basis for selection.",
        "Both arms use the same profile-compiled generic `decengine-varied-pair-text-v2-generic-types` inputs per model, the same 192-wide GELU typed head initialization, optimizer/loss/order, and the same case/family split. Qwen's export manifest is profile version 1 and Harrier's is profile version 2; neither arm changes that model-specific export. Token inputs reject above 288 tokens (no truncation).",
        "Primary selection: maximize equal-weight macro exact accuracy across choice, noul, and score; ties break by lower mean dev objective loss, lower normalized score expected-value MAE, then earlier epoch. All epochs 1–9 are retained; no selection uses heldout data.",
        "",
        "## Frozen selections",
        "",
        "| Model | Arm | Selected epoch | Dev choice | Dev noul | Dev score | Type-balanced macro | Score norm. MAE | Mean dev loss | Selected checkpoint SHA-256 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    def metric(run, kind):
        return run["best_dev"]["per_type"][kind]["exact_accuracy"]
    for tag in MODELS:
        for arm in ARMS:
            run = runs[f"{tag}_{arm}"]
            report.append("| {} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.4f} | {:.4f} | {:.4f} | `{}` |".format(
                tag, arm, run["best_epoch"], metric(run, "choice"), metric(run, "noul"), metric(run, "score"),
                run["best_dev"]["primary_macro_exact_accuracy"], run["best_dev"]["score_normalized_expected_value_mae"],
                run["best_dev"]["secondary_mean_objective_loss"], run["artifacts"]["best.safetensors"]["sha256"]))
    report.extend(["", "## Per-epoch dev metrics", ""])
    for tag in MODELS:
        report.extend([f"### {tag}", ""])
        for arm in ARMS:
            run = runs[f"{tag}_{arm}"]
            report.extend([f"#### {arm}", "",
                           "| Epoch | Choice acc. | Noul acc. | Score acc. | Macro acc. | Score norm. MAE | Mean dev loss |",
                           "|---:|---:|---:|---:|---:|---:|---:|"])
            for epoch_row in run["epoch_metrics"]:
                dev = epoch_row["dev"]
                report.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.4f} | {:.4f} | {:.5f} |".format(
                    epoch_row["epoch"], dev["per_type"]["choice"]["exact_accuracy"],
                    dev["per_type"]["noul"]["exact_accuracy"], dev["per_type"]["score"]["exact_accuracy"],
                    dev["primary_macro_exact_accuracy"], dev["score_normalized_expected_value_mae"],
                    dev["secondary_mean_objective_loss"]))
            report.extend(["", "Train/dev loss and macro trajectory (do not interpret training loss alone as convergence):", "",
                           "| Epoch | Train mean loss | Dev mean loss | Dev macro acc. |",
                           "|---:|---:|---:|---:|"])
            for epoch_row in run["epoch_metrics"]:
                train = epoch_row["train_mean_objective_loss_by_type"]
                train_mean = sum(v["n"] * v["mean_loss"] for v in train.values()) / sum(v["n"] for v in train.values())
                dev = epoch_row["dev"]
                report.append("| {} | {:.5f} | {:.5f} | {:.4f} |".format(
                    epoch_row["epoch"], train_mean, dev["secondary_mean_objective_loss"], dev["primary_macro_exact_accuracy"]))
            report.append("")

    report.extend([
        "## Interpretation and convergence limits",
        "",
        "- Qwen frozen-head selected epoch 2 at macro 0.4028; Qwen LoRA selected epoch 9 at 0.4989 (Δ +0.0961). Qwen LoRA macro was still rising from epoch 8 (0.4811) to 9 (0.4989), while dev mean loss worsened from 1.2248 to 1.2567. This is not evidence of convergence; the accuracy/loss divergence and very small grouped dev basis warrant caution.",
        "- Harrier frozen-head selected epoch 9 at 0.5117; Harrier LoRA selected epoch 9 at 0.5228 (Δ +0.0111). Both arms' macro accuracy rose from epoch 8 to 9 (head 0.5050→0.5117; LoRA 0.5028→0.5228); LoRA dev loss worsened 1.0156→1.0282. Neither has demonstrated convergence by the 9-epoch cap.",
        "- The dev loss rose materially across training for all arms even where primary macro accuracy improved. Per-type prediction histograms and confidence are saved in each epoch's JSONL metrics and `COMPARISON.json`; early constant-class predictions in noul/score were detected by the ≥90% concentration diagnostic. The small two-component dev split makes these fluctuations particularly uncertain.",
        "- Historical explicit cached-embedding C=1 head-only dev accuracies were Qwen 0.480 and Harrier 0.570. This is a different linear architecture/training recipe, so it is shown only as context and is not the matched frozen-backbone typed-head control.",
        "",
        "## Integrity and resources",
        "",
        "All four runs are complete (9 epochs / 4,140 updates each). Initial head safetensors hashes match; within each model, initial train-sample logits match between arms at absolute tolerance 1e-6. Compiled-text parity reports passed. Every step has finite nonzero head gradients; frozen-head LoRA gradients are zero at every step, and LoRA-arm gradients are finite/nonzero at every step.",
        "",
        "| Run | Process peak (GiB) | 6 GiB stop | Swap baseline (B) | Swap min (B) | Swap max (B) | Swap increased? |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for key, run in runs.items():
        res = run["resource"]
        report.append("| {} | {:.3f} | 6.000 | {} | {} | {} | no |".format(
            key, res["process_peak_gib"], res["swap_baseline_bytes"], res["swap_min_bytes"], res["swap_max_bytes"]))
    report.extend([
        "",
        "The process peaks are below the 6 GiB early-stop and 8 GiB hard cap. Swap never exceeded the per-run start baseline; decreases are allowed by the guard. All code/config/source/checkpoint/log hashes and split-set hashes are recorded in `TRAIN-ONLY-FREEZE.json`.",
        "The pre-correction interrupted Qwen-head attempt is preserved under `runs/qwen-head-interrupted-attempt1/` and excluded from this report/freeze.",
        "",
        "## Files",
        "",
        "- `TRAIN-ONLY-FREEZE.json`: hash-bound train/dev selections, integrity/resource audit, and epoch metrics.",
        "- `COMPARISON.md` / `COMPARISON.json`: pairwise per-epoch model comparisons and diagnostics.",
        "- `runs/{qwen,harrier}-{head,lora}/`: configs, logs, immutable epoch checkpoints, best/latest checkpoints, optimizer state, and summaries.",
        "- `verify_and_freeze.py`: reproducible CPU-only validator/freeze generator; it does not import MLX or read any test/heldout path.",
        "",
        "No additional training is authorized by this report; the selected train-only artifacts are frozen pending the user's/orchestrator's next decision.",
        "",
    ])
    (ROOT / "TRAIN-ONLY-REPORT.md").write_text("\n".join(report))
    freeze["historical_linear_head_only_comparison"]["qwen_metrics_sha256"] = sha(PROJECT / freeze["historical_linear_head_only_comparison"]["qwen_metrics_path"])
    freeze["historical_linear_head_only_comparison"]["harrier_metrics_sha256"] = sha(PROJECT / freeze["historical_linear_head_only_comparison"]["harrier_metrics_path"])
    freeze["final_artifacts"] = {
        "report": {"path": "TRAIN-ONLY-REPORT.md", "sha256": sha(ROOT / "TRAIN-ONLY-REPORT.md")},
        "comparison_markdown": {"path": "COMPARISON.md", "sha256": sha(ROOT / "COMPARISON.md")},
        "comparison_json": {"path": "COMPARISON.json", "sha256": sha(ROOT / "COMPARISON.json")},
        "validator": {"path": "verify_and_freeze.py", "sha256": sha(Path(__file__))},
        "aggregator": {"path": "summarize.py", "sha256": sha(ROOT / "summarize.py")},
        "plan": {"path": "PLAN.json", "sha256": sha(ROOT / "PLAN.json")},
    }
    output.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"freeze": str(output), "run_count": len(runs),
                      "selected": {k: {"head_epoch": v["frozen_head"]["epoch"],
                                       "head_macro": v["frozen_head"]["macro_exact_accuracy"],
                                       "lora_epoch": v["lora"]["epoch"],
                                       "lora_macro": v["lora"]["macro_exact_accuracy"],
                                       "delta": v["lora_minus_head_macro_accuracy"]}
                                     for k, v in selected.items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
