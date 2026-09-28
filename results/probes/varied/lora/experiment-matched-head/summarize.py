#!/usr/bin/env python3
"""Build the fixed train/dev matched-head comparison without model inference."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
EXPECTED = {
    "qwen": "Qwen/Qwen3-Embedding-0.6B",
    "harrier": "microsoft/harrier-oss-v1-0.6b",
}
ARMS = ("head", "lora")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def collect(tag, arm):
    path = RUNS / f"{tag}-{arm}"
    required = ["config.json", "initial.json", "steps.jsonl", "epochs.jsonl", "memory.jsonl",
                "summary.json", "best.json", "best.safetensors", "latest.safetensors",
                "optimizer.safetensors"]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{path}: missing required outputs {missing}")
    config = json.loads((path / "config.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    epochs = read_jsonl(path / "epochs.jsonl")
    steps = read_jsonl(path / "steps.jsonl")
    memory = read_jsonl(path / "memory.jsonl")
    if config["model"] != EXPECTED[tag] or config["arm"] != arm:
        raise ValueError(f"run identity mismatch at {path}")
    if summary["status"] != "complete" or len(epochs) != 9 or len(steps) != 4140:
        raise ValueError(f"run incomplete at {path}: status={summary['status']} epochs={len(epochs)} steps={len(steps)}")
    for epoch in range(1, 10):
        if not (path / f"epoch-{epoch:02d}.safetensors").is_file():
            raise FileNotFoundError(f"missing immutable epoch checkpoint {path}/epoch-{epoch:02d}.safetensors")
    peaks = [x["process_peak_bytes"] for x in memory]
    swaps = [x["system_swap_used_bytes"] for x in memory]
    swap_baselines = {x["swap_baseline_bytes"] for x in memory}
    if len(swap_baselines) != 1:
        raise RuntimeError(f"inconsistent swap baseline in {path}: {sorted(swap_baselines)}")
    swap_baseline = next(iter(swap_baselines))
    if max(peaks) >= 6 * 1024**3 or max(swaps) > swap_baseline:
        raise RuntimeError(f"resource guard violated in {path}: peak={max(peaks)} swap_baseline={swap_baseline} swap_max={max(swaps)}")
    if arm == "head" and any(x["lora_grad_norm"] != 0 for x in steps):
        raise RuntimeError(f"frozen-head arm has a nonzero LoRA gradient in {path}")
    if arm == "lora" and any(x["lora_grad_norm"] <= 0 for x in steps):
        raise RuntimeError(f"LoRA arm has a zero LoRA gradient in {path}")
    initial = json.loads((path / "initial.json").read_text())
    return {"path": path, "config": config, "initial": initial, "epochs": epochs,
            "steps": steps, "memory": memory, "summary": summary,
            "peak_bytes": max(peaks), "swap_baseline_bytes": swap_baseline,
            "swap_min_bytes": min(swaps), "swap_max_bytes": max(swaps)}


def load_pair(tag):
    head = collect(tag, "head")
    lora = collect(tag, "lora")
    if head["initial"]["head_init_sha256"] != lora["initial"]["head_init_sha256"]:
        raise AssertionError(f"{tag}: initial head SHA mismatch")
    if head["initial"]["sample_key"] != lora["initial"]["sample_key"]:
        raise AssertionError(f"{tag}: initial parity sample mismatch")
    left, right = head["initial"]["logits"], lora["initial"]["logits"]
    if len(left) != len(right) or any(abs(a - b) > 1e-6 for a, b in zip(left, right)):
        raise AssertionError(f"{tag}: initial arm logits differ > 1e-6")
    if [x["epoch"] for x in head["epochs"]] != list(range(1, 10)) or [x["epoch"] for x in lora["epochs"]] != list(range(1, 10)):
        raise AssertionError(f"{tag}: epoch metric numbering invalid")
    return head, lora


def overfit_diagnostic(run):
    rows = run["epochs"]
    result = []
    for i, row in enumerate(rows):
        train = row["train_mean_objective_loss_by_type"]
        train_mean = sum(v["n"] * v["mean_loss"] for v in train.values()) / sum(v["n"] for v in train.values())
        dev_loss = row["dev"]["secondary_mean_objective_loss"]
        result.append({"epoch": row["epoch"], "train_mean_objective_loss": train_mean,
                       "dev_mean_objective_loss": dev_loss,
                       "dev_macro_type_accuracy": row["dev"]["primary_macro_exact_accuracy"]})
    # Flag a possible overfit pattern only when train loss falls while dev loss
    # rises on two consecutive transitions; the raw table remains authoritative.
    flags = []
    for a, b, c in zip(result, result[1:], result[2:]):
        if (b["train_mean_objective_loss"] < a["train_mean_objective_loss"] and
                c["train_mean_objective_loss"] < b["train_mean_objective_loss"] and
                b["dev_mean_objective_loss"] > a["dev_mean_objective_loss"] and
                c["dev_mean_objective_loss"] > b["dev_mean_objective_loss"]):
            flags.append({"epochs": [a["epoch"], b["epoch"], c["epoch"]],
                          "criterion": "two consecutive train-loss declines with two consecutive dev-loss increases"})
    collapse = []
    for row in rows:
        for kind, metrics in row["dev"]["per_type"].items():
            hist = metrics["predicted_candidate_index_hist"]
            n = metrics["n"]
            max_share = max(hist.values()) / n
            if max_share >= 0.90:
                collapse.append({"epoch": row["epoch"], "type": kind,
                                 "max_predicted_candidate_share": max_share,
                                 "histogram": hist})
    return {"epoch_trajectory": result, "possible_overfit_flags": flags,
            "prediction_collapse_flags_at_90pct": collapse}


def main():
    all_results = {}
    markdown = ["# Matched-head train/dev comparison", "",
                "All rows are from the fixed grouped 460-train / 200-dev split. No heldout or test data was opened.",
                "Primary metric: equal-weight macro accuracy over choice, noul, and score. Best epoch is selected using the predeclared dev rule.", ""]
    for tag in EXPECTED:
        head, lora = load_pair(tag)
        comparison = []
        for hrow, lrow in zip(head["epochs"], lora["epochs"]):
            hm, lm = hrow["dev"], lrow["dev"]
            comparison.append({
                "epoch": hrow["epoch"],
                "head": hm,
                "lora": lm,
                "delta_lora_minus_head": {
                    "primary_macro_exact_accuracy": lm["primary_macro_exact_accuracy"] - hm["primary_macro_exact_accuracy"],
                    "secondary_mean_objective_loss": lm["secondary_mean_objective_loss"] - hm["secondary_mean_objective_loss"],
                    "score_normalized_expected_value_mae": lm["score_normalized_expected_value_mae"] - hm["score_normalized_expected_value_mae"],
                    "per_type_exact_accuracy": {k: lm["per_type"][k]["exact_accuracy"] - hm["per_type"][k]["exact_accuracy"] for k in hm["per_type"]},
                },
                "head_train_objective_loss_by_type": hrow["train_mean_objective_loss_by_type"],
                "lora_train_objective_loss_by_type": lrow["train_mean_objective_loss_by_type"],
            })
        best_head = min(head["epochs"], key=lambda r: (-r["dev"]["primary_macro_exact_accuracy"], r["dev"]["secondary_mean_objective_loss"], r["dev"]["score_normalized_expected_value_mae"], r["epoch"]))
        best_lora = min(lora["epochs"], key=lambda r: (-r["dev"]["primary_macro_exact_accuracy"], r["dev"]["secondary_mean_objective_loss"], r["dev"]["score_normalized_expected_value_mae"], r["epoch"]))
        best_head_metric = best_head["dev"]["primary_macro_exact_accuracy"]
        best_lora_metric = best_lora["dev"]["primary_macro_exact_accuracy"]
        arm_info = {}
        for name, run in (("head", head), ("lora", lora)):
            p = run["path"]
            arm_info[name] = {
                "run_dir": p.relative_to(ROOT).as_posix(),
                "config_sha256": sha(p / "config.json"),
                "initial_head_sha256": run["initial"]["head_init_sha256"],
                "best_epoch": run["summary"]["best_epoch"],
                "best_checkpoint_sha256": sha(p / "best.safetensors"),
                "latest_checkpoint_sha256": sha(p / "latest.safetensors"),
                "optimizer_checkpoint_sha256": sha(p / "optimizer.safetensors"),
                "epoch_checkpoint_sha256": {str(e): sha(p / f"epoch-{e:02d}.safetensors") for e in range(1, 10)},
                "max_process_peak_bytes": run["peak_bytes"],
                "swap_baseline_bytes": run["swap_baseline_bytes"],
                "swap_min_bytes": run["swap_min_bytes"],
                "swap_max_bytes": run["swap_max_bytes"],
                "all_steps": len(run["steps"]),
                "min_head_grad_norm": min(x["head_grad_norm"] for x in run["steps"]),
                "max_head_grad_norm": max(x["head_grad_norm"] for x in run["steps"]),
                "min_lora_grad_norm": min(x["lora_grad_norm"] for x in run["steps"]),
                "max_lora_grad_norm": max(x["lora_grad_norm"] for x in run["steps"]),
                "diagnostics": overfit_diagnostic(run),
            }
        all_results[tag] = {
            "model": EXPECTED[tag], "same_initial_head": True,
            "initial_logits_match_atol_1e-6": True,
            "selection_rule": head["config"]["selection"],
            "best_head_epoch": best_head["epoch"], "best_lora_epoch": best_lora["epoch"],
            "best_head_macro_accuracy": best_head_metric, "best_lora_macro_accuracy": best_lora_metric,
            "best_lora_improves_same_head": best_lora_metric > best_head_metric,
            "best_macro_accuracy_delta_lora_minus_head": best_lora_metric - best_head_metric,
            "arms": arm_info, "epoch_comparison": comparison,
        }
        markdown.extend([f"## {tag}", "",
                         "| Epoch | Head macro acc | LoRA macro acc | Δ macro | Head loss | LoRA loss | Head score MAE | LoRA score MAE |",
                         "|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for row in comparison:
            markdown.append("| {epoch} | {ha:.4f} | {la:.4f} | {da:+.4f} | {hl:.5f} | {ll:.5f} | {hs:.5f} | {ls:.5f} |".format(
                epoch=row["epoch"], ha=row["head"]["primary_macro_exact_accuracy"],
                la=row["lora"]["primary_macro_exact_accuracy"],
                da=row["delta_lora_minus_head"]["primary_macro_exact_accuracy"],
                hl=row["head"]["secondary_mean_objective_loss"], ll=row["lora"]["secondary_mean_objective_loss"],
                hs=row["head"]["score_normalized_expected_value_mae"], ls=row["lora"]["score_normalized_expected_value_mae"]))
        markdown.extend(["", f"Selected head epoch {best_head['epoch']} ({best_head_metric:.4f}); selected LoRA epoch {best_lora['epoch']} ({best_lora_metric:.4f}); LoRA improves at the selected epoch: **{best_lora_metric > best_head_metric}**.", ""])
    result = {"status": "TRAIN_DEV_MATCHED_COMPARISON_COMPLETE", "scope": "train/dev only; no heldout/test labels, predictions, or hashes accessed", "models": all_results}
    (ROOT / "COMPARISON.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (ROOT / "COMPARISON.md").write_text("\n".join(markdown) + "\n")
    print(json.dumps({k: {"head_best": v["best_head_epoch"], "lora_best": v["best_lora_epoch"], "head_macro": v["best_head_macro_accuracy"], "lora_macro": v["best_lora_macro_accuracy"], "lora_improves": v["best_lora_improves_same_head"]} for k, v in all_results.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
