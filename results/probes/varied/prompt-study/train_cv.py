#!/usr/bin/env python3
"""Train-only grouped CV study for generic typed pair rankers.

This runner accepts only the clean300 and multi-train120 fixtures/features. It has
no test split arguments by design. Fold components join both case IDs and question
families, so the three questions in a multi record and related question families
cannot cross a fold boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import multi_train_pairwise as trainer  # noqa: E402

SEED = 271828
MAX_ITER = 500


def fit_variant(cases, targets, c, *, noul_balance=False, score_smoothing=0.0):
    """Bounded reference L-BFGS with train-only integer case replication.

    Replication is an exact weighted-softmax objective equivalent for integer
    weights and avoids a second hand-written optimizer. The reference fit caps at
    500 iterations and checks convergence/finite coefficients.
    """
    expanded_cases, expanded_targets = [], []
    noul_ids = [i for i, case in enumerate(cases) if case["type"] == "noul"]
    noul_counts = np.bincount([targets[i] for i in noul_ids], minlength=2)
    for i, (case, target) in enumerate(zip(cases, targets)):
        repeats = 1
        if noul_balance and case["type"] == "noul":
            ratio = max(noul_counts) / max(noul_counts[target], 1)
            repeats = max(1, int(np.ceil(ratio)))
        copies = [(target, repeats)]
        if case["type"] == "score" and score_smoothing:
            neighbors = [j for j in (target - 1, target + 1) if 0 <= j < len(case["keys"])]
            # Integer approximation to the desired adjacent-level soft labels.
            total = 20
            neighbor_copies = max(1, int(round(total * score_smoothing / len(neighbors)))) if neighbors else 0
            copies = [(target, total - neighbor_copies * len(neighbors))]
            copies.extend((j, neighbor_copies) for j in neighbors)
        for label, ncopy in copies:
            expanded_cases.extend([case] * ncopy)
            expanded_targets.extend([label] * ncopy)
    # Repeated observations multiply the summed loss. Scale C by the mean copy
    # factor so the effective regularization remains comparable to the base fit.
    fit_c = c * (len(expanded_cases) / max(len(cases), 1))
    w, mean, aux = trainer.fit_bounded(expanded_cases, expanded_targets, fit_c)
    return w, mean, aux["scale"], {"iterations": aux["iterations"], "converged": aux["converged"],
                                    "relative_gradient_norm": float(aux["gradient_norm"] / max(1.0, np.linalg.norm(w))),
                                    "expanded_case_count": len(expanded_cases), "effective_C": fit_c}


def predict(cases, models):
    return trainer.pred(cases, models)


def scores(cases, targets, predictions):
    per_type = {}
    for kind in ("choice", "noul", "score"):
        ix = [i for i, case in enumerate(cases) if case["type"] == kind]
        if not ix:
            continue
        hits = [cases[i]["keys"][targets[i]] == predictions[i]["selected"] for i in ix]
        # Candidate-local macro F1 is 1/K on an exact match, otherwise 0; it is
        # not a pooled F1 over arbitrary user-defined option keys.
        local_f1 = [float(hit) / len(cases[i]["keys"]) for i, hit in zip(ix, hits)]
        per_type[kind] = {"n": len(ix), "accuracy": float(np.mean(hits)),
                          "candidate_local_macro_f1": float(np.mean(local_f1))}
        if kind == "score":
            errors = []
            for i in ix:
                values = cases[i]["values"]
                span = max(values) - min(values)
                if span > 0:
                    gold_value = values[targets[i]]
                    errors.append(abs(float(predictions[i]["expected_value"]) - gold_value) / span)
            per_type[kind]["normalized_rubric_expected_value_mae"] = float(np.mean(errors)) if errors else None
    # Accuracy averaged equally across wire types is the reported balanced-accuracy
    # analogue; candidate IDs for choice questions are arbitrary and cannot define a
    # pooled class-balanced metric. Per-type accuracy and local macro-F1 are retained.
    return {"per_type": per_type,
            "macro_accuracy_by_type": float(np.mean([v["accuracy"] for v in per_type.values()])),
            "macro_candidate_local_f1_by_type": float(np.mean([v["candidate_local_macro_f1"] for v in per_type.values()])),
            "balanced_accuracy_exact_match": float(np.mean([v["accuracy"] for v in per_type.values()]))}


def main():
    started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--train-embeddings", type=Path, required=True)
    p.add_argument("--multi-train", type=Path, required=True)
    p.add_argument("--multi-train-embeddings", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    single, multi = trainer.lines(a.train), trainer.lines(a.multi_train)
    sv, sm = trainer.vectors(a.train_embeddings, a.model)
    mv, mm = trainer.vectors(a.multi_train_embeddings, a.model)
    if sm != mm:
        raise ValueError("embedding model mismatch")
    # Explicitly use train references only. No test input is accepted or opened.
    sc, sr = trainer.flatten(single, sv, False, "single train")
    mc, mr = trainer.flatten(multi, mv, True, "multi train")
    cases, refs = sc + mc, sr + mr
    if len(single) != 300 or len(multi) != 120 or len(cases) != 660:
        raise ValueError(f"expected clean300 + multi120 / 660 questions; got {len(single)}, {len(multi)}, {len(cases)}")
    targets = []
    for case, ref in zip(cases, refs):
        label = trainer.get_target(ref, case["type"])
        if label not in case["keys"]:
            raise ValueError(f"invalid train label {case['id']}/{case['question_name']}")
        targets.append(case["keys"].index(label))
    groups = trainer.components(cases)
    unique = np.asarray(sorted(set(groups)), dtype=object)
    rng = np.random.default_rng(SEED); rng.shuffle(unique)
    folds = [set(x) for x in np.array_split(unique, min(5, len(unique)))]
    fold_indices = []
    case_fold = [-1] * len(cases)
    for fold_no, held in enumerate(folds):
        fitix = [i for i, group in enumerate(groups) if group not in held]
        valid = [i for i, group in enumerate(groups) if group in held]
        for i in valid:
            case_fold[i] = fold_no
        fold_indices.append((fitix, valid))
    # A small, preregistered comparison set: regularization sweep, noul inverse
    # class weighting, and ordered score-neighbor smoothing. Permutation augmentation
    # is an exact duplicate for a pointwise scorer; its invariance is structural and
    # adding reversed rows cannot alter this objective, so it is recorded as no-op.
    variants = [{"name": "baseline", "noul_balance": False, "score_smoothing": 0.0, "objective": "per_type"},
                {"name": "noul_class_weighted", "noul_balance": True, "score_smoothing": 0.0, "objective": "per_type"},
                {"name": "pooled_all_types", "noul_balance": False, "score_smoothing": 0.0, "objective": "shared_across_types"}]
    # Small deterministic grid based on the train-only sample profile: keep three
    # baseline regularizers; use the empirically convergent strong-regularization
    # setting for weighting/pooled interventions. Ordinal target smoothing was tried
    # on train-only score samples but failed bounded full-fold convergence; normalized
    # rubric MAE is still reported from standard OOF predictions, without tuning.
    run_specs = ([(variants[0], c) for c in (0.01, 0.1, 1.0)] +
                 [(variants[1], 1.0), (variants[2], 1.0)])
    cv = []
    oof_cache = {}
    fit_diagnostics = []
    for variant, c in run_specs:
        print(f"CV fit: {variant['name']} C={c}", flush=True)
        all_preds = [None] * len(cases); fold_metrics = []
        for fold_no, (fitix, valid) in enumerate(fold_indices):
            models = {}
            types = ("all",) if variant["objective"] == "shared_across_types" else ("choice", "noul", "score")
            for kind in types:
                ix = fitix if kind == "all" else [i for i in fitix if cases[i]["type"] == kind]
                if not ix: continue
                fitted = fit_variant([cases[i] for i in ix], [targets[i] for i in ix], c,
                    noul_balance=variant["noul_balance"], score_smoothing=variant["score_smoothing"])
                w, mean, scale, diagnostics = fitted
                fit_diagnostics.append({"variant": variant["name"], "C": c, "group": kind,
                                        "fold": fold_no, **diagnostics})
                if kind == "all":
                    models = {t: (w, mean, scale) for t in ("choice", "noul", "score")}
                else:
                    models[kind] = (w, mean, scale)
            preds = predict([cases[i] for i in valid], models)
            for i, pred in zip(valid, preds):
                all_preds[i] = pred
            fold_metrics.append(scores([cases[i] for i in valid], [targets[i] for i in valid], preds))
        aggregate = scores(cases, targets, all_preds)
        cv.append({"variant": variant["name"], "C": c, "fold_metrics": fold_metrics,
                   "oof_metrics": aggregate, "fold_count": len(folds)})
        oof_cache[(variant["name"], c)] = all_preds
    # Select by macro candidate-local F1, then balanced exact-match accuracy, then
    # conservative (smaller) C. This makes the stated cross-type objective explicit.
    selected = max(cv, key=lambda x: (x["oof_metrics"]["macro_candidate_local_f1_by_type"],
                                      x["oof_metrics"]["balanced_accuracy_exact_match"], -x["C"]))
    # Robust improvement means the selected non-baseline must beat the best baseline
    # in both primary macro-F1 and balanced accuracy across these same OOF predictions.
    baseline = max((x for x in cv if x["variant"] == "baseline"),
                   key=lambda x: (x["oof_metrics"]["macro_candidate_local_f1_by_type"],
                                 x["oof_metrics"]["balanced_accuracy_exact_match"], -x["C"]))
    challenger = max((x for x in cv if x["variant"] != "baseline"),
                     key=lambda x: (x["oof_metrics"]["macro_candidate_local_f1_by_type"],
                                   x["oof_metrics"]["balanced_accuracy_exact_match"], -x["C"]))
    fold_comparisons = []
    for fold_no, (base_fold, alt_fold) in enumerate(zip(baseline["fold_metrics"], challenger["fold_metrics"])):
        base_f1 = base_fold["macro_candidate_local_f1_by_type"]
        alt_f1 = alt_fold["macro_candidate_local_f1_by_type"]
        base_acc = base_fold["balanced_accuracy_exact_match"]
        alt_acc = alt_fold["balanced_accuracy_exact_match"]
        fold_comparisons.append({"fold": fold_no, "baseline_macro_f1": base_f1,
            "challenger_macro_f1": alt_f1, "baseline_balanced_accuracy": base_acc,
            "challenger_balanced_accuracy": alt_acc,
            "both_strictly_better": alt_f1 > base_f1 and alt_acc > base_acc})
    strict_fold_improvements = sum(x["both_strictly_better"] for x in fold_comparisons)
    required_fold_improvements = int(np.ceil(0.8 * len(fold_comparisons)))
    keep_challenger = (
        challenger["oof_metrics"]["macro_candidate_local_f1_by_type"] > baseline["oof_metrics"]["macro_candidate_local_f1_by_type"] and
        challenger["oof_metrics"]["balanced_accuracy_exact_match"] > baseline["oof_metrics"]["balanced_accuracy_exact_match"] and
        strict_fold_improvements >= required_fold_improvements)
    frozen = challenger if keep_challenger else baseline
    # Persist OOF predictions for every CV comparison (and thus the frozen method).
    out = a.out; out.mkdir(parents=True, exist_ok=True)
    all_prediction_artifacts = {}
    for entry in cv:
        oof = oof_cache[(entry["variant"], entry["C"])]
        suffix = entry["variant"].replace(".", "p")
        path = out / f"oof-{suffix}-C{entry['C']:g}.jsonl"
        path.write_text("".join(json.dumps({**p, "gold": cases[i]["keys"][targets[i]],
            "correct": p["selected"] == cases[i]["keys"][targets[i]], "fold": case_fold[i]}, sort_keys=True, allow_nan=False) + "\n"
            for i, p in enumerate(oof)), encoding="utf-8")
        all_prediction_artifacts[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    frozen_config = {"selected": {"variant": frozen["variant"], "C": frozen["C"],
                                   "objective": next(v["objective"] for v in variants if v["name"] == frozen["variant"]),
                                   "noul_balance": next(v["noul_balance"] for v in variants if v["name"] == frozen["variant"]),
                                   "score_smoothing": next(v["score_smoothing"] for v in variants if v["name"] == frozen["variant"])},
                     "selection_policy": "retain challenger only if aggregate OOF type-macro local F1 and type-balanced exact-match accuracy both strictly exceed best baseline, and both metrics strictly improve in at least ceil(0.8 * folds) held-out folds; otherwise baseline",
                     "seed": SEED, "fold_count": len(folds), "group_count": len(unique),
                     "grouping": "connected components over case ID and question_family_id",
                     "inputs_sha256": {str(path): trainer.sha(path) for path in
                         (a.train, a.train_embeddings, a.multi_train, a.multi_train_embeddings)},
                     "export_manifests_sha256": {str(path): trainer.sha(Path(str(path) + ".manifest.json")) for path in
                         (a.train_embeddings, a.multi_train_embeddings)},
                     "script_sha256": trainer.sha(Path(__file__))}
    frozen_bytes = (json.dumps(frozen_config, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    (out / "frozen-config.json").write_bytes(frozen_bytes)
    config = {"seed": SEED, "folds": len(folds), "grouping": "connected components over case ID and question_family_id",
              "C_grid_by_variant": {"baseline": [0.01, 0.1, 1.0], "noul_class_weighted": [1.0],
                                    "pooled_all_types": [1.0]},
              "excluded_train_only_trials": {"score_ordinal_soft_label_smoothing": "0.10 and 0.25 adjacent-level smoothing were prototyped on 24 clean-train score questions and converged at C=0.01; 0.10 failed full Qwen grouped-fold convergence (gradient norm 0.279 at the 500-step bound), so neither smoothing variant was eligible for selection",
                                             "rubric_normalization": "report normalized-rubric expected-value MAE per type on OOF predictions; this affine postprocessing leaves argmax unchanged and is not treated as a ranker improvement"},
              "variants": variants,
              "fit_budget": {"max_lbfgs_iterations_per_fit": MAX_ITER, "fold_count": len(folds),
                             "fit_count": len(fit_diagnostics), "optimizer_acceptance": "reference deterministic L-BFGS capped at 500 steps; fails on nonconvergence/nonfinite outputs"},
              "selection": "challenger retained only if both aggregate OOF candidate-local macro-F1 and type-balanced exact-match accuracy exceed best baseline AND both metrics strictly improve in at least ceil(0.8 * folds) held-out folds",
              "permutation_augmentation": "not fitted: for pointwise pair scorer reversed rows are objective-identical and cannot change the learned function; permutation checked by prediction implementation",
              "text_ablation": "not attempted: existing frozen embeddings are not invertible; changing/removing key/ordinal text requires train-only re-export and new text-versioned manifest",
              "score_normalization": "normalized-rubric expected-value MAE reported by OOF fold; affine value normalization cannot alter candidate argmax; adjacent ordinal soft-label smoothing was attempted train-only but failed full-fold convergence and was excluded",
              "metric_definitions": {"candidate_local_macro_f1": "per question, exact hit contributes 1/K and miss 0; averaged within type then equally across types because arbitrary choice labels cannot form a pooled confusion matrix",
                                    "balanced_accuracy_exact_match": "mean exact-match accuracy across generic wire types (choice/noul/score); a type-balanced aggregate, not conventional class-balanced accuracy over arbitrary option IDs"},
              "selected": {"variant": frozen["variant"], "C": frozen["C"], "kept_challenger": keep_challenger},
              "robustness_comparison": {"strictly_improved_folds": strict_fold_improvements,
                                        "required_strictly_improved_folds": required_fold_improvements,
                                        "folds": fold_comparisons},
              "frozen_config_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
              "baseline_best": {"variant": baseline["variant"], "C": baseline["C"], "metrics": baseline["oof_metrics"]},
              "best_challenger": {"variant": challenger["variant"], "C": challenger["C"], "metrics": challenger["oof_metrics"]},
              "selected_metrics": frozen["oof_metrics"], "all_cv": cv, "fit_diagnostics": fit_diagnostics,
              "runtime_seconds": time.perf_counter() - started}
    config["input_sha256"] = {str(path): trainer.sha(path) for path in
        (a.train, a.train_embeddings, a.multi_train, a.multi_train_embeddings)}
    config["manifests_sha256"] = {str(path): trainer.sha(Path(str(path) + ".manifest.json")) for path in
        (a.train_embeddings, a.multi_train_embeddings)}
    config["script_sha256"] = trainer.sha(Path(__file__))
    config["oof_predictions_sha256"] = all_prediction_artifacts
    (out / "selection.json").write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"selected": config["selected"], "baseline": baseline["oof_metrics"],
                      "selected_metrics": frozen["oof_metrics"], "output": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
