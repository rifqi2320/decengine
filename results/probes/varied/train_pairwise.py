#!/usr/bin/env python3
"""Train a question-conditioned linear candidate scorer on frozen embeddings.

Inputs are one case per record and exporter records contain one query vector plus
candidate vectors for that case. Test references are not consulted until the
prediction JSONL has been written. See README.md for the protocol/schema.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SEED = 271828
# Bounded regularization grid keeps fits well-conditioned for 4 x 1024-D inputs.
C_VALUES = (0.01, 0.1, 1.0, 10.0)


def local_macro_f1(correct: list[bool], candidate_counts: list[int]) -> float:
    """Mean per-question macro-F1 over that question's own candidate labels.

    With one gold decision per question, local macro-F1 is 1/K on an exact match
    (one supported class gets F1=1, other local options get 0), otherwise zero.
    It is necessarily secondary to exact-match accuracy and has this sparse-sample
    interpretation; it does not pretend arbitrary option keys form shared classes.
    """
    return float(np.mean([1.0 / k if hit else 0.0 for hit, k in zip(correct, candidate_counts)]))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(r.get("id", "")) for r in rows]
    if not rows or not all(ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{path}: need non-empty records with unique IDs")
    return rows


def case_tokens(row: dict[str, Any]) -> set[str]:
    """Leakage-audit text only; never includes candidates or reference metadata."""
    questions = row.get("request", {}).get("questions", {})
    prompt = ""
    if isinstance(questions, dict) and len(questions) == 1:
        question = next(iter(questions.values()))
        if isinstance(question, dict):
            prompt = str(question.get("prompt", ""))
    state = json.dumps(row.get("request", {}).get("state", {}), sort_keys=True, ensure_ascii=False)
    return set(re.findall(r"[\w]+", (state + " " + prompt).casefold()))


def assert_no_cross_split_siblings(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]],
                                   threshold: float = 0.88) -> None:
    """Reject exact/near duplicate scenario narratives across the locked split."""
    train_tokens = [case_tokens(row) for row in train_rows]
    test_tokens = [case_tokens(row) for row in test_rows]
    for i, left in enumerate(train_tokens):
        for j, right in enumerate(test_tokens):
            union = left | right
            similarity = len(left & right) / len(union) if union else 0.0
            if similarity >= threshold:
                raise ValueError(
                    "train/test scenario sibling or near-duplicate leakage: "
                    f"{train_rows[i]['id']} vs {test_rows[j]['id']} (token Jaccard={similarity:.3f})")


def training_cv_groups(rows: list[dict[str, Any]], threshold: float = 0.88) -> tuple[list[str], list[dict[str, Any]]]:
    """Connect families sharing exact/near-duplicate scenario text for grouped CV."""
    families = [str(row["question_family_id"]) for row in rows]
    parent = {family: family for family in set(families)}
    def root(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item
    tokens = [case_tokens(row) for row in rows]
    links = []
    for i, left in enumerate(tokens):
        for j in range(i):
            if families[i] == families[j]:
                continue
            union = left | tokens[j]
            similarity = len(left & tokens[j]) / len(union) if union else 0.0
            if similarity >= threshold:
                a, b = root(families[i]), root(families[j])
                if a != b:
                    parent[a] = b
                links.append({"left_id": str(rows[j]["id"]), "right_id": str(rows[i]["id"]),
                              "token_jaccard": similarity})
    return [root(family) for family in families], links


def manifest_info(path: Path) -> dict[str, Any] | None:
    manifest_path = Path(str(path) + ".manifest.json")
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {"path": str(manifest_path), "sha256": sha(manifest_path), "contents": manifest}


def read_vectors(path: Path, model_arg: str) -> tuple[dict[str, Any], str]:
    raw = path.read_text(encoding="utf-8")
    try:
        doc = json.loads(raw)
        records = doc.get("embeddings", []) if isinstance(doc, dict) else doc
        if not isinstance(records, list):
            records = [doc]
    except json.JSONDecodeError:
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    models = sorted({str(r["model"]) for r in records if isinstance(r, dict) and r.get("model")})
    matched = [m for m in models if m.casefold() == model_arg.casefold()]
    if not matched:
        matched = [m for m in models if model_arg.casefold() in m.casefold()]
    if models and len(matched) != 1:
        raise ValueError(f"--model must identify one exported model; found {models}")
    model = matched[0] if matched else model_arg
    result = {}
    for row in records:
        if row.get("model", model) != model:
            continue
        key = str(row["id"])
        if key in result:
            raise ValueError(f"{path}: duplicate ID {key}")
        # Current varied exporter contract: query_embedding and a candidates array
        # of {candidate_id, embedding, ...}. Keep accepting the initial prototype's
        # query/candidates map for small local smoke fixtures.
        emb = row.get("embeddings", row)
        query = np.asarray(emb.get("query_embedding", emb.get("query")), dtype=np.float64)
        candidates = emb.get("candidates")
        if isinstance(candidates, list):
            candidates = {str(c.get("candidate_id")): c.get("embedding") for c in candidates
                          if isinstance(c, dict) and c.get("candidate_id") is not None}
        if query.ndim != 1 or not np.isfinite(query).all() or not isinstance(candidates, dict):
            raise ValueError(f"{path}:{key}: expected finite query_embedding and candidate array")
        parsed = {}
        for name, vector in candidates.items():
            value = np.asarray(vector, dtype=np.float64)
            if value.ndim != 1 or value.shape != query.shape or not np.isfinite(value).all():
                raise ValueError(f"{path}:{key}: candidate {name!r} dimension/nonfinite mismatch")
            parsed[str(name)] = value
        result[key] = (query, parsed)
    return result, model


def question(row: dict[str, Any]) -> tuple[str, dict[str, str], list[float] | None, str]:
    request = row["request"]
    if len(request.get("questions", {})) != 1:
        raise ValueError(f"{row['id']}: expected exactly one question")
    qname, q = next(iter(request["questions"].items()))
    kind = q["type"]
    if kind == "choice":
        opts = {str(k): str(v) for k, v in q["options"].items()}
        if not 2 <= len(opts) <= 6:
            raise ValueError(f"{row['id']}: choice must have 2..6 options")
        values = None
    elif kind == "noul":
        # The exporter encodes the question-conditioned semantic outcomes using
        # these stable IDs; they are not candidate positions or target-derived.
        opts = {"unsupported": "Proposition not supported by the reported state",
                "supported": "Proposition supported by the reported state"}
        values = None
    elif kind == "score":
        levels = q["levels"]
        if not 3 <= len(levels) <= 5:
            raise ValueError(f"{row['id']}: score must have 3..5 ordered levels")
        opts = {str(x["label"]): str(x["criterion"]) for x in levels}
        values = [float(x["value"]) for x in levels]
    else:
        raise ValueError(f"{row['id']}: unsupported question type {kind!r}")
    return kind, opts, values, str(qname)


def pair_features(query: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    return np.concatenate((query, candidate, query * candidate, np.abs(query - candidate)))


def materialize(rows: list[dict[str, Any]], vectors: dict[str, Any]) -> list[dict[str, Any]]:
    cases = []
    for row in rows:
        case_id = str(row["id"])
        if case_id not in vectors:
            raise ValueError(f"missing embeddings for {case_id}")
        kind, options, values, qname = question(row)
        query, candidate_vecs = vectors[case_id]
        if set(candidate_vecs) != set(options):
            raise ValueError(f"{case_id}: candidate embedding keys do not match question candidates")
        keys = list(options)
        cases.append({"id": case_id, "family": str(row["question_family_id"]),
                      "domain": str(row["domain"]), "type": kind, "question_name": qname,
                      "keys": keys, "options": options, "values": values,
                      "x": np.stack([pair_features(query, candidate_vecs[k]) for k in keys])})
    return cases


def target_index(case: dict[str, Any], row: dict[str, Any]) -> int:
    target = row["reference"]["target"]
    if case["type"] == "choice":
        label = str(target["selected"])
    elif case["type"] == "noul":
        if type(target["value"]) is not bool:
            raise ValueError(f"{case['id']}: Noul target.value must be boolean")
        label = "supported" if target["value"] else "unsupported"
    else:
        label = str(target["label"])
    if label not in case["keys"]:
        raise ValueError(f"{case['id']}: target does not identify a candidate")
    return case["keys"].index(label)


def fit(cases: list[dict[str, Any]], targets: list[int], c: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    xall = np.concatenate([x["x"] for x in cases])
    mean = xall.mean(axis=0)
    scale = xall.std(axis=0)
    scale[scale < 1e-12] = 1.0
    # A single contiguous design matrix avoids thousands of tiny BLAS calls in
    # objective/gradient evaluation. Targets index candidates within each case.
    design = np.concatenate([(x["x"] - mean) / scale for x in cases], axis=0)
    offsets = np.cumsum([0] + [len(x["keys"]) for x in cases])
    target_rows = offsets[:-1] + np.asarray(targets, dtype=np.int64)
    dim = design.shape[1]
    def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
        loss = 0.5 * np.dot(w, w) / c
        grad = w / c
        with warnings.catch_warnings():
            # Some macOS OpenBLAS builds emit spurious FPE warnings for finite
            # matmuls. Suppression is narrow; outputs are checked immediately.
            warnings.filterwarnings("ignore", message=r".*encountered in matmul", category=RuntimeWarning)
            scores = design @ w
        if not np.isfinite(scores).all():
            raise FloatingPointError("non-finite logits during scorer optimization")
        delta = np.zeros(len(scores), dtype=np.float64)
        for begin, end, target in zip(offsets[:-1], offsets[1:], target_rows):
            local = scores[begin:end]
            y = int(target - begin)
            maximum = float(np.max(local))
            exp_scores = np.exp(local - maximum)
            probs = exp_scores / exp_scores.sum()
            loss += maximum + float(np.log(exp_scores.sum())) - local[y]
            delta[begin:end] = probs
            delta[target] -= 1.0
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*encountered in matmul", category=RuntimeWarning)
            grad += design.T @ delta
        if not np.isfinite(grad).all() or not np.isfinite(loss):
            raise FloatingPointError("non-finite loss/gradient during scorer optimization")
        return float(loss), grad
    # Bounded-memory L-BFGS with Armijo backtracking: convex softmax + L2 objective,
    # no external optimizer dependency, finite iter/line-search bounds.
    w = np.zeros(dim, dtype=np.float64)
    loss, grad = objective(w)
    iterations = 0
    history_s: list[np.ndarray] = []
    history_y: list[np.ndarray] = []
    history_rho: list[float] = []
    grad_norm = float(np.linalg.norm(grad))
    converged = False
    for iterations in range(1, 301):
        if grad_norm <= 1e-5 * max(1.0, float(np.linalg.norm(w))):
            converged = True
            break
        # Standard two-loop recursion for inverse-Hessian direction.
        q = grad.copy()
        alpha: list[float] = []
        for s, yvec, rho in zip(reversed(history_s), reversed(history_y), reversed(history_rho)):
            a = rho * float(np.dot(s, q)); alpha.append(a); q -= a * yvec
        if history_s:
            sy = float(np.dot(history_s[-1], history_y[-1]))
            yy = float(np.dot(history_y[-1], history_y[-1]))
            r = q * (sy / yy if yy > 0 else 1.0)
        else:
            r = q
        for s, yvec, rho, a in zip(history_s, history_y, history_rho, reversed(alpha)):
            beta = rho * float(np.dot(yvec, r))
            r += s * (a - beta)
        direction = -r
        directional = float(np.dot(grad, direction))
        if not np.isfinite(directional) or directional >= 0:
            history_s.clear(); history_y.clear(); history_rho.clear()
            direction = -grad
            directional = -float(np.dot(grad, grad))
        step = 1.0
        accepted = False
        for _line_search in range(24):
            candidate = w + step * direction
            next_loss, next_grad = objective(candidate)
            if np.isfinite(next_loss) and np.isfinite(next_grad).all() and next_loss <= loss + 1e-4 * step * directional:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
        s = candidate - w
        yvec = next_grad - grad
        curvature = float(np.dot(s, yvec))
        if curvature > 1e-12 * np.linalg.norm(s) * max(np.linalg.norm(yvec), 1e-30):
            history_s.append(s); history_y.append(yvec); history_rho.append(1.0 / curvature)
            if len(history_s) > 10:
                history_s.pop(0); history_y.pop(0); history_rho.pop(0)
        w, loss, grad = candidate, next_loss, next_grad
        grad_norm = float(np.linalg.norm(grad))
    if grad_norm <= 1e-4 * max(1.0, float(np.linalg.norm(w))):
        converged = True
    if not np.isfinite(w).all():
        raise FloatingPointError("non-finite scorer coefficients")
    if not converged:
        raise RuntimeError(f"linear scorer did not converge in {iterations} iterations (gradient norm={grad_norm:.3g})")
    return w, mean, {"scale": scale, "iterations": iterations, "objective": float(loss),
                     "gradient_norm": grad_norm, "converged": converged}


def predict(cases: list[dict[str, Any]], w: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> list[dict[str, Any]]:
    output = []
    for c in cases:
        logits = ((c["x"] - mean) / scale) @ w
        logits -= logits.max()
        probs = np.exp(logits); probs /= probs.sum()
        selected = int(np.argmax(probs))
        selected_key = c["keys"][selected]
        output.append({"id": c["id"], "question_type": c["type"], "question_name": c["question_name"],
                       "family": c["family"], "domain": c["domain"], "candidate_keys": c["keys"],
                       "candidate_logits": [float(v) for v in logits],
                       "probabilities": {k: float(probs[i]) for i, k in enumerate(c["keys"])},
                       "selected": selected_key,
                       "label": (selected_key == "supported" if c["type"] == "noul" else selected_key),
                       "value": (selected_key == "supported" if c["type"] == "noul" else None),
                       "selected_level": selected_key if c["type"] == "score" else None,
                       "confidence": float(probs[selected]),
                       "expected_value": (float(np.dot(probs, c["values"])) if c["values"] is not None else None)})
    return output


def evaluate(rows: list[dict[str, Any]], cases: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    # Called only after prediction artifacts are durably written.
    targets = [target_index(c, r) for c, r in zip(cases, rows)]
    reports: dict[str, Any] = {}
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(cases):
        buckets[f"type:{c['type']}"].append(i)
        buckets[f"family:{c['family']}"].append(i)
        buckets[f"domain:{c['domain']}"].append(i)
        buckets[f"candidate_count:{len(c['keys'])}"].append(i)
    buckets["all"] = list(range(len(cases)))
    for name, indices in buckets.items():
        if name.startswith("family:") or name.startswith("domain:") or name.startswith("candidate_count:") or name.startswith("type:") or name == "all":
            yy = [targets[i] for i in indices]
            pp = [cases[i]["keys"].index(predictions[i]["selected"]) for i in indices]
            probs = [list(predictions[i]["probabilities"].values()) for i in indices]
            correct = [yy[j] == pp[j] for j in range(len(indices))]
            report = {"n": len(indices), "accuracy": float(np.mean(correct)),
                      "macro_f1": local_macro_f1(correct, [len(cases[i]["keys"]) for i in indices]),
                      "macro_f1_definition": "mean within-question macro F1 over local candidate labels; exact match contributes 1/K, otherwise 0",
                      "mean_log_loss": float(np.mean([-np.log(max(probs[j][yy[j]], 1e-15)) for j in range(len(indices))]))}
            score_ids = [j for j, i in enumerate(indices) if cases[i]["type"] == "score"]
            if score_ids:
                actual = [float(rows[indices[j]]["reference"]["target"]["value"]) for j in score_ids]
                predicted = [float(predictions[indices[j]]["expected_value"]) for j in score_ids]
                report["score_expected_value_mae"] = float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))
            reports[name] = report
    type_reports = [v for k, v in reports.items() if k.startswith("type:")]
    reports["macro_f1_across_question_types"] = float(np.mean([r["macro_f1"] for r in type_reports])) if type_reports else None
    overall = reports["all"]
    entropies = []
    confidences = []
    for pred in predictions:
        p = np.asarray(list(pred["probabilities"].values()), dtype=np.float64)
        entropies.append(float(-np.sum(p * np.log(np.maximum(p, 1e-15)))))
        confidences.append(float(np.max(p)))
    overall["mean_predictive_entropy_nats"] = float(np.mean(entropies))
    overall["mean_max_probability"] = float(np.mean(confidences))
    # Cluster bootstrap by held-out question family (never pretend candidate pairs
    # are independent). For very few test families these intervals remain descriptive.
    family_indices: dict[str, list[int]] = defaultdict(list)
    for i, case in enumerate(cases):
        family_indices[case["family"]].append(i)
    rng = np.random.default_rng(SEED)
    families = sorted(family_indices)
    boot_f1, boot_acc = [], []
    if families:
        for _ in range(2000):
            sampled = rng.choice(families, size=len(families), replace=True)
            indices = [i for family in sampled for i in family_indices[str(family)]]
            yy = [targets[i] for i in indices]
            pp = [cases[i]["keys"].index(predictions[i]["selected"]) for i in indices]
            correct = [yy[j] == pp[j] for j in range(len(indices))]
            boot_f1.append(local_macro_f1(correct, [len(cases[i]["keys"]) for i in indices]))
            boot_acc.append(float(np.mean(correct)))
    overall["family_cluster_bootstrap_95_percentile"] = {
        "macro_f1": [float(x) for x in np.percentile(boot_f1, [2.5, 97.5])] if boot_f1 else None,
        "accuracy": [float(x) for x in np.percentile(boot_acc, [2.5, 97.5])] if boot_acc else None,
        "family_count": len(families), "replicates": 2000}
    return reports


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True); p.add_argument("--test", type=Path, required=True)
    p.add_argument("--train-embeddings", type=Path, required=True); p.add_argument("--test-embeddings", type=Path, required=True)
    p.add_argument("--model", required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--expected-counts", action="store_true", help="require clean 300 train and 200 test")
    args = p.parse_args()
    train_rows, test_rows = read_jsonl(args.train), read_jsonl(args.test)
    if args.expected_counts and (len(train_rows), len(test_rows)) != (300, 200):
        raise ValueError(f"expected clean 300 train/200 test, got {len(train_rows)}/{len(test_rows)}")
    train_ids, test_ids = {str(r["id"]) for r in train_rows}, {str(r["id"]) for r in test_rows}
    if train_ids & test_ids: raise ValueError("train/test IDs overlap")
    train_families = {str(r["question_family_id"]) for r in train_rows}
    test_families = {str(r["question_family_id"]) for r in test_rows}
    train_domains = {str(r["domain"]) for r in train_rows}
    test_domains = {str(r["domain"]) for r in test_rows}
    if train_families & test_families:
        raise ValueError("train/test question-family overlap; refusing held-out-family leakage")
    assert_no_cross_split_siblings(train_rows, test_rows)
    trv, trmodel = read_vectors(args.train_embeddings, args.model)
    tev, temodel = read_vectors(args.test_embeddings, args.model)
    if trmodel != temodel: raise ValueError("train/test exporter model mismatch")
    train_manifest = manifest_info(args.train_embeddings)
    test_manifest = manifest_info(args.test_embeddings)
    if train_manifest and test_manifest:
        identity_fields = ("model", "profile_version", "profile_sha256", "text_version",
                           "embedding_dimensions", "normalization")
        left, right = train_manifest["contents"], test_manifest["contents"]
        mismatches = [field for field in identity_fields if left.get(field) != right.get(field)]
        if mismatches:
            raise ValueError(f"train/test exporter provenance differs: {mismatches}")
    if set(trv) != train_ids or set(tev) != test_ids: raise ValueError("embedding IDs must exactly match split IDs")
    train = materialize(train_rows, trv)
    test = materialize(test_rows, tev)
    y = [target_index(c, r) for c, r in zip(train, train_rows)]
    families = sorted({c["family"] for c in train})
    if len(families) < 2: raise ValueError("at least two training question families required for group-aware CV")
    cv_group_by_row, train_context_links = training_cv_groups(train_rows)
    cv_groups = sorted(set(cv_group_by_row))
    if len(cv_groups) < 2: raise ValueError("fewer than two independent family/context groups for CV")
    # Deterministic shuffled grouped folds keep all cases in a question family
    # together while avoiding one expensive fit per family.
    rng = np.random.default_rng(SEED)
    shuffled_groups = np.asarray(cv_groups, dtype=object)
    rng.shuffle(shuffled_groups)
    family_folds = [list(x) for x in np.array_split(shuffled_groups, min(5, len(cv_groups)))]
    folds = []
    for held_out in family_folds:
        held = set(str(x) for x in held_out)
        fitting = [i for i, _case in enumerate(train) if cv_group_by_row[i] not in held]
        validation = [i for i, _case in enumerate(train) if cv_group_by_row[i] in held]
        folds.append((fitting, validation))
    cv = []
    for cvalue in C_VALUES:
        scores = []
        for fitting, validation in folds:
            w, mean, aux = fit([train[i] for i in fitting], [y[i] for i in fitting], cvalue)
            preds = predict([train[i] for i in validation], w, mean, aux["scale"])
            # Candidate keys differ by question, so pooled class-index F1 would
            # accidentally score option position. Selection is family-level exact
            # choice accuracy, not an index- or text-derived shared class label.
            correct = [train[i]["keys"][y[i]] == pred["selected"] for i, pred in zip(validation, preds)]
            scores.append(float(np.mean(correct)))
        cv.append({"C": cvalue, "held_out_family_accuracy": scores, "mean_family_accuracy": float(np.mean(scores))})
    best = max(cv, key=lambda x: (x["mean_family_accuracy"], -x["C"]))["C"]
    w, mean, aux = fit(train, y, best)
    predictions = predict(test, w, mean, aux["scale"])
    args.out.mkdir(parents=True, exist_ok=True)
    pred_path = args.out / "predictions.jsonl"
    pred_path.write_text("".join(json.dumps(x, sort_keys=True, allow_nan=False) + "\n" for x in predictions), encoding="utf-8")
    # Persist predictions before any test target/reference access.
    metrics = evaluate(test_rows, test, predictions)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.out / "models.json").write_text(json.dumps({"model": trmodel, "C": best,
        "coefficients": w.tolist(), "feature_mean": mean.tolist(), "feature_scale": aux["scale"].tolist(),
        "feature_order": ["query", "candidate", "query*candidate", "abs(query-candidate)"],
        "objective": "per-question softmax cross-entropy + L2",
        "fit_diagnostics": {k: v for k, v in aux.items() if k != "scale"}}, indent=2) + "\n", encoding="utf-8")
    metadata = {"protocol": "shared question-conditioned candidate scorer; independently train one model per embedding model",
        "model": trmodel, "train_count": len(train), "test_count": len(test), "train_family_count": len(families),
        "train_ids_sha256": hashlib.sha256("\n".join(str(r["id"]) for r in train_rows).encode()).hexdigest(),
        "test_ids_sha256": hashlib.sha256("\n".join(str(r["id"]) for r in test_rows).encode()).hexdigest(),
        "train_test_split_audit": {"family_overlap_count": len(train_families & test_families),
            "scenario_near_duplicate_threshold": 0.88, "cross_split_scenario_near_duplicate_count": 0,
            "within_train_cross_family_context_links": train_context_links,
            "within_train_cv_component_count": len(cv_groups),
            "within_train_cross_family_context_link_count": len(train_context_links),
            "train_domain_count": len(train_domains), "test_domain_count": len(test_domains),
            "domain_overlap": sorted(train_domains & test_domains),
            "domain_overlap_count": len(train_domains & test_domains),
            "domain_policy": "allowed by LOQFO design; this is not leave-one-domain-out evaluation"},
        "input_sha256": {str(x): sha(x) for x in (args.train, args.test, args.train_embeddings, args.test_embeddings)},
        "output_sha256": {name: sha(args.out / name) for name in ("predictions.jsonl", "metrics.json", "models.json")},
        "embedding_manifests": {"train": train_manifest, "test": test_manifest},
        "trainer_sha256": sha(Path(__file__)), "python": sys.version, "numpy": np.__version__,
        "selected_C": best,
        "cv": cv, "cv_protocol": "deterministic shuffled 5-fold grouped CV; family groups connected by exact/near-duplicate state+prompt context are kept intact; C selected by mean held-out-fold exact-match accuracy",
        "expected_count_enforced": args.expected_counts,
        "complete_case": {"one_question_per_case": True, "test_question_count": len(test),
                           "test_case_count": len(test), "all_test_cases_scored": len(predictions) == len(test)},
        "uncertainty": "family-cluster bootstrap 95% percentile intervals are reported for overall test accuracy/local macro-F1; only ten held-out families, so intervals are descriptive, not precise population guarantees",
        "limitations": "test family slices are held out; domains are intentionally shared under LOQFO and not a LODO estimate; one question per case; no claim fixed heads generalize; test labels used only for post-prediction metrics"}
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
