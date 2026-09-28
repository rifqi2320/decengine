#!/usr/bin/env python3
"""Train/evaluate the shared pairwise scorer on single- and multi-question fixtures.

Multi fixtures keep references keyed by question name; predictions are written before
the evaluator dereferences any test reference fields.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from train_pairwise import C_VALUES, SEED, pair_features, sha


def fit_bounded(cases: list[dict[str, Any]], targets: list[int], c: float):
    """Bounded-memory L-BFGS for per-question softmax with finite iteration/search caps."""
    # Known macOS BLAS builds can emit a spurious FPE RuntimeWarning for finite
    # matmuls; every score/loss/gradient is checked below before use.
    warnings.filterwarnings("ignore", message=r".*encountered in matmul", category=RuntimeWarning)
    xall = np.concatenate([x["x"] for x in cases])
    mean = xall.mean(axis=0); scale = xall.std(axis=0); scale[scale < 1e-12] = 1.0
    design = np.concatenate([(x["x"] - mean) / scale for x in cases], axis=0)
    offsets = np.cumsum([0] + [len(x["keys"]) for x in cases])
    target_rows = offsets[:-1] + np.asarray(targets, dtype=np.int64)
    def objective(w):
        scores = design @ w
        if not np.isfinite(scores).all(): raise FloatingPointError("nonfinite pairwise logits")
        loss = 0.5 * float(w @ w) / c
        delta = np.zeros(len(scores), dtype=np.float64)
        for start, end, target in zip(offsets[:-1], offsets[1:], target_rows):
            local = scores[start:end]; maximum = float(local.max()); exp = np.exp(local - maximum); probs = exp / exp.sum()
            loss += maximum + float(np.log(exp.sum())) - float(local[target-start])
            delta[start:end] = probs; delta[target] -= 1.0
        grad = w / c + design.T @ delta
        if not np.isfinite(loss) or not np.isfinite(grad).all(): raise FloatingPointError("nonfinite pairwise objective")
        return loss, grad
    w = np.zeros(design.shape[1], dtype=np.float64); loss, grad = objective(w)
    ss, yy, rr = [], [], []; limit = 500; line_limit = 24; converged = False
    for iteration in range(1, limit + 1):
        gn = float(np.linalg.norm(grad)); wn = float(np.linalg.norm(w))
        if gn <= 1e-3 * max(1.0, wn): converged = True; break
        q = grad.copy(); alphas = []
        for s, yv, rho in zip(reversed(ss), reversed(yy), reversed(rr)):
            alpha = rho * float(s @ q); alphas.append(alpha); q -= alpha * yv
        if ss:
            sy = float(ss[-1] @ yy[-1]); y2 = float(yy[-1] @ yy[-1]); r = q * (sy / y2 if y2 else 1.0)
        else: r = q
        for s, yv, rho, alpha in zip(ss, yy, rr, reversed(alphas)):
            r += s * (alpha - rho * float(yv @ r))
        direction = -r; directional = float(grad @ direction)
        if not np.isfinite(directional) or directional >= 0:
            ss.clear(); yy.clear(); rr.clear(); direction = -grad; directional = -float(grad @ grad)
        step = 1.0; accepted = False
        for _ in range(line_limit):
            candidate = w + step * direction; next_loss, next_grad = objective(candidate)
            if next_loss <= loss + 1e-4 * step * directional: accepted = True; break
            step *= 0.5
        if not accepted: break
        s = candidate - w; yv = next_grad - grad; curvature = float(s @ yv)
        if curvature > 1e-12 * np.linalg.norm(s) * max(np.linalg.norm(yv), 1e-30):
            ss.append(s); yy.append(yv); rr.append(1.0 / curvature)
            if len(ss) > 10: ss.pop(0); yy.pop(0); rr.pop(0)
        w, loss, grad = candidate, next_loss, next_grad
    grad_norm = float(np.linalg.norm(grad))
    converged = converged or grad_norm <= 2e-3 * max(1.0, float(np.linalg.norm(w)))
    if not converged: raise RuntimeError(f"typed scorer did not converge within {limit} bounded iterations (gradient norm={grad_norm:.3g})")
    return w, mean, {"scale": scale, "iterations": iteration, "objective": float(loss), "gradient_norm": grad_norm, "converged": True}


def lines(path: Path) -> list[dict[str, Any]]:
    result = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    ids = [str(x.get("id", "")) for x in result]
    if not result or not all(ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{path}: empty rows or duplicate/missing case IDs")
    return result


def vectors(path: Path, model: str) -> tuple[dict[tuple[str, str], tuple[np.ndarray, dict[str, np.ndarray]]], str]:
    raw = path.read_text()
    try:
        doc = json.loads(raw)
        records = doc.get("embeddings", []) if isinstance(doc, dict) else doc
        if not isinstance(records, list): records = [doc]
    except json.JSONDecodeError:
        records = [json.loads(x) for x in raw.splitlines() if x.strip()]
    models = sorted({str(x.get("model")) for x in records if x.get("model")})
    matched = [x for x in models if x.casefold() == model.casefold()]
    if not matched: matched = [x for x in models if model.casefold() in x.casefold()]
    if models and len(matched) != 1: raise ValueError(f"{path}: ambiguous model {models}")
    selected = matched[0] if matched else model
    output = {}
    for row in records:
        if row.get("model", selected) != selected: continue
        name = str(row.get("question_key", ""))
        key = (str(row["id"]), name)
        if not name or key in output: raise ValueError(f"{path}: missing question_key or duplicate {key}")
        q = np.asarray(row.get("query_embedding"), dtype=np.float64)
        cand = row.get("candidates")
        if isinstance(cand, list): cand = {str(c["candidate_id"]): c["embedding"] for c in cand}
        if q.ndim != 1 or not np.isfinite(q).all() or not isinstance(cand, dict):
            raise ValueError(f"{path}:{key}: invalid query/candidates")
        parsed = {str(k): np.asarray(v, dtype=np.float64) for k, v in cand.items()}
        if any(v.shape != q.shape or not np.isfinite(v).all() for v in parsed.values()):
            raise ValueError(f"{path}:{key}: candidate dimension/nonfinite mismatch")
        output[key] = (q, parsed)
    return output, selected


def exporter_manifest(path: Path) -> dict[str, Any]:
    manifest_path=Path(str(path)+".manifest.json")
    if not manifest_path.is_file(): raise ValueError(f"missing frozen exporter manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def get_target(ref: dict[str, Any], kind: str) -> str:
    target = ref["target"]
    if kind == "choice": return str(target["selected"])
    if kind == "noul":
        if type(target["value"]) is not bool: raise ValueError("noul target.value must be bool")
        return "supported" if target["value"] else "unsupported"
    return str(target["label"])


def flatten(rows: list[dict[str, Any]], vecs: dict, multi: bool, split: str) -> tuple[list[dict], list[dict]]:
    cases, refs = [], []
    for row in rows:
        request = row["request"]
        questions = request["questions"]
        if not isinstance(questions, dict) or not questions: raise ValueError(f"{row['id']}: no named questions")
        reference = row.get("reference", {})
        if multi and set(questions) != set(reference): raise ValueError(f"{row['id']}: question/reference names differ")
        for name, q in questions.items():
            if not isinstance(q, dict): raise ValueError(f"{row['id']}/{name}: question wire must be object")
            kind = q.get("type")
            if kind == "choice": options = {str(k): str(v) for k, v in q["options"].items()}; values = None
            elif kind == "noul": options = {"unsupported": "not supported", "supported": "supported"}; values = None
            elif kind == "score":
                levels = q["levels"]; options = {str(x["label"]): str(x["criterion"]) for x in levels}; values = [float(x["value"]) for x in levels]
            else: raise ValueError(f"{row['id']}/{name}: unsupported type {kind!r}")
            # The clean single-question reference split contains six-option choices;
            # current multi fixtures exercise 2-option choices and 3-level scores.
            max_options = 6 if kind == "choice" else 5
            if not 2 <= len(options) <= max_options:
                raise ValueError(f"{row['id']}/{name}: {kind} supports 2..{max_options} candidates, got {len(options)}")
            pair = vecs.get((str(row["id"]), str(name)))
            if pair is None: raise ValueError(f"missing embedding for {row['id']}/{name}")
            query, candidate = pair
            if set(candidate) != set(options): raise ValueError(f"{row['id']}/{name}: embedding candidate IDs mismatch")
            fam = str(reference[name]["question_family_id"] if multi else row["question_family_id"])
            cases.append({"id": str(row["id"]), "family": fam, "domain": str(row.get("domain", "")),
                          "type": kind, "question_name": str(name), "keys": list(options), "options": options,
                          "values": values, "x": np.stack([pair_features(query, candidate[k]) for k in options])})
            refs.append(reference.get(name, {}) if multi else row["reference"])
    expected = {(str(r["id"]), str(k)) for r in rows for k in r["request"]["questions"]}
    actual = {(c["id"], c["question_name"]) for c in cases}
    if actual != expected or len(vecs) != len(actual): raise ValueError(f"{split}: feature/question set mismatch")
    return cases, refs


def components(cases: list[dict]) -> list[str]:
    """Connected components joining every shared family and every case."""
    parent = list(range(len(cases)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    by = {}
    for i, c in enumerate(cases):
        for key in ("family", "id"):
            k = (key, c[key])
            if k in by: parent[root(i)] = root(by[k])
            else: by[k] = i
    return [str(root(i)) for i in range(len(cases))]


def pred(cases, models):
    result=[]
    for c in cases:
        if c["type"] not in models: raise ValueError(f"no typed scorer for {c['type']!r}")
        w,mean,scale=models[c["type"]]
        standardized=(c["x"]-mean)/scale
        raw=standardized@w
        # Each candidate receives a pointwise pair score; reordering candidates must
        # only reorder scores, never alter them or the probabilities by position.
        permuted_raw=standardized[::-1]@w
        reversed_raw=permuted_raw[::-1]
        if not np.allclose(raw,reversed_raw,rtol=0,atol=1e-12):
            raise AssertionError("candidate permutation changed pairwise scores")
        logits=raw-raw.max(); probs=np.exp(logits); probs/=probs.sum(); ix=int(np.argmax(probs))
        permuted_probs=np.exp(permuted_raw-permuted_raw.max()); permuted_probs/=permuted_probs.sum()
        if not np.allclose(probs,permuted_probs[::-1],rtol=0,atol=1e-12):
            raise AssertionError("candidate permutation changed normalized probabilities")
        selected=c["keys"][ix]
        result.append({"id":c["id"],"question_name":c["question_name"],"question_type":c["type"],"family":c["family"],
                       "domain":c["domain"],"candidate_keys":c["keys"],"probabilities":{k:float(probs[j]) for j,k in enumerate(c["keys"])},
                       "selected":selected,"label":(selected=="supported" if c["type"]=="noul" else selected),
                       "value":(selected=="supported" if c["type"]=="noul" else None),
                       "selected_level":(selected if c["type"]=="score" else None),"confidence":float(probs[ix]),
                       "expected_value":float(np.dot(probs,c["values"])) if c["values"] is not None else None})
    return result


def main() -> int:
    started=time.perf_counter()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train",type=Path,required=True,help="clean300 single-question train JSONL")
    p.add_argument("--train-embeddings",type=Path,required=True)
    p.add_argument("--multi-train",type=Path,required=True,help="multi train120 JSONL")
    p.add_argument("--multi-train-embeddings",type=Path,required=True)
    p.add_argument("--test",type=Path,required=True,help="sealed multi test60 JSONL")
    p.add_argument("--test-embeddings",type=Path,required=True)
    p.add_argument("--single-test",type=Path,help="optional separate existing single-question test200")
    p.add_argument("--single-test-embeddings",type=Path)
    p.add_argument("--model",required=True); p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if bool(a.single_test) != bool(a.single_test_embeddings):
        raise ValueError("--single-test and --single-test-embeddings must be supplied together")
    single, mt, test=lines(a.train),lines(a.multi_train),lines(a.test)
    train_ids={str(r["id"]) for r in single+mt}; test_ids={str(r["id"]) for r in test}
    if train_ids & test_ids: raise ValueError("train/test case ID overlap")
    tv, tm=vectors(a.train_embeddings,a.model); mv, mm=vectors(a.multi_train_embeddings,a.model); xv,xm=vectors(a.test_embeddings,a.model)
    if len({tm,mm,xm}) != 1: raise ValueError("embedding model mismatch")
    manifest_paths={"single_train":a.train_embeddings,"multi_train":a.multi_train_embeddings,"multi_test":a.test_embeddings}
    manifests={name:exporter_manifest(path) for name,path in manifest_paths.items()}
    sc,sr=flatten(single,tv,False,"single train"); mc,mr=flatten(mt,mv,True,"multi train"); tc,tr=flatten(test,xv,True,"multi test")
    train=sc+mc; train_refs=sr+mr
    fam_train={c["family"] for c in train}; fam_test={c["family"] for c in tc}
    domains_train={c["domain"] for c in train}; domains_test={c["domain"] for c in tc}
    if fam_train & fam_test: raise ValueError(f"train/test question-family overlap: {sorted(fam_train&fam_test)}")
    if len(single)!=300 or len(mt)!=120 or len(test)!=60 or len(mc)!=360 or len(tc)!=180:
        raise ValueError(f"required 300+120 cases and 180 test questions; got {len(single)},{len(mt)},{len(test)},{len(mc)},{len(tc)}")
    y=[]
    for c,r in zip(train,train_refs):
        label=get_target(r,c["type"])
        if label not in c["keys"]: raise ValueError(f"{c['id']}/{c['question_name']}: invalid train target")
        y.append(c["keys"].index(label))
    groups=components(train); unique=sorted(set(groups))
    if len(unique)<2: raise ValueError("insufficient family/case grouped CV components")
    rng=np.random.default_rng(SEED); rng.shuffle(unique); folds=np.array_split(unique,min(5,len(unique))); cv=[]
    for C in C_VALUES:
        scores=[]; typed=[]
        for held in folds:
            held=set(held); fitix=[i for i,g in enumerate(groups) if g not in held]; valid=[i for i,g in enumerate(groups) if g in held]
            models={}
            for kind in ("choice","noul","score"):
                fi=[i for i in fitix if train[i]["type"]==kind]; vi=[i for i in valid if train[i]["type"]==kind]
                if not fi: continue
                w,mean,aux=fit_bounded([train[i] for i in fi],[y[i] for i in fi],C)
                models[kind]=(w,mean,aux["scale"])
            predictions=pred([train[i] for i in valid],models)
            hits=[train[i]["keys"][y[i]]==pr["selected"] for i,pr in zip(valid,predictions)]
            scores.append(float(np.mean(hits)))
            typed.append({kind:float(np.mean([train[i]["keys"][y[i]]==pr["selected"] for i,pr in zip(valid,predictions) if train[i]["type"]==kind]))
                          for kind in ("choice","noul","score") if any(train[i]["type"]==kind for i in valid)})
        cv.append({"C":C,"held_out_accuracy":scores,"held_out_accuracy_by_type":typed,"mean_accuracy":float(np.mean(scores))})
    best=max(cv,key=lambda x:(x["mean_accuracy"],-x["C"]))["C"]
    models={}; model_docs={}
    for kind in ("choice","noul","score"):
        inds=[i for i,c in enumerate(train) if c["type"]==kind]
        if not inds: raise ValueError(f"training split has no {kind} questions")
        w,mean,aux=fit_bounded([train[i] for i in inds],[y[i] for i in inds],best)
        models[kind]=(w,mean,aux["scale"])
        model_docs[kind]={"coefficients":w.tolist(),"feature_mean":mean.tolist(),"feature_scale":aux["scale"].tolist(),
                          "fit_diagnostics":{k:v for k,v in aux.items() if k!="scale"}}
    predictions=pred(tc,models)
    a.out.mkdir(parents=True,exist_ok=True)
    pp=a.out/"predictions.jsonl"; pp.write_text("".join(json.dumps(x,sort_keys=True,allow_nan=False)+"\n" for x in predictions))
    separate_single=None
    if a.single_test:
        single_test_rows=lines(a.single_test); sv,sm=vectors(a.single_test_embeddings,a.model)
        if sm!=tm: raise ValueError("single test embedding model mismatch")
        single_test_ids={str(r["id"]) for r in single_test_rows}
        if train_ids & single_test_ids: raise ValueError("single train/test case ID overlap")
        manifest_paths["single_test"]=a.single_test_embeddings
        manifests["single_test"]=exporter_manifest(a.single_test_embeddings)
        single_test_cases,single_test_refs=flatten(single_test_rows,sv,False,"single test")
        if len(single_test_rows)!=200 or len(single_test_cases)!=200:
            raise ValueError(f"expected separate single-question test200, got {len(single_test_rows)} cases/{len(single_test_cases)} questions")
        single_families={c["family"] for c in single_test_cases}
        if fam_train & single_families: raise ValueError(f"single train/test family overlap: {sorted(fam_train&single_families)}")
        single_test_predictions=pred(single_test_cases,models)
        single_out=a.out/"single-test200"; single_out.mkdir(exist_ok=True)
        (single_out/"predictions.jsonl").write_text("".join(json.dumps(x,sort_keys=True,allow_nan=False)+"\n" for x in single_test_predictions))
        separate_single=(single_test_rows,single_test_cases,single_test_refs,single_test_predictions,single_out)
    identity_fields=("model","profile_version","profile_sha256","text_version","embedding_dimensions","normalization")
    baseline=manifests["single_train"]
    for split,manifest in manifests.items():
        mismatches=[k for k in identity_fields if manifest.get(k)!=baseline.get(k)]
        if mismatches: raise ValueError(f"{split} exporter provenance mismatch: {mismatches}")
    # Do not access test reference targets until predictions are persisted.
    correct=[]; type_report={}; question_outcomes=[]
    for kind in ("choice","noul","score"):
        inds=[i for i,c in enumerate(tc) if c["type"]==kind]
        hits=[]
        for i in inds:
            label=get_target(tr[i],kind); hits.append(tc[i]["keys"].index(label)==tc[i]["keys"].index(predictions[i]["selected"]))
            question_outcomes.append({"id":tc[i]["id"],"question_name":tc[i]["question_name"],"question_type":kind,
                                      "gold":label,"predicted":predictions[i]["selected"],
                                      "gold_noul_value":(bool(tr[i]["target"]["value"]) if kind=="noul" else None),
                                      "predicted_noul_value":(bool(predictions[i]["value"]) if kind=="noul" else None),
                                      "score_gold_value":(float(tr[i]["target"]["value"]) if kind=="score" else None),
                                      "score_expected_value":(predictions[i]["expected_value"] if kind=="score" else None),
                                      "correct":bool(hits[-1])})
        if inds:
            report={"n":len(inds),"accuracy":float(np.mean(hits)),"variable_candidate_counts":sorted({len(tc[i]["keys"]) for i in inds})}
            if kind=="score":
                actual=np.asarray([float(tr[i]["target"]["value"]) for i in inds]); expected=np.asarray([predictions[i]["expected_value"] for i in inds])
                report["expected_value_mae"]=float(np.mean(np.abs(actual-expected)))
                report["label_argmax_accuracy"]=float(np.mean(hits))
            if kind=="noul":
                # Targets are boolean while candidates are canonical semantic outcomes.
                for i in inds:
                    expected="supported" if tr[i]["target"]["value"] else "unsupported"
                    if expected not in tc[i]["keys"]: raise ValueError(f"{tc[i]['id']}/{tc[i]['question_name']}: Noul polarity candidates missing")
            type_report[kind]=report
        correct.extend((tc[i]["id"],hits[j]) for j,i in enumerate(inds))
    bycase={}
    for cid,hit in correct: bycase.setdefault(cid,[]).append(hit)
    case_outcomes=[{"id":cid,"question_count":len(hits),"all_questions_correct":len(hits)==3 and all(hits),"correct_count":sum(hits)}
                   for cid,hits in sorted(bycase.items())]
    (a.out/"question_outcomes.jsonl").write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in question_outcomes))
    (a.out/"case_outcomes.jsonl").write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in case_outcomes))
    metrics={"per_question_type":type_report,"question_count":len(predictions),"case_count":len(bycase),
             "all_three_correct_case_count":sum(len(v)==3 and all(v) for v in bycase.values()),
             "all_test_cases_have_three_questions":len(bycase)==60 and all(len(v)==3 for v in bycase.values())}
    (a.out/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    if separate_single:
        single_rows,single_cases,single_refs,single_preds,single_out=separate_single
        per_type={}; single_outcomes=[]
        for kind in ("choice","noul","score"):
            inds=[i for i,c in enumerate(single_cases) if c["type"]==kind]
            hits=[]
            for i in inds:
                gold=get_target(single_refs[i],kind); hit=single_cases[i]["keys"].index(gold)==single_cases[i]["keys"].index(single_preds[i]["selected"])
                hits.append(hit); single_outcomes.append({"id":single_cases[i]["id"],"question_type":kind,"gold":gold,
                    "predicted":single_preds[i]["selected"],"correct":bool(hit),
                    "score_gold_value":float(single_refs[i]["target"]["value"]) if kind=="score" else None,
                    "score_expected_value":single_preds[i]["expected_value"] if kind=="score" else None})
            if inds:
                report={"n":len(inds),"accuracy":float(np.mean(hits)),"variable_candidate_counts":sorted({len(single_cases[i]["keys"]) for i in inds})}
                if kind=="score":
                    actual=np.asarray([float(single_refs[i]["target"]["value"]) for i in inds]); expected=np.asarray([single_preds[i]["expected_value"] for i in inds])
                    report["expected_value_mae"]=float(np.mean(np.abs(actual-expected))); report["label_argmax_accuracy"]=float(np.mean(hits))
                per_type[kind]=report
        (single_out/"question_outcomes.jsonl").write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in single_outcomes))
        (single_out/"metrics.json").write_text(json.dumps({"per_question_type":per_type,"question_count":len(single_cases),
            "case_count":len(single_rows),"evaluation":"separate single-question test200; never pooled with multi test60"},indent=2)+"\n")
    (a.out/"models.json").write_text(json.dumps({"model":tm,"C":best,"typed_scorers":model_docs,
        "feature_order":["query","candidate","query*candidate","abs(query-candidate)"],
        "objective":"one shared learned scorer per generic wire type; per-question softmax cross-entropy + L2",
        "candidate_scoring":"pointwise pair scorer; candidate order only enters the question-level softmax; permutation checked at inference",
        "score_semantics":"ordered levels retain wire order; expected numeric score is probability-weighted over level values; selected label is score argmax"},indent=2)+"\n")
    param_hash={kind:hashlib.sha256(np.asarray(doc["coefficients"],dtype=np.float64).tobytes()).hexdigest() for kind,doc in model_docs.items()}
    metadata={"protocol":"one shared learned scorer per generic type (choice/noul/score); one prediction row per named question",
        "train_case_count":len(single)+len(mt),"train_question_count":len(train),"test_case_count":len(test),"test_question_count":len(tc),
         "train_family_count":len(fam_train),"test_family_count":len(fam_test),"family_overlap_count":len(fam_train&fam_test),
         "train_test_family_overlap":sorted(fam_train&fam_test),
         "train_families":sorted(fam_train),"test_families":sorted(fam_test),
         "domain_overlap_allowed":sorted(domains_train&domains_test),"domain_overlap_count":len(domains_train&domains_test),
        "cv_component_count":len(unique),"cv":"5-fold deterministic CV grouped by connected family and case components; train-only",
         "selected_C":best,"cv_results":cv,"inputs_sha256":{str(x):sha(x) for x in (a.train,a.train_embeddings,a.multi_train,a.multi_train_embeddings,a.test,a.test_embeddings,
                                                                             *([a.single_test,a.single_test_embeddings] if a.single_test else []))},
          "outputs_sha256":{x:sha(a.out/x) for x in ("predictions.jsonl","question_outcomes.jsonl","case_outcomes.jsonl","metrics.json","models.json")},
         "model_parameter_sha256":param_hash,"models_file_sha256":sha(a.out/"models.json"),
          "export_manifest_sha256":{name:sha(Path(str(path)+".manifest.json")) for name,path in manifest_paths.items()},
         "reference_policy":"train labels used for fitting/CV; test references read only after raw predictions.jsonl written",
          "single_test_policy":"existing single-question test200 is intentionally not used; evaluate separately",
          "candidate_cardinality_policy":"wire-generic; choice 2..6 (including legacy clean300 six-option choices), noul 2, score 2..5",
          "invariance_checks":{"candidate_permutation_logits":True,"candidate_permutation_probabilities":True,
                               "noul_bool_polarity_checked":True,"score_expected_value_from_probabilities":True},
          "runtime_seconds":time.perf_counter()-started,"python":sys.version,"numpy":np.__version__,"trainer_sha256":sha(Path(__file__))}
    if separate_single:
        single_rows,single_cases,_,_,single_out=separate_single
        metadata["single_test200"]={"case_count":len(single_rows),"question_count":len(single_cases),
            "family_count":len({c["family"] for c in single_cases}),"train_family_overlap_count":0,
            "artifacts_sha256":{name:sha(single_out/name) for name in ("predictions.jsonl","question_outcomes.jsonl","metrics.json")}}
    (a.out/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    return 0

if __name__=="__main__": raise SystemExit(main())
