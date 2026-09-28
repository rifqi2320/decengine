//! GPU-resident inference for pre-frozen prompt-study text/scorer variants.
use anyhow::{Context, Result, ensure};
use decengine_engine::DecisionEngine;
use decengine_models::ModelStore;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufRead, BufReader, Write},
    path::Path,
    time::Instant,
};

const VARIANTS: [&str; 4] = [
    "v2-baseline",
    "readable-state",
    "rule-in-candidate",
    "joint-context-candidate",
];
const KINDS: [&str; 3] = ["choice", "noul", "score"];
fn sha(path: &Path) -> Result<String> {
    Ok(hex::encode(Sha256::digest(fs::read(path)?)))
}
fn replace(template: &str, key: &str, value: &str) -> String {
    template.replace(key, value)
}
fn canonical(v: &Value) -> String {
    serde_json::to_string(v).expect("JSON serialization")
}
fn readable(v: &Value, depth: usize, out: &mut String) {
    let pad = "  ".repeat(depth);
    match v {
        Value::Object(map) => {
            for (key, value) in map {
                match value {
                    Value::Object(_) | Value::Array(_) => {
                        out.push_str(&format!("{pad}{key}:\n"));
                        readable(value, depth + 1, out);
                    }
                    _ => out.push_str(&format!("{pad}{key}: {}\n", canonical(value))),
                }
            }
        }
        Value::Array(items) => {
            for value in items {
                match value {
                    Value::Object(_) | Value::Array(_) => {
                        out.push_str(&format!("{pad}-\n"));
                        readable(value, depth + 1, out);
                    }
                    _ => out.push_str(&format!("{pad}- {}\n", canonical(value))),
                }
            }
        }
        _ => out.push_str(&format!("{pad}{}\n", canonical(v))),
    }
}

pub fn run(
    store: &ModelStore,
    model: &str,
    variant: &str,
    input: &Path,
    output: &Path,
    bundle_path: &Path,
    study_run: &Path,
    head_mode: &str,
) -> Result<()> {
    ensure!(
        VARIANTS.contains(&variant),
        "unknown prompt-study variant {variant}"
    );
    let mk = if model.starts_with("Qwen/") {
        "qwen"
    } else {
        "harrier"
    };
    let frozen = study_run.join("frozen").join(mk);
    let selection_path = frozen.join("selection.json");
    let selection: Value = serde_json::from_slice(&fs::read(&selection_path)?)?;
    ensure!(
        selection["selected_variant"] == "v2-baseline",
        "outer prompt selection is not frozen to v2-baseline"
    );
    let cv_config_path = study_run
        .join("cv")
        .join(mk)
        .join(variant)
        .join("frozen-config.json");
    let cv_config: Value = serde_json::from_slice(&fs::read(&cv_config_path)?)?;
    let frozen_models_path = frozen.join("models.json");
    let frozen_models: Value = serde_json::from_slice(&fs::read(&frozen_models_path)?)?;
    let bundle: Value = serde_json::from_slice(&fs::read(bundle_path)?)?;
    ensure!(
        bundle["model"].as_str() == Some(model) && bundle["variant"].as_str() == Some(variant),
        "scorer bundle model/variant mismatch"
    );
    // Earlier fixed train-only per-variant bundles predate the explicit `head_mode`
    // field; their exact SHA-256 is bound by the secondary CPU handoff. The CLI mode
    // remains explicit so these legacy bundles cannot be mistaken for shared heads.
    let bundle_mode = bundle["head_mode"].as_str().unwrap_or("per-variant");
    ensure!(bundle_mode == head_mode, "head mode/bundle mismatch");
    match head_mode {
        "shared-v2-baseline" => {
            ensure!(
                bundle["C"].as_f64() == frozen_models["C"].as_f64(),
                "shared v2 head C mismatch"
            );
            ensure!(
                bundle["provenance"]["models_sha256"].as_str()
                    == Some(sha(&frozen_models_path)?.as_str()),
                "shared v2 head frozen models hash mismatch"
            );
            for kind in KINDS {
                for field in ["weights", "mean", "scale"] {
                    ensure!(
                        bundle["scorers"][kind][field] == frozen_models["scorers"][kind][field],
                        "shared v2 head {kind}/{field} differs from reference frozen arrays"
                    );
                }
            }
        }
        "per-variant" => {
            ensure!(
                bundle["C"].as_f64() == cv_config["selected"]["C"].as_f64(),
                "per-variant head C differs from its train-only frozen CV configuration"
            );
            ensure!(
                bundle["training_scope"].as_str()
                    == Some(
                        "clean300 + multi120 only; targets read only from those train fixtures"
                    ),
                "per-variant scorer bundle training scope invalid"
            );
        }
        _ => anyhow::bail!("unsupported prompt-study head mode {head_mode:?}"),
    }
    ensure!(
        bundle["provenance"]["models_sha256"].as_str() == Some(sha(&frozen_models_path)?.as_str())
            && bundle["provenance"]["selection_sha256"].as_str()
                == Some(sha(&selection_path)?.as_str())
            && bundle["provenance"]["variant_cv_config_sha256"].as_str()
                == Some(sha(&cv_config_path)?.as_str()),
        "scorer bundle provenance does not match frozen prompt-study artifacts"
    );
    ensure!(
        bundle["scorers"].as_object().map(|m| m.len()) == Some(3)
            && KINDS.iter().all(|k| bundle["scorers"].get(*k).is_some()),
        "bundle needs all three generic typed scorers"
    );
    let declared_sha = fs::read_to_string(bundle_path.with_extension("sha256"))?;
    let bundle_sha256 = sha(bundle_path)?;
    ensure!(
        declared_sha.split_whitespace().next() == Some(bundle_sha256.as_str()),
        "scorer bundle SHA-256 sidecar mismatch"
    );
    ensure!(
        bundle["provenance"]["fixture_sha256"].as_str() == Some(sha(input)?.as_str()),
        "input does not match the pre-reference-frozen prompt-study fixture"
    );

    let resolved = store.resolve_installed(model)?;
    let model_root = resolved
        .root
        .as_ref()
        .context("installed model root missing")?;
    ensure!(
        resolved.profile.pooling == "last_token" && resolved.profile.normalize == "l2",
        "prompt-study requires last-token L2 pooling"
    );
    let install: Value = serde_json::from_slice(&fs::read(model_root.join("install.json"))?)?;
    ensure!(
        bundle["provenance"]["checkpoint_files"] == install["files"],
        "installed checkpoint hashes differ from frozen CPU-reference metadata"
    );
    let profile_hash = hex::encode(Sha256::digest(serde_json::to_vec(&resolved.profile)?));
    ensure!(
        bundle["provenance"]["profile_sha256"].as_str() == Some(profile_hash.as_str()),
        "profile serialization hash mismatch"
    );
    ensure!(
        bundle["provenance"]["tokenizer_sha256"].as_str()
            == Some(sha(&model_root.join("tokenizer.json"))?.as_str()),
        "tokenizer hash mismatch"
    );
    let tokenizer = tokenizers::Tokenizer::from_file(model_root.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("load installed tokenizer: {e}"))?;
    let mut engine = decengine_engine_mlx::MlxDecisionEngine::new()?;
    let handle = engine.load_model(&resolved.engine_spec())?;
    let rows = BufReader::new(fs::File::open(input)?)
        .lines()
        .enumerate()
        .map(|(i, l)| {
            let l = l?;
            if l.trim().is_empty() {
                return Ok(None);
            }
            serde_json::from_str::<Value>(&l)
                .with_context(|| format!("parse holdout row {}", i + 1))
                .map(Some)
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    ensure!(
        rows.len() == 60,
        "expected fresh60 holdout; got {} cases",
        rows.len()
    );
    let mut seen = BTreeSet::new();
    let mut question_count = 0usize;
    let mut predictions_out = fs::File::create(output)?;
    let mut timings_out = fs::File::create(format!("{}.timings.jsonl", output.display()))?;
    let mut warmed = BTreeSet::new();
    let mut case_totals = BTreeMap::<String, u128>::new();
    let profile = &resolved.profile;
    for row in &rows {
        let id = row["id"].as_str().context("holdout case ID missing")?;
        ensure!(seen.insert(id.to_owned()), "duplicate holdout ID {id}");
        let request = &row["request"];
        let state = request.get("state").context("request.state missing")?;
        let state_text = if variant == "readable-state" {
            let mut s = String::new();
            readable(state, 0, &mut s);
            s
        } else {
            canonical(state)
        };
        let questions = request["questions"]
            .as_object()
            .context("questions must be named map")?;
        ensure!(!questions.is_empty(), "case {id} has no questions");
        for (name, q) in questions {
            let kind = q["type"].as_str().context("question type missing")?;
            ensure!(KINDS.contains(&kind), "unknown wire type {kind}");
            let prompt = q["prompt"].as_str().context("question prompt missing")?;
            let kind_rule = match kind {
                "choice" => {
                    "select the single candidate that best satisfies the question using the reported state and candidate criteria; do not infer facts absent from the case"
                }
                "noul" => {
                    "decide whether the proposition asked about is supported by the reported state; evaluate the complete proposition from evidence and do not infer facts absent from the case"
                }
                _ => {
                    "select the rubric level whose criterion best matches the reported state and requested rating; use the level definitions and do not infer facts absent from the case"
                }
            };
            let instruction = replace(
                &replace(&profile.query_instruction_template, "{prompt}", prompt),
                "{kind}",
                kind_rule,
            );
            let raw = format!("Decision: {prompt}\nState: {state_text}");
            let query = replace(
                &replace(&profile.query_template, "{instruction}", &instruction),
                "{text}",
                &raw,
            );
            let mut keys = Vec::<String>::new();
            let mut texts = Vec::<String>::new();
            let mut score_values = Vec::<f64>::new();
            match kind {
                "choice" => {
                    for (label, criterion) in
                        q["options"].as_object().context("choice options missing")?
                    {
                        let c = replace(
                            &replace(&profile.choice_candidate_template, "{label}", label),
                            "{criterion}",
                            criterion.as_str().unwrap_or(""),
                        );
                        keys.push(label.clone());
                        texts.push(replace(&profile.candidate_template, "{text}", &c));
                    }
                }
                "score" => {
                    for level in q["levels"].as_array().context("score levels missing")? {
                        let label = level["label"]
                            .as_str()
                            .context("score label missing")?
                            .to_owned();
                        let c = replace(
                            &replace(&profile.score_candidate_template, "{label}", &label),
                            "{criterion}",
                            level["criterion"].as_str().unwrap_or(""),
                        );
                        keys.push(label);
                        texts.push(replace(&profile.candidate_template, "{text}", &c));
                        score_values.push(
                            level["value"]
                                .as_f64()
                                .context("score numeric value missing")?,
                        );
                    }
                }
                _ => {
                    for (label, candidate) in [
                        (
                            "unsupported",
                            format!(
                                "For the question ‘{prompt}’, the proposition is not supported by the reported state; the evidence does not establish that this outcome applies."
                            ),
                        ),
                        (
                            "supported",
                            format!(
                                "For the question ‘{prompt}’, the proposition is supported by the reported state; the evidence establishes that this outcome applies."
                            ),
                        ),
                    ] {
                        keys.push(label.to_owned());
                        texts.push(replace(&profile.candidate_template, "{text}", &candidate));
                    }
                }
            }
            if variant == "rule-in-candidate" || variant == "joint-context-candidate" {
                for text in &mut texts {
                    let context = if variant == "joint-context-candidate" {
                        format!(
                            "Question: {prompt}\nReported state: {state_text}\nDecision rule: {kind_rule}\n"
                        )
                    } else {
                        format!("Question: {prompt}\nDecision rule: {kind_rule}\n")
                    };
                    *text = replace(
                        &profile.candidate_template,
                        "{text}",
                        &format!("{context}Candidate: {text}"),
                    );
                }
            }
            ensure!(
                keys.len() >= 2 && keys.len() == texts.len(),
                "invalid candidate set at {id}/{name}"
            );
            let query_tokens = tokenizer
                .encode(query.as_str(), true)
                .map_err(|e| anyhow::anyhow!("tokenize query: {e}"))?
                .get_ids()
                .len();
            let candidate_tokens = texts
                .iter()
                .map(|text| {
                    tokenizer
                        .encode(text.as_str(), true)
                        .map(|e| e.get_ids().len())
                        .map_err(|e| anyhow::anyhow!("tokenize candidate: {e}"))
                })
                .collect::<Result<Vec<_>>>()?;
            ensure!(
                query_tokens > 0
                    && query_tokens <= profile.max_length
                    && candidate_tokens
                        .iter()
                        .all(|n| *n > 0 && *n <= profile.max_length),
                "overlength or empty text at {id}/{name}"
            );
            let spec = &bundle["scorers"][kind];
            let weights = spec["weights"]
                .as_array()
                .context("weights missing")?
                .iter()
                .map(|x| x.as_f64().context("invalid weight"))
                .collect::<Result<Vec<_>>>()?;
            let mean = spec["mean"]
                .as_array()
                .context("mean missing")?
                .iter()
                .map(|x| x.as_f64().context("invalid mean"))
                .collect::<Result<Vec<_>>>()?;
            let scale = spec["scale"]
                .as_array()
                .context("scale missing")?
                .iter()
                .map(|x| x.as_f64().context("invalid scale"))
                .collect::<Result<Vec<_>>>()?;
            ensure!(
                weights.len() == 4096 && mean.len() == 4096 && scale.len() == 4096,
                "unexpected pair scorer feature dimensions"
            );
            let scorer_key = format!("{variant}/{kind}");
            if warmed.insert(kind.to_owned()) {
                let _ = engine.varied_pair_probabilities_timed(
                    handle,
                    &scorer_key,
                    &query,
                    &texts,
                    &weights,
                    &mean,
                    &scale,
                )?;
            }
            let t0 = Instant::now();
            let (probs, stage) = engine.varied_pair_probabilities_timed(
                handle,
                &scorer_key,
                &query,
                &texts,
                &weights,
                &mean,
                &scale,
            )?;
            let total = t0.elapsed().as_nanos();
            *case_totals.entry(id.to_owned()).or_default() += total;
            let selected_i = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .context("empty score distribution")?
                .0;
            let selected = keys[selected_i].clone();
            let expected = if kind == "score" {
                Some(
                    probs
                        .iter()
                        .zip(&score_values)
                        .map(|(p, v)| f64::from(*p) * v)
                        .sum::<f64>(),
                )
            } else {
                None
            };
            let prediction = json!({"id":id,"question_name":name,"question_type":kind,"variant":variant,"model":model,
                "head_mode":head_mode,"scorer_bundle_sha256":bundle_sha256,
                "candidate_keys":keys,"probabilities":keys.iter().enumerate().map(|(i,k)|(k.clone(),json!(probs[i]))).collect::<serde_json::Map<String,Value>>(),
                "selected":selected,"label":if kind=="noul"{json!(selected=="supported")}else{json!(selected)},
                "value":if kind=="noul"{json!(selected=="supported")}else{Value::Null},
                "selected_level":if kind=="score"{json!(selected)}else{Value::Null},"confidence":probs[selected_i],"expected_value":expected});
            serde_json::to_writer(&mut predictions_out, &prediction)?;
            predictions_out.write_all(b"\n")?;
            serde_json::to_writer(
                &mut timings_out,
                &json!({"id":id,"question_name":name,"variant":variant,"head_mode":head_mode,"question_total_ns":total,
                "tokenization_ns":stage.tokenization_ns,"embedding_ns":stage.embedding_ns,"gpu_scorer_ns":stage.classifier_ns,
                "query_token_count":query_tokens,"candidate_token_counts":candidate_tokens}),
            )?;
            timings_out.write_all(b"\n")?;
            question_count += 1;
        }
    }
    predictions_out.flush()?;
    timings_out.flush()?;
    let mut cases = fs::File::create(format!("{}.case-timings.jsonl", output.display()))?;
    for (id, ns) in &case_totals {
        serde_json::to_writer(&mut cases, &json!({"id":id,"case_total_ns":ns}))?;
        cases.write_all(b"\n")?;
    }
    cases.flush()?;
    ensure!(
        question_count == 180,
        "fresh holdout expected 180 typed questions, got {question_count}"
    );
    eprintln!(
        "all-MLX {variant}/{mk}: {} cases {} questions; bundle={} fixture={}",
        rows.len(),
        question_count,
        sha(bundle_path)?,
        sha(input)?
    );
    Ok(())
}
