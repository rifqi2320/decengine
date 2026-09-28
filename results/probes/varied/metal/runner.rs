//! End-to-end frozen typed inference. Only query/candidate text crosses into the embedder;
//! normalized embeddings and all pair/scorer math remain MLX arrays through softmax.
use anyhow::{Context, Result, ensure};
use decengine_engine::DecisionEngine;
use decengine_models::ModelStore;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufRead, BufReader, Write},
    path::Path,
    time::Instant,
};

fn sha(path: &Path) -> Result<String> {
    Ok(hex::encode(sha2::Sha256::digest(fs::read(path)?)))
}
use sha2::Digest;
fn rep(t: &str, key: &str, val: &str) -> String {
    t.replace(key, val)
}
const KINDS: [&str; 3] = ["choice", "noul", "score"];

pub fn run(
    store: &ModelStore,
    model: &str,
    run_dir: &Path,
    input: &Path,
    output: &Path,
) -> Result<()> {
    let models_path = run_dir.join("models.json");
    let metadata_path = run_dir.join("metadata.json");
    let models_sha = sha(&models_path)?;
    let metadata_sha = sha(&metadata_path)?;
    let saved: Value = serde_json::from_slice(&fs::read(&models_path)?)?;
    let metadata: Value = serde_json::from_slice(&fs::read(&metadata_path)?)?;
    ensure!(
        saved["model"].as_str() == Some(model),
        "frozen scorer model mismatch"
    );
    let typed = saved["typed_scorers"]
        .as_object()
        .context("missing typed_scorers")?;
    ensure!(
        typed.len() == KINDS.len() && KINDS.iter().all(|k| typed.contains_key(*k)),
        "frozen scorer types mismatch"
    );
    let approval = serde_json::from_slice::<Value>(&fs::read(
        run_dir.parent().unwrap().join("selection-freeze.json"),
    )?)?;
    let key = if model.starts_with("Qwen/") {
        "qwen"
    } else {
        "harrier"
    };
    let approved = &approval["approved_runs"][key];
    ensure!(
        approval["selection_status"] == "frozen" && approved["selection_status"] == "frozen",
        "selection not frozen"
    );
    ensure!(
        approved["models_sha256"].as_str() == Some(&models_sha)
            && approved["metadata_sha256"].as_str() == Some(&metadata_sha),
        "frozen artifact hash mismatch"
    );
    ensure!(
        approved["run_dir"]
            .as_str()
            .map(|x| Path::new(x).canonicalize().ok())
            == Some(Some(run_dir.canonicalize()?)),
        "approved run path mismatch"
    );

    let resolved = store.resolve_installed(model)?;
    let model_root = resolved
        .root
        .as_ref()
        .context("installed model has no root")?;
    let tokenizer = tokenizers::Tokenizer::from_file(model_root.join("tokenizer.json"))
        .map_err(|error| anyhow::anyhow!("load installed tokenizer: {error}"))?;
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
                .with_context(|| format!("parse input row {}", i + 1))
                .map(Some)
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    ensure!(!rows.is_empty(), "fixture has no cases");
    let mut ids = BTreeSet::new();
    let profile = &resolved.profile;
    let mut out = fs::File::create(output)?;
    let mut timings = fs::File::create(format!("{}.timings.jsonl", output.display()))?;
    let mut predictions = Vec::new();
    let mut warmed = BTreeSet::new();
    let mut case_totals: BTreeMap<String, u128> = BTreeMap::new();
    for row in &rows {
        let id = row["id"].as_str().context("missing fixture id")?;
        ensure!(ids.insert(id.to_owned()), "duplicate id {id}");
        let state = serde_json::to_string(&row["request"]["state"])?;
        let qs = row["request"]["questions"]
            .as_object()
            .context("questions must be map")?;
        for (name, q) in qs {
            let kind = q["type"].as_str().context("missing question type")?;
            ensure!(KINDS.contains(&kind), "unknown question type {kind}");
            let prompt = q["prompt"].as_str().context("missing prompt")?;
            let instruction_kind = match kind {
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
            let inst = rep(
                &rep(&profile.query_instruction_template, "{prompt}", prompt),
                "{kind}",
                instruction_kind,
            );
            let raw = format!("Decision: {prompt}\nState: {state}");
            let query = rep(
                &rep(&profile.query_template, "{instruction}", &inst),
                "{text}",
                &raw,
            );
            let mut keys = Vec::new();
            let mut texts = Vec::new();
            let mut values: Vec<f64> = Vec::new();
            match kind {
                "choice" => {
                    for (label, criterion) in q["options"]
                        .as_object()
                        .context("choice options must map")?
                    {
                        keys.push(label.clone());
                        let c = rep(
                            &rep(&profile.choice_candidate_template, "{label}", label),
                            "{criterion}",
                            criterion.as_str().unwrap_or(""),
                        );
                        texts.push(rep(&profile.candidate_template, "{text}", &c));
                    }
                }
                "score" => {
                    for level in q["levels"].as_array().context("score levels must array")? {
                        let label = level["label"]
                            .as_str()
                            .context("score label missing")?
                            .to_owned();
                        keys.push(label.clone());
                        values.push(level["value"].as_f64().context("score value missing")?);
                        let c = rep(
                            &rep(&profile.score_candidate_template, "{label}", &label),
                            "{criterion}",
                            level["criterion"].as_str().unwrap_or(""),
                        );
                        texts.push(rep(&profile.candidate_template, "{text}", &c));
                    }
                }
                _ => {
                    for (key, t) in [
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
                        keys.push(key.to_owned());
                        texts.push(rep(&profile.candidate_template, "{text}", &t));
                    }
                }
            }
            ensure!(
                keys.len() >= 2 && keys.len() == texts.len(),
                "bad candidate set"
            );
            let query_token_count = tokenizer
                .encode(query.as_str(), true)
                .map_err(|error| anyhow::anyhow!("tokenize query: {error}"))?
                .get_ids()
                .len();
            let candidate_token_counts = texts
                .iter()
                .map(|text| {
                    tokenizer
                        .encode(text.as_str(), true)
                        .map(|encoding| encoding.get_ids().len())
                        .map_err(|error| anyhow::anyhow!("tokenize candidate: {error}"))
                })
                .collect::<Result<Vec<_>>>()?;
            ensure!(
                query_token_count > 0
                    && query_token_count <= profile.max_length
                    && candidate_token_counts
                        .iter()
                        .all(|n| *n > 0 && *n <= profile.max_length),
                "query or candidate token count outside profile max_length"
            );
            let total_candidate_tokens = candidate_token_counts.iter().sum::<usize>();
            let spec = &typed[kind];
            let coeff = spec["coefficients"]
                .as_array()
                .context("coefficients must array")?
                .iter()
                .map(|x| x.as_f64().context("invalid coefficient"))
                .collect::<Result<Vec<_>>>()?;
            let mean = spec["feature_mean"]
                .as_array()
                .context("feature_mean must array")?
                .iter()
                .map(|x| x.as_f64().context("invalid mean"))
                .collect::<Result<Vec<_>>>()?;
            let scale = spec["feature_scale"]
                .as_array()
                .context("feature_scale must array")?
                .iter()
                .map(|x| x.as_f64().context("invalid scale"))
                .collect::<Result<Vec<_>>>()?;
            if warmed.insert(kind.to_owned()) {
                // Warm each typed scorer and the model path before recording its first case.
                let _ = engine.varied_pair_probabilities_timed(
                    handle, kind, &query, &texts, &coeff, &mean, &scale,
                )?;
            }
            let question_start = Instant::now();
            let (probs, stage) = engine.varied_pair_probabilities_timed(
                handle, kind, &query, &texts, &coeff, &mean, &scale,
            )?;
            let total_ns = question_start.elapsed().as_nanos();
            *case_totals.entry(id.to_owned()).or_default() += total_ns;
            let ix = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
            let selected = keys[ix].clone();
            let expected = if kind == "score" {
                Some(
                    probs
                        .iter()
                        .zip(&values)
                        .map(|(p, v)| f64::from(*p) * v)
                        .sum::<f64>(),
                )
            } else {
                None
            };
            let prediction = json!({"id":id,"domain":row["domain"],"question_name":name,"question_type":kind,"candidate_keys":keys,
                "probabilities":keys.iter().enumerate().map(|(i,k)|(k.clone(),json!(probs[i]))).collect::<serde_json::Map<String,Value>>(),
                "selected":selected,"label":if kind=="noul" {json!(selected=="supported")} else {json!(selected)},
                "value":if kind=="noul" {json!(selected=="supported")} else {Value::Null},"selected_level":if kind=="score" {json!(selected)} else {Value::Null},
                "confidence":probs[ix],"expected_value":expected});
            serde_json::to_writer(&mut out, &prediction)?;
            out.write_all(b"\n")?;
            serde_json::to_writer(
                &mut timings,
                &json!({"id":id,"question_name":name,"tokenization_ns":stage.tokenization_ns,"embedding_ns":stage.embedding_ns,"gpu_scorer_ns":stage.classifier_ns,"question_total_ns":total_ns,"query_token_count":query_token_count,"candidate_token_counts":candidate_token_counts,"total_candidate_tokens":total_candidate_tokens}),
            )?;
            timings.write_all(b"\n")?;
            predictions.push(prediction);
        }
    }
    out.flush()?;
    timings.flush()?;
    let mut case_file = fs::File::create(format!("{}.case-timings.jsonl", output.display()))?;
    for (id, ns) in &case_totals {
        serde_json::to_writer(&mut case_file, &json!({"id":id,"case_total_ns":ns}))?;
        case_file.write_all(b"\n")?;
    }
    case_file.flush()?;
    eprintln!(
        "Metal inference {} typed questions; model hashes {} {}",
        predictions.len(),
        models_sha,
        sha(input)?
    );
    let _ = metadata;
    Ok(())
}
