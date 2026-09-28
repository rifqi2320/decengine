//! MLX-only feature export for the variable-question fixture.
//!
//! This intentionally exports plain vectors to disk for offline training; it is not a
//! Metal-resident classifier. Inference text construction mirrors compiler v1 templates.
use anyhow::{Context, Result, ensure};
use decengine_engine::DecisionEngine;
use decengine_models::{InstallRecord, ModelStore};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};

const TEXT_VERSION: &str = "decengine-varied-pair-text-v2-generic-types";

#[derive(Serialize)]
struct CheckpointFile {
    path: String,
    sha256: String,
    size: u64,
}

fn sha(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("read checkpoint {}", path.display()))?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

fn canonical(v: &Value) -> String {
    serde_json::to_string(v).expect("JSON serialization")
}
fn replace(template: &str, key: &str, value: &str) -> String {
    template.replace(key, value)
}
fn embed(
    engine: &mut decengine_engine_mlx::MlxDecisionEngine,
    handle: decengine_engine::ModelHandle,
    text: &str,
) -> Result<Vec<f32>> {
    let vector = engine.embed_text(handle, text)?;
    ensure!(
        vector.len() == 1024,
        "embedding dimension {} != 1024",
        vector.len()
    );
    ensure!(vector.iter().all(|x| x.is_finite()), "non-finite embedding");
    let norm = vector
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();
    ensure!(
        (norm - 1.0).abs() <= 1e-4,
        "embedding is not L2-normalized (norm={norm})"
    );
    Ok(vector)
}

pub fn export(store: &ModelStore, model: &str, input: &Path, output: &Path) -> Result<()> {
    use decengine_engine_mlx::MlxDecisionEngine;
    let resolved = store.resolve_installed(model)?;
    let root = resolved
        .root
        .as_ref()
        .context("installed model has no root")?;
    let install: InstallRecord = serde_json::from_slice(&fs::read(root.join("install.json"))?)?;
    ensure!(install.model_id == model, "installed model mismatch");
    ensure!(
        resolved.profile.pooling == "last_token" && resolved.profile.normalize == "l2",
        "profile must use last-token L2 embeddings"
    );
    let mut checkpoints = Vec::new();
    for file in &install.files {
        let path = root.join(&file.path);
        ensure!(path.is_file(), "checkpoint file missing: {}", file.path);
        ensure!(
            path.metadata()?.len() == file.size,
            "checkpoint size mismatch: {}",
            file.path
        );
        ensure!(
            sha(&path)? == file.sha256,
            "checkpoint hash mismatch: {}",
            file.path
        );
        checkpoints.push(CheckpointFile {
            path: file.path.clone(),
            sha256: file.sha256.clone(),
            size: file.size,
        });
    }
    ensure!(!checkpoints.is_empty(), "empty checkpoint manifest");
    let tokenizer_path = root.join("tokenizer.json");
    ensure!(
        tokenizer_path.is_file(),
        "tokenizer.json missing from installed model"
    );
    let tokenizer_sha256 = sha(&tokenizer_path)?;
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|error| anyhow::anyhow!("load tokenizer.json: {error}"))?;
    let profile_json = serde_json::to_vec(&resolved.profile)?;
    let profile_hash = hex::encode(Sha256::digest(profile_json));
    let mut engine = MlxDecisionEngine::new()?;
    let handle = engine.load_model(&resolved.engine_spec())?;
    let src = BufReader::new(fs::File::open(input)?);
    let mut out = BufWriter::new(fs::File::create(output)?);
    let mut seen = BTreeSet::new();
    let mut count = 0usize;
    for (line_no, line) in src.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Value =
            serde_json::from_str(&line).with_context(|| format!("input line {}", line_no + 1))?;
        let id = row["id"].as_str().context("record needs string id")?;
        ensure!(seen.insert(id.to_owned()), "duplicate record id {id}");
        let req = &row["request"];
        let state = req.get("state").context("request.state missing")?;
        let questions = req["questions"]
            .as_object()
            .context("questions must be object")?;
        ensure!(!questions.is_empty(), "record has no questions");
        for (key, question) in questions {
            let kind = question["type"].as_str().context("question.type missing")?;
            let prompt = question["prompt"]
                .as_str()
                .context("question.prompt missing")?;
            let state_text = canonical(state);
            let raw = format!("Decision: {prompt}\nState: {state_text}");
            let kind_instruction = match kind {
                "choice" => {
                    "select the single candidate that best satisfies the question using the reported state and candidate criteria; do not infer facts absent from the case"
                }
                "noul" => {
                    "decide whether the proposition asked about is supported by the reported state; evaluate the complete proposition from evidence and do not infer facts absent from the case"
                }
                "score" => {
                    "select the rubric level whose criterion best matches the reported state and requested rating; use the level definitions and do not infer facts absent from the case"
                }
                _ => anyhow::bail!("unsupported question type {kind:?}"),
            };
            let instruction = replace(
                &replace(
                    &resolved.profile.query_instruction_template,
                    "{prompt}",
                    prompt,
                ),
                "{kind}",
                kind_instruction,
            );
            let query = replace(
                &replace(
                    &resolved.profile.query_template,
                    "{instruction}",
                    &instruction,
                ),
                "{text}",
                &raw,
            );
            let query_tokens = tokenizer
                .encode(query.as_str(), true)
                .map_err(|error| anyhow::anyhow!("tokenize query: {error}"))?
                .get_ids()
                .len();
            ensure!(
                query_tokens > 0 && query_tokens <= resolved.profile.max_length,
                "query token length {query_tokens} outside 1..={}",
                resolved.profile.max_length
            );
            let candidates: Vec<(String, String, Option<f64>)> = match kind {
            "choice" => question["options"].as_object().context("choice options missing")?.iter().map(|(label, text)| {
                let templated = replace(&replace(&resolved.profile.choice_candidate_template, "{label}", label), "{criterion}", text.as_str().unwrap_or(""));
                (label.clone(), replace(&resolved.profile.candidate_template, "{text}", &templated), None)
            }).collect(),
            "score" => question["levels"].as_array().context("score levels missing")?.iter().map(|level| {
                let label = level["label"].as_str().unwrap_or("").to_owned();
                let criterion = level["criterion"].as_str().unwrap_or("");
                let value = level["value"].as_f64();
                let templated = replace(&replace(&resolved.profile.score_candidate_template, "{label}", &label), "{criterion}", criterion);
                (label, replace(&resolved.profile.candidate_template, "{text}", &templated), value)
            }).collect(),
            // Explicit semantic alternatives are conditioned on the actual proposition, not a fixed urgency vocabulary.
            "noul" => [
                ("unsupported".to_owned(), format!("For the question ‘{prompt}’, the proposition is not supported by the reported state; the evidence does not establish that this outcome applies."), Some(0.0)),
                ("supported".to_owned(), format!("For the question ‘{prompt}’, the proposition is supported by the reported state; the evidence establishes that this outcome applies."), Some(1.0)),
            ].into_iter().map(|(label, text, value)| (label, replace(&resolved.profile.candidate_template, "{text}", &text), value)).collect(),
            _ => unreachable!(),
        };
            ensure!(!candidates.is_empty(), "question has no candidates");
            let query_embedding = embed(&mut engine, handle, &query)?;
            let mut candidate_rows = Vec::new();
            for (ordinal, (label, text, numeric_ordinal)) in candidates.iter().enumerate() {
                let token_length = tokenizer
                    .encode(text.as_str(), true)
                    .map_err(|error| anyhow::anyhow!("tokenize candidate: {error}"))?
                    .get_ids()
                    .len();
                ensure!(
                    token_length > 0 && token_length <= resolved.profile.max_length,
                    "candidate token length {token_length} outside 1..={}",
                    resolved.profile.max_length
                );
                let embedding = embed(&mut engine, handle, text)?;
                candidate_rows.push(json!({"candidate_id":label,"type":kind,"order":ordinal,"label":label,"numeric_ordinal":ordinal,"score_value":numeric_ordinal,"token_length":token_length,"text":text,"text_utf8_bytes":text.len(),"embedding":embedding}));
            }
            let result = json!({"id":id,"question_key":key,"question_type":kind,"model":model,"query_text":query,"query_token_length":query_tokens,"query_utf8_bytes":query.len(),"query_embedding":query_embedding,"candidates":candidate_rows});
            serde_json::to_writer(&mut out, &result)?;
            out.write_all(b"\n")?;
            count += 1;
        }
    }
    ensure!(count > 0, "input contains no rows");
    out.flush()?;
    let manifest = json!({"schema_version":1,"model":model,"profile_version":resolved.profile.profile_version,"profile_sha256":profile_hash,"text_version":TEXT_VERSION,"embedding_dimensions":1024,"normalization":"L2","backend":"MLX/Metal","feature_transfer":"host vectors for offline training; not Metal-resident classifier","checkpoint_files":checkpoints,"tokenizer_sha256":tokenizer_sha256,"max_token_length":resolved.profile.max_length,"input_sha256":sha(input)?,"question_record_count":count,"serialization":"UTF-8 JSONL, one row per named question; state canonicalized with serde_json compact sorted-key map order; query=profile.query_template(instruction=profile.query_instruction_template(prompt, kind_instruction), text='Decision: {prompt}\\nState: {canonical_state}'); candidate=profile.candidate_template(profile-specific candidate template). Choice/score preserve candidate order as serialized by JSON object/array; binary candidates contain the complete prompt proposition. Reference, target, rationale, and domain are never read into feature text.","text_lengths":"Exact tokenizer.json token counts and UTF-8 byte lengths are recorded for each query and candidate; each is checked nonempty and <= frozen profile max_length."});
    fs::write(
        format!("{}.manifest.json", output.display()),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    eprintln!(
        "exported {count} varied feature records: {}",
        output.display()
    );
    Ok(())
}
