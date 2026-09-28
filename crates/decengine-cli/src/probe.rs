use std::path::Path;
#[cfg(feature = "mlx")]
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
};

use anyhow::Result;
#[cfg(not(feature = "mlx"))]
use anyhow::bail;
#[cfg(feature = "mlx")]
use anyhow::{Context, ensure};
#[cfg(feature = "mlx")]
use decengine_core::DecisionRequest;
#[cfg(feature = "mlx")]
use decengine_engine::DecisionEngine;
#[cfg(feature = "mlx")]
use decengine_models::InstallRecord;
use decengine_models::ModelStore;
#[cfg(feature = "mlx")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "mlx")]
use serde_json::Value;
#[cfg(feature = "mlx")]
use sha2::{Digest, Sha256};

#[cfg(feature = "mlx")]
const FEATURE_TEXT_VERSION: &str = "decengine-probe-text-v1";

#[cfg(feature = "mlx")]
#[derive(Deserialize)]
struct Case {
    id: String,
    request: DecisionRequest,
}
#[cfg(feature = "mlx")]
#[derive(Deserialize)]
struct FrozenProbe {
    classes: Vec<String>,
    coefficients: Vec<Vec<f64>>,
    intercept: Vec<f64>,
    feature_scaler_mean: Vec<f64>,
    feature_scaler_scale: Vec<f64>,
}
#[cfg(feature = "mlx")]
#[derive(Deserialize, Serialize)]
struct CheckpointFile {
    path: String,
    sha256: String,
    size: u64,
}
#[cfg(feature = "mlx")]
#[derive(Serialize)]
struct Prediction {
    model: String,
    id: String,
    owner: TaskPrediction,
    urgent: TaskPrediction,
    impact: TaskPrediction,
}
#[cfg(feature = "mlx")]
#[derive(Serialize)]
struct TaskPrediction {
    label: String,
    confidence: f32,
    probabilities: BTreeMap<String, f32>,
}
#[cfg(feature = "mlx")]
#[derive(Serialize)]
struct CaseTiming {
    id: String,
    case_wall_ns: u128,
    tokenization_ns: u128,
    embedding_ns: u128,
    classifier_ns: u128,
}

pub fn probe_decide(
    store: &ModelStore,
    model_id: &str,
    input: &Path,
    output: &Path,
    probe_dir: &Path,
) -> Result<()> {
    #[cfg(feature = "mlx")]
    {
        probe_decide_mlx(store, model_id, input, output, probe_dir)
    }
    #[cfg(not(feature = "mlx"))]
    {
        let _ = (store, model_id, input, output, probe_dir);
        bail!("probe-decide requires building decengine with --features mlx")
    }
}

#[cfg(feature = "mlx")]
fn probe_decide_mlx(
    store: &ModelStore,
    model_id: &str,
    input: &Path,
    output: &Path,
    probe_dir: &Path,
) -> Result<()> {
    let models_path = probe_dir.join("models.json");
    let metadata_path = probe_dir.join("metadata.json");
    let probes: BTreeMap<String, FrozenProbe> = serde_json::from_slice(
        &std::fs::read(&models_path).with_context(|| format!("read {}", models_path.display()))?,
    )?;
    let metadata: Value = serde_json::from_slice(
        &std::fs::read(&metadata_path)
            .with_context(|| format!("read {}", metadata_path.display()))?,
    )?;
    let manifest = &metadata["embedding_manifests"]["test"]["contents"];
    let resolved = store.resolve_installed(model_id)?;
    let root = resolved
        .root
        .as_ref()
        .context("installed model has no root")?;
    let installed: InstallRecord =
        serde_json::from_slice(&std::fs::read(root.join("install.json"))?)?;
    let profile_sha = hex::encode(Sha256::digest(serde_json::to_vec(&resolved.profile)?));
    ensure!(
        manifest["model"].as_str() == Some(model_id) && installed.model_id == model_id,
        "probe/model id mismatch"
    );
    ensure!(
        manifest["profile_version"].as_str() == Some(&resolved.profile.profile_version)
            && installed.profile_version == resolved.profile.profile_version,
        "probe profile version mismatch"
    );
    ensure!(
        manifest["profile_sha256"].as_str() == Some(&profile_sha),
        "probe profile hash does not match installed profile"
    );
    ensure!(
        manifest["text_version"].as_str() == Some(FEATURE_TEXT_VERSION)
            && manifest["embedding_dimensions"].as_u64() == Some(1024),
        "probe feature text or dimensions unsupported"
    );
    let expected: Vec<CheckpointFile> =
        serde_json::from_value(manifest["checkpoint_files"].clone())?;
    let installed_files: Vec<_> = installed
        .files
        .iter()
        .map(|f| (&f.path, &f.sha256, f.size))
        .collect();
    let expected_files: Vec<_> = expected
        .iter()
        .map(|f| (&f.path, &f.sha256, f.size))
        .collect();
    ensure!(
        installed_files == expected_files,
        "probe checkpoint hashes do not match installed checkpoint"
    );
    ensure!(
        probes.keys().map(String::as_str).collect::<BTreeSet<_>>()
            == ["impact", "owner", "urgent"].into_iter().collect(),
        "frozen models must contain owner, urgent, and impact"
    );

    let mut engine = decengine_engine_mlx::MlxDecisionEngine::new()?;
    let handle = engine.load_model(&resolved.engine_spec())?;
    let reader =
        BufReader::new(File::open(input).with_context(|| format!("open {}", input.display()))?);
    let cases = reader
        .lines()
        .enumerate()
        .filter_map(|(n, row)| match row {
            Ok(row) if row.trim().is_empty() => None,
            Ok(row) => Some(
                serde_json::from_str::<Case>(&row)
                    .with_context(|| format!("parse case line {}", n + 1)),
            ),
            Err(err) => Some(Err(err.into())),
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(!cases.is_empty(), "input contains no cases");
    let mut seen_ids = BTreeSet::new();
    for case in &cases {
        ensure!(
            seen_ids.insert(case.id.clone()),
            "duplicate case id {:?}",
            case.id
        );
    }
    let mut writer = BufWriter::new(
        File::create(output).with_context(|| format!("create {}", output.display()))?,
    );
    let timing_path = Path::new(&format!("{}.timings.jsonl", output.display())).to_path_buf();
    let mut timing_writer = BufWriter::new(
        File::create(&timing_path).with_context(|| format!("create {}", timing_path.display()))?,
    );
    // Warm the model and cache all three GPU-resident classifier weight sets before measuring.
    for task in ["owner", "urgent", "impact"] {
        let first = &cases[0];
        let compiled = decengine_compiler::compile(&first.request, &resolved.profile)?;
        let decision = compiled
            .engine_batch
            .decisions
            .iter()
            .find(|d| d.key == task)
            .with_context(|| format!("warmup case missing compiled query {task}"))?;
        let probe = &probes[task];
        let _ = engine.probe_probabilities_timed(
            handle,
            task,
            &decision.query,
            &probe.coefficients,
            &probe.intercept,
            &probe.feature_scaler_mean,
            &probe.feature_scaler_scale,
        )?;
    }
    let count = cases.len();
    for case in cases {
        ensure!(!case.id.trim().is_empty(), "case id cannot be blank");
        case.request
            .validate()
            .with_context(|| format!("validate case {}", case.id))?;
        let case_start = std::time::Instant::now();
        let compiled = decengine_compiler::compile(&case.request, &resolved.profile)?;
        let mut rows = BTreeMap::new();
        let (mut tokenization_ns, mut embedding_ns, mut classifier_ns) = (0u128, 0u128, 0u128);
        for task in ["owner", "urgent", "impact"] {
            let decision = compiled
                .engine_batch
                .decisions
                .iter()
                .find(|d| d.key == task)
                .with_context(|| format!("{} missing compiled query {task}", case.id))?;
            let probe = &probes[task];
            ensure!(
                probe.classes.len()
                    == if probe.coefficients.len() == 1 {
                        2
                    } else {
                        probe.coefficients.len()
                    },
                "invalid class/coefficient shape for {task}"
            );
            let (probabilities, stage) = engine.probe_probabilities_timed(
                handle,
                task,
                &decision.query,
                &probe.coefficients,
                &probe.intercept,
                &probe.feature_scaler_mean,
                &probe.feature_scaler_scale,
            )?;
            tokenization_ns += stage.tokenization_ns;
            embedding_ns += stage.embedding_ns;
            classifier_ns += stage.classifier_ns;
            ensure!(
                probabilities.len() == probe.classes.len(),
                "probability/class count mismatch for {task}"
            );
            let (best_index, confidence) = probabilities
                .iter()
                .enumerate()
                .reduce(|best, next| if next.1 > best.1 { next } else { best })
                .map(|(i, p)| (i, *p))
                .context("empty probabilities")?;
            let mapped = probe.classes.iter().cloned().zip(probabilities).collect();
            rows.insert(
                task,
                TaskPrediction {
                    label: probe.classes[best_index].clone(),
                    confidence,
                    probabilities: mapped,
                },
            );
        }
        let row = Prediction {
            model: model_id.to_owned(),
            id: case.id,
            owner: rows.remove("owner").unwrap(),
            urgent: rows.remove("urgent").unwrap(),
            impact: rows.remove("impact").unwrap(),
        };
        serde_json::to_writer(&mut writer, &row)?;
        writer.write_all(b"\n")?;
        let timing = CaseTiming {
            id: row.id.clone(),
            case_wall_ns: case_start.elapsed().as_nanos(),
            tokenization_ns,
            embedding_ns,
            classifier_ns,
        };
        serde_json::to_writer(&mut timing_writer, &timing)?;
        timing_writer.write_all(b"\n")?;
    }
    writer.flush()?;
    timing_writer.flush()?;
    engine.unload_model(handle)?;
    let sidecar = serde_json::json!({
        "model": model_id, "case_count": count, "probe_dir": probe_dir,
        "probe_models_sha256": hex::encode(Sha256::digest(std::fs::read(models_path)?)),
        "input_path": input,
        "input_sha256": hex::encode(Sha256::digest(std::fs::read(input)?)),
        "prediction_sha256": hex::encode(Sha256::digest(std::fs::read(output)?)),
        "profile_sha256": profile_sha, "checkpoint_files": expected,
        "text_version": FEATURE_TEXT_VERSION,
        "execution": "Device(gpu, 0); tokenization CPU; normalized embedding, scaler, linear logits, and softmax are MLX arrays on GPU; only class probability vectors are read back",
        "warmup": "one first-case execution per owner/urgent/impact classifier after model load; all 300 per-case timings exclude model load and warmup",
        "timings_path": timing_path,
        "numeric_drift": "MLX f32 arithmetic versus sklearn CPU f64; compare probabilities at 1e-4 tolerance; labels must match exactly"
    });
    std::fs::write(
        format!("{}.metadata.json", output.display()),
        serde_json::to_vec_pretty(&sidecar)?,
    )?;
    eprintln!(
        "wrote {count} GPU-resident probe predictions to {}",
        output.display()
    );
    Ok(())
}
