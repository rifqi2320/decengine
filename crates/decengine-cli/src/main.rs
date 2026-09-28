#[cfg(feature = "mlx")]
use std::collections::{BTreeMap, BTreeSet};
#[cfg(feature = "mlx")]
use std::fs::{self, File};
#[cfg(feature = "mlx")]
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::net::IpAddr;
use std::path::PathBuf;
use std::str::FromStr;

#[cfg(feature = "mlx")]
use anyhow::ensure;
use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
#[cfg(feature = "mlx")]
use decengine_core::DecisionRequest;
use decengine_engine::DecisionEngine;
#[cfg(feature = "mlx")]
use decengine_models::InstallRecord;
use decengine_models::{ModelRegistry, ModelStore};
use decengine_runtime::DecisionService;
#[cfg(feature = "mlx")]
use serde::Serialize;
#[cfg(feature = "mlx")]
use serde_json::Value;
#[cfg(feature = "mlx")]
use sha2::{Digest, Sha256};
use tokio::net::TcpListener;
use tracing::info;

mod probe;
#[cfg(feature = "mlx")]
#[path = "../../../results/probes/varied/prompt-study/metal/runner.rs"]
mod prompt_study_metal;
#[cfg(feature = "mlx")]
#[path = "../../../results/probes/varied/exporter.rs"]
mod varied_exporter;
#[cfg(feature = "mlx")]
#[path = "../../../results/probes/varied/metal/runner.rs"]
mod varied_metal;

#[derive(Debug, Parser)]
#[command(
    name = "decengine",
    version,
    about = "Local typed semantic decision runtime"
)]
struct Cli {
    #[arg(
        long,
        visible_alias = "model-store",
        env = "DECENGINE_HOME",
        global = true
    )]
    home: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Start the System One-compatible localhost server.
    Serve {
        #[arg(long, default_value = "Qwen/Qwen3-Embedding-0.6B")]
        model: String,
        #[arg(long, value_enum, default_value_t = default_engine())]
        engine: EngineKind,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8787)]
        port: u16,
        #[arg(long)]
        allow_non_loopback: bool,
    },
    /// Install an allowlisted model from Hugging Face or a local directory.
    Pull {
        model: String,
        #[arg(long = "from")]
        source: Option<PathBuf>,
    },
    /// List installed model records.
    List {
        #[arg(long)]
        json: bool,
    },
    /// Remove an installed model.
    Rm {
        model: String,
        #[arg(long)]
        yes: bool,
    },
    /// Export state and compiled task-query embeddings for probe training.
    ExportFeatures {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value = "Qwen/Qwen3-Embedding-0.6B")]
        model: String,
    },
    /// Export prompt-conditioned candidate features for varied-question fixtures (MLX only).
    VariedExportFeatures {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value = "Qwen/Qwen3-Embedding-0.6B")]
        model: String,
    },
    /// Run a frozen generic typed scorer with embeddings and pair inference resident on Metal.
    VariedMetal {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        run_dir: PathBuf,
        #[arg(long, default_value = "Qwen/Qwen3-Embedding-0.6B")]
        model: String,
    },
    /// Run one pre-frozen prompt-study text/scorer variant fully on MLX/Metal.
    PromptStudyMetal {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        scorer_bundle: PathBuf,
        #[arg(long)]
        study_run: PathBuf,
        #[arg(long)]
        variant: String,
        #[arg(long, default_value = "shared-v2-baseline")]
        head_mode: String,
        #[arg(long, default_value = "Qwen/Qwen3-Embedding-0.6B")]
        model: String,
    },
    /// Run a saved frozen logistic probe fully on MLX/Metal.
    ProbeDecide {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        probe_dir: PathBuf,
        #[arg(long, default_value = "Qwen/Qwen3-Embedding-0.6B")]
        model: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum EngineKind {
    Fixture,
    Mlx,
}

const fn default_engine() -> EngineKind {
    #[cfg(feature = "fixture-engine")]
    {
        EngineKind::Fixture
    }
    #[cfg(all(not(feature = "fixture-engine"), feature = "mlx"))]
    {
        EngineKind::Mlx
    }
    #[cfg(not(any(feature = "fixture-engine", feature = "mlx")))]
    {
        EngineKind::Mlx
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "decengine=info".into()),
        )
        .init();
    let cli = Cli::parse();
    run(cli).await
}

async fn run(cli: Cli) -> Result<()> {
    let registry = ModelRegistry::bundled().context("load model registry")?;
    let store = match cli.home {
        Some(home) => ModelStore::new(home, registry.clone()),
        None => ModelStore::from_environment(registry.clone())?,
    };

    match cli.command {
        Command::Serve {
            model,
            engine,
            host,
            port,
            allow_non_loopback,
        } => {
            let ip = IpAddr::from_str(&host).context("--host must be an IP address")?;
            if !ip.is_loopback() && !allow_non_loopback {
                bail!("refusing non-loopback bind; pass --allow-non-loopback explicitly");
            }
            let (backend, resolved): (Box<dyn DecisionEngine>, _) = match engine {
                EngineKind::Fixture => fixture_backend(&store, &model)?,
                EngineKind::Mlx => mlx_backend(&store, &model)?,
            };
            let service = DecisionService::shared(backend, resolved)?;
            let application = decengine_server::router(service, &registry);
            let address = (ip, port);
            let listener = TcpListener::bind(address).await?;
            info!(address = %listener.local_addr()?, "decengine listening");
            decengine_server::serve(listener, application).await?;
        }
        Command::Pull { model, source } => {
            let record = if let Some(source) = source {
                store.install_from_directory(&model, &source)?
            } else {
                store.pull_from_hugging_face(&model).await?
            };
            println!(
                "installed {} ({} files)",
                record.model_id,
                record.files.len()
            );
        }
        Command::List { json } => {
            let records = store.list()?;
            if json {
                println!("{}", serde_json::to_string_pretty(&records)?);
            } else if records.is_empty() {
                println!("No models installed.");
            } else {
                for record in records {
                    let bytes = record.files.iter().map(|file| file.size).sum::<u64>();
                    println!(
                        "{}\t{} files\t{} bytes",
                        record.model_id,
                        record.files.len(),
                        bytes
                    );
                }
            }
        }
        Command::Rm { model, yes } => {
            if !yes {
                bail!("refusing to remove {model:?} without --yes");
            }
            if store.remove(&model)? {
                println!("removed {model}");
            } else {
                println!("{model} was not installed");
            }
        }
        Command::ExportFeatures {
            input,
            output,
            model,
        } => {
            export_features(&store, &model, &input, &output)?;
        }
        Command::VariedExportFeatures {
            input,
            output,
            model,
        } => {
            #[cfg(feature = "mlx")]
            varied_exporter::export(&store, &model, &input, &output)?;
            #[cfg(not(feature = "mlx"))]
            {
                let _ = (input, output, model);
                bail!("varied-export-features requires building decengine with --features mlx");
            }
        }
        Command::VariedMetal {
            input,
            output,
            run_dir,
            model,
        } => {
            #[cfg(feature = "mlx")]
            varied_metal::run(&store, &model, &run_dir, &input, &output)?;
            #[cfg(not(feature = "mlx"))]
            {
                let _ = (input, output, run_dir, model);
                bail!("varied-metal requires building decengine with --features mlx");
            }
        }
        Command::PromptStudyMetal {
            input,
            output,
            scorer_bundle,
            study_run,
            variant,
            head_mode,
            model,
        } => {
            #[cfg(feature = "mlx")]
            prompt_study_metal::run(
                &store,
                &model,
                &variant,
                &input,
                &output,
                &scorer_bundle,
                &study_run,
                &head_mode,
            )?;
            #[cfg(not(feature = "mlx"))]
            {
                let _ = (
                    input,
                    output,
                    scorer_bundle,
                    study_run,
                    variant,
                    head_mode,
                    model,
                );
                bail!("prompt-study-metal requires building decengine with --features mlx");
            }
        }
        Command::ProbeDecide {
            input,
            output,
            probe_dir,
            model,
        } => {
            probe::probe_decide(&store, &model, &input, &output, &probe_dir)?;
        }
    }
    Ok(())
}

#[cfg(feature = "mlx")]
const FEATURE_TEXT_VERSION: &str = "decengine-probe-text-v1";

#[cfg(feature = "mlx")]
#[derive(Serialize)]
struct FeatureRecord {
    id: String,
    model: String,
    embeddings: FeatureEmbeddings,
}

#[cfg(feature = "mlx")]
#[derive(Serialize)]
struct FeatureEmbeddings {
    state: Vec<f32>,
    owner: Vec<f32>,
    urgent: Vec<f32>,
    impact: Vec<f32>,
}

#[cfg(feature = "mlx")]
#[derive(Serialize)]
struct FeatureManifest {
    schema_version: u32,
    model: String,
    profile_version: String,
    profile_sha256: String,
    text_version: &'static str,
    embedding_dimensions: usize,
    checkpoint_files: Vec<CheckpointFile>,
}

#[cfg(feature = "mlx")]
#[derive(Serialize)]
struct CheckpointFile {
    path: String,
    sha256: String,
    size: u64,
}

fn export_features(
    store: &ModelStore,
    model_id: &str,
    input: &PathBuf,
    output: &PathBuf,
) -> Result<()> {
    #[cfg(feature = "mlx")]
    {
        export_features_mlx(store, model_id, input, output)
    }
    #[cfg(not(feature = "mlx"))]
    {
        let _ = (store, model_id, input, output);
        bail!("export-features requires building decengine with --features mlx")
    }
}

#[cfg(feature = "mlx")]
fn export_features_mlx(
    store: &ModelStore,
    model_id: &str,
    input: &PathBuf,
    output: &PathBuf,
) -> Result<()> {
    let resolved = store.resolve_installed(model_id)?;
    let root = resolved
        .root
        .as_ref()
        .context("installed model has no root")?;
    let install: InstallRecord = serde_json::from_slice(&fs::read(root.join("install.json"))?)?;
    ensure!(
        install.model_id == model_id,
        "installed record model id mismatch"
    );

    let mut engine = decengine_engine_mlx::MlxDecisionEngine::new()?;
    let handle = engine.load_model(&resolved.engine_spec())?;
    let source =
        BufReader::new(File::open(input).with_context(|| format!("open {}", input.display()))?);
    let output_file =
        File::create(output).with_context(|| format!("create {}", output.display()))?;
    let mut writer = BufWriter::new(output_file);
    let mut seen = BTreeSet::new();
    let mut count = 0usize;
    for (line_number, line) in source.lines().enumerate() {
        let line = line.with_context(|| format!("read line {}", line_number + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(&line)
            .with_context(|| format!("parse input line {}", line_number + 1))?;
        let id = row
            .get("id")
            .and_then(Value::as_str)
            .context("each fixture needs a string id")?
            .to_owned();
        ensure!(!id.trim().is_empty(), "fixture id must not be blank");
        ensure!(seen.insert(id.clone()), "duplicate fixture id {id:?}");
        let request_value = row
            .get("request")
            .context("each fixture needs a request object")?
            .clone();
        let request: DecisionRequest = serde_json::from_value(request_value)
            .with_context(|| format!("invalid request for fixture {id:?}"))?;
        request
            .validate()
            .with_context(|| format!("invalid request for fixture {id:?}"))?;
        // State text is deliberately derived only from request.state. The compiler receives the
        // request's questions and state, never the fixture's scenario/reference/label fields.
        let compiled = decengine_compiler::compile(&request, &resolved.profile)?;
        let state_text = canonical_json(&request.state);
        let state = checked_embedding(engine.embed_text(handle, &state_text)?)?;
        let mut task_embeddings = BTreeMap::new();
        for key in ["owner", "urgent", "impact"] {
            let decision = compiled
                .engine_batch
                .decisions
                .iter()
                .find(|decision| decision.key == key)
                .with_context(|| format!("fixture {id:?} is missing compiled question {key:?}"))?;
            task_embeddings.insert(
                key,
                checked_embedding(engine.embed_text(handle, &decision.query)?)?,
            );
        }
        let embeddings = FeatureEmbeddings {
            state,
            owner: task_embeddings
                .remove("owner")
                .context("owner query embedding missing")?,
            urgent: task_embeddings
                .remove("urgent")
                .context("urgent query embedding missing")?,
            impact: task_embeddings
                .remove("impact")
                .context("impact query embedding missing")?,
        };
        let record = FeatureRecord {
            id,
            model: model_id.to_owned(),
            embeddings,
        };
        serde_json::to_writer(&mut writer, &record)?;
        writer.write_all(b"\n")?;
        count += 1;
    }
    ensure!(count > 0, "input contains no fixture rows");
    writer.flush()?;

    let checkpoint_files = install
        .files
        .iter()
        .map(|file| CheckpointFile {
            path: file.path.clone(),
            sha256: file.sha256.clone(),
            size: file.size,
        })
        .collect();
    let profile_json = serde_json::to_vec(&resolved.profile)?;
    let manifest = FeatureManifest {
        schema_version: 1,
        model: model_id.to_owned(),
        profile_version: resolved.profile.profile_version.clone(),
        profile_sha256: hex::encode(Sha256::digest(profile_json)),
        text_version: FEATURE_TEXT_VERSION,
        embedding_dimensions: 1024,
        checkpoint_files,
    };
    let manifest_path = PathBuf::from(format!("{}.manifest.json", output.display()));
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)
        .with_context(|| format!("write {}", manifest_path.display()))?;
    eprintln!(
        "exported {count} feature rows to {} (manifest {})",
        output.display(),
        manifest_path.display()
    );
    engine.unload_model(handle)?;
    Ok(())
}

#[cfg(feature = "mlx")]
fn checked_embedding(values: Vec<f32>) -> Result<Vec<f32>> {
    ensure!(
        values.len() == 1024,
        "expected 1024 embedding dimensions, got {}",
        values.len()
    );
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "embedding contains non-finite values"
    );
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    ensure!(
        (norm - 1.0).abs() < 1e-3,
        "embedding is not L2 normalized (norm {norm})"
    );
    Ok(values)
}

#[cfg(feature = "mlx")]
fn canonical_json(value: &Value) -> String {
    fn sorted(value: &Value) -> Value {
        match value {
            Value::Object(map) => Value::Object(
                map.iter()
                    .map(|(key, value)| (key.clone(), sorted(value)))
                    .collect::<BTreeMap<_, _>>()
                    .into_iter()
                    .collect(),
            ),
            Value::Array(values) => Value::Array(values.iter().map(sorted).collect()),
            _ => value.clone(),
        }
    }
    serde_json::to_string(&sorted(value)).expect("serializing JSON values cannot fail")
}

fn fixture_backend(
    store: &ModelStore,
    model: &str,
) -> Result<(Box<dyn DecisionEngine>, decengine_models::ResolvedModel)> {
    #[cfg(feature = "fixture-engine")]
    {
        Ok((
            Box::new(decengine_engine::FixtureDecisionEngine::default()),
            store.resolve_fixture(model)?,
        ))
    }
    #[cfg(not(feature = "fixture-engine"))]
    {
        let _ = (store, model);
        bail!("this binary was built without fixture-engine")
    }
}

fn mlx_backend(
    store: &ModelStore,
    model: &str,
) -> Result<(Box<dyn DecisionEngine>, decengine_models::ResolvedModel)> {
    #[cfg(feature = "mlx")]
    {
        Ok((
            Box::new(decengine_engine_mlx::MlxDecisionEngine::new()?),
            store.resolve_installed(model)?,
        ))
    }
    #[cfg(not(feature = "mlx"))]
    {
        let _ = (store, model);
        bail!("this binary was built without the mlx backend")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_required_commands() {
        assert!(Cli::try_parse_from(["decengine", "serve"]).is_ok());
        assert!(Cli::try_parse_from(["decengine", "pull", "Qwen/Qwen3-Embedding-0.6B"]).is_ok());
        assert!(Cli::try_parse_from(["decengine", "list", "--json"]).is_ok());
        assert!(
            Cli::try_parse_from(["decengine", "rm", "Qwen/Qwen3-Embedding-0.6B", "--yes"]).is_ok()
        );
        assert!(
            Cli::try_parse_from([
                "decengine",
                "probe-decide",
                "--input",
                "cases.jsonl",
                "--output",
                "out.jsonl",
                "--probe-dir",
                "frozen/qwen/task-specific"
            ])
            .is_ok()
        );
        assert!(
            Cli::try_parse_from([
                "decengine",
                "prompt-study-metal",
                "--input",
                "fresh60.jsonl",
                "--output",
                "gpu.jsonl",
                "--scorer-bundle",
                "frozen-scorers/qwen/v2-baseline.json",
                "--study-run",
                "run",
                "--variant",
                "v2-baseline",
                "--model",
                "Qwen/Qwen3-Embedding-0.6B"
            ])
            .is_ok()
        );
    }

    #[tokio::test]
    async fn refuses_public_bind_without_opt_in() {
        let cli = Cli::try_parse_from([
            "decengine",
            "--home",
            "/tmp/unused-decengine-test",
            "serve",
            "--host",
            "0.0.0.0",
        ])
        .unwrap();
        let error = run(cli).await.unwrap_err();
        assert!(error.to_string().contains("refusing non-loopback"));
    }
}
