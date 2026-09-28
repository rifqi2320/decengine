//! Allowlisted model profiles and the local model store.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use decengine_core::{DecengineError, ErrorCode, Result};
use decengine_engine::EngineModelSpec;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const QWEN_PROFILE: &str = include_str!("../../../models/manifests/qwen3-embedding-0.6b.json");
const HARRIER_PROFILE: &str = include_str!("../../../models/manifests/harrier-oss-v1-0.6b.json");

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelProfile {
    pub schema_version: u32,
    pub id: String,
    pub family: String,
    pub architecture: String,
    pub pooling: String,
    pub normalize: String,
    pub padding_side: String,
    pub query_template: String,
    pub candidate_template: String,
    #[serde(default = "default_query_instruction_template")]
    pub query_instruction_template: String,
    #[serde(default = "default_choice_candidate_template")]
    pub choice_candidate_template: String,
    #[serde(default = "default_score_candidate_template")]
    pub score_candidate_template: String,
    #[serde(default = "default_noul_unsupported")]
    pub noul_unsupported: String,
    #[serde(default = "default_noul_supported")]
    pub noul_supported: String,
    pub temperature: f32,
    pub max_length: usize,
    pub profile_version: String,
    pub required_files: Vec<String>,
}

fn default_noul_unsupported() -> String {
    "unsupported false no absent not required can wait".to_owned()
}

fn default_noul_supported() -> String {
    "supported true yes present urgent today immediate required".to_owned()
}

fn default_query_instruction_template() -> String {
    "{prompt}".to_owned()
}

fn default_choice_candidate_template() -> String {
    "{label}: {criterion}".to_owned()
}

fn default_score_candidate_template() -> String {
    "{label}: {criterion}".to_owned()
}

impl ModelProfile {
    /// Validates the versioned model profile and all file-path constraints.
    ///
    /// # Errors
    ///
    /// Returns [`DecengineError`] when the profile is unsupported or internally inconsistent.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != 1 {
            return Err(invalid_manifest("unsupported profile schema_version"));
        }
        if self.family != "qwen3-0.6b" {
            return Err(invalid_manifest("profile family must be qwen3-0.6b"));
        }
        if self.pooling != "last_token" || self.normalize != "l2" {
            return Err(invalid_manifest(
                "v1 profiles require last_token pooling and l2 normalization",
            ));
        }
        if self.padding_side != "left" {
            return Err(invalid_manifest("v1 profiles require left padding"));
        }
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(invalid_manifest("temperature must be positive and finite"));
        }
        if !self.query_template.contains("{instruction}") || !self.query_template.contains("{text}")
        {
            return Err(invalid_manifest(
                "query_template requires {instruction} and {text}",
            ));
        }
        if self.required_files.is_empty() {
            return Err(invalid_manifest("required_files must not be empty"));
        }
        for file in &self.required_files {
            let path = Path::new(file);
            if path.is_absolute()
                || path
                    .components()
                    .any(|component| matches!(component, Component::ParentDir | Component::RootDir))
            {
                return Err(invalid_manifest("required_files contains an unsafe path"));
            }
        }
        Ok(())
    }
}

fn invalid_manifest(message: impl Into<String>) -> DecengineError {
    DecengineError::new(ErrorCode::InvalidManifest, message)
}

#[derive(Debug, Clone)]
pub struct ModelRegistry {
    profiles: BTreeMap<String, ModelProfile>,
}

impl ModelRegistry {
    /// Parses and validates the model profiles embedded in this build.
    ///
    /// # Errors
    ///
    /// Returns [`DecengineError`] if a bundled manifest is invalid or duplicates another model ID.
    pub fn bundled() -> Result<Self> {
        let mut profiles = BTreeMap::new();
        for source in [QWEN_PROFILE, HARRIER_PROFILE] {
            let profile: ModelProfile = serde_json::from_str(source).map_err(|error| {
                invalid_manifest(format!("bundled profile is invalid JSON: {error}"))
            })?;
            profile.validate()?;
            if profiles.insert(profile.id.clone(), profile).is_some() {
                return Err(invalid_manifest("duplicate bundled model ID"));
            }
        }
        Ok(Self { profiles })
    }

    /// Returns the allowlisted profile for a model ID.
    ///
    /// # Errors
    ///
    /// Returns [`DecengineError`] when the model ID is not in the bundled registry.
    pub fn get(&self, model_id: &str) -> Result<&ModelProfile> {
        self.profiles.get(model_id).ok_or_else(|| {
            DecengineError::new(
                ErrorCode::UnsupportedModel,
                format!("unsupported model {model_id:?}"),
            )
        })
    }

    pub fn profiles(&self) -> impl Iterator<Item = &ModelProfile> {
        self.profiles.values()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledFile {
    pub path: String,
    pub sha256: String,
    pub size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstallRecord {
    pub model_id: String,
    pub profile_version: String,
    pub installed_at_unix: u64,
    pub files: Vec<InstalledFile>,
}

#[derive(Debug, Clone)]
pub struct ResolvedModel {
    pub profile: ModelProfile,
    pub root: Option<PathBuf>,
}

impl ResolvedModel {
    pub fn engine_spec(&self) -> EngineModelSpec {
        EngineModelSpec {
            id: self.profile.id.clone(),
            root: self.root.clone(),
            family: self.profile.family.clone(),
            temperature: self.profile.temperature,
            max_length: self.profile.max_length,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelStore {
    root: PathBuf,
    registry: ModelRegistry,
}

impl ModelStore {
    pub fn new(root: impl Into<PathBuf>, registry: ModelRegistry) -> Self {
        Self {
            root: root.into(),
            registry,
        }
    }

    /// Builds a model store from `DECENGINE_HOME`, or from `HOME` when no override is set.
    ///
    /// # Errors
    ///
    /// Returns [`DecengineError`] when neither environment variable provides a usable root.
    pub fn from_environment(registry: ModelRegistry) -> Result<Self> {
        if let Some(path) = std::env::var_os("DECENGINE_HOME") {
            return Ok(Self::new(path, registry));
        }
        let home = std::env::var_os("HOME").ok_or_else(|| {
            DecengineError::invalid_request(
                "HOME is not set; set DECENGINE_HOME to a writable directory",
            )
        })?;
        Ok(Self::new(PathBuf::from(home).join(".decengine"), registry))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn model_dir(&self, model_id: &str) -> PathBuf {
        self.root.join("models").join(model_id.replace('/', "--"))
    }

    /// Resolves and verifies a model already installed in the local store.
    ///
    /// # Errors
    ///
    /// Returns [`DecengineError`] when the model is unsupported, absent, corrupt, or does not match
    /// the active profile.
    pub fn resolve_installed(&self, model_id: &str) -> Result<ResolvedModel> {
        let profile = self.registry.get(model_id)?.clone();
        let root = self.model_dir(model_id);
        let record_path = root.join("install.json");
        if !record_path.is_file() {
            return Err(DecengineError::new(
                ErrorCode::ModelNotFound,
                format!("model {model_id:?} is not installed; run `decengine pull {model_id}`"),
            ));
        }
        Self::verify_record(&profile, &root, &record_path)?;
        Ok(ResolvedModel {
            profile,
            root: Some(root),
        })
    }

    /// Resolves an allowlisted model without requiring weights, for fixture-backed tests only.
    ///
    /// # Errors
    ///
    /// Returns [`DecengineError`] when the model ID is not allowlisted.
    pub fn resolve_fixture(&self, model_id: &str) -> Result<ResolvedModel> {
        Ok(ResolvedModel {
            profile: self.registry.get(model_id)?.clone(),
            root: None,
        })
    }

    /// Lists installation records currently present in the model store.
    ///
    /// # Errors
    ///
    /// Returns [`DecengineError`] when the store cannot be read or a record is invalid.
    pub fn list(&self) -> Result<Vec<InstallRecord>> {
        let models = self.root.join("models");
        if !models.exists() {
            return Ok(Vec::new());
        }
        let mut records = Vec::new();
        for entry in fs::read_dir(models).map_err(io_error)? {
            let path = entry.map_err(io_error)?.path().join("install.json");
            if path.is_file() {
                let record: InstallRecord = serde_json::from_slice(
                    &fs::read(&path).map_err(io_error)?,
                )
                .map_err(|error| invalid_manifest(format!("{}: {error}", path.display())))?;
                records.push(record);
            }
        }
        records.sort_by(|left, right| left.model_id.cmp(&right.model_id));
        Ok(records)
    }

    /// Removes an installed allowlisted model, returning whether it existed.
    ///
    /// # Errors
    ///
    /// Returns [`DecengineError`] when the model ID is unsupported or the directory cannot be
    /// removed.
    pub fn remove(&self, model_id: &str) -> Result<bool> {
        self.registry.get(model_id)?;
        let directory = self.model_dir(model_id);
        if !directory.exists() {
            return Ok(false);
        }
        fs::remove_dir_all(directory).map_err(io_error)?;
        Ok(true)
    }

    /// Installs model metadata and safetensors files from a local directory.
    ///
    /// # Errors
    ///
    /// Returns [`DecengineError`] when validation, copying, hashing, or the atomic install fails.
    pub fn install_from_directory(&self, model_id: &str, source: &Path) -> Result<InstallRecord> {
        let profile = self.registry.get(model_id)?.clone();
        validate_source(&profile, source)?;
        let staging = self.staging_dir(model_id);
        fs::create_dir_all(&staging).map_err(io_error)?;

        let result = (|| {
            let mut names = profile
                .required_files
                .iter()
                .cloned()
                .collect::<BTreeSet<_>>();
            for entry in fs::read_dir(source).map_err(io_error)? {
                let entry = entry.map_err(io_error)?;
                let name = entry.file_name().to_string_lossy().into_owned();
                if name.ends_with(".safetensors") || name == "model.safetensors.index.json" {
                    names.insert(name);
                }
            }
            if !names.iter().any(|name| name.ends_with(".safetensors")) {
                return Err(DecengineError::new(
                    ErrorCode::ModelLoadFailed,
                    "source has no safetensors weights",
                ));
            }
            for name in names {
                fs::copy(source.join(&name), staging.join(&name)).map_err(io_error)?;
            }
            self.commit_install(&profile, &staging)
        })();

        if result.is_err() {
            let _ = fs::remove_dir_all(&staging);
        }
        result
    }

    /// Downloads and installs an allowlisted model from Hugging Face.
    ///
    /// # Errors
    ///
    /// Returns [`DecengineError`] for unsupported models, network failures, invalid metadata, or
    /// local storage failures.
    pub async fn pull_from_hugging_face(&self, model_id: &str) -> Result<InstallRecord> {
        let profile = self.registry.get(model_id)?.clone();
        let staging = self.staging_dir(model_id);
        fs::create_dir_all(&staging).map_err(io_error)?;
        let client = reqwest::Client::builder()
            .user_agent(concat!("decengine/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(network_error)?;

        let result = async {
            for file in &profile.required_files {
                download(&client, model_id, file, &staging).await?;
            }

            let index_name = "model.safetensors.index.json";
            let index_url = hf_url(model_id, index_name);
            let index_response = client.get(index_url).send().await.map_err(network_error)?;
            if index_response.status() == StatusCode::NOT_FOUND {
                download(&client, model_id, "model.safetensors", &staging).await?;
            } else {
                let index_response = index_response.error_for_status().map_err(network_error)?;
                let bytes = index_response.bytes().await.map_err(network_error)?;
                fs::write(staging.join(index_name), &bytes).map_err(io_error)?;
                let index: SafetensorsIndex = serde_json::from_slice(&bytes).map_err(|error| {
                    invalid_manifest(format!("invalid safetensors index: {error}"))
                })?;
                let shards = index.weight_map.into_values().collect::<BTreeSet<_>>();
                if shards.is_empty() {
                    return Err(invalid_manifest("safetensors index has no shards"));
                }
                for shard in shards {
                    safe_relative_name(&shard)?;
                    download(&client, model_id, &shard, &staging).await?;
                }
            }
            self.commit_install(&profile, &staging)
        }
        .await;

        if result.is_err() {
            let _ = fs::remove_dir_all(&staging);
        }
        result
    }

    fn commit_install(&self, profile: &ModelProfile, staging: &Path) -> Result<InstallRecord> {
        let mut files = Vec::new();
        for entry in fs::read_dir(staging).map_err(io_error)? {
            let entry = entry.map_err(io_error)?;
            if entry.path().is_file() {
                let bytes = fs::read(entry.path()).map_err(io_error)?;
                files.push(InstalledFile {
                    path: entry.file_name().to_string_lossy().into_owned(),
                    sha256: hex::encode(Sha256::digest(&bytes)),
                    size: bytes.len() as u64,
                });
            }
        }
        files.sort_by(|left, right| left.path.cmp(&right.path));
        let record = InstallRecord {
            model_id: profile.id.clone(),
            profile_version: profile.profile_version.clone(),
            installed_at_unix: unix_time()?,
            files,
        };
        fs::write(
            staging.join("install.json"),
            serde_json::to_vec_pretty(&record).map_err(|error| {
                DecengineError::new(ErrorCode::InternalError, error.to_string())
            })?,
        )
        .map_err(io_error)?;

        let destination = self.model_dir(&profile.id);
        if destination.exists() {
            return Err(DecengineError::invalid_request(format!(
                "model {:?} is already installed; remove it first",
                profile.id
            )));
        }
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent).map_err(io_error)?;
        }
        fs::rename(staging, destination).map_err(io_error)?;
        Ok(record)
    }

    fn verify_record(profile: &ModelProfile, root: &Path, record_path: &Path) -> Result<()> {
        let record: InstallRecord =
            serde_json::from_slice(&fs::read(record_path).map_err(io_error)?)
                .map_err(|error| invalid_manifest(format!("invalid install record: {error}")))?;
        if record.model_id != profile.id || record.profile_version != profile.profile_version {
            return Err(DecengineError::new(
                ErrorCode::ModelLoadFailed,
                "installed model record does not match the active profile",
            ));
        }
        let recorded_names = record
            .files
            .iter()
            .map(|file| file.path.as_str())
            .collect::<BTreeSet<_>>();
        for required in &profile.required_files {
            if !recorded_names.contains(required.as_str()) {
                return Err(DecengineError::new(
                    ErrorCode::ModelLoadFailed,
                    format!("install record is missing required file {required}"),
                ));
            }
        }
        if !recorded_names
            .iter()
            .any(|name| name.ends_with(".safetensors"))
        {
            return Err(DecengineError::new(
                ErrorCode::ModelLoadFailed,
                "install record has no safetensors weights",
            ));
        }
        for file in record.files {
            safe_relative_name(&file.path)?;
            let path = root.join(&file.path);
            let bytes = fs::read(&path).map_err(|error| {
                DecengineError::new(
                    ErrorCode::ModelLoadFailed,
                    format!("cannot read {}: {error}", path.display()),
                )
            })?;
            if bytes.len() as u64 != file.size || hex::encode(Sha256::digest(&bytes)) != file.sha256
            {
                return Err(DecengineError::new(
                    ErrorCode::ModelLoadFailed,
                    format!("checksum mismatch for {}", path.display()),
                ));
            }
        }
        Ok(())
    }

    fn staging_dir(&self, model_id: &str) -> PathBuf {
        let suffix = format!("{}-{}", std::process::id(), unix_time().unwrap_or_default());
        self.root
            .join("state")
            .join("staging")
            .join(format!("{}-{suffix}", model_id.replace('/', "--")))
    }
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: BTreeMap<String, String>,
}

async fn download(client: &reqwest::Client, model_id: &str, file: &str, root: &Path) -> Result<()> {
    safe_relative_name(file)?;
    let response = client
        .get(hf_url(model_id, file))
        .send()
        .await
        .map_err(network_error)?
        .error_for_status()
        .map_err(network_error)?;
    let bytes = response.bytes().await.map_err(network_error)?;
    fs::write(root.join(file), bytes).map_err(io_error)
}

fn hf_url(model_id: &str, file: &str) -> String {
    format!("https://huggingface.co/{model_id}/resolve/main/{file}")
}

fn validate_source(profile: &ModelProfile, source: &Path) -> Result<()> {
    if !source.is_dir() {
        return Err(DecengineError::new(
            ErrorCode::ModelNotFound,
            format!("model source {} is not a directory", source.display()),
        ));
    }
    for file in &profile.required_files {
        if !source.join(file).is_file() {
            return Err(DecengineError::new(
                ErrorCode::ModelLoadFailed,
                format!("model source is missing required file {file}"),
            ));
        }
    }
    Ok(())
}

fn safe_relative_name(name: &str) -> Result<()> {
    let path = Path::new(name);
    if path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::RootDir))
    {
        return Err(invalid_manifest(format!("unsafe model file path {name:?}")));
    }
    Ok(())
}

fn unix_time() -> Result<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|error| DecengineError::new(ErrorCode::InternalError, error.to_string()))
}

// These adapters intentionally accept owned errors so they can be passed directly to `map_err`.
#[allow(clippy::needless_pass_by_value)]
fn io_error(error: std::io::Error) -> DecengineError {
    DecengineError::new(ErrorCode::ModelLoadFailed, error.to_string())
}

#[allow(clippy::needless_pass_by_value)]
fn network_error(error: reqwest::Error) -> DecengineError {
    DecengineError::new(ErrorCode::ModelLoadFailed, error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_registry_contains_only_release_models() {
        let registry = ModelRegistry::bundled().unwrap();
        let ids = registry
            .profiles()
            .map(|profile| profile.id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            ids,
            vec!["Qwen/Qwen3-Embedding-0.6B", "microsoft/harrier-oss-v1-0.6b"]
        );
    }

    #[test]
    fn rejects_unknown_model() {
        let registry = ModelRegistry::bundled().unwrap();
        assert_eq!(
            registry.get("other/model").unwrap_err().code,
            ErrorCode::UnsupportedModel
        );
    }

    #[test]
    fn installs_verifies_lists_and_removes_local_model() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source");
        fs::create_dir(&source).unwrap();
        for file in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
            fs::write(source.join(file), b"{}").unwrap();
        }
        fs::write(source.join("model.safetensors"), b"fixture weights").unwrap();

        let store = ModelStore::new(temp.path().join("store"), ModelRegistry::bundled().unwrap());
        let model_id = "Qwen/Qwen3-Embedding-0.6B";
        let record = store.install_from_directory(model_id, &source).unwrap();
        assert_eq!(record.model_id, model_id);
        assert!(store.resolve_installed(model_id).is_ok());
        assert_eq!(store.list().unwrap().len(), 1);
        assert!(store.remove(model_id).unwrap());
        assert!(!store.remove(model_id).unwrap());
    }

    #[test]
    fn detects_tampered_installed_file() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source");
        fs::create_dir(&source).unwrap();
        for file in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
            fs::write(source.join(file), b"{}").unwrap();
        }
        fs::write(source.join("model.safetensors"), b"fixture weights").unwrap();
        let store = ModelStore::new(temp.path().join("store"), ModelRegistry::bundled().unwrap());
        let model_id = "Qwen/Qwen3-Embedding-0.6B";
        store.install_from_directory(model_id, &source).unwrap();
        fs::write(store.model_dir(model_id).join("config.json"), b"tampered").unwrap();
        assert_eq!(
            store.resolve_installed(model_id).unwrap_err().code,
            ErrorCode::ModelLoadFailed
        );
    }
}
