//! Apple-Silicon MLX backend for the allowlisted Qwen3 embedding checkpoints.
//!
//! The model is deliberately kept in this crate so MLX arrays never leak over
//! the coarse `DecisionEngine` boundary.  It embeds the compiled query and each
//! candidate, then returns a temperature-scaled cosine distribution.

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::{collections::HashMap, fs, path::Path};

use decengine_engine::{
    DecisionEngine, EngineDecisionBatch, EngineDecisionBatchResult, EngineDecisionResult,
    EngineError, EngineModelSpec, ModelHandle, Result,
};

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use mlx_rs::{
    Array, Device, DeviceType, Dtype,
    builder::Builder,
    fast::{ScaledDotProductAttentionMask, scaled_dot_product_attention},
    module::Module,
    nn, ops,
    ops::indexing::IndexOp,
};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use serde::Deserialize;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use tokenizers::Tokenizer;

/// MLX implementation for Qwen3 embedding checkpoints.
#[derive(Default)]
pub struct MlxDecisionEngine {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    next_handle: u64,
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    loaded: Option<LoadedModel>,
}

/// Synchronized stage timings for one compiled query; all values are nanoseconds.
#[derive(Debug, Default, Clone, Copy)]
pub struct ProbeStageTimings {
    pub tokenization_ns: u128,
    pub embedding_ns: u128,
    pub classifier_ns: u128,
}

impl MlxDecisionEngine {
    /// Creates an MLX engine after checking that the host is supported.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::UnsupportedPlatform`] unless running on macOS on Apple Silicon.
    pub fn new() -> Result<Self> {
        platform_check()?;
        Ok(Self::default())
    }

    /// Embeds text using the currently loaded checkpoint.
    ///
    /// The returned vector is the model's last-token, L2-normalized embedding.
    pub fn embed_text(&mut self, model: ModelHandle, text: &str) -> Result<Vec<f32>> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            let loaded = self
                .loaded
                .as_mut()
                .filter(|loaded| loaded.handle == model)
                .ok_or_else(|| EngineError::Backend("unknown model handle".to_owned()))?;
            let embedding = loaded.embedder.embed(text)?;
            embedding.eval().map_err(mlx_error)?;
            let values = embedding.as_slice::<f32>().to_vec();
            if values.iter().any(|value| !value.is_finite()) {
                return Err(EngineError::Backend(
                    "model produced a non-finite embedding".to_owned(),
                ));
            }
            Ok(values)
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = (model, text);
            Err(EngineError::UnsupportedPlatform(
                "the MLX engine requires macOS on Apple Silicon".to_owned(),
            ))
        }
    }

    /// Embed a query and variable candidate texts, then apply a frozen generic pair scorer.
    /// Embeddings, pair features, standardization, logits, and softmax stay as MLX arrays;
    /// only final candidate probabilities are copied to host memory.
    pub fn varied_pair_probabilities_timed(
        &mut self,
        model: ModelHandle,
        scorer_key: &str,
        query_text: &str,
        candidate_texts: &[String],
        coefficients: &[f64],
        feature_mean: &[f64],
        feature_scale: &[f64],
    ) -> Result<(Vec<f32>, ProbeStageTimings)> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            use mlx_rs::ops::concatenate;
            let mut timings = ProbeStageTimings::default();
            let loaded = self
                .loaded
                .as_mut()
                .filter(|loaded| loaded.handle == model)
                .ok_or_else(|| EngineError::Backend("unknown model handle".to_owned()))?;
            ensure_varied_scorer(candidate_texts, coefficients, feature_mean, feature_scale)?;
            let signature = coefficients
                .iter()
                .chain(feature_mean)
                .chain(feature_scale)
                .map(|x| x.to_bits())
                .collect::<Vec<_>>();
            if let Some(cached) = loaded.probe_cache.get(scorer_key) {
                if cached.signature != signature {
                    return Err(EngineError::Backend(format!(
                        "scorer key {scorer_key:?} reused with different frozen weights"
                    )));
                }
            } else {
                // Fold StandardScaler into the linear weights on the CPU, in float64.
                // This is algebraically exact and avoids Metal flushing Harrier's subnormal
                // feature scales to zero. No inference features are constructed on CPU.
                let effective = coefficients
                    .iter()
                    .zip(feature_scale)
                    .map(|(&w, &s)| w / s)
                    .collect::<Vec<_>>();
                let bias = -feature_mean
                    .iter()
                    .zip(&effective)
                    .map(|(m, w)| m * w)
                    .sum::<f64>();
                if effective
                    .iter()
                    .chain(std::iter::once(&bias))
                    .any(|x| !x.is_finite())
                {
                    return Err(EngineError::Backend(
                        "standardized frozen scorer overflows float64".to_owned(),
                    ));
                }
                loaded.probe_cache.insert(
                    scorer_key.to_owned(),
                    ResidentProbe {
                        signature,
                        weights: array_from_f64(&effective, &[4096])?,
                        bias: array_from_f64(&[bias], &[1])?,
                        factor: Array::from_f32(1.0),
                        mean: Array::from_f32(0.0),
                        scale: Array::from_f32(1.0),
                    },
                );
            }
            let resident = loaded.probe_cache.get(scorer_key).unwrap();
            let weights = resident.weights.clone();
            let bias = resident.bias.clone();
            let embedding_start = std::time::Instant::now();
            let (query, query_token_ns) = loaded.embedder.embed_probe(query_text)?;
            let mut candidates = Vec::with_capacity(candidate_texts.len());
            let mut token_ns = query_token_ns;
            for text in candidate_texts {
                let (candidate, ns) = loaded.embedder.embed_probe(text)?;
                token_ns += ns;
                candidates.push(candidate);
            }
            let mut feature_rows = Vec::with_capacity(candidate_texts.len());
            for candidate in candidates {
                let mul = query.multiply(candidate.clone()).map_err(mlx_error)?;
                let abs = query
                    .subtract(candidate.clone())
                    .map_err(mlx_error)?
                    .abs()
                    .map_err(mlx_error)?;
                feature_rows
                    .push(concatenate(&[&query, &candidate, &mul, &abs], 0).map_err(mlx_error)?);
            }
            let x = ops::stack(&feature_rows, 0).map_err(mlx_error)?;
            timings.tokenization_ns = token_ns;
            x.eval().map_err(mlx_error)?;
            timings.embedding_ns = embedding_start
                .elapsed()
                .as_nanos()
                .saturating_sub(token_ns);
            let scorer_start = std::time::Instant::now();
            let logits = x
                .matmul(&weights)
                .map_err(mlx_error)?
                .add(bias)
                .map_err(mlx_error)?;
            let probs = ops::softmax(&logits, None).map_err(mlx_error)?;
            probs.eval().map_err(mlx_error)?;
            timings.classifier_ns = scorer_start.elapsed().as_nanos();
            let values = probs.as_slice::<f32>().to_vec();
            if values.len() != candidate_texts.len() || values.iter().any(|x| !x.is_finite()) {
                return Err(EngineError::Backend(
                    "varied scorer returned invalid probabilities".to_owned(),
                ));
            }
            Ok((values, timings))
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = (
                model,
                scorer_key,
                query_text,
                candidate_texts,
                coefficients,
                feature_mean,
                feature_scale,
            );
            Err(EngineError::UnsupportedPlatform(
                "the MLX engine requires macOS on Apple Silicon".to_owned(),
            ))
        }
    }

    /// Runs a frozen sklearn multinomial/binary logistic probe while the embedding and
    /// classifier math remain MLX arrays on Metal. Only the final class probabilities
    /// are copied to host memory.
    #[allow(clippy::too_many_arguments)]
    pub fn probe_probabilities(
        &mut self,
        model: ModelHandle,
        probe_key: &str,
        text: &str,
        coefficients: &[Vec<f64>],
        intercept: &[f64],
        scaler_mean: &[f64],
        scaler_scale: &[f64],
    ) -> Result<Vec<f32>> {
        self.probe_probabilities_timed(
            model,
            probe_key,
            text,
            coefficients,
            intercept,
            scaler_mean,
            scaler_scale,
        )
        .map(|(probabilities, _)| probabilities)
    }

    /// Timed counterpart to [`Self::probe_probabilities`]. The embedding and
    /// classifier stage boundaries are evaluated synchronously to make timings meaningful.
    #[allow(clippy::too_many_arguments)]
    pub fn probe_probabilities_timed(
        &mut self,
        model: ModelHandle,
        probe_key: &str,
        text: &str,
        coefficients: &[Vec<f64>],
        intercept: &[f64],
        scaler_mean: &[f64],
        scaler_scale: &[f64],
    ) -> Result<(Vec<f32>, ProbeStageTimings)> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            let mut timings = ProbeStageTimings::default();
            let loaded = self
                .loaded
                .as_mut()
                .filter(|loaded| loaded.handle == model)
                .ok_or_else(|| EngineError::Backend("unknown model handle".to_owned()))?;
            let classes = if coefficients.len() == 1 {
                2
            } else {
                coefficients.len()
            };
            if scaler_mean.len() != 1024
                || scaler_scale.len() != 1024
                || coefficients.iter().any(|row| row.len() != 1024)
                || intercept.len() != coefficients.len()
                || scaler_scale.iter().any(|x| !x.is_finite() || *x <= 0.0)
                || scaler_mean
                    .iter()
                    .chain(intercept)
                    .chain(coefficients.iter().flatten())
                    .any(|x| !x.is_finite())
            {
                return Err(EngineError::Backend(
                    "invalid frozen probe dimensions or values".to_owned(),
                ));
            }
            let signature = scaler_mean
                .iter()
                .chain(scaler_scale)
                .chain(intercept)
                .chain(coefficients.iter().flatten())
                .map(|x| x.to_bits())
                .collect::<Vec<_>>();
            if let Some(cached) = loaded.probe_cache.get(probe_key) {
                if cached.signature != signature {
                    return Err(EngineError::Backend(format!(
                        "probe key {probe_key:?} reused with different frozen weights"
                    )));
                }
            } else {
                let weights = coefficients.iter().flatten().copied().collect::<Vec<_>>();
                let weights = array_from_f64(&weights, &[coefficients.len() as i32, 1024])?;
                let bias = array_from_f64(intercept, &[intercept.len() as i32])?;
                // StandardScaler can emit subnormal scales for nearly-constant
                // Harrier dimensions. Metal may flush those to zero, so algebraically
                // precondition only those dimensions: (x-mean)/scale = x/scale-mean/scale.
                let mut input_factor = Vec::with_capacity(1024);
                let mut adjusted_mean = Vec::with_capacity(1024);
                let mut adjusted_scale = Vec::with_capacity(1024);
                for (&mean, &scale) in scaler_mean.iter().zip(scaler_scale) {
                    if scale < f64::from(f32::MIN_POSITIVE) {
                        input_factor.push(1.0 / scale);
                        adjusted_mean.push(mean / scale);
                        adjusted_scale.push(1.0);
                    } else {
                        input_factor.push(1.0);
                        adjusted_mean.push(mean);
                        adjusted_scale.push(scale);
                    }
                }
                let factor = array_from_f64(&input_factor, &[1024])?;
                let mean = array_from_f64(&adjusted_mean, &[1024])?;
                let scale = array_from_f64(&adjusted_scale, &[1024])?;
                loaded.probe_cache.insert(
                    probe_key.to_owned(),
                    ResidentProbe {
                        signature,
                        weights,
                        bias,
                        factor,
                        mean,
                        scale,
                    },
                );
            }
            let resident = loaded
                .probe_cache
                .get(probe_key)
                .expect("probe cache inserted above");
            let weights = resident.weights.clone();
            let bias = resident.bias.clone();
            let factor = resident.factor.clone();
            let mean = resident.mean.clone();
            let scale = resident.scale.clone();
            let embedding_start = std::time::Instant::now();
            let (embedding, tokenization_ns) = loaded.embedder.embed_probe(text)?;
            timings.tokenization_ns = tokenization_ns;
            embedding.eval().map_err(mlx_error)?;
            timings.embedding_ns = embedding_start
                .elapsed()
                .as_nanos()
                .saturating_sub(tokenization_ns);
            let classifier_start = std::time::Instant::now();
            let scaled = embedding
                .multiply(factor)
                .map_err(mlx_error)?
                .subtract(mean)
                .map_err(mlx_error)?
                .divide(scale)
                .map_err(mlx_error)?;
            let mut logits = weights
                .matmul(&scaled)
                .map_err(mlx_error)?
                .add(bias)
                .map_err(mlx_error)?;
            if coefficients.len() == 1 {
                // sklearn's binary LogisticRegression stores only the class-1 logit.
                logits =
                    ops::stack(&[Array::from_f32(0.0), logits.index(0)], 0).map_err(mlx_error)?;
            }
            let probabilities = ops::softmax(&logits, None).map_err(mlx_error)?;
            probabilities.eval().map_err(mlx_error)?;
            timings.classifier_ns = classifier_start.elapsed().as_nanos();
            let values = probabilities.as_slice::<f32>().to_vec();
            if values.len() != classes || values.iter().any(|x| !x.is_finite()) {
                return Err(EngineError::Backend(format!(
                    "probe produced invalid probabilities: got {} expected {classes}, finite={}",
                    values.len(),
                    values.iter().all(|x| x.is_finite())
                )));
            }
            Ok((values, timings))
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = (
                model,
                probe_key,
                text,
                coefficients,
                intercept,
                scaler_mean,
                scaler_scale,
            );
            Err(EngineError::UnsupportedPlatform(
                "the MLX engine requires macOS on Apple Silicon".to_owned(),
            ))
        }
    }
}

impl DecisionEngine for MlxDecisionEngine {
    fn name(&self) -> &'static str {
        "mlx"
    }

    fn load_model(&mut self, model: &EngineModelSpec) -> Result<ModelHandle> {
        platform_check()?;
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            Device::set_default(&Device::gpu());
            let active = Device::try_default().map_err(mlx_error)?;
            if !matches!(active.get_type().map_err(mlx_error)?, DeviceType::Gpu)
                || active.get_index().map_err(mlx_error)? != 0
            {
                return Err(EngineError::Backend(format!(
                    "probe inference requires Device(gpu, 0), active device is {active}"
                )));
            }
        }
        if model.family != "qwen3-0.6b" {
            return Err(EngineError::UnsupportedModel(model.id.clone()));
        }
        if self.loaded.is_some() {
            return Err(EngineError::Backend(
                "v1 MLX engines support one resident model; unload it before loading another"
                    .to_owned(),
            ));
        }

        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            let root = model.root.as_ref().ok_or_else(|| {
                EngineError::ModelNotFound(format!("{} has no installed model directory", model.id))
            })?;
            let embedder = QwenEmbedder::load(root, model.max_length)?;
            self.next_handle += 1;
            let handle = ModelHandle(self.next_handle);
            self.loaded = Some(LoadedModel {
                handle,
                temperature: model.temperature,
                embedder,
                candidate_cache: HashMap::new(),
                probe_cache: HashMap::new(),
            });
            Ok(handle)
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = model;
            unreachable!("platform_check returned on an unsupported platform")
        }
    }

    fn evaluate(
        &mut self,
        model: ModelHandle,
        batch: EngineDecisionBatch,
    ) -> Result<EngineDecisionBatchResult> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            let loaded = self
                .loaded
                .as_mut()
                .filter(|loaded| loaded.handle == model)
                .ok_or_else(|| EngineError::Backend("unknown model handle".to_owned()))?;
            let mut candidate_cache_hits = 0;
            let mut candidate_cache_misses = 0;
            let mut decisions = Vec::with_capacity(batch.decisions.len());

            for decision in batch.decisions {
                if decision.candidates.is_empty() {
                    return Err(EngineError::Backend(format!(
                        "decision {:?} has no candidates",
                        decision.key
                    )));
                }
                let candidate_embeddings =
                    if let Some(cached) = loaded.candidate_cache.get(&decision.cache_key) {
                        candidate_cache_hits += 1;
                        cached.clone()
                    } else {
                        candidate_cache_misses += 1;
                        let values = decision
                            .candidates
                            .iter()
                            .map(|candidate| loaded.embedder.embed(&candidate.text))
                            .collect::<Result<Vec<_>>>()?;
                        let values = ops::stack(&values, 0).map_err(mlx_error)?;
                        loaded
                            .candidate_cache
                            .insert(decision.cache_key.clone(), values.clone());
                        values
                    };
                let query = loaded.embedder.embed(&decision.query)?;
                let logits = candidate_embeddings.matmul(&query).map_err(mlx_error)?;
                let probabilities = ops::softmax(
                    &logits
                        .divide(Array::from_f32(loaded.temperature))
                        .map_err(mlx_error)?,
                    None,
                )
                .map_err(mlx_error)?;
                probabilities.eval().map_err(mlx_error)?;
                let probabilities = probabilities.as_slice::<f32>().to_vec();
                if probabilities
                    .iter()
                    .any(|probability| !probability.is_finite())
                {
                    return Err(EngineError::Backend(
                        "model produced a non-finite score distribution".to_owned(),
                    ));
                }
                let confidence = normalized_entropy_confidence(&probabilities);
                decisions.push(EngineDecisionResult {
                    key: decision.key,
                    probabilities,
                    confidence,
                });
            }

            Ok(EngineDecisionBatchResult {
                decisions,
                candidate_cache_hits,
                candidate_cache_misses,
            })
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = (model, batch);
            Err(EngineError::UnsupportedPlatform(
                "the MLX engine requires macOS on Apple Silicon".to_owned(),
            ))
        }
    }

    fn unload_model(&mut self, model: ModelHandle) -> Result<()> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        if self
            .loaded
            .as_ref()
            .is_some_and(|loaded| loaded.handle == model)
        {
            self.loaded = None;
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        let _ = model;
        Ok(())
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
struct LoadedModel {
    handle: ModelHandle,
    temperature: f32,
    embedder: QwenEmbedder,
    candidate_cache: HashMap<String, Array>,
    probe_cache: HashMap<String, ResidentProbe>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
struct ResidentProbe {
    signature: Vec<u64>,
    weights: Array,
    bias: Array,
    factor: Array,
    mean: Array,
    scale: Array,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Deserialize)]
struct QwenConfig {
    vocab_size: i32,
    hidden_size: i32,
    intermediate_size: i32,
    num_hidden_layers: usize,
    num_attention_heads: i32,
    num_key_value_heads: i32,
    head_dim: i32,
    rms_norm_eps: f32,
    rope_theta: f32,
    max_position_embeddings: usize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
struct QwenEmbedder {
    model: QwenModel,
    tokenizer: Tokenizer,
    max_length: usize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl QwenEmbedder {
    fn load(root: &Path, requested_max_length: usize) -> Result<Self> {
        let config_path = root.join("config.json");
        let tokenizer_path = root.join("tokenizer.json");
        if !config_path.is_file() || !tokenizer_path.is_file() {
            return Err(EngineError::ModelLoad(format!(
                "{} is missing config.json or tokenizer.json",
                root.display()
            )));
        }
        let config =
            serde_json::from_slice::<QwenConfig>(&fs::read(&config_path).map_err(model_io)?)
                .map_err(|error| {
                    EngineError::ModelLoad(format!("invalid {}: {error}", config_path.display()))
                })?;
        validate_config(&config)?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|error| {
            EngineError::ModelLoad(format!("cannot load {}: {error}", tokenizer_path.display()))
        })?;
        let weights = load_weights(root)?;
        let model = QwenModel::from_weights(&config, weights)?;
        Ok(Self {
            model,
            tokenizer,
            max_length: requested_max_length.min(config.max_position_embeddings),
        })
    }

    fn embed(&mut self, text: &str) -> Result<Array> {
        let encoding = self.tokenizer.encode(text, true).map_err(|error| {
            EngineError::Backend(format!("tokenizing inference input failed: {error}"))
        })?;
        let ids = encoding.get_ids();
        if ids.is_empty() {
            return Err(EngineError::Backend(
                "tokenizer produced an empty input".to_owned(),
            ));
        }
        if ids.len() > self.max_length {
            return Err(EngineError::ContextTooLong(format!(
                "input has {} tokens but this model allows {}",
                ids.len(),
                self.max_length
            )));
        }
        let ids = Array::from(ids)
            .reshape(&[
                1,
                i32::try_from(ids.len()).map_err(|_| {
                    EngineError::ContextTooLong("token count exceeds MLX index range".to_owned())
                })?,
            ])
            .map_err(mlx_error)?;
        let hidden = self.model.forward(&ids)?;
        let pooled = hidden
            .index((0, -1))
            .as_dtype(Dtype::Float32)
            .map_err(mlx_error)?;
        l2_normalize(&pooled)
    }

    fn embed_probe(&mut self, text: &str) -> Result<(Array, u128)> {
        let tokenization_start = std::time::Instant::now();
        let encoding = self.tokenizer.encode(text, true).map_err(|error| {
            EngineError::Backend(format!("tokenizing inference input failed: {error}"))
        })?;
        let tokenization_ns = tokenization_start.elapsed().as_nanos();
        let ids = encoding.get_ids();
        if ids.is_empty() {
            return Err(EngineError::Backend(
                "tokenizer produced an empty input".to_owned(),
            ));
        }
        if ids.len() > self.max_length {
            return Err(EngineError::ContextTooLong(format!(
                "input has {} tokens but this model allows {}",
                ids.len(),
                self.max_length
            )));
        }
        let ids = Array::from(ids)
            .reshape(&[
                1,
                i32::try_from(ids.len()).map_err(|_| {
                    EngineError::ContextTooLong("token count exceeds MLX index range".to_owned())
                })?,
            ])
            .map_err(mlx_error)?;
        let hidden = self.model.forward(&ids)?;
        let pooled = hidden
            .index((0, -1))
            .as_dtype(Dtype::Float32)
            .map_err(mlx_error)?;
        Ok((l2_normalize(&pooled)?, tokenization_ns))
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
struct QwenModel {
    embedding: nn::Embedding,
    layers: Vec<QwenLayer>,
    norm: nn::RmsNorm,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl QwenModel {
    fn from_weights(config: &QwenConfig, mut weights: HashMap<String, Array>) -> Result<Self> {
        let prefix = if weights.contains_key("model.embed_tokens.weight") {
            "model."
        } else {
            ""
        };
        let mut embedding =
            nn::Embedding::new(config.vocab_size, config.hidden_size).map_err(mlx_error)?;
        embedding.weight.value =
            take_weight(&mut weights, &format!("{prefix}embed_tokens.weight"))?;
        let layers = (0..config.num_hidden_layers)
            .map(|index| QwenLayer::from_weights(config, prefix, index, &mut weights))
            .collect::<Result<Vec<_>>>()?;
        let mut norm = rms_norm(config.hidden_size, config.rms_norm_eps)?;
        norm.weight.value = take_weight(&mut weights, &format!("{prefix}norm.weight"))?;
        Ok(Self {
            embedding,
            layers,
            norm,
        })
    }

    fn forward(&mut self, ids: &Array) -> Result<Array> {
        let mut hidden = self.embedding.forward(ids).map_err(mlx_error)?;
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden)?;
        }
        self.norm.forward(&hidden).map_err(mlx_error)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
struct QwenLayer {
    attention: QwenAttention,
    mlp: QwenMlp,
    input_norm: nn::RmsNorm,
    post_attention_norm: nn::RmsNorm,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl QwenLayer {
    fn from_weights(
        config: &QwenConfig,
        model_prefix: &str,
        index: usize,
        weights: &mut HashMap<String, Array>,
    ) -> Result<Self> {
        let prefix = format!("{model_prefix}layers.{index}");
        let mut input_norm = rms_norm(config.hidden_size, config.rms_norm_eps)?;
        input_norm.weight.value =
            take_weight(weights, &format!("{prefix}.input_layernorm.weight"))?;
        let mut post_attention_norm = rms_norm(config.hidden_size, config.rms_norm_eps)?;
        post_attention_norm.weight.value = take_weight(
            weights,
            &format!("{prefix}.post_attention_layernorm.weight"),
        )?;
        Ok(Self {
            attention: QwenAttention::from_weights(
                config,
                &format!("{prefix}.self_attn"),
                weights,
            )?,
            mlp: QwenMlp::from_weights(config, &format!("{prefix}.mlp"), weights)?,
            input_norm,
            post_attention_norm,
        })
    }

    fn forward(&mut self, hidden: &Array) -> Result<Array> {
        let attention = self
            .attention
            .forward(&self.input_norm.forward(hidden).map_err(mlx_error)?)?;
        let hidden = hidden.add(attention).map_err(mlx_error)?;
        let mlp = self.mlp.forward(
            &self
                .post_attention_norm
                .forward(&hidden)
                .map_err(mlx_error)?,
        )?;
        hidden.add(mlp).map_err(mlx_error)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
struct QwenAttention {
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    o_proj: nn::Linear,
    q_norm: nn::RmsNorm,
    k_norm: nn::RmsNorm,
    rope: nn::Rope,
    heads: i32,
    key_value_heads: i32,
    head_dim: i32,
    scale: f32,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl QwenAttention {
    fn from_weights(
        config: &QwenConfig,
        prefix: &str,
        weights: &mut HashMap<String, Array>,
    ) -> Result<Self> {
        let mut q_proj = linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
        )?;
        q_proj.weight.value = take_weight(weights, &format!("{prefix}.q_proj.weight"))?;
        let mut k_proj = linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
        )?;
        k_proj.weight.value = take_weight(weights, &format!("{prefix}.k_proj.weight"))?;
        let mut v_proj = linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
        )?;
        v_proj.weight.value = take_weight(weights, &format!("{prefix}.v_proj.weight"))?;
        let mut o_proj = linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
        )?;
        o_proj.weight.value = take_weight(weights, &format!("{prefix}.o_proj.weight"))?;
        let mut q_norm = rms_norm(config.head_dim, config.rms_norm_eps)?;
        q_norm.weight.value = take_weight(weights, &format!("{prefix}.q_norm.weight"))?;
        let mut k_norm = rms_norm(config.head_dim, config.rms_norm_eps)?;
        k_norm.weight.value = take_weight(weights, &format!("{prefix}.k_norm.weight"))?;
        let rope = nn::RopeBuilder::new(config.head_dim)
            .base(config.rope_theta)
            .build()
            .expect("Qwen3 RoPE builder is infallible after config validation");
        let head_dim = f32::from(i16::try_from(config.head_dim).map_err(|_| {
            EngineError::ModelLoad(
                "Qwen3 attention head dimension exceeds supported range".to_owned(),
            )
        })?);
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rope,
            heads: config.num_attention_heads,
            key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            scale: head_dim.powf(-0.5),
        })
    }

    fn forward(&mut self, hidden: &Array) -> Result<Array> {
        let batch = hidden.dim(0);
        let length = hidden.dim(1);
        let mut query = self
            .q_proj
            .forward(hidden)
            .map_err(mlx_error)?
            .reshape(&[batch, length, self.heads, self.head_dim])
            .map_err(mlx_error)?;
        let mut key = self
            .k_proj
            .forward(hidden)
            .map_err(mlx_error)?
            .reshape(&[batch, length, self.key_value_heads, self.head_dim])
            .map_err(mlx_error)?;
        let value = self
            .v_proj
            .forward(hidden)
            .map_err(mlx_error)?
            .reshape(&[batch, length, self.key_value_heads, self.head_dim])
            .map_err(mlx_error)?
            .transpose_axes(&[0, 2, 1, 3])
            .map_err(mlx_error)?;
        query = self
            .q_norm
            .forward(&query)
            .map_err(mlx_error)?
            .transpose_axes(&[0, 2, 1, 3])
            .map_err(mlx_error)?;
        key = self
            .k_norm
            .forward(&key)
            .map_err(mlx_error)?
            .transpose_axes(&[0, 2, 1, 3])
            .map_err(mlx_error)?;
        query = self.rope.forward(&query).map_err(mlx_error)?;
        key = self.rope.forward(&key).map_err(mlx_error)?;
        let output = scaled_dot_product_attention(
            &query,
            &key,
            &value,
            self.scale,
            ScaledDotProductAttentionMask::Causal,
            None,
        )
        .map_err(mlx_error)?;
        let output = output
            .transpose_axes(&[0, 2, 1, 3])
            .map_err(mlx_error)?
            .reshape(&[batch, length, self.heads * self.head_dim])
            .map_err(mlx_error)?;
        self.o_proj.forward(&output).map_err(mlx_error)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
struct QwenMlp {
    gate: nn::Linear,
    up: nn::Linear,
    down: nn::Linear,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl QwenMlp {
    fn from_weights(
        config: &QwenConfig,
        prefix: &str,
        weights: &mut HashMap<String, Array>,
    ) -> Result<Self> {
        let mut gate_proj = linear(config.hidden_size, config.intermediate_size)?;
        gate_proj.weight.value = take_weight(weights, &format!("{prefix}.gate_proj.weight"))?;
        let mut up_proj = linear(config.hidden_size, config.intermediate_size)?;
        up_proj.weight.value = take_weight(weights, &format!("{prefix}.up_proj.weight"))?;
        let mut down_proj = linear(config.intermediate_size, config.hidden_size)?;
        down_proj.weight.value = take_weight(weights, &format!("{prefix}.down_proj.weight"))?;
        Ok(Self {
            gate: gate_proj,
            up: up_proj,
            down: down_proj,
        })
    }

    fn forward(&mut self, hidden: &Array) -> Result<Array> {
        let gate = nn::silu(self.gate.forward(hidden).map_err(mlx_error)?).map_err(mlx_error)?;
        let up = self.up.forward(hidden).map_err(mlx_error)?;
        self.down
            .forward(&gate.multiply(up).map_err(mlx_error)?)
            .map_err(mlx_error)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn linear(input: i32, output: i32) -> Result<nn::Linear> {
    nn::LinearBuilder::new(input, output)
        .bias(false)
        .build()
        .map_err(mlx_error)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn rms_norm(dimensions: i32, eps: f32) -> Result<nn::RmsNorm> {
    nn::RmsNormBuilder::new(dimensions)
        .eps(eps)
        .build()
        .map_err(mlx_error)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn load_weights(root: &Path) -> Result<HashMap<String, Array>> {
    let mut paths = fs::read_dir(root)
        .map_err(model_io)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension().and_then(|extension| extension.to_str()) == Some("safetensors")
        })
        .collect::<Vec<_>>();
    paths.sort();
    if paths.is_empty() {
        return Err(EngineError::ModelLoad(format!(
            "{} has no safetensors weights",
            root.display()
        )));
    }
    let mut weights = HashMap::new();
    for path in paths {
        let loaded = Array::load_safetensors(&path).map_err(|error| {
            EngineError::ModelLoad(format!("cannot load {}: {error}", path.display()))
        })?;
        for (name, value) in loaded {
            if weights.insert(name.clone(), value).is_some() {
                return Err(EngineError::ModelLoad(format!("duplicate weight {name:?}")));
            }
        }
    }
    Ok(weights)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn take_weight(weights: &mut HashMap<String, Array>, name: &str) -> Result<Array> {
    let value = weights
        .remove(name)
        .ok_or_else(|| EngineError::ModelLoad(format!("checkpoint is missing {name}")))?;
    value.eval().map_err(mlx_error)?;
    Ok(value)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn validate_config(config: &QwenConfig) -> Result<()> {
    let valid = config.vocab_size > 0
        && config.hidden_size > 0
        && config.intermediate_size > 0
        && config.num_hidden_layers > 0
        && config.num_attention_heads > 0
        && config.num_key_value_heads > 0
        && config.head_dim > 0
        && config.num_attention_heads % config.num_key_value_heads == 0
        && config.rms_norm_eps.is_finite()
        && config.rms_norm_eps > 0.0
        && config.rope_theta.is_finite()
        && config.rope_theta > 0.0
        && config.max_position_embeddings > 0;
    if valid {
        Ok(())
    } else {
        Err(EngineError::ModelLoad(
            "config.json is not a supported Qwen3 decoder configuration".to_owned(),
        ))
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn l2_normalize(values: &Array) -> Result<Array> {
    let norm = values
        .square()
        .map_err(mlx_error)?
        .sum(None)
        .map_err(mlx_error)?
        .sqrt()
        .map_err(mlx_error)?;
    values.divide(norm).map_err(mlx_error)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn array_from_f64(values: &[f64], shape: &[i32]) -> Result<Array> {
    let values = values.iter().map(|value| *value as f32).collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(EngineError::Backend(
            "probe parameters overflowed float32".to_owned(),
        ));
    }
    Ok(Array::from_slice(&values, shape))
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn ensure_varied_scorer(
    candidates: &[String],
    coefficients: &[f64],
    mean: &[f64],
    scale: &[f64],
) -> Result<()> {
    if candidates.len() < 2
        || coefficients.len() != 4096
        || mean.len() != 4096
        || scale.len() != 4096
        || coefficients
            .iter()
            .chain(mean)
            .chain(scale)
            .any(|x| !x.is_finite())
        || scale.iter().any(|x| *x <= 0.0)
    {
        return Err(EngineError::Backend(
            "invalid generic varied scorer dimensions or values".to_owned(),
        ));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn normalized_entropy_confidence(probabilities: &[f32]) -> f32 {
    if probabilities.len() <= 1 {
        return 1.0;
    }
    let entropy = -probabilities
        .iter()
        .filter(|probability| **probability > 0.0)
        .map(|probability| probability * probability.ln())
        .sum::<f32>();
    let count = f32::from(u16::try_from(probabilities.len()).unwrap_or(u16::MAX));
    (1.0 - entropy / count.ln()).clamp(0.0, 1.0)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[allow(clippy::needless_pass_by_value)]
fn mlx_error(error: mlx_rs::error::Exception) -> EngineError {
    EngineError::Backend(error.to_string())
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[allow(clippy::needless_pass_by_value)]
fn model_io(error: std::io::Error) -> EngineError {
    EngineError::ModelLoad(error.to_string())
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[allow(clippy::unnecessary_wraps)]
fn platform_check() -> Result<()> {
    Ok(())
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn platform_check() -> Result<()> {
    Err(EngineError::UnsupportedPlatform(
        "the MLX engine requires macOS on Apple Silicon".to_owned(),
    ))
}

#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
mod tests {
    use super::{QwenConfig, normalized_entropy_confidence, validate_config};

    #[test]
    fn validates_qwen3_embedding_configuration() {
        let config = QwenConfig {
            vocab_size: 151_669,
            hidden_size: 1_024,
            intermediate_size: 3_072,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 32_768,
        };
        assert!(validate_config(&config).is_ok());
    }

    #[test]
    fn derives_confidence_from_a_distribution() {
        assert!(normalized_entropy_confidence(&[0.5, 0.5]) < 1e-6);
        assert!(normalized_entropy_confidence(&[0.99, 0.01]) > 0.9);
    }
}
