//! Coarse-grained inference boundary.
//!
//! Implementations own the complete tensor/scoring pipeline and return only small
//! candidate distributions. MLX arrays and device types never cross this crate API.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("unsupported platform: {0}")]
    UnsupportedPlatform(String),
    #[error("model not found: {0}")]
    ModelNotFound(String),
    #[error("unsupported model: {0}")]
    UnsupportedModel(String),
    #[error("model load failed: {0}")]
    ModelLoad(String),
    #[error("context too long: {0}")]
    ContextTooLong(String),
    #[error("out of memory: {0}")]
    OutOfMemory(String),
    #[error("engine failure: {0}")]
    Backend(String),
}

pub type Result<T> = std::result::Result<T, EngineError>;

#[derive(Debug, Clone, PartialEq)]
pub struct EngineModelSpec {
    pub id: String,
    pub root: Option<PathBuf>,
    pub family: String,
    pub temperature: f32,
    pub max_length: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelHandle(pub u64);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionKind {
    Choice,
    Noul,
    Score,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineDecisionBatch {
    pub decisions: Vec<EngineDecision>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineDecision {
    pub key: String,
    pub kind: DecisionKind,
    pub query: String,
    pub candidates: Vec<EngineCandidate>,
    pub cache_key: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineCandidate {
    pub id: String,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score_value: Option<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EngineDecisionBatchResult {
    pub decisions: Vec<EngineDecisionResult>,
    pub candidate_cache_hits: usize,
    pub candidate_cache_misses: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EngineDecisionResult {
    pub key: String,
    pub probabilities: Vec<f32>,
    pub confidence: f32,
}

pub trait DecisionEngine: Send {
    fn name(&self) -> &'static str;

    /// Loads one model and returns the backend-owned handle.
    ///
    /// # Errors
    ///
    /// Returns an error when the model is unsupported, missing, invalid, or cannot be loaded by
    /// the backend.
    fn load_model(&mut self, model: &EngineModelSpec) -> Result<ModelHandle>;

    /// Evaluates a compiled decision batch with a previously loaded model.
    ///
    /// # Errors
    ///
    /// Returns an error when the handle is unknown, an input exceeds backend limits, or inference
    /// fails.
    fn evaluate(
        &mut self,
        model: ModelHandle,
        batch: EngineDecisionBatch,
    ) -> Result<EngineDecisionBatchResult>;

    /// Releases backend resources associated with a model handle.
    ///
    /// # Errors
    ///
    /// Returns an error when the backend cannot release the requested model.
    fn unload_model(&mut self, model: ModelHandle) -> Result<()>;
}

#[cfg(feature = "fixture-engine")]
mod fixture {
    use std::collections::{BTreeSet, HashMap};

    use super::{
        DecisionEngine, EngineDecisionBatch, EngineDecisionBatchResult, EngineDecisionResult,
        EngineError, EngineModelSpec, ModelHandle, Result,
    };

    /// Deterministic lexical backend for API development and tests only.
    ///
    /// It is intentionally named `FixtureDecisionEngine` so it cannot be confused with
    /// model inference. Release scripts never enable this feature.
    pub struct FixtureDecisionEngine {
        next_handle: u64,
        loaded: HashMap<ModelHandle, EngineModelSpec>,
        candidate_cache: HashMap<String, Vec<BTreeSet<String>>>,
    }

    impl Default for FixtureDecisionEngine {
        fn default() -> Self {
            Self {
                next_handle: 1,
                loaded: HashMap::new(),
                candidate_cache: HashMap::new(),
            }
        }
    }

    impl DecisionEngine for FixtureDecisionEngine {
        fn name(&self) -> &'static str {
            "fixture"
        }

        fn load_model(&mut self, model: &EngineModelSpec) -> Result<ModelHandle> {
            let handle = ModelHandle(self.next_handle);
            self.next_handle += 1;
            self.loaded.insert(handle, model.clone());
            Ok(handle)
        }

        fn evaluate(
            &mut self,
            model: ModelHandle,
            batch: EngineDecisionBatch,
        ) -> Result<EngineDecisionBatchResult> {
            let spec = self
                .loaded
                .get(&model)
                .ok_or_else(|| EngineError::Backend("unknown model handle".to_owned()))?;
            let mut hits = 0;
            let mut misses = 0;
            let mut results = Vec::with_capacity(batch.decisions.len());

            for decision in batch.decisions {
                let candidate_tokens =
                    if let Some(tokens) = self.candidate_cache.get(&decision.cache_key) {
                        hits += 1;
                        tokens.clone()
                    } else {
                        misses += 1;
                        let tokens = decision
                            .candidates
                            .iter()
                            .map(|candidate| tokenize(&candidate.text))
                            .collect::<Vec<_>>();
                        self.candidate_cache
                            .insert(decision.cache_key.clone(), tokens.clone());
                        tokens
                    };

                let query_tokens = tokenize(&decision.query);
                let logits = candidate_tokens
                    .iter()
                    .enumerate()
                    .map(|(index, candidate)| {
                        lexical_similarity(&query_tokens, candidate)
                            + deterministic_tiebreak(&decision.candidates[index].id)
                    })
                    .collect::<Vec<_>>();
                let probabilities = softmax(&logits, spec.temperature.max(0.01));
                let confidence = normalized_entropy_confidence(&probabilities);
                results.push(EngineDecisionResult {
                    key: decision.key,
                    probabilities,
                    confidence,
                });
            }

            Ok(EngineDecisionBatchResult {
                decisions: results,
                candidate_cache_hits: hits,
                candidate_cache_misses: misses,
            })
        }

        fn unload_model(&mut self, model: ModelHandle) -> Result<()> {
            self.loaded.remove(&model);
            Ok(())
        }
    }

    fn tokenize(text: &str) -> BTreeSet<String> {
        text.to_lowercase()
            .split(|character: char| !character.is_alphanumeric())
            .filter(|token| token.len() > 1)
            .map(ToOwned::to_owned)
            .collect()
    }

    fn lexical_similarity(left: &BTreeSet<String>, right: &BTreeSet<String>) -> f32 {
        if left.is_empty() || right.is_empty() {
            return 0.0;
        }
        let shared = bounded_count(left.intersection(right).count());
        shared / (bounded_count(left.len()) * bounded_count(right.len())).sqrt()
    }

    fn deterministic_tiebreak(value: &str) -> f32 {
        let hash = value.bytes().fold(2_166_136_261_u32, |state, byte| {
            state.wrapping_mul(16_777_619) ^ u32::from(byte)
        });
        f32::from(u16::try_from(hash % 997).expect("modulo 997 always fits in u16")) / 997_000.0
    }

    fn softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
        let scaled = logits
            .iter()
            .map(|value| value / temperature)
            .collect::<Vec<_>>();
        let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exponents = scaled
            .iter()
            .map(|value| (value - max).exp())
            .collect::<Vec<_>>();
        let denominator = exponents.iter().sum::<f32>();
        exponents
            .into_iter()
            .map(|value| value / denominator)
            .collect()
    }

    fn normalized_entropy_confidence(probabilities: &[f32]) -> f32 {
        if probabilities.len() <= 1 {
            return 1.0;
        }
        let entropy = -probabilities
            .iter()
            .filter(|probability| **probability > 0.0)
            .map(|probability| probability * probability.ln())
            .sum::<f32>();
        (1.0 - entropy / bounded_count(probabilities.len()).ln()).clamp(0.0, 1.0)
    }

    fn bounded_count(value: usize) -> f32 {
        f32::from(u16::try_from(value).unwrap_or(u16::MAX))
    }

    #[cfg(test)]
    mod tests {
        use super::{FixtureDecisionEngine, normalized_entropy_confidence, softmax};
        use crate::{
            DecisionEngine, DecisionKind, EngineCandidate, EngineDecision, EngineDecisionBatch,
            EngineModelSpec,
        };

        fn model() -> EngineModelSpec {
            EngineModelSpec {
                id: "fixture".to_owned(),
                root: None,
                family: "fixture".to_owned(),
                temperature: 0.1,
                max_length: 1024,
            }
        }

        fn batch(cache_key: &str) -> EngineDecisionBatch {
            EngineDecisionBatch {
                decisions: vec![EngineDecision {
                    key: "route".to_owned(),
                    kind: DecisionKind::Choice,
                    query: "refund charged twice".to_owned(),
                    candidates: vec![
                        EngineCandidate {
                            id: "billing".to_owned(),
                            text: "billing payments refund charge".to_owned(),
                            score_value: None,
                        },
                        EngineCandidate {
                            id: "sales".to_owned(),
                            text: "sales pricing contracts".to_owned(),
                            score_value: None,
                        },
                    ],
                    cache_key: cache_key.to_owned(),
                }],
            }
        }

        #[test]
        fn chooses_lexically_matching_candidate() {
            let mut engine = FixtureDecisionEngine::default();
            let handle = engine.load_model(&model()).unwrap();
            let output = engine.evaluate(handle, batch("a")).unwrap();
            let probabilities = &output.decisions[0].probabilities;
            assert!(probabilities[0] > 0.9);
            assert!(probabilities[0] > probabilities[1]);
            assert_eq!(output.candidate_cache_misses, 1);
        }

        #[test]
        fn reuses_candidate_cache() {
            let mut engine = FixtureDecisionEngine::default();
            let handle = engine.load_model(&model()).unwrap();
            engine.evaluate(handle, batch("same")).unwrap();
            let output = engine.evaluate(handle, batch("same")).unwrap();
            assert_eq!(output.candidate_cache_hits, 1);
            assert_eq!(output.candidate_cache_misses, 0);
        }

        #[test]
        fn probabilities_are_normalized() {
            let probabilities = softmax(&[1.0, 2.0, 3.0], 0.5);
            let sum = probabilities.iter().sum::<f32>();
            assert!((sum - 1.0).abs() < 1e-6);
            assert!(normalized_entropy_confidence(&probabilities) <= 1.0);
        }
    }
}

#[cfg(feature = "fixture-engine")]
pub use fixture::FixtureDecisionEngine;
