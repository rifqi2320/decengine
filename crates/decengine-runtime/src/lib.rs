//! Shared decision orchestration used by both native and HTTP frontends.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use decengine_compiler::compile;
use decengine_core::{
    DecengineError, DecisionRequest, DecisionResponse, DecisionResult, ErrorCode, Result, Usage,
};
use decengine_engine::{
    DecisionEngine, DecisionKind, EngineDecision, EngineDecisionResult, EngineError, ModelHandle,
};
use decengine_models::ResolvedModel;
use parking_lot::Mutex;
use uuid::Uuid;

struct EngineState {
    engine: Box<dyn DecisionEngine>,
    model: ModelHandle,
}

pub struct DecisionService {
    state: Mutex<EngineState>,
    resolved: ResolvedModel,
}

impl DecisionService {
    /// Loads a resolved model into an engine and creates the shared orchestration service.
    ///
    /// # Errors
    ///
    /// Returns [`DecengineError`] when the backend cannot load the resolved model.
    pub fn new(mut engine: Box<dyn DecisionEngine>, resolved: ResolvedModel) -> Result<Self> {
        let handle = engine
            .load_model(&resolved.engine_spec())
            .map_err(map_engine_error)?;
        Ok(Self {
            state: Mutex::new(EngineState {
                engine,
                model: handle,
            }),
            resolved,
        })
    }

    /// Creates a reference-counted decision service.
    ///
    /// # Errors
    ///
    /// Returns [`DecengineError`] when the backend cannot load the resolved model.
    pub fn shared(engine: Box<dyn DecisionEngine>, resolved: ResolvedModel) -> Result<Arc<Self>> {
        Self::new(engine, resolved).map(Arc::new)
    }

    pub fn model_id(&self) -> &str {
        &self.resolved.profile.id
    }

    pub fn engine_name(&self) -> &'static str {
        self.state.lock().engine.name()
    }

    /// Compiles and evaluates one validated decision request.
    ///
    /// # Errors
    ///
    /// Returns [`DecengineError`] for invalid input, a model mismatch, backend failure, or invalid
    /// backend output.
    pub fn decide(&self, request: &DecisionRequest) -> Result<DecisionResponse> {
        if let Some(requested) = &request.model {
            if requested != self.model_id() {
                return Err(DecengineError::new(
                    ErrorCode::UnsupportedModel,
                    format!(
                        "engine loaded model {:?}, but request selected {requested:?}",
                        self.model_id()
                    ),
                ));
            }
        }

        let compiled = compile(request, &self.resolved.profile)?;
        let descriptions = compiled.engine_batch.decisions.clone();
        let mut state = self.state.lock();
        let handle = state.model;
        let evaluated = state
            .engine
            .evaluate(handle, compiled.engine_batch)
            .map_err(map_engine_error)?;
        drop(state);

        if descriptions.len() != evaluated.decisions.len() {
            return Err(DecengineError::new(
                ErrorCode::EngineError,
                "engine returned a different decision count",
            ));
        }
        let mut results = BTreeMap::new();
        for (description, evidence) in descriptions.iter().zip(&evaluated.decisions) {
            if description.key != evidence.key {
                return Err(DecengineError::new(
                    ErrorCode::EngineError,
                    "engine returned decisions out of order",
                ));
            }
            let result = materialize_result(description, evidence)?;
            results.insert(description.key.clone(), result);
        }

        Ok(DecisionResponse {
            id: format!("dec_{}", Uuid::new_v4().simple()),
            model: self.model_id().to_owned(),
            created: unix_time()?,
            results,
            usage: Usage {
                input_characters: compiled.input_characters,
                questions: request.questions.len(),
                candidates: compiled.candidates,
                candidate_cache_hits: evaluated.candidate_cache_hits,
                candidate_cache_misses: evaluated.candidate_cache_misses,
            },
        })
    }
}

impl Drop for DecisionService {
    fn drop(&mut self) {
        let state = self.state.get_mut();
        let _ = state.engine.unload_model(state.model);
    }
}

fn materialize_result(
    decision: &EngineDecision,
    evidence: &EngineDecisionResult,
) -> Result<DecisionResult> {
    validate_probabilities(decision, &evidence.probabilities)?;
    if !evidence.confidence.is_finite() {
        return Err(DecengineError::new(
            ErrorCode::EngineError,
            format!("invalid confidence for {:?}", decision.key),
        ));
    }
    let confidence = evidence.confidence.clamp(0.0, 1.0);

    match decision.kind {
        DecisionKind::Choice => {
            let selected = evidence
                .probabilities
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(index, _)| decision.candidates[index].id.clone())
                .ok_or_else(|| {
                    DecengineError::new(ErrorCode::EngineError, "empty choice distribution")
                })?;
            let probabilities = decision
                .candidates
                .iter()
                .zip(&evidence.probabilities)
                .map(|(candidate, probability)| (candidate.id.clone(), *probability))
                .collect();
            Ok(DecisionResult::Choice {
                selected,
                probabilities,
                confidence,
            })
        }
        DecisionKind::Noul => {
            let supported = decision
                .candidates
                .iter()
                .position(|candidate| candidate.id == "supported")
                .ok_or_else(|| {
                    DecengineError::new(
                        ErrorCode::EngineError,
                        "noul decision is missing supported candidate",
                    )
                })?;
            Ok(DecisionResult::Noul {
                value: evidence.probabilities[supported],
                confidence,
            })
        }
        DecisionKind::Score => {
            let value = decision
                .candidates
                .iter()
                .zip(&evidence.probabilities)
                .map(|(candidate, probability)| {
                    candidate.score_value.unwrap_or_default() * probability
                })
                .sum();
            let distribution = decision
                .candidates
                .iter()
                .zip(&evidence.probabilities)
                .map(|(candidate, probability)| (candidate.id.clone(), *probability))
                .collect();
            let legend = decision
                .candidates
                .iter()
                .map(|candidate| (candidate.id.clone(), candidate.text.clone()))
                .collect();
            Ok(DecisionResult::Score {
                value,
                distribution,
                legend,
                confidence,
            })
        }
    }
}

fn validate_probabilities(decision: &EngineDecision, probabilities: &[f32]) -> Result<()> {
    if probabilities.len() != decision.candidates.len() || probabilities.is_empty() {
        return Err(DecengineError::new(
            ErrorCode::EngineError,
            format!("invalid distribution length for {:?}", decision.key),
        ));
    }
    if probabilities
        .iter()
        .any(|probability| !probability.is_finite() || *probability < 0.0 || *probability > 1.0)
    {
        return Err(DecengineError::new(
            ErrorCode::EngineError,
            format!("invalid probability for {:?}", decision.key),
        ));
    }
    let sum = probabilities.iter().sum::<f32>();
    if (sum - 1.0).abs() > 1e-3 {
        return Err(DecengineError::new(
            ErrorCode::EngineError,
            format!("distribution for {:?} sums to {sum}", decision.key),
        ));
    }
    Ok(())
}

fn map_engine_error(error: EngineError) -> DecengineError {
    let (code, message) = match error {
        EngineError::UnsupportedPlatform(message) => (
            ErrorCode::UnsupportedModel,
            format!("unsupported platform: {message}"),
        ),
        EngineError::ModelNotFound(message) => (
            ErrorCode::ModelNotFound,
            format!("model not found: {message}"),
        ),
        EngineError::UnsupportedModel(message) => (
            ErrorCode::UnsupportedModel,
            format!("unsupported model: {message}"),
        ),
        EngineError::ModelLoad(message) => (
            ErrorCode::ModelLoadFailed,
            format!("model load failed: {message}"),
        ),
        EngineError::ContextTooLong(message) => (
            ErrorCode::ContextTooLong,
            format!("context too long: {message}"),
        ),
        EngineError::OutOfMemory(message) => {
            (ErrorCode::OutOfMemory, format!("out of memory: {message}"))
        }
        EngineError::Backend(message) => {
            (ErrorCode::EngineError, format!("engine failure: {message}"))
        }
    };
    DecengineError::new(code, message)
}

fn unix_time() -> Result<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|error| DecengineError::new(ErrorCode::InternalError, error.to_string()))
}

#[cfg(all(test, feature = "fixture-engine"))]
mod tests {
    use super::*;
    use decengine_core::{Question, ScoreLevel};
    use decengine_engine::FixtureDecisionEngine;
    use decengine_models::{ModelRegistry, ModelStore};
    use serde_json::json;

    fn service() -> DecisionService {
        let registry = ModelRegistry::bundled().unwrap();
        let store = ModelStore::new("unused", registry);
        let model = store.resolve_fixture("Qwen/Qwen3-Embedding-0.6B").unwrap();
        DecisionService::new(Box::new(FixtureDecisionEngine::default()), model).unwrap()
    }

    fn request() -> DecisionRequest {
        DecisionRequest {
            model: Some("Qwen/Qwen3-Embedding-0.6B".to_owned()),
            state: json!({"subject": "refund charged twice", "deadline": "today"}),
            questions: BTreeMap::from([
                (
                    "route".to_owned(),
                    Question::Choice {
                        prompt: "Which team should handle this?".to_owned(),
                        options: BTreeMap::from([
                            ("billing".to_owned(), "payments charges refunds".to_owned()),
                            (
                                "sales".to_owned(),
                                "pricing purchasing contracts".to_owned(),
                            ),
                        ]),
                    },
                ),
                (
                    "severity".to_owned(),
                    Question::Score {
                        prompt: "How severe?".to_owned(),
                        levels: vec![
                            ScoreLevel {
                                label: "low".to_owned(),
                                criterion: "minor inconvenience".to_owned(),
                                value: 0.0,
                            },
                            ScoreLevel {
                                label: "high".to_owned(),
                                criterion: "critical business impact".to_owned(),
                                value: 1.0,
                            },
                        ],
                    },
                ),
                (
                    "urgent".to_owned(),
                    Question::Noul {
                        prompt: "Does this require attention today?".to_owned(),
                    },
                ),
            ]),
        }
    }

    #[test]
    fn produces_typed_results_from_one_core() {
        let response = service().decide(&request()).unwrap();
        assert_eq!(response.results.len(), 3);
        assert!(matches!(
            response.results.get("route"),
            Some(DecisionResult::Choice { .. })
        ));
        assert!(matches!(
            response.results.get("urgent"),
            Some(DecisionResult::Noul { .. })
        ));
        assert!(matches!(
            response.results.get("severity"),
            Some(DecisionResult::Score { .. })
        ));
        assert_eq!(response.usage.candidate_cache_misses, 3);
    }

    #[test]
    fn second_request_hits_candidate_cache() {
        let service = service();
        service.decide(&request()).unwrap();
        let response = service.decide(&request()).unwrap();
        assert_eq!(response.usage.candidate_cache_hits, 3);
    }

    #[test]
    fn rejects_request_for_different_model() {
        let mut request = request();
        request.model = Some("microsoft/harrier-oss-v1-0.6b".to_owned());
        assert_eq!(
            service().decide(&request).unwrap_err().code,
            ErrorCode::UnsupportedModel
        );
    }
}
