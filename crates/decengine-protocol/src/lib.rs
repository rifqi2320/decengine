//! Versioned wire types for the localhost compatibility frontend.

use decengine_core::{DecengineError, DecisionRequest, DecisionResponse, Result};
use decengine_models::ModelProfile;
use serde::{Deserialize, Serialize};

pub const WIRE_VERSION: &str = "systemone-v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SystemOneRequest(pub DecisionRequest);

impl SystemOneRequest {
    /// Validates the wire request and unwraps its core representation.
    ///
    /// # Errors
    ///
    /// Returns [`DecengineError`] when request validation fails.
    pub fn into_core(self) -> Result<DecisionRequest> {
        self.0.validate()?;
        Ok(self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SystemOneResponse(pub DecisionResponse);

impl From<DecisionResponse> for SystemOneResponse {
    fn from(value: DecisionResponse) -> Self {
        Self(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelCard>,
}

impl ModelList {
    pub fn from_profiles<'a>(profiles: impl Iterator<Item = &'a ModelProfile>) -> Self {
        Self {
            object: "list".to_owned(),
            data: profiles
                .map(|profile| ModelCard {
                    id: profile.id.clone(),
                    object: "model".to_owned(),
                    owned_by: profile
                        .id
                        .split_once('/')
                        .map_or_else(|| "unknown".to_owned(), |(owner, _)| owner.to_owned()),
                    capabilities: vec!["choice".to_owned(), "noul".to_owned(), "score".to_owned()],
                })
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCard {
    pub id: String,
    pub object: String,
    pub owned_by: String,
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub model: String,
    pub engine: String,
    pub wire_version: String,
}

/// Decodes and validates a `SystemOne` JSON request.
///
/// # Errors
///
/// Returns [`DecengineError`] when the payload is invalid JSON or violates request constraints.
pub fn decode_request(bytes: &[u8]) -> Result<DecisionRequest> {
    let request: SystemOneRequest = serde_json::from_slice(bytes).map_err(|error| {
        DecengineError::invalid_request(format!("request is not valid JSON: {error}"))
    })?;
    request.into_core()
}

#[cfg(test)]
mod tests {
    use super::*;
    use decengine_core::{DecisionResult, Usage};
    use decengine_models::ModelRegistry;
    use std::collections::BTreeMap;

    #[test]
    fn request_fixture_round_trips() {
        let source = include_bytes!("../../../tests/protocol/fixtures/systemone-request.json");
        let request = decode_request(source).unwrap();
        assert_eq!(request.questions.len(), 3);
        assert_eq!(request.model.as_deref(), Some("Qwen/Qwen3-Embedding-0.6B"));
        let encoded = serde_json::to_vec(&SystemOneRequest(request)).unwrap();
        assert!(decode_request(&encoded).is_ok());
    }

    #[test]
    fn response_shape_is_stable() {
        let response = DecisionResponse {
            id: "dec_test".to_owned(),
            model: "Qwen/Qwen3-Embedding-0.6B".to_owned(),
            created: 0,
            results: BTreeMap::from([(
                "route".to_owned(),
                DecisionResult::Choice {
                    selected: "billing".to_owned(),
                    probabilities: BTreeMap::from([
                        ("billing".to_owned(), 0.9),
                        ("sales".to_owned(), 0.1),
                    ]),
                    confidence: 0.7,
                },
            )]),
            usage: Usage::default(),
        };
        let json = serde_json::to_value(SystemOneResponse(response)).unwrap();
        assert_eq!(json["results"]["route"]["type"], "choice");
        assert_eq!(json["results"]["route"]["selected"], "billing");
    }

    #[test]
    fn models_fixture_contains_both_profiles() {
        let registry = ModelRegistry::bundled().unwrap();
        let models = ModelList::from_profiles(registry.profiles());
        assert_eq!(models.object, "list");
        assert_eq!(models.data.len(), 2);
    }
}
