//! Loopback HTTP compatibility frontend.

use std::sync::Arc;

use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use decengine_core::{DecengineError, ErrorCode, ErrorEnvelope};
use decengine_models::ModelRegistry;
use decengine_protocol::{
    HealthResponse, ModelList, SystemOneRequest, SystemOneResponse, WIRE_VERSION,
};
use decengine_runtime::DecisionService;
use tokio::net::TcpListener;

#[derive(Clone)]
struct AppState {
    service: Arc<DecisionService>,
    models: ModelList,
}

pub fn router(service: Arc<DecisionService>, registry: &ModelRegistry) -> Router {
    let state = AppState {
        service,
        models: ModelList::from_profiles(registry.profiles()),
    };
    Router::new()
        .route("/v1/systemone", post(decide))
        .route("/v1/models", get(models))
        .route("/healthz", get(health))
        .with_state(state)
}

/// Serves the configured router until the listener fails or the task is cancelled.
///
/// # Errors
///
/// Returns [`std::io::Error`] when the HTTP listener or connection service fails.
pub async fn serve(listener: TcpListener, application: Router) -> std::io::Result<()> {
    axum::serve(listener, application).await
}

async fn decide(
    State(state): State<AppState>,
    request: std::result::Result<Json<SystemOneRequest>, JsonRejection>,
) -> std::result::Result<Json<SystemOneResponse>, ApiError> {
    let Json(request) = request.map_err(|error| {
        ApiError(DecengineError::invalid_request(format!(
            "invalid JSON request: {}",
            error.body_text()
        )))
    })?;
    let request = request.into_core().map_err(ApiError)?;
    let response = state.service.decide(&request).map_err(ApiError)?;
    Ok(Json(response.into()))
}

async fn models(State(state): State<AppState>) -> Json<ModelList> {
    Json(state.models)
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_owned(),
        model: state.service.model_id().to_owned(),
        engine: state.service.engine_name().to_owned(),
        wire_version: WIRE_VERSION.to_owned(),
    })
}

struct ApiError(DecengineError);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match self.0.code {
            ErrorCode::InvalidRequest | ErrorCode::InvalidQuestion => StatusCode::BAD_REQUEST,
            ErrorCode::ModelNotFound => StatusCode::NOT_FOUND,
            ErrorCode::UnsupportedModel => StatusCode::UNPROCESSABLE_ENTITY,
            ErrorCode::ContextTooLong => StatusCode::PAYLOAD_TOO_LARGE,
            ErrorCode::OutOfMemory => StatusCode::INSUFFICIENT_STORAGE,
            ErrorCode::ModelLoadFailed
            | ErrorCode::EngineError
            | ErrorCode::InvalidManifest
            | ErrorCode::InternalError => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (status, Json(ErrorEnvelope::from(&self.0))).into_response()
    }
}

#[cfg(all(test, feature = "fixture-engine"))]
mod tests {
    use super::*;
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use decengine_engine::FixtureDecisionEngine;
    use decengine_models::ModelStore;
    use tower::ServiceExt;

    fn application() -> Router {
        let registry = ModelRegistry::bundled().unwrap();
        let store = ModelStore::new("unused", registry.clone());
        let resolved = store.resolve_fixture("Qwen/Qwen3-Embedding-0.6B").unwrap();
        let service =
            DecisionService::shared(Box::new(FixtureDecisionEngine::default()), resolved).unwrap();
        router(service, &registry)
    }

    #[tokio::test]
    async fn health_reports_loaded_engine() {
        let response = application()
            .oneshot(Request::get("/healthz").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["engine"], "fixture");
    }

    #[tokio::test]
    async fn models_lists_two_allowlisted_checkpoints() {
        let response = application()
            .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["data"].as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn post_decision_matches_wire_contract() {
        let body = include_bytes!("../../../tests/protocol/fixtures/systemone-request.json");
        let response = application()
            .oneshot(
                Request::post("/v1/systemone")
                    .header("content-type", "application/json")
                    .body(Body::from(body.as_slice()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 16 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["results"]["route"]["type"], "choice");
        assert_eq!(json["model"], "Qwen/Qwen3-Embedding-0.6B");
    }

    #[tokio::test]
    async fn bad_json_uses_stable_error_envelope() {
        let response = application()
            .oneshot(
                Request::post("/v1/systemone")
                    .header("content-type", "application/json")
                    .body(Body::from("{"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "invalid_request");
    }
}
