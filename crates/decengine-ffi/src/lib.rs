//! Panic-contained C ABI consumed by the pure-Python CFFI wrapper.

use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::ptr;

use decengine_core::{DecengineError, DecisionRequest, ErrorCode};
use decengine_engine::DecisionEngine;
use decengine_models::{ModelRegistry, ModelStore};
use decengine_runtime::DecisionService;
use libc::c_char;
use serde::Deserialize;

const ABI_VERSION: u32 = 1;
static VERSION: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").expect("empty CString"));
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeStatus {
    Ok = 0,
    InvalidArgument = 1,
    ModelNotFound = 2,
    UnsupportedModel = 3,
    ModelLoadFailed = 4,
    ContextTooLong = 5,
    EngineError = 6,
    OutOfMemory = 7,
    InvalidManifest = 8,
    InternalError = 255,
}

#[allow(non_camel_case_types)]
pub struct de_engine {
    service: DecisionService,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct EngineOptions {
    #[serde(default)]
    engine: Option<String>,
    #[serde(default)]
    model_store: Option<PathBuf>,
    #[serde(default)]
    query_template: Option<String>,
    #[serde(default)]
    candidate_template: Option<String>,
    #[serde(default)]
    query_instruction_template: Option<String>,
    #[serde(default)]
    choice_candidate_template: Option<String>,
    #[serde(default)]
    score_candidate_template: Option<String>,
    #[serde(default)]
    noul_unsupported: Option<String>,
    #[serde(default)]
    noul_supported: Option<String>,
    #[serde(default)]
    temperature: Option<f32>,
}

#[unsafe(no_mangle)]
pub extern "C" fn de_abi_version() -> u32 {
    ABI_VERSION
}

#[unsafe(no_mangle)]
pub extern "C" fn de_version() -> *const c_char {
    VERSION.as_ptr().cast()
}

#[unsafe(no_mangle)]
/// Creates an engine and writes its owning handle to `out_engine`.
///
/// # Safety
///
/// `model_id` must point to a valid NUL-terminated UTF-8 string. `options_json` may be null or must
/// point to the same. `out_engine` must be a valid writable pointer; the returned handle must later
/// be passed exactly once to [`de_engine_destroy`].
pub unsafe extern "C" fn de_engine_create(
    model_id: *const c_char,
    options_json: *const c_char,
    out_engine: *mut *mut de_engine,
) -> DeStatus {
    ffi_boundary(|| {
        if out_engine.is_null() {
            return Err(DecengineError::invalid_request(
                "out_engine must not be null",
            ));
        }
        // SAFETY: the pointer was checked for null and the ABI requires it to be writable.
        unsafe { ptr::write(out_engine, ptr::null_mut()) };
        let model_id = unsafe { required_utf8(model_id, "model_id") }?;
        let options = unsafe { parse_options(options_json) }?;
        let service = create_service(model_id, &options)?;
        let value = Box::new(de_engine { service });
        // SAFETY: out_engine is a checked caller-owned output slot.
        unsafe { ptr::write(out_engine, Box::into_raw(value)) };
        Ok(())
    })
}

#[unsafe(no_mangle)]
/// Evaluates a JSON decision request and writes an allocated JSON string to the output slot.
///
/// # Safety
///
/// `engine` must be a live handle returned by [`de_engine_create`]. `request_json` must point to a
/// valid NUL-terminated UTF-8 string. `out_response_json` must be writable; a successful result must
/// later be passed exactly once to [`de_string_free`].
pub unsafe extern "C" fn de_engine_decide_json(
    engine: *mut de_engine,
    request_json: *const c_char,
    out_response_json: *mut *mut c_char,
) -> DeStatus {
    ffi_boundary(|| {
        if engine.is_null() {
            return Err(DecengineError::invalid_request("engine must not be null"));
        }
        if out_response_json.is_null() {
            return Err(DecengineError::invalid_request(
                "out_response_json must not be null",
            ));
        }
        // SAFETY: the pointer was checked for null and the ABI requires it to be writable.
        unsafe { ptr::write(out_response_json, ptr::null_mut()) };
        let request_json = unsafe { required_utf8(request_json, "request_json") }?;
        let request: DecisionRequest = serde_json::from_str(request_json).map_err(|error| {
            DecengineError::invalid_request(format!("invalid request JSON: {error}"))
        })?;
        // SAFETY: engine is non-null and was allocated by de_engine_create.
        let response = unsafe { &*engine }.service.decide(&request)?;
        let json = serde_json::to_string(&response)
            .map_err(|error| DecengineError::new(ErrorCode::InternalError, error.to_string()))?;
        let value = CString::new(json).map_err(|_| {
            DecengineError::new(ErrorCode::InternalError, "response contained a NUL byte")
        })?;
        // SAFETY: out_response_json is a checked caller-owned output slot.
        unsafe { ptr::write(out_response_json, value.into_raw()) };
        Ok(())
    })
}

#[unsafe(no_mangle)]
/// Destroys an engine handle created by [`de_engine_create`].
///
/// # Safety
///
/// `engine` must be null or a live handle returned by [`de_engine_create`] that has not previously
/// been destroyed.
pub unsafe extern "C" fn de_engine_destroy(engine: *mut de_engine) {
    if engine.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the ABI requires a pointer returned by de_engine_create exactly once.
        drop(unsafe { Box::from_raw(engine) });
    }));
}

#[unsafe(no_mangle)]
/// Frees a response string allocated by [`de_engine_decide_json`].
///
/// # Safety
///
/// `value` must be null or a live pointer returned through `out_response_json` that has not
/// previously been freed.
pub unsafe extern "C" fn de_string_free(value: *mut c_char) {
    if value.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the ABI requires a pointer returned by de_engine_decide_json exactly once.
        drop(unsafe { CString::from_raw(value) });
    }));
}

#[unsafe(no_mangle)]
pub extern "C" fn de_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| slot.borrow().as_ptr())
}

fn create_service(
    model_id: &str,
    options: &EngineOptions,
) -> Result<DecisionService, DecengineError> {
    let registry = ModelRegistry::bundled()?;
    let root = options
        .model_store
        .clone()
        .or_else(|| std::env::var_os("DECENGINE_HOME").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from(".decengine"));
    let store = ModelStore::new(root, registry);
    let engine_name = options.engine.as_deref().unwrap_or(default_engine_name());
    let (engine, mut model): (Box<dyn DecisionEngine>, _) = match engine_name {
        "fixture" => {
            #[cfg(feature = "fixture-engine")]
            {
                (
                    Box::new(decengine_engine::FixtureDecisionEngine::default()),
                    store.resolve_fixture(model_id)?,
                )
            }
            #[cfg(not(feature = "fixture-engine"))]
            {
                return Err(DecengineError::new(
                    ErrorCode::UnsupportedModel,
                    "this native library was built without fixture-engine",
                ));
            }
        }
        "mlx" => {
            #[cfg(feature = "mlx")]
            {
                let engine = decengine_engine_mlx::MlxDecisionEngine::new().map_err(|error| {
                    DecengineError::new(ErrorCode::ModelLoadFailed, error.to_string())
                })?;
                (Box::new(engine), store.resolve_installed(model_id)?)
            }
            #[cfg(not(feature = "mlx"))]
            {
                return Err(DecengineError::new(
                    ErrorCode::UnsupportedModel,
                    "this native library was built without the mlx backend",
                ));
            }
        }
        other => {
            return Err(DecengineError::invalid_request(format!(
                "unknown engine {other:?}; expected fixture or mlx"
            )));
        }
    };
    if let Some(template) = &options.query_template {
        model.profile.query_template = template.clone();
    }
    if let Some(template) = &options.candidate_template {
        model.profile.candidate_template = template.clone();
    }
    if let Some(template) = &options.query_instruction_template {
        model.profile.query_instruction_template = template.clone();
    }
    if let Some(template) = &options.choice_candidate_template {
        model.profile.choice_candidate_template = template.clone();
    }
    if let Some(template) = &options.score_candidate_template {
        model.profile.score_candidate_template = template.clone();
    }
    if let Some(text) = &options.noul_unsupported {
        model.profile.noul_unsupported = text.clone();
    }
    if let Some(text) = &options.noul_supported {
        model.profile.noul_supported = text.clone();
    }
    if let Some(temperature) = options.temperature {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(DecengineError::invalid_request(
                "temperature must be positive and finite",
            ));
        }
        model.profile.temperature = temperature;
    }
    DecisionService::new(engine, model)
}

fn default_engine_name() -> &'static str {
    #[cfg(feature = "fixture-engine")]
    {
        "fixture"
    }
    #[cfg(all(not(feature = "fixture-engine"), feature = "mlx"))]
    {
        "mlx"
    }
    #[cfg(not(any(feature = "fixture-engine", feature = "mlx")))]
    {
        "unavailable"
    }
}

unsafe fn parse_options(pointer: *const c_char) -> Result<EngineOptions, DecengineError> {
    if pointer.is_null() {
        return Ok(EngineOptions::default());
    }
    let value = unsafe { required_utf8(pointer, "options_json") }?;
    if value.trim().is_empty() {
        return Ok(EngineOptions::default());
    }
    serde_json::from_str(value)
        .map_err(|error| DecengineError::invalid_request(format!("invalid options JSON: {error}")))
}

unsafe fn required_utf8<'a>(pointer: *const c_char, name: &str) -> Result<&'a str, DecengineError> {
    if pointer.is_null() {
        return Err(DecengineError::invalid_request(format!(
            "{name} must not be null"
        )));
    }
    // SAFETY: the ABI requires a valid NUL-terminated string for non-null inputs.
    unsafe { CStr::from_ptr(pointer) }
        .to_str()
        .map_err(|_| DecengineError::invalid_request(format!("{name} must be UTF-8")))
}

fn ffi_boundary(function: impl FnOnce() -> Result<(), DecengineError>) -> DeStatus {
    match catch_unwind(AssertUnwindSafe(function)) {
        Ok(Ok(())) => {
            set_last_error("");
            DeStatus::Ok
        }
        Ok(Err(error)) => {
            set_last_error(&error.message);
            status_for(error.code)
        }
        Err(_) => {
            set_last_error("internal panic contained at the C ABI boundary");
            DeStatus::InternalError
        }
    }
}

fn status_for(code: ErrorCode) -> DeStatus {
    match code {
        ErrorCode::InvalidRequest | ErrorCode::InvalidQuestion => DeStatus::InvalidArgument,
        ErrorCode::ModelNotFound => DeStatus::ModelNotFound,
        ErrorCode::UnsupportedModel => DeStatus::UnsupportedModel,
        ErrorCode::ModelLoadFailed => DeStatus::ModelLoadFailed,
        ErrorCode::ContextTooLong => DeStatus::ContextTooLong,
        ErrorCode::EngineError => DeStatus::EngineError,
        ErrorCode::OutOfMemory => DeStatus::OutOfMemory,
        ErrorCode::InvalidManifest => DeStatus::InvalidManifest,
        ErrorCode::InternalError => DeStatus::InternalError,
    }
}

fn set_last_error(message: &str) {
    let sanitized = message.replace('\0', "\\0");
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = CString::new(sanitized).expect("sanitized error has no NUL");
    });
}

#[cfg(all(test, feature = "fixture-engine"))]
mod tests {
    use super::*;

    #[test]
    fn abi_version_is_stable() {
        assert_eq!(de_abi_version(), 1);
    }

    #[test]
    fn null_output_pointer_is_rejected() {
        let model = CString::new("Qwen/Qwen3-Embedding-0.6B").unwrap();
        let status = unsafe { de_engine_create(model.as_ptr(), ptr::null(), ptr::null_mut()) };
        assert_eq!(status, DeStatus::InvalidArgument);
    }

    #[test]
    fn fixture_round_trip_uses_alloc_and_free_contract() {
        let model = CString::new("Qwen/Qwen3-Embedding-0.6B").unwrap();
        let options = CString::new(r#"{"engine":"fixture"}"#).unwrap();
        let mut engine = ptr::null_mut();
        assert_eq!(
            unsafe { de_engine_create(model.as_ptr(), options.as_ptr(), &raw mut engine) },
            DeStatus::Ok
        );

        let request = CString::new(
            r#"{"state":"refund","questions":{"route":{"type":"choice","prompt":"Which?","options":{"billing":"refund","sales":"pricing"}}}}"#,
        )
        .unwrap();
        let mut response = ptr::null_mut();
        assert_eq!(
            unsafe { de_engine_decide_json(engine, request.as_ptr(), &raw mut response) },
            DeStatus::Ok
        );
        let json = unsafe { CStr::from_ptr(response) }.to_str().unwrap();
        assert!(json.contains("billing"));
        unsafe {
            de_string_free(response);
            de_engine_destroy(engine);
        }
    }

    #[test]
    fn malformed_json_returns_diagnostic() {
        let model = CString::new("Qwen/Qwen3-Embedding-0.6B").unwrap();
        let mut engine = ptr::null_mut();
        assert_eq!(
            unsafe { de_engine_create(model.as_ptr(), ptr::null(), &raw mut engine) },
            DeStatus::Ok
        );
        let request = CString::new("not json").unwrap();
        let mut response = ptr::null_mut();
        assert_eq!(
            unsafe { de_engine_decide_json(engine, request.as_ptr(), &raw mut response) },
            DeStatus::InvalidArgument
        );
        let error = unsafe { CStr::from_ptr(de_last_error()) }.to_str().unwrap();
        assert!(error.contains("invalid request JSON"));
        unsafe { de_engine_destroy(engine) };
    }
}
