#![cfg(all(
    feature = "hardware-tests",
    target_os = "macos",
    target_arch = "aarch64"
))]

//! Release-blocking hardware tests live here rather than in the portable suite.
//! They are ignored so a model download is always an explicit operator action.

use std::path::{Path, PathBuf};

use decengine_engine::{DecisionEngine, EngineModelSpec};
use decengine_engine_mlx::MlxDecisionEngine;

const REQUIRED_MODELS: [(&str, &str); 2] = [
    ("Qwen/Qwen3-Embedding-0.6B", "DECENGINE_QWEN_DIR"),
    ("microsoft/harrier-oss-v1-0.6b", "DECENGINE_HARRIER_DIR"),
];

#[test]
#[ignore = "requires local model files and official golden vectors"]
fn both_required_checkpoints_load_and_have_golden_fixtures() {
    let fixture_root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/model-parity/golden");
    for (model_id, directory_variable) in REQUIRED_MODELS {
        let root = required_directory(directory_variable);
        let mut engine = MlxDecisionEngine::new().expect("Apple-Silicon MLX must initialize");
        let handle = engine
            .load_model(&EngineModelSpec {
                id: model_id.to_owned(),
                root: Some(root),
                family: "qwen3-0.6b".to_owned(),
                temperature: 1.0,
                max_length: 32_768,
            })
            .unwrap_or_else(|error| panic!("{model_id} did not load through MLX: {error}"));
        engine
            .unload_model(handle)
            .unwrap_or_else(|error| panic!("{model_id} did not unload cleanly: {error}"));

        let name = model_id.replace('/', "--");
        assert!(
            fixture_root.join(format!("{name}.json")).is_file(),
            "missing golden fixture for {model_id}"
        );
    }
}

fn required_directory(variable: &str) -> PathBuf {
    let value = std::env::var_os(variable)
        .unwrap_or_else(|| panic!("{variable} must point to an installed checkpoint"));
    let path = Path::new(&value);
    assert!(path.is_dir(), "{variable} is not a directory");
    path.to_owned()
}
