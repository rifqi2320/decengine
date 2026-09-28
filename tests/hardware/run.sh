#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "hardware suite requires macOS on Apple Silicon" >&2
    exit 2
fi
: "${DECENGINE_QWEN_DIR:?set DECENGINE_QWEN_DIR to an installed Qwen checkpoint}"
: "${DECENGINE_HARRIER_DIR:?set DECENGINE_HARRIER_DIR to an installed Harrier checkpoint}"

for model_dir in "$DECENGINE_QWEN_DIR" "$DECENGINE_HARRIER_DIR"; do
    test -f "$model_dir/config.json"
    test -f "$model_dir/tokenizer.json"
done

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECENGINE_DISABLE_NETWORK=1

cargo test --manifest-path "$repo_dir/Cargo.toml" \
    --locked -p decengine-engine-mlx --features hardware-tests --test model_parity \
    --release -- --ignored --nocapture

echo "hardware acceptance passed for both required checkpoints"
