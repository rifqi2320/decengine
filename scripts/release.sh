#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "release builds require macOS on Apple Silicon" >&2
    exit 1
fi
if grep -q '^## Red release gate' "$repo_dir/docs/IMPLEMENTATION_STATUS.md"; then
    echo "release blocked: unresolved MLX implementation gate" >&2
    exit 1
fi
if [[ -z "${DECENGINE_QWEN_DIR:-}" || -z "${DECENGINE_HARRIER_DIR:-}" ]]; then
    echo "release requires DECENGINE_QWEN_DIR and DECENGINE_HARRIER_DIR" >&2
    exit 1
fi

python3 "$repo_dir/scripts/check-boundaries.py"
cargo test --locked --manifest-path "$repo_dir/Cargo.toml" --workspace --all-targets
bash "$repo_dir/tests/hardware/run.sh"

cargo build --manifest-path "$repo_dir/Cargo.toml" \
    --locked --release -p decengine-cli --no-default-features --features mlx
cargo build --manifest-path "$repo_dir/Cargo.toml" \
    --locked --release -p decengine-ffi --no-default-features --features mlx

release_root="$repo_dir/dist/decengine-darwin-arm64"
rm -rf "$release_root"
mkdir -p "$release_root/bin" "$release_root/lib" "$release_root/include" \
    "$release_root/python" "$release_root/LICENSES"
cp "$repo_dir/target/release/decengine" "$release_root/bin/"
cp "$repo_dir/target/release/libdecengine.dylib" "$release_root/lib/"
cp "$repo_dir/include/decengine.h" "$release_root/include/"
cp -R "$repo_dir/python/decengine" "$release_root/python/"
cp "$repo_dir/LICENSE" "$release_root/LICENSES/decengine-MIT.txt"

tar -C "$repo_dir/dist" -czf "$repo_dir/dist/decengine-darwin-arm64.tar.gz" \
    decengine-darwin-arm64
shasum -a 256 "$repo_dir/dist/decengine-darwin-arm64.tar.gz" \
    >"$repo_dir/dist/decengine-darwin-arm64.tar.gz.sha256"
