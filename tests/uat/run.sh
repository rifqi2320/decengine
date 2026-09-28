#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fixture_path="$(python3 "$repo_dir/tests/native/build_fixture.py" --output-dir "$repo_dir/build/test-native")"

PYTHONPATH="$repo_dir/python" \
DECENGINE_LIB_PATH="$fixture_path" \
python3 "$repo_dir/tests/uat/developer_acceptance.py"

if command -v cargo >/dev/null 2>&1; then
    port="${DECENGINE_UAT_PORT:-18787}"
    DECENGINE_HOME="$repo_dir/build/uat-home" \
        cargo run --quiet --locked --manifest-path "$repo_dir/Cargo.toml" \
        -p decengine-cli --features fixture-engine -- \
        serve --engine fixture --host 127.0.0.1 --port "$port" \
        >"$repo_dir/build/uat-server.log" 2>&1 &
    server_pid=$!
    cleanup() {
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    }
    trap cleanup EXIT INT TERM
    python3 "$repo_dir/tests/uat/systemone_black_box.py" \
        --base-url "http://127.0.0.1:$port"
    cleanup
    trap - EXIT INT TERM
else
    echo "HTTP UAT SKIP: cargo is unavailable in this runner" >&2
fi
