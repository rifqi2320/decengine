#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
version="$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$repo_dir/Cargo.toml" | head -1)"
output="${1:-$repo_dir/dist/decengine-$version-source.zip}"

if ! git -C "$repo_dir" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "source packaging requires a git repository" >&2
    exit 1
fi
if ! git -C "$repo_dir" diff --quiet || ! git -C "$repo_dir" diff --cached --quiet; then
    echo "source packaging requires a clean tracked worktree" >&2
    exit 1
fi

mkdir -p "$(dirname "$output")"
git -C "$repo_dir" archive \
    --format=zip \
    --prefix=decengine/ \
    --output="$output" \
    HEAD
python3 "$repo_dir/scripts/verify-source-archive.py" "$output"
printf '%s\n' "$output"
