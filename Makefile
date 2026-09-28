.PHONY: test test-rust test-python uat lint format-check clean source-zip

test: test-rust test-python

test-rust:
	cargo test --locked --workspace --all-targets --features fixture-engine

test-python:
	PYTHONPATH=python python3 -m unittest discover -s tests/python -v

uat:
	bash tests/uat/run.sh

lint:
	cargo fmt --all -- --check
	cargo clippy --locked --workspace --all-targets --features fixture-engine -- -D warnings
	PYTHONPATH=python python3 -m ruff check python tests
	PYTHONPATH=python python3 -m mypy python/decengine

format-check:
	cargo fmt --all -- --check

clean:
	cargo clean
	rm -rf build dist .coverage .pytest_cache .mypy_cache .ruff_cache
	find python tests -type d -name __pycache__ -prune -exec rm -rf {} +

source-zip:
	bash scripts/source-zip.sh
