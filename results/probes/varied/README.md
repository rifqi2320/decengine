# Varied-question MLX feature exporter

Build on Apple Silicon with `cargo build -p decengine-cli --release --features mlx`.
Export with:

```sh
target/release/decengine --home "$MODEL_STORE" varied-export-features \
  --input cases.jsonl --output features.jsonl --model Qwen/Qwen3-Embedding-0.6B
```

Run separately for Qwen and Harrier; the model profile is resolved from the installed
model record. This is an offline host-vector export using real MLX/Metal embedding
inference. It is not a Metal-resident classifier.

## Deterministic text and row format

For each record, only `request.state` and the named `request.questions` items are used;
the exporter emits one JSONL row per `(id, question_key)` with a query and candidates.
State text is compact `serde_json` serialization of the parsed JSON value (object keys
are sorted by serde_json's map representation). The query is compiled exactly as:

1. `kind_instruction` is generic per wire type (select a candidate from criteria, assess
   the yes/no proposition, or apply the supplied score rubric); it does not encode an
   owner, urgency, impact, option label, or question name.
2. `instruction = profile.query_instruction_template` with `{prompt}` then `{kind}`
   replaced by the question prompt and instruction.
3. `raw = "Decision: {prompt}\nState: {state_json}"`.
4. `query = profile.query_template` with `{instruction}` then `{text}` replaced.

Choice candidate text applies `profile.choice_candidate_template` (`{label}` and
`{criterion}` replaced) then `profile.candidate_template` (`{text}` replaced), in
serde_json's deterministic object-key order. Score uses the corresponding
`score_candidate_template`, preserving the levels array order. Noul explicitly
constructs two outcomes from the complete question proposition: “not supported by the
reported state” and “supported by the reported state”; each is then wrapped by the
profile candidate template. The two candidate alternatives therefore vary with the
actual prompt instead of using a fixed urgency word bag.

Each output row contains the compiled query text, exact tokenizer token count, UTF-8
byte count and 1024d L2 embedding. Candidate rows additionally contain ID, type,
zero-based order and numeric ordinal, label, score value when applicable, candidate
text, token/byte lengths, and the embedding. The exporter checks all encoded strings
against the profile max token length, verifies dimensions, finiteness, and unit L2 norm.
It verifies all installed checkpoint file sizes and SHA-256 hashes and records the
checkpoint and tokenizer hashes plus profile version/hash in the sidecar manifest.

Only the prompt/request fields are dereferenced for feature creation; reference target,
rationale, and labels are not copied into query/candidate text. Candidate labels present
in the public request are naturally included by the same compiler-like candidate
templates, as with ordinary inference. The JSON vectors are transferred to host memory
for offline training.

For all-MLX/Metal inference of the frozen generic typed scorer, including GPU-resident
pair-feature construction and scoring with no host embedding transfer, see
[`metal/README.md`](metal/README.md). `varied-export-features` remains an offline training
feature exporter and is not that inference path.

## Multi-question shared-scorer run

The exporter accepts multi-question records with `request.questions` as a map of names
to ordinary choice/noul/score question wires. `multi_train_pairwise.py` combines
clean300 single-question training families with multi train120 (360 questions), does
train-only grouped CV with family and case components kept intact, and writes raw test
predictions before reading test targets. For example:

```sh
python3 results/probes/varied/multi_train_pairwise.py \
  --train clean300-train.jsonl --train-embeddings clean300-train.features.jsonl \
  --multi-train multi-train120.jsonl --multi-train-embeddings multi-train120.features.jsonl \
  --test multi-test60.jsonl --test-embeddings multi-test60.features.jsonl \
  --single-test single-test200.jsonl --single-test-embeddings single-test200.features.jsonl \
  --model Qwen/Qwen3-Embedding-0.6B --out runs/multi-qwen
```

The runner enforces 300/120/60 cases and 360/180 multi questions, disjoint IDs and
families, reports per-type accuracy and the number of test cases correct on all three
questions. Existing single-question test200 is optionally exported and scored in a
separate `single-test200/` output directory, never pooled with multi test60. Embeddings
are offline host vectors; the exporter itself runs real MLX/Metal inference. Exporter
manifests must agree on model/profile/text versions across splits.
