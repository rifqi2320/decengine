# Laya comparison runner

`compare_300.py` runs the public `convaiinnovations/laya` **English root
checkpoint** directly through the model's supported `laya.load(...).predict(...)`
Python SDK. The Hugging Face card describes a Transformers/PyTorch
non-autoregressive typed-decision model; it does not advertise an MLX checkpoint
or `laya-mlx` runtime. The optional local `/Users/IDSP35241/Documents/Code/laya-mlx`
checkout was not present when this runner was prepared. No decengine/JEV backend
or trained probe artifact is used.

## Install and run

In an environment with PyTorch, Transformers and the public Laya SDK installed
(the runner itself does not install packages; the model card's SDK install is
`python3 -m pip install 'laya>=0.3.7'`):

```sh
python3 results/laya/compare_300.py --dry-run
python3 results/laya/compare_300.py --device mps
```

Omit `--device` to let Laya choose its normal device. Use `--revision` to select
a Hub branch, tag, or commit; the resolved snapshot commit and SHA-256 of
`model.safetensors` are recorded. By default outputs go to a fresh timestamped
`results/laya/runs/<UTC timestamp>/`; `--out` selects another new directory.
The model is public; this runner does not read or use a JEV/API key.

## Comparable output and limitations

The runner validates the same ordered 300-case fixture as
`results/probes/compare_300.py` and writes `laya-predictions.jsonl` with
`id`, `owner`, `urgent`, and `impact` scalar labels, plus `metrics.json` using
the same pooled300/per-question/complete-case accuracy and Wilson interval
shape. It also emits:

* `raw-results.jsonl`: raw SDK responses, normalized labels, errors, option
  marker counts, exact per-question input-token counts and per-decision latency;
* `timings.jsonl`: concise per-case and per-decision times/token lengths;
* `run-metadata.json`: fixture and weight hashes, resolved revision, SDK/runtime
  versions, actual device/dtype, config limits, and process peak-RSS data;
* `checksums.json`: hashes for the run outputs.

Fixture `prompt`/`options`/`levels` are explicitly translated to Laya's
`instructions`/`criteria` schema. Urgency is the `noul` probability thresholded
at 0.5. Impact is the most probable ordinal rubric label (argmax of Laya's
probabilities), not its expected score. The full level label and criterion are
passed through to Laya. This mapping makes the scalar output evaluable against
the fixture labels, but it is not a claim that Laya's ordinal score primitive
is identical to an external scalar classifier.

Laya supports answering questions together in one forward pass. To expose
actual owner/urgent/impact latency separately, this runner intentionally makes
three single-question calls per case and reports their sum as
`case_inference_ms_sum_of_three_calls`; these are not batched-call latency or
model FLOPs. Token lengths are recorded per input and can anchor FLOPs estimates,
but FLOPs are not estimated by this runner. Calls include SDK/model inference
but exclude Python startup and checkpoint loading. Raw per-decision SDK errors
are retained and corresponding scalar predictions are `null`.

No benchmark is run during setup; only fixture validation (`--dry-run`) is
needed to check readiness. To start only once the orchestrator confirms the
other 300-case run is no longer active, use the command above.
