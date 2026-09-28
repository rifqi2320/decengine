# Official Laya CPU parity fixture

`cpu-reference.json` captures the official published CPU forward pass for
`test-001` from `benchmarks/cases/probe-test-300.jsonl`, using all three
decisions (`owner`, `urgent`, `impact`). It stores each exact unpadded input
sequence (`input_ids`, `attention_mask`), option-marker positions/mask, raw
option logits, raw action logits and action probabilities, calibrated option
probabilities, and the official SDK-shaped answer. The urgent input is 109
tokens (short); owner and impact are 180 and 167 tokens (>128).

## Pinned model and runtime

* Hub repo/revision: `convaiinnovations/laya`
  `5e7b2b1b8ca2ecdd3f2322d94069c9b6ce7e844b`.
* Root checkpoint SHA-256:
  `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`.
* Model, `rl_agent_api.py`, `rl_common.py`, `rl_agent_config.json`, encoder
  config, and tokenizer are downloaded to
  `/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/laya-root/checkpoint`.
  The per-file hashes and fixture hash are in the JSON artifact.
* Isolated Python 3.12.12 environment at
  `/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/laya-root/venv`;
  resolved versions are pinned in `requirements.txt`. Transformers is exactly
  5.0.0, matching the encoder config's `transformers_version` field.
* Inference imports `RLAgent` directly from the pinned Hub snapshot and invokes
  its published `system_one` method. The installed `laya==0.3.7` package is not
  used for inference. The runner hooks only the official forward pass to capture
  its actual model inputs/outputs. CPU device, four CPU threads, no autocast;
  no MPS or GPU inference was run.

## Reproduce

From the repository root, with the isolated environment and downloaded store
still present:

```sh
export HF_HOME=/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/laya-root/hf-cache
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 LAYA_CPU_THREADS=4
/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/laya-root/venv/bin/python \
  results/laya/reference/run_reference.py --case-id test-001
```

The runner checks the pinned checkpoint digest, captures one case only, and
overwrites `cpu-reference.json`. No 300-case inference or timing benchmark is
part of this reference setup.
