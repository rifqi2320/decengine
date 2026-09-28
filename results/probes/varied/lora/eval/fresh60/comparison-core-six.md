# Frozen model comparison: LoRA fresh-60

Fixture SHA-256: `2848d28c79120c2fcf3ab49b755c402386f60ff96d0ac69dce801a606df44629`

| Model | Choice | Noul | Score | All three | Score expected-value MAE | Score ordinal MAE |
|---|---:|---:|---:|---:|---:|---:|
| qwen_lora | 15/60 (25.0%) | 30/60 (50.0%) | 20/60 (33.3%) | 0/60 (0.0%) | 30.132 | 0.933 |
| harrier_lora | 15/60 (25.0%) | 30/60 (50.0%) | 20/60 (33.3%) | 0/60 (0.0%) | 30.161 | 0.933 |
| qwen_frozen_baseline_full660 | 20/60 (33.3%) | 37/60 (61.7%) | 10/60 (16.7%) | 2/60 (3.3%) | 52.076 | 1.583 |
| harrier_frozen_baseline_full660 | 16/60 (26.7%) | 37/60 (61.7%) | 16/60 (26.7%) | 2/60 (3.3%) | 38.113 | 1.267 |
| qwen_headonly_same460 | 21/60 (35.0%) | 37/60 (61.7%) | 6/60 (10.0%) | 2/60 (3.3%) | 44.825 | 1.500 |
| harrier_headonly_same460 | 17/60 (28.3%) | 35/60 (58.3%) | 20/60 (33.3%) | 3/60 (5.0%) | 35.873 | 1.133 |

Latency comparisons are only meaningful with the recorded timing scopes below; do not conflate model-only forward time, per-question/case time, or end-to-end startup/load time.

```json
{
  "timings": {
    "qwen_lora": {
      "question_inference_ms": {
        "scope": "sequential individual text backbone forwards + typed scorer; tokenizer/compilation and load excluded",
        "n": 180,
        "p50": 126.77335400000001,
        "p95": 164.87324199999998
      },
      "three_question_case_sum_ms": {
        "scope": "sum of three question inference scopes; tokenizer/compilation and model load excluded",
        "n": 60,
        "p50": 397.7869375,
        "p95": 439.153886
      }
    },
    "harrier_lora": {
      "question_inference_ms": {
        "scope": "sequential individual text backbone forwards + typed scorer; tokenizer/compilation and load excluded",
        "n": 180,
        "p50": 126.2561255,
        "p95": 164.3975771
      },
      "three_question_case_sum_ms": {
        "scope": "sum of three question inference scopes; tokenizer/compilation and model load excluded",
        "n": 60,
        "p50": 397.29279199999996,
        "p95": 436.4980312
      }
    },
    "qwen_frozen_baseline_full660": {
      "case_total_ms": {
        "n": 60,
        "p50": 320.93535399999996,
        "p95": 344.7769281,
        "scope": "sum of 3 question calls/case; excludes model load; warmup excluded"
      },
      "question_total_ms": {
        "n": 180,
        "p50": 101.9993335,
        "p95": 127.45909585,
        "scope": "tokenization + query/candidate embedding + Metal scorer; excludes model load; warmup excluded"
      }
    },
    "harrier_frozen_baseline_full660": {
      "case_total_ms": {
        "n": 60,
        "p50": 320.5260425,
        "p95": 344.91560695,
        "scope": "sum of 3 question calls/case; excludes model load; warmup excluded"
      },
      "question_total_ms": {
        "n": 180,
        "p50": 102.05947950000001,
        "p95": 126.91669404999999,
        "scope": "tokenization + query/candidate embedding + Metal scorer; excludes model load; warmup excluded"
      }
    },
    "qwen_headonly_same460": {
      "case_total_ms": {
        "n": 60,
        "p50": 322.0725,
        "p95": 348.1571077,
        "scope": "three-question per-case sum; excludes model load/warmup"
      },
      "question_total_ms": {
        "n": 180,
        "p50": 103.05431250000001,
        "p95": 128.70307499999998,
        "scope": "tokenization+query/candidate embedding+Metal scorer; excludes model load/warmup"
      }
    },
    "harrier_headonly_same460": {
      "case_total_ms": {
        "n": 60,
        "p50": 322.578001,
        "p95": 347.6380812499999,
        "scope": "three-question per-case sum; excludes model load/warmup"
      },
      "question_total_ms": {
        "n": 180,
        "p50": 103.010125,
        "p95": 128.83049194999998,
        "scope": "tokenization+query/candidate embedding+Metal scorer; excludes model load/warmup"
      }
    }
  },
  "provenance": {
    "qwen_lora": {
      "run_dir": "/Users/IDSP35241/Documents/Code/decengine/results/probes/varied/lora/eval/fresh60/predictions/qwen-lora",
      "predictions_sha256": "c47fb09e32e0ce9bbac2cda44cc016d587adbef8aa0e024ad9791f80397be2cc",
      "metadata_sha256": "ff415654d64c3a2e75de6f4b4ab9e361810db0aaa07557868b1cada335764cbf",
      "runtime": "MLX/Metal",
      "device": "Device(gpu, 0)",
      "dtype": null,
      "compute_assumptions": {
        "forward_policy": "one query/candidate text forward individually; no batching",
        "candidate_count": "fixture supplied 2-5 per question",
        "flops": "not estimated; no hardware-normalized compute claim"
      },
      "resource_guard": {
        "monitor": "external footprint + vm.swapusage sampled during process",
        "baseline_swap_bytes": 1087635456,
        "max_swap_bytes": 1087635456,
        "swap_never_above_baseline": true,
        "swap_samples": 69,
        "max_process_peak_footprint_bytes": 5712676712,
        "early_stop_bytes": 6442450944,
        "absolute_cap_bytes": 8589934592,
        "peak_below_early_stop": true,
        "samples": 69
      },
      "baseline_protocol": null,
      "training_question_count": null,
      "inference_policy_id": "lora-fresh60-inference-extrapolation-512-v1",
      "inference_policy_sha256": "d1fb0c2e142c066e135705cdd7d77a313d65e5eb730d2b4ed6459f7ee044a32c",
      "unseen_length_extrapolation": {
        "authorized": true,
        "inputs_over_training_limit": 180,
        "maximum_compiled_token_count": 375,
        "truncation": false
      }
    },
    "harrier_lora": {
      "run_dir": "/Users/IDSP35241/Documents/Code/decengine/results/probes/varied/lora/eval/fresh60/predictions/harrier-lora",
      "predictions_sha256": "e7ce1c8c7912ad896d38cae6cb7d7a0ca6ad29318d7d2d52b37a5746d01bc3b9",
      "metadata_sha256": "063d5acc8a350f1f5d4d74551b4f655b4c6e6d1f55c109dbed41cf20a3a5ed2d",
      "runtime": "MLX/Metal",
      "device": "Device(gpu, 0)",
      "dtype": null,
      "compute_assumptions": {
        "forward_policy": "one query/candidate text forward individually; no batching",
        "candidate_count": "fixture supplied 2-5 per question",
        "flops": "not estimated; no hardware-normalized compute claim"
      },
      "resource_guard": {
        "monitor": "external footprint + vm.swapusage sampled during process",
        "baseline_swap_bytes": 1087635456,
        "max_swap_bytes": 1087635456,
        "swap_never_above_baseline": true,
        "swap_samples": 69,
        "max_process_peak_footprint_bytes": 5715199848,
        "early_stop_bytes": 6442450944,
        "absolute_cap_bytes": 8589934592,
        "peak_below_early_stop": true,
        "samples": 69
      },
      "baseline_protocol": null,
      "training_question_count": null,
      "inference_policy_id": "lora-fresh60-inference-extrapolation-512-v1",
      "inference_policy_sha256": "d1fb0c2e142c066e135705cdd7d77a313d65e5eb730d2b4ed6459f7ee044a32c",
      "unseen_length_extrapolation": {
        "authorized": true,
        "inputs_over_training_limit": 180,
        "maximum_compiled_token_count": 375,
        "truncation": false
      }
    },
    "qwen_frozen_baseline_full660": {
      "run_dir": "/Users/IDSP35241/Documents/Code/decengine/results/probes/varied/lora/eval/fresh60/predictions/qwen-frozen-baseline-full660",
      "predictions_sha256": "1baeb0cd61470529e80d33473a7908bd4a71e1eae6269f97632b09221aef1e12",
      "metadata_sha256": "8e371f592439136b008f0a832e1170243b737af7da1e4e9b047f48a6a32e29ad",
      "runtime": "decengine varied-metal (MLX/Metal)",
      "device": "Apple M4 Pro (freeze record)",
      "dtype": "profile-native MLX inference",
      "compute_assumptions": {
        "batching": "varied-metal native protocol",
        "execution": "query/candidate embeddings and pair scorer on MLX/Metal",
        "flops": "not estimated"
      },
      "resource_guard": {
        "baseline_swap_bytes": 1112801280,
        "guard_samples": 21,
        "limit_absolute_bytes": 8589934592,
        "limit_early_stop_bytes": 6442450944,
        "max_process_peak_footprint_bytes": 3991783248,
        "max_swap_bytes": 1112801280,
        "process_peak_below_early_stop": true,
        "swap_unchanged": true
      },
      "baseline_protocol": "historical frozen typed scorer fit/selected on full 660 training questions; not same split as 460-question LoRA train subset",
      "training_question_count": null,
      "inference_policy_id": null,
      "inference_policy_sha256": null,
      "unseen_length_extrapolation": null
    },
    "harrier_frozen_baseline_full660": {
      "run_dir": "/Users/IDSP35241/Documents/Code/decengine/results/probes/varied/lora/eval/fresh60/predictions/harrier-frozen-baseline-full660",
      "predictions_sha256": "b86bc07f2bd3ff1f5fc119a1e8cc59e630398a01dc164a790c3a8a396f1d2754",
      "metadata_sha256": "f54901bdff5158aa1fe05c23a5a0bbf65b52e0ecb7c297081e66d43e1c86a2fe",
      "runtime": "decengine varied-metal (MLX/Metal)",
      "device": "Apple M4 Pro (freeze record)",
      "dtype": "profile-native MLX inference",
      "compute_assumptions": {
        "batching": "varied-metal native protocol",
        "execution": "query/candidate embeddings and pair scorer on MLX/Metal",
        "flops": "not estimated"
      },
      "resource_guard": {
        "baseline_swap_bytes": 1112801280,
        "guard_samples": 22,
        "limit_absolute_bytes": 8589934592,
        "limit_early_stop_bytes": 6442450944,
        "max_process_peak_footprint_bytes": 3986884408,
        "max_swap_bytes": 1112801280,
        "process_peak_below_early_stop": true,
        "swap_unchanged": true
      },
      "baseline_protocol": "historical frozen typed scorer fit/selected on full 660 training questions; not same split as 460-question LoRA train subset",
      "training_question_count": null,
      "inference_policy_id": null,
      "inference_policy_sha256": null,
      "unseen_length_extrapolation": null
    },
    "qwen_headonly_same460": {
      "run_dir": "/Users/IDSP35241/Documents/Code/decengine/results/probes/varied/lora/eval/fresh60/predictions/qwen-headonly-same460",
      "predictions_sha256": "a704c39005e5fe0cbec4f78331d0238a625e27fd436fd6b2af70c7692fb46ae8",
      "metadata_sha256": "d4931dbacba19860466ad39d1adf8a5ff983996d495585b98086822ec10d5681",
      "runtime": "decengine varied-metal (MLX/Metal)",
      "device": "Apple M4 Pro",
      "dtype": "profile-native MLX inference",
      "compute_assumptions": "not provided",
      "resource_guard": {
        "absolute_cap_bytes": 8589934592,
        "baseline_swap_bytes": 1112801280,
        "early_stop_bytes": 6442450944,
        "max_process_peak_footprint_bytes": 3985753912,
        "max_swap_bytes": 1112801280,
        "sample_count": 22,
        "swap_unchanged": true
      },
      "baseline_protocol": "new controlled head-only C=1.0 fit on exactly frozen 460 train indices; no dev or holdout labels used in fit",
      "training_question_count": 460,
      "inference_policy_id": null,
      "inference_policy_sha256": null,
      "unseen_length_extrapolation": null
    },
    "harrier_headonly_same460": {
      "run_dir": "/Users/IDSP35241/Documents/Code/decengine/results/probes/varied/lora/eval/fresh60/predictions/harrier-headonly-same460",
      "predictions_sha256": "798491a9741934fe5c35d04ba22fdf7400e5c72f36ec4672ed452ad9ee9463c9",
      "metadata_sha256": "b5c815264efa52f1b3c871ec04614bd7c1adea7f31b2de51ec802f875b4fac4e",
      "runtime": "decengine varied-metal (MLX/Metal)",
      "device": "Apple M4 Pro",
      "dtype": "profile-native MLX inference",
      "compute_assumptions": "not provided",
      "resource_guard": {
        "absolute_cap_bytes": 8589934592,
        "baseline_swap_bytes": 1112801280,
        "early_stop_bytes": 6442450944,
        "max_process_peak_footprint_bytes": 3986474808,
        "max_swap_bytes": 1112801280,
        "sample_count": 22,
        "swap_unchanged": true
      },
      "baseline_protocol": "new controlled head-only C=1.0 fit on exactly frozen 460 train indices; no dev or holdout labels used in fit",
      "training_question_count": 460,
      "inference_policy_id": null,
      "inference_policy_sha256": null,
      "unseen_length_extrapolation": null
    }
  }
}
```

Compute assumptions are copied from each frozen run's metadata. Missing assumptions/timing scopes are explicitly reported as not provided; no cross-device FLOPs or latency normalization is inferred.
