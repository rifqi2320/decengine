# Corrected 9-epoch matched-head comparison on reused fresh60

Fixture SHA-256: `2848d28c79120c2fcf3ab49b755c402386f60ff96d0ac69dce801a606df44629`

> **Exploratory / seen holdout:** this exact 60-case fixture was already scored previously. No tuning or selection used these outputs.

| System | Choice | Noul | Score | All three | Score expected-value MAE | Score selected-level value MAE | Score ordinal MAE | Case p50/p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen_head_epoch2_matched | 19/60 (31.7%) | 30/60 (50.0%) | 20/60 (33.3%) | 0/60 (0.0%) | 30.163 | 32.267 | 0.933 | 387.8 / 426.8 |
| qwen_lora_epoch9_matched | 15/60 (25.0%) | 35/60 (58.3%) | 21/60 (35.0%) | 3/60 (5.0%) | 30.610 | 31.433 | 0.900 | 391.0 / 428.4 |
| harrier_head_epoch9_matched | 26/60 (43.3%) | 30/60 (50.0%) | 21/60 (35.0%) | 2/60 (3.3%) | 30.472 | 38.367 | 1.167 | 388.4 / 426.3 |
| harrier_lora_epoch9_matched | 15/60 (25.0%) | 36/60 (60.0%) | 16/60 (26.7%) | 2/60 (3.3%) | 30.626 | 42.633 | 1.300 | 390.5 / 427.8 |
| qwen_frozen_baseline_full660 | 20/60 (33.3%) | 37/60 (61.7%) | 10/60 (16.7%) | 2/60 (3.3%) | 52.076 | 54.600 | 1.583 | 320.9 / 344.8 |
| harrier_frozen_baseline_full660 | 16/60 (26.7%) | 37/60 (61.7%) | 16/60 (26.7%) | 2/60 (3.3%) | 38.113 | 41.267 | 1.267 | 320.5 / 344.9 |
| qwen_same460_linear | 21/60 (35.0%) | 37/60 (61.7%) | 6/60 (10.0%) | 2/60 (3.3%) | 44.825 | 51.550 | 1.500 | 322.1 / 348.2 |
| harrier_same460_linear | 17/60 (28.3%) | 35/60 (58.3%) | 20/60 (33.3%) | 3/60 (5.0%) | 35.873 | 36.567 | 1.133 | 322.6 / 347.6 |
| laya_archived | 18/60 (30.0%) | 33/60 (55.0%) | 19/60 (31.7%) | 1/60 (1.7%) | n/a | 35.600 | 1.067 | 494.8 / 520.7 |
| jev_archived | 26/60 (43.3%) | 56/60 (93.3%) | 14/60 (23.3%) | 6/60 (10.0%) | n/a | 35.367 | 1.033 | 1064.4 / 2259.0 |

## Provenance and diagnostics

```json
{
  "provenance": {
    "qwen_head_epoch2_matched": {
      "run_dir": "/Users/IDSP35241/Documents/Code/decengine/results/probes/varied/lora/experiment-matched-head/eval/fresh60/qwen-head-epoch02-total10g",
      "prediction_path": "/Users/IDSP35241/Documents/Code/decengine/results/probes/varied/lora/experiment-matched-head/eval/fresh60/qwen-head-epoch02-total10g/predictions.jsonl",
      "predictions_sha256": "e11c257a90494e1b369ef9fab90510887efe460cf32ca86f41ffde6d5df069cf",
      "metadata_sha256": "3d3df897d95be2aa838f21d27d9809da3419d05e92f20ebc33cbd812fee82a6f",
      "runtime": "MLX/Metal",
      "policy_sha256": "1a9414f4647570f1f06de9e9e6ee8710caa0e9a4aff63fbbcfc14c2042b7ab11",
      "model_or_adapter_sha256": "58bef9ac575faf62344d7e176e5c80910fff7370dd8314295cc2d9a23d0437b8",
      "resource_summary": {
        "baseline_swap_bytes": 1866926653,
        "max_positive_swap_growth_bytes": 0,
        "max_process_peak_bytes": 2376025864,
        "max_process_peak_plus_positive_swap_growth_bytes": 2376025864,
        "process_early_stop_bytes": 6442450944,
        "absolute_process_cap_bytes": 8589934592,
        "total_process_peak_plus_swap_cap_bytes": 10737418240,
        "external_monitor_samples": 72,
        "external_max_process_peak_bytes": 2376025864,
        "external_max_swap_growth_bytes": 0
      },
      "baseline_protocol": null,
      "fixture_reused_previously_scored": true
    },
    "qwen_lora_epoch9_matched": {
      "run_dir": "/Users/IDSP35241/Documents/Code/decengine/results/probes/varied/lora/experiment-matched-head/eval/fresh60/qwen-lora-epoch09-total10g",
      "prediction_path": "/Users/IDSP35241/Documents/Code/decengine/results/probes/varied/lora/experiment-matched-head/eval/fresh60/qwen-lora-epoch09-total10g/predictions.jsonl",
      "predictions_sha256": "44bdad3bac19486e060fb21c67fc6f833bdde81a7f52621a350d9374639b8365",
      "metadata_sha256": "1da9dc969f8c223de6f22fc31bb33fc261a39bd5885e10e02268d9cb3c69c9e6",
      "runtime": "MLX/Metal",
      "policy_sha256": "a641c64ac7f8fda46d587422f0fe20b94fcd1353246eb9f0c3cd1051313177f7",
      "model_or_adapter_sha256": "cc6ceb9cd13819cd162ec494630660c159aa27e2e37cf5b82391b36ef5f6bf06",
      "resource_summary": {
        "baseline_swap_bytes": 1858538045,
        "max_positive_swap_growth_bytes": 447280579,
        "max_process_peak_bytes": 2405861176,
        "max_process_peak_plus_positive_swap_growth_bytes": 2853141755,
        "process_early_stop_bytes": 6442450944,
        "absolute_process_cap_bytes": 8589934592,
        "total_process_peak_plus_swap_cap_bytes": 10737418240,
        "external_monitor_samples": 73,
        "external_max_process_peak_bytes": 2405861176,
        "external_max_positive_swap_growth_bytes": 447280579
      },
      "baseline_protocol": null,
      "fixture_reused_previously_scored": true
    },
    "harrier_head_epoch9_matched": {
      "run_dir": "/Users/IDSP35241/Documents/Code/decengine/results/probes/varied/lora/experiment-matched-head/eval/fresh60/harrier-head-epoch09-total10g",
      "prediction_path": "/Users/IDSP35241/Documents/Code/decengine/results/probes/varied/lora/experiment-matched-head/eval/fresh60/harrier-head-epoch09-total10g/predictions.jsonl",
      "predictions_sha256": "253bfd899c352060eb1f49bd75f445cb3f7ef7c8687de5f4fb7bf667e8f9e157",
      "metadata_sha256": "6fef4c7f0f5955f292d292d8a01d1910ca44e38f6f3c802c3da7f578710a4da4",
      "runtime": "MLX/Metal",
      "policy_sha256": "57455ae8d88322ea7703d4d67b1ee317ad198b57071b35ebc57be29a01f1a590",
      "model_or_adapter_sha256": "92145ec36c4eeec1d89e70818313f57a758bd696b121ddcd20d8cd1153b59c28",
      "resource_summary": {
        "baseline_swap_bytes": 2289041408,
        "max_process_peak_bytes": 2377189152,
        "max_positive_swap_growth_observed_bytes": 0,
        "max_process_peak_plus_swap_growth_bytes": 2377189152,
        "early_stop_bytes": 6442450944,
        "absolute_stop_bytes": 8589934592,
        "standalone_swap_growth_stop_bytes": null,
        "combined_stop_bytes": 10737418240,
        "swap_zero_growth_required": false
      },
      "baseline_protocol": null,
      "fixture_reused_previously_scored": true
    },
    "harrier_lora_epoch9_matched": {
      "run_dir": "/Users/IDSP35241/Documents/Code/decengine/results/probes/varied/lora/experiment-matched-head/eval/fresh60/harrier-lora-epoch09-total10g",
      "prediction_path": "/Users/IDSP35241/Documents/Code/decengine/results/probes/varied/lora/experiment-matched-head/eval/fresh60/harrier-lora-epoch09-total10g/predictions.jsonl",
      "predictions_sha256": "cdf4b89346facca60a664103177e9aa7867e081e74df11f529adabc6facfdd3c",
      "metadata_sha256": "433d93cb3bf2ec40a84725146dde40aaa139f4311c4bc6eb4c181ca5a3bbd4dc",
      "runtime": "MLX/Metal",
      "policy_sha256": "57455ae8d88322ea7703d4d67b1ee317ad198b57071b35ebc57be29a01f1a590",
      "model_or_adapter_sha256": "06f1f13eb90fb84ce8bfac8a444fad7af2312e53c2c23745342256aa567bc10f",
      "resource_summary": {
        "baseline_swap_bytes": 2289041408,
        "max_process_peak_bytes": 2385774344,
        "max_positive_swap_growth_observed_bytes": 0,
        "max_process_peak_plus_swap_growth_bytes": 2385774344,
        "early_stop_bytes": 6442450944,
        "absolute_stop_bytes": 8589934592,
        "standalone_swap_growth_stop_bytes": null,
        "combined_stop_bytes": 10737418240,
        "swap_zero_growth_required": false
      },
      "baseline_protocol": null,
      "fixture_reused_previously_scored": true
    },
    "qwen_frozen_baseline_full660": {
      "run_dir": "/Users/IDSP35241/Documents/Code/decengine/results/probes/varied/lora/eval/fresh60/predictions/qwen-frozen-baseline-full660",
      "prediction_path": "/Users/IDSP35241/Documents/Code/decengine/results/probes/varied/lora/eval/fresh60/predictions/qwen-frozen-baseline-full660/predictions.jsonl",
      "predictions_sha256": "1baeb0cd61470529e80d33473a7908bd4a71e1eae6269f97632b09221aef1e12",
      "metadata_sha256": "8e371f592439136b008f0a832e1170243b737af7da1e4e9b047f48a6a32e29ad",
      "runtime": "decengine varied-metal (MLX/Metal)",
      "policy_sha256": null,
      "model_or_adapter_sha256": null,
      "resource_summary": null,
      "baseline_protocol": "independent historical frozen scorer trained/selected on full 660 questions",
      "fixture_reused_previously_scored": true
    },
    "harrier_frozen_baseline_full660": {
      "run_dir": "/Users/IDSP35241/Documents/Code/decengine/results/probes/varied/lora/eval/fresh60/predictions/harrier-frozen-baseline-full660",
      "prediction_path": "/Users/IDSP35241/Documents/Code/decengine/results/probes/varied/lora/eval/fresh60/predictions/harrier-frozen-baseline-full660/predictions.jsonl",
      "predictions_sha256": "b86bc07f2bd3ff1f5fc119a1e8cc59e630398a01dc164a790c3a8a396f1d2754",
      "metadata_sha256": "f54901bdff5158aa1fe05c23a5a0bbf65b52e0ecb7c297081e66d43e1c86a2fe",
      "runtime": "decengine varied-metal (MLX/Metal)",
      "policy_sha256": null,
      "model_or_adapter_sha256": null,
      "resource_summary": null,
      "baseline_protocol": "independent historical frozen scorer trained/selected on full 660 questions",
      "fixture_reused_previously_scored": true
    },
    "qwen_same460_linear": {
      "run_dir": "/Users/IDSP35241/Documents/Code/decengine/results/probes/varied/lora/eval/fresh60/predictions/qwen-headonly-same460",
      "prediction_path": "/Users/IDSP35241/Documents/Code/decengine/results/probes/varied/lora/eval/fresh60/predictions/qwen-headonly-same460/predictions.jsonl",
      "predictions_sha256": "a704c39005e5fe0cbec4f78331d0238a625e27fd436fd6b2af70c7692fb46ae8",
      "metadata_sha256": "d4931dbacba19860466ad39d1adf8a5ff983996d495585b98086822ec10d5681",
      "runtime": "decengine varied-metal (MLX/Metal)",
      "policy_sha256": null,
      "model_or_adapter_sha256": null,
      "resource_summary": null,
      "baseline_protocol": "separate C=1 head-only linear comparator fitted on exact 460 training questions",
      "fixture_reused_previously_scored": true
    },
    "harrier_same460_linear": {
      "run_dir": "/Users/IDSP35241/Documents/Code/decengine/results/probes/varied/lora/eval/fresh60/predictions/harrier-headonly-same460",
      "prediction_path": "/Users/IDSP35241/Documents/Code/decengine/results/probes/varied/lora/eval/fresh60/predictions/harrier-headonly-same460/predictions.jsonl",
      "predictions_sha256": "798491a9741934fe5c35d04ba22fdf7400e5c72f36ec4672ed452ad9ee9463c9",
      "metadata_sha256": "b5c815264efa52f1b3c871ec04614bd7c1adea7f31b2de51ec802f875b4fac4e",
      "runtime": "decengine varied-metal (MLX/Metal)",
      "policy_sha256": null,
      "model_or_adapter_sha256": null,
      "resource_summary": null,
      "baseline_protocol": "separate C=1 head-only linear comparator fitted on exact 460 training questions",
      "fixture_reused_previously_scored": true
    },
    "laya_archived": {
      "run_dir": "/Users/IDSP35241/Documents/Code/decengine/results/laya/varied/runs/20260924T054309737898Z-cacheguard",
      "prediction_path": "/Users/IDSP35241/Documents/Code/decengine/results/laya/varied/runs/20260924T054309737898Z-cacheguard/laya-predictions.jsonl",
      "predictions_sha256": "0273e0c8b9a453de8688f0542f1382820e984613dcaafe09aa1b8c348bd78181",
      "metadata_sha256": "d49530fcbb79b08a7d867a86a57e05677641d3898a9a3710031bc086d4cc333c",
      "runtime": "mlx-metal",
      "policy_sha256": null,
      "model_or_adapter_sha256": null,
      "resource_summary": null,
      "baseline_protocol": null,
      "fixture_reused_previously_scored": true
    },
    "jev_archived": {
      "run_dir": "/Users/IDSP35241/Documents/Code/decengine/results/jev/varied/runs/20260924T042126036050Z",
      "prediction_path": "/Users/IDSP35241/Documents/Code/decengine/results/jev/varied/runs/20260924T042126036050Z/jev-predictions.jsonl",
      "predictions_sha256": "b0f8e192e1b0effa0f9c7e5c665ccdd09a719dc6f1abe7188cf353e1693fa86f",
      "metadata_sha256": "10e1c30d89c2cc7fa666088d35dc19dfaedae74df53c6c4cc0818a4ba27573bd",
      "runtime": null,
      "policy_sha256": null,
      "model_or_adapter_sha256": null,
      "resource_summary": null,
      "baseline_protocol": null,
      "fixture_reused_previously_scored": true
    }
  },
  "timing_scopes": {
    "qwen_head_epoch2_matched": {
      "scope": "sum of three per-question timings; model load/preflight/reference excluded",
      "n": 60,
      "p50_ms": 387.8458335,
      "p95_ms": 426.80982950000003,
      "question_scope": {
        "scope": "per question: individual query/candidate forwards plus typed scorer; excludes load/preflight/reference",
        "n": 180,
        "p50": 123.898646,
        "p95": 160.69458575
      }
    },
    "qwen_lora_epoch9_matched": {
      "scope": "sum of three per-question timings; model load/preflight/reference excluded",
      "n": 60,
      "p50_ms": 390.9686665,
      "p95_ms": 428.4255744,
      "question_scope": {
        "scope": "per-question individual query/candidate forwards plus typed scorer; excludes model load/preflight/reference",
        "n": 180,
        "p50": 124.3370835,
        "p95": 161.70642080000002
      }
    },
    "harrier_head_epoch9_matched": {
      "scope": "sum of three per-question timings; model load/preflight/reference excluded",
      "n": 60,
      "p50_ms": 388.3605625,
      "p95_ms": 426.29533635,
      "question_scope": {
        "scope": "sequential individual text forwards plus typed scorer; excludes model load, preflight and reference scoring",
        "n": 180,
        "p50": 123.892458,
        "p95": 160.61943749999998
      }
    },
    "harrier_lora_epoch9_matched": {
      "scope": "sum of three per-question timings; model load/preflight/reference excluded",
      "n": 60,
      "p50_ms": 390.52054150000004,
      "p95_ms": 427.81191075000004,
      "question_scope": {
        "scope": "sequential individual text forwards plus typed scorer; excludes model load, preflight and reference scoring",
        "n": 180,
        "p50": 124.3164375,
        "p95": 161.172483
      }
    },
    "qwen_frozen_baseline_full660": {
      "n": 60,
      "p50": 320.93535399999996,
      "p95": 344.7769281,
      "scope": "sum of 3 question calls/case; excludes model load; warmup excluded"
    },
    "harrier_frozen_baseline_full660": {
      "n": 60,
      "p50": 320.5260425,
      "p95": 344.91560695,
      "scope": "sum of 3 question calls/case; excludes model load; warmup excluded"
    },
    "qwen_same460_linear": {
      "n": 60,
      "p50": 322.0725,
      "p95": 348.1571077,
      "scope": "three-question per-case sum; excludes model load/warmup"
    },
    "harrier_same460_linear": {
      "n": 60,
      "p50": 322.578001,
      "p95": 347.6380812499999,
      "scope": "three-question per-case sum; excludes model load/warmup"
    },
    "laya_archived": {
      "scope": "runner wall time from before the first question through case completion; includes forwards/evaluation, per-call footprint+swap sampling, cache cleanup/GC, and incremental artifact writes; excludes model load",
      "n": 60,
      "p50": 494.821063,
      "p95": 520.7280187499999,
      "mean": 494.00033820000004
    },
    "jev_archived": {
      "scope": "one JEV request containing the three named questions; archived scope",
      "n": 60,
      "p50_ms": 1064.3935000000001,
      "p95_ms": 2259.0365440499995
    }
  },
  "candidate_position_distribution": {
    "qwen_head_epoch2_matched": {
      "choice": {
        "0": 13,
        "1": 24,
        "3": 6,
        "2": 12,
        "4": 5
      },
      "noul": {
        "0": 60
      },
      "score": {
        "2": 58,
        "4": 2
      }
    },
    "qwen_lora_epoch9_matched": {
      "choice": {
        "0": 19,
        "1": 19,
        "3": 6,
        "4": 5,
        "2": 11
      },
      "noul": {
        "1": 31,
        "0": 29
      },
      "score": {
        "2": 58,
        "4": 2
      }
    },
    "harrier_head_epoch9_matched": {
      "choice": {
        "0": 14,
        "1": 29,
        "2": 4,
        "3": 7,
        "4": 6
      },
      "noul": {
        "0": 60
      },
      "score": {
        "2": 40,
        "4": 16,
        "3": 4
      }
    },
    "harrier_lora_epoch9_matched": {
      "choice": {
        "0": 20,
        "1": 19,
        "3": 6,
        "4": 4,
        "2": 11
      },
      "noul": {
        "0": 50,
        "1": 10
      },
      "score": {
        "2": 30,
        "3": 18,
        "4": 12
      }
    },
    "qwen_frozen_baseline_full660": {
      "choice": {
        "0": 10,
        "1": 29,
        "2": 11,
        "4": 8,
        "3": 2
      },
      "noul": {
        "1": 53,
        "0": 7
      },
      "score": {
        "0": 47,
        "3": 2,
        "1": 3,
        "2": 3,
        "4": 5
      }
    },
    "harrier_frozen_baseline_full660": {
      "choice": {
        "0": 20,
        "1": 18,
        "3": 6,
        "4": 4,
        "2": 12
      },
      "noul": {
        "1": 45,
        "0": 15
      },
      "score": {
        "2": 36,
        "4": 12,
        "3": 10,
        "1": 2
      }
    },
    "qwen_same460_linear": {
      "choice": {
        "0": 14,
        "1": 22,
        "3": 6,
        "2": 10,
        "4": 8
      },
      "noul": {
        "0": 19,
        "1": 41
      },
      "score": {
        "1": 16,
        "3": 12,
        "2": 5,
        "0": 22,
        "4": 5
      }
    },
    "harrier_same460_linear": {
      "choice": {
        "0": 18,
        "1": 21,
        "3": 6,
        "4": 4,
        "2": 11
      },
      "noul": {
        "1": 49,
        "0": 11
      },
      "score": {
        "1": 5,
        "2": 38,
        "4": 8,
        "3": 9
      }
    },
    "laya_archived": {
      "choice": {
        "1": 21,
        "0": 22,
        "4": 2,
        "2": 11,
        "3": 4
      },
      "noul": {
        "1": 57,
        "0": 3
      },
      "score": {
        "2": 56,
        "4": 4
      }
    },
    "jev_archived": {
      "choice": {
        "0": 16,
        "3": 11,
        "1": 17,
        "4": 4,
        "2": 12
      },
      "noul": {
        "1": 32,
        "0": 28
      },
      "score": {
        "2": 32,
        "1": 28
      }
    }
  },
  "collapsed_candidate_position_types": {
    "qwen_head_epoch2_matched": [
      "noul"
    ],
    "qwen_lora_epoch9_matched": [],
    "harrier_head_epoch9_matched": [
      "noul"
    ],
    "harrier_lora_epoch9_matched": [],
    "qwen_frozen_baseline_full660": [],
    "harrier_frozen_baseline_full660": [],
    "qwen_same460_linear": [],
    "harrier_same460_linear": [],
    "laya_archived": [],
    "jev_archived": []
  },
  "first_question_structural_parity": {
    "qwen_head_epoch2_matched": {
      "all_rows_match_fixture_candidate_schema": true,
      "row_count": 180,
      "first_case_candidate_order_valid": true
    },
    "qwen_lora_epoch9_matched": {
      "all_rows_match_fixture_candidate_schema": true,
      "row_count": 180,
      "first_case_candidate_order_valid": true
    },
    "harrier_head_epoch9_matched": {
      "all_rows_match_fixture_candidate_schema": true,
      "row_count": 180,
      "first_case_candidate_order_valid": true
    },
    "harrier_lora_epoch9_matched": {
      "all_rows_match_fixture_candidate_schema": true,
      "row_count": 180,
      "first_case_candidate_order_valid": true
    },
    "qwen_frozen_baseline_full660": {
      "all_rows_match_fixture_candidate_schema": true,
      "row_count": 180,
      "first_case_candidate_order_valid": true
    },
    "harrier_frozen_baseline_full660": {
      "all_rows_match_fixture_candidate_schema": true,
      "row_count": 180,
      "first_case_candidate_order_valid": true
    },
    "qwen_same460_linear": {
      "all_rows_match_fixture_candidate_schema": true,
      "row_count": 180,
      "first_case_candidate_order_valid": true
    },
    "harrier_same460_linear": {
      "all_rows_match_fixture_candidate_schema": true,
      "row_count": 180,
      "first_case_candidate_order_valid": true
    },
    "laya_archived": {
      "all_rows_match_fixture_candidate_schema": true,
      "row_count": 180,
      "first_case_candidate_order_valid": true
    },
    "jev_archived": {
      "all_rows_match_fixture_candidate_schema": true,
      "row_count": 180,
      "first_case_candidate_order_valid": true
    }
  }
}
```

Historical full-660 baselines and same-460 linear comparators are separate systems; the old three-epoch LoRA runs are deliberately excluded. Laya and JEV outputs were reused without rerunning either system.
