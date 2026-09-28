# Frozen domain-generalization folds

This protocol was frozen from fixture scenario metadata and reference labels only. No model execution or prediction files were used. The machine-readable, exhaustive ID assignment is `folds.json` (600 case records plus ten explicit folds).

## Recommendation and fold use

Use the **original `probe-train-300.jsonl` only for fitting and development** and `probe-test-300.jsonl` only as a sealed external evaluation set. Do not pool all 600 rows for training or tuning: that would turn cases from the nominal test fixture into training data and remove its independent-test status.

There are ten leave-one-industry-out (LOIO) folds. For fold `d`:

* Fit on `fit_train_ids`: train-source cases whose canonical industry is not `d`.
* Use `development_validation_ids`: train-source cases in `d` as the outer held-out-domain validation slice. If choosing hyperparameters, do so using only the other training data (for example, an inner LOIO within `fit_train_ids`) and reserve this slice for the outer estimate. If it is used to choose hyperparameters, label its score selection-biased and do not present it as an unbiased outer-fold estimate. Do not tune on `sealed_test_ids`.
* `sealed_test_ids` are the 30 test-source cases in `d`. Keep them untouched during all fold-level selection. After choices are frozen, either report these as the final per-domain test evaluation from a model trained on the permitted train source (state the training protocol), or use the train-source LOIO estimates as the primary tuning evidence and make one final, aggregate test evaluation. Never use held-out test performance to select a fold's hyperparameters.

The ten domain blocks and train/validation/sealed-test support are:

| Held-out industry | Fit train | Train-domain validation | Sealed test |
|---|---:|---:|---:|
| education | 285 | 15 | 30 |
| energy_utilities | 262 | 38 | 30 |
| financial_services | 277 | 23 | 30 |
| healthcare | 278 | 22 | 30 |
| hospitality | 268 | 32 | 30 |
| manufacturing | 278 | 22 | 30 |
| public_services | 249 | 51 | 30 |
| retail_ecommerce | 276 | 24 | 30 |
| software_saas | 262 | 38 | 30 |
| transport_logistics | 265 | 35 | 30 |

This is a domain-held-out **development** protocol, not ten independent test folds: the 300-row test source is external and must not participate in tuning. The test source has 30 cases per canonical domain. Fit sizes vary with the long-tail source-industry labels in train. A selected model's final test result should be reported once after all choices are frozen, not repeatedly used to select among fold/model variants.

## Canonicalization

The fixed target set is `healthcare`, `manufacturing`, `hospitality`, `financial_services`, `retail_ecommerce`, `transport_logistics`, `education`, `energy_utilities`, `software_saas`, and `public_services`. Each case's original `scenario.industry`, canonical `industry`, source and ID are explicitly recorded in `folds.json`. Apply the following ordered, case-insensitive substring rules to normalized labels (lowercase; spaces and hyphens become `_`); first matching group wins:

1. `healthcare`: `health`, `medical`, `veterinary`, `pharma`, `clinical`, `biotech`, `laboratory`.
2. `manufacturing`: `manufactur`, `semiconductor`, `chemical`, `automotive`, `watercraft`.
3. `hospitality`: `hospital`, `hotel`, `tourism`, `travel`, `event`, `arts`, `museum`, `sports`, `food_service`.
4. `financial_services`: `bank`, `insurance`, `financial`, `lending`, `credit_union`, `accounting`.
5. `retail_ecommerce`: `retail`, `commerce`, `ecommerce`, `fashion`, `consumer_goods`, `consumer_electronics`.
6. `transport_logistics`: `transport`, `transit`, `aviation`, `rail`, `postal`, `shipping`, `logistic`, `warehouse`, `fleet`, `marine`, `vehicle`.
7. `education`: `education`, `school`, `university`, `library`, `childcare`.
8. `energy_utilities`: `energy`, `electricity`, `utility`, `water_`, `mining`, `forestry`, `agriculture`, `environment`, `waste`, `food_manufacturing`, `food_distribution`, `cold_storage`.
9. `software_saas`: `software`, `cloud`, `cyber`, `telecommunication`, `telehealth`, `media`, `broadcast`, `publishing`, `research`, `technology`.
10. Otherwise `public_services` (a deliberately broad residual bucket, not a claim that all such sectors are semantically public services).

The ordering is material (e.g. `telehealth` maps to healthcare before the software rule; `automotive_repair` maps to manufacturing). The canonicalization is a coarse fixed evaluation grouping, not a learned mapping. Check `folds.json` rather than independently re-deriving it downstream.

## Fixture support and limitations

Canonical counts are train/test respectively: healthcare 22/30, manufacturing 22/30, hospitality 32/30, financial services 23/30, retail/e-commerce 24/30, transport/logistics 35/30, education 15/30, energy/utilities 38/30, software/SaaS 38/30, public services 51/30. Test labels are balanced at 30 per target group, but train is not. The train fixture has many one- and two-case source-industry labels; grouping improves per-domain support but several canonical groups remain small (education n=15, healthcare/manufacturing n=22, financial n=23, retail n=24).

Across the 600 references, owner counts are safety_compliance 131, technical_support 103, operations 100, billing 80, sales 67, account_access 62, other 57; urgent counts are false 337 / true 263; impact counts are low 179 / moderate 211 / high 210. No owner, urgency, or impact label is absent from the complete corpus. However, individual train held-out domains are sparse: education has only 15 validation cases and no sales label; healthcare has no billing/account_access/other labels; retail has only five urgent cases and no owner class with more than nine examples; transport has only one low-impact case. Report per-class support and avoid interpreting unstable per-domain class metrics as precise. No unsupported owner/impact class should be imputed or merged based on model results.

## Leakage and duplication checks

Fixture-level audit compared only case summaries across train and test after lowercase/alphanumeric normalization. There were zero exact cross-source summary duplicates. Character-sequence ratio screening found zero pairs at `>=0.85`, and one pair at `>=0.75` (`train-102` vs `test-019`, retail vs retail/e-commerce, ratio about 0.777). This does not rule out paraphrase/template leakage. Both sources reuse the same task/question and output schema by construction, so those prompt fields are not independent evidence; the 10-domain normalization also intentionally merges related sectors (e.g. aliases). Preserve the source boundary and avoid splitting on individual aliases as though they were independent domains. No exact scenario-template IDs were present in the checked fixture fields; `scenario.ambiguity` is not a template identifier.

## Manifest schema

`cases` has one record per fixture row, including `id`, `source_split`, `source_file`, raw `source_industry`, canonical `industry`, ambiguity, and reference labels. Each `folds[industry]` explicitly lists fit, train-domain validation, and sealed test IDs. This file is the frozen split artifact; do not regenerate it from predictions or alter assignment after viewing test performance.
