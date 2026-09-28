# Interrupted train-only CV attempt (preserved note)

The first Qwen run against clean300 + multi train120 used the initial experimental
optimizer and was interrupted before CV completed. It emitted no predictions,
metrics, selected config, or model artifacts. A later attempt with the modified
optimizer was also interrupted while the custom optimizer was proving unstable.
Those partial attempts are intentionally not treated as evidence and were not
overwritten with a successful run. The reproducible bounded runner and complete
results live in the model-specific `*-trainonly` output directories.

Only train fixture/features were used. No test-60, matched-test, or test200 labels
or predictions were read by these attempts.
