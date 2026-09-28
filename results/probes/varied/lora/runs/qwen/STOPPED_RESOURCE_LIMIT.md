# Qwen partial run stopped for resource safety

This directory is **not a completed train/dev run** and its best epoch is not a
final selected model. It was stopped after the host exceeded the subsequently
specified 8 GiB per-process budget. No test/heldout input or labels were accessed.

- Host: Apple M4 Pro, 24 GiB unified memory.
- Python PID: 53934; wrapper shell PID: 53931.
- OS footprint near stop: 17.0 GB `IOAccelerator (graphics)`; measured process peak
  17.3 GB.
- System swap near stop: 27.9 GB / 28.7 GB used; `memory_pressure` reported 9%
  system free.
- Stop: SIGINT at an MLX gradient evaluation, exit 130. No Harrier full-training
  process was started.
- Safely committed Qwen checkpoint boundary: epoch 2, step 920 (`latest.json`). Four
  epoch-3 updates were logged afterward but are uncommitted. Resume must replay from
  the committed boundary; do not treat those log rows as completed optimization.
- Grouped dev: epoch 1 loss 0.97537 / accuracy 0.40; epoch 2 loss 1.12008 /
  accuracy 0.46. Best-by-loss so far is epoch 1, but this partial run is not frozen
  or presented as the final selected result.
- After exit: no training/inference process remained; swap returned to 1.57 GB of
  3 GB configured and system-wide free memory to 83%.

The actual run used micro-batch one question, four upper q/v LoRA layers, rank 8,
alpha 16, and 512-token hard rejection (no truncation). It exceeded the new user cap
despite micro-batch one. A separate, unrun safer code configuration is documented in
`../../FEASIBILITY.md`; do not launch without approval.
