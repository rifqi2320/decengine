# Capped Qwen run stopped by memory guard

This was a safety probe of the updated full-run path, **not a completed epoch and
not a trained model selection**. No test/heldout input or label was accessed.

- Configuration: question micro-batch 1, 288-token reject-only limit, two upper
  q/v LoRA layers at rank 4, 10-step Metal cache/memory check, process stop at 6 GiB
  and system-swap stop at 4 GiB.
- The run stopped at step 10 when OS `phys_footprint_peak` reached 6,472,418,960 B
  (about 6.03 GiB), above the 6 GiB early-stop threshold but below the hard 8 GiB
  cap. Current process footprint at measurement was 3,355,248,272 B. MLX active
  allocator was 1,204,928,404 B and MLX allocator peak 1,892,488,166 B.
- System swap remained 1,640,622,981 B through the measurement; no swap growth was
  observed. After process exit, current system swap was about 1.55 GB / 3 GB and
  memory pressure reported 85% free.
- This run's PID was not recorded by the early memory logger. The process exited
  with the guard exception; a process query after exit showed no trainer/inference
  process. Future logs should include `os.getpid()`.
- Only ten step rows are present; `latest.json` remains at the initial state
  (`completed_epoch=0`, `step=0`). No epoch checkpoint or model selection exists.
- The first-step gradient/loss decrease and parity still passed in the instrumented
  smoke directories; this does not establish multi-step fit under the cap.

The prior less-restricted Qwen run under `runs/qwen/` remains separately documented
as a 17.3 GB peak incident. Neither directory should be represented as a completed
LoRA experiment.
