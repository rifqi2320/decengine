# Read-only LoRA memory diagnosis (2026-09-24)

## Scope and status

This note is based only on saved train-only logs/configuration, the trainer source,
and previously captured OS memory reports. No model was loaded and no inference,
training, test, or heldout operation was run for this diagnosis. No training file was
modified for it. No test/heldout source or label was opened.

There is no completed two-model LoRA experiment. The unrestricted Qwen run has two
completed epochs but was interrupted after a 17.3 GB process peak. The later capped
Qwen attempt stopped at step 10. Harrier has smoke-only artifacts; no full Harrier
train/dev run exists.

## Saved memory readings

`runs/qwen-capped/memory.jsonl` has only three snapshots. `process_peak_footprint`
is a lifetime high-water value at that snapshot; it does not identify the exact
update at which the peak occurred. There are no memory observations for steps 2–9.

| Snapshot | MLX active | MLX peak | Process current footprint | Process peak footprint | System swap used |
| --- | ---: | ---: | ---: | ---: | ---: |
| startup/resume | 1,194,904,648 B | 0 B after reset | 1,556,334,200 B | 1,556,366,968 B | 1,640,622,981 B |
| after step 1 | 1,204,958,652 B | 1,601,603,872 B | 1,514,833,528 B | 1,997,407,864 B | 1,640,622,981 B |
| after step 10 | 1,204,928,404 B | 1,892,488,166 B | 3,355,248,272 B | 6,472,418,960 B | 1,640,622,981 B |

At step 10, MLX active memory was about 1.12 GiB and MLX's allocator peak about
1.76 GiB, while OS process peak was about 6.03 GiB. The 4.58 GB (4.27 GiB) difference
between those peak counters is not attributed by the saved logs. The memory guard stopped
the run at that snapshot: the internal 6 GiB stop was crossed by about 28.6 MiB,
but the user's 8 GiB absolute cap was not crossed. Current footprint was lower than
the high-water mark, so reading only current RSS/footprint would hide the peak.

The previous `runs/qwen/` run has no per-step memory file. Its stop record preserves
Python PID 53934 / wrapper PID 53931 and an OS `Footprint` peak of 17.3 GB, with
17 GB in `IOAccelerator (graphics)` in `vmmap`. At the captured point, system swap
was 27,903.88 MiB used of 28,672 MiB and system-wide free memory was 9%. Its last
committed checkpoint was epoch 2 / step 920; four later step rows (921–924) were
uncommitted. No exact per-step active-memory series exists for that run; do not
interpolate one from training loss/time logs. After stop, swap settled near 1.48–1.57
GB of the then-configured 3 GB and free memory returned to 83–85%; no LoRA process
remained.

The two one-minibatch memory-instrumented smokes (actual checkpoint load and update)
peaked at 1,934,902,904 B for Qwen and 1,936,524,944 B for Harrier. Their MLX peaks
were about 1.60 GB, and swap remained unchanged at 1,640,622,981 B. Loss decreased and
LoRA gradients were nonzero, but one step is not a predictor for multi-step allocator
growth.

## Train-only sequence lengths

Saved profile-compiled train feature metadata covers 2,788 text inputs per model
(query plus candidates over clean300 + multi train120): min 9, p50 24, p90 167,
p95 179, p99 257, max 266 tokens. The capped trainer used a 288-token guard and
rejects overlength input; it does not truncate. Thus 288 preserves every inspected
train/dev input with a 22-token margin. No test token distribution was inspected.

Do not reduce the cap below 266 while keeping this train set: p90/p95/p99 already
show a substantial tail, and token counts do not show where answer-relevant evidence
occurs. Shortening to 128 would affect at least the top decile of texts and could
remove evidence. Excluding cases over a short cap would change the family/domain
composition and require a new preregistered dataset/split; neither truncation nor
silent exclusion is evidence-safe. Keep 288 plus explicit rejection, or use a
different smaller train set only after approval and a fresh data-selection protocol.

## Source-level diagnosis

Relevant path in `train.py`:

- `batch_inputs` tokenizes one query and all K candidates for a question, pads those
  `1+K` rows to that question's longest sequence, and creates new MLX token arrays.
- `PreferenceModel.__call__` runs the decoder on the variable `(1+K, width)` shape,
  gathers actual last-token states, and normalizes them.
- `TypedScorer.__call__` broadcasts the one query over candidates and materializes
  concatenated `(q, c, q*c, abs(q-c))` features. This tensor is FP32 and scales only
  with K x 4096; it is small beside decoder state/Metal allocations.
- Each update creates a new `loss_fn`/`value_and_grad` closure and new per-question
  arrays, calls `mx.eval(loss, grads)`, updates AdamW, then calls
  `mx.eval(model.parameters(), optimizer.state)`. There is no explicit gradient
  accumulation. A new `mx.clear_cache()` is called at step 1 and then only every 10
  steps in the capped run.
- The model uses two upper q/v LoRA layers, rank 4, in the capped run; base parameters
  are frozen BF16 and only adapters/head are FP32. No KV inference cache is passed.
- Dev scoring is per question, and `steps.jsonl` retains only scalar metrics. The
  epoch loss list holds Python floats, not graphs or arrays.

Assessment of proposed causes:

| Cause | Evidence from saved source/logs | Assessment |
| --- | --- | --- |
| Optimizer state / accumulating gradients | AdamW state has fixed-size moments for roughly a small scorer plus LoRA; optimizer is updated every step and there is no gradient-accumulation loop. Logged MLX active memory stays ~1.20 GB from step 1 to 10. | Unlikely to explain multi-GB rise. |
| Per-step concatenations / query-candidate reuse | One query and K candidates are encoded together; a query embedding is broadcast to K rows for small pair features. Arrays are regenerated per question; no reuse cache is accumulated. | Tensor sizes are modest, but their varying shapes contribute to allocation churn. |
| Sequence tails | Each question has a different padded width and K; train text max 266. No per-step token length or K is logged with the memory sample. | Variable shape signatures are a plausible contributor, but cannot be linked to the observed peak from current logs. |
| Lazy graph retention | Loss and gradients are explicitly evaluated before optimizer update, which argues against an unmaterialized whole-epoch graph. But loop locals (`loss_fn`, `loss_and_grad`, `grads`, `flat`, token arrays) are not explicitly deleted before the next iteration. | Not demonstrated as a leak; one-update lifetime overlap is possible. Explicit `del` could reduce peak, but is not sufficient evidence of the multi-GB rise. |
| MLX allocator cache | MLX active stayed ~1.20 GB; MLX peak only grew to ~1.89 GB. Cache clearing every 10 steps did not stop OS process peak reaching 6.47 GB by the step-10 sample. | Some cache retention is plausible; `mx.clear_cache()` at that cadence did not solve it. |
| Metal/IOAccelerator or compiled-shape allocation | Prior 17.3 GB run attributed ~17 GB to `IOAccelerator (graphics)`; capped run's OS high-water greatly exceeded MLX active/peak counters. Variable widths/batch rows create different shapes. | Best-fitting hypothesis, not proven. Saved data lacks per-region/per-step samples in the capped run. |

The current data cannot distinguish driver allocation/cache from transient lazy
graphs or compiler/kernel caches. It also cannot show whether the 6.47 GB high-water
is a short transient or persistent allocation: current footprint at step 10 was
3.36 GB. The first unrestricted run's RSS was not representative of Metal usage.

## Code-only mitigations and risks (not applied here)

1. **Keep a strict OS guard, not just MLX counters.** Sample process `phys_footprint`
   every optimizer update, not every 10; use a stop threshold around 5 GiB for a 6
   GiB operating budget, with the 8 GiB absolute cap never relaxed. Record current
   and lifetime peak, PID, step, K, batch width, total tokens, active/peak MLX bytes,
   and system swap. A monitor is not a hard OS quota: a single update may transiently
   exceed the threshold before the next sample.
2. **Explicitly release per-step references after `mx.eval` and the optimizer update**
   (`del loss, grads, flat, loss_and_grad, loss_fn, tokens, lengths, after` when
   defined), then clear the MLX cache every step. This may release graph/scratch
   buffers; it may also have no effect on Metal/IOAccelerator reservations, as cache
   clearing every 10 did not cap the peak. Deleting `grads` too early would be wrong;
   do it only after `optimizer.update` and evaluation have completed.
3. **Bucket sequence widths without truncation.** Right-pad to fixed ceilings (e.g.
   64/128/192/288) and gather at each original `length-1`; because every train string
   is <=266, this preserves all text/evidence. It reduces shape variants, but increases
   FLOPs/padding (especially for the p50=24 tail) and has not been memory-tested.
   Bucket candidate count too only if dummy rows are masked out of the softmax; an
   incorrect mask changes the within-question objective.
4. **Rematerialize only the upper trainable block(s)** with MLX checkpointing if its
   parameter-capture semantics are verified. This trades recomputation for saved
   activations. A real LoRA gradient test for both query and candidate rows is
   mandatory after wrapping; captured/frozen parameters may otherwise lose gradients.
5. **Reduce trainable scope to 1 upper q/v layer, rank 2, and retain question
   micro-batch 1.** This lowers optimizer/adapter state (already small) and some
   activation use, but likely will not by itself fix a multi-GB OS/Metal cache issue.
   Do not introduce gradient accumulation; it increases live graph retention unless
   implemented as safe, evaluated per-microbatch gradients.
6. **Avoid candidate-wise sequential recomputation as a quick patch.** Exact
   within-question CE needs all candidate logits/probabilities and gradients. Pairwise
   re-encoding can increase work and retain multiple graphs unless implemented as a
   two-pass analytic-gradient/rematerialization scheme; it must preserve the exact
   local objective and query-side gradient.

## Feasibility conclusion

The new run is under the 8 GiB hard ceiling through its ten observed updates, but the
6.03 GiB peak at step 10 after only 1.8 GiB one-step smoke is a sharp enough rise that
full compliance is not established. The earlier 17.3 GB run confirms that memory can
grow well beyond the cap on this 24 GiB host. It is **not currently realistic to
promise a full two-model 3-epoch local run under 6 GiB**, and the evidence is
insufficient to certify even the absolute 8 GiB cap over 1,380 updates/model. A
fixed-shape/rematerialized implementation might make it fit, but requires a new
approved low-resource smoke and continuous OS-footprint monitoring before any full
run. If the cap is non-negotiable, the reliable recommendation is smaller explicitly
approved training data/model or remote hardware with a suitable memory budget—not
raising the limit. This was read-only diagnosis; no further compute was launched.
