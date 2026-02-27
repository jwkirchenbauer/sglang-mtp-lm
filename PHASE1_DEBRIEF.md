# PHASE 1 Debrief: Static-k MTP Integration in SGLang

## Summary
Phase 1 delivered the first working end-to-end integration of the custom MTP decoding path in SGLang for real eval workloads.

Checkpoint commit: `7153c0205` (`First working version of a static k MTP model integration within SGL`).

This phase prioritized:
1. Correctness and liveness for `mtp_enabled=true` with `k=1` and `k>1`.
2. KV-cache correctness with delayed commit semantics.
3. Compatibility with continuous batching under constrained policies.
4. Practical observability to debug token/phase/position parity.

## Implemented Scope (What Works)

## Core MTP Runtime
- Added fixed-k MTP runtime path integrated into decode flow.
- Supports multi-token emission per decode step for `k>1`.
- Preserves a dedicated MTP codepath even for `k=1` (no fallback to standard decode when `mtp_enabled=true`).

## Phase-Aware Decode
- Request-level MTP phase state (`seed` vs `steady`) implemented.
- Decode input row construction implemented by phase.
- Emit window logic implemented so sampler emits exactly `mtp_k` tokens (not all decode query positions).

## KV Allocation + Commit Semantics
- Decode allocation path updated for multi-slot decode writes.
- `req_to_token` mapping updated for multi-token decode placement.
- Delayed KV commit behavior implemented to match MTP recomputation semantics.
- Provisional cache lifecycle handling added (carry/free/commit sequencing).

## Scheduler and Batching Behavior
- Decode grouping includes MTP characteristics to avoid incompatible mixed decode batches.
- Runtime guards for mixed MTP/non-MTP decode batches and mixed `mtp_k` conditions.

## Backend and Runtime Guards
- MTP path guarded to flashinfer backend.
- Overlap schedule + MTP is fail-loud (requires `--disable-overlap-schedule` for now).
- Explicit guardrails added for known unsupported combinations.

## Sampling / Forward Batch / Positioning
- Decode `q_len>1` metadata propagated through forward batch.
- Position handling updated for flattened decode inputs.
- Seed anchor correction added for first-step parity alignment.

## Observability and Debugging
- Added MTP stepwise debug trace plumbing and first-step argmax logging.
- Debug payload includes phase/input row/argmax/pending/committed/KV transitions.
- Added decoded token text in debug outputs for fast parity inspection.

## Metrics Stability
- Timing field ordering fixed for finished requests.
- Decode metric emission guarded to avoid negative/invalid decode-time artifacts in response metrics.

## Public/Runtime Interface Notes
- Existing sampling args remain the MTP surface:
  - `mtp_enabled`
  - `mtp_k`
  - `mtp_mask_id` (or min/max mask id options)
- Current required runtime policy:
  - `attention_backend=flashinfer`
  - `disable_overlap_schedule=true` for MTP
- Typical usage:
  - `--gen_kwargs "temperature=0,top_k=1,mtp_enabled=true,mtp_k=<k>,mtp_mask_id=<id>"`

## Files Touched in Phase 1
- `python/sglang/srt/layers/attention/flashinfer_backend.py`
- `python/sglang/srt/layers/logits_processor.py`
- `python/sglang/srt/managers/schedule_batch.py`
- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/managers/scheduler_output_processor_mixin.py`
- `python/sglang/srt/managers/tokenizer_manager.py`
- `python/sglang/srt/mem_cache/common.py`
- `python/sglang/srt/model_executor/forward_batch_info.py`
- `python/sglang/srt/model_executor/model_runner.py`
- `python/sglang/srt/sampling/sampling_params.py`

## Known Caveats / Deferred Work

## P0 (Important Before Fair Perf Conclusions)
1. Re-enable overlap for MTP (`disable_overlap_schedule` currently required).
2. Re-enable/optimize CUDA graph usage for MTP `k>1`.
3. Ensure MTP+overlap correctness and throughput under concurrency (future-map/token-shape contract).

## P1 (Major Productization)
1. Confidence-adaptive `k` policy on top of static-k scaffolding.
2. Mixed-phase decode improvements beyond strict homogeneous-phase batching.
3. Broader backend support parity (currently flashinfer-first).

## P2 (Completeness + Quality)
1. Per-token logprob behavior for multi-token accepted steps.
2. More robust benchmark metrics definitions for non-streaming eval workflows.
3. Optional cleanup/removal of temporary debug instrumentation once parity work closes.

## Benchmarking and Interpretation Caveats
- Scheduler `gen throughput (token/s)` is interval-based and can be bursty.
- Non-streaming request timing can skew per-request decode window metrics.
- Warm-cache vs cold-cache differences are material.
- Comparing `k` values should normalize:
  - concurrency (`num_concurrent`)
  - prompt set/order
  - cache state (restart or `/flush_cache`)
  - aggregation metric (total completion tokens / wall time)

## Test Status Snapshot (Phase 1)
- Baseline decode and MTP `k=1` functional parity path: passed.
- MTP `k>1` generation no longer crashes on previously observed shape/liveness errors: passed.
- Single prompt qualitative sanity and GSM-style eval runs: passed.
- Concurrent `sglang-generate` server-mode runs show expected scaling trend with larger `k`: observed.
- Remaining: formalized benchmark protocol and overlap/cudagraph restoration.

## Phase 2 Proposed Entry Tasks (Ordered)
1. Implement MTP-aware overlap future-map contract and remove fail-loud requirement.
2. Add MTP-safe CUDA graph strategy (shape bucketing by phase/q_len).
3. Add adaptive-k decision policy using existing runtime hooks (`mtp_effective_k` path).
4. Re-benchmark vs baseline decode and Eagle3 under identical server-mode conditions.
5. Lock final public docs once overlap/cudagraph/adaptive-k status is settled.

## Assumptions and Defaults
- Primary goal in Phase 1 was correctness-first, not full optimization parity.
- Static `k` per request is the current supported optimization model.
- MTP remains backend-constrained (flashinfer) and overlap-constrained by design in this checkpoint.
- Temporary debug instrumentation is acceptable until parity and perf restoration work completes.
