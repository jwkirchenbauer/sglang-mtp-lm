# PHASE 2A Debrief: Adaptive MTP Stabilization With Temporary q_len>1 CUDA Graph Bypass

## Summary
Phase 2A re-scoped to correctness-first stabilization for MTP decode with overlap disabled:
1. Keep static-k and `conf_adapt` decode behavior correct.
2. Keep CUDA graph active for `decode_q_len_per_req == 1`.
3. Temporarily bypass CUDA graph replay for `decode_q_len_per_req > 1`.
4. Keep MTP+overlap fail-loud until q>1 graph and overlap reintegration are intentionally resumed.

This checkpoint is intended as a safe handoff state that runs correctly for the validated static/adaptive workflows, with known performance limitations for q>1 decode due eager fallback.

## Implemented Scope (What Works)

## Strategy Parsing and API Surface
- `SamplingParams` now supports canonical list-style strategy input:
  - `mtp_strategy=["conf_adapt", <threshold>]`
- `conf_adapt` threshold validation and normalization:
  - Accepts threshold from list and/or `mtp_conf_threshold`.
  - Rejects malformed values and conflicting thresholds.
  - Enforces `mtp_strategy` requires `mtp_enabled=true`.
- `mtp_k` remains max-k.

## Runtime MTP Semantics
- Static-k path preserved and functioning for `k=1`, `k=2`, `k=3`.
- `conf_adapt` path implemented with contiguous-prefix confidence rule:
  - Compute top-1 confidence per emitted candidate.
  - Select longest contiguous prefix above threshold.
  - Fallback to 1 token minimum.
- Effective-k metadata is propagated through decode output processing.
- Request-level MTP state tracks per-step pending/commit behavior and debug state.

## Decode Batch Construction / Scheduling
- Decode rows support phase-aware MTP shapes (seed vs steady).
- Scheduling logic includes MTP decode bucket awareness for adaptive shape divergence.
- Added temporary fail-loud policy for MTP+overlap:
  - MTP requests require server start with `--disable-overlap-schedule`.

## CUDA Graph Policy (Temporary)
- Graph replay remains enabled and used for standard decode (`q_len==1`).
- Temporary phase-2A guard:
  - If decode batch has `decode_q_len_per_req > 1`, force eager forward (`can_run_cuda_graph=false`).
- One-time warning emitted per process on first bypass, with payload:
  - `decode_q_len_per_req`
  - `mtp_phase`
  - `mtp_strategy_kind`

## Observability
- MTP debug trace includes runtime fields needed for diagnosis:
  - `can_run_cuda_graph`
  - phase before/after step
  - decode q_len and decode-k for the step
  - effective-k (adaptive/static)
  - token rows (`input`, `pending`, `committed`)
  - positions and seq-len base
  - finish reason (length/stop)

## Validation Status (User-Run)

## Passed / Confirmed
1. Policy guard behavior:
- MTP request with overlap enabled fails loudly with explicit message to use `--disable-overlap-schedule`.

2. Baseline correctness with overlap OFF, graph OFF:
- Static `k=1/2/3` and eval outputs reported sane.
- Prior gibberish/invalid negative-id behavior no longer present in this configuration.

3. Stabilized graph behavior with overlap OFF, graph ON:
- `k=1` decode uses CUDA graph.
- `k=2/3` decode now bypasses graph (expected temporary policy).
- Output quality is consistent with graph-OFF behavior in tested runs.

4. Adaptive sanity:
- `conf_adapt` path functions and produces expected behavior under tested workflows.

## Current Functional Tradeoff
- q>1 decode runs eagerly, so expected speedups from multi-token decode+graph replay are not yet realized.
- This is intentional for correctness stabilization.

## Files Touched in Phase 2A Workstream
- `python/sglang/srt/sampling/sampling_params.py`
- `python/sglang/srt/layers/logits_processor.py`
- `python/sglang/srt/managers/schedule_batch.py`
- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/managers/scheduler_output_processor_mixin.py`
- `python/sglang/srt/managers/tokenizer_manager.py`
- `python/sglang/srt/managers/overlap_utils.py`
- `python/sglang/srt/model_executor/forward_batch_info.py`
- `python/sglang/srt/model_executor/model_runner.py`
- `python/sglang/srt/model_executor/cuda_graph_runner.py`
- `python/sglang/srt/layers/attention/flashinfer_backend.py`

## Known Issues / Deferred Work

## P0: q_len>1 CUDA Graph Replay (Primary Blocker)
- Multi-token decode graph replay path remains unstable when enabled directly.
- Historical failure modes seen during bring-up:
  - early EOS collapse,
  - gibberish / unstable token trajectories,
  - inconsistent first-step argmax in some runs,
  - capture/replay metadata mismatch issues.
- Current mitigation is explicit bypass for q>1 decode.

## P1: Overlap Reintegration (Intentionally Deferred)
- MTP+overlap is currently blocked by policy.
- Overlap-related MTP scaffolding exists but is out of active acceptance scope.
- Re-enable only after q>1 graph stability is achieved and validated.

## P1: Performance Gap
- With q>1 eager fallback, adaptive/static multi-token decode does not yet get intended graph acceleration.
- Correctness accepted; performance optimization deferred.

## Phase 2B Restart Plan (Concrete TODOs)
1. Build a deterministic q>1 graph repro harness:
- single-request curl harness with fixed prompt,
- k=2 and k=3 traces,
- capture first 3 decode steps (`input/pending/committed/positions/effective_k`).

2. Isolate q>1 graph path in controlled switch:
- keep default bypass ON,
- add temporary opt-in flag/env to force q>1 graph for debugging only.

3. Stabilize decode-as-prefill replay path:
- validate metadata init/capture/replay invariants,
- verify seq-len/prefix-len clamping for padded synthetic rows,
- verify wrapper selection and forward_metadata freshness each replay.

4. Re-enable q>1 graph only after A/B parity gate:
- graph ON (q>1) must match graph OFF qualitatively and accuracy-wise,
- no systematic early EOS collapse,
- no KV leaks / allocator imbalance.

5. Revisit overlap reintegration only after step 4:
- keep homogeneous adaptive bucket policy,
- validate overlap future-map token-window/effective-k contract end-to-end,
- remove temporary overlap guardrails once stable.

## Recommended Bring-Up Commands
- Baseline correctness:
  - server: `--disable-overlap-schedule --disable-cuda-graph`
- Graph-stabilized Phase2A mode:
  - server: `--disable-overlap-schedule`

## Notes for Future Session
- Treat this checkpoint as correctness-stable but performance-incomplete.
- First acceptance criterion for next phase is safe q>1 graph replay re-enable, not overlap.
- Keep using MTP debug trace fields introduced in Phase2A as primary parity signal.
