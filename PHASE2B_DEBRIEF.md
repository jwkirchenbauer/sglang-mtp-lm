# PHASE 2B Debrief: Adaptive MTP Concurrent Correctness Investigation

## Summary
Phase 2B focused on correctness-first stabilization for adaptive MTP under realistic concurrent server load while preserving runtime safety.

Primary goals were:
1. prevent server crash under adaptive concurrent decode,
2. enforce allocator/cache invariants strongly enough to localize producer bugs,
3. gather decision-quality evidence before re-enabling broader performance work.

Current checkpoint:
1. crash risk is mitigated by idle repair safety nets (no hard process failure in the validated matrix),
2. residual issue remains: adaptive-concurrent still produces end-of-run orphan radix KV recovery at first quiescent tick,
3. two leading hypotheses were falsified by instrumentation:
   - terminal-slab release leak hypothesis: falsified (`missing_from_terminal_release_count=0`),
   - seed-anchor-old-loc leak hypothesis: falsified (`seed_anchor_old_loc_state=cached`),
4. full conf_adapt CUDA-graph performance claims remain blocked by unresolved producer-path correctness.

Pause decision: stop this phase here and hand off with evidence plus a decision-complete resume plan.

## Public API / Interface Changes
No user-facing API, CLI, request schema, or model behavior interface changes were introduced in this phase.

Internal debug/diagnostic surface documented and used:
1. `SGLANG_MTP_KV_LEAK_DEBUG` (internal env flag, default off),
2. idle invariant summary logs:
   - `legacy_missing_tokens`,
   - `orphan_token_count`,
   - `overlap_token_count`,
3. terminal release ownership audit logs:
   - `curr_step_token_count`,
   - `freed_by_tail_cleanup_count`,
   - `freed_by_overalloc_range_count`,
   - `retained_as_committed_count`,
   - `missing_from_terminal_release_count`,
   - `seed_anchor_old_loc`,
   - `seed_anchor_old_loc_state`.

## Implemented Scope (What Works)
1. Seed-anchor ownership-aware handling is landed and prevents the prior free-list overlap/corruption class seen earlier in this stream.
2. Idle orphan/overlap repair safety net is active and effective for liveness; servers no longer hard-crash on this class.
3. Static and adaptive workflows are functionally stable in tested single-request modes.
4. Baseline and candidate both complete the validated test matrix without process crash in current guardrail configuration.

## Validation Matrix (Baseline vs Candidate)
| Scenario | Baseline outcome | Candidate outcome | Crash? | Idle invariant result | Notes |
|---|---|---|---|---|---|
| Static `k=3`, single | PASS | PASS | No | Clean after startup artifact | One-time startup orphan (`count=1, sample=[13]`) observed and tolerated |
| Static `k=3`, concurrent=8 | PASS | PASS | No | Clean or effectively clean after run | No terminal crash |
| Adaptive single (`conf_adapt`) | PASS | PASS | No | Clean after run in tested cases | No recurrent runtime orphan recovery pattern |
| Adaptive concurrent (`conf_adapt`, concurrent=8) | PASS (completes) | PASS (completes) | No | Residual orphan recovery at first quiescent tick | `orphan_token_count>0`, `overlap_token_count=0`, recovery log emitted |

## Instrumentation Added in Phase 2B
1. Idle radix invariant summary:
   - one compact quiescent-tick summary with `legacy_missing_tokens`, `orphan_token_count`, `overlap_token_count`.
2. Orphan/overlap repair observability:
   - explicit `Recovered orphan radix KV tokens...` logging with sample,
   - explicit overlap repair logging when applicable.
3. Terminal release ownership audit under `SGLANG_MTP_KV_LEAK_DEBUG`:
   - token ownership partitioning at release (`tail cleanup`, `overalloc range`, `retained committed`),
   - residual terminal set (`missing_from_terminal_release`),
   - seed-anchor old-location state probe (`cached/free/missing`).

## Findings: Confirmed / Falsified Hypotheses
### Confirmed
1. Residual issue is isolated to adaptive-concurrent end-of-run quiescence checking and recovery.
2. `overlap_token_count=0` while orphan recovery still happens, so free-list/cache overlap is not the active residual failure mode.
3. Orphan counts vary by run (for example, `15` and `23` were observed), which does not fit a single fixed terminal-slab leak model.

### Falsified
1. Terminal release slab leak hypothesis is falsified:
   - repeated terminal audits report `missing_from_terminal_release_count=0`.
2. Seed-anchor old-location leak hypothesis is falsified:
   - repeated terminal audits report `seed_anchor_old_loc_state=cached`.
3. Therefore the active producer path is upstream of terminal release accounting and not explained by current seed-anchor ownership handling.

## Residual Known Issue (Unresolved)
Symptom:
1. adaptive-concurrent workloads leave orphan radix KV tokens at the first quiescent tick.

Representative signature:
1. `Idle radix invariant summary ... orphan_token_count>0, overlap_token_count=0`,
2. followed immediately by `Recovered orphan radix KV tokens during idle check...`.

Scope:
1. observed on baseline and candidate servers,
2. observed with and without current candidate CUDA-graph behavior.

Impact:
1. correctness uncertainty remains despite liveness,
2. safety net masks crash but does not close root cause,
3. blocks clean acceptance for adaptive-concurrent correctness and blocks definitive conf_adapt performance claims.

## Operational Status at Pause
Go:
1. continue controlled experiments using current safety-net runtime,
2. continue instrumentation-first producer localization.

No-go:
1. do not claim root-cause closure,
2. do not claim adaptive-concurrent correctness is fully resolved,
3. do not claim final conf_adapt CUDA-graph performance conclusions.

## Resume Plan (Phase 2C Entry Tasks)
1. Focus producer-path tracing before terminal release (non-terminal/defer/requeue paths).
2. Add per-step ownership accounting around adaptive bucket deferral/requeue transitions.
3. Correlate orphan token IDs directly to request lifecycle metadata (`rid`, step index, bucket key, queue transition).
4. Implement a targeted producer fix only after producer evidence is deterministic.
5. Keep idle repair enabled until multiple adaptive-concurrent runs are clean:
   - `orphan_token_count=0`,
   - `overlap_token_count=0`,
   - no post-workload recovery logs.

## Files Touched in Phase 2B
At minimum, the Phase 2B stabilization/instrumentation stream touched:
1. `python/sglang/srt/environ.py`
2. `python/sglang/srt/managers/schedule_batch.py`
3. `python/sglang/srt/managers/scheduler_output_processor_mixin.py`
4. `python/sglang/srt/mem_cache/common.py`
5. `python/sglang/srt/managers/scheduler_runtime_checker_mixin.py`
6. `python/sglang/srt/managers/scheduler.py`
7. `python/sglang/srt/managers/schedule_policy.py`
8. `python/sglang/srt/layers/attention/flashinfer_backend.py`
9. `python/sglang/srt/server_args.py`
10. `python/sglang/srt/model_executor/model_runner.py`
11. `scripts/playground/mtp_cudagraph_ab.py`

## Test Cases and Scenarios (Observed)
1. static single: validated.
2. static concurrent: validated.
3. adaptive single: validated.
4. adaptive concurrent baseline: completes; residual orphan recovery at quiescence.
5. adaptive concurrent candidate: completes; residual orphan recovery at quiescence.
6. `k` variation checks:
   - `k=8` runs showed residual orphan recovery (commonly around `15` in sampled runs),
   - `k=5` run also showed residual orphan recovery and non-trivial count variation (for example `23` observed), indicating no simple fixed terminal-slab scaling explanation.
7. startup artifact:
   - one-time startup recovery (`count=1, sample=[13]`) treated as separate from runtime adaptive-concurrent residual issue.

## Assumptions and Defaults
1. `legacy_missing_tokens` remains intentionally retained and not redefined in this phase.
2. Idle repair remains enabled by default as a guardrail.
3. Debug instrumentation remains disabled unless explicitly enabled (for example `SGLANG_MTP_KV_LEAK_DEBUG=true`).
4. User-run runtime tests/benchmarks remain the source of truth for observed behavior.
