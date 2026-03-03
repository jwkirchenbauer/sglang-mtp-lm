# PHASE 2C Debrief: Adaptive Concurrency Correctness Closure and Phase 3 Handoff

## Summary
Phase 2C closed the main adaptive decoding correctness failure under concurrency. The key closure was scheduler lifecycle correctness for adaptive mixed-shape decode (pause/resume instead of waiting/prefill bounce), plus resume-path batch state completion needed to avoid runtime crashes. Baseline and candidate both completed the 4-way `lm_eval` matrix with matching sample outputs and matching accuracy. What remains open is optimization work: adaptive CUDA graph strategy (and later overlap, if needed) plus scaling/performance characterization.

## Checkpoint Context
1. Repository: `sglang-mtp-lm`
2. Branch at closure: `main`
3. HEAD at closure capture: `5faca0a95`
4. Validation and debugging included iterative local edits and uncommitted working-tree state during bring-up; acceptance evidence is from explicit run artifacts listed below.

## Primary Outcome
1. Adaptive concurrency correctness issue was resolved:
   - prior failures (`Scheduler hit an exception`, ownership-gap style failures, pause/resume decode crashes) were closed.
2. Adaptive performance optimization remains open:
   - correctness is stable, but adaptive throughput is still behind static high-throughput behavior in tested conditions.

## Authoritative Evidence

## User-Run Observations (Current Session)
1. Candidate static `k=4` with high concurrency (`128`) is good and fast.
2. Candidate `conf_adapt` at high concurrency is correct but slow.
3. Candidate `conf_adapt` at concurrency `8` completes cleanly with no bad warning lines.

## Agent-Run Evidence (Phase 2C.2)
1. Quick-driver stabilization path:
   - iterative crash-repro and fixes on allocated nodes.
   - final focused concurrent validation passed gates and completed without scheduler exceptions.
   - quick open-ended text parity still showed mismatches and is treated as weaker evidence.
2. Full 4-way `lm_eval` matrix:
   - baseline single: exit `0`
   - candidate single: exit `0`
   - baseline concurrent=8: exit `0`
   - candidate concurrent=8: exit `0`
3. `lm_eval` sample parity:
   - single: baseline vs candidate mismatches = `0`
   - concurrent=8: baseline vs candidate mismatches = `0`
4. `lm_eval` metric parity:
   - all four scenarios: `exact_match,strict-match = 0.65625`
5. Log-gate status:
   - both baseline and candidate server logs: PASS in acceptance runs
   - startup singleton orphan recovery (`count=1`) tolerated by default gate policy; runtime orphan/overlap/repair signatures absent.

## Artifact Roots (Authoritative)
1. Quick and iterative bring-up artifacts:
   - `/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase2c2_full_20260303_071254`
2. Acceptance `lm_eval` 4-way artifacts:
   - `/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase2c2_lmeval_20260303_074212`

## Critical Learning #1: HF Source-of-Truth Must Stay Front-and-Center
1. This file must be reviewed at the start of Phase 3 and kept central during implementation:
   - [`modeling_llama.py`](/capstor/scratch/cscs/jkirchen/singleshot-root/singleshot/litgpt/transformers_local/llama/modeling_llama.py)
2. This remains the source of truth for:
   - decoding logic semantics,
   - abstract KV ownership/commit behavior,
   - adaptive flow intent that SGL runtime behavior should replicate.
3. Practical lesson:
   - keeping this HF reference explicit early would likely have reduced time-to-fix on core adaptive semantic/lifecycle issues.

## Critical Learning #2: Node-Allocated Agent-Run Debugging Was a Force Multiplier
1. The ability for the agent to allocate nodes, launch servers, run matrices, and debug from real logs materially accelerated closure.
2. Keep these docs as standard handoff anchors:
   - Generic node/env workflow: [`DEBUG_ENV_WORKFLOW.md`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/DEBUG_ENV_WORKFLOW.md)
   - Phase-specific reproducible matrix workflow: [`PHASE2C2_4WAY_TEST_WORKFLOW.md`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/PHASE2C2_4WAY_TEST_WORKFLOW.md)

## Critical Learning #3: `fixed_window` Was Explored, But Is Not the Main Bet
1. `fixed_window` strategy was explored during this phase.
2. Caveat:
   - treat `fixed_window` as provisional and not a default Phase 3 design commitment.
3. Preferred Phase 3 hypothesis:
   - adaptive CUDA graph friendliness should be achievable by compiling/replaying the required adaptive `q_len` set observed during runs.
4. Decision:
   - prioritize adaptive graph shape-set strategy before investing further in `fixed_window`.

## What to Trust Less on Restart
1. Quick-driver open-ended text parity mismatches are weaker correctness signals than `lm_eval` sample parity + metric parity.
2. One-time startup orphan singleton (`count=1`) and warmup extra HTTP-200 counts should not be treated as runtime correctness failures.
3. Short-run warmup throughput deltas should not drive architecture decisions for Phase 3.

## Phase 3 Entry Objectives

## Objective A (First-Class): Adaptive CUDA Graph Strategy for `hf_exact`
1. Build adaptive graph strategy around observed `q_len` shape set for `hf_exact`.
2. Keep correctness parity against eager as hard requirement.
3. Go/No-Go gate:
   - GO only if no scheduler exceptions, no runtime repair/orphan events, and no baseline/candidate sample or metric drift in acceptance matrix.
   - NO-GO if any ownership-gap signature, orphan/overlap repair event, or parity regression appears.

## Objective B: Scaling and Performance Characterization
1. Measure throughput/latency scaling across concurrency levels and `k` for static and adaptive.
2. Separate warmup from steady-state and preserve reproducibility controls.
3. Go/No-Go gate:
   - GO if adaptive graph path demonstrates stable correctness and meaningful speedup over eager adaptive.
   - NO-GO if speedup depends on unstable behavior or introduces correctness regressions.

## Objective C (Optional, Gated): Overlap + MTP Reintegration
1. Consider overlap only after Objective A correctness is stable.
2. Treat as secondary optimization, not required for initial adaptive graph closure.
3. Go/No-Go gate:
   - GO only after overlap path shows no correctness drift vs non-overlap under matched workloads.
   - NO-GO if overlap reintroduces lifecycle/KV ownership instability.

## Files/Areas Touched in 2C (Major)
1. `python/sglang/srt/managers/scheduler.py`
2. `python/sglang/srt/managers/schedule_batch.py`
3. `python/sglang/srt/managers/scheduler_output_processor_mixin.py`
4. `python/sglang/srt/managers/scheduler_metrics_mixin.py`
5. `python/sglang/srt/sampling/sampling_params.py`
6. `python/sglang/srt/model_executor/model_runner.py`
7. `python/sglang/srt/model_executor/forward_batch_info.py`
8. `python/sglang/srt/layers/logits_processor.py`
9. `python/sglang/srt/mem_cache/common.py`
10. `scripts/playground/mtp_log_gate.py`
11. `scripts/playground/mtp_quick_driver.py`
12. `scripts/playground/mtp_cudagraph_ab.py`
13. `DEBUG_ENV_WORKFLOW.md`
14. `PHASE2C2_4WAY_TEST_WORKFLOW.md`

## Test Cases and Scenarios (Documented Outcomes)

| Scenario | Server Mode | Traffic Mode | Result | Notable Warnings/Errors | Interpretation Strength |
|---|---|---|---|---|---|
| Static `k=4` high-concurrency run | candidate | concurrent=128 | PASS (user report) | None flagged | High |
| `conf_adapt` high-concurrency run | candidate | high concurrency | PASS (user report), slow | None flagged | High for correctness, Medium for performance interpretation |
| `conf_adapt` run | candidate | concurrent=8 | PASS (user report) | None flagged | High |
| Quick-driver matrix bring-up | baseline + candidate | single + concurrent=8 | Iterative stabilization; final concurrent gates PASS | Quick text parity mismatches can occur | Medium (debug signal) |
| `lm_eval` acceptance matrix | baseline + candidate | single + concurrent=8 | PASS (`exit 0` all four), sample parity `0` mismatches, metric parity `0.65625` all four | Startup singleton tolerated by gate policy | Very High |

## Assumptions and Defaults
1. Phase 2C debrief is cumulative across 2C.x work.
2. `hf_exact` is the adaptive closure mode carried into Phase 3.
3. `fixed_window` is not a required path for Phase 3 success.
4. Overlap remains out of scope until adaptive graph correctness is stable.
5. `lm_eval` parity and metrics are the primary acceptance signals; quick-driver is primarily a fast debug loop.

