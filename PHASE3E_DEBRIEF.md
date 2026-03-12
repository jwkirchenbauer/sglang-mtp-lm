# PHASE 3E Debrief: Backend Parity Gate for Offline `sglang` + 32B Next Plan

> Code-state anchor: `0023c1096`
> This debrief's results/analysis are tied to the implementation available at that commit.

## Summary
Phase 3E was a preliminary backend-parity gate before spending additional time on 32B model bring-up inside `lm_eval`.

Primary outcomes:
1. Completed a `limit=512` head-to-head comparison between the server `/generate` path (`lm_eval --model sglang-generate`) and the in-process/offline path (`lm_eval --model sglang`) for the main Phase3D MTP checkpoint.
2. Established that the two backends are close enough to proceed with 32B debugging and smoke evaluation.
3. Observed exact parity only at `k=1, c=1`; the other three tested cases showed small but non-zero accuracy deltas.
4. Confirmed that the large quality drop from `k=1` to `k=3` is not a server-only artifact; it reproduces on both backends.
5. Converted the parity result into a concrete 32B testing ladder centered on in-process `sglang` inside `lm_eval`.

## Checkpoint Context
1. Repository: `sglang-mtp-lm`
2. Branch at debrief finalization: `main`
3. Debrief code-state anchor commit: `0023c1096`
4. Main parity-tested MTP checkpoint:
   - [`/capstor/scratch/cscs/jkirchen/singleshot-root/singleshot/outputs/daint_prod_ift_mask_fix/daint_prod_ift_mask_fix_1N4n_9d30cad5/step-00100160`](/capstor/scratch/cscs/jkirchen/singleshot-root/singleshot/outputs/daint_prod_ift_mask_fix/daint_prod_ift_mask_fix_1N4n_9d30cad5/step-00100160)
5. Canonical Phase3E parity root:
   - [`outputs/phase3e_head2head_pairwise_exportall_limit512_20260312_051810`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3e_head2head_pairwise_exportall_limit512_20260312_051810)
6. Canonical summary source:
   - [`parity_summary.json`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3e_head2head_pairwise_exportall_limit512_20260312_051810/parity_summary.json)
7. Canonical server log:
   - [`server.log`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3e_head2head_pairwise_exportall_limit512_20260312_051810/server_backend/server.log)
8. Non-canonical / discarded attempts kept for provenance:
   - [`outputs/phase3e_head2head_20260312_040711`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3e_head2head_20260312_040711) as the earlier `limit=128` probe.
   - [`outputs/phase3e_head2head_limit512_20260312_043237`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3e_head2head_limit512_20260312_043237) as the interrupted sequential `limit=512` attempt.
   - [`outputs/phase3e_head2head_pairwise_limit512_20260312_051550`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3e_head2head_pairwise_limit512_20260312_051550) as the failed pairwise launch before fixing env export behavior.

## Why Phase3E Exists
Phase3E is a narrow gate between the measurement work in Phase3D and the next intended objective: testing larger checkpoints, including a 32B model, through the in-process `sglang` integration in `lm_eval`.

The goal here was not to reproduce the entire Phase3D Pareto dataset or match those end-to-end numbers exactly. The goal was narrower:
1. Hold the task, prompt style, and core MTP request semantics close to the Phase3D setup.
2. Compare `lm_eval --model sglang-generate` against `lm_eval --model sglang` directly.
3. Decide whether the in-process path is trustworthy enough to use as the main path for 32B debugging.

## Evaluation Protocol
1. Task family:
   - `gsm8k_cot_singleshot`
2. Run depth:
   - `limit=512`
3. Cases:
   - `k in {1, 3}`
   - `c in {1, 8}`
4. Shared MTP mask token:
   - `128259`
5. Shared prompt/eval style:
   - chat template enabled
   - fewshot as multiturn
6. Primary speed signal:
   - SGLang per-request decode metrics
   - server log decode summaries where available
7. Secondary speed signal:
   - client-observed elapsed walltime
8. Important caveat on the final usable `512` run:
   - it was executed on the `debug` partition as a pairwise server/offline comparison on separate GPUs inside one full-node allocation so the matrix would fit under the partition time cap.

## Public APIs / Interfaces / Types
None. Phase3E is a docs-only debrief phase. No repo API, type, or workflow code changes are part of this phase record.

## Final Results Snapshot
Canonical summary source:
[`parity_summary.json`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3e_head2head_pairwise_exportall_limit512_20260312_051810/parity_summary.json)

| case | server flex/strict | offline flex/strict | req decode p50 server/offline | elapsed s server/offline |
| --- | ---: | ---: | ---: | ---: |
| `k1_c1` | `0.6504 / 0.6387` | `0.6504 / 0.6387` | `386.1 / 388.6` | `505.67 / 506.21` |
| `k1_c8` | `0.6484 / 0.6348` | `0.6426 / 0.6289` | `348.5 / 358.2` | `127.79 / 136.55` |
| `k3_c1` | `0.5352 / 0.5020` | `0.5312 / 0.4980` | `964.5 / 978.8` | `175.11 / 196.98` |
| `k3_c8` | `0.5254 / 0.4883` | `0.5234 / 0.4844` | `846.1 / 853.6` | `54.23 / 82.82` |

Server batch decode p50 values derived from
[`server.log`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3e_head2head_pairwise_exportall_limit512_20260312_051810/server_backend/server.log):
1. `k1_c1`: `144.22 tok/s`
2. `k1_c8`: `759.845 tok/s`
3. `k3_c1`: `316.93 tok/s`
4. `k3_c8`: `1680.215 tok/s`

## Interpretation
1. `k=1, c=1` achieved exact parity on both Flex EM and Strict EM.
2. The remaining three cases showed small but real accuracy deltas:
   - `k=1, c=8`: server `+3/512` on both Flex and Strict.
   - `k=3, c=1`: server `+2/512` on both Flex and Strict.
   - `k=3, c=8`: server `+1/512` Flex and `+2/512` Strict.
3. Engine-side per-request decode throughput remained close across backends in all four cases.
4. The large quality drop from `k=1` to `k=3` is not a server/backend artifact; it appears on both the server and in-process paths.
5. The in-process/offline path did not emit the same batch decode summaries into `lm_eval.log` that the standalone server produced in `server.log`, so per-request metrics were the cleanest common metric for apples-to-apples speed comparison.
6. Because the final canonical run was pairwise on `debug`, the result is strong enough to proceed to 32B bring-up, but it should not be treated as the last word on backend determinism.

## Workflow Takeaways
This phase reinforced the Phase3D operational rules rather than replacing them.

1. Full-node allocations remain the required baseline on this system; do not try to treat this workflow as a single-GPU allocation pattern.
2. Activate the environment with `sslm_sgl_env` before the allocation, then carry that shell state into the Slurm work.
3. Do not attempt to "reactivate" inside the allocation with plain `conda` semantics; that is not the supported workflow.
4. If nested `srun` shells are needed, preserve the carried environment explicitly with `--export=ALL`.
5. This phase should be read as a direct extension of the Phase3D "Key Takeaways to Never Forget," not as a new workflow doctrine.

## 32B Next Plan
Target checkpoint:
[`step-00025040`](/capstor/scratch/cscs/jkirchen/singleshot-root/singleshot/outputs/daint_prod_large_models/daint_prod_large_models_16N64n_1f611930/step-00025040)

Important checkpoint note:
1. `pytorch_model.bin` is present in the checkpoint directory and is the intended HF-style load target for `pretrained=...`.
2. The directory also contains the expected HF-style config/modeling files needed for the load attempt.

Required 32B eval specifics:
1. Backend: in-process `lm_eval --model sglang`
2. Tensor parallel degree: `tp_size=4`
3. Prompt style: no chat template
4. MTP mask token: `151669`
5. EOS handling: `stop_token_ids=151645+151643`

Staged bring-up ladder:
1. Preflight load with `non_mtp`, `limit=32`, `batch_size=1`.
2. If that succeeds, run `mtp_k1` with static graph capture enabled for `k=1`.
3. If that succeeds, repeat `k=1` at `c=8`.
4. Only then try `k=3`.
5. Only after viability is established should prompt count / eval depth be increased beyond smoke depth.

Current command-shape reference:
[`phase3e_32b_offline_smoke.sh`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/_helpers/phase3e_32b_offline_smoke.sh)

Current helper-aligned launch assumptions:
1. `trust_remote_code=true`
2. `dtype=bfloat16`
3. `attention_backend=flashinfer`
4. `disable_overlap_schedule=true`
5. `mem_fraction_static=0.70`
6. `cuda_graph_max_bs=16`
7. `max_running_requests=16`

Fallback ladder if bring-up is unstable:
1. Lower `cuda_graph_max_bs` from `16` to `8`.
2. If still unstable, disable the MTP CUDA graph flags.
3. Lower `mem_fraction_static` from `0.70` to `0.66`, then to `0.62`.

Immediate extension once the helper succeeds:
1. Add `mtp_k3` to the same offline smoke flow after `non_mtp` and `mtp_k1` pass.

## Assumptions and Defaults
1. Phase3E is a docs-only debrief phase summarizing existing benchmark artifacts.
2. The canonical Phase3E parity artifact is the pairwise `debug`-partition `limit=512` run, and its caveat is intentionally preserved in the interpretation.
3. No code or workflow-script changes are part of this commit.
