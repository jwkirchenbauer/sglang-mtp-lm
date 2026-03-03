# Debug Environment Workflow (Interactive HPC)

This is the tested workflow for running debug commands in the same style as manual terminal usage on CSCS.

## Why this workflow

`sslm_sgl_env` and the `uenv`/conda chain depend on an interactive shell.  
Non-interactive one-shot shell execution may not load the function or may fail in `uenv`.

## Standard flow

1. Start an interactive login shell:
```bash
bash -li
```

2. Activate project environment:
```bash
sslm_sgl_env
```

3. Request an interactive allocation (1 task/node, 4 GPUs):
```bash
salloc --partition=debug --job-name=debug_sgl_server -t29 -N1 --ntasks-per-node=1 --gpus-per-node=4 -c18
```
If you need more than one "session", you can target individual GPUs on the same node for each session after the
allocation is live. Alternately, you can use the `normal` partition which allows runtimes up to 1440 mins, but
takes a lot longer to schedule (sometimes even for 1 or 2 hr requests) whereas debug should come back immediately.

4. Run a simple smoke test with `srun`:
```bash
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --gpus-per-node=4 \
  bash -lc 'echo SRUN_HOST=$(hostname); echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES; nvidia-smi -L | wc -l'
```

Expected shape of output:
- `SRUN_HOST=<allocated-node>`
- `CUDA_VISIBLE_DEVICES=0,1,2,3`
- `4` (GPU count)

5. Run real debug/test commands (e.g. project scripts).

6. Exit cleanly to release resources:
```bash
exit   # leave allocation shell
exit   # leave sslm_sgl_env shell
exit   # leave outer login shell (if opened for this session)
```

## Notes for automation agents

- Use a persistent PTY/TTY session, not one-shot non-interactive command execution.
- Send commands incrementally (as typed input), and poll for prompt/output between steps.
- Wait for `salloc` readiness message (`Nodes ... are ready for job`) before running `srun`.
- Entering a new `srun --pty bash -li` shell resets shell-scoped exports. Re-run `sslm_sgl_env` and re-export critical vars (`MODEL_PATH`, `RUN_ROOT`, debug flags) inside that shell.
- If you use `set -u`, unset vars will abort the active step. Prefer explicit `export ...` before long command blocks.
- `GET /health_generate` can return `503` until warmup is complete; poll with retries and also check server PID liveness to fail fast on crashes.
- For live introspection without interrupting the main interactive step, use overlapping probes such as:
  `srun --overlap --jobid "$SLURM_JOB_ID" --nodes=1 --ntasks=1 bash -lc '<diagnostic command>'`.
- Running one server per GPU is supported. Keep per-scenario artifacts isolated under a run root (for example `sglang-mtp-lm/outputs/<run_id>/...`) to simplify postmortem triage.
- The `debug` partition is intentionally short-lived and effectively singleton for this user; do not start multiple debug allocations in parallel.
- This singleton usage is intentional to cap resource expenditure and provide lightweight sandboxing during automated debugging.
- Always perform cleanup exits so the allocation is relinquished.
