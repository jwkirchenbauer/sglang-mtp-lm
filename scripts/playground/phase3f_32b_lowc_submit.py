#!/usr/bin/env python3
"""Submit and monitor the Phase3F 32B low-concurrency sweep."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


REPO_ROOT = Path("/capstor/scratch/cscs/jkirchen/sglang-mtp-lm")
DEFAULT_MODEL_PATH = Path(
    "/capstor/scratch/cscs/jkirchen/singleshot-root/singleshot/outputs/"
    "daint_prod_large_models/daint_prod_large_models_16N64n_1f611930/step-00025040"
)
WORKER_SCRIPT = REPO_ROOT / "scripts/playground/phase3f_32b_lowc_worker.sh"
METRICS_SCRIPT = REPO_ROOT / "scripts/playground/phase3d_lowc_metrics_summary.py"
PLOT_SCRIPT = REPO_ROOT / "scripts/playground/phase3d_pareto_plot_by_strategy.py"
DEBRIEF_PATH = REPO_ROOT / "PHASE3F_DEBRIEF.md"

FULL_CONCURRENCY = [1, 2, 4, 8, 16]
SMOKE_CONCURRENCY = [1]
PRECHECK_STRATEGIES = [
    "non_mtp",
    "static_k3",
    "conf_adapt_k8_t09",
]
FULL_STRATEGIES = [
    "non_mtp",
    "static_k1",
    "static_k2",
    "static_k3",
    "static_k4",
    "static_k5",
    "static_k8",
    "static_k16",
    "conf_adapt_k3_t09",
    "conf_adapt_k3_t06",
    "conf_adapt_k8_t09",
    "conf_adapt_k8_t06",
    "conf_adapt_k16_t09",
    "conf_adapt_k16_t06",
]
RUNTIME_PROFILES = [
    {"name": "cg16_mem070", "cuda_graph_max_bs": 16, "mem_fraction_static": "0.70"},
    {"name": "cg8_mem070", "cuda_graph_max_bs": 8, "mem_fraction_static": "0.70"},
    {"name": "cg8_mem066", "cuda_graph_max_bs": 8, "mem_fraction_static": "0.66"},
    {"name": "cg8_mem062", "cuda_graph_max_bs": 8, "mem_fraction_static": "0.62"},
]
TERMINAL_PREFIXES = (
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
)
PENDING_PREFIXES = ("PENDING",)
RUNNING_PREFIXES = ("CONFIGURING", "COMPLETING", "RUNNING", "STAGE_OUT")
CONDA_ENV = "/capstor/scratch/cscs/jkirchen/daint_291_129_singleshot"
UENV_VIEW = "prgenv-gnu/25.11:v1"
MODULE_LINE = "ml aws-ofi-nccl cuda nccl libfabric"
DEBUG_TIME_LIMIT_MAX_MINUTES = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True, default=None)
    parser.add_argument("--mask-id", type=int, required=True)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Run root. Default: outputs/phase3f_32b_lowc_robust_<timestamp>/",
    )
    parser.add_argument(
        "--resume-run-root",
        type=Path,
        default=None,
        help="Resume an existing Phase3F run root in place.",
    )
    parser.add_argument("--partition-normal", type=str, default="normal")
    parser.add_argument("--partition-debug", type=str, default="debug")
    parser.add_argument("--limit-full", type=int, default=512)
    parser.add_argument("--limit-smoke", type=int, default=32)
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=FULL_CONCURRENCY,
        help="Full-sweep concurrency list. Default: 1 2 4 8 16",
    )
    parser.add_argument("--max-live-jobs", type=int, default=3)
    parser.add_argument("--debug-mirror-slots", type=int, default=1)
    parser.add_argument("--max-resubmit-rounds", type=int, default=2)
    parser.add_argument("--minutes-preflight", type=int, default=45)
    parser.add_argument("--minutes-full", type=int, default=180)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--mirror-after-seconds", type=int, default=60)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--attention-backend", type=str, default="flashinfer")
    parser.add_argument("--max-gen-toks", type=int, default=128)
    parser.add_argument("--max-running-requests", type=int, default=16)
    parser.add_argument("--stop-token-ids", type=str, default="151645+151643")
    parser.add_argument("--visible-devices", type=str, default="0,1,2,3")
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the preflight gate and use the first runtime profile directly.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write launcher scripts and manifests without submitting jobs.",
    )
    return parser.parse_args()


def ts_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def strategy_slug(strategy: str) -> str:
    return strategy.replace(".", "p").replace("+", "plus").replace("/", "_")


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_clean_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)


def format_walltime(minutes: int) -> str:
    total = max(1, int(minutes)) * 60
    hours, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d}"


def parse_time_token(raw: str) -> Optional[dt.datetime]:
    token = raw.strip()
    if not token or token in {"N/A", "None", "Unknown", "Unknown/Invalid"}:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
        try:
            parsed = dt.datetime.strptime(token, fmt)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=dt.timezone.utc)
            return parsed.astimezone(dt.timezone.utc)
        except ValueError:
            continue
    return None


def state_is_pending(state: Optional[str]) -> bool:
    return bool(state) and state.startswith(PENDING_PREFIXES)


def state_is_terminal(state: Optional[str]) -> bool:
    return bool(state) and state.startswith(TERMINAL_PREFIXES)


def state_is_running(state: Optional[str]) -> bool:
    return bool(state) and state.startswith(RUNNING_PREFIXES)


def state_is_success(state: Optional[str]) -> bool:
    return bool(state) and state.startswith("COMPLETED")


def attempt_sort_key(attempt: Dict[str, Any]) -> tuple[float, float]:
    start = parse_time_token(str(attempt.get("start_time") or ""))
    start_ts = start.timestamp() if start is not None else float("inf")
    submitted_ts = float(attempt.get("submitted_ts") or 0.0)
    return (start_ts, submitted_ts)


def read_status(attempt_dir: Path) -> str:
    status_file = attempt_dir / "status.txt"
    if not status_file.exists():
        return "missing"
    return status_file.read_text(encoding="utf-8").strip() or "missing"


def record_event(target: Dict[str, Any], event: str, **kwargs: Any) -> None:
    target.setdefault("events", []).append({"ts": iso_now(), "event": event, **kwargs})


def build_stage_jobs(
    *,
    stage_root: Path,
    stage_name: str,
    strategies: Sequence[str],
    concurrency: Sequence[int],
    limit: int,
    minutes: int,
    port_base: int,
) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    stage_root = stage_root.resolve()
    (stage_root / "shards").mkdir(parents=True, exist_ok=True)
    (stage_root / "_attempts").mkdir(parents=True, exist_ok=True)
    for idx, strategy in enumerate(strategies):
        slug = strategy_slug(strategy)
        jobs.append(
            {
                "stage_name": stage_name,
                "strategy": strategy,
                "slug": slug,
                "job_index": idx,
                "canonical_shard_dir": str(stage_root / "shards" / slug),
                "attempts_root": str(stage_root / "_attempts" / slug),
                "port": int(port_base + idx),
                "limit": int(limit),
                "concurrency": [int(x) for x in concurrency],
                "minutes": int(minutes),
                "current_round": 0,
                "current_winner_partition": None,
                "current_winner_job_id": None,
                "winner_partition": None,
                "winner_job_id": None,
                "winner_attempt_dir": None,
                "final_status": "pending",
                "done": False,
                "success": False,
                "attempts": [],
                "events": [],
            }
        )
    return jobs


def build_worker_args(
    *,
    job: Dict[str, Any],
    attempt_dir: Path,
    profile: Dict[str, Any],
    args: argparse.Namespace,
) -> List[str]:
    return [
        "bash",
        str(WORKER_SCRIPT),
        "--strategy",
        str(job["strategy"]),
        "--shard-dir",
        str(attempt_dir),
        "--port",
        str(job["port"]),
        "--model-path",
        str(args.model_path),
        "--mask-id",
        str(args.mask_id),
        "--task",
        "gsm8k_cot_singleshot",
        "--limit",
        str(job["limit"]),
        "--concurrency",
        ",".join(str(x) for x in job["concurrency"]),
        "--max-gen-toks",
        str(args.max_gen_toks),
        "--tp-size",
        str(args.tp_size),
        "--dtype",
        str(args.dtype),
        "--attention-backend",
        str(args.attention_backend),
        "--mem-fraction-static",
        str(profile["mem_fraction_static"]),
        "--cuda-graph-max-bs",
        str(profile["cuda_graph_max_bs"]),
        "--max-running-requests",
        str(args.max_running_requests),
        "--stop-token-ids",
        str(args.stop_token_ids),
        "--visible-devices",
        str(args.visible_devices),
    ]


def build_run_name(run_tag: str, job: Dict[str, Any], round_idx: int, partition: str) -> str:
    stage_tag = "pf" if job["stage_name"] == "preflight" else "fw"
    slug = str(job["slug"])[:10]
    part_tag = "n" if partition == "normal" else "d"
    return f"p3f{run_tag}_{stage_tag}{int(job['job_index']):02d}_r{round_idx}_{part_tag}_{slug}"


def build_sbatch_script(
    *,
    job: Dict[str, Any],
    round_idx: int,
    partition: str,
    profile: Dict[str, Any],
    args: argparse.Namespace,
    run_tag: str,
) -> Dict[str, Any]:
    attempt_root = Path(str(job["attempts_root"]))
    attempt_dir = attempt_root / f"round{round_idx}_{partition}"
    launcher_dir = attempt_dir / "launcher"
    launcher_dir.mkdir(parents=True, exist_ok=True)
    run_name = build_run_name(run_tag=run_tag, job=job, round_idx=round_idx, partition=partition)
    actual_partition = (
        str(args.partition_debug) if partition == "debug" else str(args.partition_normal)
    )
    requested_minutes = int(job["minutes"])
    if actual_partition == str(args.partition_debug):
        requested_minutes = min(requested_minutes, DEBUG_TIME_LIMIT_MAX_MINUTES)
    script_path = launcher_dir / f"{run_name}.sbatch"
    slurm_output = attempt_dir / "slurm-%j.out"
    worker_cmd = " ".join(
        subprocess.list2cmdline([tok]) if False else tok for tok in []
    )
    del worker_cmd
    worker_line = " ".join(
        __import__("shlex").quote(tok)
        for tok in build_worker_args(job=job, attempt_dir=attempt_dir, profile=profile, args=args)
    )
    inner_cmd = "; ".join(
        [
            "set -euo pipefail",
            "source ~/.bashrc",
            MODULE_LINE,
            f"conda_activate {__import__('shlex').quote(CONDA_ENV)}",
            worker_line,
        ]
    )
    script = "\n".join(
        [
            "#!/usr/bin/env bash",
            f"#SBATCH --job-name={run_name}",
            f"#SBATCH --output={slurm_output}",
            f"#SBATCH --error={slurm_output}",
            f"#SBATCH --partition={actual_partition}",
            f"#SBATCH --time={format_walltime(requested_minutes)}",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks-per-node=1",
            "#SBATCH --gpus-per-node=4",
            "#SBATCH --cpus-per-task=18",
            "",
            "set -euo pipefail",
            "echo HOSTNAME=$(hostname)",
            f"echo RUN_NAME={run_name}",
            f"echo PARTITION={actual_partition}",
            f"echo PROFILE={profile['name']}",
            f"echo MODEL_PATH={args.model_path}",
            f"echo ATTEMPT_DIR={attempt_dir}",
            "date -u",
            f"uenv run --view=modules {UENV_VIEW} -- bash -lc {__import__('shlex').quote(inner_cmd)}",
            "",
        ]
    )
    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(0o755)
    return {
        "round": int(round_idx),
        "partition": partition,
        "partition_requested": actual_partition,
        "role": "primary" if partition == "normal" else "mirror",
        "job_name": run_name,
        "attempt_dir": str(attempt_dir),
        "script_path": str(script_path),
        "slurm_output_pattern": str(slurm_output),
        "minutes_requested": requested_minutes,
        "profile_name": str(profile["name"]),
        "cuda_graph_max_bs": int(profile["cuda_graph_max_bs"]),
        "mem_fraction_static": str(profile["mem_fraction_static"]),
    }


def submit_sbatch(script_path: Path) -> tuple[int, str, str, Optional[int]]:
    proc = subprocess.run(
        ["sbatch", "--parsable", str(script_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    out = proc.stdout.strip()
    err = proc.stderr.strip()
    job_id: Optional[int] = None
    if proc.returncode == 0 and out:
        token = out.split(";", 1)[0].strip()
        if token.isdigit():
            job_id = int(token)
    return proc.returncode, out, err, job_id


def cancel_job(job_id: int) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["scancel", str(job_id)],
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def query_job_info(job_id: int) -> Dict[str, Optional[str]]:
    squeue = subprocess.run(
        ["squeue", "-h", "-j", str(job_id), "-o", "%A|%T|%P|%S"],
        check=False,
        capture_output=True,
        text=True,
    )
    if squeue.returncode == 0:
        for line in squeue.stdout.splitlines():
            parts = [x.strip() for x in line.split("|")]
            if len(parts) >= 4 and parts[0].isdigit() and int(parts[0]) == int(job_id):
                return {
                    "state": parts[1],
                    "partition": parts[2] or None,
                    "start_time": parts[3] or None,
                    "end_time": None,
                }

    sacct = subprocess.run(
        [
            "sacct",
            "-P",
            "-n",
            "-j",
            str(job_id),
            "--format=JobIDRaw,State,Partition,Start,End",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if sacct.returncode == 0:
        for line in sacct.stdout.splitlines():
            parts = [x.strip() for x in line.split("|")]
            if len(parts) < 5:
                continue
            if parts[0] != str(job_id):
                continue
            return {
                "state": parts[1] or None,
                "partition": parts[2] or None,
                "start_time": parts[3] or None,
                "end_time": parts[4] or None,
            }

    return {"state": None, "partition": None, "start_time": None, "end_time": None}


def refresh_attempt(attempt: Dict[str, Any]) -> None:
    job_id = attempt.get("job_id")
    if not job_id:
        return
    info = query_job_info(int(job_id))
    if info["state"]:
        attempt["state"] = info["state"]
    if info["partition"]:
        attempt["partition_resolved"] = info["partition"]
    if info["start_time"]:
        attempt["start_time"] = info["start_time"]
    if info["end_time"]:
        attempt["end_time"] = info["end_time"]


def current_round_attempts(job: Dict[str, Any]) -> List[Dict[str, Any]]:
    round_idx = int(job["current_round"])
    return [a for a in job["attempts"] if int(a["round"]) == round_idx]


def attempt_by_partition(job: Dict[str, Any], partition: str) -> Optional[Dict[str, Any]]:
    for attempt in current_round_attempts(job):
        if attempt["partition"] == partition:
            return attempt
    return None


def choose_winner(job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    attempts = current_round_attempts(job)
    if not attempts:
        return None
    active = [a for a in attempts if not state_is_pending(str(a.get("state") or ""))]
    if not active:
        return None
    active.sort(key=attempt_sort_key)
    return active[0]


def request_cancel(attempt: Optional[Dict[str, Any]], manifest: Dict[str, Any]) -> None:
    if attempt is None:
        return
    if attempt.get("cancel_requested_at"):
        return
    if state_is_terminal(str(attempt.get("state") or "")):
        return
    job_id = attempt.get("job_id")
    if not job_id:
        return
    rc, out, err = cancel_job(int(job_id))
    attempt["cancel_requested_at"] = iso_now()
    attempt["cancel_returncode"] = rc
    attempt["cancel_stdout"] = out
    attempt["cancel_stderr"] = err
    record_event(attempt, "cancel_requested", returncode=rc)
    if rc == 0:
        attempt["cancelled"] = True
    write_json(Path(str(manifest["manifest_path"])), manifest)


def promote_winner(job: Dict[str, Any], attempt: Dict[str, Any]) -> None:
    canonical = Path(str(job["canonical_shard_dir"]))
    canonical.parent.mkdir(parents=True, exist_ok=True)
    ensure_clean_path(canonical)
    canonical.symlink_to(Path(str(attempt["attempt_dir"])).resolve(), target_is_directory=True)


def prepare_next_round(job: Dict[str, Any]) -> None:
    job["current_round"] = int(job["current_round"]) + 1
    job["current_winner_partition"] = None
    job["current_winner_job_id"] = None


def active_attempt_count(jobs: Sequence[Dict[str, Any]], partition: str) -> int:
    count = 0
    for job in jobs:
        if bool(job.get("done")):
            continue
        for attempt in current_round_attempts(job):
            if attempt["partition"] != partition:
                continue
            if not state_is_terminal(str(attempt.get("state") or "")):
                count += 1
    return count


def stage_debug_mirror_slots(stage_name: str, args: argparse.Namespace) -> int:
    if stage_name != "preflight":
        return 0
    return int(args.debug_mirror_slots)


def submit_attempt_if_needed(
    *,
    job: Dict[str, Any],
    round_idx: int,
    partition: str,
    profile: Dict[str, Any],
    args: argparse.Namespace,
    run_tag: str,
    manifest: Dict[str, Any],
) -> bool:
    if attempt_by_partition(job, partition) is not None:
        return False

    attempt = build_sbatch_script(
        job=job,
        round_idx=round_idx,
        partition=partition,
        profile=profile,
        args=args,
        run_tag=run_tag,
    )
    attempt["submitted_at"] = iso_now()
    attempt["submitted_ts"] = time.time()
    attempt["state"] = "DRY_RUN" if args.dry_run else "SUBMITTING"
    attempt["partition_resolved"] = partition
    attempt["events"] = []
    record_event(attempt, "script_written")

    if args.dry_run:
        job["attempts"].append(attempt)
        record_event(job, "dry_run_attempt", partition=partition, round=round_idx)
        write_json(Path(str(manifest["manifest_path"])), manifest)
        return True

    rc, out, err, job_id = submit_sbatch(Path(str(attempt["script_path"])))
    if rc != 0 or job_id is None:
        record_event(
            job,
            "submit_failed",
            partition=partition,
            round=round_idx,
            returncode=rc,
            stdout=out,
            stderr=err,
        )
        write_json(Path(str(manifest["manifest_path"])), manifest)
        return False

    attempt["submit_returncode"] = rc
    attempt["submit_stdout"] = out
    attempt["submit_stderr"] = err
    attempt["job_id"] = int(job_id)
    attempt["state"] = "PENDING"
    attempt["slurm_output"] = str(Path(str(attempt["attempt_dir"])) / f"slurm-{job_id}.out")
    record_event(attempt, "submitted", job_id=job_id)
    job["attempts"].append(attempt)
    record_event(job, "submitted", partition=partition, round=round_idx, job_id=job_id)
    write_json(Path(str(manifest["manifest_path"])), manifest)
    return True


def maybe_submit_debug_mirror(
    *,
    stage_name: str,
    jobs: Sequence[Dict[str, Any]],
    profile: Dict[str, Any],
    args: argparse.Namespace,
    run_tag: str,
    manifest: Dict[str, Any],
) -> bool:
    if stage_debug_mirror_slots(stage_name, args) <= 0:
        return False
    active_debug = active_attempt_count(jobs, "debug")
    if active_debug >= stage_debug_mirror_slots(stage_name, args):
        return False
    total_live = active_attempt_count(jobs, "normal") + active_debug
    if total_live >= int(args.max_live_jobs):
        return False

    candidates: List[Dict[str, Any]] = []
    now = time.time()
    for job in jobs:
        if bool(job.get("done")):
            continue
        if job.get("current_winner_partition") is not None:
            continue
        normal = attempt_by_partition(job, "normal")
        debug = attempt_by_partition(job, "debug")
        if normal is None or debug is not None:
            continue
        state = str(normal.get("state") or "")
        if not state_is_pending(state):
            continue
        if now - float(normal.get("submitted_ts") or 0.0) < int(args.mirror_after_seconds):
            continue
        candidates.append(job)

    if not candidates:
        return False

    candidates.sort(
        key=lambda j: float(
            (attempt_by_partition(j, "normal") or {}).get("submitted_ts") or 0.0
        )
    )
    return submit_attempt_if_needed(
        job=candidates[0],
        round_idx=int(candidates[0]["current_round"]),
        partition="debug",
        profile=profile,
        args=args,
        run_tag=run_tag,
        manifest=manifest,
    )


def maybe_submit_primary_jobs(
    *,
    stage_name: str,
    jobs: Sequence[Dict[str, Any]],
    profile: Dict[str, Any],
    args: argparse.Namespace,
    run_tag: str,
    manifest: Dict[str, Any],
) -> bool:
    changed = False
    max_primary_jobs = max(1, int(args.max_live_jobs) - stage_debug_mirror_slots(stage_name, args))
    while True:
        active_primary = active_attempt_count(jobs, "normal")
        active_total = active_primary + active_attempt_count(jobs, "debug")
        if active_primary >= max_primary_jobs or active_total >= int(args.max_live_jobs):
            break

        target: Optional[Dict[str, Any]] = None
        for job in jobs:
            if bool(job.get("done")):
                continue
            if attempt_by_partition(job, "normal") is not None:
                continue
            target = job
            break
        if target is None:
            break
        if not submit_attempt_if_needed(
            job=target,
            round_idx=int(target["current_round"]),
            partition="normal",
            profile=profile,
            args=args,
            run_tag=run_tag,
            manifest=manifest,
        ):
            break
        changed = True
    return changed


def refresh_jobs(jobs: Sequence[Dict[str, Any]]) -> None:
    for job in jobs:
        for attempt in current_round_attempts(job):
            if state_is_terminal(str(attempt.get("state") or "")):
                continue
            refresh_attempt(attempt)


def all_done(jobs: Sequence[Dict[str, Any]]) -> bool:
    return all(bool(job.get("done")) for job in jobs)


def complete_job_success(job: Dict[str, Any], winner: Dict[str, Any], manifest: Dict[str, Any]) -> None:
    promote_winner(job, winner)
    job["done"] = True
    job["success"] = True
    job["final_status"] = "ok"
    job["winner_partition"] = winner.get("partition_requested") or winner["partition"]
    job["winner_job_id"] = winner.get("job_id")
    job["winner_attempt_dir"] = winner["attempt_dir"]
    record_event(
        job,
        "completed",
        winner_partition=winner.get("partition_requested") or winner["partition"],
        winner_role=winner["partition"],
        job_id=winner.get("job_id"),
    )
    write_json(Path(str(manifest["manifest_path"])), manifest)


def complete_job_failure(job: Dict[str, Any], reason: str, manifest: Dict[str, Any]) -> None:
    job["done"] = True
    job["success"] = False
    job["final_status"] = reason
    record_event(job, "failed", reason=reason)
    write_json(Path(str(manifest["manifest_path"])), manifest)


def maybe_advance_job(job: Dict[str, Any], args: argparse.Namespace, manifest: Dict[str, Any]) -> bool:
    changed = False
    attempts = current_round_attempts(job)
    if not attempts:
        return False

    if job.get("current_winner_partition") is None:
        winner = choose_winner(job)
        if winner is not None:
            job["current_winner_partition"] = winner["partition"]
            job["current_winner_job_id"] = winner.get("job_id")
            record_event(
                job,
                "winner_selected",
                partition=winner["partition"],
                job_id=winner.get("job_id"),
                round=job["current_round"],
            )
            for attempt in attempts:
                if attempt is winner:
                    continue
                request_cancel(attempt, manifest)
            changed = True

    winner_partition = job.get("current_winner_partition")
    winner = attempt_by_partition(job, str(winner_partition)) if winner_partition else None
    if winner is not None:
        for attempt in attempts:
            if attempt is winner:
                continue
            request_cancel(attempt, manifest)
        if state_is_terminal(str(winner.get("state") or "")):
            status = read_status(Path(str(winner["attempt_dir"])))
            if state_is_success(str(winner.get("state") or "")) and status == "ok":
                complete_job_success(job, winner, manifest)
                return True
            if int(job["current_round"]) < int(args.max_resubmit_rounds):
                record_event(
                    job,
                    "resubmit",
                    round=job["current_round"],
                    winner_partition=winner["partition"],
                    state=winner.get("state"),
                    status_file=status,
                )
                prepare_next_round(job)
                write_json(Path(str(manifest["manifest_path"])), manifest)
                return True
            complete_job_failure(
                job,
                reason=f"winner_{winner.get('state')}_{status}",
                manifest=manifest,
            )
            return True
        return changed

    if all(state_is_terminal(str(a.get("state") or "")) for a in attempts):
        successes = [
            a
            for a in attempts
            if state_is_success(str(a.get("state") or ""))
            and read_status(Path(str(a["attempt_dir"]))) == "ok"
        ]
        if successes:
            successes.sort(key=attempt_sort_key)
            job["current_winner_partition"] = successes[0]["partition"]
            job["current_winner_job_id"] = successes[0].get("job_id")
            complete_job_success(job, successes[0], manifest)
            return True
        if int(job["current_round"]) < int(args.max_resubmit_rounds):
            record_event(job, "resubmit", round=job["current_round"], reason="all_attempts_terminal")
            prepare_next_round(job)
            write_json(Path(str(manifest["manifest_path"])), manifest)
            return True
        complete_job_failure(job, reason="all_attempts_terminal", manifest=manifest)
        return True

    return changed


def run_metrics(stage_root: Path) -> int:
    proc = subprocess.run(
        [sys.executable, str(METRICS_SCRIPT), "--run-root", str(stage_root)],
        check=False,
    )
    return int(proc.returncode)


def run_plot(stage_root: Path) -> int:
    input_tsv = stage_root / "summary" / "lowc_metrics_complete.tsv"
    output_png = stage_root / "summary" / "lowc_pareto_by_strategy_32b_no_eagle3.png"
    proc = subprocess.run(
        [
            sys.executable,
            str(PLOT_SCRIPT),
            "--input-tsv",
            str(input_tsv),
            "--output-png",
            str(output_png),
            "--title",
            "32B Low-Concurrency Pareto Frontier (No Eagle3)",
        ],
        check=False,
    )
    return int(proc.returncode)


def write_debrief(run_root: Path, args: argparse.Namespace, profile: Dict[str, Any]) -> None:
    summary_root = run_root / "summary"
    debrief = "\n".join(
        [
            "# PHASE 3F Debrief: 32B Low-Concurrency Pareto Recreation",
            "",
            f"> Generated on {dt.date.today().isoformat()} from the Phase3F submitter.",
            "",
            "## Summary",
            "1. Recreated the Phase3D low-concurrency analysis shape for the 32B final checkpoint.",
            "2. Used `chat_off` only and removed Eagle3 from the strategy set.",
            "3. Produced the canonical Phase3D-style summary tables plus the final Pareto plot.",
            "",
            "## Checkpoint Context",
            f"1. Checkpoint: `{args.model_path}`",
            f"2. MTP mask id: `{args.mask_id}`",
            f"3. Runtime profile: `{profile['name']}` (`cuda_graph_max_bs={profile['cuda_graph_max_bs']}`, `mem_fraction_static={profile['mem_fraction_static']}`)",
            "4. Prompt mode: `chat_off`",
            "5. Backend: standalone server + `lm_eval --model sglang-generate`",
            "",
            "## Canonical Artifacts",
            f"1. Run root: [`{run_root}`]({run_root})",
            f"2. Complete table: [`{summary_root / 'lowc_metrics_complete.tsv'}`]({summary_root / 'lowc_metrics_complete.tsv'})",
            f"3. Pareto points: [`{summary_root / 'pareto_points.tsv'}`]({summary_root / 'pareto_points.tsv'})",
            f"4. Pareto plot: [`{summary_root / 'lowc_pareto_by_strategy_32b_no_eagle3.png'}`]({summary_root / 'lowc_pareto_by_strategy_32b_no_eagle3.png'})",
            "",
            "## Notes",
            "1. Strategy coverage matches Phase3D minus Eagle3: `non_mtp`, `static_k{1,2,3,4,5,8,16}`, `conf_adapt_k{3,8,16}_t{06,09}`.",
            "2. Concurrency coverage is `c={1,2,4,8,16}` at `limit=512`.",
            "3. The Phase3D summary and plotting scripts were reused unchanged by keeping the shard layout compatible.",
            "",
        ]
    )
    DEBRIEF_PATH.write_text(debrief + "\n", encoding="utf-8")


def execute_stage(
    *,
    stage_name: str,
    stage_root: Path,
    jobs: List[Dict[str, Any]],
    profile: Dict[str, Any],
    args: argparse.Namespace,
    manifest: Dict[str, Any],
    resume_stage_record: Optional[Dict[str, Any]] = None,
) -> bool:
    stage_root.mkdir(parents=True, exist_ok=True)
    (stage_root / "summary").mkdir(parents=True, exist_ok=True)
    run_tag = re.sub(r"[^a-zA-Z0-9]", "", stage_root.name)[-8:].lower() or "p3f"
    if resume_stage_record is None:
        stage_record = {
            "stage_name": stage_name,
            "stage_root": str(stage_root),
            "profile": dict(profile),
            "jobs": jobs,
            "status": "dry_run" if args.dry_run else "running",
            "started_at": iso_now(),
            "events": [],
        }
        manifest.setdefault("stages", []).append(stage_record)
    else:
        stage_record = resume_stage_record
        stage_record["status"] = "dry_run" if args.dry_run else "running"
        stage_record["resumed_at"] = iso_now()
        stage_record["profile"] = dict(profile)
        stage_record["jobs"] = jobs
    write_json(Path(str(manifest["manifest_path"])), manifest)

    if args.dry_run:
        for job in jobs:
            submit_attempt_if_needed(
                job=job,
                round_idx=int(job["current_round"]),
                partition="normal",
                profile=profile,
                args=args,
                run_tag=run_tag,
                manifest=manifest,
            )
        record_event(stage_record, "dry_run_enumerated", shard_count=len(jobs))
        write_json(Path(str(manifest["manifest_path"])), manifest)
        return True

    while not all_done(jobs):
        refresh_jobs(jobs)
        progressed = False

        for job in jobs:
            if bool(job.get("done")):
                continue
            if maybe_advance_job(job, args, manifest):
                progressed = True

        if maybe_submit_debug_mirror(
            stage_name=stage_name,
            jobs=jobs,
            profile=profile,
            args=args,
            run_tag=run_tag,
            manifest=manifest,
        ):
            progressed = True

        if maybe_submit_primary_jobs(
            stage_name=stage_name,
            jobs=jobs,
            profile=profile,
            args=args,
            run_tag=run_tag,
            manifest=manifest,
        ):
            progressed = True

        if not all_done(jobs):
            if progressed:
                write_json(Path(str(manifest["manifest_path"])), manifest)
            time.sleep(max(5, int(args.poll_seconds)))

    success = all(bool(job.get("success")) for job in jobs)
    stage_record["status"] = "ok" if success else "failed"
    stage_record["finished_at"] = iso_now()
    metrics_rc = run_metrics(stage_root)
    stage_record["metrics_returncode"] = metrics_rc
    plot_rc: Optional[int] = None
    if metrics_rc == 0:
        plot_rc = run_plot(stage_root)
    stage_record["plot_returncode"] = plot_rc
    write_json(Path(str(manifest["manifest_path"])), manifest)
    return success


def validate_args(args: argparse.Namespace) -> None:
    if args.resume_run_root is not None and args.run_root is not None:
        raise ValueError("--run-root and --resume-run-root are mutually exclusive")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    if not WORKER_SCRIPT.exists():
        raise FileNotFoundError(f"Worker script not found: {WORKER_SCRIPT}")
    if not METRICS_SCRIPT.exists():
        raise FileNotFoundError(f"Metrics script not found: {METRICS_SCRIPT}")
    if not PLOT_SCRIPT.exists():
        raise FileNotFoundError(f"Plot script not found: {PLOT_SCRIPT}")
    if int(args.max_live_jobs) < 1:
        raise ValueError("--max-live-jobs must be >= 1")
    if int(args.debug_mirror_slots) < 0:
        raise ValueError("--debug-mirror-slots must be >= 0")
    if int(args.debug_mirror_slots) >= int(args.max_live_jobs):
        raise ValueError("--debug-mirror-slots must be less than --max-live-jobs")
    if int(args.limit_full) <= 0 or int(args.limit_smoke) <= 0:
        raise ValueError("--limit-full and --limit-smoke must be > 0")
    if int(args.max_resubmit_rounds) < 0:
        raise ValueError("--max-resubmit-rounds must be >= 0")
    if int(args.mirror_after_seconds) < 0:
        raise ValueError("--mirror-after-seconds must be >= 0")
    conc = [int(x) for x in args.concurrency]
    if not conc or any(x <= 0 for x in conc):
        raise ValueError("--concurrency values must all be > 0")
    if len(set(conc)) != len(conc):
        raise ValueError("--concurrency values must be unique")


def main() -> int:
    args = parse_args()
    if args.model_path is None:
        args.model_path = DEFAULT_MODEL_PATH
    validate_args(args)

    if args.resume_run_root is not None:
        run_root = args.resume_run_root.resolve()
        manifest_path = run_root / "phase3f_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found for resume: {manifest_path}")
        manifest = read_json(manifest_path)
        manifest["manifest_path"] = str(manifest_path)
        manifest["run_root"] = str(run_root)
        if not manifest.get("stages"):
            raise ValueError("Resume manifest has no stages to continue.")
        stage_record = manifest["stages"][-1]
        if stage_record.get("status") == "ok":
            return 0
        if stage_record.get("status") == "failed":
            raise ValueError("Cannot resume a failed stage without manual cleanup.")

        stage_name = str(stage_record["stage_name"])
        stage_root = Path(str(stage_record["stage_root"]))
        jobs = list(stage_record["jobs"])
        profile = dict(stage_record["profile"])

        ok = execute_stage(
            stage_name=stage_name,
            stage_root=stage_root,
            jobs=jobs,
            profile=profile,
            args=args,
            manifest=manifest,
            resume_stage_record=stage_record,
        )
        manifest["chosen_profile"] = dict(profile)
        manifest["final_status"] = "ok" if ok else "failed"
        write_json(manifest_path, manifest)

        if ok and not args.dry_run and stage_name == "full":
            write_debrief(run_root=run_root, args=args, profile=profile)
        return 0 if ok else 1

    run_root = args.run_root
    if run_root is None:
        run_root = REPO_ROOT / "outputs" / f"phase3f_32b_lowc_robust_{ts_tag()}"
    run_root = run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    manifest_path = run_root / "phase3f_manifest.json"
    manifest: Dict[str, Any] = {
        "manifest_path": str(manifest_path),
        "run_root": str(run_root),
        "created_at": iso_now(),
        "model_path": str(args.model_path),
        "mask_id": int(args.mask_id),
        "partition_normal": str(args.partition_normal),
        "partition_debug": str(args.partition_debug),
        "limit_full": int(args.limit_full),
        "limit_smoke": int(args.limit_smoke),
        "concurrency": [int(x) for x in args.concurrency],
        "smoke_concurrency": list(SMOKE_CONCURRENCY),
        "precheck_strategies": list(PRECHECK_STRATEGIES),
        "full_strategies": list(FULL_STRATEGIES),
        "runtime_profiles": list(RUNTIME_PROFILES),
        "dry_run": bool(args.dry_run),
        "skip_preflight": bool(args.skip_preflight),
        "stages": [],
    }
    write_json(manifest_path, manifest)

    chosen_profile: Optional[Dict[str, Any]] = None

    if not args.skip_preflight:
        for profile in RUNTIME_PROFILES:
            preflight_root = run_root / "preflight" / str(profile["name"])
            jobs = build_stage_jobs(
                stage_root=preflight_root,
                stage_name="preflight",
                strategies=PRECHECK_STRATEGIES,
                concurrency=SMOKE_CONCURRENCY,
                limit=int(args.limit_smoke),
                minutes=int(args.minutes_preflight),
                port_base=34100,
            )
            ok = execute_stage(
                stage_name="preflight",
                stage_root=preflight_root,
                jobs=jobs,
                profile=profile,
                args=args,
                manifest=manifest,
            )
            if args.dry_run:
                chosen_profile = dict(profile)
                break
            if ok:
                chosen_profile = dict(profile)
                break
        if chosen_profile is None:
            manifest["final_status"] = "preflight_failed"
            write_json(manifest_path, manifest)
            return 1
    else:
        chosen_profile = dict(RUNTIME_PROFILES[0])

    full_jobs = build_stage_jobs(
        stage_root=run_root,
        stage_name="full",
        strategies=FULL_STRATEGIES,
        concurrency=[int(x) for x in args.concurrency],
        limit=int(args.limit_full),
        minutes=int(args.minutes_full),
        port_base=34200,
    )
    ok = execute_stage(
        stage_name="full",
        stage_root=run_root,
        jobs=full_jobs,
        profile=chosen_profile,
        args=args,
        manifest=manifest,
    )
    manifest["chosen_profile"] = dict(chosen_profile)
    manifest["final_status"] = "ok" if ok else "failed"
    write_json(manifest_path, manifest)

    if ok and not args.dry_run:
        write_debrief(run_root=run_root, args=args, profile=chosen_profile)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
