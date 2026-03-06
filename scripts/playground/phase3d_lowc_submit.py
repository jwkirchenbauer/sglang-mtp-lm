#!/usr/bin/env python3
"""Submit and monitor the Phase3D low-concurrency sweep via launch_daint."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path("/capstor/scratch/cscs/jkirchen/sglang-mtp-lm")
DEFAULT_MODEL_PATH = Path(
    "/capstor/scratch/cscs/jkirchen/singleshot-root/singleshot/outputs/"
    "daint_prod_ift_mask_fix/daint_prod_ift_mask_fix_1N4n_9d30cad5/step-00100160"
)
DEFAULT_LAUNCH_DAINT = Path("/capstor/scratch/cscs/jkirchen/llnl-tools/launch_daint.py")
WORKER_SCRIPT = REPO_ROOT / "scripts/playground/phase3d_lowc_worker.sh"
METRICS_SCRIPT = REPO_ROOT / "scripts/playground/phase3d_lowc_metrics_summary.py"

FULL_CONCURRENCY = [1, 2, 4, 8, 16]
SMOKE_CONCURRENCY = [1]

MTP_STRATEGIES = [
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
EAGLE3_STRATEGY = "eagle3"


@dataclass
class JobSpec:
    stage: str
    kind: str
    strategy: str
    shard_dir: Path
    port: int
    limit: int
    concurrency: List[int]
    minutes: int
    run_name: str


@dataclass
class SubmissionRecord:
    stage: str
    kind: str
    strategy: str
    run_name: str
    shard_dir: str
    limit: int
    concurrency: List[int]
    minutes: int
    port: int
    job_id: Optional[int]
    command: List[str]
    returncode: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Run root. Default: outputs/phase3d_lowc_robust_<timestamp>/",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="MTP/non-MTP model checkpoint path.",
    )
    parser.add_argument(
        "--launch-daint",
        type=Path,
        default=DEFAULT_LAUNCH_DAINT,
        help="Path to launch_daint.py.",
    )
    parser.add_argument("--partition", type=str, default="normal")
    parser.add_argument("--minutes", type=int, default=45)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the two smoke jobs and launch the full matrix directly.",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit jobs and return without waiting/collecting metrics.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write launch scripts but do not submit sbatch jobs.",
    )
    parser.add_argument(
        "--max-resubmit-rounds",
        type=int,
        default=2,
        help="Resubmission rounds for failed full shards.",
    )
    return parser.parse_args()


def ts_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def strategy_slug(strategy: str) -> str:
    return strategy.replace(".", "p").replace("+", "plus").replace("/", "_")


def build_custom_invocation(
    *,
    spec: JobSpec,
    model_path: Path,
    task: str,
    mask_id: int,
    max_gen_toks: int,
) -> str:
    parts = [
        "bash",
        str(WORKER_SCRIPT),
        "--kind",
        spec.kind,
        "--strategy",
        spec.strategy,
        "--shard-dir",
        str(spec.shard_dir),
        "--port",
        str(spec.port),
        "--task",
        task,
        "--limit",
        str(spec.limit),
        "--concurrency",
        ",".join(str(x) for x in spec.concurrency),
        "--max-gen-toks",
        str(max_gen_toks),
    ]
    if spec.kind == "mtp":
        parts.extend(
            [
                "--model-path",
                str(model_path),
                "--mask-id",
                str(mask_id),
            ]
        )
    return " ".join(shlex.quote(x) for x in parts)


def build_launch_cmd(
    *,
    launch_daint: Path,
    spec: JobSpec,
    model_path: Path,
    partition: str,
    task: str,
    mask_id: int,
    max_gen_toks: int,
    dry_run: bool,
) -> List[str]:
    custom_invocation = build_custom_invocation(
        spec=spec,
        model_path=model_path,
        task=task,
        mask_id=mask_id,
        max_gen_toks=max_gen_toks,
    )
    cmd = [
        sys.executable,
        str(launch_daint),
        "--custom_invocation",
        custom_invocation,
        "--run_name",
        spec.run_name,
        "--pass_run_name",
        "false",
        "--partition",
        partition,
        "--minutes",
        str(spec.minutes),
        "--nodes",
        "1",
        "--gpus_per_node",
        "1",
        "--ntasks_per_node",
        "1",
        "--cpus_per_node",
        "18",
        "--cpus_per_task",
        "18",
        "--output_dir",
        str(spec.shard_dir),
        "--sub_output_dir_name",
        "launcher",
        "--nccl_cfg",
        "eager",
        "--squonce",
        "false",
    ]
    if dry_run:
        cmd.append("--dryrun")
    return cmd


def extract_job_id(stdout: str, stderr: str) -> Optional[int]:
    text = "\n".join([stdout, stderr])
    m = re.search(r"Successfully submitted batch job (\d+)", text)
    if m is None:
        return None
    return int(m.group(1))


def submit_job(cmd: Sequence[str]) -> Tuple[int, str, str, Optional[int]]:
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    job_id = extract_job_id(proc.stdout, proc.stderr)
    return proc.returncode, proc.stdout, proc.stderr, job_id


def wait_for_jobs(job_ids: Sequence[int], poll_seconds: int) -> None:
    pending = {int(x) for x in job_ids}
    while pending:
        query = ",".join(str(x) for x in sorted(pending))
        proc = subprocess.run(
            ["squeue", "-h", "-j", query, "-o", "%A"],
            check=False,
            capture_output=True,
            text=True,
        )
        live: set[int] = set()
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.isdigit():
                    live.add(int(line))
        pending = pending & live
        if pending:
            print(f"Waiting on jobs: {sorted(pending)}", flush=True)
            time.sleep(max(5, int(poll_seconds)))


def wait_for_run_names(run_names: Sequence[str], poll_seconds: int) -> None:
    pending = set(run_names)
    user = os.environ.get("USER", "")
    while pending:
        proc = subprocess.run(
            ["squeue", "-h", "-u", user, "-o", "%j"],
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
        live: set[str] = set()
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                name = line.strip()
                if name:
                    live.add(name)
        pending = pending & live
        if pending:
            print(f"Waiting on run_names: {sorted(pending)}", flush=True)
            time.sleep(max(5, int(poll_seconds)))


def resolve_job_id_by_run_name(run_name: str, timeout_s: int = 30) -> Optional[int]:
    user = os.environ.get("USER", "")
    if not user:
        return None
    deadline = time.time() + max(1, int(timeout_s))
    while time.time() < deadline:
        proc = subprocess.run(
            ["squeue", "-h", "-u", user, "-n", run_name, "-o", "%A"],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            ids: List[int] = []
            for line in proc.stdout.splitlines():
                line = line.strip()
                if line.isdigit():
                    ids.append(int(line))
            if ids:
                return max(ids)
        time.sleep(2)
    return None


def jobspec_to_dict(spec: JobSpec) -> Dict[str, object]:
    return {
        "stage": spec.stage,
        "kind": spec.kind,
        "strategy": spec.strategy,
        "shard_dir": str(spec.shard_dir),
        "port": int(spec.port),
        "limit": int(spec.limit),
        "concurrency": list(spec.concurrency),
        "minutes": int(spec.minutes),
        "run_name": spec.run_name,
    }


def read_status(shard_dir: Path) -> str:
    status_file = shard_dir / "status.txt"
    if not status_file.exists():
        return "missing"
    return status_file.read_text(encoding="utf-8").strip() or "missing"


def create_jobs(
    *,
    run_root: Path,
    minutes: int,
    stage: str,
    smoke: bool,
) -> List[JobSpec]:
    jobs: List[JobSpec] = []
    run_tag = re.sub(r"[^a-zA-Z0-9]", "", run_root.name)[-8:].lower() or "p3d"
    conc = SMOKE_CONCURRENCY if smoke else FULL_CONCURRENCY
    limit = 32 if smoke else 512

    if smoke:
        smoke_specs = [
            ("mtp", "non_mtp"),
            ("eagle3", EAGLE3_STRATEGY),
        ]
        base_port = 32100
        for idx, (kind, strategy) in enumerate(smoke_specs):
            slug = f"{strategy_slug(strategy)}_smoke"
            shard_dir = run_root / "preflight" / slug
            run_name = f"p3d{run_tag}_smk{idx:02d}_{strategy_slug(strategy)[:14]}"
            jobs.append(
                JobSpec(
                    stage=stage,
                    kind=kind,
                    strategy=strategy,
                    shard_dir=shard_dir,
                    port=base_port + idx,
                    limit=limit,
                    concurrency=list(conc),
                    minutes=minutes,
                    run_name=run_name,
                )
            )
        return jobs

    full_specs: List[Tuple[str, str]] = [("mtp", s) for s in MTP_STRATEGIES] + [
        ("eagle3", EAGLE3_STRATEGY)
    ]
    base_port = 32200
    for idx, (kind, strategy) in enumerate(full_specs):
        slug = strategy_slug(strategy)
        shard_dir = run_root / "shards" / slug
        run_name = f"p3d{run_tag}_{idx:02d}_{slug[:16]}"
        jobs.append(
            JobSpec(
                stage=stage,
                kind=kind,
                strategy=strategy,
                shard_dir=shard_dir,
                port=base_port + idx,
                limit=limit,
                concurrency=list(conc),
                minutes=minutes,
                run_name=run_name,
            )
        )
    return jobs


def run_stage(
    *,
    stage_name: str,
    jobs: Sequence[JobSpec],
    args: argparse.Namespace,
    model_path: Path,
    manifest: Dict[str, object],
) -> Tuple[List[SubmissionRecord], List[JobSpec]]:
    submitted: List[SubmissionRecord] = []
    for spec in jobs:
        spec.shard_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_launch_cmd(
            launch_daint=args.launch_daint,
            spec=spec,
            model_path=model_path,
            partition=str(args.partition),
            task="gsm8k_cot_singleshot",
            mask_id=128259,
            max_gen_toks=128,
            dry_run=bool(args.dry_run),
        )
        print(f"Submitting {spec.stage}:{spec.strategy} -> {spec.shard_dir}", flush=True)
        rc, out, err, job_id = submit_job(cmd)
        if job_id is None and rc == 0 and not args.dry_run:
            job_id = resolve_job_id_by_run_name(spec.run_name, timeout_s=40)
        if out.strip():
            print(out.rstrip(), flush=True)
        if err.strip():
            print(err.rstrip(), file=sys.stderr, flush=True)
        rec = SubmissionRecord(
            stage=spec.stage,
            kind=spec.kind,
            strategy=spec.strategy,
            run_name=spec.run_name,
            shard_dir=str(spec.shard_dir),
            limit=spec.limit,
            concurrency=list(spec.concurrency),
            minutes=spec.minutes,
            port=spec.port,
            job_id=job_id,
            command=list(cmd),
            returncode=rc,
        )
        submitted.append(rec)

    manifest.setdefault("submissions", [])
    manifest["submissions"].extend(asdict(x) for x in submitted)

    if args.no_wait or args.dry_run:
        return submitted, []

    job_ids = [x.job_id for x in submitted if x.returncode == 0 and x.job_id is not None]
    if job_ids:
        wait_for_jobs(job_ids=job_ids, poll_seconds=int(args.poll_seconds))
    else:
        run_names = [x.run_name for x in submitted if x.returncode == 0]
        if run_names:
            wait_for_run_names(run_names=run_names, poll_seconds=int(args.poll_seconds))
        time.sleep(5)

    failed_specs: List[JobSpec] = []
    for spec in jobs:
        status = read_status(spec.shard_dir)
        if status != "ok":
            failed_specs.append(spec)
            print(f"[warn] {stage_name}:{spec.strategy} status={status}", flush=True)
    return submitted, failed_specs


def run_metrics(run_root: Path) -> int:
    cmd = [
        sys.executable,
        str(METRICS_SCRIPT),
        "--run-root",
        str(run_root),
    ]
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def main() -> int:
    args = parse_args()
    if not REPO_ROOT.exists():
        raise FileNotFoundError(f"Repo not found: {REPO_ROOT}")
    if not args.launch_daint.exists():
        raise FileNotFoundError(f"launch_daint.py not found: {args.launch_daint}")
    if not WORKER_SCRIPT.exists():
        raise FileNotFoundError(f"Worker script not found: {WORKER_SCRIPT}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    run_root = args.run_root
    if run_root is None:
        run_root = REPO_ROOT / "outputs" / f"phase3d_lowc_robust_{ts_tag()}"
    run_root = run_root.resolve()
    (run_root / "summary").mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, object] = {
        "run_root": str(run_root),
        "created_at": dt.datetime.now().isoformat(),
        "model_path": str(args.model_path),
        "launch_daint": str(args.launch_daint),
        "partition": str(args.partition),
        "minutes": int(args.minutes),
        "dry_run": bool(args.dry_run),
        "skip_preflight": bool(args.skip_preflight),
        "no_wait": bool(args.no_wait),
        "full_concurrency": FULL_CONCURRENCY,
        "smoke_concurrency": SMOKE_CONCURRENCY,
        "mtp_strategies": MTP_STRATEGIES,
        "eagle3_strategy": EAGLE3_STRATEGY,
        "submissions": [],
        "failures": [],
    }

    if not args.skip_preflight:
        preflight_jobs = create_jobs(
            run_root=run_root,
            minutes=max(20, int(args.minutes)),
            stage="preflight",
            smoke=True,
        )
        _, preflight_failures = run_stage(
            stage_name="preflight",
            jobs=preflight_jobs,
            args=args,
            model_path=args.model_path,
            manifest=manifest,
        )
        if preflight_failures and not (args.no_wait or args.dry_run):
            manifest["failures"] = [jobspec_to_dict(x) for x in preflight_failures]
            manifest_path = run_root / "summary" / "run_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            print(f"Preflight failed. Manifest: {manifest_path}", file=sys.stderr)
            return 1

    full_jobs = create_jobs(
        run_root=run_root,
        minutes=int(args.minutes),
        stage="full",
        smoke=False,
    )
    _, full_failures = run_stage(
        stage_name="full",
        jobs=full_jobs,
        args=args,
        model_path=args.model_path,
        manifest=manifest,
    )

    if not (args.no_wait or args.dry_run):
        current_failures = list(full_failures)
        for round_idx in range(1, int(args.max_resubmit_rounds) + 1):
            if not current_failures:
                break
            print(f"Resubmission round {round_idx} for {len(current_failures)} shard(s).", flush=True)
            retry_jobs: List[JobSpec] = []
            for spec in current_failures:
                retry_jobs.append(
                    JobSpec(
                        stage=f"retry{round_idx}",
                        kind=spec.kind,
                        strategy=spec.strategy,
                        shard_dir=spec.shard_dir,
                        port=spec.port,
                        limit=spec.limit,
                        concurrency=list(spec.concurrency),
                        minutes=int(args.minutes) + (30 * round_idx),
                        run_name=f"{spec.run_name}_r{round_idx}",
                    )
                )
            _, current_failures = run_stage(
                stage_name=f"retry{round_idx}",
                jobs=retry_jobs,
                args=args,
                model_path=args.model_path,
                manifest=manifest,
            )
        full_failures = current_failures

    manifest["failures"] = [jobspec_to_dict(x) for x in full_failures]
    manifest_path = run_root / "summary" / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")

    if args.no_wait or args.dry_run:
        return 0

    if run_metrics(run_root) != 0:
        return 1
    return 1 if full_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
