"""
Usage examples:

# 1) Compare baseline vs candidate servers directly
python3 scripts/playground/mtp_cudagraph_ab.py \
  --baseline-url http://127.0.0.1:30000 \
  --candidate-url http://127.0.0.1:30001

# 2) Candidate-only smoke run (single endpoint)
python3 scripts/playground/mtp_cudagraph_ab.py \
  --url http://127.0.0.1:30000

# 3) Include conf_adapt negative check
python3 scripts/playground/mtp_cudagraph_ab.py \
  --baseline-url http://127.0.0.1:30000 \
  --candidate-url http://127.0.0.1:30001 \
  --include-conf-adapt

# 4) Single-GPU sequential workflow:
#    a) run baseline server, then:
python3 scripts/playground/mtp_cudagraph_ab.py \
  --url http://127.0.0.1:30000 \
  --output-json /tmp/mtp_baseline.json
#
#    b) run candidate server, then:
python3 scripts/playground/mtp_cudagraph_ab.py \
  --url http://127.0.0.1:30000 \
  --output-json /tmp/mtp_candidate.json
#
#    c) compare saved reports:
python3 scripts/playground/mtp_cudagraph_ab.py \
  --baseline-report /tmp/mtp_baseline.json \
  --candidate-report /tmp/mtp_candidate.json
"""

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests


DEFAULT_PROMPT = (
    "Janet lays 16 eggs per day. She eats 3 for breakfast and sells the rest for $2 each. "
    "How much does she earn per day?"
)
TRACE_FIELDS = [
    "can_run_cuda_graph",
    "phase_before_step",
    "phase_after_step",
    "decode_q_len_per_req_runtime",
    "decode_k_for_step_runtime",
    "effective_k_runtime",
    "finished_reason",
    "input_row_token_ids",
    "pending_token_ids",
    "committed_token_ids",
    "positions_row",
    "seq_len_base_used_for_next_decode",
]


@dataclass
class Scenario:
    name: str
    sampling_params: Dict[str, Any]
    expected_seed_q: int
    expected_steady_q: int
    is_conf_adapt: bool


def _to_int_list(values: List[Any]) -> List[int]:
    return [int(v) for v in values]


def build_scenarios(
    k_values: List[int],
    mask_id: int,
    max_new_tokens: int,
    include_conf_adapt: bool,
    conf_adapt_k: int,
    conf_adapt_threshold: float,
) -> List[Scenario]:
    scenarios: List[Scenario] = []
    for k in k_values:
        k = int(k)
        if k < 1:
            raise ValueError(f"Invalid k={k}. k must be >= 1.")
        scenarios.append(
            Scenario(
                name=f"static_k{k}",
                sampling_params={
                    "temperature": 0,
                    "top_k": 1,
                    "max_new_tokens": max_new_tokens,
                    "mtp_enabled": True,
                    "mtp_k": k,
                    "mtp_mask_id": mask_id,
                    "custom_params": {
                        "mtp_debug_trace": True,
                        "mtp_debug_max_steps": max_new_tokens,
                    },
                },
                expected_seed_q=k,
                expected_steady_q=(2 * k - 1) if k > 1 else 1,
                is_conf_adapt=False,
            )
        )

    if include_conf_adapt:
        scenarios.append(
            Scenario(
                name=f"conf_adapt_k{conf_adapt_k}_t{conf_adapt_threshold}",
                sampling_params={
                    "temperature": 0,
                    "top_k": 1,
                    "max_new_tokens": max_new_tokens,
                    "mtp_enabled": True,
                    "mtp_k": conf_adapt_k,
                    "mtp_strategy": ["conf_adapt", conf_adapt_threshold],
                    "mtp_mask_id": mask_id,
                    "custom_params": {
                        "mtp_debug_trace": True,
                        "mtp_debug_max_steps": max_new_tokens,
                    },
                },
                expected_seed_q=conf_adapt_k,
                expected_steady_q=1,  # expected typical conf_adapt behavior after seed
                is_conf_adapt=True,
            )
        )

    return scenarios


def post_generate(url: str, text: str, sampling_params: Dict[str, Any], timeout_s: int) -> Dict[str, Any]:
    payload = {
        "text": text,
        "sampling_params": sampling_params,
        "stream": False,
    }
    resp = requests.post(f"{url.rstrip('/')}/generate", json=payload, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def simplify_trace_rows(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in trace:
        if row is None:
            out.append({"_row_is_none": True})
            continue
        out.append({k: row.get(k) for k in TRACE_FIELDS})
    return out


def first_seed_and_steady_rows(
    trace: List[Dict[str, Any]], steady_steps: int
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    if not trace:
        return None, []
    seed = trace[0]
    steady = []
    for row in trace:
        if row is None:
            continue
        if row.get("phase_before_step") == "steady":
            steady.append(row)
        if len(steady) >= steady_steps:
            break
    return seed, steady


def find_trace_issues(
    trace: List[Dict[str, Any]],
    scenario: Scenario,
    expect_q_gt1_graph: bool,
) -> List[str]:
    issues = []
    if trace is None:
        return ["missing_mtp_debug_trace"]
    if len(trace) == 0:
        return ["empty_mtp_debug_trace"]

    for idx, row in enumerate(trace):
        if row is None:
            issues.append(f"trace_row_none_at_step_{idx}")
            continue
        if row.get("phase_before_step") is None:
            issues.append(f"trace_phase_missing_at_step_{idx}")
        if row.get("decode_q_len_per_req_runtime") is None:
            issues.append(f"trace_q_len_missing_at_step_{idx}")

    seed = trace[0]
    if seed is not None:
        seed_q = seed.get("decode_q_len_per_req_runtime")
        if seed_q is not None and int(seed_q) != int(scenario.expected_seed_q):
            issues.append(
                f"seed_q_mismatch_expected_{scenario.expected_seed_q}_got_{seed_q}"
            )

    for idx, row in enumerate(trace):
        if row is None:
            continue
        phase = row.get("phase_before_step")
        q_runtime = row.get("decode_q_len_per_req_runtime")
        if phase == "steady" and q_runtime is not None and not scenario.is_conf_adapt:
            if int(q_runtime) != int(scenario.expected_steady_q):
                issues.append(
                    f"steady_q_mismatch_at_step_{idx}_expected_{scenario.expected_steady_q}_got_{q_runtime}"
                )

    if expect_q_gt1_graph and not scenario.is_conf_adapt and scenario.expected_seed_q > 1:
        for idx, row in enumerate(trace):
            if row is None:
                continue
            if row.get("can_run_cuda_graph") is not True:
                issues.append(f"q_gt1_graph_not_enabled_at_step_{idx}")
                break

    if scenario.is_conf_adapt:
        # conf_adapt q>1 should stay eager for q>1 rows.
        for idx, row in enumerate(trace):
            if row is None:
                continue
            q_runtime = row.get("decode_q_len_per_req_runtime")
            can_graph = row.get("can_run_cuda_graph")
            if q_runtime is not None and int(q_runtime) > 1 and can_graph is True:
                issues.append(f"conf_adapt_q_gt1_used_graph_at_step_{idx}")
                break

    return issues


def compare_traces(
    baseline_trace: List[Dict[str, Any]],
    candidate_trace: List[Dict[str, Any]],
    steady_steps_to_compare: int,
) -> List[str]:
    diffs = []
    base_seed, base_steady = first_seed_and_steady_rows(baseline_trace, steady_steps_to_compare)
    cand_seed, cand_steady = first_seed_and_steady_rows(candidate_trace, steady_steps_to_compare)

    if base_seed is None or cand_seed is None:
        return ["missing_seed_step"]

    def cmp_row(label: str, row_a: Dict[str, Any], row_b: Dict[str, Any], fields: List[str]):
        for field in fields:
            if row_a.get(field) != row_b.get(field):
                diffs.append(
                    f"{label}.{field}: baseline={row_a.get(field)} candidate={row_b.get(field)}"
                )

    cmp_row(
        "seed",
        base_seed,
        cand_seed,
        [
            "pending_token_ids",
            "committed_token_ids",
            "decode_q_len_per_req_runtime",
            "effective_k_runtime",
        ],
    )

    if len(base_steady) < steady_steps_to_compare:
        diffs.append(
            f"baseline_has_only_{len(base_steady)}_steady_steps_expected_{steady_steps_to_compare}"
        )
    if len(cand_steady) < steady_steps_to_compare:
        diffs.append(
            f"candidate_has_only_{len(cand_steady)}_steady_steps_expected_{steady_steps_to_compare}"
        )

    steps = min(len(base_steady), len(cand_steady), steady_steps_to_compare)
    for i in range(steps):
        cmp_row(
            f"steady[{i}]",
            base_steady[i],
            cand_steady[i],
            [
                "pending_token_ids",
                "committed_token_ids",
                "decode_q_len_per_req_runtime",
                "effective_k_runtime",
            ],
        )

    return diffs


def run_suite(
    url: str,
    text: str,
    scenarios: List[Scenario],
    timeout_s: int,
    expect_q_gt1_graph: bool,
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for scenario in scenarios:
        raw = post_generate(url, text, scenario.sampling_params, timeout_s)
        trace = raw.get("meta_info", {}).get("mtp_debug_trace")
        if trace is None:
            trace = []
        issues = find_trace_issues(
            trace=trace,
            scenario=scenario,
            expect_q_gt1_graph=expect_q_gt1_graph,
        )
        results[scenario.name] = {
            "scenario": scenario.name,
            "sampling_params": scenario.sampling_params,
            "issues": issues,
            "trace_len": len(trace),
            "trace": simplify_trace_rows(trace),
        }
    return results


def print_single_endpoint_summary(url: str, suite: Dict[str, Dict[str, Any]]) -> None:
    print(f"\n=== Endpoint Summary: {url} ===")
    for name, result in suite.items():
        issues = result["issues"]
        status = "PASS" if len(issues) == 0 else "WARN"
        print(f"[{status}] {name} trace_len={result['trace_len']}")
        if issues:
            for issue in issues:
                print(f"  - {issue}")
        trace = result["trace"]
        preview = [trace[0]] if trace else []
        if len(trace) > 1:
            preview.append(trace[1])
        if len(trace) > 2:
            preview.append(trace[2])
        if len(trace) > 3:
            preview.append(trace[-1])
        print("  preview_steps=", json.dumps(preview, ensure_ascii=True))


def print_ab_summary(
    baseline_url: str,
    candidate_url: str,
    baseline_suite: Dict[str, Dict[str, Any]],
    candidate_suite: Dict[str, Dict[str, Any]],
    steady_steps_to_compare: int,
) -> int:
    print(f"\n=== A/B Summary ===")
    print(f"baseline={baseline_url}")
    print(f"candidate={candidate_url}")

    total_failures = 0
    scenario_names = sorted(set(baseline_suite.keys()) | set(candidate_suite.keys()))
    for name in scenario_names:
        b = baseline_suite.get(name)
        c = candidate_suite.get(name)
        if b is None or c is None:
            total_failures += 1
            print(f"[FAIL] {name}: missing_in_one_suite")
            continue

        scenario_failures = []
        scenario_failures.extend([f"baseline:{x}" for x in b["issues"]])
        scenario_failures.extend([f"candidate:{x}" for x in c["issues"]])
        diffs = compare_traces(
            baseline_trace=b["trace"],
            candidate_trace=c["trace"],
            steady_steps_to_compare=steady_steps_to_compare,
        )
        scenario_failures.extend([f"diff:{x}" for x in diffs])

        if scenario_failures:
            total_failures += 1
            print(f"[FAIL] {name}")
            for failure in scenario_failures:
                print(f"  - {failure}")
        else:
            print(f"[PASS] {name}")

    return total_failures


def load_suite_from_report(path: str, side: str) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)

    if "suite" in report:
        return str(report.get("url", path)), report["suite"]

    if side == "baseline" and "baseline_suite" in report:
        return str(report.get("baseline_url", path)), report["baseline_suite"]
    if side == "candidate" and "candidate_suite" in report:
        return str(report.get("candidate_url", path)), report["candidate_suite"]

    raise ValueError(
        f"Unsupported report format in {path}. Expected keys: suite OR "
        "baseline_suite/candidate_suite."
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default=None, help="Single endpoint mode.")
    parser.add_argument("--baseline-url", type=str, default=None)
    parser.add_argument("--candidate-url", type=str, default=None)
    parser.add_argument("--baseline-report", type=str, default=None)
    parser.add_argument("--candidate-report", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--mask-id", type=int, default=128259)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--steady-steps-to-compare", type=int, default=3)
    parser.add_argument("--include-conf-adapt", action="store_true")
    parser.add_argument("--conf-adapt-k", type=int, default=8)
    parser.add_argument("--conf-adapt-threshold", type=float, default=0.9)
    parser.add_argument(
        "--allow-qgt1-eager",
        action="store_true",
        help="Allow q>1 static scenarios to run eagerly (do not require can_run_cuda_graph=true).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any scenario fails in A/B mode or has issues in single-endpoint mode.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save full raw summary report.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    endpoint_single_mode = args.url is not None
    endpoint_ab_mode = args.baseline_url is not None and args.candidate_url is not None
    report_ab_mode = (
        args.baseline_report is not None and args.candidate_report is not None
    )

    if not (endpoint_single_mode or endpoint_ab_mode or report_ab_mode):
        raise ValueError(
            "Provide one mode: --url OR --baseline-url/--candidate-url OR "
            "--baseline-report/--candidate-report."
        )
    if endpoint_single_mode and (endpoint_ab_mode or report_ab_mode):
        raise ValueError("Do not mix --url with A/B endpoint/report modes.")
    if endpoint_ab_mode and report_ab_mode:
        raise ValueError("Use either endpoint A/B mode OR report A/B mode, not both.")

    scenarios = build_scenarios(
        k_values=_to_int_list(args.k_values),
        mask_id=int(args.mask_id),
        max_new_tokens=int(args.max_new_tokens),
        include_conf_adapt=bool(args.include_conf_adapt),
        conf_adapt_k=int(args.conf_adapt_k),
        conf_adapt_threshold=float(args.conf_adapt_threshold),
    )
    expect_qgt1_graph = not bool(args.allow_qgt1_eager)

    report: Dict[str, Any] = {"scenarios": [s.name for s in scenarios]}
    exit_code = 0

    if endpoint_single_mode:
        suite = run_suite(
            url=args.url,
            text=args.prompt,
            scenarios=scenarios,
            timeout_s=args.timeout,
            expect_q_gt1_graph=expect_qgt1_graph,
        )
        print_single_endpoint_summary(args.url, suite)
        report["mode"] = "single"
        report["url"] = args.url
        report["suite"] = suite
        if args.strict:
            for result in suite.values():
                if len(result["issues"]) > 0:
                    exit_code = 1
                    break
    elif endpoint_ab_mode:
        baseline_suite = run_suite(
            url=args.baseline_url,
            text=args.prompt,
            scenarios=scenarios,
            timeout_s=args.timeout,
            expect_q_gt1_graph=False,
        )
        candidate_suite = run_suite(
            url=args.candidate_url,
            text=args.prompt,
            scenarios=scenarios,
            timeout_s=args.timeout,
            expect_q_gt1_graph=expect_qgt1_graph,
        )
        failures = print_ab_summary(
            baseline_url=args.baseline_url,
            candidate_url=args.candidate_url,
            baseline_suite=baseline_suite,
            candidate_suite=candidate_suite,
            steady_steps_to_compare=args.steady_steps_to_compare,
        )
        report["mode"] = "ab"
        report["baseline_url"] = args.baseline_url
        report["candidate_url"] = args.candidate_url
        report["baseline_suite"] = baseline_suite
        report["candidate_suite"] = candidate_suite
        report["failures"] = failures
        if args.strict and failures > 0:
            exit_code = 1
    else:
        baseline_url, baseline_suite = load_suite_from_report(
            args.baseline_report, "baseline"
        )
        candidate_url, candidate_suite = load_suite_from_report(
            args.candidate_report, "candidate"
        )
        failures = print_ab_summary(
            baseline_url=baseline_url,
            candidate_url=candidate_url,
            baseline_suite=baseline_suite,
            candidate_suite=candidate_suite,
            steady_steps_to_compare=args.steady_steps_to_compare,
        )
        report["mode"] = "ab_report"
        report["baseline_report"] = args.baseline_report
        report["candidate_report"] = args.candidate_report
        report["baseline_url"] = baseline_url
        report["candidate_url"] = candidate_url
        report["failures"] = failures
        if args.strict and failures > 0:
            exit_code = 1

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=True)
        print(f"\nWrote report: {args.output_json}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
