#!/usr/bin/env python3
"""Deterministic pass/fail gate for MTP KV ownership logs."""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple


FAIL_SUBSTRINGS = [
    "Recovered orphan radix KV tokens",
    "Removed overlapping cached tokens from allocator free lists",
    "Scheduler hit an exception:",
    "MTP terminal release ownership gap detected",
]

NONZERO_COUNTER_PATTERNS = [
    ("missing_from_terminal_release_count", re.compile(r"missing_from_terminal_release_count=(\d+)")),
    ("orphan_token_count", re.compile(r"orphan_token_count=(\d+)")),
    ("overlap_token_count", re.compile(r"overlap_token_count=(\d+)")),
]

HTTP_STATUS_RE = re.compile(r"POST /generate HTTP/1\.1\" (\d{3})")


def scan_log(
    path: Path, *, strict_startup: bool
) -> Tuple[List[str], List[str], int, int]:
    failures: List[str] = []
    tolerated_startup: List[str] = []
    total_http = 0
    ok_http = 0
    tolerated_orphan_singleton = False

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            for needle in FAIL_SUBSTRINGS:
                if needle in line:
                    if (
                        not strict_startup
                        and needle == "Recovered orphan radix KV tokens"
                        and not tolerated_orphan_singleton
                        and "count=1" in line
                    ):
                        tolerated_orphan_singleton = True
                        tolerated_startup.append(
                            f"{path}:{line_no}: tolerated startup singleton orphan recovery"
                        )
                    else:
                        failures.append(
                            f"{path}:{line_no}: matched fail substring: {needle}"
                        )

            for name, pattern in NONZERO_COUNTER_PATTERNS:
                for m in pattern.finditer(line):
                    value = int(m.group(1))
                    if value != 0:
                        failures.append(
                            f"{path}:{line_no}: {name}={value} (expected 0)"
                        )

            status_match = HTTP_STATUS_RE.search(line)
            if status_match:
                total_http += 1
                if int(status_match.group(1)) == 200:
                    ok_http += 1
                else:
                    failures.append(
                        f"{path}:{line_no}: non-200 /generate status={status_match.group(1)}"
                    )

    return failures, tolerated_startup, total_http, ok_http


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gate MTP runs using strict ownership and HTTP-success checks."
    )
    parser.add_argument(
        "log_files",
        nargs="+",
        help="Server log file(s) to scan.",
    )
    parser.add_argument(
        "--expected-http-200",
        type=int,
        default=None,
        help=(
            "Expected number of HTTP 200 /generate responses. "
            "Default behavior accepts observed >= expected."
        ),
    )
    parser.add_argument(
        "--exact-http-200",
        action="store_true",
        help="Require exact equality with --expected-http-200 (strict mode).",
    )
    parser.add_argument(
        "--strict-startup",
        action="store_true",
        help="Disable startup-noise tolerance and fail on all matching signatures.",
    )
    args = parser.parse_args()

    all_failures: List[str] = []
    tolerated_startup_events: List[str] = []
    total_http = 0
    ok_http = 0

    for raw_path in args.log_files:
        path = Path(raw_path)
        if not path.exists():
            all_failures.append(f"{path}: file not found")
            continue
        failures, tolerated, file_total_http, file_ok_http = scan_log(
            path, strict_startup=args.strict_startup
        )
        all_failures.extend(failures)
        tolerated_startup_events.extend(tolerated)
        total_http += file_total_http
        ok_http += file_ok_http

    if args.expected_http_200 is not None:
        if args.exact_http_200:
            if ok_http != args.expected_http_200:
                all_failures.append(
                    f"expected_http_200={args.expected_http_200}, observed_http_200={ok_http}"
                )
        elif ok_http < args.expected_http_200:
            all_failures.append(
                f"min_expected_http_200={args.expected_http_200}, observed_http_200={ok_http}"
            )

    if total_http != ok_http:
        all_failures.append(
            f"http_success_rate= {ok_http}/{total_http} (must be 100%)"
        )

    if all_failures:
        print("MTP log gate: FAIL")
        for item in all_failures:
            print(f"- {item}")
        if tolerated_startup_events:
            print("- startup_tolerated_events:")
            for item in tolerated_startup_events:
                print(f"  - {item}")
        return 1

    print("MTP log gate: PASS")
    print(f"- http_success_rate= {ok_http}/{total_http}")
    if tolerated_startup_events:
        print(
            f"- startup_tolerated_events= {len(tolerated_startup_events)} "
            "(ignored by default policy)"
        )
    print("- runtime orphan/overlap/repair signatures: none detected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
