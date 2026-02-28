from __future__ import annotations

import logging
import time
import warnings
from typing import TYPE_CHECKING

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.utils.common import ceil_align, raise_error_or_warn
from sglang.srt.utils.request_logger import disable_request_logging
from sglang.srt.utils.watchdog import WatchdogRaw

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)
IDLE_MEM_CHECK_SKIP_LOG_INTERVAL_SEC = 5.0


class SchedulerRuntimeCheckerMixin:
    def _get_radix_token_set_accounting(self: Scheduler):
        # Explicit token-set accounting for radix cache correctness checks.
        # This path is only valid when page_size==1 and cached values are token ids.
        if self.page_size != 1:
            return None
        if not hasattr(self.tree_cache, "all_values_flatten"):
            return None

        try:
            free_tokens = set(
                self.token_to_kv_pool_allocator.free_pages.tolist()
                + self.token_to_kv_pool_allocator.release_pages.tolist()
            )
            cached_tokens = set(self.tree_cache.all_values_flatten().tolist())
            expected_tokens = set(range(1, self.token_to_kv_pool_allocator.size + 1))
            missing_tokens = sorted(expected_tokens - free_tokens - cached_tokens)
            overlap_tokens = sorted(free_tokens & cached_tokens)
            return {
                "missing_tokens": missing_tokens,
                "overlap_tokens": overlap_tokens,
            }
        except Exception:
            return None

    def _get_orphan_radix_token_indices(self: Scheduler):
        accounting = self._get_radix_token_set_accounting()
        if accounting is None:
            return []
        return accounting["missing_tokens"]

    def _get_radix_invariant_summary(self: Scheduler):
        if self.is_hybrid_swa or (self.is_hybrid_ssm and self.tree_cache.supports_mamba()):
            return None

        _, _, available_size, evictable_size = self._get_token_info()
        protected_size = self.tree_cache.protected_size()
        legacy_missing_tokens = (
            self.max_total_num_tokens - protected_size - (available_size + evictable_size)
        )
        accounting = self._get_radix_token_set_accounting()
        if accounting is None:
            orphan_token_count = 0
            overlap_token_count = 0
        else:
            orphan_token_count = len(accounting["missing_tokens"])
            overlap_token_count = len(accounting["overlap_tokens"])
        return legacy_missing_tokens, orphan_token_count, overlap_token_count

    def _log_idle_radix_invariant_summary(self: Scheduler):
        summary = self._get_radix_invariant_summary()
        if summary is None:
            return
        legacy_missing_tokens, orphan_token_count, overlap_token_count = summary
        logger.info(
            "Idle radix invariant summary (first quiescent tick). "
            f"{legacy_missing_tokens=}, {orphan_token_count=}, {overlap_token_count=}"
        )

    def _try_reclaim_orphan_radix_tokens(self: Scheduler) -> int:
        orphan_tokens = self._get_orphan_radix_token_indices()
        if len(orphan_tokens) == 0:
            return 0

        self.token_to_kv_pool_allocator.free(
            torch.tensor(
                orphan_tokens,
                dtype=torch.int64,
                device=self.token_to_kv_pool_allocator.free_pages.device,
            )
        )
        logger.warning(
            "Recovered orphan radix KV tokens during idle check. "
            f"count={len(orphan_tokens)}, sample={orphan_tokens[:16]}"
        )
        return len(orphan_tokens)

    def _try_repair_radix_free_cache_overlap(self: Scheduler) -> int:
        accounting = self._get_radix_token_set_accounting()
        if accounting is None:
            return 0

        overlap_tokens = accounting["overlap_tokens"]
        if len(overlap_tokens) == 0:
            return 0

        allocator = self.token_to_kv_pool_allocator
        overlap_tensor = torch.tensor(
            overlap_tokens,
            dtype=allocator.free_pages.dtype,
            device=allocator.free_pages.device,
        )

        removed_from_free = 0
        removed_from_release = 0
        if len(allocator.free_pages) > 0:
            keep_mask = ~torch.isin(allocator.free_pages, overlap_tensor)
            removed_from_free = int((~keep_mask).sum().item())
            allocator.free_pages = allocator.free_pages[keep_mask]
        if len(allocator.release_pages) > 0:
            keep_mask = ~torch.isin(allocator.release_pages, overlap_tensor)
            removed_from_release = int((~keep_mask).sum().item())
            allocator.release_pages = allocator.release_pages[keep_mask]

        removed_total = removed_from_free + removed_from_release
        if removed_total > 0:
            logger.warning(
                "Removed overlapping cached tokens from allocator free lists during idle check. "
                f"overlap_count={len(overlap_tokens)}, removed_from_free={removed_from_free}, "
                f"removed_from_release={removed_from_release}, sample={overlap_tokens[:16]}"
            )
        return removed_total

    def _recompute_radix_lock_counters(self: Scheduler) -> bool:
        # Best-effort repair for Python radix cache lock counters.
        # Some defer/requeue interleavings can drift protected/evictable counters
        # even when tree structure itself is still valid.
        tree = self.tree_cache
        if not (
            hasattr(tree, "root_node")
            and hasattr(tree, "protected_size_")
            and hasattr(tree, "evictable_size_")
        ):
            return False

        protected = 0
        evictable = 0
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            children = getattr(node, "children", None)
            if children:
                for child in children.values():
                    stack.append(child)

            if node is tree.root_node:
                continue
            if getattr(node, "evicted", False):
                continue

            key_len = len(getattr(node, "key", []))
            if key_len <= 0:
                continue
            if getattr(node, "lock_ref", 0) > 0:
                protected += key_len
            else:
                evictable += key_len

        old_protected = int(tree.protected_size_)
        old_evictable = int(tree.evictable_size_)
        if old_protected == protected and old_evictable == evictable:
            return False

        tree.protected_size_ = protected
        tree.evictable_size_ = evictable
        logger.warning(
            "Recomputed radix lock counters during idle check. "
            f"old_protected={old_protected}, old_evictable={old_evictable}, "
            f"new_protected={protected}, new_evictable={evictable}"
        )
        return True

    def _get_token_info(self: Scheduler):
        available_size = self.token_to_kv_pool_allocator.available_size()
        evictable_size = self.tree_cache.evictable_size()
        num_used = self.max_total_num_tokens - (available_size + evictable_size)
        token_usage = num_used / self.max_total_num_tokens
        return num_used, token_usage, available_size, evictable_size

    def _get_mamba_token_info(self: Scheduler):
        is_mamba_radix_cache = (
            self.tree_cache.supports_mamba() and self.tree_cache.is_tree_cache()
        )
        full_available_size = self.token_to_kv_pool_allocator.available_size()
        full_evictable_size = (
            self.tree_cache.full_evictable_size() if is_mamba_radix_cache else 0
        )
        mamba_available_size = self.req_to_token_pool.mamba_pool.available_size()
        mamba_evictable_size = (
            self.tree_cache.mamba_evictable_size() if is_mamba_radix_cache else 0
        )
        full_num_used = self.token_to_kv_pool_allocator.size - (
            full_available_size + full_evictable_size
        )
        mamba_num_used = self.req_to_token_pool.mamba_pool.size - (
            mamba_available_size + mamba_evictable_size
        )
        full_token_usage = full_num_used / self.token_to_kv_pool_allocator.size
        mamba_usage = mamba_num_used / self.req_to_token_pool.mamba_pool.size
        return (
            full_num_used,
            mamba_num_used,
            full_token_usage,
            mamba_usage,
            full_available_size,
            full_evictable_size,
            mamba_available_size,
            mamba_evictable_size,
        )

    def _get_swa_token_info(self: Scheduler):
        full_available_size = self.token_to_kv_pool_allocator.full_available_size()
        full_evictable_size = self.tree_cache.full_evictable_size()
        swa_available_size = self.token_to_kv_pool_allocator.swa_available_size()
        swa_evictable_size = self.tree_cache.swa_evictable_size()
        full_num_used = self.full_tokens_per_layer - (
            full_available_size + full_evictable_size
        )
        swa_num_used = self.swa_tokens_per_layer - (
            swa_available_size + swa_evictable_size
        )
        full_token_usage = full_num_used / self.full_tokens_per_layer
        swa_token_usage = swa_num_used / self.swa_tokens_per_layer
        return (
            full_num_used,
            swa_num_used,
            full_token_usage,
            swa_token_usage,
            full_available_size,
            full_evictable_size,
            swa_available_size,
            swa_evictable_size,
        )

    def _check_hybrid_memory(self: Scheduler):
        (
            full_num_used,
            swa_num_used,
            _,
            _,
            full_available_size,
            full_evictable_size,
            swa_available_size,
            swa_evictable_size,
        ) = self._get_swa_token_info()
        memory_leak = full_num_used != 0 or swa_num_used != 0
        token_msg = (
            f"{self.full_tokens_per_layer=}, {full_available_size=}, {full_evictable_size=}, {self.tree_cache.full_protected_size()=}\n"
            f"{self.swa_tokens_per_layer=}, {swa_available_size=}, {swa_evictable_size=}, {self.tree_cache.swa_protected_size()=}\n"
        )
        return memory_leak, token_msg

    def _check_mamba_memory(self: Scheduler):
        (
            full_num_used,
            mamba_num_used,
            _,
            _,
            full_available_size,
            full_evictable_size,
            mamba_available_size,
            mamba_evictable_size,
        ) = self._get_mamba_token_info()
        memory_leak = (
            full_num_used != self.tree_cache.full_protected_size()
            or mamba_num_used != self.tree_cache.mamba_protected_size()
        )
        if memory_leak:
            free_full_pages = set(
                self.token_to_kv_pool_allocator.free_pages.tolist()
                + self.token_to_kv_pool_allocator.release_pages.tolist()
            )
            cached_full_pages = set(self.tree_cache.all_values_flatten().tolist())
            expected_full_pages = set(
                range(1, self.token_to_kv_pool_allocator.size + 1)
            )
            leaked_full_pages = (
                expected_full_pages - free_full_pages - cached_full_pages
            )
            free_mamba_pages = set(
                self.req_to_token_pool.mamba_pool.free_slots.tolist()
            )
            cached_mamba_pages = set(
                self.tree_cache.all_mamba_values_flatten().tolist()
            )
            expected_mamba_pages = set(range(self.req_to_token_pool.mamba_pool.size))
            leaked_mamba_pages = (
                expected_mamba_pages - free_mamba_pages - cached_mamba_pages
            )
            token_msg = (
                f"{full_available_size=}, {full_evictable_size=}, {self.token_to_kv_pool_allocator.size=}, {self.tree_cache.full_protected_size()=}\n"
                f"{mamba_available_size=}, {mamba_evictable_size=}, {self.req_to_token_pool.mamba_pool.size=}, {self.tree_cache.mamba_protected_size()=}, leaked_full_pages={leaked_full_pages if len(leaked_full_pages) > 0 else None}, leaked_mamba_pages={leaked_mamba_pages if len(leaked_mamba_pages) > 0 else None}\n"
            )
        else:
            token_msg = (
                f"{full_available_size=}, {full_evictable_size=}, {self.token_to_kv_pool_allocator.size=}, {self.tree_cache.full_protected_size()=}\n"
                f"{mamba_available_size=}, {mamba_evictable_size=}, {self.req_to_token_pool.mamba_pool.size=}, {self.tree_cache.mamba_protected_size()=}\n"
            )
        return memory_leak, token_msg

    def _check_radix_cache_memory(self: Scheduler):
        _, _, available_size, evictable_size = self._get_token_info()
        protected_size = self.tree_cache.protected_size()
        expected_available_or_evictable = self.max_total_num_tokens - protected_size
        actual_available_or_evictable = available_size + evictable_size
        legacy_missing_tokens = (
            expected_available_or_evictable - actual_available_or_evictable
        )

        accounting = self._get_radix_token_set_accounting()
        if accounting is not None:
            orphan_tokens = accounting["missing_tokens"]
            overlap_tokens = accounting["overlap_tokens"]
            memory_leak = len(orphan_tokens) > 0 or len(overlap_tokens) > 0
        else:
            orphan_tokens = (
                self._get_orphan_radix_token_indices()
                if legacy_missing_tokens != 0
                else []
            )
            overlap_tokens = []
            memory_leak = legacy_missing_tokens != 0

        (
            mtp_tail_req_count,
            mtp_tail_token_count,
            mtp_tail_sample_rids,
        ) = self._summarize_outstanding_mtp_tail_refs()
        token_msg = (
            f"{self.max_total_num_tokens=}, {available_size=}, {evictable_size=}, "
            f"{protected_size=}, legacy_missing_tokens={legacy_missing_tokens}, "
            f"{mtp_tail_req_count=}, "
            f"{mtp_tail_token_count=}, {mtp_tail_sample_rids=}, "
            f"orphan_token_count={len(orphan_tokens)}, "
            f"orphan_token_sample={orphan_tokens[:16]}, "
            f"overlap_token_count={len(overlap_tokens)}, "
            f"overlap_token_sample={overlap_tokens[:16]}\n"
        )
        return memory_leak, token_msg

    def _summarize_outstanding_mtp_tail_refs(self: Scheduler):
        reqs = []
        seen_rids = set()
        waiting_queue = getattr(self, "waiting_queue", None)
        if waiting_queue is not None:
            reqs.extend(waiting_queue)
        running_batch = getattr(self, "running_batch", None)
        if running_batch is not None:
            reqs.extend(getattr(running_batch, "reqs", []))

        req_count = 0
        token_count = 0
        sample_rids = []

        for req in reqs:
            rid = getattr(req, "rid", None)
            if rid is None or rid in seen_rids:
                continue
            seen_rids.add(rid)

            tail_tokens = 0
            if getattr(req, "mtp_prev_step_cache_loc", None) is not None:
                tail_tokens += int(req.mtp_prev_step_cache_loc.numel())
            if getattr(req, "mtp_overlap_prev_step_cache_loc", None) is not None:
                tail_tokens += int(req.mtp_overlap_prev_step_cache_loc.numel())

            if tail_tokens <= 0:
                continue

            req_count += 1
            token_count += tail_tokens
            if len(sample_rids) < 4:
                sample_rids.append(rid)

        return req_count, token_count, sample_rids

    def _get_batch_uncached_size(self: Scheduler, batch: ScheduleBatch) -> int:
        ret = 0
        for req in batch.reqs:
            assert req.kv_committed_freed == req.kv_overallocated_freed
            uncached_len = 0
            if not req.kv_committed_freed:
                allocated_len = req.kv_allocated_len
                if self.page_size > 1:
                    allocated_len = ceil_align(allocated_len, self.page_size)
                    assert req.cache_protected_len % self.page_size == 0
                uncached_len = allocated_len - req.cache_protected_len

            ret += uncached_len

        return ret

    def self_check_during_busy(self: Scheduler):
        current_batch: ScheduleBatch = self.last_batch

        if current_batch is None:
            return

        spec_topk = self.server_args.speculative_eagle_topk or 1
        if spec_topk > 1:
            warnings.warn(
                "Runtime memory check (busy) is not supported when speculation topk > 1."
            )
            return

        _, _, available_size, evictable_size = self._get_token_info()
        protected_size = self.tree_cache.protected_size()

        uncached_size = self._get_batch_uncached_size(current_batch)

        if (
            current_batch.forward_mode.is_extend()
            and self.running_batch is not None
            and not self.running_batch.is_empty()
        ):
            uncached_size += self._get_batch_uncached_size(self.running_batch)

        if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get() > 1:
            log_msg = f"[Mem Check (BUSY)] {available_size=}, {evictable_size=}, {protected_size=}, {uncached_size=}"
            logger.info(log_msg)

        total_tokens = available_size + evictable_size + protected_size + uncached_size
        assert (
            total_tokens == self.max_total_num_tokens
        ), f"Mem Leak Detected! {total_tokens=} vs {self.max_total_num_tokens=}"

    def _check_req_pool(self: Scheduler):
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            req_total_size = (
                self.req_to_token_pool.size + self.req_to_token_pool.pre_alloc_size
            )
        else:
            req_total_size = self.req_to_token_pool.size

        if len(self.req_to_token_pool.free_slots) != req_total_size:
            msg = (
                "req_to_token_pool memory leak detected!"
                f"available_size={len(self.req_to_token_pool.free_slots)}, "
                f"total_size={self.req_to_token_pool.size}\n"
            )
            raise_error_or_warn(
                self,
                envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.get(),
                "count_req_pool_leak_warnings",
                msg,
            )

    def check_memory(self: Scheduler):
        if self.is_hybrid_swa:
            memory_leak, token_msg = self._check_hybrid_memory()
        elif self.is_hybrid_ssm and self.tree_cache.supports_mamba():
            memory_leak, token_msg = self._check_mamba_memory()
        else:
            memory_leak, token_msg = self._check_radix_cache_memory()
            if memory_leak:
                recovered = self._try_reclaim_orphan_radix_tokens()
                if recovered > 0:
                    memory_leak, token_msg = self._check_radix_cache_memory()
            if memory_leak:
                repaired_overlap = self._try_repair_radix_free_cache_overlap()
                if repaired_overlap > 0:
                    memory_leak, token_msg = self._check_radix_cache_memory()
            if memory_leak:
                repaired = self._recompute_radix_lock_counters()
                if repaired:
                    memory_leak, token_msg = self._check_radix_cache_memory()

        if memory_leak:
            msg = "token_to_kv_pool_allocator memory leak detected! " f"{token_msg}"
            raise_error_or_warn(
                self,
                envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.get(),
                "count_memory_leak_warnings",
                msg,
            )

        self._check_req_pool()

        if (
            self.enable_metrics
            and self.current_scheduler_metrics_enabled
            and time.perf_counter() > self.metrics_collector.last_log_time + 30
        ):
            # During idle time, also collect metrics every 30 seconds.
            if self.is_hybrid_swa:
                (
                    full_num_used,
                    swa_num_used,
                    full_token_usage,
                    swa_token_usage,
                    _,
                    _,
                    _,
                    _,
                ) = self._get_swa_token_info()
                num_used = max(full_num_used, swa_num_used)
                token_usage = max(full_token_usage, swa_token_usage)
            elif self.is_hybrid_ssm:
                (
                    num_used,
                    _,
                    token_usage,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = self._get_mamba_token_info()
            else:
                num_used, token_usage, _, _ = self._get_token_info()
            num_running_reqs = len(self.running_batch.reqs)
            self.stats.num_running_reqs = num_running_reqs
            self.stats.num_used_tokens = num_used
            self.stats.token_usage = round(token_usage, 2)
            self.stats.gen_throughput = 0
            self.stats.num_queue_reqs = len(self.waiting_queue)
            self.stats.num_grammar_queue_reqs = len(self.grammar_manager)
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.stats.num_prefill_prealloc_queue_reqs = len(
                    self.disagg_prefill_bootstrap_queue.queue
                )
                self.stats.num_prefill_inflight_queue_reqs = len(
                    self.disagg_prefill_inflight_queue
                )
            if self.disaggregation_mode == DisaggregationMode.DECODE:
                self.stats.num_decode_prealloc_queue_reqs = len(
                    self.disagg_decode_prealloc_queue.queue
                )
                self.stats.num_decode_transfer_queue_reqs = len(
                    self.disagg_decode_transfer_queue.queue
                )
            self.metrics_collector.log_stats(self.stats)
        self._publish_kv_events()

    def check_tree_cache(self: Scheduler):
        if (
            self.tree_cache.is_tree_cache()
            and (self.is_hybrid_swa and self.tree_cache.supports_swa())
            or (self.is_hybrid_ssm and self.tree_cache.supports_mamba())
        ):
            self.tree_cache.sanity_check()

    def self_check_during_idle(self: Scheduler):
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            if len(self.disagg_prefill_inflight_queue) > 0:
                self._idle_quiescent_streak = 0
                self._idle_quiescent_summary_logged = False
                return
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            queue_size = (
                len(self.waiting_queue)
                + len(self.disagg_decode_transfer_queue.queue)
                + len(self.disagg_decode_prealloc_queue.queue)
            )
            if self.server_args.disaggregation_decode_enable_offload_kvcache:
                queue_size += len(self.decode_offload_manager.ongoing_offload)
            if queue_size:
                self._idle_quiescent_streak = 0
                self._idle_quiescent_summary_logged = False
                return
        else:
            has_inflight_batches = not self._is_no_request()
            has_waiting = len(self.waiting_queue) > 0
            has_running = not self.running_batch.is_empty()
            has_chunked = self.chunked_req is not None
            has_dllm_staging = bool(
                self.dllm_config is not None and self.dllm_manager.any_staging_reqs()
            )
            has_grammar_queued = len(self.grammar_manager) > 0
            if (
                has_inflight_batches
                or has_waiting
                or has_running
                or has_chunked
                or has_dllm_staging
                or has_grammar_queued
            ):
                self._idle_quiescent_streak = 0
                self._idle_quiescent_summary_logged = False
                self._idle_epoch_has_activity = True
                now = time.perf_counter()
                last_log_ts = getattr(self, "_idle_mem_check_skip_log_ts", 0.0)
                if now - last_log_ts >= IDLE_MEM_CHECK_SKIP_LOG_INTERVAL_SEC:
                    (
                        mtp_tail_req_count,
                        mtp_tail_token_count,
                        mtp_tail_sample_rids,
                    ) = self._summarize_outstanding_mtp_tail_refs()
                    logger.info(
                        "Skip idle memory check because scheduler is not quiescent. "
                        f"{has_inflight_batches=}, {has_waiting=}, {has_running=}, "
                        f"{has_chunked=}, {has_dllm_staging=}, {has_grammar_queued=}, "
                        f"{mtp_tail_req_count=}, "
                        f"{mtp_tail_token_count=}, {mtp_tail_sample_rids=}"
                    )
                    self._idle_mem_check_skip_log_ts = now
                return
            self._idle_quiescent_streak = (
                getattr(self, "_idle_quiescent_streak", 0) + 1
            )
            if (
                self._idle_quiescent_streak == 1
                and getattr(self, "_idle_epoch_has_activity", False)
                and not getattr(self, "_idle_quiescent_summary_logged", False)
            ):
                self._log_idle_radix_invariant_summary()
                self._idle_quiescent_summary_logged = True
                self._idle_epoch_has_activity = False
            if self._idle_quiescent_streak < 2:
                # Give one extra idle tick for allocator/radix accounting to settle
                # after the last active batch drains.
                return

        self.check_memory()
        self.check_tree_cache()
        self.new_token_ratio = self.init_new_token_ratio
        self.maybe_sleep_on_idle()


def create_scheduler_watchdog(
    scheduler: Scheduler, watchdog_timeout: float, soft: bool = False
) -> WatchdogRaw:
    def dump_info() -> str:
        if scheduler.is_initializing or disable_request_logging():
            return ""
        if scheduler.is_hybrid_swa:
            _, info_msg = scheduler._check_hybrid_memory()
        elif scheduler.is_hybrid_ssm and scheduler.tree_cache.supports_mamba():
            _, info_msg = scheduler._check_mamba_memory()
        else:
            _, info_msg = scheduler._check_radix_cache_memory()
        return (
            f"{scheduler.cur_batch.batch_size()=}\n"
            f"{scheduler.cur_batch.reqs=}\n"
            f"{info_msg}"
        )

    return WatchdogRaw(
        debug_name="Scheduler",
        get_counter=lambda: getattr(scheduler, "forward_ct", 0),
        is_active=lambda: scheduler.is_initializing
        or getattr(scheduler, "cur_batch", None) is not None,
        watchdog_timeout=watchdog_timeout,
        soft=soft,
        dump_info=dump_info,
    )
