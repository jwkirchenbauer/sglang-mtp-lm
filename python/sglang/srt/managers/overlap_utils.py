from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import torch

from sglang.srt.speculative.spec_utils import spec_need_hidden_states
from sglang.srt.utils import get_compiler_backend, is_npu

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ModelWorkerBatch
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.speculative.eagle_info import EagleDraftInput
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

_is_npu = is_npu()


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def _resolve_future_token_ids(input_ids, future_token_ids_map, future_ref_stride):
    future_refs = torch.clamp(-input_ids, min=0)
    row_indices = torch.div(future_refs, future_ref_stride, rounding_mode="trunc")
    col_indices = torch.remainder(future_refs, future_ref_stride)
    input_ids[:] = torch.where(
        input_ids < 0,
        future_token_ids_map[row_indices, col_indices],
        input_ids,
    )


@dataclass
class FutureIndices:
    indices: torch.Tensor
    interval: Optional[slice] = None


class FutureMap:
    # Use a fixed stride for future token reference encoding.
    # Encoded reference: -(future_index * stride + token_offset)
    # token_offset is 0 for normal decode, and [0, effective_k) for MTP windows.
    FUTURE_TOKEN_REF_STRIDE = 128

    def __init__(
        self,
        max_running_requests: int,
        chunked_prefill_size: int,
        context_len: int,
        device: torch.device,
        spec_algo: Optional[SpeculativeAlgorithm] = None,
    ):
        # FIXME: the calculation of future_limit and future_buffer_len maybe too conservative
        self.future_ct = 0

        # Circular buffer layout (wraps in this order):
        # Running decode batch -> Prefill chunk 1 -> ... -> Prefill chunk N
        # A running decode batch's result will be resolved after all prefill chunks are done.
        # reserve `max_num_chunks` extra future slots on top of `max_running_requests * 3`.
        max_num_chunks = (
            (context_len + chunked_prefill_size - 1) // chunked_prefill_size
            if chunked_prefill_size
            else 0
        )
        self.future_limit = max_running_requests * (3 + max_num_chunks)
        # Adding 2 * max_running_requests to future_limit ensures the buffer is sufficiently large.
        self.future_buffer_len = self.future_limit + 2 * max_running_requests
        self.device = device
        self.spec_algo = spec_algo

        if self.spec_algo.is_none():
            # For non-speculative decoding, keep a token window per future entry.
            # The first column is used by normal decode; additional columns are used
            # by adaptive/static MTP overlap relay.
            self.buf_initialized = True
            self.token_ids_buf = torch.empty(
                (self.future_buffer_len, self.FUTURE_TOKEN_REF_STRIDE),
                dtype=torch.int64,
                device=self.device,
            )
            self.mtp_effective_k_buf = torch.ones(
                (self.future_buffer_len,), dtype=torch.int32, device=self.device
            )
            self.mtp_window_len_buf = torch.ones(
                (self.future_buffer_len,), dtype=torch.int32, device=self.device
            )
        else:
            # For speculative decoding, we lazily initialize the buffers
            # This is to make the shape derivation easier.
            self.buf_initialized = False

    def _lazy_init_buf(self, draft_input: EagleDraftInput):
        self.buf_initialized = True

        # Get a reference for each tensor
        topk_p0 = draft_input.topk_p[0]
        topk_index0 = draft_input.topk_index[0]
        verified_id0 = draft_input.verified_id[0]
        new_seq_lens0 = draft_input.new_seq_lens[0]

        self.topk_p_buf = torch.empty(
            (self.future_buffer_len, *topk_p0.shape),
            dtype=topk_p0.dtype,
            device=self.device,
        )
        self.topk_index_buf = torch.empty(
            (self.future_buffer_len, *topk_index0.shape),
            dtype=topk_index0.dtype,
            device=self.device,
        )
        self.verified_id_buf = torch.empty(
            (self.future_buffer_len, *verified_id0.shape),
            dtype=verified_id0.dtype,
            device=self.device,
        )
        self.new_seq_lens_buf = torch.empty(
            (self.future_buffer_len, *new_seq_lens0.shape),
            dtype=new_seq_lens0.dtype,
            device=self.device,
        )

        if spec_need_hidden_states():
            hidden_states0 = draft_input.hidden_states[0]
            self.hidden_states_buf = torch.empty(
                (self.future_buffer_len, *hidden_states0.shape),
                dtype=hidden_states0.dtype,
                device=self.device,
            )

    def alloc_future_indices(self, bs: int) -> FutureIndices:
        """Update the circular buffer pointer and allocate future indices."""
        cur_future_ct = self.future_ct
        self.future_ct = (cur_future_ct + bs) % self.future_limit
        start = cur_future_ct + 1
        end = cur_future_ct + 1 + bs
        indices = torch.arange(start, end, dtype=torch.int64, device=self.device)
        return FutureIndices(indices=indices, interval=slice(start, end))

    def resolve_future(self, model_worker_batch: ModelWorkerBatch):
        if self.spec_algo.is_none():
            _resolve_future_token_ids(
                model_worker_batch.input_ids,
                self.token_ids_buf,
                self.FUTURE_TOKEN_REF_STRIDE,
            )
        else:
            # TODO(lsyin): write future indices into spec_info.future_indices
            draft_input: EagleDraftInput = model_worker_batch.spec_info
            if draft_input is None:
                # FIXME(lsyin): No future exists, only for prefill batch, not compatible with mixed mode
                return
            indices = draft_input.future_indices.indices
            # The indices tensor was allocated on the default stream but is
            # used here on the forward stream. Meanwhile, the old spec_info
            # holding this tensor will lose all Python references (replaced at
            # model_worker_batch.spec_info and batch.spec_info), so the
            # caching allocator (torch GC) could reclaim the memory before
            # the GPU finishes reading it.
            indices.record_stream(torch.get_device_module(self.device).current_stream())
            draft_input.topk_p = self.topk_p_buf[indices]
            draft_input.topk_index = self.topk_index_buf[indices]
            draft_input.verified_id = self.verified_id_buf[indices]
            draft_input.new_seq_lens = self.new_seq_lens_buf[indices]
            if spec_need_hidden_states():
                draft_input.hidden_states = self.hidden_states_buf[indices]

    def is_empty_slice(self, s: slice) -> bool:
        start, stop, step = s.indices(self.future_buffer_len)
        if step > 0:
            return start >= stop
        else:
            return start <= stop

    def store_to_map(
        self, future_indices: FutureIndices, batch_result: GenerationBatchResult
    ):
        if self.spec_algo.is_none():
            intv = future_indices.interval
            next_token_ids = batch_result.next_token_ids
            if isinstance(next_token_ids, list):
                if len(next_token_ids) == 0:
                    raise ValueError("Empty next_token_ids in overlap future map store.")
                if isinstance(next_token_ids[0], torch.Tensor):
                    next_token_ids = torch.stack(next_token_ids)
                else:
                    next_token_ids = torch.tensor(
                        next_token_ids, dtype=torch.int64, device=self.device
                    )
            if next_token_ids.ndim == 1:
                token_windows = next_token_ids.unsqueeze(1)
            elif next_token_ids.ndim == 2:
                token_windows = next_token_ids
            else:
                raise ValueError(
                    "Unsupported next_token_ids rank for overlap future map: "
                    f"{next_token_ids.ndim}."
                )

            window_len = int(token_windows.shape[1])
            if window_len > self.FUTURE_TOKEN_REF_STRIDE:
                raise ValueError(
                    "MTP future token window exceeds overlap reference stride. "
                    f"window_len={window_len}, stride={self.FUTURE_TOKEN_REF_STRIDE}."
                )

            token_windows = token_windows.to(
                device=self.device, dtype=torch.int64, non_blocking=True
            )
            self.token_ids_buf[intv].zero_()
            self.token_ids_buf[intv, :window_len] = token_windows
            self.mtp_window_len_buf[intv] = window_len

            effective_k = None
            if batch_result.logits_output is not None:
                effective_k = getattr(
                    batch_result.logits_output, "mtp_effective_k_per_req", None
                )
            if effective_k is None:
                self.mtp_effective_k_buf[intv] = 1
            else:
                if not isinstance(effective_k, torch.Tensor):
                    effective_k = torch.tensor(
                        effective_k, dtype=torch.int32, device=self.device
                    )
                else:
                    effective_k = effective_k.to(
                        device=self.device, dtype=torch.int32, non_blocking=True
                    )
                self.mtp_effective_k_buf[intv] = torch.clamp(
                    effective_k, min=1, max=window_len
                )
        else:
            draft_input: EagleDraftInput = batch_result.next_draft_input
            self.store_to_map_for_new_batch(future_indices, draft_input)

    def _make_future_token_refs(
        self,
        indices: torch.Tensor,
        token_offset: Union[int, torch.Tensor] = 0,
    ) -> torch.Tensor:
        if isinstance(token_offset, int):
            offsets = torch.full_like(indices, token_offset)
        else:
            offsets = token_offset.to(device=indices.device, dtype=indices.dtype)
        if torch.any(offsets >= self.FUTURE_TOKEN_REF_STRIDE) or torch.any(offsets < 0):
            raise ValueError(
                "Invalid future token offset for overlap reference encoding: "
                f"offsets must be in [0, {self.FUTURE_TOKEN_REF_STRIDE - 1}]."
            )
        return -(
            indices.to(dtype=torch.int64) * self.FUTURE_TOKEN_REF_STRIDE
            + offsets.to(dtype=torch.int64)
        )

    def make_single_token_future_refs(self, indices: torch.Tensor) -> torch.Tensor:
        return self._make_future_token_refs(indices, token_offset=0)

    def build_mtp_pending_token_refs(
        self,
        future_indices: FutureIndices,
        effective_k_per_req: Sequence[int],
    ) -> List[List[int]]:
        indices = future_indices.indices.detach().to("cpu").tolist()
        if len(indices) != len(effective_k_per_req):
            raise ValueError(
                "Mismatch between future indices and effective-k metadata for MTP overlap relay: "
                f"indices={len(indices)}, effective_k={len(effective_k_per_req)}."
            )

        pending_refs: List[List[int]] = []
        for future_idx, effective_k in zip(indices, effective_k_per_req):
            ek = int(effective_k)
            if ek < 1:
                ek = 1
            if ek > self.FUTURE_TOKEN_REF_STRIDE:
                raise ValueError(
                    "MTP effective-k exceeds overlap reference stride. "
                    f"effective_k={ek}, stride={self.FUTURE_TOKEN_REF_STRIDE}."
                )
            refs = [
                int(-(future_idx * self.FUTURE_TOKEN_REF_STRIDE + token_offset))
                for token_offset in range(ek)
            ]
            pending_refs.append(refs)
        return pending_refs

    def store_to_map_for_new_batch(
        self, future_indices: FutureIndices, draft_input: EagleDraftInput
    ):
        intv = future_indices.interval
        if self.is_empty_slice(intv):
            # idle indices in dp attention do not need store info
            return

        if not self.buf_initialized:
            self._lazy_init_buf(draft_input)

        self.topk_p_buf[intv] = draft_input.topk_p
        self.topk_index_buf[intv] = draft_input.topk_index
        self.verified_id_buf[intv] = draft_input.verified_id
        self.new_seq_lens_buf[intv] = draft_input.new_seq_lens
        if spec_need_hidden_states():
            self.hidden_states_buf[intv] = draft_input.hidden_states
