# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Sampling parameters for text generation."""

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Union

# sre_parse is deprecated in Python 3.11+, use re._parser instead
try:
    import re._parser as sre_parse
except ImportError:
    import sre_parse  # Python < 3.11

_SAMPLING_EPS = 1e-6
TOP_K_ALL = 1 << 30

logger = logging.getLogger(__name__)


class SamplingParams:
    """
    The sampling parameters.

    See docs/backend/sampling_params.md or
    https://docs.sglang.io/backend/sampling_params.html
    for the documentation.
    """

    def __init__(
        self,
        max_new_tokens: int = 128,
        stop: Optional[Union[str, List[str]]] = None,
        stop_token_ids: Optional[List[int]] = None,
        stop_regex: Optional[Union[str, List[str]]] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        min_new_tokens: int = 0,
        n: int = 1,
        json_schema: Optional[str] = None,
        regex: Optional[str] = None,
        ebnf: Optional[str] = None,
        structural_tag: Optional[str] = None,
        ignore_eos: bool = False,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
        no_stop_trim: bool = False,
        custom_params: Optional[Dict[str, Any]] = None,
        stream_interval: Optional[int] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        sampling_seed: Optional[int] = None,
        mtp_enabled: bool = False,
        mtp_k: int = 1,
        mtp_strategy: Optional[Union[str, Sequence[Any]]] = None,
        mtp_conf_threshold: Optional[float] = None,
        mtp_temperature: Optional[float] = None,
        mtp_mask_id: Optional[int] = None,
        mtp_min_mask_id: Optional[int] = None,
        mtp_max_mask_id: Optional[int] = None,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.stop_strs = stop
        if stop_token_ids:
            self.stop_token_ids = set(stop_token_ids)
        else:
            self.stop_token_ids = None
        self.stop_regex_strs = stop_regex
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        self.min_new_tokens = min_new_tokens
        self.regex = regex
        self.n = n
        self.json_schema = json_schema
        self.ebnf = ebnf
        self.structural_tag = structural_tag
        self.ignore_eos = ignore_eos
        self.skip_special_tokens = skip_special_tokens
        self.spaces_between_special_tokens = spaces_between_special_tokens
        self.no_stop_trim = no_stop_trim
        self.custom_params = custom_params
        self.stream_interval = stream_interval
        self.logit_bias = logit_bias
        self.sampling_seed = sampling_seed
        self.mtp_enabled = mtp_enabled
        self.mtp_k = mtp_k
        self.mtp_strategy = mtp_strategy
        self.mtp_conf_threshold = mtp_conf_threshold
        self.mtp_temperature = mtp_temperature
        self.mtp_mask_id = mtp_mask_id
        self.mtp_min_mask_id = mtp_min_mask_id
        self.mtp_max_mask_id = mtp_max_mask_id
        self.mtp_strategy_kind: Optional[str] = None

        # Process some special cases
        if 0 <= self.temperature < _SAMPLING_EPS:
            # top_k = 1 means greedy sampling
            self.temperature = 1.0
            self.top_k = 1
        if self.top_k == -1:
            self.top_k = TOP_K_ALL  # whole vocabulary

    def _parse_mtp_strategy(self):
        self.mtp_strategy_kind = None
        if self.mtp_strategy is None:
            return

        # Canonical strategy input is a list of basic literals, e.g.
        # ["conf_adapt", 0.9]. Keep a minimal string fallback for legacy callers.
        strategy_name: Optional[str] = None
        threshold_val: Optional[float] = None

        if isinstance(self.mtp_strategy, str):
            strategy = self.mtp_strategy.strip().lower()
            if strategy in {"", "none"}:
                self.mtp_strategy = None
                return
            strategy_name = strategy
        elif isinstance(self.mtp_strategy, Sequence):
            strategy_items = list(self.mtp_strategy)
            if len(strategy_items) == 0:
                self.mtp_strategy = None
                return
            name_raw = strategy_items[0]
            if not isinstance(name_raw, str):
                raise ValueError(
                    "mtp_strategy list must start with a strategy name string, "
                    f"got {type(name_raw).__name__}."
                )
            strategy_name = name_raw.strip().lower()
            if strategy_name in {"", "none"}:
                self.mtp_strategy = None
                return
            if len(strategy_items) > 2:
                raise ValueError(
                    "mtp_strategy list supports at most two items: "
                    "['conf_adapt', <threshold>]."
                )
            if len(strategy_items) == 2:
                threshold_raw = strategy_items[1]
                try:
                    threshold_val = float(threshold_raw)
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        "Invalid conf_adapt threshold in mtp_strategy list: "
                        f"{self.mtp_strategy!r}."
                    ) from e
        else:
            raise ValueError(
                "mtp_strategy must be a list like ['conf_adapt', <threshold>] "
                f"or a legacy strategy string, got {type(self.mtp_strategy).__name__}."
            )

        if strategy_name == "conf_adapt":
            if threshold_val is None:
                if self.mtp_conf_threshold is None:
                    raise ValueError(
                        "mtp_strategy=['conf_adapt', <threshold>] requires a numeric threshold "
                        "(or set mtp_conf_threshold)."
                    )
                threshold_val = float(self.mtp_conf_threshold)

            if self.mtp_conf_threshold is not None and not math.isclose(
                float(self.mtp_conf_threshold),
                threshold_val,
                rel_tol=0.0,
                abs_tol=1e-8,
            ):
                raise ValueError(
                    "Conflicting MTP threshold values from mtp_strategy and mtp_conf_threshold."
                )

            self.mtp_strategy_kind = "conf_adapt"
            self.mtp_conf_threshold = threshold_val
            self.mtp_strategy = ["conf_adapt", threshold_val]
            return

        raise ValueError(
            "Unsupported mtp_strategy. Expected ['conf_adapt', <threshold>] "
            f"(or legacy 'conf_adapt'), got {self.mtp_strategy!r}."
        )

    def verify(self, vocab_size):
        self._parse_mtp_strategy()

        if self.temperature < 0.0:
            raise ValueError(
                f"temperature must be non-negative, got {self.temperature}."
            )
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}.")
        if self.top_k < 1 or self.top_k == -1:
            raise ValueError(
                f"top_k must be -1 (disable) or at least 1, got {self.top_k}."
            )
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError(
                "frequency_penalty must be in [-2, 2], got "
                f"{self.frequency_penalty}."
            )
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError(
                "presence_penalty must be in [-2, 2], got " f"{self.presence_penalty}."
            )
        if not 0.0 <= self.repetition_penalty <= 2.0:
            raise ValueError(
                "repetition_penalty must be in [0, 2], got "
                f"{self.repetition_penalty}."
            )
        if not 0 <= self.min_new_tokens:
            raise ValueError(
                f"min_new_tokens must be in [0, max_new_tokens], got "
                f"{self.min_new_tokens}."
            )
        if self.max_new_tokens is not None:
            if self.max_new_tokens < 0:
                raise ValueError(
                    f"max_new_tokens must be at least 0, got {self.max_new_tokens}."
                )
            if not self.min_new_tokens <= self.max_new_tokens:
                raise ValueError(
                    f"min_new_tokens must be in [0, max_new_tokens({self.max_new_tokens})], got "
                    f"{self.min_new_tokens}."
                )
        if self.logit_bias is not None:
            for token_id in self.logit_bias:
                if not 0 <= int(token_id) < vocab_size:
                    raise ValueError(
                        f"logit_bias must has keys in [0, {vocab_size - 1}], got "
                        f"{token_id}."
                    )

        if self.mtp_k < 1:
            raise ValueError(f"mtp_k must be >= 1, got {self.mtp_k}.")
        if self.mtp_conf_threshold is not None and not (
            0.0 <= self.mtp_conf_threshold <= 1.0
        ):
            raise ValueError(
                f"mtp_conf_threshold must be in [0, 1], got {self.mtp_conf_threshold}."
            )
        if self.mtp_temperature is not None and self.mtp_temperature < 0.0:
            raise ValueError(
                f"mtp_temperature must be non-negative, got {self.mtp_temperature}."
            )
        if self.mtp_min_mask_id is not None and self.mtp_max_mask_id is not None:
            if self.mtp_min_mask_id > self.mtp_max_mask_id:
                raise ValueError(
                    "mtp_min_mask_id must be <= mtp_max_mask_id, got "
                    f"{self.mtp_min_mask_id} > {self.mtp_max_mask_id}."
                )
        if self.mtp_enabled and self.mtp_k > 1:
            if self.mtp_mask_id is None and self.mtp_min_mask_id is None:
                raise ValueError(
                    "MTP decode requires either mtp_mask_id or mtp_min_mask_id when mtp_k > 1."
                )
        if self.mtp_enabled:
            # First-pass MTP runtime support is greedy-only.
            if self.n != 1:
                raise ValueError("mtp_enabled currently requires n == 1.")
            if self.top_k != 1:
                raise ValueError("mtp_enabled currently requires greedy decoding (top_k == 1).")
            if self.top_p != 1.0 or self.min_p != 0.0:
                raise ValueError(
                    "mtp_enabled currently does not support top-p/min-p sampling."
                )
            if (
                self.frequency_penalty != 0.0
                or self.presence_penalty != 0.0
                or self.repetition_penalty != 1.0
            ):
                raise ValueError(
                    "mtp_enabled currently does not support repetition/frequency/presence penalties."
                )
        if self.mtp_strategy_kind is not None and not self.mtp_enabled:
            raise ValueError("mtp_strategy requires mtp_enabled=true.")

        grammars = [
            self.json_schema,
            self.regex,
            self.ebnf,
        ]  # since mutually exclusive, only one can be set
        if sum(x is not None for x in grammars) > 1:
            raise ValueError("Only one of regex, json_schema, or ebnf can be set.")

    def normalize(self, tokenizer):
        # Process stop strings
        if self.stop_strs is None:
            self.stop_strs = []
            self.stop_str_max_len = 0
        else:
            if isinstance(self.stop_strs, str):
                self.stop_strs = [self.stop_strs]

            stop_str_max_len = 0
            for stop_str in self.stop_strs:
                if tokenizer is not None:
                    stop_str_ids = tokenizer.encode(stop_str, add_special_tokens=False)
                    stop_str_max_len = max(stop_str_max_len, len(stop_str_ids))
                else:
                    stop_str_max_len = max(stop_str_max_len, len(stop_str))
            self.stop_str_max_len = stop_str_max_len

        # Process stop regex strings
        if self.stop_regex_strs is None:
            self.stop_regex_strs = []
            self.stop_regex_max_len = 0
        else:
            if isinstance(self.stop_regex_strs, str):
                self.stop_regex_strs = [self.stop_regex_strs]

            stop_regex_max_len = 0
            for stop_regex in self.stop_regex_strs:
                stop_regex_max_len = max(
                    stop_regex_max_len, get_max_seq_length(stop_regex)
                )

            self.stop_regex_max_len = stop_regex_max_len


# This function gets a strict upperbound on the maximum number of tokens that would need
# to be buffered to match the input regex string
# NOTE: in the worst case, one character that needs to be buffered corresponds to one
# token
def get_max_seq_length(regex_str: str):
    return _max_length_from_subpattern(sre_parse.parse(regex_str))


MAX_LEN = 2**30


def _max_length_from_subpattern(subpattern: sre_parse.SubPattern):
    total = 0
    for token, value in subpattern:
        if token in {
            sre_parse.LITERAL,  # `value` is any one character
            sre_parse.IN,  # Any character within `value`
            sre_parse.ANY,  # "."
        }:
            total += 1
        elif token == sre_parse.SUBPATTERN:
            # EG: (a\d+) ->
            # [(SUBPATTERN,
            #   (1, 0, 0, [(LITERAL, 97),
            #              (MAX_REPEAT, (1, MAXREPEAT, [(IN, [(CATEGORY, CATEGORY_DIGIT)])]))]))]
            _, _, _, inner_subpattern = value
            total += _max_length_from_subpattern(inner_subpattern)
        elif token == sre_parse.BRANCH:
            _, branches = value
            total += max(_max_length_from_subpattern(branch) for branch in branches)
        elif token in {sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT}:
            _, max_num_repeat, inner_subpattern = value
            if max_num_repeat == sre_parse.MAXREPEAT:
                total += MAX_LEN
            else:
                total += max_num_repeat * _max_length_from_subpattern(inner_subpattern)
        elif token == sre_parse.AT:
            # These are zero-width assertions like ^, $, and \b that don't add to the max
            # length
            total += 0
        else:
            logger.warning(f"Got unhandled regex token: {token}")

            total += MAX_LEN

    return total
