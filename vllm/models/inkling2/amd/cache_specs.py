"""Explicit positional contract for Inkling2 cache-splice planning.

Relative-bias attention has no rotary keys. Convolution caches contain raw
stream inputs, also with no positional rotation. Both remain approximate
causal-prefix reuse (not semantic forgetting or equivalent to cold replay).
"""

from dataclasses import dataclass
from typing import ClassVar
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec


@dataclass(frozen=True)
class RelativeFullAttentionSpec(FullAttentionSpec):
    leyline_position_encoding: ClassVar[str] = "relative"


@dataclass(frozen=True)
class RelativeSlidingWindowSpec(SlidingWindowSpec):
    leyline_position_encoding: ClassVar[str] = "relative"


@dataclass(frozen=True)
class RotaryFullAttentionSpec(FullAttentionSpec):
    leyline_position_encoding: ClassVar[str] = "rotary"


@dataclass(frozen=True)
class RotarySlidingWindowSpec(SlidingWindowSpec):
    leyline_position_encoding: ClassVar[str] = "rotary"
