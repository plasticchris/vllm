"""Inkling-only logit workspace reduction; preserve torch operation order."""

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from .tuning import OPTIMIZED


def softcap_owned_logits(logits, soft_cap, scale=1.0):
    """Only for newly-produced, inference-only logits (never caller inputs)."""
    if soft_cap is not None:
        logits.div_(soft_cap)
        logits.tanh_()
        logits.mul_(soft_cap)
    if scale != 1.0:
        logits.mul_(scale)
    return logits


class InklingLogitsProcessor(LogitsProcessor):
    def forward(self, lm_head, hidden_states, embedding_bias=None, skip_gather=False):
        if not OPTIMIZED or self.logits_as_input:
            return super().forward(lm_head, hidden_states, embedding_bias, skip_gather)
        # _get_logits owns the new GEMM/collective output; it may be a
        # strided view after vocabulary padding is cropped. No caller-owned
        # hidden state or parameter is modified.
        logits = self._get_logits(hidden_states, lm_head, embedding_bias, skip_gather)
        if logits is None:
            return None
        return softcap_owned_logits(logits, self.soft_cap, self.scale)
