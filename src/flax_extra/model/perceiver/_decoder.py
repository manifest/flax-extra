r"""The Perceiver decoder."""

from typing import Any, Optional
import jax.numpy as jnp
from flax import linen as nn
from flax_extra.layer._feedforward import FeedForward, FeedForwardCt
from flax_extra.layer._attention import (
    KVQAttention,
    KVQAttentionCt,
    cross_attention_mask,
)
from flax_extra.model.perceiver._cross_attention_block import CrossAttentionBlock


Array = jnp.ndarray
Precision = Any


class Decoder(nn.Module):
    r"""A single cross-attention block projects latent features
    to outputs of desired dimension.

    .. math::

        \begin{aligned}
            & \textrm{Decoder}( \\
            & \quad z \in \sR^{\nBatchSize \times \nSeqLen_{z} \times d_{z}} \\
            & \quad o \in \sR^{\nBatchSize \times \nSeqLen_{o} \times d_{o}} \\
            & \quad mask \in \sR^{\nBatchSize \times \nSeqLen_{z} \times \nSeqLen_{o}} \\
            & \quad \_ \\
            & \quad \theta \gets CrossAttentionBlock() \\
            & ) \\
            & \rightarrow \sR^{\nBatchSize \times \nSeqLen_{o} \times d_{o}}
        \end{aligned}

    Args:
        latents: latent features.
        targets: inputs for decoder's query of desired dimension.
        target_mask: a padding mask indicating at which positions values
            of the targets are valid.

    Returns:
        outputs with the same dimension as inputs for decoder's query.
    """

    attention: KVQAttentionCt = KVQAttention
    r"""a type of the cross-attention."""

    feed_forward: FeedForwardCt = FeedForward
    r"""a type of the feed-forward."""

    dropout_rate: float = 0.0
    r"""probababilistic rate for dropout."""

    deterministic: bool = True
    r"""whether to perform deterministically or not."""

    use_q_residual: int = False
    r"""whether to include a residual to the query.
    Consider omitting the residual if the semantics of decoder's query
    and outputs are different (e.g. if queries are positions and outputs
    are pixels)."""

    precision: Optional[Precision] = None
    r"""numerical precision of the computation.
    See :attr:`jax.lax.Precision` for details."""

    @nn.compact
    def __call__(  # type: ignore[override] # pylint: disable=arguments-differ
        self,
        latents: Array,
        targets: Array,
        target_mask: Optional[Array],
    ) -> Array:
        attention_mask = None
        if target_mask is not None:
            attention_mask = cross_attention_mask(
                mask_q=target_mask,
                mask_kv=jnp.ones(latents.shape[:2], dtype=jnp.int32),
            )

        return CrossAttentionBlock(
            attention=self.attention,
            feed_forward=self.feed_forward,
            dropout_rate=self.dropout_rate,
            deterministic=self.deterministic,
            use_q_residual=self.use_q_residual,
            precision=self.precision,
        )(inputs_q=targets, inputs_kv=latents, mask=attention_mask)
