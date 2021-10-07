r"""The Perceiver encoder."""

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


class Encoder(nn.Module):
    r"""A single cross-attention block projects a high-dimensional
    input vectors to a fixed-dimensional vectors of latent features.

    .. math::

        \begin{aligned}
            & \textrm{Encoder}( \\
            & \quad x \in \sR^{\nBatchSize \times \nSeqLen_{x} \times d_{x}} \\
            & \quad z \in \sR^{\nBatchSize \times \nSeqLen_{z} \times d_{z}} \\
            & \quad mask \in \sR^{\nBatchSize \times \nSeqLen_{z} \times \nSeqLen_{x}} \\
            & \quad \_ \\
            & \quad \theta \gets CrossAttentionBlock() \\
            & ) \\
            & \rightarrow \sR^{\nBatchSize \times \nSeqLen_{z} \times d_{z}}
        \end{aligned}

    Args:
        inputs: a high-dimensional input array.
        latents: inputs for encoder's query of desired dimension.
        input_mask: a padding mask indicating at which positions values
            of the inputs are valid.

    Returns:
        latent features with the same dimension as inputs for encoder's query.
    """

    attention: KVQAttentionCt = KVQAttention
    r"""a type of the cross-attention."""

    feed_forward: FeedForwardCt = FeedForward
    r"""a type of the feed-forward."""

    dropout_rate: float = 0.0
    r"""probababilistic rate for dropout."""

    deterministic: bool = True
    r"""whether to perform deterministically or not."""

    use_q_residual: bool = True
    r"""whether to include a residual to the query.
    Consider omitting the residual if the semantics of encoder's query
    and latent features are different (e.g. if queries are positions
    and latents are pixels)."""

    precision: Optional[Precision] = None
    r"""numerical precision of the computation.
    See :attr:`jax.lax.Precision` for details."""

    @nn.compact
    def __call__(  # type: ignore[override] # pylint: disable=arguments-differ
        self,
        inputs: Array,
        latents: Array,
        input_mask: Optional[Array],
    ) -> Array:
        attention_mask = None
        if input_mask is not None:
            attention_mask = cross_attention_mask(
                mask_q=jnp.ones(latents.shape[:2], dtype=jnp.int32),
                mask_kv=input_mask,
            )

        return CrossAttentionBlock(
            attention=self.attention,
            feed_forward=self.feed_forward,
            dropout_rate=self.dropout_rate,
            deterministic=self.deterministic,
            use_q_residual=self.use_q_residual,
            precision=self.precision,
        )(inputs_q=latents, inputs_kv=inputs, mask=attention_mask)
