r"""The Perceiver processor."""

from typing import Any, Optional
from functools import partial
import jax.numpy as jnp
from flax import linen as nn
from flax_extra import combinator as cb
from flax_extra.layer._feedforward import FeedForward, FeedForwardCt
from flax_extra.layer._attention import SelfAttention, SelfAttentionCt
from flax_extra.model.perceiver._self_attention_block import SelfAttentionBlock

Array = jnp.ndarray
Precision = Any


class Processor(nn.Module):
    r"""Self-attends latent features.

    Self-attention blocks grouped in shards. Weights of the blocks
    are shared across shards.

    .. math::

        \begin{aligned}
            & \textrm{Processor}( \\
            & \quad z \in \sR^{\nBatchSize \times \nSeqLen_{z} \times d_{z}} \\
            & \quad \_ \\
            & \quad \theta \gets n_{blocks} \times SelfAttentionBlock() \\
            & ) \\
            & \rightarrow \sR^{\nBatchSize \times \nSeqLen_{z} \times d_{z}}
        \end{aligned}

    Args:
        latents: an array of latent features.

    Returns:
        an array of latent features.
    """

    n_shards: int = 8
    r"""a number of shards."""

    n_blocks: int = 6
    r"""a number of self-attention blocks building up a single shard."""

    attention: SelfAttentionCt = SelfAttention
    r"""a type of the self-attention."""

    feed_forward: FeedForwardCt = FeedForward
    r"""a type of the feed-forward."""

    dropout_rate: float = 0.0
    r"""probababilistic rate for dropout."""

    deterministic: bool = True
    r"""whether to perform deterministically or not."""

    precision: Optional[Precision] = None
    r"""numerical precision of the computation.
    See :attr:`jax.lax.Precision` for details."""

    @nn.compact
    def __call__(self, latents: Array) -> Array:  # type: ignore[override] # pylint: disable=arguments-differ
        self_attention_blocks = [
            partial(
                SelfAttentionBlock(
                    attention=self.attention,
                    feed_forward=self.feed_forward,
                    dropout_rate=self.dropout_rate,
                    deterministic=self.deterministic,
                    precision=self.precision,
                ),
                mask=None,
            )
            for _ in range(self.n_blocks)
        ]

        # Processor block (latent-space Transformer: a number of self-attentions).
        blocks = cb.serial(
            [block for _ in range(self.n_shards) for block in self_attention_blocks]
        )

        return blocks(latents)  # type: ignore
