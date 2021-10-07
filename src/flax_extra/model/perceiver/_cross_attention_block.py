r"""Cross attention block of the Perceiver model."""

from typing import Any, Optional
from functools import partial
import jax.numpy as jnp
from flax import linen as nn
from flax_extra import combinator as cb
from flax_extra.layer._feedforward import FeedForward, FeedForwardCt
from flax_extra.layer._attention import KVQAttention, KVQAttentionCt

Array = jnp.ndarray
Precision = Any


class CrossAttentionBlock(nn.Module):
    r"""A block of a cross-attention module and following
    feed-forward module.

    .. math::

        \begin{aligned}
            & \textrm{SelfAttentionBlock}( \\
            & \quad x_{q} \in \sR^{\nBatchSize \times \nSeqLen_{x_{q}} \times d_{x_{q}}} \\
            & \quad x_{kv} \in \sR^{\nBatchSize \times \nSeqLen_{x_{kv}} \times d_{x_{kv}}} \\
            & \quad \_ \\
            & \quad \theta \gets 2 \times LayerNorm() \\
            & \quad \theta \gets CrossAttentionBlock() \\
            & \quad \theta \gets LayerNorm() \\
            & \quad \theta \gets FeedForward() \\
            & ) \\
            & \rightarrow \sR^{\nBatchSize \times \nSeqLen_{x_{q}} \times d_{x_{kv}}}
        \end{aligned}

    Args:
        inputs_q: inputs for the query.
        inputs_kv: inputs for key and value.
        mask: a mask tensor with boolean values indicating whether
            a particular query attends to a particular key.

    Returns:
        latent features.
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
    Consider omitting the residual if the semantics of query and output
    are different, e.g. if queries are positions and outputs are pixels."""

    precision: Optional[Precision] = None
    r"""numerical precision of the computation.
    See :attr:`jax.lax.Precision` for details."""

    @nn.compact
    def __call__(  # type: ignore[override] # pylint: disable=arguments-differ
        self,
        inputs_q: Array,
        inputs_kv: Array,
        mask: Optional[Array],
    ) -> Array:
        # Optionally include a residual to the query.
        # Consider omitting the residual if the semantics of query and output
        # are different, e.g. if queries are positions and outputs are pixels.
        maybe_residual = cb.residual if self.use_q_residual else cb.serial

        block = cb.serial(
            maybe_residual(  # type: ignore
                cb.parallel(nn.LayerNorm(epsilon=1e-5), nn.LayerNorm(epsilon=1e-5)),
                partial(
                    self.attention,
                    deterministic=self.deterministic,
                    precision=self.precision,
                )(),
                nn.Dropout(
                    rate=self.dropout_rate,
                    deterministic=self.deterministic,
                ),
            ),
            cb.residual(
                nn.LayerNorm(epsilon=1e-5),
                self.feed_forward(),
                nn.Dropout(
                    rate=self.dropout_rate,
                    deterministic=self.deterministic,
                ),
            ),
        )

        return block(inputs_q, inputs_kv, mask)  # type: ignore
