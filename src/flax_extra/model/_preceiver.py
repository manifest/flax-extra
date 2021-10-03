"""Layers of the perceiver model."""

from typing import Any, Optional, Callable
from functools import partial
from redex import combinator as cb
import jax.numpy as jnp
from flax import linen as nn
from flax_extra.layer._feedforward import FeedForward
from flax_extra.layer._attention import (
    KVQAttention,
    SelfAttention,
    make_cross_attention_mask,
)

Array = jnp.ndarray
Precision = Any
CrossAttentionT = Callable[..., nn.Module]
SelfAttentionT = Callable[..., nn.Module]
FeedForwardT = Callable[..., nn.Module]


class CrossAttentionBlock(nn.Module):
    """A block of a cross-attention layer and a feed-forward layer."""

    attention: CrossAttentionT = KVQAttention
    """a type of the cross-attention layer."""

    feed_forward: FeedForwardT = FeedForward
    """a type of the feed-forward layer."""

    dropout_rate: float = 0.
    """probababilistic rate for dropout."""

    deterministic: bool = True
    """whether to perform deterministically or not."""

    use_q_residual: bool = True
    """whether to include a residual to the query.
    Consider omitting the residual if the semantics of query and output
    are different, e.g. if queries are positions and outputs are pixels."""

    precision: Optional[Precision] = None
    """numerical precision of the computation.
    See :attr:`jax.lax.Precision` for details."""

    @nn.compact
    # pylint: disable=arguments-differ
    def __call__(  # type: ignore[override]
        self,
        inputs_q: Array,
        inputs_kv: Array,
        mask: Optional[Array],
    ) -> Array:
        """Computes a cross-attention between two sequences.

        Args:
            inputs_q: inputs for the query.
            inputs_kv: inputs for key and value.
            mask: an attention mask indicating which attention values are valid.

        Returns:
            an array of latent features.
        """
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


class LatentTransformerBlock(nn.Module):
    """Transformer-style self-attention blocks."""

    attention: SelfAttentionT = SelfAttention
    """a type of the self-attention layer."""

    feed_forward: FeedForwardT = FeedForward
    """a type of the feed-forward layer."""

    dropout_rate: float = 0.
    """probababilistic rate for dropout."""

    deterministic: bool = True
    """whether to perform deterministically or not."""

    precision: Optional[Precision] = None
    """numerical precision of the computation.
    See :attr:`jax.lax.Precision` for details."""

    @nn.compact
    # pylint: disable=arguments-differ
    def __call__(  # type: ignore[override]
        self,
        latents: Array,
        mask: Optional[Array],
    ) -> Array:
        """Iteratively self-attends to the latent features.

        Args:
            latents: an array of latent features.
            mask: an attention mask indicating which attention values are valid.

        Returns:
            an array of latent features.
        """
        block = cb.serial(
            cb.residual(
                nn.LayerNorm(epsilon=1e-5),
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
                nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic),
            ),
        )

        return block(latents, mask)  # type: ignore


class PerceiverEncoder(nn.Module):
    """A single cross-attention block projects a high-dimensional
    input array to a fixed-dimensional latent bottleneck.

    Outputs have the same dimension as encoder's query.
    """

    attention: CrossAttentionT = KVQAttention
    """a type of the cross-attention layer."""

    feed_forward: FeedForwardT = FeedForward
    """a type of the feed-forward layer."""

    dropout_rate: float = 0.
    """probababilistic rate for dropout."""

    deterministic: bool = True
    """whether to perform deterministically or not."""

    use_q_residual: bool = True
    """whether to include a residual to the query.
    Consider omitting the residual if the semantics of encoder's query
    and latent features are different (e.g. if queries are positions
    and latents are pixels)."""

    precision: Optional[Precision] = None
    """numerical precision of the computation.
    See :attr:`jax.lax.Precision` for details."""

    @nn.compact
    # pylint: disable=arguments-differ
    def __call__(  # type: ignore[override]
        self,
        inputs: Array,
        latents: Array,
        input_mask: Optional[Array],
    ) -> Array:
        """Computes a cross-attention between inputs and encoder's query
        producing latent features.

        Args:
            inputs: a high-dimensional input array.
            latents: inputs for encoder's query of desired dimension.
            input_mask: a padding mask indicating at which positions values
                of the inputs are valid.

        Returns:
            an array of latent features with the same dimension as encoder's
            query.
        """
        attention_mask = None
        if input_mask is not None:
            attention_mask = make_cross_attention_mask(
                q_mask=jnp.ones(latents.shape[:2], dtype=jnp.int32),
                kv_mask=input_mask,
            )

        return CrossAttentionBlock(
            attention=self.attention,
            feed_forward=self.feed_forward,
            dropout_rate=self.dropout_rate,
            deterministic=self.deterministic,
            use_q_residual=self.use_q_residual,
            precision=self.precision,
        )(inputs_q=latents, inputs_kv=inputs, mask=attention_mask)


class PerceiverProcessor(nn.Module):
    """Multiple self-attention blocks grouped in shards.

    Weights of the grouped blocks are shared across shards.
    """

    n_shards: int = 8
    """a number of shards."""

    n_blocks: int = 6
    """a number of self-attention blocks building up a single shard."""

    attention: SelfAttentionT = SelfAttention
    """a type of the self-attention layer."""

    feed_forward: FeedForwardT = FeedForward
    """a type of the feed-forward layer."""

    dropout_rate: float = 0.
    """probababilistic rate for dropout."""

    deterministic: bool = True
    """whether to perform deterministically or not."""

    precision: Optional[Precision] = None
    """numerical precision of the computation.
    See :attr:`jax.lax.Precision` for details."""

    @nn.compact
    # pylint: disable=arguments-differ
    def __call__(self, latents: Array) -> Array:  # type: ignore[override]
        """Iteratively self-attends the latent features to themselves.

        Args:
            latents: an array of latent features.

        Returns:
            an array of latent features.
        """
        self_attention_blocks = [
            partial(
                LatentTransformerBlock(
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


class PerceiverDecoder(nn.Module):
    """A single cross-attention block projects latent features
    to outputs of desired dimension.

    Outputs have the same dimension as decoder's query.
    """

    attention: CrossAttentionT = KVQAttention
    """a type of the cross-attention layer."""

    feed_forward: FeedForwardT = FeedForward
    """a type of the feed-forward layer."""

    dropout_rate: float = 0.
    """probababilistic rate for dropout."""

    deterministic: bool = True
    """whether to perform deterministically or not."""

    use_q_residual: int = False
    """whether to include a residual to the query.
    Consider omitting the residual if the semantics of decoder's query
    and outputs are different (e.g. if queries are positions and outputs
    are pixels)."""

    precision: Optional[Precision] = None
    """numerical precision of the computation.
    See :attr:`jax.lax.Precision` for details."""

    @nn.compact
    # pylint: disable=arguments-differ
    def __call__(  # type: ignore[override]
        self,
        latents: Array,
        targets: Array,
        target_mask: Optional[Array],
    ) -> Array:
        """Computes a cross-attention between latent features and decoder's
        query producing outputs.

        Args:
            latents: an array of latent features.
            targets: inputs for decoder's query of desired dimension.
            target_mask: a padding mask indicating at which positions values
                of the targets are valid.

        Returns:
            an array of outputs with the same dimension as decoder's query.
        """
        attention_mask = None
        if target_mask is not None:
            attention_mask = make_cross_attention_mask(
                q_mask=target_mask,
                kv_mask=jnp.ones(latents.shape[:2], dtype=jnp.int32),
            )

        return CrossAttentionBlock(
            attention=self.attention,
            feed_forward=self.feed_forward,
            dropout_rate=self.dropout_rate,
            deterministic=self.deterministic,
            use_q_residual=self.use_q_residual,
            precision=self.precision,
        )(inputs_q=targets, inputs_kv=latents, mask=attention_mask)
