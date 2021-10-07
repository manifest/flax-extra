r"""Trainable decoding."""

from typing import cast, Callable
from jax import numpy as jnp
from flax import linen as nn
from flax_extra import combinator as cb

Array = jnp.ndarray

EmbeddingDecodingFn = Callable[[Array], Array]
EmbeddingDecodingCt = Callable[..., EmbeddingDecodingFn]
PostprocessingFn = Callable[[Array], Array]
PostprocessingCt = Callable[..., PostprocessingFn]


class Decoding(nn.Module):
    r"""Decodes a vector at each position (i.e. time step)
    of the sequence using an optional embedding decoding
    and postprocessing step.

    The postprocessing is intended to change shape
    of the sequence to desired.

    .. math::

        \begin{aligned}
            & \textrm{Decoding}( \\
            & \quad y \in \sR^{\nBatchSize \times \nSeqLen \times d} \\
            & \quad \_ \\
            & \quad \theta \gets EmbeddingDecoding() \\
            & \quad \theta \gets Postprocessing() \\
            & ) \\
            & \rightarrow \sR^{\nBatchSize \times \dots d^{\prime}}
        \end{aligned}

    Args:
        outputs: model's outputs.

    Returns:
        decoded outputs.
    """

    embedding_decoding: EmbeddingDecodingCt = cast(
        EmbeddingDecodingCt,
        lambda: cb.identity(n_in=1),
    )
    r"""a type of the embedding decoder
    (e.g. :class:`flax_extra.layer.EmbedDecoding` or
    :class:`flax.linen.Dense`)."""

    postprocessing: PostprocessingCt = cast(
        PostprocessingCt,
        lambda: cb.identity(n_in=1),
    )
    r"""a constructor for a module or function that accepts decoded outputs
    of as its single argument and returns the preprocessed outputs as a
    single output."""

    @nn.compact
    def __call__(self, outputs: Array) -> Array:  # type: ignore[override] # pylint: disable=arguments-differ
        # Decode embeddings.
        decoded_outputs = self.embedding_decoding()(outputs)

        # Postprocess.
        postprocessed_outputs = self.postprocessing()(decoded_outputs)
        return postprocessed_outputs


DecodingCt = Callable[..., Decoding]
