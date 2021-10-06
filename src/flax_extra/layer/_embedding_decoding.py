r"""Embedding decoding."""

from jax import numpy as jnp
from flax import linen as nn

Array = jnp.ndarray


class EmbedDecoding(nn.Module):
    r"""Projects encoded vectors at each position (i.e. time step)
    of the sequence to its original representation using projection
    tensor of the embeding encoder.

    .. math::

        \begin{aligned}
            & \textrm{Decoding}( \\
            & \quad y \in \sR^{\nBatchSize \times \nSeqLen \times d} \\
            & \quad \_ \\
            & \quad b_{x} \in \sR^{d_{x}} \\
            & ) \\
            & \rightarrow \sR^{\nBatchSize \times \nSeqLen \times d_{x}}
        \end{aligned}

    Args:
        outputs: model's outputs.

    Returns:
        decoded outputs.
    """

    embedding: Array
    r"""embedding projection tensor of the embedding encoder
    (e.g. :class:`flax.linen.Embed`)."""

    @nn.compact
    def __call__(self, embeddings: Array) -> Array:  # type: ignore[override] # pylint: disable=arguments-differ
        vocab_size, d_model = self.embedding.shape
        batch_size, seq_len, _ = embeddings.shape
        output = jnp.matmul(
            jnp.reshape(embeddings, (-1, d_model)),
            jnp.transpose(self.embedding),
        )
        bias = self.param(
            "bias",
            nn.zeros,
            (vocab_size,),
            jnp.float32,
        )
        output = output + bias
        return jnp.reshape(output, (batch_size, seq_len, vocab_size))  # type: ignore
