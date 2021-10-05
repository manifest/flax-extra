r"""Trainable positional padding."""

from typing import List
from jax import numpy as jnp
from flax import linen as nn
from flax_extra.layer._trainable_positional_encoding import TrainablePositionalEncoding

Array = jnp.ndarray
Positions = List[int]


class TrainablePositionalPadding(nn.Module):
    r"""Trainable positional padding.

    Learns a padding vector and applies it to a vector at each position
    (i.e. time step) of the sequence to pad up vector's dimension
    to desired value.

    .. math::

        \begin{aligned}
            & \textrm{TrainablePositionalPadding}( \\
            & \quad x \in \sR^{\nBatchSize \times T \times d} \\
            & \quad \_ \\
            & \quad w \in  \sR^{1 \times d^{\prime}} \\
            & ) \\
            & \rightarrow \sR^{\nBatchSize \times T \times d + d^{\prime}}
        \end{aligned}

    Args:
        inputs: a sequence.

    Returns:
        a sequence with padding along feature dimension.
    """

    d_max: int
    r"""a desired dimension for vectors in the sequence."""

    @nn.compact
    def __call__(self, inputs: Array) -> Array:  # type: ignore[override] # pylint: disable=arguments-differ
        batch_size = inputs.shape[0]
        d_input = inputs.shape[-1]
        d_padding = self.d_max - d_input
        padding = TrainablePositionalEncoding(
            seqlen=1,
            dimension=d_padding,
        )

        padded_input_shape = list(inputs.shape)
        padded_input_shape[-1] = d_padding
        pad_tokens = jnp.broadcast_to(
            padding(batch_size=batch_size),
            shape=padded_input_shape,
        )

        padded_inputs = jnp.concatenate(
            [inputs, pad_tokens],
            axis=-1,
        )
        return padded_inputs  # type: ignore
