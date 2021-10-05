r"""Trainable positional maskin."""

import jax
from jax import numpy as jnp
from flax import linen as nn
from flax_extra.layer._trainable_positional_encoding import TrainablePositionalEncoding

Array = jnp.ndarray


class TrainablePositionalMasking(nn.Module):
    r"""Learns a mask vector and applies it to a vector at each position
    (i.e. time step) of the sequence according specified probabilistic rate.

    .. math::

        \begin{aligned}
            & \textrm{TrainablePositionalMasking}( \\
            & \quad x \in \sR^{\nBatchSize \times T \times d} \\
            & \quad \_ \\
            & \quad w \in \sR^{1 \times d} \\
            & ) \\
            & \rightarrow \sR^{\nBatchSize \times T \times d}
        \end{aligned}

    Args:
        inputs: a sequence.

    Returns:
        a masked sequence.
    """

    rate: float
    r"""a probability for a vector at each position being masked."""

    @nn.compact
    def __call__(self, inputs: Array) -> Array:  # type: ignore[override] # pylint: disable=arguments-differ
        batch_size, seqlen_input, d_input = inputs.shape
        masking = TrainablePositionalEncoding(
            seqlen=1,
            dimension=d_input,
        )

        mask_weight = jax.random.bernoulli(
            self.make_rng("params"),
            self.rate,
            shape=(batch_size, seqlen_input),
        )[:, :, jnp.newaxis]

        mask_token = masking(batch_size=batch_size)
        masked_inputs = (1 - mask_weight) * inputs + mask_weight * mask_token
        return masked_inputs  # type: ignore
