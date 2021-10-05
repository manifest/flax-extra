r"""The array reshaping operator."""

from dataclasses import dataclass
from jax import numpy as jnp
from flax import linen as nn

Array = jnp.ndarray


@dataclass
class ReshapeBatch:
    r"""Changes the shape of an array preserving its batch dimension.

    >>> from jax import numpy as jnp
    >>> from flax_extra import operator as op
    >>> reshape = op.ReshapeBatch(shape=(1, 6))
    >>> reshape(jnp.ones((2, 2, 3))).shape
    (2, 1, 6)
    """

    shape: tuple[int, ...]
    r"""an array shape excluding its batch dimension."""

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        batch_size = inputs.shape[0]
        return jnp.reshape(inputs, (batch_size, *self.shape))  # type: ignore
