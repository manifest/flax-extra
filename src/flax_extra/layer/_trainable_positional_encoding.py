"""Trainable positional encoding."""

from typing import Any, Callable, List, Sequence, Optional, Union
import jax
import jax.numpy as jnp
from jax.core import NamedShape
from jax.random import KeyArray
from flax import linen as nn

Array = jnp.ndarray
Shape = Optional[Union[Sequence[int], NamedShape]]
Dtype = Any
Positions = List[int]
InitFn = Callable[[KeyArray, Shape, Dtype], Array]

## TODO: move to initializers.
def _truncated_normal(
    stddev: float = 1.0,
    mean: float = 0.0,
    dtype: Dtype = jnp.float32,
) -> InitFn:
    def init(key: KeyArray, shape: Shape, dtype: Dtype = dtype) -> Array:
        target_mean = jax.lax.convert_element_type(mean, dtype)
        target_stddev = jax.lax.convert_element_type(stddev, dtype)
        unscaled = jax.random.truncated_normal(key, -2.0, 2.0, shape, dtype)
        return target_stddev * unscaled + target_mean  # type: ignore

    return init


class TrainablePositionalEncoding(nn.Module):
    """Trainable positional encoding."""

    seqlen: int
    """a length of the sequence."""

    dimension: int
    """a disared dimension of the positional encoding."""

    kernel_init: InitFn = _truncated_normal(stddev=0.02)
    """weights initializer."""

    dtype: Dtype = jnp.float32
    """desired data type."""

    @nn.compact
    # pylint: disable=arguments-differ
    def __call__(  # type: ignore[override]
        self,
        batch_size: int,
        output_positions: Optional[Positions] = None,
    ) -> Array:
        if output_positions is not None:
            raise ValueError(
                "TrainablePositionalEncoding don't accept output positions."
            )

        positional_encodings = self.param(
            "positional_encoding",
            self.kernel_init,  # type: ignore
            (self.seqlen, self.dimension),
            self.dtype,
        )
        return jnp.broadcast_to(  # type: ignore
            positional_encodings[None, :, :],
            (batch_size, *positional_encodings.shape),
        )
