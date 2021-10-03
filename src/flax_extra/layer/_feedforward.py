"""FeedForward layer."""

from typing import Any, Callable, Iterable
import jax
from jax import numpy as jnp
from jax.random import KeyArray
from redex import combinator as cb
from flax import linen as nn

Array = jnp.ndarray
Shape = Iterable[int]
Dtype = Any
InitFn = Callable[[KeyArray, Shape, Dtype], Array]


class FeedForward(nn.Module):
    """A dense layer encouraging sparsity."""

    widening_factor: int = 4
    """determines a hidden dimension of the layer as a product of
    the inputs dimension by the factor value."""

    activation: Callable[..., Array] = jax.nn.gelu
    """a nonlinear function."""

    kernel_init: InitFn = nn.initializers.lecun_normal()
    """a weights initializer."""

    @nn.compact
    # pylint: disable=arguments-differ
    def __call__(self, inputs: Array) -> Array:  # type: ignore[override]
        d_output = inputs.shape[-1]
        return cb.serial(
            nn.Dense(
                features=self.widening_factor * d_output,
                kernel_init=self.kernel_init,  # type: ignore
            ),
            self.activation,
            nn.Dense(
                features=d_output,
                kernel_init=self.kernel_init,  # type: ignore
            ),
        )(
            inputs,  # type: ignore
        )
