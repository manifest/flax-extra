r"""FeedForward layer."""

from typing import Any, Callable, Sequence, Optional, Union
import jax
from jax import numpy as jnp
from jax.core import NamedShape
from jax.random import KeyArray
from flax import linen as nn
from flax_extra import combinator as cb

Array = jnp.ndarray
Shape = Optional[Union[Sequence[int], NamedShape]]
Dtype = Any
InitFn = Callable[[KeyArray, Shape, Dtype], Array]


class FeedForward(nn.Module):
    r"""Learns a dense vector representation.

    .. math::

        \begin{aligned}
            & \textrm{FeedForward}( \\
            & \quad x \in \sR^{\nBatchSize \times \dots d} \\
            & \quad \_ \\
            & \quad w_{h} \in \sR^{d_{x} \times d \cdot \text{factor}} \\
            & \quad w_{o} \in \sR^{d \cdot \text{factor} \times d} \\
            & \quad b_{h} \in \sR^{d \cdot \text{factor}} \\
            & \quad b_{o} \in \sR^{d} \\
            & ) \\
            & \rightarrow \sR^{\nBatchSize \times \dots d}
        \end{aligned}

    Args:
        inputs: a tensor.

    Returns:
        a tensor of the same shape as the input tensor.
    """

    widening_factor: int = 1
    r"""determines a hidden dimension of the layer as a product of
    the inputs dimension by the factor value."""

    activation: Callable[..., Array] = jax.nn.gelu
    r"""a nonlinear function."""

    kernel_init: InitFn = nn.initializers.lecun_normal()
    r"""an initializer for weights."""

    @nn.compact
    def __call__(self, inputs: Array) -> Array:  # type: ignore[override] # pylint: disable=arguments-differ
        d_output = inputs.shape[-1]
        block = cb.serial(
            nn.Dense(
                features=self.widening_factor * d_output,
                kernel_init=self.kernel_init,  # type: ignore
            ),
            self.activation,
            nn.Dense(
                features=d_output,
                kernel_init=self.kernel_init,  # type: ignore
            ),
        )

        return block(inputs)  # type: ignore


FeedForwardFn = Callable[[Array], Array]
FeedForwardCt = Callable[..., FeedForwardFn]
