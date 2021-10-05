r"""The array rearranging operator."""

from dataclasses import dataclass, field
from typing import Mapping
from jax import numpy as jnp
from flax import linen as nn
import einops

Array = jnp.ndarray


@dataclass
class Rearrange:
    r"""Rearranges the shape of an array according to the pattern.

    >>> from jax import numpy as jnp
    >>> from flax_extra import operator as op
    >>> rearrange = op.Rearrange(pattern="(a da) db -> a (da db)", bindings=dict(da=2))
    >>> rearrange(jnp.ones((4,3))).shape
    (2, 6)
    """

    pattern: str
    r"""a rearrangement pattern."""

    bindings: Mapping[str, int] = field(default_factory=dict)
    r"""bindings for dimensions specified in the pattern."""

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        return einops.rearrange(inputs, pattern=self.pattern, **self.bindings)  # type: ignore
