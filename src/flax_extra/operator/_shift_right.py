"""The shift-right operator."""

from typing import Any
from dataclasses import dataclass
from jax import numpy as jnp

Array = Any


@dataclass
class ShiftRight:
    """Inserts padding to shift the sequence.

    >>> from jax import numpy as jnp
    >>> from flax_extra import operator as op
    >>> shift_right = op.ShiftRight(axis=0, n_positions=1)
    >>> shift_right(jnp.array([1, 2, 3]))
    DeviceArray([0, 1, 2], dtype=int32)
    """

    n_positions: int = 1
    """a number of positions to shift."""

    pad_id: int = 0
    """a padding identifier to insert."""

    axis: int = 0
    """the operation will be performed along this axis."""

    def __call__(self, inputs: Array) -> Array:
        def pad_width() -> Array:
            acc = [(0, 0)] * len(inputs.shape)
            acc[self.axis] = (self.n_positions, 0)
            return acc

        padded = jnp.pad(
            inputs,
            pad_width=pad_width(),
            mode="constant",
            constant_values=self.pad_id,
        )
        return jnp.take(
            padded,
            jnp.arange(padded.shape[self.axis] - self.n_positions),
            axis=self.axis,
        )
