r"""The shift-right operator."""

from typing import List
from dataclasses import dataclass
from jax import numpy as jnp

Array = jnp.ndarray


@dataclass
class ShiftRight:
    r"""Inserts padding to shift the sequence.

    >>> from jax import numpy as jnp
    >>> from flax_extra import operator as op
    >>> shift_right = op.ShiftRight(axis=0, n_positions=1)
    >>> shift_right(jnp.array([1, 2, 3]))
    DeviceArray([0, 1, 2], dtype=int32)
    """

    n_positions: int = 1
    r"""a number of positions to shift."""

    pad_id: int = 0
    r"""a padding identifier to insert."""

    axis: int = 0
    r"""the operation will be performed along this axis."""

    def __call__(self, inputs: Array) -> Array:
        def pad_width() -> List[tuple[int, int]]:
            acc = [(0, 0)] * len(inputs.shape)
            acc[self.axis] = (self.n_positions, 0)
            return acc

        padded = jnp.pad(
            inputs,
            pad_width=pad_width(),
            mode="constant",
            constant_values=self.pad_id,
        )
        return jnp.take(  # type: ignore
            padded,
            jnp.arange(padded.shape[self.axis] - self.n_positions),
            axis=self.axis,
        )
