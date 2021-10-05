r"""Operator is just a function."""

from flax_extra.operator._rearrange import Rearrange
from flax_extra.operator._reshape import ReshapeBatch
from flax_extra.operator._shift_right import ShiftRight

__all__ = [
    "Rearrange",
    "ReshapeBatch",
    "ShiftRight",
]
