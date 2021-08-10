"""Operator is just a function."""

from redex.operator import add, sub, identity
from flax_extra.operator._shift_right import ShiftRight

__all__ = [
    "add",
    "sub",
    "identity",
    "ShiftRight",
]
