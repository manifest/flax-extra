"""The concatenate combinator."""

from redex.function import Signature
from redex.stack import stackmethod, verify_stack_size, Stack
from redex.combinator._base import Combinator
from jax import numpy as jnp

# pylint: disable=too-few-public-methods
class Concatenate(Combinator):
    """The concatenate combinator."""

    axis: int
    """an axis on which to concatenate arrays."""

    @stackmethod
    def __call__(self, stack: Stack) -> Stack:
        verify_stack_size(self, stack, self.signature)
        n_in = self.signature.n_in
        concatenated = jnp.concatenate(stack[:n_in], self.axis)
        return (concatenated, *stack[self.signature.n_in :])


def concatenate(axis: int = -1, n_in: int = 2) -> Concatenate:
    """Creates a concatenate combinator.

    Concatenates input arrays on desired axis.

    >>> from jax import numpy as jnp
    >>> from flax_extra import combinator as cb
    >>> concatenate = cb.concatenate()
    >>> concatenate(jnp.array([1, 2]), jnp.array([3]))
    DeviceArray([1, 2, 3], dtype=int32)

    Args:
        n_in: a number of inputs.

    Returns:
        a combinator.
    """
    return Concatenate(axis=axis, signature=Signature(n_in=n_in, n_out=1))
