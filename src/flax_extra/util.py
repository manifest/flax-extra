"""General utility functions."""

from typing import Any, Optional
import jax
from jax import numpy as jnp

Array = jnp.ndarray
ArrayTree = Any


def originate(tree: ArrayTree) -> ArrayTree:
    """Originate replicated pytree object.

    Assuming that each replica of the (per device) replicated pytree is
    representing a copy of the origin pytree, the function simply retrieves
    the first replica of the replicated pytree.

    Args:
        tree: replicated pytree.

    Returns:
        a regular pytree.
    """
    return jax.tree_map(lambda x: x[0], tree)


def batch_per_device(inputs: Array, n_devices: Optional[int] = None) -> Array:
    """Splits the head axis of an array evenly across the number of devices.

    The function changes input array shape as follows:

    ::

        (head, *tail) -> (n_devices, reduced_head // n_devices, *tail)

    Args:
        inputs: an array.
        n_devices: number of devices. Defaults to all available devices.

    Returns:
        an array with items of the leading axis mapped to the number of devices.
    """
    if n_devices is None:
        n_devices = jax.local_device_count()

    head, *tail = inputs.shape
    return inputs.reshape(n_devices, head // n_devices, *tail)  # type: ignore
