"""Random number generation."""

from typing import List, Mapping, Generator, Optional, Union
import jax
from jax import random, numpy as jnp


Array = jnp.ndarray
Sequence = Generator[Array, None, None]


def split_per_device(key: Array, n_devices: Optional[int] = None) -> Array:
    """Splits a random number generator key according to the number of devices."""
    if n_devices is None:
        n_devices = jax.local_device_count()

    return jax.random.split(key, num=n_devices)


def into_collection(key: Array, labels: List[str]) -> Mapping[str, Array]:
    """Splits a random number generator key into a few.
    New keys are associated with provided collection labels.

    Args:
        labels: a collection of labels.
        key: an initial random number generator key.

    Returns:
        a dictionary with random number generator keys as values
        and collection labels as keys.
    """
    keys = {labels[i]: k for i, k in enumerate(random.split(key=key, num=len(labels)))}
    return keys


def into_sequence(key: Array) -> Sequence:
    """Creates a generator of random number generator keys.

    Args:
        key: an initial generator key.

    Yields:
        a new random number generator key.
    """
    initial_key = key
    while True:
        initial_key, key = random.split(key=initial_key, num=2)
        yield key


def sequence(seed: int, num: int = 1) -> Union[Sequence, List[Sequence]]:
    """Creates generators of random number generator keys.

    .. code-block:: python

        from flax_extra import random
        rnkeyg = random.sequence(seed=0)
        next(rnkeyg)

    Args:
        seed: a seed integer to create an initial generator.
        num: the number of sequences to produce.

    Yields:
        a new random number generator key.
    """
    initial_key = random.PRNGKey(seed=seed)
    if num == 1:
        return into_sequence(key=initial_key)

    keys = random.split(key=initial_key, num=num)
    return [into_sequence(key) for key in keys]
