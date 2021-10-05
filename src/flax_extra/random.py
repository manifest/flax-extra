r"""Random number generation."""

from typing import List, Mapping, Generator, Optional, Union
import jax
from jax import random, numpy as jnp
from jax.random import KeyArray


Array = jnp.ndarray
KeyGenerator = Generator[KeyArray, None, None]


def split_per_device(key: KeyArray, n_devices: Optional[int] = None) -> KeyArray:
    r"""Splits a random number generator key according to the number of devices."""
    if n_devices is None:
        n_devices = jax.local_device_count()

    return jax.random.split(key, num=n_devices)


def into_collection(key: KeyArray, labels: List[str]) -> Mapping[str, KeyArray]:
    r"""Splits a random number generator key into a few.
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


def into_sequence(key: KeyArray) -> KeyGenerator:
    r"""Creates a generator of random number generator keys.

    Args:
        key: an initial generator key.

    Yields:
        a new random number generator key.
    """
    initial_key = key
    while True:
        initial_key, key = random.split(key=initial_key, num=2)
        yield key


def sequence(seed: int, num: int = 1) -> Union[KeyGenerator, List[KeyGenerator]]:
    r"""Creates generators of random number generator keys.

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
