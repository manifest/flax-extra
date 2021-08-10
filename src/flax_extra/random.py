"""Random number generation."""

from typing import List, Mapping, Generator, Union
from jax import random, numpy as jnp


Sequence = Generator[jnp.ndarray, None, None]


def into_collection(key: jnp.ndarray, labels: List[str]) -> Mapping[str, jnp.ndarray]:
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


def into_sequence(key: jnp.ndarray) -> Sequence:
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