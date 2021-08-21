"""Batch types and data processing functions."""

from typing import Union, Generator
from functools import partial
from jax import numpy as jnp
import redex
from flax_extra import util

Array = jnp.ndarray
Inputs = tuple[Array, ...]  # type: ignore
Targets = Inputs
Batch = tuple[Inputs, Targets]
UnnormalizedInputs = Union[Inputs, Array]
UnnormalizedBatch = Union[Batch, tuple[UnnormalizedInputs, UnnormalizedInputs]]
DataStream = Generator[UnnormalizedBatch, None, None]


def normalize_batch(batch: UnnormalizedBatch) -> Batch:
    """Converts a batch to the normalized form.

    In normalized form, a `Batch` is a tuple that consist of:
        `Inputs`: a single or multiple arguments to model's `apply` function.
        `Targets`: arguments for a loss function. Targets could be an empty tuple in the case of unsupervised learning.

    Args:
        batch: a batch to normalize.

    Returns:
        a normalized batch of arrays.
    """
    return tuple(map(redex.util.expand_to_tuple, batch))  # type: ignore


def normalize_batch_per_device(batch: UnnormalizedBatch, n_devices: int) -> Batch:
    """Converts a batch to the normalized form splitting head axis of
    inputs and targets evenly across the number of devices.

    Args:
        batch: a batch to normalize.

    Returns:
        a normalized batch of shared arrays.
    """
    batch_per_device = partial(util.batch_per_device, n_devices=n_devices)

    def normalize_items(group: UnnormalizedInputs) -> Inputs:
        return tuple(map(batch_per_device, redex.util.expand_to_tuple(group)))

    return tuple(map(normalize_items, batch))  # type: ignore
