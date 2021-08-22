"""Batch types and data processing functions.

The batch, inputs, and targets in **Training API** usually
represented using unnormalized form for user convenience.

During construction of :class:`TrainLoop` data get normalized.

In that form, a batch is a tuple, `((x,...), (y,...))`, that consist of:

- inputs, `(x,...)`, a tuple of a single array or multiple arrays.
    Inputs get passed to model's `apply(x,...)` function as arguments.
- targets, `(y,...)`, a tuple of arbitrary size or an empty tuple.
    Targets along with model outputs, `(o,...)`, get passed to a
    `loss(o,...,y,...)` function as arguments.
"""

from typing import Union, Generator
from functools import partial
from jax import numpy as jnp
import redex
from flax_extra import util

Array = jnp.ndarray

Inputs = tuple[Array, ...]  # type: ignore
Targets = Inputs
Outputs = Inputs
Batch = tuple[Inputs, Targets]
UnnormalizedInputs = Union[Inputs, Array]
UnnormalizedBatch = Union[Batch, tuple[UnnormalizedInputs, UnnormalizedInputs]]
DataStream = Generator[UnnormalizedBatch, None, None]


def normalize_batch(batch: UnnormalizedBatch) -> Batch:
    """Converts a batch to its normalized form.

    Args:
        batch: a batch to normalize.

    Returns:
        a normalized batch.
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
