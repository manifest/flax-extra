r"""Metrics associated with checkpoints."""

from typing import Mapping
from dataclasses import dataclass
from flax_extra.checkpoint import Checkpoint

Metrics = Mapping[str, float]
r"""Metrics grouped by categories such as training statistics,
evaluation metrics, etc."""


@dataclass
class Summary(Checkpoint):
    r"""A checkpoint with related metrics."""

    metrics: Mapping[str, Metrics]
