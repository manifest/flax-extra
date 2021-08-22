"""The checkpoint file format."""

import typing
from typing import Any
from dataclasses import field
from flax.core.frozen_dict import FrozenDict
from flax.struct import PyTreeNode, TNode
import optax
from flax_extra.checkpoint._summary import Metrics

# pylint: disable=too-few-public-methods
class CheckpointFile(PyTreeNode):
    """The file format to store checkpoint on the local file system."""

    model_params: FrozenDict
    """parameters of the model at the checkpoint."""

    model_state: FrozenDict
    """a state of the model at the checkpoint."""

    optimizer_state: optax.OptState
    """a state of the optimizer at the checkpoint."""

    step: int
    """a step number this checkpoint was occured."""

    metrics: Metrics = field(default_factory=dict)
    """metrics at the current checkpoint."""

    version: int = 0
    """the file format version."""

    if typing.TYPE_CHECKING:

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            # This stub informs a type checker that this method is overridden.
            super().__init__()

        def replace(self: TNode, **overrides: Any) -> TNode:
            # This stub informs a type checker that this method is overridden.
            pass
