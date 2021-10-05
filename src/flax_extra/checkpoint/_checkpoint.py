r"""A representation of the training state."""

from typing import Any
from dataclasses import dataclass
from flax.core.frozen_dict import FrozenDict
import optax


FrozenVars = FrozenDict[Any, Any]


# pylint: disable=too-many-instance-attributes
@dataclass
class Checkpoint:
    r"""A representation of the training state at particular recurrent step."""

    model_params: FrozenVars
    r"""parameters of the model at the checkpoint."""

    model_state: FrozenVars
    r"""a state of the model at the checkpoint."""

    optimizer_state: optax.OptState
    r"""a state of the optimizer at the checkpoint."""

    grads: FrozenVars
    r"""gradients at the checkpoint."""

    loss: float
    r"""a loss at the checkpoint."""

    n_completed_steps: int
    r"""a number of completed steps between previous checkpoint and this one."""

    elapsed_time: float
    r"""an elapsed time between previous checkpoint and this one."""

    step: int
    r"""a step number this checkpoint was occured."""
