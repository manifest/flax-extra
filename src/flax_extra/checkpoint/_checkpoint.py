"""A representation of the training state."""

from dataclasses import dataclass
from flax.core.frozen_dict import FrozenDict
import optax


# pylint: disable=too-many-instance-attributes
@dataclass
class Checkpoint:
    """A representation of the training state at particular recurrent step."""

    model_params: FrozenDict
    """parameters of the model at the checkpoint."""

    model_state: FrozenDict
    """a state of the model at the checkpoint."""

    optimizer_state: optax.OptState
    """a state of the optimizer at the checkpoint."""

    grads: FrozenDict
    """gradients at the checkpoint."""

    loss: float
    """a loss at the checkpoint."""

    n_completed_steps: int
    """a number of completed steps between previous checkpoint and this one."""

    elapsed_time: float
    """an elapsed time between previous checkpoint and this one."""

    step: int
    """a step number this checkpoint was occured."""
