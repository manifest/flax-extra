"""File format and operations for checkpoints."""

import typing
from typing import Any, Callable, Mapping, Optional, Union
from dataclasses import dataclass
from flax.core.frozen_dict import FrozenDict
from flax.struct import PyTreeNode, TNode
from flax.training import checkpoints
import optax

# pylint: disable=too-few-public-methods
class CheckpointFile(PyTreeNode):
    """A file format for the checkpoint."""

    model_params: FrozenDict
    """parameters of the model at the checkpoint."""

    model_state: FrozenDict
    """a state of the model at the checkpoint."""

    optimizer_state: optax.OptState
    """a state of the optimizer at the checkpoint."""

    step: int
    """a step number this checkpoint was occured."""

    version: int = 0
    """the file format version."""

    if typing.TYPE_CHECKING:

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            # This stub informs a type checker that this method is overridden.
            super().__init__()

        def replace(self: TNode, **overrides: Any) -> TNode:
            # This stub informs a type checker that this method is overridden.
            pass


@dataclass
class CheckpointFileReader:
    """A reader for the checkpoint file format."""

    dir: str
    """a directory path for checkpoint files."""

    ## The `target` is required to restore type information.
    ## It is used to initialize `CheckpointFile` at step 0.
    target: Callable[..., Any]
    """init function of the model related to the checkpoint.
    It is used to restore type information."""

    def __call__(self, initializer: CheckpointFile) -> CheckpointFile:
        """Reads the latest checkpoint from the file system.

        If the checkpoint file doesn't exist, given initial checkpoint
        will be returned.

        Args:
            initializer: an initial checkpoint at step 0.
                It is required to restore type information.

        Returns:
            either the loaded checkpoint file or initial checkpoint.
        """
        return self.read(initializer)  # type: ignore

    def read(
        self,
        initializer: Optional[CheckpointFile],
        prefix: Optional[str] = None,
    ) -> Union[CheckpointFile, Mapping[str, Any]]:
        """Reads the latest checkpoint from the file system.

        If the checkpoint file doesn't exist, given initial checkpoint
        will be returned.

        If an initial checkpoint isn't given, the loaded checkpoint
        will of the `Mapping[str, Any]` type.

        Args:
            initializer: an initial checkpoint at step 0.
                It is required to restore type information.

        Returns:
            either the loaded checkpoint file or initial checkpoint.
        """
        if initializer and not isinstance(initializer, CheckpointFile):
            raise TypeError(
                f"Cannot load a checkpoint from `{self.dir}`. "
                "Expecting initializer of `CheckpointFile` or `None` type, "
                f"but have got `{type(initializer)}`."
            )

        if prefix is None:
            prefix = _regular_checkpoint_prefix()

        return checkpoints.restore_checkpoint(  # type:ignore
            self.dir,
            target=initializer,
            prefix=prefix,
        )


def _regular_checkpoint_prefix() -> str:
    return "regular_"
