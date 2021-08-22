"""Reading checkpoints from the local file system."""

from typing import Any, Callable, Mapping, Optional, Union
from dataclasses import dataclass
from flax.training import checkpoints
from flax_extra.checkpoint._checkpoint_file import CheckpointFile
from flax_extra.checkpoint._checkpoint_file_writer import regular_checkpoint_prefix


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

        Raises:
            TypeError: if the initializer is not of the :class:`Checkpoint` type.
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

        Raises:
            TypeError: if the initializer is not of the :class:`Checkpoint` type.
        """
        if initializer and not isinstance(initializer, CheckpointFile):
            raise TypeError(
                f"Cannot load a checkpoint from `{self.dir}`. "
                "Expecting initializer of `CheckpointFile` or `None` type, "
                f"but have got `{type(initializer)}`."
            )

        if prefix is None:
            prefix = regular_checkpoint_prefix()

        return checkpoints.restore_checkpoint(  # type:ignore
            self.dir,
            target=initializer,
            prefix=prefix,
        )
