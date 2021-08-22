"""Writing checkpoints to the local file system."""

from typing import Optional, Union
from dataclasses import dataclass
from flax.training import checkpoints
from flax_extra import util
from flax_extra.checkpoint._checkpoint import Checkpoint
from flax_extra.checkpoint._summary import Summary
from flax_extra.checkpoint._checkpoint_file import CheckpointFile


@dataclass
class CheckpointFileWriter:
    """A writer for the checkpoint file format."""

    output_dir: str
    """a directory path for checkpoint files."""

    keep: int = 1
    """a number of checkpoints to keep."""

    overwrite: bool = True
    """whether to override previous checkpoints."""

    def __call__(
        self,
        checkpoint: Union[Checkpoint, Summary],
    ) -> Union[Checkpoint, Summary]:
        """Writes a checkpoint to the file system.

        Args:
            checkpoint: a checkpoint to write.

        Returns:
            an original checkpoint.

        Raises:
            TypeError: if the checkpoint is neither of :class:`Checkpoint`
            nor :class:`Summary` type.
        """
        return self.write(checkpoint)

    def write(
        self,
        checkpoint: Union[Checkpoint, Summary],
        prefix: Optional[str] = None,
    ) -> Union[Checkpoint, Summary]:
        """Writes a checkpoint to the file system.

        Args:
            checkpoint: a checkpoint to write.
            prefix: a string that will be added to the output file name.

        Returns:
            an original checkpoint.

        Raises:
            TypeError: if the checkpoint is neither of :class:`Checkpoint`
            nor :class:`Summary` type.
        """
        if not isinstance(checkpoint, (Checkpoint, Summary)):
            raise TypeError(
                f"Cannot write a checkpoint to `{self.output_dir}`. "
                "Expecting checkpoint of `Checkpoint` or `Summary` type, "
                f"but have got `{type(checkpoint)}`."
            )

        if prefix is None:
            prefix = regular_checkpoint_prefix()

        checkpoint_file = CheckpointFile(
            model_params=util.originate(checkpoint.model_params),
            model_state=util.originate(checkpoint.model_state),
            optimizer_state=util.originate(checkpoint.optimizer_state),
            step=checkpoint.step,
            metrics=checkpoint.metrics if isinstance(checkpoint, Summary) else {},
        )

        checkpoints.save_checkpoint(
            self.output_dir,
            target=checkpoint_file,
            step=checkpoint.step,
            prefix=prefix,
            keep=self.keep,
            overwrite=self.overwrite,
        )

        return checkpoint


def regular_checkpoint_prefix() -> str:
    """Returns a prefix for a regular checkpoint."""
    return "regular_"
