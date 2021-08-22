"""Reading the best metric checkpoints from the local file system."""

from dataclasses import dataclass
from flax_extra.checkpoint._checkpoint_file import CheckpointFile
from flax_extra.checkpoint._checkpoint_file_reader import CheckpointFileReader
from flax_extra.checkpoint._best_checkpoint_file_writer import (
    lowest_checkpoint_prefix,
    highest_checkpoint_prefix,
)


@dataclass
class LowestCheckpointFileReader(CheckpointFileReader):
    """A reader of a lowest metric checkpoint file."""

    metric: str
    """a metric label."""

    group: str = "eval"
    """a group label."""

    def __call__(self, initializer: CheckpointFile) -> CheckpointFile:
        """Reads the latest checkpoint related to the lowest metric value
        from the file system.

        Args:
            initializer: an initial checkpoint at step 0.
                It is required to restore type information.

        Returns:
            either the loaded checkpoint file or initial checkpoint.

        Raises:
            TypeError: if the initializer is not of the :class:`Checkpoint` type.
        """
        prefix = lowest_checkpoint_prefix(
            metric_label=self.metric,
            group_label=self.group,
        )
        return super().read(initializer, prefix=prefix)  # type: ignore


@dataclass
class HighestCheckpointFileReader(CheckpointFileReader):
    """A reader of a highest metric checkpoint file."""

    metric: str
    """a metric label."""

    group: str = "eval"
    """a group label."""

    def __call__(self, initializer: CheckpointFile) -> CheckpointFile:
        """Reads the latest checkpoint related to the highest metric value
        from the file system.

        Args:
            initializer: an initial checkpoint at step 0.
                It is required to restore type information.

        Returns:
            either the loaded checkpoint file or initial checkpoint.

        Raises:
            TypeError: if the initializer is not of the :class:`Checkpoint` type.
        """
        prefix = highest_checkpoint_prefix(
            metric_label=self.metric,
            group_label=self.group,
        )
        return super().read(initializer, prefix=prefix)  # type: ignore
