"""Writing the best metric checkpoints to the local file system."""

import abc
from typing import Any, Optional
from flax.training import checkpoints
from flax_extra import console
from flax_extra.checkpoint._summary import Summary
from flax_extra.checkpoint._checkpoint_file_writer import CheckpointFileWriter


class BestCheckpointFileWriter(CheckpointFileWriter, metaclass=abc.ABCMeta):
    """A base class for the best metric checkpoint writers."""

    @property
    def best_metric(self) -> float:
        """the current value for the best metric."""
        return self._best_metric

    @abc.abstractmethod
    def default_metric_value(self) -> float:
        """an initial value for the best metric."""

    @abc.abstractmethod
    def is_best(self, metric: float) -> bool:
        """Determines whether the metric value is the best.

        Args:
            metric: a new value to compare the current best value with.

        Returns:
            `True` if a new value is the best value, `False` otherwise.
        """

    def __init__(
        self,
        metric: str,
        group: str,
        prefix: str,
        **kwds: Any,
    ) -> None:
        """Initializes the writer.

        Args:
            metric: a metric label.
            group: a group label (such as "train" or "eval").
            prefix: a string that will be added to the output file name.
        """
        super().__init__(**kwds)

        self._group_label = group
        self._metric_label = metric
        self._prefix = prefix
        self._best_metric = (
            _load_best_metric(
                self._group_label,
                self._metric_label,
                self.output_dir,
                self._prefix,
            )
            or self.default_metric_value()
        )

    def __call__(self, summary: Summary) -> Summary:  # type: ignore
        """Keeps track of the best metric value and writes a checkpoint
        to the local file system when the value get updated.

        Args:
            summary: a checkpoint that also incudes metrics.

        Returns:
            an original summary.

        Raises:
            TypeError: if the checkpoint is not of the :class:`Summary` type.
        """
        if not isinstance(summary, Summary):
            raise TypeError(
                f"Cannot write a checkpoint to `{self.output_dir}`. "
                "Expecting checkpoint of type `Summary`, "
                f"but have got `{type(summary)}`."
            )

        metric = _read_best_metric(summary, self._group_label, self._metric_label)
        if self.is_best(metric):
            self._best_metric = metric
            _ = super().write(summary, self._prefix)
            return summary

        return summary


class LowestCheckpointFileWriter(BestCheckpointFileWriter):
    """A checkpoint writer that writes a checkpoint each time the lowest metric
    value is observed."""

    def __init__(
        self,
        metric: str,
        group: str = "eval",
        stdout: bool = True,
        **kwds: Any,
    ) -> None:
        """Initializes the lowest metric checkpoint writer.

        Args:
            metric: a metric label.
            group: a group label.
            stdout: whether to write to stdout.
        """
        prefix = lowest_checkpoint_prefix(metric_label=metric, group_label=group)
        super().__init__(metric, group, prefix=prefix, **kwds)

        console.log(
            f"The lowest value for the metric {self._group_label}/{self._metric_label} "
            f"is set to {self._best_metric:.8f}.",
            stdout=stdout,
        )

    def default_metric_value(self) -> float:
        return float("+inf")

    def is_best(self, metric: float) -> bool:
        return min(self._best_metric, metric) == metric


class HighestCheckpointFileWriter(BestCheckpointFileWriter):
    """A checkpoint writer that writes a checkpoint each time the highest metric
    value is observed."""

    def __init__(
        self,
        metric: str,
        group: str = "eval",
        stdout: bool = True,
        **kwds: Any,
    ):
        """Initializes the highest metric checkpoint writer.

        Args:
            metric: a metric label.
            group: a group label.
            stdout: whether to write to stdout.
        """
        prefix = lowest_checkpoint_prefix(metric_label=metric, group_label=group)
        super().__init__(metric, group, prefix=prefix, **kwds)

        console.log(
            f"The highest value for the metric {self._group_label}/{self._metric_label} "
            f"is set to {self._best_metric:.8f}.",
            stdout=stdout,
        )

    def default_metric_value(self) -> float:
        return float("-inf")

    def is_best(self, metric: float) -> bool:
        return max(self._best_metric, metric) == metric


def lowest_checkpoint_prefix(group_label: str, metric_label: str) -> str:
    """Returns a prefix for a lowest metric checkpoint."""
    return f"lowest_{group_label}_{metric_label}_"


def highest_checkpoint_prefix(group_label: str, metric_label: str) -> str:
    """Returns a prefix for a highest metric checkpoint."""
    return f"highest_{group_label}_{metric_label}_"


def _load_best_metric(
    group_label: str,
    metric_label: str,
    checkpoint_dir: str,
    prefix: str,
) -> Optional[float]:
    metric = None

    checkpoint_dict = checkpoints.restore_checkpoint(
        checkpoint_dir,
        target=None,
        prefix=prefix,
    )

    if checkpoint_dict:
        group = checkpoint_dict.get("metrics").get(group_label)
        if group is None:
            raise ValueError(
                f"Cannot load the metric {group_label}/{metric_label} "
                f"from `{checkpoint_dir}`, because restored checkpoint "
                "does not contain the metric's group."
            )

        metric = group.get(metric_label)
        if metric is None:
            raise ValueError(
                f"Cannot load the metric {group_label}/{metric_label} "
                f"from `{checkpoint_dir}`, because restored checkpoint "
                "does not contain the metric."
            )

    return metric


def _read_best_metric(
    summary: Summary,
    group_label: str,
    metric_label: str,
) -> float:
    group = summary.metrics.get(group_label)
    if group is None:
        raise ValueError(
            f"Cannot read the metric {group_label}/{metric_label} "
            "from a summary, because it does not contain metric's group."
        )

    metric = group.get(metric_label)
    if metric is None:
        raise ValueError(
            f"Cannot read the metric {group_label}/{metric_label} "
            "from a summary, because it does not contain the metric."
        )

    return metric
