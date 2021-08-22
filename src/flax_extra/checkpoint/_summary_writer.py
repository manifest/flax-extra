"""Writing summaries to the Tensorboard."""

from dataclasses import dataclass
import contextlib
from flax.metrics import tensorboard
from flax_extra.checkpoint._summary import Summary


@dataclass
class SummaryWriter:
    """The writer persists summaries in Tensorboard file format
    to the specified directory on the local file system."""

    output_dir: str

    def __call__(self, summary: Summary) -> Summary:
        with contextlib.closing(tensorboard.SummaryWriter(self.output_dir)) as writer:
            step = summary.step
            for group_label, group in summary.metrics.items():
                for metric_label, metric_value in group.items():
                    writer.scalar(
                        f"{group_label}/{metric_label}",
                        metric_value,
                        step,
                    )

        return summary
