"""Logging a summary to stdout."""

from typing import Optional
from functools import reduce
from flax_extra import console
from flax_extra.checkpoint._summary import Summary, Metrics

# pylint: disable=too-few-public-methods
class SummaryLogger:
    """The writer prints summaries to stdout."""

    def __init__(self) -> None:
        self._formatting_info: Optional[tuple[int, int]] = None

    def __call__(self, summary: Summary) -> Summary:
        if self._formatting_info is None:
            self._formatting_info = _infer_summary_formatting_info(summary)

        step = summary.step

        console.log_step(
            step,
            f"Ran {summary.n_completed_steps} train steps in {summary.elapsed_time:.2f} seconds",
        )

        group_size, metric_size = self._formatting_info
        for group_label, group in summary.metrics.items():
            for metric_label, metric_value in group.items():
                console.log_step(
                    step,
                    f"{group_label.ljust(group_size)} {metric_label.rjust(metric_size)} "
                    f"| {metric_value:.8f}",
                )
        console.log("", stdout=True)

        return summary


def _infer_summary_formatting_info(summary: Summary) -> tuple[int, int]:
    def count(acc: tuple[int, int], group: tuple[str, Metrics]) -> tuple[int, int]:
        group_label, metrics = group
        max_group_size, max_metric_size = acc
        metric_size = max(map(len, metrics.keys()))
        return (
            max(max_group_size, len(group_label)),
            max(max_metric_size, metric_size),
        )

    return reduce(count, summary.metrics.items(), (0, 0))
