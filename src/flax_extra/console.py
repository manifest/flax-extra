"""Logging to cosole."""
import sys


def log(message: str, stdout: bool) -> None:
    """Writes a message to stdout.

    Args:
        message: an arbitrary text.
        stdout: whether to write or not to stdout.
    """
    if stdout:
        print(message)
        sys.stdout.flush()


def log_step(step: int, message: str, stdout: bool = True) -> None:
    """Writes a step-related message to stdout.

    Args:
        step: the step number.
        message: an arbitrary text.
        stdout: whether to write or not to stdout.
    """
    log(f"Step {step:6d}: {message}", stdout=stdout)
