"""Evaluating models."""

from typing import Any, Callable, List, Mapping, MutableMapping, Optional
from functools import partial
from dataclasses import dataclass
import jax
from jax import numpy as jnp
import redex
from flax.core.frozen_dict import FrozenDict
from flax_extra import random, util
from flax_extra.batch import normalize_batch_per_device, Batch, DataStream
from flax_extra.checkpoint import Checkpoint, Metrics, Summary

Array = jnp.ndarray
ArrayTree = Any
MetricSpecs = Mapping[str, Callable[..., float]]

EvaluationFn = Callable[[FrozenDict, FrozenDict, Batch, Array], Metrics]


@dataclass
class EvalTask:
    """The evaluation task describes how to evaluate a model."""

    apply: Callable[..., Any]
    """an apply function of the model (a linen module)."""

    metrics: MetricSpecs
    """evaluation metrics with corresponding labels."""

    data: DataStream
    """a data stream of evaluation examples.

    An item, yielded by the data stream, must consist of inputs and targets.
    Each of them may be represented as a single array or multiple arrays.
    Targets may be represented as an empty tuple.

    Following forms are acceptable:

    - `(x, y)`
    - `((x,...), (y,...))`
    - `((x,...), ())`
    - etc.

    Inputs get passed to model's `apply(x,...)` function as arguments.

    Targets along with model outputs, `(o,...)`, get passed to a
    `loss(o,...,y,...)` function as arguments.
    """


@dataclass
class EvalTaskRunner:
    """The evaluation task runner holds everything required to evaluate
    the model (e.g. data generator, etc.)."""

    evaluation: EvaluationFn
    """a function performing evaluation of the model."""

    data_generator: DataStream
    """a data stream of evaluation examples."""

    randnumkey_generator: random.Sequence
    """a random number key generator."""

    n_devices: int
    """a number of devices to parallelize evaluation."""

    def run(
        self,
        model_params: FrozenDict,
        model_state: FrozenDict,
    ) -> Metrics:
        """Runs a single evaluation step.

        Args:
            model_params: parameters of the model.
            model_state: a state of the model.

        Returns:
            evaluation metrics.
        """
        batch = normalize_batch_per_device(
            batch=next(self.data_generator),
            n_devices=self.n_devices,
        )

        return util.originate(  # type: ignore
            self.evaluation(
                model_params,
                model_state,
                batch,
                random.split_per_device(
                    next(self.randnumkey_generator),
                    self.n_devices,
                ),
            )
        )


class EvalLoop:
    """The evaluation loop performs a few steps evaluating a model
    then returns averaged metrics."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        task: EvalTask,
        rnkey: Array,
        n_steps: int = 1,
        collections: Optional[Mapping[str, List[str]]] = None,
        n_devices: Optional[int] = None,
    ):
        """Initializes the evaluation loop.

        Args:
            task: a discription of an evaluation task.
            rnkey: a random number generator key.
            n_steps: a number of steps to run.
            collections: labels that will be used in creation of
                random number generator keys for variable collections
                of the linen module.
            n_devices: a number of devices to parallelize evaluation.
                if `None` all available devices will be used.
                Must match the value specified for the training loop.
        """
        if n_devices is None:
            n_devices = jax.local_device_count()

        if collections is None:
            collections = dict(apply=[])

        rnkeyg = random.into_sequence(key=rnkey)

        ## Setup the eval task runner.
        evaluation = _setup_evaluation(
            task.apply,
            task.metrics,
            collections["apply"],
        )
        self._eval_task = EvalTaskRunner(
            evaluation=evaluation,
            data_generator=task.data,
            randnumkey_generator=rnkeyg,
            n_devices=n_devices,
        )

        self._n_steps = n_steps

    @property
    def n_steps(self) -> int:
        """a number of steps in the loop."""
        return self._n_steps

    def __call__(self, checkpoint: Checkpoint) -> Summary:
        ## Average evaluation metrics over `self._n_steps`.
        eval_metrics: MutableMapping[str, float] = self._eval_task.run(  # type: ignore
            checkpoint.model_params,
            checkpoint.model_state,
        )
        for step in range(1, self._n_steps):
            del step
            update = self._eval_task.run(
                checkpoint.model_params,
                checkpoint.model_state,
            )
            for k in update.keys():
                eval_metrics[k] += update[k]
        for k in eval_metrics.keys():
            eval_metrics[k] /= self._n_steps

        ## Produce summary.
        seconds_per_step = checkpoint.elapsed_time / checkpoint.n_completed_steps
        train_stats = dict(
            seconds_per_step=seconds_per_step,
            gradients_l2norm=_tree_l2norm(checkpoint.grads),
            weights_l2norm=_tree_l2norm(checkpoint.model_params),
            loss=_tree_l2norm(checkpoint.loss),
        )

        return Summary(
            metrics=dict(
                train=train_stats,
                eval=eval_metrics,
            ),
            **vars(checkpoint),
        )


def _tree_l2norm(tree: ArrayTree) -> float:
    """Computes L2-norm for each leaf of the pytree object, then average the result.

    Args:
        tree: a pytree.

    Returns:
        a L2-norm averaged over all pytree leaves.
    """

    def add_norm(acc: tuple[int, int], leaf: Array) -> tuple[int, int]:
        n_total, average_total = acc
        return (
            n_total + 1,
            average_total + jnp.linalg.norm(leaf),
        )

    n_total, average_total = jax.tree_util.tree_reduce(add_norm, tree, (0, 0))
    return average_total / n_total


def _evaluate(
    outputs: tuple[Array, ...],
    targets: tuple[Array, ...],
    metric_specs: MetricSpecs,
) -> Metrics:
    def evaluate(spec: tuple[str, Callable[..., float]]) -> tuple[str, float]:
        metric_label, metric_func = spec
        metric = metric_func(*outputs, *targets)
        return metric_label, metric

    return dict(map(evaluate, metric_specs.items()))


def _setup_evaluation(
    apply: Callable[..., Any],
    metric_specs: MetricSpecs,
    collections: List[str],
) -> EvaluationFn:
    # Disable mutable during evaluation because we are going to perform just
    # a single pass through the model and use only model outputs in metric
    # calculations.
    apply = partial(apply, mutable=False)

    def evaluation(
        params: FrozenDict,
        state: FrozenDict,
        batch: Batch,
        rngkey: Array,
    ) -> Metrics:
        rngkeys = random.into_collection(key=rngkey, labels=collections)
        variables = {"params": params, **state}
        del params
        del state
        inputs, targets = batch
        outputs = apply(variables, *inputs, rngs=rngkeys)
        outputs = redex.util.expand_to_tuple(outputs)
        metrics = _evaluate(outputs, targets, metric_specs)
        averaged_metrics = jax.lax.pmean(metrics, axis_name="replica")  # type: ignore
        return averaged_metrics  # type: ignore

    return jax.pmap(evaluation, axis_name="replica", donate_argnums=(2, 3))
