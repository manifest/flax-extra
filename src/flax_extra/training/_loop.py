"""Training models."""

from typing import (
    Any,
    Callable,
    Final,
    Generator,
    List,
    Mapping,
    Iterable,
    Optional,
    Union,
)
from functools import partial
from dataclasses import dataclass
import math
import time
import jax
from jax import numpy as jnp
from jax.interpreters.xla import Device
import redex
from flax.core.scope import CollectionFilter, DenyList
from flax.core.frozen_dict import FrozenDict
import optax
from flax_extra import console, random
from flax_extra.batch import (
    normalize_batch,
    normalize_batch_per_device,
    Batch,
    DataStream,
    Inputs,
    UnnormalizedInputs,
)
from flax_extra.checkpoint import Checkpoint, CheckpointFile, CheckpointFileReader

Array = jnp.ndarray

InitializationFnResult = tuple[FrozenDict, FrozenDict]
InitializationFn = Callable[
    [Array, Inputs],
    InitializationFnResult,
]

ForwardPropagationFnResult = tuple[float, optax.OptState]
ForwardBackwardPropagationFnResult = tuple[ForwardPropagationFnResult, FrozenDict]
ForwardBackwardPropagationFn = Callable[
    [FrozenDict, FrozenDict, Batch, Array],
    ForwardBackwardPropagationFnResult,
]

UpdateFnResult = tuple[optax.OptState, FrozenDict, FrozenDict, FrozenDict, float]
UpdateFn = Callable[
    [optax.OptState, FrozenDict, FrozenDict, Batch, Array],
    UpdateFnResult,
]

BYTE_UNIT_SIZE: Final[int] = 1024
BYTE_UNITS: Final[List[str]] = ["B", "kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]


@dataclass
class TrainTask:
    """The training task describes how to train a model."""

    apply: Callable[..., Any]
    """an apply function of the model (a linen module)."""

    optimizer: optax.GradientTransformation
    """an optimizer."""

    loss: Callable[..., float]
    """a loss function.

    The function must accept as input arguments all
    model outputs and targets.
    """

    data: DataStream
    """a data stream of training examples.

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
class TrainTaskRunner:
    """The training task runner holds everything required for training
    the model on a single task (e.g. data generator, optimizer state, etc.)."""

    update: UpdateFn
    """a function performing update of the model parameters."""

    optimizer_state: optax.OptState
    """a state of the optimizer."""

    data_generator: DataStream
    """a data stream of training examples."""

    randnumkey_generator: random.Sequence
    """a random number key generator."""

    n_devices: int
    """a number of devices to parallelize training."""

    def run(
        self,
        model_params: FrozenDict,
        model_state: FrozenDict,
    ) -> tuple[FrozenDict, FrozenDict, FrozenDict, float]:
        """Runs a single training step and updates the optimizer state.

        Args:
            model_params: parameters of the model at the current step.
            model_state: a state of the model at the current step.

        Returns:
            updates for model parameters and state along with trainig stats.
        """
        batch = normalize_batch_per_device(
            batch=next(self.data_generator),
            n_devices=self.n_devices,
        )

        self.optimizer_state, *rest, average_grads, average_loss = self.update(
            self.optimizer_state,
            model_params,
            model_state,
            batch,
            random.split_per_device(
                key=next(self.randnumkey_generator), n_devices=self.n_devices
            ),
        )

        return (  # type: ignore
            *rest,
            average_grads,
            average_loss,
        )


class TrainLoop:
    """The training loop updates model parameters and yields checkpoints
    describing training state at specified steps."""

    # pylint: disable=too-many-arguments, too-many-locals
    def __init__(
        self,
        init: Union[Callable[..., Any], CheckpointFileReader],
        task: TrainTask,
        rnkey: Array,
        input_sample: Optional[UnnormalizedInputs] = None,
        n_steps_per_checkpoint: int = 1,
        n_steps: int = 0,
        collections: Optional[Mapping[str, List[str]]] = None,
        mutable_collections: CollectionFilter = DenyList("intermediates"),
        devices: Optional[List[Device]] = None,
        stdout: bool = True,
    ) -> None:
        """Initializes the training loop.

        Args:
            init: an init function of the model (linen module) or an
                instance of :class:`CheckpointFileReader` to initialize the
                training loop from a checkpoint stored on file system.
                If :class:`CheckpointFileReader` is specified, but checkpoint
                file isn't available, an initial checkpoint at step 0
                will be used.
            task: a discription of a training task.
            rnkey: a random number generator key.
            input_sample: a sigle batched training example. If `None`,
                the first training example yielded by data stream
                will be used.
            n_steps_per_checkpoint: a number of steps between checkpoints.
            n_steps: the total number of steps to run.
            collections: labels that will be used in creation of
                random number generator keys for variable collections
                of the linen module.
            mutable_collections: labels that specifies which collections
                should be treated as mutable.
            devices: a list of devices to parallelize the training,
                if `None` all available devices will be used.
            stdout: whether to print informational messages to stdout.
        """
        if devices is None:
            devices = jax.devices()

        if collections is None:
            collections = dict(init=["params"], apply=[])

        if input_sample is None:
            normalized_input_sample = normalize_batch(next(task.data))[0]
        else:
            normalized_input_sample = redex.util.expand_to_tuple(input_sample)

        n_devices = len(devices)
        rnkeyg = random.into_sequence(key=rnkey)
        self._n_steps_per_checkpoint = n_steps_per_checkpoint
        self._n_steps = n_steps
        self._stdout = stdout

        ## Initialization.
        initialization_start_time = time.time()
        use_checkpoint_file = isinstance(init, CheckpointFileReader)
        initialization = _setup_initialization(
            init.target if use_checkpoint_file else init,  # type: ignore
            collections["init"],
            mutable_collections,
        )
        model_params, model_state = initialization(
            next(rnkeyg),
            normalized_input_sample,
        )
        optimizer_state = task.optimizer.init(model_params)
        ## An initial checkpoint at step 0.
        initializer = CheckpointFile(
            model_params=model_params,
            model_state=model_state,
            optimizer_state=optimizer_state,
            step=0,
        )
        if use_checkpoint_file:
            checkpoint_loading_start_time = time.time()
            loaded_checkpoint_file = init(initializer=initializer)
            ## If the checkpoint file exists – use it as an initializer,
            ## otherwise fall back to an initial checkpoint at step 0.
            checkpoint_loading_elapsed_time = (
                time.time() - checkpoint_loading_start_time
            )
            if id(initializer) != id(loaded_checkpoint_file):
                initializer = loaded_checkpoint_file
                console.log(
                    f"A checkpoint was loaded in {checkpoint_loading_elapsed_time:.2f} seconds.",
                    stdout=stdout,
                )
        initialization_elapsed_time = time.time() - initialization_start_time
        console.log(
            f"Total model initialization time is {initialization_elapsed_time:.2f} seconds.",
            stdout=stdout,
        )

        self._step = initializer.step
        self._model_params = jax.device_put_replicated(
            initializer.model_params, devices
        )
        self._model_state = jax.device_put_replicated(initializer.model_state, devices)

        ## Setup the training task runner.
        forward_backward_propagation = _setup_forward_backward_propagation(
            task.apply,
            task.loss,
            collections["apply"],
            mutable_collections,
        )
        update = _setup_update(
            task.optimizer,
            forward_backward_propagation,
        )
        self._train_task = TrainTaskRunner(
            update=update,
            optimizer_state=jax.device_put_replicated(
                initializer.optimizer_state, devices
            ),
            data_generator=task.data,
            randnumkey_generator=rnkeyg,
            n_devices=n_devices,
        )

    @property
    def step(self) -> int:
        """the current step."""
        return self._step

    @property
    def n_steps_per_checkpoint(self) -> int:
        """a number of steps between checkpoints."""
        return self._n_steps_per_checkpoint

    @n_steps_per_checkpoint.setter
    def n_steps_per_checkpoint(self, value: int) -> None:
        """updates the number of steps between checkpoints."""
        self._n_steps_per_checkpoint = value

    @property
    def n_steps(self) -> int:
        """the total number of steps in the loop."""
        return self._n_steps

    @n_steps.setter
    def n_steps(self, value: int) -> None:
        """updates the total number of steps in the loop."""
        self._n_steps = value

    def __iter__(self) -> Iterable[Checkpoint]:
        return self._run(n_steps=self._n_steps)

    def __next__(self) -> Checkpoint:
        return next(self._run(n_steps=self._n_steps))

    def next_checkpoint(self) -> Checkpoint:
        """Runs a number of steps remaining for the next checkpoint.

        Returns:
            a checkpoint.
        """
        n_steps = _next_checkpoint_step(self._step, self.n_steps_per_checkpoint)
        return next(self._run(n_steps=n_steps))

    def next_step(self) -> Checkpoint:
        """Runs a single step.

        Returns:
            a checkpoint.
        """
        return next(self._run(n_steps=self._step + 1, n_steps_per_checkpoint=1))

    def run(self, n_steps: int) -> Generator[Checkpoint, None, None]:
        """Runs an arbitrary number of steps yelding checkpoints.

        Args:
            n_steps: a number of steps to ran.

        Yields:
            a checkpoint.
        """
        return self._run(n_steps=n_steps)

    def _run(
        self,
        n_steps: Optional[int] = None,
        n_steps_per_checkpoint: Optional[int] = None,
    ) -> Generator[Checkpoint, None, None]:
        """Runs an arbitrary number of steps yelding checkpoints.

        Args:
            n_steps: a number of steps to ran. The `n_steps` property
                of the loop will be updated. If set to `None`,
                `n_steps` property of the loop is used.
            n_steps_per_checkpoint: determines how often checkpoints
                will be yeilded. The `n_steps_per_checkpoint` property
                of the loop won't be updated. If set to `None`,
                `n_steps_per_checkpoint` property of the loop is used.

        Yields:
            a checkpoint.
        """
        if n_steps is not None:
            self._n_steps = n_steps

        n_remaining_steps = self._n_steps - self._step
        if n_remaining_steps <= 0:
            console.log(
                f"Stop training, already reached the total training steps {self._n_steps}.",
                stdout=self._stdout,
            )
            return None

        if n_steps_per_checkpoint is None:
            n_steps_per_checkpoint = self._n_steps_per_checkpoint

        console.log(
            f"Total number of trainable weights: {_params_total_size(self._model_params)} "
            f"{_format_total_bytes(_params_total_bytes(self._model_params))}."
            "\n",
            stdout=self._stdout,
        )

        n_steps_between_checkpoints = 0
        start_time = time.time()
        for self._step in range(self._step + 1, self._n_steps + 1):
            n_steps_between_checkpoints += 1

            ## Train.
            self._model_params, self._model_state, grads, loss = self._train_task.run(
                self._model_params,
                self._model_state,
            )

            ## Produce a checkpoint each `n_steps_per_checkpoint` step.
            if self._is_checkpoint_step(n_steps_per_checkpoint):
                elapsed_time = time.time() - start_time
                yield Checkpoint(
                    self._model_params,
                    self._model_state,
                    self._train_task.optimizer_state,
                    grads,
                    loss,
                    n_steps_between_checkpoints,
                    elapsed_time,
                    self._step,
                )

                start_time = time.time()
                n_steps_between_checkpoints = 0

        ## Produce a checkpoint for the latest step.
        if not self._is_checkpoint_step(n_steps_per_checkpoint):
            elapsed_time = time.time() - start_time
            yield Checkpoint(
                self._model_params,
                self._model_state,
                self._train_task.optimizer_state,
                grads,
                loss,
                n_steps_between_checkpoints,
                elapsed_time,
                self._step,
            )

        return None

    def _is_checkpoint_step(self, n_steps_per_checkpoint: int) -> bool:
        """Determines whether the current step is a checkpoint step.

        Args:
            n_steps_per_checkpoint: a number of steps per checkpoint.

        Returns:
            `True` if the current step is a checkpoint step, otherwise `False`.
        """
        return (self._step - 1) % n_steps_per_checkpoint == 0


def _next_checkpoint_step(current_step: int, n_steps_per_checkpoint: int) -> int:
    """Computes a step number the next checkpoint will occur.

    Args:
        current_step: the current step.
        n_steps_per_checkpoint: a number of steps per checkpoint.

    Returns:
        a step number.
    """
    return (
        1
        + n_steps_per_checkpoint
        + n_steps_per_checkpoint * ((current_step - 1) // n_steps_per_checkpoint)
    )


def _params_total_size(params: FrozenDict) -> int:
    """Computes the total number of model parameters.

    Args:
        params: model parameters

    Returns:
        a number of parameters.
    """

    def count(acc: int, leaf: Array) -> int:
        return acc + leaf.size

    return jax.tree_util.tree_reduce(count, params, 0)


def _params_total_bytes(params: FrozenDict) -> int:
    """Conputes the total byte size of model parameters.

    Args:
        params: model parameters.

    Return:
        a byte size.
    """

    def count(acc: int, leaf: Array) -> int:
        return acc + leaf.nbytes  # type: ignore

    return jax.tree_util.tree_reduce(count, params, 0)


def _format_total_bytes(size: int) -> str:
    """Formats byte size to human readable text format.

    Args:
        size: byte size.

    Returns:
         a formatted text.
    """
    if size < BYTE_UNIT_SIZE:
        return f"= {size} B"

    ratio = math.floor(math.log2(size) / 10)
    scaled_size = size / (BYTE_UNIT_SIZE ** ratio)
    scaled_size = math.floor(scaled_size * 10) / 10.0
    return f"~ {scaled_size} {BYTE_UNITS[ratio]}"


def _setup_initialization(
    init: Callable[..., Any],
    collections: List[str],
    mutable_collections: CollectionFilter,
) -> InitializationFn:
    # We don't need to change `mutable` during training,
    # so we set its value and JIT-compile the function.
    init = partial(init, mutable=mutable_collections)

    def initialization(
        rngkey: Array,
        inputs: Inputs,
    ) -> InitializationFnResult:
        rngkeys = random.into_collection(key=rngkey, labels=collections)
        state, params = init(rngkeys, *inputs).pop("params")
        return params, state

    return jax.jit(initialization)


def _setup_forward_backward_propagation(
    apply: Callable[..., Any],
    loss: Callable[..., float],
    collections: List[str],
    mutable_collections: CollectionFilter,
) -> ForwardBackwardPropagationFn:
    # We don't need to change `mutable` during training,
    # so we set its value and JIT-compile the function.
    apply = partial(apply, mutable=mutable_collections)

    def forward_propagation(
        params: FrozenDict,
        state: FrozenDict,
        batch: Batch,
        rngkey: Array,
    ) -> ForwardPropagationFnResult:
        rngkeys = random.into_collection(key=rngkey, labels=collections)
        variables = {"params": params, **state}
        del params
        del state
        inputs, targets = batch
        outputs, variables = apply(variables, *inputs, rngs=rngkeys)
        outputs = redex.util.expand_to_tuple(outputs)
        state_diff, params = variables.pop("params")
        del variables
        del params
        return loss(*outputs, *targets), state_diff

    ## We will JIT-compile `update` function that wraps this one.
    return jax.value_and_grad(forward_propagation, has_aux=True)


def _setup_update(
    optimizer: optax.GradientTransformation,
    forward_backward_propagation: Callable[..., Any],
) -> UpdateFn:
    def update(
        optimizer_state: optax.OptState,
        model_params: FrozenDict,
        model_state: FrozenDict,
        batch: Batch,
        rngkey: Array,
    ) -> UpdateFnResult:
        (loss, state_diff), grads = forward_backward_propagation(
            model_params,
            model_state,
            batch,
            rngkey,
        )

        averaged_loss = jax.lax.pmean(loss, axis_name="replica")  # type: ignore
        averaged_grads = jax.lax.pmean(grads, axis_name="replica")  # type: ignore
        ## TODO: It's not clear what may be in the model state,
        ## but we need to aggregate it in some way.
        averaged_state_diff = jax.lax.pmean(state_diff, axis_name="replica")  # type: ignore
        param_updates, updated_optimizer_state = optimizer.update(
            averaged_grads,
            optimizer_state,
        )
        updated_model_params = optax.apply_updates(model_params, param_updates)
        updated_model_state = {**model_state, **averaged_state_diff}
        return (
            updated_optimizer_state,
            updated_model_params,
            updated_model_state,
            averaged_grads,
            averaged_loss,
        )

    return jax.pmap(update, axis_name="replica", donate_argnums=(3, 4))
