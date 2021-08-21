"""LSTM layer."""

from redex import operator as op
from redex import combinator as cb
from flax import linen as nn
from flax.linen.recurrent import Array

LSTMState = tuple[Array, Array]

# TODO: add type annotation to `nn.LSTMCell`.
class LSTMCell(nn.LSTMCell):
    """LSTM cell."""

    def __call__(self, carry: LSTMState, inputs: Array) -> tuple[LSTMState, Array]:
        # pylint: disable=useless-super-delegation
        return super().__call__(carry, inputs)  # type:ignore


class LSTM(nn.Module):
    """LSTM running on axis 1.

    The layer scans over each time step of the input and returns its hidden
    output state for the last time step (hidden cell state is dropped).
    """

    d_hidden: int
    """depth of a hidden state. LSTM has (output state and cell state)."""

    @nn.compact
    # pylint: disable=arguments-differ
    def __call__(self, inputs: Array) -> Array:
        return cb.serial(
            cb.branch(self.initial_state, op.identity),
            nn.scan(
                LSTMCell,
                variable_broadcast="params",
                split_rngs={"params": False},
                in_axes=1,
                out_axes=1,
            )(),
            cb.drop(n_in=2),
        )(inputs)

    def initial_state(self, inputs: Array) -> LSTMState:
        """Creates an LSTM state."""
        batch_size = inputs.shape[0]
        return nn.LSTMCell.initialize_carry(  # type:ignore
            self.make_rng("carry"),
            (batch_size,),
            self.d_hidden,
        )
