r"""LSTM layer."""

from flax import linen as nn
from flax.linen.recurrent import Array
from flax_extra import combinator as cb

LSTMState = tuple[Array, Array]

# TODO: add type annotation to `nn.LSTMCell`.
class LSTMCell(nn.LSTMCell):
    r"""LSTM cell."""

    def __call__(  # type: ignore[override]
        self,
        carry: LSTMState,
        inputs: Array,
    ) -> tuple[LSTMState, Array]:
        # pylint: disable=useless-super-delegation
        return super().__call__(carry, inputs)  # type:ignore


class LSTM(nn.Module):
    r"""LSTM running on time axis.

    The layer scans over each time step of the input and returns its hidden
    output state for the last time step (hidden cell state is dropped).
    """

    d_hidden: int
    r"""depth of a hidden state. LSTM has (output state and cell state)."""

    @nn.compact
    def __call__(self, inputs: Array) -> Array:  # type: ignore[override] # pylint: disable=arguments-differ
        return cb.serial(
            cb.branch(self.initial_state, cb.identity(n_in=1)),
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
        r"""Creates an LSTM state."""
        batch_size = inputs.shape[0]
        return nn.LSTMCell.initialize_carry(  # type:ignore
            self.make_rng("carry"),
            (batch_size,),
            self.d_hidden,
        )
