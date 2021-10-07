r"""RNN language model."""

from typing import Callable
from flax import linen as nn
from flax.linen.recurrent import Array
from flax_extra import combinator as cb
from flax_extra import layer as xn
from flax_extra import operator as xp


RNNT = Callable[..., nn.Module]


class RNNLM(nn.Module):
    r"""RNN language model."""

    vocab_size: int
    r"""size of the vocabulary."""

    d_model: int = 512
    r"""depth of the model."""

    n_layers: int = 2
    r"""a number of RNN layers."""

    dropout_rate: float = 0.1
    r"""probababilistic rate for dropout."""

    rnn_type: RNNT = xn.LSTM
    r"""a type of the RNN."""

    deterministic: bool = True
    r"""whether to perform deterministically or not."""

    @nn.compact
    def __call__(self, inputs: Array) -> Array:  # type: ignore[override] # pylint: disable=arguments-differ
        return cb.serial(
            xp.ShiftRight(axis=1),
            nn.Embed(
                num_embeddings=self.vocab_size,
                features=self.d_model,
                embedding_init=nn.initializers.normal(stddev=1.0),
            ),
            [self.rnn_type(d_hidden=self.d_model) for _ in range(self.n_layers)],
            nn.Dropout(
                rate=self.dropout_rate,
                deterministic=self.deterministic,
            ),
            nn.Dense(features=self.vocab_size),
        )(inputs)
