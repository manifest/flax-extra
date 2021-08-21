"""RNN models."""

from flax import linen as nn
from flax.linen.recurrent import Array
from redex import combinator as cb
from flax_extra import layer as xn
from flax_extra import operator as xp


class RNNLM(nn.Module):
    """RNN language model."""

    vocab_size: int
    """size of the vocabulary."""

    d_model: int = 512
    """depth of the model."""

    n_layers: int = 2
    """a number of RNN layers."""

    dropout_rate: float = 0.1
    """dropout rate (how much to drop out)."""

    rnn_type: nn.Module = xn.LSTM
    """a type of the RNN."""

    deterministic: bool = True
    """the flag specifies whether the model must perform deterministically
        or not."""

    @nn.compact
    # pylint: disable=arguments-differ
    def __call__(self, inputs: Array) -> Array:
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
