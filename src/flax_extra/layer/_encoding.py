r"""Encoding."""

from typing import cast, Callable, List, Optional
from jax import numpy as jnp
from flax import linen as nn

from flax_extra import combinator as cb

Array = jnp.ndarray
Positions = List[int]

PreprocessingFn = Callable[[Array], Array]
PreprocessingCt = Callable[..., PreprocessingFn]
PositionalEncodingFn = Callable[[int, Optional[Positions]], Array]
PositionalEncodingCt = Callable[..., PositionalEncodingFn]
BinaryOperator = Callable[[Array, Array], Array]


class Encoding(nn.Module):
    r"""Encodes a vector at each position (i.e. time step)
    of the sequence using an optional preprocessing step
    and a positional encoding.

    Preprocessed inputs get reshaped to
    :math:`\sR^{\nBatchSize \times T \times d}`.
    Then, optionally, positional encodings get concatenated
    or added to the sequence.

    .. math::

        \begin{aligned}
            & \textrm{Encoding}( \\
            & \quad x \in \sR^{\nBatchSize \times T \times d} \\
            & \quad \_ \\
            & \quad \theta \gets Preprocessing() \\
            & \quad \theta \gets PositionalEncoding() \\
            & ) \\
            & \rightarrow \sR^{\nBatchSize \times T \times d^{\prime}}
        \end{aligned}

        \begin{aligned}
            & \textrm{Encoding}( \\
            & \quad x \in \sR^{\nBatchSize \times T \times d} \\
            & \quad t \in \sN^{\nSeqLen^{\prime}} \\
            & \quad \_ \\
            & \quad \theta \gets Preprocessing() \\
            & \quad \theta \gets PositionalEncoding() \\
            & ) \\
            & \rightarrow \sR^{\nBatchSize \times T^{\prime} \times d^{\prime}}
        \end{aligned}

    Args:
        inputs: an input sequence.
        output_positions: a subset of positions (i.e. time steps) within
            the sequence encoding will be calculated.

    Returns:
        encoded vectors.
    """

    preprocessing: PreprocessingCt = cast(PreprocessingCt, lambda: cb.identity(n_in=1))
    r"""a constructor for a module or function that accepts inputs as its single argument
    and returns the preprocessed inputs as a single output."""

    positional_encoding: Optional[PositionalEncodingCt] = None
    r"""a type of the positional encoding (e.g.
    :class:`flax_extra.layer.TrainablePositionalEncoding`
    or :class:`flax_extra.layer.FourierPositionEncoding`)."""

    aggregation: BinaryOperator = cast(BinaryOperator, cb.concatenate())
    r"""a binary operation (e.g. :class:`flax_extra.combinator.concatenate`
    or :class:`flax_extra.combinator.add`)."""

    @nn.compact
    def __call__(  # type: ignore[override] # pylint: disable=arguments-differ
        self,
        inputs: Array,
        output_positions: Optional[Positions],
    ) -> Array:
        batch_size = inputs.shape[0]

        # Preprocess.
        preprocessed_inputs = self.preprocessing()(inputs)

        # Reshape to a single feature dimension.
        d_preprocessed_inputs = preprocessed_inputs.shape[-1]
        preprocessed_inputs_1d = jnp.reshape(
            preprocessed_inputs,
            (batch_size, -1, d_preprocessed_inputs),
        )

        if self.positional_encoding is None:
            return preprocessed_inputs_1d  # type: ignore

        ## Enhance with positional information.
        return self.aggregation(  # type: ignore
            preprocessed_inputs_1d,
            self.positional_encoding()(  # pylint: disable=not-callable
                batch_size,
                output_positions,
            ),
        )


EncodingCt = Callable[..., Encoding]
