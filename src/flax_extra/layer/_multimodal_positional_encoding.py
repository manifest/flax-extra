r"""Multimodal positional encoding."""

from typing import Any, Callable, List, Optional
from functools import reduce
import jax.numpy as jnp
from flax import linen as nn
from flax_extra.layer._trainable_positional_padding import TrainablePositionalPadding
from flax_extra.layer._encoding import PositionalEncodingCt

Array = jnp.ndarray
Positions = List[int]


def _normalize_modalities(
    values: Optional[List[Any]],
    n_modalities: int,
) -> Optional[List[Any]]:
    if values is None:
        values = [None] * n_modalities
    return _verify_modalities(values, n_modalities)


def _verify_modalities(
    values: List[Any],
    n_modalities: int,
) -> Optional[List[Any]]:
    n_values = len(values)
    if n_modalities != n_values:
        return None
    return values


_EncodeInitializer = tuple[int, List[int], List[Array]]
_EncodeData = tuple[PositionalEncodingCt, Optional[Positions]]


class MultimodalPositionalEncoding(nn.Module):
    r"""Encodes positional encoding for each modality.

    Each modality is padded to the same feature dimension and
    gets concatenatenated to form a single sequence.

    .. math::

        \begin{aligned}
            & \textrm{MultimodalPositionalEncoding}( \\
            & \quad m \in \sN \\
            & \quad \_ \\
            & \quad \theta \gets PositionalEncoding() \\
            & \quad \theta \gets TrainablePositionalPadding() \\
            & ) \\
            & \rightarrow \\
            & \quad h \in \sR^{\nBatchSize \times T^{\prime} \times d^{\prime}} \\
            & \quad l_{x} \in n_{modailies} \times \sN
        \end{aligned}

        \begin{aligned}
            & \textrm{MultimodalPositionalEncoding}( \\
            & \quad m \in \sN \\
            & \quad t \in n_{mod} \times \sN^{\nSeqLen^{\prime}} \\
            & \quad \_ \\
            & \quad \theta \gets PositionalEncoding() \\
            & \quad \theta \gets TrainablePositionalPadding() \\
            & ) \\
            & \rightarrow \\
            & \quad h \in \sR^{\nBatchSize \times T^{\prime} \times d^{\prime}} \\
            & \quad l_{x} \in n_{modailies} \times \sN
        \end{aligned}

    Args:
        batch_size: a batch size of the sequence.
        multimodal_output_positions: a subset of positions (i.e. time steps) within
            each modality positional encoding will be calculated.

    Returns:
        positional encoding vectors combined in a single sequence
            and sequence lengths of original modalities.
    """

    modalities: List[PositionalEncodingCt]
    r"""a list of constructors for a positional encoding module (e.g.
    :class:`flax_extra.layer.TrainablePositionalEncoding`
    or :class:`flax_extra.layer.FourierPositionEncoding`)
    for each modality."""

    d_reserved: int = 1
    r"""a number of reserved feature dimensions."""

    @property
    def n_modalities(self) -> int:
        r"""a number of modalities."""
        return len(self.modalities)

    @nn.compact
    def __call__(  # type: ignore[override] # pylint: disable=arguments-differ
        self,
        batch_size: int,
        multimodal_output_positions: Optional[List[Optional[Positions]]],
    ) -> tuple[Array, List[int]]:
        multimodal_output_positions = _normalize_modalities(
            multimodal_output_positions,
            self.n_modalities,
        )
        if multimodal_output_positions is None:
            raise ValueError(
                f"The number of {self.__name__} modalities {self.n_modalities} "
                f"doesn't match the number of provided subsamples."
            )

        # Single modality.
        if self.n_modalities == 1:
            positional_encoding = self.modalities[0]
            encoded_inputs = positional_encoding()(
                batch_size,
                multimodal_output_positions[0],
            )
            seqlen_encoded_input = encoded_inputs.shape[1]
            return encoded_inputs, [seqlen_encoded_input]

        # Multimodal.
        # Encode modalities, count input lengths and max input dimension.
        def encode(acc: _EncodeInitializer, data: _EncodeData) -> _EncodeInitializer:
            positional_encoding, output_positions = data
            d_input_max, seqlen_inputs, multimodal_encoded_inputs = acc
            encoded_inputs = positional_encoding()(
                batch_size,
                output_positions,
            )
            d_input = encoded_inputs.shape[-1]
            seqlen_input = encoded_inputs.shape[1]
            return (
                max(d_input_max, d_input),
                seqlen_inputs + [seqlen_input],
                multimodal_encoded_inputs + [encoded_inputs],
            )

        encode_initializer: _EncodeInitializer = (0, [], [])
        d_input_max, seqlen_inputs, multimodal_encoded_inputs = reduce(
            encode,
            zip(self.modalities, multimodal_output_positions),
            encode_initializer,
        )
        d_input_max += self.d_reserved

        # Pad inputs along their feature dimension axis.
        def pad(acc: List[Array], inputs: Array) -> List[Array]:
            padded_inputs = TrainablePositionalPadding(d_max=d_input_max)(inputs)
            return acc + [padded_inputs]

        pad_initializer: List[Array] = []
        multimodal_padded_inputs = reduce(
            pad,
            multimodal_encoded_inputs,
            pad_initializer,
        )

        return jnp.concatenate(multimodal_padded_inputs, axis=1), seqlen_inputs


MultimodalPositionalEncodingCt = Callable[..., MultimodalPositionalEncoding]
