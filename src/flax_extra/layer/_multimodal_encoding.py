r"""Multimodal encoding."""

from typing import Callable, List, Optional
from functools import reduce
import jax.numpy as jnp
from flax import linen as nn
from flax_extra.layer._trainable_positional_padding import TrainablePositionalPadding
from flax_extra.layer._trainable_positional_masking import TrainablePositionalMasking
from flax_extra.layer._multimodal_positional_encoding import (
    _normalize_modalities,
    _verify_modalities,
)
from flax_extra.layer._encoding import EncodingCt

Array = jnp.ndarray
Positions = List[int]

_EncodeInitializer = tuple[int, List[int], List[Array]]
_EncodeData = tuple[Array, Optional[Positions], EncodingCt]


class MultimodalEncoding(nn.Module):
    r"""Encodes multiple modalities.

    Each modality is padded to the same feature dimension and
    gets concatenatenated to form a single sequence.

    .. math::

        \begin{aligned}
            & \textrm{MultimodalEncoding}( \\
            & \quad x \in n_{mod} \times \sR^{m \times T \times d} \\
            & \quad \_ \\
            & \quad \theta \gets PositionalEncoding() \\
            & \quad \theta \gets TrainablePositionalPadding() \\
            & \quad \theta \gets TrainablePositionalMasking() \\
            & ) \\
            & \rightarrow \\
            & \quad h \in \sR^{\nBatchSize \times T^{\prime} \times d^{\prime}} \\
            & \quad l_{x} \in n_{mod} \times \sN
        \end{aligned}

        \begin{aligned}
            & \textrm{MultimodalEncoding}( \\
            & \quad x \in n_{mod} \times \sR^{m \times T \times d} \\
            & \quad t \in n_{mod} \times \sN^{\nSeqLen^{\prime}} \\
            & \quad \_ \\
            & \quad \theta \gets PositionalEncoding() \\
            & \quad \theta \gets TrainablePositionalPadding() \\
            & \quad \theta \gets TrainablePositionalMasking() \\
            & ) \\
            & \rightarrow \\
            & \quad h \in \sR^{\nBatchSize \times T^{\prime} \times d^{\prime}} \\
            & \quad l_{x} \in n_{mod} \times \sN
        \end{aligned}

    Args:
        multimodal_inputs: a batch size of the sequence.
        multimodal_output_positions: a subset of positions (i.e. time steps) within
            each modality encoding will be calculated.

    Returns:
        encoded vectors combined in a single sequence
            and sequence lengths of original modalities.
    """

    modalities: List[EncodingCt]
    r"""a list of constructors for an encoding module (e.g.
    :class:`flax_extra.layer.Encoding`) for each modality."""

    mask_rates: Optional[List[int]] = None
    r"""a probability for each modality being masked."""

    d_reserved: int = 1
    r"""a number of reserved feature dimensions."""

    @property
    def n_modalities(self) -> int:
        r"""a number of modalities."""
        return len(self.modalities)

    @nn.compact
    def __call__(  # type: ignore[override] # pylint: disable=arguments-differ, disable=too-many-locals
        self,
        multimodal_inputs: List[Array],
        multimodal_output_positions: Optional[List[Optional[Positions]]],
    ) -> tuple[Array, List[int]]:
        if _verify_modalities(multimodal_inputs, self.n_modalities) is None:
            raise ValueError(
                f"The number of {self.__name__} modalities {self.n_modalities} "
                f"doesn't match the number of its arguments."
            )

        multimodal_output_positions = _normalize_modalities(
            multimodal_output_positions,
            self.n_modalities,
        )
        if multimodal_output_positions is None:
            raise ValueError(
                f"The number of {self.__name__} modalities {self.n_modalities} "
                f"doesn't match the number of provided output positions."
            )

        mask_rates = _normalize_modalities(self.mask_rates, self.n_modalities)
        if mask_rates is None:
            raise ValueError(
                f"The number of {self.__name__} modalities {self.n_modalities} "
                f"doesn't match the number of provided mask rates."
            )

        # Single modality.
        if len(self.modalities) == 1:
            encoding = self.modalities[0]
            encoded_inputs = encoding()(
                multimodal_inputs[0],
                multimodal_output_positions[0],
            )
            seqlen_encoded_input = encoded_inputs.shape[1]
            return encoded_inputs, [seqlen_encoded_input]

        # Multimodal.
        # Encode modalities, count input lengths and max input dimension.
        def encode(acc: _EncodeInitializer, data: _EncodeData) -> _EncodeInitializer:
            inputs, output_positions, encoding = data
            d_input_max, seqlen_inputs, multimodal_encoded_inputs = acc
            encoded_inputs = encoding()(inputs, output_positions)
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
            zip(multimodal_inputs, multimodal_output_positions, self.modalities),
            encode_initializer,
        )
        d_input_max += self.d_reserved

        # Pad inputs along their dimension axis, then apply masking.
        def padmask(acc: List[Array], data: tuple[float, Array]) -> List[Array]:
            mask_rate, inputs = data
            padded_inputs = TrainablePositionalPadding(d_max=d_input_max)(inputs)
            if mask_rate is not None:
                padded_inputs = TrainablePositionalMasking(rate=mask_rate)(
                    padded_inputs
                )
            return acc + [padded_inputs]

        padmask_initializer: List[Array] = []
        multimodal_padded_inputs = reduce(
            padmask,
            zip(mask_rates, multimodal_encoded_inputs),
            padmask_initializer,
        )

        return jnp.concatenate(multimodal_padded_inputs, axis=1), seqlen_inputs


MultimodalEncodingCt = Callable[..., MultimodalEncoding]
