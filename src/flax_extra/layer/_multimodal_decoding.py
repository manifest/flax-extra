r"""Multimodal decoding."""

from typing import cast, Callable, List, Union
from functools import reduce
import jax.numpy as jnp
from flax import linen as nn
from flax_extra.layer._decoding import EmbeddingDecodingCt, DecodingCt
from flax_extra.layer._multimodal_positional_encoding import _verify_modalities
from flax_extra import combinator as cb

Array = jnp.ndarray

MultimodalDecodingFn = Callable[[Array, List[int]], List[Array]]
MultimodalDecodingCt = Callable[..., MultimodalDecodingFn]


def _split_modalities(inputs: Array, seqlens: List[int]) -> List[Array]:
    r"""Partitions a multimodal tensor into tensors for each modality."""

    def split(acc: tuple[int, List[Array]], seqlen: int) -> tuple[int, List[Array]]:
        index, modalities = acc
        modality = inputs[:, index : index + seqlen]
        return (
            index + seqlen,
            modalities + [modality],
        )

    initializer: tuple[int, List[Array]] = (0, [])
    _index, modalities = reduce(split, seqlens, initializer)
    return modalities


_DecodeInitializer = List[Array]
_DecodeData = tuple[Array, DecodingCt]


class MultimodalDecoding(nn.Module):
    r"""Given original sequence lengths, partitions a multimodal tensor
    into tensors for each modality and decode each of them.

    .. math::

        \begin{aligned}
            & \textrm{MultimodalDecoding}( \\
            & \quad y \in \sR^{\nBatchSize \times \nSeqLen \times d} \\
            & \quad l_{x} \in n_{mod} \times \sN \\
            & \quad \_ \\
            & \quad \theta \gets MultimodalEmbeddingDecoding() \\
            & \quad \theta \gets n_{mod} \times Decoding() \\
            & ) \\
            & \rightarrow n_{mod} \times \sR^{\nBatchSize \times \dots d^{\prime}}
        \end{aligned}

    Args:
        outputs: multimodal model's outputs.
        seqlen_outputs: sequence lengths of original modalities.

    Returns:
        partitioned modalities.
    """

    modalities: List[DecodingCt]
    r"""a list of constructors for a decoding module (e.g.
    :class:`flax_extra.layer.Decoding`) for each modality."""

    multimodal_embedding_decoding: EmbeddingDecodingCt = cast(
        EmbeddingDecodingCt,
        lambda: cb.identity(n_in=1),
    )
    r"""a constructor for a embedding decoder module
    (e.g. :class:`flax_extra.layer.EmbedDecoding` or
    :class:`flax.linen.Dense`)."""

    @nn.compact
    def __call__(  # type: ignore[override] # pylint: disable=arguments-differ
        self,
        outputs: Array,
        seqlen_outputs: List[int],
    ) -> Union[Array, List[Array]]:
        if self.d_hidden is not None:
            outputs = nn.Dense(features=self.d_hidden)(outputs)

        n_modalities = len(self.modalities)
        if _verify_modalities(seqlen_outputs, n_modalities) is None:
            raise ValueError(
                f"The number of {self.__name__} modalities {n_modalities} "
                f"doesn't match the number of output sequences."
            )

        # Single modality.
        if n_modalities == 1:
            decoding = self.modalities[0]
            return decoding()(outputs)

        # Multimodal.
        multimodal_outputs = _split_modalities(outputs, seqlen_outputs)

        def decode(acc: _DecodeInitializer, data: _DecodeData) -> _DecodeInitializer:
            outputs, decoding = data
            decoded_outputs = decoding()(outputs)
            return acc + [decoded_outputs]

        decode_initializer: _DecodeInitializer = []
        return reduce(
            decode, zip(multimodal_outputs, self.modalities), decode_initializer
        )
