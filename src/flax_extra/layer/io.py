r"""Input and output encoding utility functions."""

from typing import cast, Any
from functools import partial
from flax_extra.layer._encoding import EncodingCt, PositionalEncodingCt
from flax_extra.layer._decoding import DecodingCt
from flax_extra.layer._multimodal_encoding import (
    MultimodalEncoding,
    MultimodalEncodingCt,
)
from flax_extra.layer._multimodal_positional_encoding import (
    MultimodalPositionalEncoding,
    MultimodalPositionalEncodingCt,
)
from flax_extra.layer._multimodal_decoding import (
    MultimodalDecoding,
    MultimodalDecodingCt,
)


def _initialize_multimodal(kind: type, *modalities: Any, **kvargs: Any) -> Any:
    n_modalities = len(modalities)
    if n_modalities < 1:
        raise ValueError(
            f"Cannot initialize {kind} " "at least one modality is required."
        )
    if n_modalities == 1:
        modality = modalities[0]
        modality = partial(modality, **kvargs)
        return partial(kind, modalities=[modality])
    return partial(kind, modalities=modalities, **kvargs)


def input_encoding(
    *modalities: EncodingCt,
    **kvargs: Any,
) -> MultimodalEncodingCt:
    r"""Defines an encoding for single or multimodal input."""
    return cast(
        MultimodalEncodingCt,
        _initialize_multimodal(MultimodalEncoding, *modalities, **kvargs),
    )


def output_decoding(
    *modalities: DecodingCt,
    **kvargs: Any,
) -> MultimodalDecodingCt:
    r"""Defines a decoding for single or multimodal output."""
    return cast(
        MultimodalDecodingCt,
        _initialize_multimodal(MultimodalDecoding, *modalities, **kvargs),
    )


def query_encoding(
    *modalities: PositionalEncodingCt,
    **kvargs: Any,
) -> MultimodalPositionalEncodingCt:
    r"""Defines a decoder or encoder query."""
    return cast(
        MultimodalPositionalEncodingCt,
        _initialize_multimodal(MultimodalPositionalEncoding, *modalities, **kvargs),
    )


target_encoding = input_encoding
r"""Defines an encoding for single or multimodal target."""
