r"""Flax linen layers."""

from flax_extra.layer import io
from flax_extra.layer._lstm import LSTM, LSTMCell
from flax_extra.layer._feedforward import (
    FeedForward,
    FeedForwardCt,
)
from flax_extra.layer._encoding import (
    Encoding,
    EncodingCt,
    PositionalEncodingCt,
)
from flax_extra.layer._fourier_positional_encoding import FourierPositionEncoding
from flax_extra.layer._trainable_positional_encoding import TrainablePositionalEncoding
from flax_extra.layer._trainable_positional_padding import TrainablePositionalPadding
from flax_extra.layer._trainable_positional_masking import TrainablePositionalMasking
from flax_extra.layer._decoding import (
    Decoding,
    DecodingCt,
    EmbeddingDecodingCt,
)
from flax_extra.layer._embedding_decoding import EmbedDecoding
from flax_extra.layer._multimodal_encoding import (
    MultimodalEncoding,
    MultimodalEncodingCt,
)
from flax_extra.layer._multimodal_positional_encoding import (
    MultimodalPositionalEncoding,
    MultimodalPositionalEncodingCt,
)
from flax_extra.layer._multimodal_decoding import MultimodalDecoding
from flax_extra.layer._attention import (
    Attention,
    QKVAttention,
    QKVAttentionCt,
    KVQAttention,
    KVQAttentionCt,
    SelfAttention,
    SelfAttentionCt,
    attend,
    cross_attention_mask,
)

__all__ = [
    "io",
    "LSTM",
    "LSTMCell",
    "FeedForward",
    "FeedForwardCt",
    "Encoding",
    "EncodingCt",
    "PositionalEncodingCt",
    "FourierPositionEncoding",
    "TrainablePositionalEncoding",
    "TrainablePositionalPadding",
    "TrainablePositionalMasking",
    "Decoding",
    "DecodingCt",
    "EmbeddingDecodingCt",
    "EmbedDecoding",
    "MultimodalEncoding",
    "MultimodalEncodingCt",
    "MultimodalPositionalEncoding",
    "MultimodalPositionalEncodingCt",
    "MultimodalDecoding",
    "Attention",
    "QKVAttention",
    "QKVAttentionCt",
    "KVQAttention",
    "KVQAttentionCt",
    "SelfAttention",
    "SelfAttentionCt",
    "attend",
    "cross_attention_mask",
]
