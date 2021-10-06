r"""Flax linen layers."""

from flax_extra.layer._lstm import LSTM, LSTMCell
from flax_extra.layer._feedforward import FeedForward, FeedForwardFn, FeedForwardCt
from flax_extra.layer._encoding import Encoding
from flax_extra.layer._fourier_positional_encoding import FourierPositionEncoding
from flax_extra.layer._trainable_positional_encoding import TrainablePositionalEncoding
from flax_extra.layer._trainable_positional_padding import TrainablePositionalPadding
from flax_extra.layer._trainable_positional_masking import TrainablePositionalMasking
from flax_extra.layer._decoding import Decoding
from flax_extra.layer._embedding_decoding import EmbedDecoding
from flax_extra.layer._multimodal_encoding import MultimodalEncoding
from flax_extra.layer._multimodal_positional_encoding import (
    MultimodalPositionalEncoding,
)
from flax_extra.layer._multimodal_decoding import MultimodalDecoding
from flax_extra.layer._attention import (
    Attention,
    QKVAttention,
    QKVAttentionFn,
    QKVAttentionCt,
    KVQAttention,
    KVQAttentionFn,
    KVQAttentionCt,
    SelfAttention,
    SelfAttentionFn,
    SelfAttentionCt,
    attend,
    cross_attention_mask,
)

__all__ = [
    "LSTM",
    "LSTMCell",
    "FeedForward",
    "FeedForwardFn",
    "FeedForwardCt",
    "Encoding",
    "FourierPositionEncoding",
    "TrainablePositionalEncoding",
    "TrainablePositionalPadding",
    "TrainablePositionalMasking",
    "Decoding",
    "EmbedDecoding",
    "MultimodalEncoding",
    "MultimodalPositionalEncoding",
    "MultimodalDecoding",
    "Attention",
    "QKVAttention",
    "QKVAttentionFn",
    "QKVAttentionCt",
    "KVQAttention",
    "KVQAttentionFn",
    "KVQAttentionCt",
    "SelfAttention",
    "SelfAttentionFn",
    "SelfAttentionCt",
    "attend",
    "cross_attention_mask",
]
