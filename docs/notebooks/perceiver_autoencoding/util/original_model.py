import pickle
from jax import numpy as jnp
import haiku as hk
from flax.core import freeze
from flax.traverse_util import unflatten_dict

PARAM_MAP = {
    "encoder/~/trainable_position_encoding:pos_embs": "params/EncodedQueryEncoding/TrainablePositionalEncoding_0/positional_encoding",
    "multimodal_preprocessor/audio_padding:pos_embs": "params/InputEncoding/TrainablePositionalPadding_0/TrainablePositionalEncoding_0/positional_encoding",
    "multimodal_preprocessor/image_padding:pos_embs": "params/InputEncoding/TrainablePositionalPadding_1/TrainablePositionalEncoding_0/positional_encoding",
    "multimodal_preprocessor/label_padding:pos_embs": "params/InputEncoding/TrainablePositionalPadding_2/TrainablePositionalEncoding_0/positional_encoding",
    "multimodal_preprocessor/audio_mask_token:pos_embs": "params/InputEncoding/TrainablePositionalMasking_0/TrainablePositionalEncoding_0/positional_encoding",
    "multimodal_preprocessor/image_mask_token:pos_embs": "params/InputEncoding/TrainablePositionalMasking_1/TrainablePositionalEncoding_0/positional_encoding",
    "multimodal_preprocessor/label_mask_token:pos_embs": "params/InputEncoding/TrainablePositionalMasking_2/TrainablePositionalEncoding_0/positional_encoding",
    "classification_decoder/~/basic_decoder/~/trainable_position_encoding:pos_embs": "params/DecoderQueryEncoding/TrainablePositionalEncoding_0/positional_encoding",
    "multimodal_decoder/~decoder_query/audio_padding:pos_embs": "params/DecoderQueryEncoding/TrainablePositionalPadding_0/TrainablePositionalEncoding_0/positional_encoding",
    "multimodal_decoder/~decoder_query/image_padding:pos_embs": "params/DecoderQueryEncoding/TrainablePositionalPadding_1/TrainablePositionalEncoding_0/positional_encoding",
    "multimodal_decoder/~decoder_query/label_padding:pos_embs": "params/DecoderQueryEncoding/TrainablePositionalPadding_2/TrainablePositionalEncoding_0/positional_encoding",
    "multimodal_decoder/~/basic_decoder/output:b": "params/OutputDecoding/Dense_0/bias",
    "multimodal_decoder/~/basic_decoder/output:w": "params/OutputDecoding/Dense_0/kernel",
    "audio_postprocessor/linear:b": "params/OutputDecoding/Decoding_0/Dense_0/bias",
    "audio_postprocessor/linear:w": "params/OutputDecoding/Decoding_0/Dense_0/kernel",
    "projection_postprocessor/linear:b": "params/OutputDecoding/Decoding_1/Dense_0/bias",
    "projection_postprocessor/linear:w": "params/OutputDecoding/Decoding_1/Dense_0/kernel",
    "classification_postprocessor/linear:b": "params/OutputDecoding/Decoding_2/Dense_0/bias",
    "classification_postprocessor/linear:w": "params/OutputDecoding/Decoding_2/Dense_0/kernel",
    "multimodal_decoder/~/basic_decoder/cross_attention/attention/linear:b": "params/Decoder/CrossAttentionBlock_0/Attention_0/query/bias",
    "multimodal_decoder/~/basic_decoder/cross_attention/attention/linear:w": "params/Decoder/CrossAttentionBlock_0/Attention_0/query/kernel",
    "multimodal_decoder/~/basic_decoder/cross_attention/attention/linear_1:b": "params/Decoder/CrossAttentionBlock_0/Attention_0/key/bias",
    "multimodal_decoder/~/basic_decoder/cross_attention/attention/linear_1:w": "params/Decoder/CrossAttentionBlock_0/Attention_0/key/kernel",
    "multimodal_decoder/~/basic_decoder/cross_attention/attention/linear_2:b": "params/Decoder/CrossAttentionBlock_0/Attention_0/value/bias",
    "multimodal_decoder/~/basic_decoder/cross_attention/attention/linear_2:w": "params/Decoder/CrossAttentionBlock_0/Attention_0/value/kernel",
    "multimodal_decoder/~/basic_decoder/cross_attention/attention/linear_3:b": "params/Decoder/CrossAttentionBlock_0/Attention_0/out/bias",
    "multimodal_decoder/~/basic_decoder/cross_attention/attention/linear_3:w": "params/Decoder/CrossAttentionBlock_0/Attention_0/out/kernel",
    "multimodal_decoder/~/basic_decoder/cross_attention/layer_norm:offset": "params/Decoder/CrossAttentionBlock_0/LayerNorm_0/bias",
    "multimodal_decoder/~/basic_decoder/cross_attention/layer_norm:scale": "params/Decoder/CrossAttentionBlock_0/LayerNorm_0/scale",
    "multimodal_decoder/~/basic_decoder/cross_attention/layer_norm_1:offset": "params/Decoder/CrossAttentionBlock_0/LayerNorm_1/bias",
    "multimodal_decoder/~/basic_decoder/cross_attention/layer_norm_1:scale": "params/Decoder/CrossAttentionBlock_0/LayerNorm_1/scale",
    "multimodal_decoder/~/basic_decoder/cross_attention/layer_norm_2:offset": "params/Decoder/CrossAttentionBlock_0/LayerNorm_2/bias",
    "multimodal_decoder/~/basic_decoder/cross_attention/layer_norm_2:scale": "params/Decoder/CrossAttentionBlock_0/LayerNorm_2/scale",
    "multimodal_decoder/~/basic_decoder/cross_attention/mlp/linear:b": "params/Decoder/CrossAttentionBlock_0/FeedForward_0/Dense_0/bias",
    "multimodal_decoder/~/basic_decoder/cross_attention/mlp/linear:w": "params/Decoder/CrossAttentionBlock_0/FeedForward_0/Dense_0/kernel",
    "multimodal_decoder/~/basic_decoder/cross_attention/mlp/linear_1:b": "params/Decoder/CrossAttentionBlock_0/FeedForward_0/Dense_1/bias",
    "multimodal_decoder/~/basic_decoder/cross_attention/mlp/linear_1:w": "params/Decoder/CrossAttentionBlock_0/FeedForward_0/Dense_1/kernel",
    "encoder/~/cross_attention/attention/linear:b": "params/Encoder/CrossAttentionBlock_0/Attention_0/query/bias",
    "encoder/~/cross_attention/attention/linear:w": "params/Encoder/CrossAttentionBlock_0/Attention_0/query/kernel",
    "encoder/~/cross_attention/attention/linear_1:b": "params/Encoder/CrossAttentionBlock_0/Attention_0/key/bias",
    "encoder/~/cross_attention/attention/linear_1:w": "params/Encoder/CrossAttentionBlock_0/Attention_0/key/kernel",
    "encoder/~/cross_attention/attention/linear_2:b": "params/Encoder/CrossAttentionBlock_0/Attention_0/value/bias",
    "encoder/~/cross_attention/attention/linear_2:w": "params/Encoder/CrossAttentionBlock_0/Attention_0/value/kernel",
    "encoder/~/cross_attention/attention/linear_3:b": "params/Encoder/CrossAttentionBlock_0/Attention_0/out/bias",
    "encoder/~/cross_attention/attention/linear_3:w": "params/Encoder/CrossAttentionBlock_0/Attention_0/out/kernel",
    "encoder/~/cross_attention/layer_norm:offset": "params/Encoder/CrossAttentionBlock_0/LayerNorm_0/bias",
    "encoder/~/cross_attention/layer_norm:scale": "params/Encoder/CrossAttentionBlock_0/LayerNorm_0/scale",
    "encoder/~/cross_attention/layer_norm_1:offset": "params/Encoder/CrossAttentionBlock_0/LayerNorm_1/bias",
    "encoder/~/cross_attention/layer_norm_1:scale": "params/Encoder/CrossAttentionBlock_0/LayerNorm_1/scale",
    "encoder/~/cross_attention/layer_norm_2:offset": "params/Encoder/CrossAttentionBlock_0/LayerNorm_2/bias",
    "encoder/~/cross_attention/layer_norm_2:scale": "params/Encoder/CrossAttentionBlock_0/LayerNorm_2/scale",
    "encoder/~/cross_attention/mlp/linear:b": "params/Encoder/CrossAttentionBlock_0/FeedForward_0/Dense_0/bias",
    "encoder/~/cross_attention/mlp/linear:w": "params/Encoder/CrossAttentionBlock_0/FeedForward_0/Dense_0/kernel",
    "encoder/~/cross_attention/mlp/linear_1:b": "params/Encoder/CrossAttentionBlock_0/FeedForward_0/Dense_1/bias",
    "encoder/~/cross_attention/mlp/linear_1:w": "params/Encoder/CrossAttentionBlock_0/FeedForward_0/Dense_1/kernel",
    "encoder/~/self_attention/attention/linear:b": "params/Processor/SelfAttentionBlock_0/SelfAttention_0/query/bias",
    "encoder/~/self_attention/attention/linear:w": "params/Processor/SelfAttentionBlock_0/SelfAttention_0/query/kernel",
    "encoder/~/self_attention/attention/linear_1:b": "params/Processor/SelfAttentionBlock_0/SelfAttention_0/key/bias",
    "encoder/~/self_attention/attention/linear_1:w": "params/Processor/SelfAttentionBlock_0/SelfAttention_0/key/kernel",
    "encoder/~/self_attention/attention/linear_2:b": "params/Processor/SelfAttentionBlock_0/SelfAttention_0/value/bias",
    "encoder/~/self_attention/attention/linear_2:w": "params/Processor/SelfAttentionBlock_0/SelfAttention_0/value/kernel",
    "encoder/~/self_attention/attention/linear_3:b": "params/Processor/SelfAttentionBlock_0/SelfAttention_0/out/bias",
    "encoder/~/self_attention/attention/linear_3:w": "params/Processor/SelfAttentionBlock_0/SelfAttention_0/out/kernel",
    "encoder/~/self_attention/layer_norm:offset": "params/Processor/SelfAttentionBlock_0/LayerNorm_0/bias",
    "encoder/~/self_attention/layer_norm:scale": "params/Processor/SelfAttentionBlock_0/LayerNorm_0/scale",
    "encoder/~/self_attention/layer_norm_1:offset": "params/Processor/SelfAttentionBlock_0/LayerNorm_1/bias",
    "encoder/~/self_attention/layer_norm_1:scale": "params/Processor/SelfAttentionBlock_0/LayerNorm_1/scale",
    "encoder/~/self_attention/mlp/linear:b": "params/Processor/SelfAttentionBlock_0/FeedForward_0/Dense_0/bias",
    "encoder/~/self_attention/mlp/linear:w": "params/Processor/SelfAttentionBlock_0/FeedForward_0/Dense_0/kernel",
    "encoder/~/self_attention/mlp/linear_1:b": "params/Processor/SelfAttentionBlock_0/FeedForward_0/Dense_1/bias",
    "encoder/~/self_attention/mlp/linear_1:w": "params/Processor/SelfAttentionBlock_0/FeedForward_0/Dense_1/kernel",
    "encoder/~/self_attention_1/attention/linear:b": "params/Processor/SelfAttentionBlock_1/SelfAttention_0/query/bias",
    "encoder/~/self_attention_1/attention/linear:w": "params/Processor/SelfAttentionBlock_1/SelfAttention_0/query/kernel",
    "encoder/~/self_attention_1/attention/linear_1:b": "params/Processor/SelfAttentionBlock_1/SelfAttention_0/key/bias",
    "encoder/~/self_attention_1/attention/linear_1:w": "params/Processor/SelfAttentionBlock_1/SelfAttention_0/key/kernel",
    "encoder/~/self_attention_1/attention/linear_2:b": "params/Processor/SelfAttentionBlock_1/SelfAttention_0/value/bias",
    "encoder/~/self_attention_1/attention/linear_2:w": "params/Processor/SelfAttentionBlock_1/SelfAttention_0/value/kernel",
    "encoder/~/self_attention_1/attention/linear_3:b": "params/Processor/SelfAttentionBlock_1/SelfAttention_0/out/bias",
    "encoder/~/self_attention_1/attention/linear_3:w": "params/Processor/SelfAttentionBlock_1/SelfAttention_0/out/kernel",
    "encoder/~/self_attention_1/layer_norm:offset": "params/Processor/SelfAttentionBlock_1/LayerNorm_0/bias",
    "encoder/~/self_attention_1/layer_norm:scale": "params/Processor/SelfAttentionBlock_1/LayerNorm_0/scale",
    "encoder/~/self_attention_1/layer_norm_1:offset": "params/Processor/SelfAttentionBlock_1/LayerNorm_1/bias",
    "encoder/~/self_attention_1/layer_norm_1:scale": "params/Processor/SelfAttentionBlock_1/LayerNorm_1/scale",
    "encoder/~/self_attention_1/mlp/linear:b": "params/Processor/SelfAttentionBlock_1/FeedForward_0/Dense_0/bias",
    "encoder/~/self_attention_1/mlp/linear:w": "params/Processor/SelfAttentionBlock_1/FeedForward_0/Dense_0/kernel",
    "encoder/~/self_attention_1/mlp/linear_1:b": "params/Processor/SelfAttentionBlock_1/FeedForward_0/Dense_1/bias",
    "encoder/~/self_attention_1/mlp/linear_1:w": "params/Processor/SelfAttentionBlock_1/FeedForward_0/Dense_1/kernel",
    "encoder/~/self_attention_2/attention/linear:b": "params/Processor/SelfAttentionBlock_2/SelfAttention_0/query/bias",
    "encoder/~/self_attention_2/attention/linear:w": "params/Processor/SelfAttentionBlock_2/SelfAttention_0/query/kernel",
    "encoder/~/self_attention_2/attention/linear_1:b": "params/Processor/SelfAttentionBlock_2/SelfAttention_0/key/bias",
    "encoder/~/self_attention_2/attention/linear_1:w": "params/Processor/SelfAttentionBlock_2/SelfAttention_0/key/kernel",
    "encoder/~/self_attention_2/attention/linear_2:b": "params/Processor/SelfAttentionBlock_2/SelfAttention_0/value/bias",
    "encoder/~/self_attention_2/attention/linear_2:w": "params/Processor/SelfAttentionBlock_2/SelfAttention_0/value/kernel",
    "encoder/~/self_attention_2/attention/linear_3:b": "params/Processor/SelfAttentionBlock_2/SelfAttention_0/out/bias",
    "encoder/~/self_attention_2/attention/linear_3:w": "params/Processor/SelfAttentionBlock_2/SelfAttention_0/out/kernel",
    "encoder/~/self_attention_2/layer_norm:offset": "params/Processor/SelfAttentionBlock_2/LayerNorm_0/bias",
    "encoder/~/self_attention_2/layer_norm:scale": "params/Processor/SelfAttentionBlock_2/LayerNorm_0/scale",
    "encoder/~/self_attention_2/layer_norm_1:offset": "params/Processor/SelfAttentionBlock_2/LayerNorm_1/bias",
    "encoder/~/self_attention_2/layer_norm_1:scale": "params/Processor/SelfAttentionBlock_2/LayerNorm_1/scale",
    "encoder/~/self_attention_2/mlp/linear:b": "params/Processor/SelfAttentionBlock_2/FeedForward_0/Dense_0/bias",
    "encoder/~/self_attention_2/mlp/linear:w": "params/Processor/SelfAttentionBlock_2/FeedForward_0/Dense_0/kernel",
    "encoder/~/self_attention_2/mlp/linear_1:b": "params/Processor/SelfAttentionBlock_2/FeedForward_0/Dense_1/bias",
    "encoder/~/self_attention_2/mlp/linear_1:w": "params/Processor/SelfAttentionBlock_2/FeedForward_0/Dense_1/kernel",
    "encoder/~/self_attention_3/attention/linear:b": "params/Processor/SelfAttentionBlock_3/SelfAttention_0/query/bias",
    "encoder/~/self_attention_3/attention/linear:w": "params/Processor/SelfAttentionBlock_3/SelfAttention_0/query/kernel",
    "encoder/~/self_attention_3/attention/linear_1:b": "params/Processor/SelfAttentionBlock_3/SelfAttention_0/key/bias",
    "encoder/~/self_attention_3/attention/linear_1:w": "params/Processor/SelfAttentionBlock_3/SelfAttention_0/key/kernel",
    "encoder/~/self_attention_3/attention/linear_2:b": "params/Processor/SelfAttentionBlock_3/SelfAttention_0/value/bias",
    "encoder/~/self_attention_3/attention/linear_2:w": "params/Processor/SelfAttentionBlock_3/SelfAttention_0/value/kernel",
    "encoder/~/self_attention_3/attention/linear_3:b": "params/Processor/SelfAttentionBlock_3/SelfAttention_0/out/bias",
    "encoder/~/self_attention_3/attention/linear_3:w": "params/Processor/SelfAttentionBlock_3/SelfAttention_0/out/kernel",
    "encoder/~/self_attention_3/layer_norm:offset": "params/Processor/SelfAttentionBlock_3/LayerNorm_0/bias",
    "encoder/~/self_attention_3/layer_norm:scale": "params/Processor/SelfAttentionBlock_3/LayerNorm_0/scale",
    "encoder/~/self_attention_3/layer_norm_1:offset": "params/Processor/SelfAttentionBlock_3/LayerNorm_1/bias",
    "encoder/~/self_attention_3/layer_norm_1:scale": "params/Processor/SelfAttentionBlock_3/LayerNorm_1/scale",
    "encoder/~/self_attention_3/mlp/linear:b": "params/Processor/SelfAttentionBlock_3/FeedForward_0/Dense_0/bias",
    "encoder/~/self_attention_3/mlp/linear:w": "params/Processor/SelfAttentionBlock_3/FeedForward_0/Dense_0/kernel",
    "encoder/~/self_attention_3/mlp/linear_1:b": "params/Processor/SelfAttentionBlock_3/FeedForward_0/Dense_1/bias",
    "encoder/~/self_attention_3/mlp/linear_1:w": "params/Processor/SelfAttentionBlock_3/FeedForward_0/Dense_1/kernel",
    "encoder/~/self_attention_4/attention/linear:b": "params/Processor/SelfAttentionBlock_4/SelfAttention_0/query/bias",
    "encoder/~/self_attention_4/attention/linear:w": "params/Processor/SelfAttentionBlock_4/SelfAttention_0/query/kernel",
    "encoder/~/self_attention_4/attention/linear_1:b": "params/Processor/SelfAttentionBlock_4/SelfAttention_0/key/bias",
    "encoder/~/self_attention_4/attention/linear_1:w": "params/Processor/SelfAttentionBlock_4/SelfAttention_0/key/kernel",
    "encoder/~/self_attention_4/attention/linear_2:b": "params/Processor/SelfAttentionBlock_4/SelfAttention_0/value/bias",
    "encoder/~/self_attention_4/attention/linear_2:w": "params/Processor/SelfAttentionBlock_4/SelfAttention_0/value/kernel",
    "encoder/~/self_attention_4/attention/linear_3:b": "params/Processor/SelfAttentionBlock_4/SelfAttention_0/out/bias",
    "encoder/~/self_attention_4/attention/linear_3:w": "params/Processor/SelfAttentionBlock_4/SelfAttention_0/out/kernel",
    "encoder/~/self_attention_4/layer_norm:offset": "params/Processor/SelfAttentionBlock_4/LayerNorm_0/bias",
    "encoder/~/self_attention_4/layer_norm:scale": "params/Processor/SelfAttentionBlock_4/LayerNorm_0/scale",
    "encoder/~/self_attention_4/layer_norm_1:offset": "params/Processor/SelfAttentionBlock_4/LayerNorm_1/bias",
    "encoder/~/self_attention_4/layer_norm_1:scale": "params/Processor/SelfAttentionBlock_4/LayerNorm_1/scale",
    "encoder/~/self_attention_4/mlp/linear:b": "params/Processor/SelfAttentionBlock_4/FeedForward_0/Dense_0/bias",
    "encoder/~/self_attention_4/mlp/linear:w": "params/Processor/SelfAttentionBlock_4/FeedForward_0/Dense_0/kernel",
    "encoder/~/self_attention_4/mlp/linear_1:b": "params/Processor/SelfAttentionBlock_4/FeedForward_0/Dense_1/bias",
    "encoder/~/self_attention_4/mlp/linear_1:w": "params/Processor/SelfAttentionBlock_4/FeedForward_0/Dense_1/kernel",
    "encoder/~/self_attention_5/attention/linear:b": "params/Processor/SelfAttentionBlock_5/SelfAttention_0/query/bias",
    "encoder/~/self_attention_5/attention/linear:w": "params/Processor/SelfAttentionBlock_5/SelfAttention_0/query/kernel",
    "encoder/~/self_attention_5/attention/linear_1:b": "params/Processor/SelfAttentionBlock_5/SelfAttention_0/key/bias",
    "encoder/~/self_attention_5/attention/linear_1:w": "params/Processor/SelfAttentionBlock_5/SelfAttention_0/key/kernel",
    "encoder/~/self_attention_5/attention/linear_2:b": "params/Processor/SelfAttentionBlock_5/SelfAttention_0/value/bias",
    "encoder/~/self_attention_5/attention/linear_2:w": "params/Processor/SelfAttentionBlock_5/SelfAttention_0/value/kernel",
    "encoder/~/self_attention_5/attention/linear_3:b": "params/Processor/SelfAttentionBlock_5/SelfAttention_0/out/bias",
    "encoder/~/self_attention_5/attention/linear_3:w": "params/Processor/SelfAttentionBlock_5/SelfAttention_0/out/kernel",
    "encoder/~/self_attention_5/layer_norm:offset": "params/Processor/SelfAttentionBlock_5/LayerNorm_0/bias",
    "encoder/~/self_attention_5/layer_norm:scale": "params/Processor/SelfAttentionBlock_5/LayerNorm_0/scale",
    "encoder/~/self_attention_5/layer_norm_1:offset": "params/Processor/SelfAttentionBlock_5/LayerNorm_1/bias",
    "encoder/~/self_attention_5/layer_norm_1:scale": "params/Processor/SelfAttentionBlock_5/LayerNorm_1/scale",
    "encoder/~/self_attention_5/mlp/linear:b": "params/Processor/SelfAttentionBlock_5/FeedForward_0/Dense_0/bias",
    "encoder/~/self_attention_5/mlp/linear:w": "params/Processor/SelfAttentionBlock_5/FeedForward_0/Dense_0/kernel",
    "encoder/~/self_attention_5/mlp/linear_1:b": "params/Processor/SelfAttentionBlock_5/FeedForward_0/Dense_1/bias",
    "encoder/~/self_attention_5/mlp/linear_1:w": "params/Processor/SelfAttentionBlock_5/FeedForward_0/Dense_1/kernel",
    "encoder/~/self_attention_6/attention/linear:b": "params/Processor/SelfAttentionBlock_6/SelfAttention_0/query/bias",
    "encoder/~/self_attention_6/attention/linear:w": "params/Processor/SelfAttentionBlock_6/SelfAttention_0/query/kernel",
    "encoder/~/self_attention_6/attention/linear_1:b": "params/Processor/SelfAttentionBlock_6/SelfAttention_0/key/bias",
    "encoder/~/self_attention_6/attention/linear_1:w": "params/Processor/SelfAttentionBlock_6/SelfAttention_0/key/kernel",
    "encoder/~/self_attention_6/attention/linear_2:b": "params/Processor/SelfAttentionBlock_6/SelfAttention_0/value/bias",
    "encoder/~/self_attention_6/attention/linear_2:w": "params/Processor/SelfAttentionBlock_6/SelfAttention_0/value/kernel",
    "encoder/~/self_attention_6/attention/linear_3:b": "params/Processor/SelfAttentionBlock_6/SelfAttention_0/out/bias",
    "encoder/~/self_attention_6/attention/linear_3:w": "params/Processor/SelfAttentionBlock_6/SelfAttention_0/out/kernel",
    "encoder/~/self_attention_6/layer_norm:offset": "params/Processor/SelfAttentionBlock_6/LayerNorm_0/bias",
    "encoder/~/self_attention_6/layer_norm:scale": "params/Processor/SelfAttentionBlock_6/LayerNorm_0/scale",
    "encoder/~/self_attention_6/layer_norm_1:offset": "params/Processor/SelfAttentionBlock_6/LayerNorm_1/bias",
    "encoder/~/self_attention_6/layer_norm_1:scale": "params/Processor/SelfAttentionBlock_6/LayerNorm_1/scale",
    "encoder/~/self_attention_6/mlp/linear:b": "params/Processor/SelfAttentionBlock_6/FeedForward_0/Dense_0/bias",
    "encoder/~/self_attention_6/mlp/linear:w": "params/Processor/SelfAttentionBlock_6/FeedForward_0/Dense_0/kernel",
    "encoder/~/self_attention_6/mlp/linear_1:b": "params/Processor/SelfAttentionBlock_6/FeedForward_0/Dense_1/bias",
    "encoder/~/self_attention_6/mlp/linear_1:w": "params/Processor/SelfAttentionBlock_6/FeedForward_0/Dense_1/kernel",
    "encoder/~/self_attention_7/attention/linear:b": "params/Processor/SelfAttentionBlock_7/SelfAttention_0/query/bias",
    "encoder/~/self_attention_7/attention/linear:w": "params/Processor/SelfAttentionBlock_7/SelfAttention_0/query/kernel",
    "encoder/~/self_attention_7/attention/linear_1:b": "params/Processor/SelfAttentionBlock_7/SelfAttention_0/key/bias",
    "encoder/~/self_attention_7/attention/linear_1:w": "params/Processor/SelfAttentionBlock_7/SelfAttention_0/key/kernel",
    "encoder/~/self_attention_7/attention/linear_2:b": "params/Processor/SelfAttentionBlock_7/SelfAttention_0/value/bias",
    "encoder/~/self_attention_7/attention/linear_2:w": "params/Processor/SelfAttentionBlock_7/SelfAttention_0/value/kernel",
    "encoder/~/self_attention_7/attention/linear_3:b": "params/Processor/SelfAttentionBlock_7/SelfAttention_0/out/bias",
    "encoder/~/self_attention_7/attention/linear_3:w": "params/Processor/SelfAttentionBlock_7/SelfAttention_0/out/kernel",
    "encoder/~/self_attention_7/layer_norm:offset": "params/Processor/SelfAttentionBlock_7/LayerNorm_0/bias",
    "encoder/~/self_attention_7/layer_norm:scale": "params/Processor/SelfAttentionBlock_7/LayerNorm_0/scale",
    "encoder/~/self_attention_7/layer_norm_1:offset": "params/Processor/SelfAttentionBlock_7/LayerNorm_1/bias",
    "encoder/~/self_attention_7/layer_norm_1:scale": "params/Processor/SelfAttentionBlock_7/LayerNorm_1/scale",
    "encoder/~/self_attention_7/mlp/linear:b": "params/Processor/SelfAttentionBlock_7/FeedForward_0/Dense_0/bias",
    "encoder/~/self_attention_7/mlp/linear:w": "params/Processor/SelfAttentionBlock_7/FeedForward_0/Dense_0/kernel",
    "encoder/~/self_attention_7/mlp/linear_1:b": "params/Processor/SelfAttentionBlock_7/FeedForward_0/Dense_1/bias",
    "encoder/~/self_attention_7/mlp/linear_1:w": "params/Processor/SelfAttentionBlock_7/FeedForward_0/Dense_1/kernel",
}

def reshape_attention_kernel(weights, n_heads=8):
    d_inputs, d_hidden = weights.shape
    d_head_hidden = d_hidden // n_heads
    return jnp.reshape(weights, (d_inputs, n_heads, d_head_hidden))

def reshape_attention_bias(weights, n_heads=8):
    d_hidden, = weights.shape
    d_head_hidden = d_hidden // n_heads
    return jnp.reshape(weights, (n_heads, d_head_hidden))

def variables(path):
    with open(path, "rb") as f:
        original_params = pickle.loads(f.read())

    flat_initial_variables = {}
    for parent_key, parent_value in hk.data_structures.to_immutable_dict(original_params).items():
        for key, weights in hk.data_structures.to_immutable_dict(parent_value).items():
            full_key = f"{parent_key}:{key}"
            mapped_full_key = tuple(PARAM_MAP[full_key].split("/"))

            if (("query" in mapped_full_key)
                or ("key" in mapped_full_key)
                or ("value" in mapped_full_key)):

                n_heads = 8
                if "Encoder" in mapped_full_key:
                    n_heads = 1
                if "Decoder" in mapped_full_key:
                    n_heads = 1

                if "kernel" in mapped_full_key:
                    flat_initial_variables[mapped_full_key] = reshape_attention_kernel(weights, n_heads=n_heads)

                if "bias" in mapped_full_key:
                    flat_initial_variables[mapped_full_key] = reshape_attention_bias(weights, n_heads=n_heads)

            else:
                flat_initial_variables[mapped_full_key] = weights

    return freeze(unflatten_dict(flat_initial_variables))
