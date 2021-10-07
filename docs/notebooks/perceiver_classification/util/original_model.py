import pickle
from jax import numpy as jnp
import haiku as hk
from flax.core import freeze
from flax.traverse_util import unflatten_dict

PARAM_MAP = {
    "classification_decoder/~/basic_decoder/cross_attention/attention/linear:b": "params/Decoder/CrossAttentionBlock_0/Attention_0/query/bias",
    "classification_decoder/~/basic_decoder/cross_attention/attention/linear:w": "params/Decoder/CrossAttentionBlock_0/Attention_0/query/kernel",
    "classification_decoder/~/basic_decoder/cross_attention/attention/linear_1:b": "params/Decoder/CrossAttentionBlock_0/Attention_0/key/bias",
    "classification_decoder/~/basic_decoder/cross_attention/attention/linear_1:w": "params/Decoder/CrossAttentionBlock_0/Attention_0/key/kernel",
    "classification_decoder/~/basic_decoder/cross_attention/attention/linear_2:b": "params/Decoder/CrossAttentionBlock_0/Attention_0/value/bias",
    "classification_decoder/~/basic_decoder/cross_attention/attention/linear_2:w": "params/Decoder/CrossAttentionBlock_0/Attention_0/value/kernel",
    "classification_decoder/~/basic_decoder/cross_attention/attention/linear_3:b": "params/Decoder/CrossAttentionBlock_0/Attention_0/out/bias",
    "classification_decoder/~/basic_decoder/cross_attention/attention/linear_3:w": "params/Decoder/CrossAttentionBlock_0/Attention_0/out/kernel",
    "classification_decoder/~/basic_decoder/cross_attention/mlp/linear:b": "params/Decoder/CrossAttentionBlock_0/FeedForward_0/Dense_0/bias",
    "classification_decoder/~/basic_decoder/cross_attention/mlp/linear:w": "params/Decoder/CrossAttentionBlock_0/FeedForward_0/Dense_0/kernel",
    "classification_decoder/~/basic_decoder/cross_attention/mlp/linear_1:b": "params/Decoder/CrossAttentionBlock_0/FeedForward_0/Dense_1/bias",
    "classification_decoder/~/basic_decoder/cross_attention/mlp/linear_1:w": "params/Decoder/CrossAttentionBlock_0/FeedForward_0/Dense_1/kernel",
    "classification_decoder/~/basic_decoder/cross_attention/layer_norm:offset": "params/Decoder/CrossAttentionBlock_0/LayerNorm_0/bias",
    "classification_decoder/~/basic_decoder/cross_attention/layer_norm:scale": "params/Decoder/CrossAttentionBlock_0/LayerNorm_0/scale",
    "classification_decoder/~/basic_decoder/cross_attention/layer_norm_1:offset": "params/Decoder/CrossAttentionBlock_0/LayerNorm_1/bias",
    "classification_decoder/~/basic_decoder/cross_attention/layer_norm_1:scale": "params/Decoder/CrossAttentionBlock_0/LayerNorm_1/scale",
    "classification_decoder/~/basic_decoder/cross_attention/layer_norm_2:offset": "params/Decoder/CrossAttentionBlock_0/LayerNorm_2/bias",
    "classification_decoder/~/basic_decoder/cross_attention/layer_norm_2:scale": "params/Decoder/CrossAttentionBlock_0/LayerNorm_2/scale",
    "classification_decoder/~/basic_decoder/output:b": "params/OutputDecoding/Decoding_0/Dense_0/bias",
    "classification_decoder/~/basic_decoder/output:w": "params/OutputDecoding/Decoding_0/Dense_0/kernel",
    "classification_decoder/~/basic_decoder/~/trainable_position_encoding:pos_embs": "params/DecoderQueryEncoding/TrainablePositionalEncoding_0/positional_encoding",
    "perceiver_encoder/~/cross_attention/attention/linear:b": "params/Encoder/CrossAttentionBlock_0/Attention_0/query/bias",
    "perceiver_encoder/~/cross_attention/attention/linear:w": "params/Encoder/CrossAttentionBlock_0/Attention_0/query/kernel",
    "perceiver_encoder/~/cross_attention/attention/linear_1:b": "params/Encoder/CrossAttentionBlock_0/Attention_0/key/bias",
    "perceiver_encoder/~/cross_attention/attention/linear_1:w": "params/Encoder/CrossAttentionBlock_0/Attention_0/key/kernel",
    "perceiver_encoder/~/cross_attention/attention/linear_2:b": "params/Encoder/CrossAttentionBlock_0/Attention_0/value/bias",
    "perceiver_encoder/~/cross_attention/attention/linear_2:w": "params/Encoder/CrossAttentionBlock_0/Attention_0/value/kernel",
    "perceiver_encoder/~/cross_attention/attention/linear_3:b": "params/Encoder/CrossAttentionBlock_0/Attention_0/out/bias",
    "perceiver_encoder/~/cross_attention/attention/linear_3:w": "params/Encoder/CrossAttentionBlock_0/Attention_0/out/kernel",
    "perceiver_encoder/~/cross_attention/mlp/linear:b": "params/Encoder/CrossAttentionBlock_0/FeedForward_0/Dense_0/bias",
    "perceiver_encoder/~/cross_attention/mlp/linear:w": "params/Encoder/CrossAttentionBlock_0/FeedForward_0/Dense_0/kernel",
    "perceiver_encoder/~/cross_attention/mlp/linear_1:b": "params/Encoder/CrossAttentionBlock_0/FeedForward_0/Dense_1/bias",
    "perceiver_encoder/~/cross_attention/mlp/linear_1:w": "params/Encoder/CrossAttentionBlock_0/FeedForward_0/Dense_1/kernel",
    "perceiver_encoder/~/cross_attention/layer_norm:offset": "params/Encoder/CrossAttentionBlock_0/LayerNorm_0/bias",
    "perceiver_encoder/~/cross_attention/layer_norm:scale": "params/Encoder/CrossAttentionBlock_0/LayerNorm_0/scale",
    "perceiver_encoder/~/cross_attention/layer_norm_1:offset": "params/Encoder/CrossAttentionBlock_0/LayerNorm_1/bias",
    "perceiver_encoder/~/cross_attention/layer_norm_1:scale": "params/Encoder/CrossAttentionBlock_0/LayerNorm_1/scale",
    "perceiver_encoder/~/cross_attention/layer_norm_2:offset": "params/Encoder/CrossAttentionBlock_0/LayerNorm_2/bias",
    "perceiver_encoder/~/cross_attention/layer_norm_2:scale": "params/Encoder/CrossAttentionBlock_0/LayerNorm_2/scale",
    "perceiver_encoder/~/self_attention/attention/linear:b": "params/Processor/SelfAttentionBlock_0/SelfAttention_0/query/bias",
    "perceiver_encoder/~/self_attention/attention/linear:w": "params/Processor/SelfAttentionBlock_0/SelfAttention_0/query/kernel",
    "perceiver_encoder/~/self_attention/attention/linear_1:b": "params/Processor/SelfAttentionBlock_0/SelfAttention_0/key/bias",
    "perceiver_encoder/~/self_attention/attention/linear_1:w": "params/Processor/SelfAttentionBlock_0/SelfAttention_0/key/kernel",
    "perceiver_encoder/~/self_attention/attention/linear_2:b": "params/Processor/SelfAttentionBlock_0/SelfAttention_0/value/bias",
    "perceiver_encoder/~/self_attention/attention/linear_2:w": "params/Processor/SelfAttentionBlock_0/SelfAttention_0/value/kernel",
    "perceiver_encoder/~/self_attention/attention/linear_3:b": "params/Processor/SelfAttentionBlock_0/SelfAttention_0/out/bias",
    "perceiver_encoder/~/self_attention/attention/linear_3:w": "params/Processor/SelfAttentionBlock_0/SelfAttention_0/out/kernel",
    "perceiver_encoder/~/self_attention/mlp/linear:b": "params/Processor/SelfAttentionBlock_0/FeedForward_0/Dense_0/bias",
    "perceiver_encoder/~/self_attention/mlp/linear:w": "params/Processor/SelfAttentionBlock_0/FeedForward_0/Dense_0/kernel",
    "perceiver_encoder/~/self_attention/mlp/linear_1:b": "params/Processor/SelfAttentionBlock_0/FeedForward_0/Dense_1/bias",
    "perceiver_encoder/~/self_attention/mlp/linear_1:w": "params/Processor/SelfAttentionBlock_0/FeedForward_0/Dense_1/kernel",
    "perceiver_encoder/~/self_attention/layer_norm:offset": "params/Processor/SelfAttentionBlock_0/LayerNorm_0/bias",
    "perceiver_encoder/~/self_attention/layer_norm:scale": "params/Processor/SelfAttentionBlock_0/LayerNorm_0/scale",
    "perceiver_encoder/~/self_attention/layer_norm_1:offset": "params/Processor/SelfAttentionBlock_0/LayerNorm_1/bias",
    "perceiver_encoder/~/self_attention/layer_norm_1:scale": "params/Processor/SelfAttentionBlock_0/LayerNorm_1/scale",
    "perceiver_encoder/~/self_attention_1/attention/linear:b": "params/Processor/SelfAttentionBlock_1/SelfAttention_0/query/bias",
    "perceiver_encoder/~/self_attention_1/attention/linear:w": "params/Processor/SelfAttentionBlock_1/SelfAttention_0/query/kernel",
    "perceiver_encoder/~/self_attention_1/attention/linear_1:b": "params/Processor/SelfAttentionBlock_1/SelfAttention_0/key/bias",
    "perceiver_encoder/~/self_attention_1/attention/linear_1:w": "params/Processor/SelfAttentionBlock_1/SelfAttention_0/key/kernel",
    "perceiver_encoder/~/self_attention_1/attention/linear_2:b": "params/Processor/SelfAttentionBlock_1/SelfAttention_0/value/bias",
    "perceiver_encoder/~/self_attention_1/attention/linear_2:w": "params/Processor/SelfAttentionBlock_1/SelfAttention_0/value/kernel",
    "perceiver_encoder/~/self_attention_1/attention/linear_3:b": "params/Processor/SelfAttentionBlock_1/SelfAttention_0/out/bias",
    "perceiver_encoder/~/self_attention_1/attention/linear_3:w": "params/Processor/SelfAttentionBlock_1/SelfAttention_0/out/kernel",
    "perceiver_encoder/~/self_attention_1/mlp/linear:b": "params/Processor/SelfAttentionBlock_1/FeedForward_0/Dense_0/bias",
    "perceiver_encoder/~/self_attention_1/mlp/linear:w": "params/Processor/SelfAttentionBlock_1/FeedForward_0/Dense_0/kernel",
    "perceiver_encoder/~/self_attention_1/mlp/linear_1:b": "params/Processor/SelfAttentionBlock_1/FeedForward_0/Dense_1/bias",
    "perceiver_encoder/~/self_attention_1/mlp/linear_1:w": "params/Processor/SelfAttentionBlock_1/FeedForward_0/Dense_1/kernel",
    "perceiver_encoder/~/self_attention_1/layer_norm:offset": "params/Processor/SelfAttentionBlock_1/LayerNorm_0/bias",
    "perceiver_encoder/~/self_attention_1/layer_norm:scale": "params/Processor/SelfAttentionBlock_1/LayerNorm_0/scale",
    "perceiver_encoder/~/self_attention_1/layer_norm_1:offset": "params/Processor/SelfAttentionBlock_1/LayerNorm_1/bias",
    "perceiver_encoder/~/self_attention_1/layer_norm_1:scale": "params/Processor/SelfAttentionBlock_1/LayerNorm_1/scale",
    "perceiver_encoder/~/self_attention_2/attention/linear:b": "params/Processor/SelfAttentionBlock_2/SelfAttention_0/query/bias",
    "perceiver_encoder/~/self_attention_2/attention/linear:w": "params/Processor/SelfAttentionBlock_2/SelfAttention_0/query/kernel",
    "perceiver_encoder/~/self_attention_2/attention/linear_1:b": "params/Processor/SelfAttentionBlock_2/SelfAttention_0/key/bias",
    "perceiver_encoder/~/self_attention_2/attention/linear_1:w": "params/Processor/SelfAttentionBlock_2/SelfAttention_0/key/kernel",
    "perceiver_encoder/~/self_attention_2/attention/linear_2:b": "params/Processor/SelfAttentionBlock_2/SelfAttention_0/value/bias",
    "perceiver_encoder/~/self_attention_2/attention/linear_2:w": "params/Processor/SelfAttentionBlock_2/SelfAttention_0/value/kernel",
    "perceiver_encoder/~/self_attention_2/attention/linear_3:b": "params/Processor/SelfAttentionBlock_2/SelfAttention_0/out/bias",
    "perceiver_encoder/~/self_attention_2/attention/linear_3:w": "params/Processor/SelfAttentionBlock_2/SelfAttention_0/out/kernel",
    "perceiver_encoder/~/self_attention_2/mlp/linear:b": "params/Processor/SelfAttentionBlock_2/FeedForward_0/Dense_0/bias",
    "perceiver_encoder/~/self_attention_2/mlp/linear:w": "params/Processor/SelfAttentionBlock_2/FeedForward_0/Dense_0/kernel",
    "perceiver_encoder/~/self_attention_2/mlp/linear_1:b": "params/Processor/SelfAttentionBlock_2/FeedForward_0/Dense_1/bias",
    "perceiver_encoder/~/self_attention_2/mlp/linear_1:w": "params/Processor/SelfAttentionBlock_2/FeedForward_0/Dense_1/kernel",
    "perceiver_encoder/~/self_attention_2/layer_norm:offset": "params/Processor/SelfAttentionBlock_2/LayerNorm_0/bias",
    "perceiver_encoder/~/self_attention_2/layer_norm:scale": "params/Processor/SelfAttentionBlock_2/LayerNorm_0/scale",
    "perceiver_encoder/~/self_attention_2/layer_norm_1:offset": "params/Processor/SelfAttentionBlock_2/LayerNorm_1/bias",
    "perceiver_encoder/~/self_attention_2/layer_norm_1:scale": "params/Processor/SelfAttentionBlock_2/LayerNorm_1/scale",
    "perceiver_encoder/~/self_attention_3/attention/linear:b": "params/Processor/SelfAttentionBlock_3/SelfAttention_0/query/bias",
    "perceiver_encoder/~/self_attention_3/attention/linear:w": "params/Processor/SelfAttentionBlock_3/SelfAttention_0/query/kernel",
    "perceiver_encoder/~/self_attention_3/attention/linear_1:b": "params/Processor/SelfAttentionBlock_3/SelfAttention_0/key/bias",
    "perceiver_encoder/~/self_attention_3/attention/linear_1:w": "params/Processor/SelfAttentionBlock_3/SelfAttention_0/key/kernel",
    "perceiver_encoder/~/self_attention_3/attention/linear_2:b": "params/Processor/SelfAttentionBlock_3/SelfAttention_0/value/bias",
    "perceiver_encoder/~/self_attention_3/attention/linear_2:w": "params/Processor/SelfAttentionBlock_3/SelfAttention_0/value/kernel",
    "perceiver_encoder/~/self_attention_3/attention/linear_3:b": "params/Processor/SelfAttentionBlock_3/SelfAttention_0/out/bias",
    "perceiver_encoder/~/self_attention_3/attention/linear_3:w": "params/Processor/SelfAttentionBlock_3/SelfAttention_0/out/kernel",
    "perceiver_encoder/~/self_attention_3/mlp/linear:b": "params/Processor/SelfAttentionBlock_3/FeedForward_0/Dense_0/bias",
    "perceiver_encoder/~/self_attention_3/mlp/linear:w": "params/Processor/SelfAttentionBlock_3/FeedForward_0/Dense_0/kernel",
    "perceiver_encoder/~/self_attention_3/mlp/linear_1:b": "params/Processor/SelfAttentionBlock_3/FeedForward_0/Dense_1/bias",
    "perceiver_encoder/~/self_attention_3/mlp/linear_1:w": "params/Processor/SelfAttentionBlock_3/FeedForward_0/Dense_1/kernel",
    "perceiver_encoder/~/self_attention_3/layer_norm:offset": "params/Processor/SelfAttentionBlock_3/LayerNorm_0/bias",
    "perceiver_encoder/~/self_attention_3/layer_norm:scale": "params/Processor/SelfAttentionBlock_3/LayerNorm_0/scale",
    "perceiver_encoder/~/self_attention_3/layer_norm_1:offset": "params/Processor/SelfAttentionBlock_3/LayerNorm_1/bias",
    "perceiver_encoder/~/self_attention_3/layer_norm_1:scale": "params/Processor/SelfAttentionBlock_3/LayerNorm_1/scale",
    "perceiver_encoder/~/self_attention_4/attention/linear:b": "params/Processor/SelfAttentionBlock_4/SelfAttention_0/query/bias",
    "perceiver_encoder/~/self_attention_4/attention/linear:w": "params/Processor/SelfAttentionBlock_4/SelfAttention_0/query/kernel",
    "perceiver_encoder/~/self_attention_4/attention/linear_1:b": "params/Processor/SelfAttentionBlock_4/SelfAttention_0/key/bias",
    "perceiver_encoder/~/self_attention_4/attention/linear_1:w": "params/Processor/SelfAttentionBlock_4/SelfAttention_0/key/kernel",
    "perceiver_encoder/~/self_attention_4/attention/linear_2:b": "params/Processor/SelfAttentionBlock_4/SelfAttention_0/value/bias",
    "perceiver_encoder/~/self_attention_4/attention/linear_2:w": "params/Processor/SelfAttentionBlock_4/SelfAttention_0/value/kernel",
    "perceiver_encoder/~/self_attention_4/attention/linear_3:b": "params/Processor/SelfAttentionBlock_4/SelfAttention_0/out/bias",
    "perceiver_encoder/~/self_attention_4/attention/linear_3:w": "params/Processor/SelfAttentionBlock_4/SelfAttention_0/out/kernel",
    "perceiver_encoder/~/self_attention_4/mlp/linear:b": "params/Processor/SelfAttentionBlock_4/FeedForward_0/Dense_0/bias",
    "perceiver_encoder/~/self_attention_4/mlp/linear:w": "params/Processor/SelfAttentionBlock_4/FeedForward_0/Dense_0/kernel",
    "perceiver_encoder/~/self_attention_4/mlp/linear_1:b": "params/Processor/SelfAttentionBlock_4/FeedForward_0/Dense_1/bias",
    "perceiver_encoder/~/self_attention_4/mlp/linear_1:w": "params/Processor/SelfAttentionBlock_4/FeedForward_0/Dense_1/kernel",
    "perceiver_encoder/~/self_attention_4/layer_norm:offset": "params/Processor/SelfAttentionBlock_4/LayerNorm_0/bias",
    "perceiver_encoder/~/self_attention_4/layer_norm:scale": "params/Processor/SelfAttentionBlock_4/LayerNorm_0/scale",
    "perceiver_encoder/~/self_attention_4/layer_norm_1:offset": "params/Processor/SelfAttentionBlock_4/LayerNorm_1/bias",
    "perceiver_encoder/~/self_attention_4/layer_norm_1:scale": "params/Processor/SelfAttentionBlock_4/LayerNorm_1/scale",
    "perceiver_encoder/~/self_attention_5/attention/linear:b": "params/Processor/SelfAttentionBlock_5/SelfAttention_0/query/bias",
    "perceiver_encoder/~/self_attention_5/attention/linear:w": "params/Processor/SelfAttentionBlock_5/SelfAttention_0/query/kernel",
    "perceiver_encoder/~/self_attention_5/attention/linear_1:b": "params/Processor/SelfAttentionBlock_5/SelfAttention_0/key/bias",
    "perceiver_encoder/~/self_attention_5/attention/linear_1:w": "params/Processor/SelfAttentionBlock_5/SelfAttention_0/key/kernel",
    "perceiver_encoder/~/self_attention_5/attention/linear_2:b": "params/Processor/SelfAttentionBlock_5/SelfAttention_0/value/bias",
    "perceiver_encoder/~/self_attention_5/attention/linear_2:w": "params/Processor/SelfAttentionBlock_5/SelfAttention_0/value/kernel",
    "perceiver_encoder/~/self_attention_5/attention/linear_3:b": "params/Processor/SelfAttentionBlock_5/SelfAttention_0/out/bias",
    "perceiver_encoder/~/self_attention_5/attention/linear_3:w": "params/Processor/SelfAttentionBlock_5/SelfAttention_0/out/kernel",
    "perceiver_encoder/~/self_attention_5/mlp/linear:b": "params/Processor/SelfAttentionBlock_5/FeedForward_0/Dense_0/bias",
    "perceiver_encoder/~/self_attention_5/mlp/linear:w": "params/Processor/SelfAttentionBlock_5/FeedForward_0/Dense_0/kernel",
    "perceiver_encoder/~/self_attention_5/mlp/linear_1:b": "params/Processor/SelfAttentionBlock_5/FeedForward_0/Dense_1/bias",
    "perceiver_encoder/~/self_attention_5/mlp/linear_1:w": "params/Processor/SelfAttentionBlock_5/FeedForward_0/Dense_1/kernel",
    "perceiver_encoder/~/self_attention_5/layer_norm:offset": "params/Processor/SelfAttentionBlock_5/LayerNorm_0/bias",
    "perceiver_encoder/~/self_attention_5/layer_norm:scale": "params/Processor/SelfAttentionBlock_5/LayerNorm_0/scale",
    "perceiver_encoder/~/self_attention_5/layer_norm_1:offset": "params/Processor/SelfAttentionBlock_5/LayerNorm_1/bias",
    "perceiver_encoder/~/self_attention_5/layer_norm_1:scale": "params/Processor/SelfAttentionBlock_5/LayerNorm_1/scale",
    "perceiver_encoder/~/trainable_position_encoding:pos_embs": "params/EncodedQueryEncoding/TrainablePositionalEncoding_0/positional_encoding",
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
        original_variables = pickle.loads(f.read())

    original_params = original_variables["params"]
    assert len(original_variables["state"]) == 0

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
