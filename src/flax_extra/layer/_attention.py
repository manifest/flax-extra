"""Attention layers."""

from typing import Any, Callable, Sequence, Optional, Type, Union
from functools import partial
import jax
import jax.numpy as jnp
from jax.core import NamedShape
from jax.random import KeyArray
from flax import linen as nn

Array = jnp.ndarray
Shape = Optional[Union[Sequence[int], NamedShape]]
Dtype = Any
Precision = Any
InitFn = Callable[[KeyArray, Shape, Dtype], Array]

# pylint: disable=too-many-arguments, too-many-locals
def attend(
    query: Array,
    key: Array,
    value: Array,
    mask: Optional[Array] = None,
    dropout_rngkey: Optional[KeyArray] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    precision: Optional[Precision] = None,
) -> Array:
    """Computes multi-head attention using a query, key and value.

    Args:
        query: a query vector.
        key: a key vector.
        value: a value vector.
        mask: a boolean matrix indicating which attention values are valid.
        dropout_rngkey: random number generator key for dropout.
        dropout_rate: probababilistic rate for attention dropout.
        deterministic: whether to perform deterministically or not.
        precision: numerical precision of the computation.
            See :attr:`jax.lax.Precision` for details.
    Returns:
        an attention vector with shape.
    """
    batch_size, q_seqlen, n_heads, d_head_qk = query.shape
    _, kv_seqlen, _, d_head_v = value.shape
    d_v = n_heads * d_head_v

    query_shape = (batch_size, q_seqlen, n_heads, d_head_qk)
    assert query.shape == query_shape, (
        f"A shape of the query {query.shape}"
        f"doesn't match an expected value {query_shape}."
    )
    key_shape = (batch_size, kv_seqlen, n_heads, d_head_qk)
    assert key.shape == key_shape, (
        f"A shape of the key {query.shape}"
        f"doesn't match an expected value {key_shape}."
    )
    value_shape = (batch_size, kv_seqlen, n_heads, d_head_v)
    assert value.shape == value_shape, (
        f"A shape of the value {value.shape} "
        f"doesn't match an expected value {value_shape}."
    )

    # Scaled dot product attention weights.
    weights = jnp.einsum("bthd,bThd->bhtT", query, key, precision=precision)
    scale = 1.0 / jnp.sqrt(d_head_qk)
    weights *= scale

    if mask is not None:
        mask_shape = (batch_size, q_seqlen, kv_seqlen)
        assert mask.shape == mask_shape, (
            f"A shape of the mask {mask.shape} "
            f"doesn't match an expected value {mask_shape}."
        )

        # Use large_k instead of np.NINF because np.NINF breaks for causal-masked
        # left-padded sampling.
        large_k = jnp.array(
            1e4 if weights.dtype == jnp.float16 else 1e30,
            dtype=weights.dtype,
        )
        weights = jnp.where(mask[:, None, :, :], weights, -large_k)

    # Normalized probabilities.
    normalized = jax.nn.softmax(weights)
    if not deterministic and dropout_rate > 0.0:
        keep_probs = 1.0 - dropout_rate
        normalized = jax.random.bernoulli(dropout_rngkey, keep_probs, weights.shape)

    # Attention vectors.
    attention = jnp.einsum(
        "bhtT,bThd->bthd",
        normalized,
        value,
        precision=precision,
    )
    attention = jnp.reshape(attention, (batch_size, q_seqlen, d_v))

    if mask is not None:
        # If all attended tokens are masked, or for masked tokens
        # some rows of logits gets completely masked, in which case the softmax
        # gives a uniform row and we obtain non-zero outputs where it should be
        # zero. We force zeros.
        wiped_attention = jnp.all(mask == 0, axis=2, keepdims=True)
        assert wiped_attention.shape == (batch_size, q_seqlen, 1,), (
            f"A shape of the wiped attention matrix {wiped_attention.shape} "
            f"doesn't match expected value {(batch_size, q_seqlen, 1)}."
        )

        attention = jnp.where(wiped_attention, jnp.zeros_like(attention), attention)

    attention_shape = (batch_size, q_seqlen, d_v)
    assert attention.shape == attention_shape, (
        f"A shape of the attention vector {attention.shape} "
        f"doesn't match expected value {attention_shape}."
    )

    return attention  # type: ignore


def make_cross_attention_mask(q_mask: Array, kv_mask: Array) -> Array:
    """Creates a cross-attention mask matrix.

    Args:
        q_mask: a padding mask indicating at which positions values
            of the query are valid.
        kv_mask: a padding mask indicating at which positions values
            of the key are valid.

    Returns:
        a boolean matrix indicating which attention values are valid.
    """
    batch_size, q_seqlen = q_mask.shape
    _, k_seqlen = kv_mask.shape
    mask = jax.vmap(jnp.outer)(q_mask, kv_mask)
    assert mask.shape == (batch_size, q_seqlen, k_seqlen)
    return mask  # type: ignore


LikeFn = Callable[[Array, Array, Array, Array], Any]


def _like_value(*args: Any, index: int) -> Any:
    """Returns a value at a particular index."""
    arg = args[index]
    if arg is None:
        raise ValueError(
            "Cannot infer a like-value, becase "
            "the target value has not been initialized yet."
        )
    return arg


def _like_shape(*args: Any, index: int) -> Any:
    """Returns the shape of a value at a particular index."""
    return _like_value(*args, index=index).shape[-1]


def _init_like(*args: Any, value: Union[Any, LikeFn], kind: Type[Any] = int) -> Any:
    """Returns the value as is if possible. Otherwise call the like-function."""
    if isinstance(value, kind):
        return value
    return value(*args)


d_like_input_q = partial(_like_shape, index=0)
d_like_input_kv = partial(_like_shape, index=1)
d_like_qk = partial(_like_value, index=2)
d_like_v = partial(_like_value, index=3)


class Attention(nn.Module):
    """The multi-head cross-attention.

    Queries coming from one sequence and keys coming from another.

    May be used with
    - a padding mask to prevent attending to the padding of these sequences.
    """

    n_heads: int = 8
    """a number of attention heads."""

    d_qk: Union[int, LikeFn] = d_like_input_q
    """a dimension of the query array (the key has the same dimension).
        Defaults to the same dimension as the dimension of query's inputs."""

    d_v: Union[int, LikeFn] = d_like_qk
    """a dimension of the value array.
        Defaults to the same dimension as the key's dimesnion."""

    d_output: Union[int, LikeFn] = d_like_v
    """an output dimension.
        Defaults to the same dimension as the values's dimesnion."""

    kernel_init: InitFn = nn.initializers.lecun_normal()
    """weights initializer."""

    bias_init: InitFn = nn.initializers.zeros
    """a bias initializer."""

    use_bias: bool = True
    """wether to use bias."""

    dropout_rate: float = 0.0
    """probababilistic rate for attention dropout."""

    deterministic: bool = True
    """whether to perform deterministically or not."""

    precision: Optional[Precision] = None
    """numerical precision of the computation.
        See :attr:`jax.lax.Precision` for details."""

    @nn.compact
    # pylint: disable=arguments-differ
    def __call__(  # type: ignore[override]
        self,
        inputs_q: Array,
        inputs_kv: Array,
        mask: Optional[Array],
    ) -> Array:
        """Computes an attention array.

        Args:
            inputs_q: inputs for the query.
            inputs_kv: inputs for key and value.
            mask: a padding or/and a look-ahead mask indicating which attention
                values are valid.
        Returns:
            an attention vector with shape.
        """
        d_qk = _init_like(inputs_q, inputs_kv, None, None, value=self.d_qk)
        d_v = _init_like(inputs_q, inputs_kv, d_qk, None, value=self.d_v)
        d_output = _init_like(inputs_q, inputs_kv, d_qk, d_v, value=self.d_output)

        if d_qk % self.n_heads != 0:
            raise ValueError(
                f"Dimensions of the query and key {d_qk} must be divisible by "
                f"the number of heads {self.n_heads}."
            )
        d_head_qk = d_qk // self.n_heads

        if d_v % self.n_heads != 0:
            raise ValueError(
                f"The dimension of the value {d_v} must be divisible by "
                f"the number of heads {self.n_heads}."
            )
        d_head_v = d_v // self.n_heads

        # Project input query, key, and value to desired dimensions.
        dense = partial(
            nn.DenseGeneral,
            axis=-1,
            kernel_init=self.kernel_init,  # type:ignore
            bias_init=self.bias_init,  # type:ignore
            use_bias=self.use_bias,
            precision=self.precision,
        )
        query = dense(features=(self.n_heads, d_head_qk), name="query")(inputs_q)
        key = dense(features=(self.n_heads, d_head_qk), name="key")(inputs_kv)
        value = dense(features=(self.n_heads, d_head_v), name="value")(inputs_kv)

        dropout_rngkey = None
        if not self.deterministic and self.dropout_rate > 0.0:
            dropout_rngkey = self.make_rng("dropout")

        attention = attend(
            query,
            key,
            value,
            mask=mask,
            dropout_rngkey=dropout_rngkey,
            dropout_rate=self.dropout_rate,
            deterministic=self.deterministic,
        )

        # Project attention vectors to desired output dimension.
        return nn.Dense(  # type:ignore
            features=d_output,
            kernel_init=self.kernel_init,  # type:ignore
            bias_init=self.bias_init,  # type:ignore
            use_bias=self.use_bias,
            name="out",
        )(attention)


QKVAttention = Attention  # d_qk=d_like_input_q, d_output=d_like_v
"""A variation of multi-head cross-attention where key's dimension
defaults to the dimension of query's inputs and output dimension defaults
to the value's dimension."""

KVQAttention = partial(Attention, d_qk=d_like_input_kv, d_output=d_like_input_q)
"""A variation of  multi-head cross-attention where query's dimension
defaults to the dimension of key's inputs and output dimension defaults to the
dimension of query's inputs."""


class SelfAttention(Attention):
    """The multi-head self-attention.

    Queries and keys coming from the same sequnce.

    May be used with
    - a padding mask to prevent attending to the padding of the sequence.
    - a look-ahead mask to prevent a query at the given position attending to the
    following keys (e.g. greater then the current positions).
    """

    @nn.compact
    def __call__(  # type: ignore[override]
        self,
        inputs_q: Array,
        mask: Optional[Array],
    ) -> Array:
        """Computes an attention array.

        Args:
            inputs_q: inputs for query, key, and value.
            mask: a padding or/and a look-ahead mask indicating which attention
                values are valid.
        Returns:
            an attention vector with shape.
        """
        return super().__call__(
            inputs_q=inputs_q,
            inputs_kv=inputs_q,
            mask=mask,
        )
