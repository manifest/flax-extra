r"""Attention layers."""

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
    r"""Computes an attention vector for each position (i.e. time step)
    of the query sequence to positions of the key sequence.

    .. math::

        \begin{aligned}
            & \textrm{attend}( \\
            & \quad q \in \sR^{\nBatchSize \times \nSeqLen_{q} \times n_{heads} \times d_{qk}} \\
            & \quad k \in \sR^{\nBatchSize \times \nSeqLen_{kv} \times n_{heads} \times d_{qk}} \\
            & \quad v \in \sR^{\nBatchSize \times \nSeqLen_{kv} \times n_{heads} \times d_{v}} \\
            & \quad mask \in \sR^{\nBatchSize \times \nSeqLen_{q} \times \nSeqLen_{kv}} \\
            & ) \\
            & \rightarrow \sR^{\nBatchSize \times \nSeqLen_{q} \times d_{v}}
        \end{aligned}

    Args:
        query: a query vector.
        key: a key vector.
        value: a value vector.
        mask: a mask tensor with boolean values indicating whether
            a particular query attends to a particular key.
        dropout_rngkey: random number generator key for dropout.
        dropout_rate: probababilistic rate for attention dropout.
        deterministic: whether to perform deterministically or not.
        precision: numerical precision of the computation.
            See :attr:`jax.lax.Precision` for details.

    Returns:
        attention vectors.
    """
    batch_size, seqlen_q, n_heads, d_head_qk = query.shape
    _, seqlen_kv, _, d_head_v = value.shape
    d_v = n_heads * d_head_v

    query_shape = (batch_size, seqlen_q, n_heads, d_head_qk)
    assert query.shape == query_shape, (
        f"A shape of the query {query.shape}"
        f"doesn't match an expected value {query_shape}."
    )
    key_shape = (batch_size, seqlen_kv, n_heads, d_head_qk)
    assert key.shape == key_shape, (
        f"A shape of the key {query.shape}"
        f"doesn't match an expected value {key_shape}."
    )
    value_shape = (batch_size, seqlen_kv, n_heads, d_head_v)
    assert value.shape == value_shape, (
        f"A shape of the value {value.shape} "
        f"doesn't match an expected value {value_shape}."
    )

    # Scaled dot product attention weights.
    weights = jnp.einsum("bthd,bThd->bhtT", query, key, precision=precision)
    scale = 1.0 / jnp.sqrt(d_head_qk)
    weights *= scale

    if mask is not None:
        mask_shape = (batch_size, seqlen_q, seqlen_kv)
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
        keep_prob = 1.0 - dropout_rate
        dropout_mask = jax.random.bernoulli(dropout_rngkey, keep_prob, weights.shape)
        normalized = jax.lax.select(
            dropout_mask, normalized / keep_prob, jnp.zeros_like(normalized)
        )

    # Attention vectors.
    attention = jnp.einsum(
        "bhtT,bThd->bthd",
        normalized,
        value,
        precision=precision,
    )
    attention = jnp.reshape(attention, (batch_size, seqlen_q, d_v))

    if mask is not None:
        # If all attended tokens are masked, or for masked tokens
        # some rows of logits gets completely masked, in which case the softmax
        # gives a uniform row and we obtain non-zero outputs where it should be
        # zero. We force zeros.
        wiped_attention = jnp.all(mask == 0, axis=2, keepdims=True)
        assert wiped_attention.shape == (batch_size, seqlen_q, 1,), (
            f"A shape of the wiped attention matrix {wiped_attention.shape} "
            f"doesn't match expected value {(batch_size, seqlen_q, 1)}."
        )

        attention = jnp.where(wiped_attention, jnp.zeros_like(attention), attention)

    attention_shape = (batch_size, seqlen_q, d_v)
    assert attention.shape == attention_shape, (
        f"A shape of the attention vector {attention.shape} "
        f"doesn't match expected value {attention_shape}."
    )

    return attention  # type: ignore


def cross_attention_mask(mask_q: Array, mask_kv: Array) -> Array:
    r"""Creates a cross-attention mask tensor with boolean values indicating
    whether a particular query attends to a particular key.

    .. math::

            \begin{aligned}
                & \textrm{cross_attention_mask}( \\
                & \quad mask_{q} \in \sR^{\nBatchSize \times \nSeqLen_{q}} \\
                & \quad mask_{kv} \in \sR^{\nBatchSize \times \nSeqLen_{kv}} \\
                & ) \\
                & \rightarrow \sR^{\nBatchSize \times \nSeqLen_{q} \times \nSeqLen_{kv}}
            \end{aligned}

    Args:
        mask_q: a padding mask indicating at which positions values
            of the query are valid.
        mask_kv: a padding mask indicating at which positions values
            of the key are valid.

    Returns:
        a boolean matrix indicating which attention values are valid.
    """
    batch_size, seqlen_q = mask_q.shape
    _, seqlen_k = mask_kv.shape
    mask = jax.vmap(jnp.outer)(mask_q, mask_kv)
    assert mask.shape == (batch_size, seqlen_q, seqlen_k)
    return mask  # type: ignore


LikeFn = Callable[[Array, Array, Array, Array], Any]


def _like_value(*args: Any, index: int) -> Any:
    r"""Returns value of the argument at the particular index."""
    arg = args[index]
    if arg is None:
        raise ValueError(
            "Cannot infer a like-value, becase "
            "the target value has not been initialized yet."
        )
    return arg


def _like_shape(*args: Any, index: int) -> Any:
    r"""Returns shape of the argument at the particular index."""
    return _like_value(*args, index=index).shape[-1]


def _init_like(*args: Any, value: Union[Any, LikeFn], kind: Type[Any] = int) -> Any:
    r"""Returns value of the argument if its type matches the specified kind.
    Otherwise calls the argument."""

    if isinstance(value, kind):
        return value
    return value(*args)


d_like_input_q = partial(_like_shape, index=0)
d_like_input_kv = partial(_like_shape, index=1)
d_like_qk = partial(_like_value, index=2)
d_like_v = partial(_like_value, index=3)


class Attention(nn.Module):
    r"""Learns an attention vector for each position (i.e. time step)
    of the query sequence to all unmasked positions of the key sequence.

    .. math::

        \begin{aligned}
            & \textrm{Attention}( \\
            & \quad x_{q} \in \sR^{\nBatchSize \times \nSeqLen_{q} \times d_{x_{q}}} \\
            & \quad x_{kv} \in \sR^{\nBatchSize \times \nSeqLen_{kv} \times d_{x_{kv}}} \\
            & \quad mask \in \sR^{\nBatchSize \times \nSeqLen_{q} \times \nSeqLen_{kv}} \\
            & \quad \_ \\
            & \quad w_{q} \in \sR^{d_{x_{q}} \times n_{heads} \times d_{qk} \gets d_{x_{q}}} \\
            & \quad w_{k} \in \sR^{d_{x_{kv}} \times n_{heads} \times d_{qk} \gets d_{x_{q}}} \\
            & \quad w_{v} \in \sR^{d_{x_{kv}} \times n_{heads} \times d_{v} \gets d_{qk}} \\
            & \quad w_{o} \in \sR^{d_{v} \times d_{o} \gets d_{v}} \\
            & \quad b_{q} \in \sR^{n_{heads} \times d_{qk} \gets d_{x_{q}}} \\
            & \quad b_{k} \in \sR^{n_{heads} \times d_{qk} \gets d_{x_{q}}} \\
            & \quad b_{v} \in \sR^{n_{heads} \times d_{v} \gets d_{qk}} \\
            & \quad b_{o} \in \sR^{d_{o} \gets d_{v}} \\
            & ) \\
            & \rightarrow \sR^{\nBatchSize \times \nSeqLen_{q} \times d_{o} \gets d_{v}}
        \end{aligned}

    Args:
        inputs_q: inputs for the query.
        inputs_kv: inputs for key and value.
        mask: a mask tensor with boolean values indicating whether
            a particular query attends to a particular key.

    Returns:
        attention vectors.
    """

    n_heads: int = 8
    r"""a number of attention heads."""

    d_qk: Union[int, LikeFn] = d_like_input_q
    r"""a dimension of the query array (the key has the same dimension).
        Defaults to the same dimension as the dimension of query's inputs."""

    d_v: Union[int, LikeFn] = d_like_qk
    r"""a dimension of the value array.
        Defaults to the same dimension as the key's dimesnion."""

    d_output: Union[int, LikeFn] = d_like_v
    r"""an output dimension.
        Defaults to the same dimension as the values's dimesnion."""

    kernel_init: InitFn = nn.initializers.lecun_normal()
    r"""an initializer for weights."""

    bias_init: InitFn = nn.initializers.zeros
    r"""an initializer for bias."""

    use_bias: bool = True
    r"""wether to use bias."""

    dropout_rate: float = 0.0
    r"""probababilistic rate for attention dropout."""

    deterministic: bool = True
    r"""whether to perform deterministically or not."""

    precision: Optional[Precision] = None
    r"""numerical precision of the computation.
        See :attr:`jax.lax.Precision` for details."""

    @nn.compact
    def __call__(  # type: ignore[override] # pylint: disable=arguments-differ
        self,
        inputs_q: Array,
        inputs_kv: Array,
        mask: Optional[Array],
    ) -> Array:
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
r"""A variation of multi-head cross-attention where key's dimension
defaults to the dimension of query's inputs and output dimension defaults
to the value's dimension.

.. math::

    \begin{aligned}
        & \textrm{Attention}( \\
        & \quad x_{q} \in \sR^{\nBatchSize \times \nSeqLen_{q} \times d_{x_{q}}} \\
        & \quad x_{kv} \in \sR^{\nBatchSize \times \nSeqLen_{kv} \times d_{x_{kv}}} \\
        & \quad mask \in \sR^{\nBatchSize \times \nSeqLen_{q} \times \nSeqLen_{kv}} \\
        & \quad \_ \\
        & \quad w_{q} \in \sR^{d_{x_{q}} \times n_{heads} \times d_{qk} \gets d_{x_{q}}} \\
        & \quad w_{k} \in \sR^{d_{x_{kv}} \times n_{heads} \times d_{qk} \gets d_{x_{q}}} \\
        & \quad w_{v} \in \sR^{d_{x_{kv}} \times n_{heads} \times d_{v} \gets d_{qk}} \\
        & \quad w_{o} \in \sR^{d_{v} \times d_{o} \gets d_{v}} \\
        & \quad b_{q} \in \sR^{n_{heads} \times d_{qk} \gets d_{x_{q}}} \\
        & \quad b_{k} \in \sR^{n_{heads} \times d_{qk} \gets d_{x_{q}}} \\
        & \quad b_{v} \in \sR^{n_{heads} \times d_{v} \gets d_{qk}} \\
        & \quad b_{o} \in \sR^{d_{o} \gets d_{v}} \\
        & ) \\
        & \rightarrow \sR^{\nBatchSize \times \nSeqLen_{q} \times d_{o} \gets d_{v}}
    \end{aligned}
"""


KVQAttention = partial(Attention, d_qk=d_like_input_kv, d_output=d_like_input_q)
r"""A variation of  multi-head cross-attention where query's dimension
defaults to the dimension of key's inputs and output dimension defaults to the
dimension of query's inputs.

.. math::

    \begin{aligned}
        & \textrm{KVQAttention}( \\
        & \quad x_{q} \in \sR^{\nBatchSize \times \nSeqLen_{q} \times d_{x_{q}}} \\
        & \quad x_{kv} \in \sR^{\nBatchSize \times \nSeqLen_{kv} \times d_{x_{kv}}} \\
        & \quad mask \in \sR^{\nBatchSize \times \nSeqLen_{q} \times \nSeqLen_{kv}} \\
        & \quad \_ \\
        & \quad w_{q} \in \sR^{d_{x_{q}} \times n_{heads} \times d_{qk} \gets d_{x_{kv}}} \\
        & \quad w_{k} \in \sR^{d_{x_{kv}} \times n_{heads} \times d_{qk} \gets d_{x_{kv}}} \\
        & \quad w_{v} \in \sR^{d_{x_{kv}} \times n_{heads} \times d_{v} \gets d_{qk}} \\
        & \quad w_{o} \in \sR^{d_{v} \times d_{o} \gets d_{x_{q}}} \\
        & \quad b_{q} \in \sR^{n_{heads} \times d_{qk} \gets d_{x_{kv}}} \\
        & \quad b_{k} \in \sR^{n_{heads} \times d_{qk} \gets d_{x_{kv}}} \\
        & \quad b_{v} \in \sR^{n_{heads} \times d_{v} \gets d_{qk}} \\
        & \quad b_{o} \in \sR^{d_{o} \gets d_{x_{q}}} \\
        & ) \\
        & \rightarrow \sR^{\nBatchSize \times \nSeqLen_{q} \times d_{o} \gets d_{v}}
    \end{aligned}
"""


class SelfAttention(Attention):
    r"""Learns an attention of each position (i.e. time step)
    of the sequence to all unmasked positions of the same sequence.

    .. math::

        \begin{aligned}
            & \textrm{SelfAttention}( \\
            & \quad x_{qkv} \in \sR^{\nBatchSize \times \nSeqLen_{qkv} \times d_{x_{qkv}}} \\
            & \quad mask \in \sR^{\nBatchSize \times \nSeqLen_{qkv} \times \nSeqLen_{qkv}} \\
            & \quad \_ \\
            & \quad w_{q} \in \sR^{d_{x_{qkv}} \times n_{heads} \times d_{qk} \gets d_{x_{qkv}}} \\
            & \quad w_{k} \in \sR^{d_{x_{qkv}} \times n_{heads} \times d_{qk} \gets d_{x_{qkv}}} \\
            & \quad w_{v} \in \sR^{d_{x_{qkv}} \times n_{heads} \times d_{v} \gets d_{qk}} \\
            & \quad w_{o} \in \sR^{d_{v} \times d_{o} \gets d_{v}} \\
            & \quad b_{q} \in \sR^{n_{heads} \times d_{qk} \gets d_{x_{qkv}}} \\
            & \quad b_{k} \in \sR^{n_{heads} \times d_{qk} \gets d_{x_{qkv}}} \\
            & \quad b_{v} \in \sR^{n_{heads} \times d_{v} \gets d_{qk}} \\
            & \quad b_{o} \in \sR^{d_{o} \gets d_{v}} \\
            & ) \\
            & \rightarrow \sR^{\nBatchSize \times \nSeqLen_{q} \times d_{o} \gets d_{v}}
        \end{aligned}

    Args:
        inputs_qkv: inputs for the query, key, and value.
        mask: a mask tensor with boolean values indicating whether
            a particular query attends to a particular key.

    Returns:
        attention vectors.
    """

    @nn.compact
    def __call__(  # type: ignore[override]
        self,
        inputs_qkv: Array,
        mask: Optional[Array],
    ) -> Array:
        return super().__call__(
            inputs_q=inputs_qkv,
            inputs_kv=inputs_qkv,
            mask=mask,
        )


QKVAttentionFn = Callable[[Array, Array, Array], Array]
QKVAttentionCt = Callable[..., QKVAttention]
KVQAttentionFn = Callable[[Array, Array, Array], Array]
KVQAttentionCt = Callable[..., KVQAttentionFn]
SelfAttentionFn = Callable[[Array, Array], Array]
SelfAttentionCt = Callable[..., SelfAttentionFn]
