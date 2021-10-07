r"""The Perceiver IO model."""

from typing import cast, Any, List, Optional, Union
from jax import numpy as jnp
from flax import linen as nn
from flax_extra import combinator as cb
from flax_extra.layer._attention import (
    SelfAttention,
    SelfAttentionCt,
    KVQAttention,
    KVQAttentionCt,
)
from flax_extra.layer._feedforward import (
    FeedForward,
    FeedForwardCt,
)
from flax_extra.layer._multimodal_encoding import (
    MultimodalEncodingCt,
)
from flax_extra.layer._multimodal_positional_encoding import (
    MultimodalPositionalEncodingCt,
)
from flax_extra.layer._multimodal_decoding import (
    MultimodalDecodingCt,
)
from flax_extra.model.perceiver._encoder import Encoder
from flax_extra.model.perceiver._processor import Processor
from flax_extra.model.perceiver._decoder import Decoder

Array = jnp.ndarray
Precision = Any
Positions = List[int]


class PerceiverIO(nn.Module):
    r"""Maps arbitrary input tensors to arbitrary output tensors
    in a domain agnostic process.

    The bulk of the computation happens in a latent space whose
    size is typically smaller than the inputs and outputs,
    which makes the process computationally tractable even
    for very large inputs and outputs.

    .. math::

        \begin{aligned}
            & \textrm{PerceiverIO}( \\
            & \quad x \in n_{mod_{x}} \times \sR^{m \times T_{x} \times d_{x}} \\
            & \quad mask_{x} \in \sR^{\nBatchSize \times \nSeqLen_{x}} \\
            & \quad y \in n_{mod_{y}} \times \sR^{m \times T_{y} \times d_{y}} \\
            & \quad mask_{y} \in \sR^{\nBatchSize \times \nSeqLen_{y}} \\
            & \quad t_{y} \in n_{mod_{y}} \times \sN^{\nSeqLen^{\prime}_{y}} \\
            & \quad \_ \\
            & \quad \theta \gets MultimodalEncoding() \\
            & \quad \theta \gets MultimodalPositionalEncoding() \\
            & \quad \theta \gets Encoder() \\
            & \quad \theta \gets Processor() \\
            & \quad \theta \gets MultimodalEncoding() \\
            & \quad \theta \gets MultimodalPositionalEncoding() \mid MultimodalEncoding() \\
            & \quad \theta \gets Decoder() \\
            & \quad \theta \gets MultimodalDecoding() \\
            & ) \\
            & \rightarrow n_{mod_{y}} \times \sR^{\nBatchSize \times \dots d^{\prime}_{y}}
        \end{aligned}

    Args:
        inputs: a single or multimodal input sequence(s).
        input_mask: a padding mask indicating at which positions values
            of the inputs are valid.
        targets: a single or multimodal target sequence(s).
            If provided, `decoder_query_encoding`
            must be of :class:`flax_extra.layer.MultimodalEncoding` type.
            Otherwise :class:`flax_extra.layer.MultimodalPositionalEncoding`
            type must be used.
        target_mask: a padding mask indicating at which positions values
            of the targets are valid.
        output_positions: a subset of positions (i.e. time steps) within
            each modality encoding will be calculated.

    Returns:
        a single or multimodal output sequence(s).
    """

    input_encoding: MultimodalEncodingCt
    r"""preprocessing and encoding of a single or multimodal input.
    Use :meth:`flax_extra.layer.io.input_encoding`
    to define the module type."""

    encoder_query_encoding: MultimodalPositionalEncodingCt
    r"""an encoding for encoder's query.
    Use :meth:`flax_extra.layer.io.query_encoding`
    to define the module type."""

    decoder_query_encoding: Union[MultimodalPositionalEncodingCt, MultimodalEncodingCt]
    r"""an encoding for decoder's query.
    Use :meth:`flax_extra.layer.io.query_encoding`
    to define the module type."""

    output_decoding: MultimodalDecodingCt
    r"""output decoding and postrocessing.
    Use :meth:`flax_extra.layer.io.output_decoding`
    to define the module type."""

    n_processor_shards: int = 8
    r"""a number of shards.
    See :class:`flax_extra.model.perceiver.Processor`."""

    n_processor_blocks: int = 6
    r"""a number of self-attention blocks building up a single shard.
    See :class:`flax_extra.model.perceiver.Processor`."""

    processor_attention: SelfAttentionCt = SelfAttention
    r"""a type of the self-attention for processor.
    See :class:`flax_extra.layer.SelfAttention`."""

    processor_feed_forward: FeedForwardCt = FeedForward
    r"""a type of the feed-forward for processor.
    See :class:`flax_extra.layer.FeedForward`."""

    encoder_attention: KVQAttentionCt = KVQAttention
    r"""a type of the cross-attention for encoder.
    See :class:`flax_extra.layer.KVQAttention`."""

    encoder_feed_forward: FeedForwardCt = FeedForward
    r"""a type of the feed-forward for encoder.
    See :class:`flax_extra.layer.FeedForward`."""

    use_encoder_q_residual: bool = True
    r"""whether to include a residual to the query.
    Consider omitting the residual if the semantics of encoder's query
    and latent features are different (e.g. if queries are positions
    and latents are pixels)."""

    decoder_attention: KVQAttentionCt = KVQAttention
    r"""a type of the cross-attention for decoder.
    See :class:`flax_extra.layer.KVQAttention`."""

    decoder_feed_forward: FeedForwardCt = FeedForward
    r"""a type of the feed-forward for decoder.
    See :class:`flax_extra.layer.FeedForward`."""

    use_decoder_q_residual: bool = False
    r"""whether to include a residual to the query.
    Consider omitting the residual if the semantics of decoder's query
    and outputs are different (e.g. if queries are positions and outputs
    are pixels)."""

    deterministic: bool = True
    r"""whether to perform deterministically or not."""

    precision: Optional[Precision] = None
    r"""numerical precision of the computation.
    See :attr:`jax.lax.Precision` for details."""

    @nn.compact
    def __call__(  # type: ignore[override] # pylint: disable=arguments-differ, disable=too-many-arguments), disable=too-many-locals
        self,
        inputs: Union[Array, List[Array]],
        input_mask: Optional[Array] = None,
        targets: Union[None, Array, List[Array]] = None,
        target_mask: Optional[Array] = None,
        output_positions: Union[None, Positions, List[Optional[Positions]]] = None,
    ) -> Union[Array, List[Array]]:
        encoder_query_encoding = self.encoder_query_encoding(
            name="EncodedQueryEncoding"
        )
        input_encoding = self.input_encoding(
            name="InputEncoding",
        )
        decoder_query_encoding = self.decoder_query_encoding(
            name="DecoderQueryEncoding"
        )
        io_block = cb.serial(
            Encoder(
                attention=self.encoder_attention,
                feed_forward=self.encoder_feed_forward,
                use_q_residual=self.use_encoder_q_residual,
                deterministic=self.deterministic,
                precision=self.precision,
                name="Encoder",
            ),
            Processor(
                n_shards=self.n_processor_shards,
                n_blocks=self.n_processor_blocks,
                attention=self.processor_attention,
                feed_forward=self.processor_feed_forward,
                deterministic=self.deterministic,
                precision=self.precision,
                name="Processor",
            ),
            Decoder(
                attention=self.decoder_attention,
                feed_forward=self.decoder_feed_forward,
                use_q_residual=self.use_decoder_q_residual,
                deterministic=self.deterministic,
                precision=self.precision,
                name="Decoder",
            ),
        )
        output_decoding = self.output_decoding(
            name="OutputDecoding",
        )

        # Cast inputs to a list of a single or
        # multiple sequences per each input modality.
        if input_encoding.n_modalities == 1:
            inputs = cast(List[Array], [inputs])
        inputs = cast(List[Array], inputs)

        # Cast targets and output positions to a list of a single or
        # multiple sequences per each output modality.
        if output_decoding.n_modalities == 1:
            if targets is not None:
                targets = cast(List[Array], [targets])
            if output_positions is not None:
                output_positions = cast(List[Optional[Positions]], [output_positions])
        targets = cast(Optional[List[Array]], targets)
        output_positions = cast(Optional[List[Optional[Positions]]], output_positions)

        batch_size = inputs[0].shape[0]

        encoded_inputs, seqlen_inputs = input_encoding(
            multimodal_inputs=inputs,
            multimodal_output_positions=None,
        )
        del seqlen_inputs

        encoder_query, seqlen_latents = encoder_query_encoding(batch_size, None)
        del seqlen_latents

        decoder_query, seqlen_outputs = decoder_query_encoding(
            batch_size if targets is None else targets,  # type: ignore
            output_positions,
        )

        outputs = io_block(  # type: ignore
            encoded_inputs,  # type: ignore
            encoder_query,
            input_mask,
            decoder_query,
            target_mask,
        )

        return output_decoding(outputs, seqlen_outputs)  # type: ignore
