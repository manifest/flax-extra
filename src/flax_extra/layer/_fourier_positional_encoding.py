r"""Fourier positional encoding."""

from typing import Any, Callable, List, Sequence, Optional, Union
from functools import reduce
import operator
import jax.numpy as jnp
from jax.core import NamedShape
from jax.random import KeyArray
from flax import linen as nn

Array = jnp.ndarray
Shape = Optional[Union[Sequence[int], NamedShape]]
Dtype = Any
Positions = List[int]
InitFn = Callable[[KeyArray, Shape, Dtype], Array]


def _build_fourier_encodings(
    spatial_encodings: Array,
    seqshape: tuple[int, ...],
    n_bands: int,
    use_sine_only: bool = False,
) -> Array:
    r"""Generates Fourier positional encoding vectors with linear spacing.

    The function maps spatial coordinates to the surface of a higher dimensional
    hypersphere with a set of sinusoids.

    Args:
        spatial_encodings: an unbatched vectors representing multi-dimensional
            spatial positions, :math:`\in \sR^{\nSeqLen \times d}`.
        seqshape: a shape of the sequence.
        n_bands: a number of frequency bands to use.
        use_sine_only: whether to use a single phase (i.e. sine only) or two phase
            (i.e. sine and cosine) for each frequency band.

    Returns:
        an unbatched vectors of Fourier features,
            :math:`\sR^{\nSeqLen \times d}`.
    """
    seqlen, d_spatial_encoding = spatial_encodings.shape

    min_frequency = 1.0
    frequency_bands = jnp.stack(
        [
            jnp.linspace(min_frequency, seqlen / 2.0, num=n_bands, endpoint=True)
            for seqlen in seqshape
        ],
        axis=0,
    )

    scaled_frequency_bands = spatial_encodings[:, :, None] * frequency_bands[None, :, :]
    d_encoding = reduce(operator.mul, scaled_frequency_bands.shape[1:])
    frequencies = jnp.reshape(scaled_frequency_bands, (-1, d_encoding))
    assert (
        d_spatial_encoding * n_bands == d_encoding
    ), f"An invalid dimention of the Fourier positional encoding {d_encoding}."

    if use_sine_only:
        encodings = jnp.sin(jnp.pi * frequencies)
        assert encodings.shape == (
            seqlen,
            d_spatial_encoding * n_bands,
        ), f"An invalid shape of the sine-only Fourier positional encodings {encodings.shape}."
    else:
        encodings = jnp.concatenate(
            [
                jnp.sin(jnp.pi * frequencies),
                jnp.cos(jnp.pi * frequencies),
            ],
            axis=-1,
        )
        assert encodings.shape == (
            seqlen,
            2 * d_spatial_encoding * n_bands,
        ), f"An invalid shape of the sine-cosine Fourier positional encodings {encodings.shape}."

    return encodings  # type: ignore


def _build_spatial_positional_encodings(
    seqshape: tuple[Any, ...],
    bounds: tuple[float, float],
) -> Array:
    r"""Generates encoding vectors for multi-dimensional spatial positions.

    A positional encoding vector at each position may be seen as a
    multi-dimensional representation of the position across all spatial
    dimensions (or similarly, as a coordinate in the
    :math:`[\textrm{lower_bound} \times \textrm{upper_bound}]^{d}` space).

    Args:
        seqshape: a shape of the sequence.
        lower_bound: a minimal value for a position.
        upper_bound: a maximum value for a position.

    Returns:
        an unbatched vectors representing multi-dimensional spatial positions,
            :math:`\sR^{\nSeqLen \times d}`.
    """
    lower_bound, upper_bound = bounds
    points = [
        jnp.linspace(
            lower_bound,
            upper_bound,
            num=seqlen,
            endpoint=True,
            dtype=jnp.float32,
        )
        for seqlen in seqshape
    ]
    coordinate_grids = jnp.meshgrid(*points, indexing="ij")
    coordinates = jnp.stack(coordinate_grids, axis=-1)
    assert coordinates.shape == (*seqshape, len(seqshape)), (
        f"An invalid shape of the coordinate grid {coordinates.shape} "
        "for spatial positional encoding."
    )

    seqlen = reduce(operator.mul, seqshape)
    coordinates = jnp.reshape(coordinates, (seqlen, -1))
    return coordinates  # type: ignore


def _scale_spatial_positional_coordinates(
    coordinates: Array,
    seqshape: tuple[int, ...],
    bounds: tuple[float, float],
) -> Array:
    lower_bound, upper_bound = bounds
    distance = abs(lower_bound) + abs(upper_bound)
    return lower_bound + coordinates * distance / jnp.array(seqshape)  # type: ignore


def _to_spatial_positional_encodings(
    output_positions: Positions,
    seqshape: tuple[int, ...],
    bounds: tuple[float, float],
) -> Array:
    return _scale_spatial_positional_coordinates(
        jnp.stack(jnp.unravel_index(output_positions, shape=seqshape), axis=1),
        seqshape=seqshape,
        bounds=bounds,
    )


class FourierPositionEncoding(nn.Module):
    r"""Computes a vector of Fourier (or sinusoidal) features
    that may be seen as a multi-dimensional representation of
    the position across all its spatial dimensions (or similarly,
    as a single encoding of different time axes).

    .. math::

        \begin{aligned}
            & \textrm{FourierPositionEncoding}( \\
            & \quad m \in \sN \\
            & ) \\
            & \rightarrow \sR^{\nBatchSize \times T \times d}
        \end{aligned}

        \begin{aligned}
            & \textrm{FourierPositionEncoding}( \\
            & \quad m \in \sN \\
            & \quad t \in \sN^{\nSeqLen^{\prime}} \\
            & ) \\
            & \rightarrow \sR^{\nBatchSize \times T \times T^{\prime} \times d}
        \end{aligned}

    Output dimension may vary depending on parameters of the module:

    - For a single phase (i.e. sine only), `use_sine_only=True`.
        - With spatial coordinates get concatenated:
            :math:`d = n_{bands} \cdot len(seqshape) + len(seqshape)`
        - Otherwise:
            :math:`d = n_{bands} \cdot len(seqshape)`
    - For two phase (i.e. sine and cosine) for each frequency band.
        - With spatial coordinates get concatenated:
            :math:`d = n_{bands} \cdot len(seqshape) \cdot 2 + len(seqshape)`
        - Otherwise:
            :math:`d = n_{bands} \cdot len(seqshape) \cdot 2`

    Args:
        batch_size: a batch size of the sequence.
        output_positions: a subset of positions (i.e. time steps) within
            the sequence positional encoding will be calculated.

    Returns:
        positional encoding vectors.
    """

    seqshape: tuple[int, ...]
    r"""a shape of the sequence."""

    n_bands: int
    r"""a number of frequency bands to use."""

    spatial_coordinate_bounds: tuple[float, float] = (-1.0, 1.0)
    r"""lower and upper bound for spatial coordinates."""

    use_spatial_coordinates: bool = True
    r"""whether to concatenate the spatial coordinates to the Fourier features."""

    use_sine_only: bool = False
    r"""whether to use a single phase (i.e. sine only) or two phase (i.e. sine and cosine)
    for each frequency band."""

    @nn.compact
    def __call__(  # type: ignore[override] # pylint: disable=arguments-differ
        self,
        batch_size: int,
        output_positions: Optional[Positions] = None,
    ) -> Array:
        if output_positions is None:
            spatial_encodings = _build_spatial_positional_encodings(
                seqshape=self.seqshape,
                bounds=self.spatial_coordinate_bounds,
            )
        else:
            spatial_encodings = _to_spatial_positional_encodings(
                output_positions=output_positions,
                seqshape=self.seqshape,
                bounds=self.spatial_coordinate_bounds,
            )

        fourier_encodings = _build_fourier_encodings(
            spatial_encodings,
            n_bands=self.n_bands,
            seqshape=self.seqshape,
            use_sine_only=self.use_sine_only,
        )
        if self.use_spatial_coordinates:
            fourier_encodings = jnp.concatenate(
                [spatial_encodings, fourier_encodings],
                axis=-1,
            )

        fourier_encodings = jnp.broadcast_to(
            fourier_encodings[None],
            (batch_size, *fourier_encodings.shape),
        )
        return fourier_encodings
