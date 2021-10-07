r"""Bytes tokenizer."""

from typing import AnyStr, List, Optional, Mapping
from dataclasses import dataclass
from jax import numpy as jnp
import numpy as np

Array = jnp.ndarray


@dataclass
class BytesTokenizer:
    r"""Tokenizer mapping text strings to their UTF-8 bytes."""

    reserved_ids: Mapping[str, int]
    r"""reserved token identifiers."""

    @property
    def vocab_size(self) -> int:
        r"""a vocabulary size."""
        return self.d_reserved + 256

    @property
    def d_reserved(self) -> int:
        r"""a number of reserved tokens."""
        return len(self.reserved_ids)

    def to_tokens(self, ids: Array) -> str:
        r"""Maps text characters to UTF-8 bytes.

        Args:
            ids: a byte sequence.

        Returns:
            a sequence of text characters.
        """
        ids_no_special = ids[ids >= self.d_reserved] - self.d_reserved
        buf: bytes = ids_no_special.astype(np.uint8).tobytes()
        return buf.decode("utf-8", errors="replace")

    def to_ids(self, tokens: AnyStr) -> Array:
        r"""Maps UTF-8 bytes to text characters.

        Args:
            tokens: a sequence of text characters.

        Returns:
            a byte sequence.
        """
        if isinstance(tokens, str):
            tokens = tokens.encode("utf-8")  # type: ignore
        ids = np.frombuffer(tokens, np.uint8).astype(np.int32)
        return ids + self.d_reserved  # type: ignore

    def pad(self, inputs: Array, max_length: int) -> Array:
        r"""Pads the sequence up to desired length.

        Args:
            inputs: an input sequence.
            max_length: desired sequence length.

        Returns:
            a padded sequence.
        """
        if "PAD" not in self.reserved_ids:
            raise ValueError(
                "Cannot pad a token sequence, because "
                "reserved token `PAD` is missing."
            )
        return _pad(inputs, max_length, self.reserved_ids.get("PAD"))


def bytes_tokenizer(reserved_tokens: List[str]) -> BytesTokenizer:
    r"""Creates a tokenizer mapping text strings to their UTF-8 bytes.

    Args:
        reserved_tokens: a list of reserved tokens.

    Returns:
        a tokenizer.
    """
    reserved_ids = {token: id for id, token in enumerate(reserved_tokens)}
    return BytesTokenizer(reserved_ids=reserved_ids)


def _pad(inputs: Array, max_length: int, pad_token: Optional[int]) -> Array:
    length = inputs.shape[1]
    if length > max_length:
        raise ValueError(
            "Cannot pad a token sequence, because "
            f"an example length {length} exceeds the maximum length {max_length}."
        )

    pad_length = max_length - length
    padded_inputs = np.pad(
        inputs,
        pad_width=((0, 0), (0, pad_length)),
        constant_values=pad_token,
    )

    return padded_inputs  # type: ignore
