"""Special-token IDs shared by every tokenizer and dataset (ADR-005).

These IDs are fixed constants, not configuration: the embedding ``padding_idx``, the
masks and the loss all rely on them. Tokenizer configs must record the same values,
and a mismatch is an error.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "BOS_ID",
    "BOS_PIECE",
    "EOS_ID",
    "EOS_PIECE",
    "NUM_SPECIAL_TOKENS",
    "PAD_ID",
    "PAD_PIECE",
    "UNK_ID",
    "UNK_PIECE",
]

PAD_ID: Final = 0
UNK_ID: Final = 1
BOS_ID: Final = 2
EOS_ID: Final = 3

#: Content tokens (synthetic or learned) start at this ID.
NUM_SPECIAL_TOKENS: Final = 4

PAD_PIECE: Final = "<pad>"
UNK_PIECE: Final = "<unk>"
BOS_PIECE: Final = "<s>"
EOS_PIECE: Final = "</s>"
