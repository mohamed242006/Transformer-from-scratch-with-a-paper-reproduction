"""Tokenizer interface and the SentencePiece backend (ADR-005).

This module is the only place in the package that imports ``sentencepiece``
(enforced by ``tests/policy/test_layering.py``).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import sentencepiece as spm

from aiayn.data.config import TokenizerConfig
from aiayn.data.special_tokens import (
    BOS_ID,
    BOS_PIECE,
    EOS_ID,
    EOS_PIECE,
    PAD_ID,
    PAD_PIECE,
    UNK_ID,
    UNK_PIECE,
)
from aiayn.utils.hashing import sha256_bytes

__all__ = ["SentencePieceTokenizer", "Tokenizer", "train_sentencepiece"]

_SPECIALS_TO_DROP = frozenset({PAD_ID, BOS_ID, EOS_ID})

# Full-width letters, a ligature, a decomposed umlaut and odd spacing: differs under
# different normalization rules (e.g. nmt_nfkc vs identity).
_NORMALIZATION_PROBE = (
    "".join(chr(0xFF21 + ord(c) - ord("A")) for c in "HELLO")  # full-width letters
    + "  "
    + chr(0xFB01)
    + "ne"  # fi ligature
    + "  u"
    + chr(0x0308)
    + "ber"  # decomposed umlaut
    + chr(0x00A0)
    + "Stra"
    + chr(0x00DF)
    + "e 123"  # no-break space, sharp s
)


@runtime_checkable
class Tokenizer(Protocol):
    """What the data pipeline needs from a tokenizer. Encoding never adds BOS/EOS."""

    @property
    def vocab_size(self) -> int: ...

    @property
    def fingerprint(self) -> str: ...

    def encode(self, text: str) -> list[int]: ...

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]: ...

    def decode(self, ids: Sequence[int]) -> str: ...


class SentencePieceTokenizer:
    """A trained SentencePiece model, validated against the project's special IDs.

    The serialized model is held in memory, so the object pickles cheaply and does not
    depend on the file staying in place (Windows ``spawn`` workers re-create it).
    """

    def __init__(self, model_path: str | Path) -> None:
        self._path = Path(model_path)
        self._proto = self._path.read_bytes()
        self._sp = self._load(self._proto, self._path)

    @staticmethod
    def _load(proto: bytes, path: Path) -> Any:
        sp = spm.SentencePieceProcessor(model_proto=proto)
        actual = (sp.pad_id(), sp.unk_id(), sp.bos_id(), sp.eos_id())
        expected = (PAD_ID, UNK_ID, BOS_ID, EOS_ID)
        if actual != expected:
            raise ValueError(
                f"{path}: special IDs (pad, unk, bos, eos) are {actual}, expected {expected}. "
                "Retrain the tokenizer with aiayn.data.tokenizer.train_sentencepiece."
            )
        return sp

    @property
    def path(self) -> Path:
        return self._path

    @property
    def vocab_size(self) -> int:
        return int(self._sp.get_piece_size())

    @property
    def fingerprint(self) -> str:
        """Content hash identifying the segmentation; stored with every run (ADR-005).

        Not the file hash: SentencePiece embeds its training arguments (including the
        output path) in the model file, so identical tokenizers trained in different
        directories or machines would get different file hashes. The fingerprint covers
        every piece, score and piece type, plus the encoding of a probe string that
        exercises Unicode normalization.
        """
        sp = self._sp
        lines = [
            f"{i}\t{sp.id_to_piece(i)}\t{sp.get_score(i)!r}\t"
            f"{int(sp.is_control(i))}{int(sp.is_unknown(i))}{int(sp.is_byte(i))}"
            for i in range(sp.get_piece_size())
        ]
        lines.append("probe\t" + " ".join(map(str, sp.encode(_NORMALIZATION_PROBE))))
        return sha256_bytes("\n".join(lines).encode("utf-8"))

    def encode(self, text: str) -> list[int]:
        return [int(i) for i in self._sp.encode(text, out_type=int)]

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        encoded = self._sp.encode(list(texts), out_type=int)
        return [[int(i) for i in ids] for ids in encoded]

    def decode(self, ids: Sequence[int]) -> str:
        """Decode content IDs; PAD/BOS/EOS are dropped, UNK renders as ' ⁇ '."""
        kept = [int(i) for i in ids if int(i) not in _SPECIALS_TO_DROP]
        return str(self._sp.decode(kept))

    def id_to_piece(self, token_id: int) -> str:
        return str(self._sp.id_to_piece(token_id))

    def __getstate__(self) -> dict[str, Any]:
        return {"path": self._path, "proto": self._proto}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._path = state["path"]
        self._proto = state["proto"]
        self._sp = self._load(self._proto, self._path)

    def __repr__(self) -> str:
        return (
            f"SentencePieceTokenizer(path={str(self._path)!r}, vocab_size={self.vocab_size}, "
            f"fingerprint={self.fingerprint[:12]})"
        )


def train_sentencepiece(
    input_files: Sequence[str | Path],
    model_prefix: str | Path,
    config: TokenizerConfig,
) -> Path:
    """Train a joint SentencePiece model and return the path of the ``.model`` file.

    ``input_files`` must contain **training-split text only** (ADR-005: no leakage).
    """
    files = [Path(f) for f in input_files]
    if not files:
        raise ValueError("at least one input file is required")
    for f in files:
        if not f.is_file():
            raise FileNotFoundError(f)
        if "," in str(f):
            raise ValueError(f"SentencePiece treats ',' as a file separator: {f}")
    prefix = Path(model_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=",".join(str(f) for f in files),
        model_prefix=str(prefix),
        model_type=config.model_type,
        vocab_size=config.vocab_size,
        character_coverage=config.character_coverage,
        normalization_rule_name=config.normalization_rule_name,
        input_sentence_size=config.input_sentence_size,
        shuffle_input_sentence=True,
        num_threads=config.num_threads,
        hard_vocab_limit=config.hard_vocab_limit,
        pad_id=config.pad_id,
        unk_id=config.unk_id,
        bos_id=config.bos_id,
        eos_id=config.eos_id,
        pad_piece=PAD_PIECE,
        unk_piece=UNK_PIECE,
        bos_piece=BOS_PIECE,
        eos_piece=EOS_PIECE,
        minloglevel=2,
    )
    model_path = prefix.with_name(prefix.name + ".model")
    if not model_path.is_file():
        raise RuntimeError(f"SentencePiece did not produce {model_path}")
    return model_path
