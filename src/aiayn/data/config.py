"""Configuration schema for the data group (``configs/data/*.yaml``; ADR-007)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from aiayn.data.special_tokens import BOS_ID, EOS_ID, PAD_ID, UNK_ID
from aiayn.utils.config import OMIT_IF_DEFAULT, load_config

__all__ = [
    "SPLITS",
    "ArchiveSource",
    "BatchingConfig",
    "CopyTaskConfig",
    "DataConfig",
    "DownloadSpec",
    "FilterConfig",
    "Split",
    "TokenizerConfig",
    "TranslationConfig",
    "load_data_config",
]

Split = Literal["train", "valid", "test"]
SPLITS: tuple[Split, ...] = ("train", "valid", "test")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LANG = re.compile(r"^[a-z]{2,3}$")


def _positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


@dataclass(frozen=True)
class BatchingConfig:
    """Token-budget batching (ADR-009)."""

    max_tokens: int = 4096
    max_sentences: int | None = None
    pool_size: int = 4096
    eval_max_tokens: int | None = None
    pad_to_multiple_of: int = 1
    num_workers: int = 0

    def __post_init__(self) -> None:
        _positive("max_tokens", self.max_tokens)
        _positive("pool_size", self.pool_size)
        _positive("pad_to_multiple_of", self.pad_to_multiple_of)
        if self.max_sentences is not None:
            _positive("max_sentences", self.max_sentences)
        if self.eval_max_tokens is not None:
            _positive("eval_max_tokens", self.eval_max_tokens)
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")


@dataclass(frozen=True)
class CopyTaskConfig:
    """Synthetic copy/reverse task."""

    mode: Literal["copy", "reverse"]
    vocab_size: int
    min_len: int
    max_len: int
    num_train: int
    num_valid: int
    num_test: int

    def __post_init__(self) -> None:
        if self.vocab_size <= 4:
            raise ValueError("vocab_size must exceed the 4 special tokens")
        if not 1 <= self.min_len <= self.max_len:
            raise ValueError(f"need 1 <= min_len <= max_len, got {self.min_len}, {self.max_len}")
        for name in ("num_train", "num_valid", "num_test"):
            _positive(name, getattr(self, name))


@dataclass(frozen=True)
class DownloadSpec:
    """One pinned source file (ADR-002: checksum-verified)."""

    url: str
    sha256: str
    split: Split
    lang: str
    compression: Literal["gzip", "none"] = "gzip"

    def __post_init__(self) -> None:
        if not _SHA256.match(self.sha256):
            raise ValueError(f"sha256 must be 64 lowercase hex characters: {self.sha256!r}")
        if not self.url.startswith(("https://", "file://")):
            raise ValueError(f"only https:// and file:// URLs are allowed: {self.url!r}")


@dataclass(frozen=True)
class ArchiveSource:
    """One pinned archive turned into all splits by a named recipe (ADR-023).

    ``iwslt14_fairseq``: the IWSLT'14 DE-EN TED archive, split exactly as fairseq's
    ``prepare-iwslt14.sh`` (train/valid membership, test concatenation), keeping the raw
    cased text. ``*_lines`` pin the expected sentence counts (0 = unchecked).
    """

    url: str
    sha256: str
    recipe: Literal["iwslt14_fairseq"]
    train_lines: int = 0
    valid_lines: int = 0
    test_lines: int = 0

    def __post_init__(self) -> None:
        if not _SHA256.match(self.sha256):
            raise ValueError(f"sha256 must be 64 lowercase hex characters: {self.sha256!r}")
        if not self.url.startswith(("https://", "file://")):
            raise ValueError(f"only https:// and file:// URLs are allowed: {self.url!r}")
        if min(self.train_lines, self.valid_lines, self.test_lines) < 0:
            raise ValueError("expected line counts must be >= 0")


@dataclass(frozen=True)
class FilterConfig:
    """Length filtering, applied to tensor lengths S and T (ADR-009)."""

    max_len: int
    min_len: int = 1
    max_len_ratio: float | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.min_len < self.max_len:
            raise ValueError(f"need 1 <= min_len < max_len, got {self.min_len}, {self.max_len}")
        if self.max_len_ratio is not None and self.max_len_ratio < 1.0:
            raise ValueError("max_len_ratio must be >= 1")


@dataclass(frozen=True)
class TokenizerConfig:
    """SentencePiece settings. Every key is required (ADR-005 compliance)."""

    model_type: Literal["bpe", "unigram"]
    vocab_size: int
    character_coverage: float
    normalization_rule_name: str
    input_sentence_size: int
    num_threads: int
    hard_vocab_limit: bool
    pad_id: int
    unk_id: int
    bos_id: int
    eos_id: int

    def __post_init__(self) -> None:
        _positive("vocab_size", self.vocab_size)
        _positive("num_threads", self.num_threads)
        if not 0.0 < self.character_coverage <= 1.0:
            raise ValueError("character_coverage must be in (0, 1]")
        if self.input_sentence_size < 0:
            raise ValueError("input_sentence_size must be >= 0 (0 = use all sentences)")
        expected = (PAD_ID, UNK_ID, BOS_ID, EOS_ID)
        actual = (self.pad_id, self.unk_id, self.bos_id, self.eos_id)
        if actual != expected:
            raise ValueError(f"special IDs (pad, unk, bos, eos) must be {expected}, got {actual}")


@dataclass(frozen=True)
class TranslationConfig:
    """A parallel corpus: sources, filtering and tokenizer."""

    src_lang: str
    tgt_lang: str
    filter: FilterConfig
    tokenizer: TokenizerConfig
    downloads: tuple[DownloadSpec, ...] = ()
    root: str = "data"
    #: Alternative to ``downloads`` (ADR-023); omitted from serialization when unset so
    #: existing configs keep their hashes.
    archive: ArchiveSource | None = field(default=None, metadata=OMIT_IF_DEFAULT)

    def __post_init__(self) -> None:
        for lang in (self.src_lang, self.tgt_lang):
            if not _LANG.match(lang):
                raise ValueError(f"language codes must be 2-3 lowercase letters: {lang!r}")
        if self.src_lang == self.tgt_lang:
            raise ValueError("src_lang and tgt_lang must differ")
        if self.archive is not None:
            if self.downloads:
                raise ValueError("give either 'downloads' or 'archive', not both")
            if self.archive.recipe == "iwslt14_fairseq" and {self.src_lang, self.tgt_lang} != {
                "de",
                "en",
            }:
                raise ValueError("recipe iwslt14_fairseq provides de and en only")
            return
        seen: set[tuple[str, str]] = set()
        for spec in self.downloads:
            if spec.lang not in (self.src_lang, self.tgt_lang):
                raise ValueError(f"download language {spec.lang!r} is neither source nor target")
            key = (spec.split, spec.lang)
            if key in seen:
                raise ValueError(f"duplicate download for split/lang {key}")
            seen.add(key)
        missing = [
            (s, lang)
            for s in SPLITS
            for lang in (self.src_lang, self.tgt_lang)
            if (s, lang) not in seen
        ]
        if missing:
            raise ValueError(f"downloads missing for split/lang pairs: {missing}")


@dataclass(frozen=True)
class DataConfig:
    """Top-level data configuration. Exactly one task section must be present."""

    name: str
    task: Literal["copy", "translation"]
    seed: int = 0
    batching: BatchingConfig = field(default_factory=BatchingConfig)
    copy: CopyTaskConfig | None = None
    translation: TranslationConfig | None = None

    def __post_init__(self) -> None:
        if not re.match(r"^[a-z0-9_]+$", self.name):
            raise ValueError(f"name must be lowercase snake_case: {self.name!r}")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.task == "copy" and (self.copy is None or self.translation is not None):
            raise ValueError("task 'copy' requires a 'copy' section and no 'translation' section")
        if self.task == "translation" and (self.translation is None or self.copy is not None):
            raise ValueError(
                "task 'translation' requires a 'translation' section and no 'copy' section"
            )


def load_data_config(path: str | Path, overrides: tuple[str, ...] = ()) -> DataConfig:
    """Load and validate a ``configs/data/*.yaml`` file."""
    return load_config(path, DataConfig, overrides)
