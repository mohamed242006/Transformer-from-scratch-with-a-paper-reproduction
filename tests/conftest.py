"""Shared fixtures. Tests never touch the network or the repository's data/ folder."""

from __future__ import annotations

import gzip
import random
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from aiayn.data.config import TokenizerConfig
from aiayn.data.special_tokens import NUM_SPECIAL_TOKENS
from aiayn.utils.hashing import sha256_file

if TYPE_CHECKING:
    from aiayn.models import ModelConfig

REPO_ROOT = Path(__file__).resolve().parents[1]

_EN = [
    "a",
    "the",
    "dog",
    "cat",
    "man",
    "woman",
    "child",
    "runs",
    "jumps",
    "sits",
    "on",
    "in",
    "under",
    "red",
    "blue",
    "green",
    "big",
    "small",
]
_DE = [
    "ein",
    "der",
    "hund",
    "katze",
    "mann",
    "frau",
    "kind",
    "läuft",
    "springt",
    "sitzt",
    "auf",
    "in",
    "unter",
    "rot",
    "blau",
    "grün",
    "groß",
    "klein",
]
assert len(_EN) == len(_DE), "word lists must align one-to-one"


def make_parallel_corpus(n: int, seed: int = 0) -> tuple[list[str], list[str]]:
    """Synthetic EN/DE-like sentence pairs (word-for-word aligned, includes umlauts/ß)."""
    rng = random.Random(seed)
    src, tgt = [], []
    for _ in range(n):
        idx = [rng.randrange(len(_EN)) for _ in range(rng.randint(3, 12))]
        src.append(" ".join(_EN[i] for i in idx) + " .")
        tgt.append(" ".join(_DE[i] for i in idx) + " .")
    return src, tgt


def tiny_tokenizer_config(vocab_size: int = 120) -> TokenizerConfig:
    return TokenizerConfig(
        model_type="bpe",
        vocab_size=vocab_size,
        character_coverage=1.0,
        normalization_rule_name="nmt_nfkc",
        input_sentence_size=0,
        num_threads=1,
        hard_vocab_limit=False,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )


class CharTokenizer:
    """Deterministic test double satisfying the Tokenizer protocol (1 char = 1 ID)."""

    def __init__(self, alphabet: str = "abcdefghijklmnopqrstuvwxyz .") -> None:
        self._ids = {c: i + NUM_SPECIAL_TOKENS for i, c in enumerate(alphabet)}
        self._chars = {i: c for c, i in self._ids.items()}

    @property
    def vocab_size(self) -> int:
        return NUM_SPECIAL_TOKENS + len(self._ids)

    @property
    def fingerprint(self) -> str:
        return "char-tokenizer"

    def encode(self, text: str) -> list[int]:
        return [self._ids.get(c, 1) for c in text]

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self._chars.get(i, "") for i in ids)


@pytest.fixture
def char_tokenizer() -> CharTokenizer:
    return CharTokenizer()


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Gzipped synthetic parallel corpus for train/valid/test, as the downloader expects."""
    root = tmp_path_factory.mktemp("corpus")
    sizes = {"train": 600, "valid": 60, "test": 60}
    for split_id, (split, n) in enumerate(sizes.items()):
        src, tgt = make_parallel_corpus(n, seed=split_id)
        for lang, lines in (("en", src), ("de", tgt)):
            with gzip.open(root / f"{split}.{lang}.gz", "wt", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
    return root


def corpus_downloads(corpus_dir: Path) -> list[dict[str, str]]:
    """DownloadSpec mappings (file:// URLs + real checksums) for ``corpus_dir``."""
    specs = []
    for split in ("train", "valid", "test"):
        for lang in ("en", "de"):
            path = corpus_dir / f"{split}.{lang}.gz"
            specs.append(
                {
                    "url": path.as_uri(),
                    "sha256": sha256_file(path),
                    "split": split,
                    "lang": lang,
                    "compression": "gzip",
                }
            )
    return specs


def small_model_config(**overrides: object) -> ModelConfig:
    """A fast ModelConfig for CPU tests (d=32, 2+2 layers, dropout 0 unless overridden)."""
    from aiayn.models import ModelConfig

    base: dict[str, object] = {
        "name": "test_small",
        "d_model": 32,
        "num_heads": 4,
        "d_ff": 64,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "dropout": 0.0,
        "max_positions": 64,
    }
    base.update(overrides)
    return ModelConfig(**base)  # type: ignore[arg-type]
