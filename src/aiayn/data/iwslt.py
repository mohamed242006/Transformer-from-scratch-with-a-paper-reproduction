"""IWSLT'14 DE-EN from fairseq's archive, split exactly like ``prepare-iwslt14.sh`` (ADR-023).

fairseq's script (examples/translation/prepare-iwslt14.sh) is the de facto definition of
the benchmark's splits. This module reproduces its **membership** decisions while keeping
the **raw, cased, untokenized** text (protocol A: our SentencePiece model sees raw text):

* train source: ``train.tags.de-en.{de,en}``; lines containing ``<url>``, ``<talkid>`` or
  ``<keywords>`` are dropped and ``<title>``/``<description>`` tags removed (per file).
* cleaning (``clean-corpus-n.perl -ratio 1.5 … 1 175``) is decided on the Moses-tokenized
  text (``sacremoses``, escaping on, as ``tokenizer.perl``'s default), exactly as fairseq
  does; the kept pairs are written in raw form.
* valid = every 23rd kept pair (``NR % 23 == 0``), train = the rest.
* test = segments of dev2010, TEDX.dev2012, tst2010, tst2011, tst2012, in that order.

Expected counts (fairseq): 160,239 / 7,283 / 6,750. The archive is read member by member
(``tarfile.extractfile``): nothing is extracted to disk, so archive paths cannot escape.
"""

from __future__ import annotations

import re
import tarfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

__all__ = ["IWSLT14_TEST_SETS", "IwsltSplits", "clean_pair", "read_iwslt14"]

LANGS = ("de", "en")
IWSLT14_TEST_SETS = (
    "IWSLT14.TED.dev2010",
    "IWSLT14.TEDX.dev2012",
    "IWSLT14.TED.tst2010",
    "IWSLT14.TED.tst2011",
    "IWSLT14.TED.tst2012",
)
_TRAIN_DROP = ("<url>", "<talkid>", "<keywords>")
_TRAIN_STRIP = ("<title>", "</title>", "<description>", "</description>")
_SEG_OPEN = re.compile(r'<seg id="[0-9]*">\s*')
_SEG_CLOSE = re.compile(r"\s*</seg>\s*")
_WS = re.compile(r"\s+")
CLEAN_RATIO = 1.5
CLEAN_MIN, CLEAN_MAX = 1, 175
VALID_EVERY = 23

Tokenize = Callable[[str, str], str]


@dataclass(frozen=True)
class IwsltSplits:
    """Raw text per split and language: ``splits[split][lang]`` -> list of lines."""

    splits: dict[str, dict[str, list[str]]]

    def counts(self) -> dict[str, int]:
        return {s: len(v["de"]) for s, v in self.splits.items()}


def _member(tf: tarfile.TarFile, name: str) -> list[str]:
    info = tf.getmember(name)
    if not info.isfile():
        raise ValueError(f"{name}: not a regular file in the archive")
    handle = tf.extractfile(info)
    assert handle is not None
    text = handle.read().decode("utf-8")
    if text.endswith("\n"):
        text = text[:-1]
    return [line.removesuffix("\r") for line in text.split("\n")] if text else []


def _train_lines(lines: Iterable[str]) -> list[str]:
    out = []
    for line in lines:
        if any(tag in line for tag in _TRAIN_DROP):
            continue
        for tag in _TRAIN_STRIP:
            line = line.replace(tag, "")
        out.append(line.strip())
    return out


def _segments(lines: Iterable[str]) -> list[str]:
    out = []
    for line in lines:
        if "<seg id" not in line:
            continue
        line = _SEG_OPEN.sub("", line)
        line = _SEG_CLOSE.sub("", line)
        out.append(line.strip())
    return out


def _word_count(tokenized: str) -> int:
    """``clean-corpus-n.perl``: drop ``|``, collapse spaces, split on single spaces."""
    text = _WS.sub(" ", tokenized.replace("|", "")).strip()
    return len(text.split(" ")) if text else 0


def clean_pair(de_tok: str, en_tok: str) -> bool:
    """True if the tokenized pair survives ``clean-corpus-n.perl -ratio 1.5 … 1 175``."""
    a, b = _word_count(de_tok), _word_count(en_tok)
    if a == 0 or b == 0:
        return False
    if not (CLEAN_MIN <= a <= CLEAN_MAX and CLEAN_MIN <= b <= CLEAN_MAX):
        return False
    return a / b <= CLEAN_RATIO and b / a <= CLEAN_RATIO


def moses_tokenizer() -> Tokenize:
    """``tokenizer.perl -l <lang>`` equivalent (escaping on, no aggressive dash split)."""
    from sacremoses import MosesTokenizer

    cache: dict[str, MosesTokenizer] = {}

    def tokenize(text: str, lang: str) -> str:
        if lang not in cache:
            cache[lang] = MosesTokenizer(lang=lang)
        return str(cache[lang].tokenize(text, escape=True, return_str=True))

    return tokenize


def read_iwslt14(archive: Path, *, tokenize: Tokenize | None = None) -> IwsltSplits:
    """Build raw train/valid/test from fairseq's ``de-en.tgz`` (see module docstring)."""
    tokenize = tokenize or moses_tokenizer()
    with tarfile.open(archive, "r:gz") as tf:
        train = {
            lang: _train_lines(_member(tf, f"de-en/train.tags.de-en.{lang}")) for lang in LANGS
        }
        if len(train["de"]) != len(train["en"]):
            raise ValueError(
                f"train sides misaligned after tag filtering: {len(train['de'])} de vs "
                f"{len(train['en'])} en"
            )
        test: dict[str, list[str]] = {lang: [] for lang in LANGS}
        for name in IWSLT14_TEST_SETS:
            parts = {
                lang: _segments(_member(tf, f"de-en/{name}.de-en.{lang}.xml")) for lang in LANGS
            }
            if len(parts["de"]) != len(parts["en"]):
                raise ValueError(f"{name}: {len(parts['de'])} de vs {len(parts['en'])} en segments")
            for lang in LANGS:
                test[lang] += parts[lang]
    kept = [
        i
        for i, (de, en) in enumerate(zip(train["de"], train["en"], strict=True))
        if clean_pair(tokenize(de, "de"), tokenize(en, "en"))
    ]
    splits: dict[str, dict[str, list[str]]] = {
        "train": {lang: [] for lang in LANGS},
        "valid": {lang: [] for lang in LANGS},
        "test": test,
    }
    for position, index in enumerate(kept, start=1):
        target = "valid" if position % VALID_EVERY == 0 else "train"
        for lang in LANGS:
            splits[target][lang].append(train[lang][index])
    return IwsltSplits(splits)
