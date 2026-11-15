from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from aiayn.data.datasets import TranslationDataset, read_lines, read_parallel_text
from aiayn.data.special_tokens import BOS_ID, EOS_ID
from tests.conftest import CharTokenizer


def test_read_lines_splits_only_on_newline(tmp_path: Path) -> None:
    lf, crlf = chr(10), chr(13) + chr(10)
    ls = chr(0x2028)  # LINE SEPARATOR: str.splitlines() would split here
    path = tmp_path / "t.txt"
    path.write_bytes(("one" + crlf + "two" + ls + "still two" + lf + "three" + lf).encode())
    assert read_lines(path) == ["one", "two" + ls + "still two", "three"]


def test_read_lines_without_trailing_newline_and_empty(tmp_path: Path) -> None:
    (tmp_path / "a").write_text("x\ny", encoding="utf-8")
    (tmp_path / "b").write_text("", encoding="utf-8")
    assert read_lines(tmp_path / "a") == ["x", "y"]
    assert read_lines(tmp_path / "b") == []


def test_parallel_files_must_align(tmp_path: Path) -> None:
    (tmp_path / "s").write_text("a\nb\n", encoding="utf-8")
    (tmp_path / "t").write_text("a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="misaligned"):
        read_parallel_text(tmp_path / "s", tmp_path / "t")


def test_framing_storage_and_lengths(char_tokenizer: CharTokenizer) -> None:
    ds = TranslationDataset(["ab", "abc"], ["x", "xyz."], char_tokenizer, max_len=10)
    assert len(ds) == 2
    ex = ds[1]
    assert ex.src.tolist() == [*char_tokenizer.encode("abc"), EOS_ID]
    assert ex.tgt.tolist() == [BOS_ID, *char_tokenizer.encode("xyz."), EOS_ID]
    np.testing.assert_array_equal(ds.src_lengths, [3, 4])  # S = n + 1
    np.testing.assert_array_equal(ds.tgt_lengths, [2, 5])  # T = m + 1
    assert ds.num_tokens == (7, 9)
    assert ds.tokenizer_fingerprint == "char-tokenizer"


def test_filtering_statistics(char_tokenizer: CharTokenizer) -> None:
    src = ["ok", "", "toolongsource", "a", "abcd"]
    tgt = ["ok", "x", "y", "", "a"]
    ds = TranslationDataset(src, tgt, char_tokenizer, max_len=6, max_len_ratio=3.0)
    stats = ds.filter_stats
    assert (stats.total, stats.kept, stats.too_short, stats.too_long, stats.bad_ratio) == (
        5,
        1,
        2,
        1,
        1,
    )
    assert stats.dropped == 4
    assert ds.line_number(0) == 0


def test_length_limit_is_inclusive(char_tokenizer: CharTokenizer) -> None:
    # n = 5 -> S = 6 == max_len (kept); n = 6 -> S = 7 (dropped)
    ds = TranslationDataset(["aaaaa", "aaaaaa"], ["b", "b"], char_tokenizer, max_len=6)
    assert len(ds) == 1 and int(ds.src_lengths.max()) == 6


def test_all_filtered_is_an_error(char_tokenizer: CharTokenizer) -> None:
    with pytest.raises(ValueError, match="survived"):
        TranslationDataset([""], [""], char_tokenizer, max_len=5)


def test_bounds_and_pickle(char_tokenizer: CharTokenizer) -> None:
    ds = TranslationDataset(["ab"], ["cd"], char_tokenizer, max_len=10)
    with pytest.raises(IndexError):
        ds[1]
    clone = pickle.loads(pickle.dumps(ds))
    assert clone[0].src.tolist() == ds[0].src.tolist()


def test_examples_are_views_not_copies(char_tokenizer: CharTokenizer) -> None:
    ds = TranslationDataset(["abc"], ["de"], char_tokenizer, max_len=10)
    assert ds[0].src.base is not None  # slice of the flat storage
