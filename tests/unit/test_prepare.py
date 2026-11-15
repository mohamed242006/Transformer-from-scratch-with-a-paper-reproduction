"""Download / checksum / extraction, exercised offline through file:// URLs."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from aiayn.data.config import DataConfig
from aiayn.data.datasets import read_lines
from aiayn.data.prepare import ChecksumError, data_paths, download, prepare_translation_data
from aiayn.utils.config import from_dict
from aiayn.utils.hashing import sha256_bytes, sha256_file
from tests.conftest import corpus_downloads, tiny_tokenizer_config


def translation_config(corpus_dir: Path, root: Path) -> DataConfig:
    tok = tiny_tokenizer_config()
    raw = {
        "name": "synthetic",
        "task": "translation",
        "translation": {
            "src_lang": "en",
            "tgt_lang": "de",
            "root": str(root),
            "downloads": corpus_downloads(corpus_dir),
            "filter": {"max_len": 64},
            "tokenizer": {k: getattr(tok, k) for k in tok.__dataclass_fields__},
        },
    }
    return from_dict(DataConfig, raw)


def test_sha256_helpers(tmp_path: Path) -> None:
    path = tmp_path / "f"
    path.write_bytes(b"abc")
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert sha256_file(path) == sha256_bytes(b"abc") == expected


def test_download_verifies_and_caches(tmp_path: Path) -> None:
    source = tmp_path / "src.bin"
    source.write_bytes(b"payload")
    dest = tmp_path / "out" / "dest.bin"
    digest = sha256_bytes(b"payload")
    assert download(source.as_uri(), dest, sha256=digest).read_bytes() == b"payload"
    source.unlink()  # a cached, verified file is reused without re-downloading
    assert download(source.as_uri(), dest, sha256=digest) == dest


def test_checksum_mismatch_leaves_nothing_behind(tmp_path: Path) -> None:
    source = tmp_path / "src.bin"
    source.write_bytes(b"tampered")
    dest = tmp_path / "dest.bin"
    with pytest.raises(ChecksumError):
        download(source.as_uri(), dest, sha256="0" * 64)
    assert not dest.exists() and not (tmp_path / "dest.bin.part").exists()


def test_disallowed_scheme(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="scheme"):
        download("http://example.com/x", tmp_path / "x", sha256="0" * 64)


def test_prepare_translation_data_end_to_end(corpus_dir: Path, tmp_path: Path) -> None:
    config = translation_config(corpus_dir, tmp_path / "data")
    paths = prepare_translation_data(config)
    assert paths == data_paths(config)
    for split, n in (("train", 600), ("valid", 60), ("test", 60)):
        assert len(read_lines(paths.text_file(split, "en"))) == n
        assert len(read_lines(paths.text_file(split, "de"))) == n
    assert paths.tokenizer_model.name == "synthetic_bpe120.model"


def test_config_requires_all_split_language_pairs(corpus_dir: Path, tmp_path: Path) -> None:
    config = translation_config(corpus_dir, tmp_path)
    assert config.translation is not None
    raw = {
        "name": "synthetic",
        "task": "translation",
        "translation": {
            "src_lang": "en",
            "tgt_lang": "de",
            "downloads": copy.deepcopy(corpus_downloads(corpus_dir))[:-1],
            "filter": {"max_len": 64},
            "tokenizer": {
                k: getattr(config.translation.tokenizer, k)
                for k in config.translation.tokenizer.__dataclass_fields__
            },
        },
    }
    with pytest.raises(ValueError, match="missing"):
        from_dict(DataConfig, raw)


def test_mirror_dirs_allow_offline_preparation(tmp_path: Path) -> None:
    """Kaggle without internet: archives come from /kaggle/input-style mirrors (ADR-013)."""
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / "train.en.gz").write_bytes(b"archive")
    url = "https://unreachable.invalid/data/train.en.gz"  # never contacted
    dest = tmp_path / "raw" / "train.en.gz"
    out = download(url, dest, sha256=sha256_bytes(b"archive"), mirror_dirs=[tmp_path, mirror])
    assert out.read_bytes() == b"archive"


def test_mirror_files_are_checksum_verified(tmp_path: Path) -> None:
    (tmp_path / "train.en.gz").write_bytes(b"corrupted")
    with pytest.raises(ChecksumError):
        download(
            "https://unreachable.invalid/train.en.gz",
            tmp_path / "out" / "train.en.gz",
            sha256=sha256_bytes(b"original"),
            mirror_dirs=[tmp_path],
        )
