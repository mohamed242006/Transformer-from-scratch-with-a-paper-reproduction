"""IWSLT'14 recipe (ADR-023) on a synthetic archive with fairseq's layout: tag filtering,
segment extraction, clean-corpus rules, every-23rd validation split, test order, pinned
counts, idempotent preparation and the config schema."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Any

import pytest

from aiayn.data.config import ArchiveSource, DataConfig, load_data_config
from aiayn.data.iwslt import IWSLT14_TEST_SETS, clean_pair, read_iwslt14
from aiayn.data.prepare import ChecksumError, data_paths, prepare_translation_data
from aiayn.utils.config import ConfigError, config_to_dict, from_dict
from aiayn.utils.hashing import sha256_file

REPO = Path(__file__).resolve().parents[2]


def whitespace(text: str, lang: str) -> str:
    return " ".join(text.split())


def _add(tf: tarfile.TarFile, name: str, text: str) -> None:
    data = text.encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def make_archive(path: Path, pairs: int = 50) -> Path:
    header = [
        "<url>http://ted.com/talk</url>",
        "<keywords>talks, science</keywords>",
        "<talkid>1</talkid>",
    ]
    de = [*header, "<title>Ein Titel</title>", "<description>Eine Beschreibung</description>"]
    en = [*header, "<title>A title</title>", "<description>A description</description>"]
    for i in range(pairs):
        de.append(f"  Satz Nummer {i} hier .  ")
        en.append(f"Sentence number {i} here .")
    de += ["kurz", " ".join(["wort"] * 176)]  # ratio 1:4 -> dropped; 176 words -> dropped
    en += ["this is far too long", " ".join(["word"] * 176)]
    with tarfile.open(path, "w:gz") as tf:
        _add(tf, "de-en/train.tags.de-en.de", "\n".join(de) + "\n")
        _add(tf, "de-en/train.tags.de-en.en", "\n".join(en) + "\n")
        for k, name in enumerate(IWSLT14_TEST_SETS):
            for lang in ("de", "en"):
                segs = "\n".join(
                    f'<seg id="{j}"> {name[-7:]} {lang} {j} </seg>' for j in range(1, k + 3)
                )
                xml = f'<mteval>\n<doc docid="1">\n<url>x</url>\n{segs}\n</doc>\n</mteval>\n'
                _add(tf, f"de-en/{name}.de-en.{lang}.xml", xml)
    return path


def test_clean_pair_rules() -> None:
    assert clean_pair("a b c", "a b")  # ratio 1.5 is allowed
    assert not clean_pair("a b c d", "a b")  # ratio 2
    assert not clean_pair("", "a")
    assert not clean_pair(" ".join(["w"] * 176), " ".join(["w"] * 170))
    assert clean_pair("a | b", "a b")  # '|' is removed before counting (clean-corpus-n.perl)


def test_recipe_filters_tags_cleans_and_splits(tmp_path: Path) -> None:
    splits = read_iwslt14(make_archive(tmp_path / "de-en.tgz"), tokenize=whitespace)
    train, valid, test = (splits.splits[s] for s in ("train", "valid", "test"))
    # 1 title + 1 description line + 50 sentences kept; the two bad pairs dropped.
    kept = 52
    assert len(valid["de"]) == kept // 23 and len(train["de"]) == kept - kept // 23
    assert train["de"][0] == "Ein Titel" and train["en"][1] == "A description"
    assert all("<" not in line for line in train["de"] + train["en"])
    assert train["de"][2] == "Satz Nummer 0 hier ."  # raw text, outer whitespace stripped
    assert valid["en"][0] == "Sentence number 20 here ."  # kept pair #23
    # Test = dev2010, TEDX.dev2012, tst2010, tst2011, tst2012 in order, tags removed.
    assert len(test["de"]) == sum(k + 2 for k in range(5)) == len(test["en"])
    assert test["en"][0] == "dev2010 en 1" and test["en"][-1] == "tst2012 en 6"


def test_misaligned_archives_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.tgz"
    with tarfile.open(path, "w:gz") as tf:
        _add(tf, "de-en/train.tags.de-en.de", "a\nb\n")
        _add(tf, "de-en/train.tags.de-en.en", "a\n")
    with pytest.raises(ValueError, match="misaligned"):
        read_iwslt14(path, tokenize=whitespace)


def archive_config(tmp_path: Path, archive: Path, **counts: int) -> DataConfig:
    raw = config_to_dict(load_data_config(REPO / "configs" / "data" / "iwslt14.yaml"))
    raw["translation"]["root"] = str(tmp_path / "data")
    raw["translation"]["archive"].update(
        url=archive.resolve().as_uri(), sha256=sha256_file(archive), **counts
    )
    return from_dict(DataConfig, raw)


def test_prepare_writes_splits_and_is_idempotent(tmp_path: Path) -> None:
    archive = make_archive(tmp_path / "de-en.tgz")
    config = archive_config(tmp_path, archive, train_lines=50, valid_lines=2, test_lines=20)
    paths = prepare_translation_data(config)
    manifest = json.loads((paths.processed_dir / "MANIFEST.json").read_text("utf-8"))
    assert manifest["counts"] == {"train": 50, "valid": 2, "test": 20}
    assert set(manifest["files"]) == {
        f"{s}.{lang}" for s in ("train", "valid", "test") for lang in ("de", "en")
    }
    stamp = paths.text_file("train", "de").stat().st_mtime_ns
    prepare_translation_data(config)  # manifest matches: nothing is rewritten
    assert paths.text_file("train", "de").stat().st_mtime_ns == stamp
    paths.text_file("test", "en").write_text("tampered\n", encoding="utf-8")
    prepare_translation_data(config)  # hash mismatch: rebuilt
    assert paths.text_file("test", "en").read_text("utf-8").startswith("dev2010 en 1")


def test_unexpected_counts_fail_loudly(tmp_path: Path) -> None:
    archive = make_archive(tmp_path / "de-en.tgz")
    config = archive_config(tmp_path, archive, train_lines=160215, valid_lines=0, test_lines=0)
    with pytest.raises(ChecksumError, match="split sizes"):
        prepare_translation_data(config)


# ----------------------------------------------------------------------------- config schema
def test_iwslt_config_and_schema_evolution() -> None:
    iwslt = load_data_config(REPO / "configs" / "data" / "iwslt14.yaml")
    assert iwslt.translation is not None and iwslt.translation.archive is not None
    assert (iwslt.translation.src_lang, iwslt.translation.tgt_lang) == ("de", "en")
    assert iwslt.translation.tokenizer.vocab_size == 10000
    assert config_to_dict(iwslt)["translation"]["archive"]["recipe"] == "iwslt14_fairseq"
    # The new optional field is omitted while unset: Multi30k serializes (and hashes) as before.
    multi30k = config_to_dict(load_data_config(REPO / "configs" / "data" / "multi30k.yaml"))
    assert "archive" not in multi30k["translation"]
    assert data_paths(iwslt).processed_dir.name == "iwslt14"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            {
                "downloads": [
                    {"split": "train", "lang": "de", "url": "https://x/y", "sha256": "0" * 64}
                ]
            },
            "not both",
        ),
        ({"src_lang": "fr"}, "de and en"),
    ],
)
def test_archive_config_validation(change: dict[str, Any], message: str) -> None:
    raw = config_to_dict(load_data_config(REPO / "configs" / "data" / "iwslt14.yaml"))
    raw["translation"].update(change)
    with pytest.raises(ConfigError, match=message):
        from_dict(DataConfig, raw)
    with pytest.raises(ValueError, match="https"):
        ArchiveSource(url="http://x", sha256="0" * 64, recipe="iwslt14_fairseq")
