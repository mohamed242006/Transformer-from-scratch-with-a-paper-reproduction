"""SentencePiece wrapper (ADR-005). Trains tiny models on synthetic text in a temp dir."""

from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path

import pytest
import sentencepiece as spm

from aiayn.data.special_tokens import BOS_ID, EOS_ID, PAD_ID, UNK_ID
from aiayn.data.tokenizer import SentencePieceTokenizer, Tokenizer, train_sentencepiece
from aiayn.utils.hashing import sha256_file
from tests.conftest import make_parallel_corpus, tiny_tokenizer_config


@pytest.fixture(scope="module")
def train_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    src, tgt = make_parallel_corpus(800, seed=1)
    path = tmp_path_factory.mktemp("tok") / "train.txt"
    path.write_text("\n".join(src + tgt) + "\n", encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def tokenizer(train_file: Path) -> SentencePieceTokenizer:
    model = train_sentencepiece([train_file], train_file.parent / "m", tiny_tokenizer_config())
    return SentencePieceTokenizer(model)


def test_special_ids_and_protocol(tokenizer: SentencePieceTokenizer) -> None:
    assert isinstance(tokenizer, Tokenizer)
    assert tokenizer.id_to_piece(PAD_ID) == "<pad>"
    assert tokenizer.id_to_piece(UNK_ID) == "<unk>"
    assert tokenizer.id_to_piece(BOS_ID) == "<s>"
    assert tokenizer.id_to_piece(EOS_ID) == "</s>"
    assert tokenizer.vocab_size <= 120


@pytest.mark.parametrize("text", ["the dog runs on a big red cat .", "der hund läuft groß ."])
def test_round_trip(tokenizer: SentencePieceTokenizer, text: str) -> None:
    ids = tokenizer.encode(text)
    assert ids and all(i not in (PAD_ID, BOS_ID, EOS_ID) for i in ids)  # no specials added
    assert tokenizer.decode(ids) == text


def test_batch_encoding_matches_single(tokenizer: SentencePieceTokenizer) -> None:
    texts = ["a dog .", "ein hund .", ""]
    assert tokenizer.encode_batch(texts) == [tokenizer.encode(t) for t in texts]


def test_unseen_characters_map_to_unk(tokenizer: SentencePieceTokenizer) -> None:
    assert UNK_ID in tokenizer.encode("日本")


def test_decode_drops_structural_specials(tokenizer: SentencePieceTokenizer) -> None:
    ids = tokenizer.encode("a cat .")
    assert tokenizer.decode([BOS_ID, *ids, EOS_ID, PAD_ID, PAD_ID]) == "a cat ."


def test_fingerprint_is_a_content_hash(
    tokenizer: SentencePieceTokenizer, train_file: Path, tmp_path: Path
) -> None:
    # The file hash depends on the output path embedded by SentencePiece...
    assert len(tokenizer.fingerprint) == 64
    assert tokenizer.fingerprint != sha256_file(tokenizer.path)
    # ...while a different vocabulary or normalization changes the fingerprint.
    smaller = train_sentencepiece([train_file], tmp_path / "s", tiny_tokenizer_config(80))
    assert SentencePieceTokenizer(smaller).fingerprint != tokenizer.fingerprint
    identity = replace(tiny_tokenizer_config(), normalization_rule_name="identity")
    other_norm = train_sentencepiece([train_file], tmp_path / "n", identity)
    assert SentencePieceTokenizer(other_norm).fingerprint != tokenizer.fingerprint


def test_pickle_round_trip(tokenizer: SentencePieceTokenizer) -> None:
    clone = pickle.loads(pickle.dumps(tokenizer))
    assert clone.encode("the cat sits .") == tokenizer.encode("the cat sits .")
    assert clone.fingerprint == tokenizer.fingerprint


def test_training_is_deterministic_across_output_locations(
    train_file: Path, tmp_path: Path
) -> None:
    a = SentencePieceTokenizer(
        train_sentencepiece([train_file], tmp_path / "a", tiny_tokenizer_config())
    )
    b = SentencePieceTokenizer(
        train_sentencepiece([train_file], tmp_path / "sub" / "b", tiny_tokenizer_config())
    )
    assert a.fingerprint == b.fingerprint
    text = "the small child jumps under a green dog ."
    assert a.encode(text) == b.encode(text)


def test_rejects_models_with_foreign_special_ids(train_file: Path, tmp_path: Path) -> None:
    # SentencePiece defaults: unk=0, bos=1, eos=2, pad disabled.
    spm.SentencePieceTrainer.train(
        input=str(train_file), model_prefix=str(tmp_path / "x"), vocab_size=40, minloglevel=2
    )
    with pytest.raises(ValueError, match="special IDs"):
        SentencePieceTokenizer(tmp_path / "x.model")


def test_training_input_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        train_sentencepiece([], tmp_path / "m", tiny_tokenizer_config())
    with pytest.raises(FileNotFoundError):
        train_sentencepiece([tmp_path / "missing.txt"], tmp_path / "m", tiny_tokenizer_config())
