"""Config-driven construction of tokenizers, datasets, samplers and DataLoaders."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from aiayn.data.collate import Collator
from aiayn.data.config import SPLITS, DataConfig, Split
from aiayn.data.copy_task import CopyTaskDataset
from aiayn.data.datasets import SequencePairDataset, TranslationDataset, read_parallel_text
from aiayn.data.example import Example
from aiayn.data.prepare import data_paths
from aiayn.data.sampler import TokenBudgetBatchSampler
from aiayn.data.special_tokens import PAD_ID
from aiayn.data.tokenizer import SentencePieceTokenizer, Tokenizer, train_sentencepiece

__all__ = [
    "build_batch_sampler",
    "build_dataloader",
    "build_dataset",
    "load_tokenizer",
    "train_tokenizer",
]


def train_tokenizer(config: DataConfig, base: str | Path | None = None) -> Path:
    """Train the joint tokenizer on the **training split** of both languages (ADR-005)."""
    paths = data_paths(config, base)
    assert config.translation is not None
    t = config.translation
    inputs = [paths.text_file("train", t.src_lang), paths.text_file("train", t.tgt_lang)]
    return train_sentencepiece(inputs, paths.tokenizer_prefix, t.tokenizer)


def load_tokenizer(config: DataConfig, base: str | Path | None = None) -> SentencePieceTokenizer:
    """Load the trained tokenizer and check it matches the configured vocabulary size."""
    paths = data_paths(config, base)
    model = paths.tokenizer_model
    if not model.is_file():
        raise FileNotFoundError(f"{model} not found; run scripts/train_tokenizer.py first")
    tokenizer = SentencePieceTokenizer(model)
    assert config.translation is not None
    expected = config.translation.tokenizer.vocab_size
    if config.translation.tokenizer.hard_vocab_limit and tokenizer.vocab_size != expected:
        raise ValueError(f"{model}: vocab_size {tokenizer.vocab_size} != configured {expected}")
    return tokenizer


def build_dataset(
    config: DataConfig,
    split: Split,
    *,
    tokenizer: Tokenizer | None = None,
    base: str | Path | None = None,
) -> CopyTaskDataset | TranslationDataset:
    """Build the dataset for ``split``. Translation datasets need a ``tokenizer``."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    if config.copy is not None:
        c = config.copy
        sizes = {"train": c.num_train, "valid": c.num_valid, "test": c.num_test}
        return CopyTaskDataset(
            num_examples=sizes[split],
            vocab_size=c.vocab_size,
            min_len=c.min_len,
            max_len=c.max_len,
            seed=config.seed,
            stream=SPLITS.index(split),
            mode=c.mode,
        )

    assert config.translation is not None
    if tokenizer is None:
        raise ValueError("translation datasets require a tokenizer")
    t = config.translation
    paths = data_paths(config, base)
    src, tgt = read_parallel_text(
        paths.text_file(split, t.src_lang), paths.text_file(split, t.tgt_lang)
    )
    return TranslationDataset(
        src,
        tgt,
        tokenizer,
        max_len=t.filter.max_len,
        min_len=t.filter.min_len,
        max_len_ratio=t.filter.max_len_ratio,
    )


def build_batch_sampler(
    config: DataConfig, dataset: SequencePairDataset, split: Split
) -> TokenBudgetBatchSampler:
    """Training split: shuffled pools. Other splits: deterministic length order."""
    b = config.batching
    train = split == "train"
    max_tokens = b.max_tokens if train or b.eval_max_tokens is None else b.eval_max_tokens
    return TokenBudgetBatchSampler(
        dataset.src_lengths,
        dataset.tgt_lengths,
        max_tokens=max_tokens,
        max_sentences=b.max_sentences,
        shuffle=train,
        pool_size=b.pool_size,
        seed=config.seed,
    )


def build_dataloader(
    config: DataConfig,
    dataset: SequencePairDataset,
    split: Split,
    *,
    pin_memory: bool = False,
) -> DataLoader[Example]:
    """DataLoader yielding ``Batch`` objects.

    Call ``loader.batch_sampler.set_epoch(e)`` before each training epoch for
    reproducible, epoch-varying batches.
    """
    b = config.batching
    return DataLoader(
        dataset,  # type: ignore[arg-type]  # structural Dataset (Protocol), see SequencePairDataset
        batch_sampler=build_batch_sampler(config, dataset, split),
        collate_fn=Collator(PAD_ID, b.pad_to_multiple_of),
        num_workers=b.num_workers,
        pin_memory=pin_memory,
        persistent_workers=b.num_workers > 0,
    )
