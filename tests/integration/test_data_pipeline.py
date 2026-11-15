"""End-to-end data pipeline: config -> dataset -> sampler -> DataLoader -> Batch -> masks."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest
import torch

from aiayn.data.collate import Batch
from aiayn.data.config import DataConfig, load_data_config
from aiayn.data.loading import build_dataloader, build_dataset, load_tokenizer, train_tokenizer
from aiayn.data.prepare import prepare_translation_data
from aiayn.data.sampler import TokenBudgetBatchSampler
from aiayn.data.special_tokens import BOS_ID, EOS_ID, PAD_ID
from aiayn.modules.masking import (
    assert_valid_attention_mask,
    decoder_attention_mask,
    encoder_attention_mask,
)
from tests.conftest import REPO_ROOT
from tests.unit.test_prepare import translation_config

pytestmark = pytest.mark.integration

COPY_CONFIG = REPO_ROOT / "configs" / "data" / "copy_task.yaml"


def check_batch(batch: Batch, max_tokens: int) -> None:
    b, s = batch.src.shape
    t = batch.tgt_in.shape[1]
    assert batch.tgt_out.shape == (b, t)
    assert b * max(s, t) <= max_tokens
    src_mask = encoder_attention_mask(batch.src, PAD_ID)
    tgt_mask = decoder_attention_mask(batch.tgt_in, PAD_ID)
    assert src_mask.shape == (b, 1, 1, s) and tgt_mask.shape == (b, 1, t, t)
    assert_valid_attention_mask(src_mask, num_keys=s)
    assert_valid_attention_mask(tgt_mask, num_queries=t, num_keys=t)
    assert (batch.tgt_in[:, 0] == BOS_ID).all() and not (batch.tgt_in == EOS_ID).any()


def run_epoch(config: DataConfig, split: str, **kwargs: object) -> list[Batch]:
    dataset = build_dataset(config, split, **kwargs)  # type: ignore[arg-type]
    loader = build_dataloader(config, dataset, split)  # type: ignore[arg-type]
    batches = list(loader)
    seen = sorted(i for b in batches for i in b.indices.tolist())
    assert seen == list(range(len(dataset))), "every example exactly once per epoch"
    return batches


@pytest.mark.parametrize("split", ["train", "valid", "test"])
def test_copy_task_pipeline(split: str) -> None:
    config = load_data_config(COPY_CONFIG, ("copy.num_train=2000",))
    for batch in run_epoch(config, split):
        check_batch(batch, config.batching.max_tokens)
        content_in = batch.tgt_in[:, 1:]
        content_src = batch.src[:, : content_in.shape[1]]
        # Copy task: target content equals source content (where target is not padding).
        real = (content_in != PAD_ID) & (content_in != EOS_ID)
        assert torch.equal(content_in[real], content_src[real])


def test_copy_task_epochs_are_reproducible_and_distinct() -> None:
    config = load_data_config(COPY_CONFIG, ("copy.num_train=1000",))
    dataset = build_dataset(config, "train")
    loader = build_dataloader(config, dataset, "train")
    sampler = loader.batch_sampler
    assert isinstance(sampler, TokenBudgetBatchSampler)
    epochs = []
    for epoch in (0, 1, 0):
        sampler.set_epoch(epoch)
        epochs.append([b.indices.tolist() for b in loader])
    assert epochs[0] == epochs[2] and epochs[0] != epochs[1]


def test_translation_pipeline_end_to_end(corpus_dir: Path, tmp_path: Path) -> None:
    config = translation_config(corpus_dir, tmp_path / "data")
    prepare_translation_data(config)
    model_path = train_tokenizer(config)
    assert model_path.is_file()
    tokenizer = load_tokenizer(config)
    for split in ("train", "valid"):
        dataset = build_dataset(config, split, tokenizer=tokenizer)
        assert dataset.filter_stats.dropped == 0  # type: ignore[union-attr]
        for batch in run_epoch(config, split, tokenizer=tokenizer):
            check_batch(batch, config.batching.max_tokens)
    # Decoding a batch row recovers the original sentence.
    valid = build_dataset(config, "valid", tokenizer=tokenizer)
    ex = valid[0]
    src_text = (
        (tmp_path / "data" / "processed" / "synthetic" / "valid.en")
        .read_text(encoding="utf-8")
        .split("\n")[0]
    )
    assert tokenizer.decode(ex.src.tolist()) == src_text


def test_pipeline_objects_pickle_for_spawned_workers() -> None:
    config = load_data_config(COPY_CONFIG, ("copy.num_train=500",))
    dataset = build_dataset(config, "train")
    loader = build_dataloader(config, dataset, "train")
    for obj in (dataset, loader.batch_sampler, loader.collate_fn):
        pickle.loads(pickle.dumps(obj))


@pytest.mark.slow
def test_multiprocess_loading_matches_single_process() -> None:
    """Spawned workers (the only start method on Windows) yield identical batches."""
    base = load_data_config(COPY_CONFIG, ("copy.num_train=600",))
    multi = load_data_config(COPY_CONFIG, ("copy.num_train=600", "batching.num_workers=2"))
    single_batches = run_epoch(base, "train")
    multi_batches = run_epoch(multi, "train")
    assert len(single_batches) == len(multi_batches)
    for a, b in zip(single_batches, multi_batches, strict=True):
        assert torch.equal(a.src, b.src) and torch.equal(a.tgt_out, b.tgt_out)
