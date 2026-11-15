"""Build a split's DataLoader from a config and print statistics, shapes and masks.

Usage:
    python scripts/inspect_data.py --config configs/data/copy_task.yaml
    python scripts/inspect_data.py --config configs/data/multi30k.yaml --split valid
"""

from __future__ import annotations

import argparse
import time

from aiayn.data.collate import Batch
from aiayn.data.config import load_data_config
from aiayn.data.datasets import TranslationDataset
from aiayn.data.loading import build_dataloader, build_dataset, load_tokenizer
from aiayn.data.sampler import TokenBudgetBatchSampler
from aiayn.data.special_tokens import PAD_ID
from aiayn.data.tokenizer import SentencePieceTokenizer
from aiayn.modules.masking import (
    assert_valid_attention_mask,
    decoder_attention_mask,
    encoder_attention_mask,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    config = load_data_config(args.config, tuple(args.override))
    tokenizer: SentencePieceTokenizer | None = None
    if config.translation is not None:
        tokenizer = load_tokenizer(config)

    t0 = time.perf_counter()
    dataset = build_dataset(config, args.split, tokenizer=tokenizer)
    print(
        f"[{config.name}/{args.split}] {len(dataset):,} examples built in "
        f"{time.perf_counter() - t0:.2f}s"
    )
    if isinstance(dataset, TranslationDataset):
        print(f"  filter: {dataset.filter_stats}")
    print(
        f"  S: min {dataset.src_lengths.min()} mean {dataset.src_lengths.mean():.1f} "
        f"max {dataset.src_lengths.max()} | T: min {dataset.tgt_lengths.min()} "
        f"mean {dataset.tgt_lengths.mean():.1f} max {dataset.tgt_lengths.max()}"
    )

    loader = build_dataloader(config, dataset, args.split)
    sampler = loader.batch_sampler
    assert isinstance(sampler, TokenBudgetBatchSampler)
    print(f"  batches: {len(sampler):,}  padding efficiency: {sampler.padding_efficiency():.3f}")

    t0 = time.perf_counter()
    n_examples = 0
    first: Batch | None = None
    for batch in loader:
        n_examples += batch.batch_size
        first = first or batch
    print(f"  full pass: {n_examples:,} examples in {time.perf_counter() - t0:.2f}s")
    assert first is not None and n_examples == len(dataset)

    b = first
    src_mask = encoder_attention_mask(b.src, PAD_ID)
    tgt_mask = decoder_attention_mask(b.tgt_in, PAD_ID)
    assert_valid_attention_mask(src_mask, num_keys=b.src.shape[1])
    assert_valid_attention_mask(tgt_mask, num_queries=b.tgt_in.shape[1], num_keys=b.tgt_in.shape[1])
    print("  first batch shapes:")
    for name in ("src", "tgt_in", "tgt_out", "src_lengths", "tgt_lengths", "indices"):
        tensor = getattr(b, name)
        print(f"    {name:<12} {tuple(tensor.shape)} {tensor.dtype}")
    print(f"    src_mask     {tuple(src_mask.shape)} {src_mask.dtype}")
    print(f"    tgt_mask     {tuple(tgt_mask.shape)} {tgt_mask.dtype}")
    print(f"    tokens: src {b.num_src_tokens}, tgt {b.num_tgt_tokens}")

    row = 0
    print(f"  example (dataset index {int(b.indices[row])}):")
    for name in ("src", "tgt_in", "tgt_out"):
        ids = [i for i in getattr(b, name)[row].tolist() if i != PAD_ID]
        text = f"  -> {tokenizer.decode(ids)!r}" if tokenizer is not None else ""
        print(f"    {name:<8} {ids}{text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
