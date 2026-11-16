"""Evaluation pipeline: decode a split of a run's dataset and score it (ADR-006, ADR-019).

References are the **raw, untokenized, unfiltered** lines of the split, in corpus order,
so scores are comparable with sacreBLEU numbers elsewhere. Outputs go to
``<run_dir>/eval/``: ``<tag>.hyp`` (one line per source sentence) and ``<tag>.json``
(scores, signatures, settings, timing).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from aiayn.data.config import Split
from aiayn.data.copy_task import CopyTaskDataset
from aiayn.data.datasets import read_parallel_text
from aiayn.data.prepare import data_paths, prepare_translation_data
from aiayn.evaluation.metrics import bleu_by_length, corpus_scores
from aiayn.inference.decoding import DecodeConfig
from aiayn.inference.translator import Translator, load_run

__all__ = ["evaluate_run"]


def _copy_split(info: Any, split: Split) -> tuple[list[list[int]], list[str]]:
    c = info.data_config.copy
    sizes = {"train": c.num_train, "valid": c.num_valid, "test": c.num_test}
    dataset = CopyTaskDataset(
        num_examples=sizes[split],
        vocab_size=c.vocab_size,
        min_len=c.min_len,
        max_len=c.max_len,
        seed=info.data_config.seed,
        stream=("train", "valid", "test").index(split),
        mode=c.mode,
    )
    sources, references = [], []
    for i in range(len(dataset)):
        ex = dataset[i]
        sources.append(ex.src[:-1].tolist())
        references.append(" ".join(map(str, ex.tgt[1:-1].tolist())))
    return sources, references


def evaluate_run(
    run_dir: str | Path,
    *,
    split: Split = "test",
    checkpoint: str = "best",
    decode: DecodeConfig | None = None,
    average_last: int | None = None,
    device: str = "auto",
    max_tokens: int = 4096,
    prepare_data: bool = False,
    write: bool = True,
) -> dict[str, Any]:
    """Translate ``split`` with the chosen checkpoint and return the scores."""
    decode = decode or DecodeConfig()
    run_dir = Path(run_dir)
    info = load_run(run_dir)
    translator = Translator.from_run(
        run_dir,
        checkpoint=checkpoint,
        average_last=average_last,
        device=device,
        max_tokens=max_tokens,
    )
    start = time.perf_counter()
    if info.data_config.translation is not None:
        t = info.data_config.translation
        paths = data_paths(info.data_config)
        if prepare_data and not paths.text_file(split, t.src_lang).is_file():
            prepare_translation_data(info.data_config)
        sources_text, references = read_parallel_text(
            paths.text_file(split, t.src_lang), paths.text_file(split, t.tgt_lang)
        )
        assert info.tokenizer is not None
        source_ids = info.tokenizer.encode_batch(sources_text)
        hyps = translator.translate_ids(source_ids, decode)
        hypotheses = [info.tokenizer.decode(h.tokens) for h in hyps]
        source_lengths = [len(s.split()) for s in sources_text]  # words, for readable buckets
    else:
        source_ids, references = _copy_split(info, split)
        hyps = translator.translate_ids(source_ids, decode)
        hypotheses = [" ".join(map(str, h.tokens)) for h in hyps]
        source_lengths = [len(s) for s in source_ids]
    seconds = time.perf_counter() - start

    scores = corpus_scores(hypotheses, references)
    named = checkpoint in ("best", "last", "averaged")
    ckpt_tag = checkpoint if named else Path(checkpoint).stem
    search_tag = "greedy" if decode.beam_size == 1 else f"beam{decode.beam_size}"
    tag = f"{split}.{ckpt_tag}.{search_tag}"
    result: dict[str, Any] = {
        "run": info.name,
        "split": split,
        "checkpoint": checkpoint,
        "average_last": average_last,
        "decode": asdict(decode),
        **scores,
        "bleu_by_length": bleu_by_length(hypotheses, references, source_lengths),
        "truncated_sources": translator.truncated,
        "decode_seconds": round(seconds, 2),
        "sentences_per_second": round(len(hypotheses) / max(seconds, 1e-9), 1),
        **translator.describe(),
    }
    if write:
        out = run_dir / "eval"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{tag}.hyp").write_text("\n".join(hypotheses) + "\n", encoding="utf-8")
        (out / f"{tag}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["output"] = str(out / f"{tag}.json")
    return result
