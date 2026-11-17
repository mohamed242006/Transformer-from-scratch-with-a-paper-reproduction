"""Translation API over a trained run directory (ADR-013 layout, ADR-019).

    translator = Translator.from_run("runs/multi30k_tiny_s1", checkpoint="averaged")
    translator.translate(["A man rides a bike."])            # -> ["Ein Mann fährt Fahrrad."]

Checkpoint choices: ``"best"`` (best validation NLL), ``"last"`` (``last.pt``, falling back
to ``last.prev.pt``), ``"averaged"`` (mean of the run's ``avg/`` snapshots, or the stored
``averaged.pt`` of a slimmed run), or a path.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from aiayn.data.collate import Collator
from aiayn.data.config import DataConfig
from aiayn.data.example import Example, frame_source, frame_target
from aiayn.data.special_tokens import PAD_ID
from aiayn.data.tokenizer import SentencePieceTokenizer
from aiayn.inference.averaging import average_checkpoint_files, load_weights, snapshot_paths
from aiayn.inference.decoding import DecodeConfig, Hypothesis, beam_search, greedy_decode
from aiayn.models import EncoderDecoder, build_model, model_config_from_dict
from aiayn.utils.config import from_dict, load_yaml

__all__ = ["RunInfo", "Translator", "load_run"]


class RunInfo:
    """Configs and tokenizer recovered from a run directory (no training code needed)."""

    def __init__(self, run_dir: Path) -> None:
        raw = load_yaml(run_dir / "config.yaml")
        self.run_dir = run_dir
        self.name = str(raw.get("name", run_dir.name))
        self.model_config = model_config_from_dict(raw["model"], "model")
        self.data_config = from_dict(DataConfig, raw["data"], "data")
        self.tokenizer: SentencePieceTokenizer | None = None
        if self.data_config.translation is not None:
            models = sorted((run_dir / "tokenizer").glob("*.model"))
            if not models:
                raise FileNotFoundError(f"{run_dir}/tokenizer/*.model missing")
            self.tokenizer = SentencePieceTokenizer(models[0])
            self.vocab_size = self.tokenizer.vocab_size
        else:
            assert self.data_config.copy is not None
            self.vocab_size = self.data_config.copy.vocab_size


def load_run(run_dir: str | Path) -> RunInfo:
    return RunInfo(Path(run_dir))


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _weights(run_dir: Path, checkpoint: str, average_last: int | None) -> dict[str, torch.Tensor]:
    ckpt_dir = run_dir / "checkpoints"
    if checkpoint == "best":
        return load_weights(ckpt_dir / "best.pt")
    if checkpoint == "last":
        try:
            return load_weights(ckpt_dir / "last.pt")
        except Exception:  # truncated by a killed session (ADR-013): use the previous state
            return load_weights(ckpt_dir / "last.prev.pt")
    if checkpoint == "averaged":
        if average_last is None and (ckpt_dir / "averaged.pt").is_file():
            return load_weights(ckpt_dir / "averaged.pt")  # slimmed run (ADR-022)
        return average_checkpoint_files(snapshot_paths(run_dir, average_last))
    return load_weights(Path(checkpoint))


class Translator:
    """Batched translation with greedy or beam search (``DecodeConfig.beam_size == 1``
    selects greedy). Sources longer than the model's positional table are truncated and
    counted in ``self.truncated``."""

    def __init__(
        self,
        model: EncoderDecoder,
        tokenizer: SentencePieceTokenizer | None,
        *,
        device: torch.device,
        max_tokens: int = 4096,
    ) -> None:
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_tokens = max_tokens
        self.truncated = 0
        cuda_fp16 = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 7
        self.autocast_dtype: torch.dtype | None = torch.float16 if cuda_fp16 else None

    @classmethod
    def from_run(
        cls,
        run_dir: str | Path,
        *,
        checkpoint: str = "best",
        average_last: int | None = None,
        device: str = "auto",
        max_tokens: int = 4096,
    ) -> Translator:
        info = load_run(run_dir)
        model = build_model(info.model_config, vocab_size=info.vocab_size, pad_id=PAD_ID, seed=0)
        model.load_state_dict(_weights(info.run_dir, checkpoint, average_last))
        return cls(model, info.tokenizer, device=_resolve_device(device), max_tokens=max_tokens)

    # ------------------------------------------------------------------ #
    def translate(self, sentences: Sequence[str], config: DecodeConfig | None = None) -> list[str]:
        """Raw text in, detokenized text out (SentencePiece handles detokenization)."""
        if self.tokenizer is None:
            raise ValueError("this run has no tokenizer (synthetic task): use translate_ids")
        ids = self.tokenizer.encode_batch(list(sentences))
        hyps = self.translate_ids(ids, config)
        return [self.tokenizer.decode(h.tokens) for h in hyps]

    def translate_ids(
        self, sources: Sequence[Sequence[int]], config: DecodeConfig | None = None
    ) -> list[Hypothesis]:
        """Content token IDs in (no BOS/EOS), hypotheses out, in input order."""
        config = config or DecodeConfig()
        limit = self.model.max_positions - 1  # room for EOS
        framed = []
        for ids in sources:
            if len(ids) > limit:
                self.truncated += 1
                ids = list(ids)[:limit]
            framed.append(frame_source(ids))
        results: list[Hypothesis | None] = [None] * len(framed)
        for batch_indices in self._batches([len(f) for f in framed], config.beam_size):
            examples = [
                Example(i, framed[i], frame_target([])) for i in batch_indices
            ]  # targets are unused at inference
            src = Collator(PAD_ID)(examples).src.to(self.device)
            decode = greedy_decode if config.beam_size == 1 else beam_search
            for i, hyp in zip(
                batch_indices,
                decode(self.model, src, config, autocast_dtype=self.autocast_dtype),
                strict=True,
            ):
                results[i] = hyp
        assert all(r is not None for r in results)
        return [r for r in results if r is not None]

    def _batches(self, lengths: list[int], beam: int) -> list[list[int]]:
        """Length-sorted batches whose ``beam × sentences × longest`` fits ``max_tokens``."""
        order = sorted(range(len(lengths)), key=lengths.__getitem__)
        batches: list[list[int]] = []
        current: list[int] = []
        longest = 0
        for i in order:
            new_longest = max(longest, lengths[i])
            if current and beam * (len(current) + 1) * new_longest > self.max_tokens:
                batches.append(current)
                current, new_longest = [], lengths[i]
            current.append(i)
            longest = new_longest
        if current:
            batches.append(current)
        return batches

    def describe(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "autocast": str(self.autocast_dtype),
            "parameters": sum(p.numel() for p in self.model.parameters()),
        }
