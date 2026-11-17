"""Greedy and beam-search decoding (paper §6.1; ADR-019).

Both decoders:

* work with any ``EncoderDecoder`` (Transformer or RNN baseline; ADR-021);
* run the encoder once per batch and recompute the decoder over the prefix at each step
  (no KV cache or carried RNN state yet: a planned extension; exact for every model
  because decoders are causal; cost is fine at sentence length);
* never generate PAD or BOS;
* stop each sentence at EOS or at ``max_len = max_len_a · |x| + max_len_b`` generated tokens
  (paper: input length + 50, so a = 1, b = 50), where EOS is forced on the last step;
* score hypotheses by the sum of token log-probabilities (float32) and, for ranking,
  divide by the GNMT length penalty ``lp(|Y|) = ((5 + |Y|) / 6) ** alpha`` with |Y|
  counting generated tokens including EOS.

Beam search follows the standard batched scheme: from the ``k`` live beams take the top
``2k`` candidates; EOS candidates ranked within the top ``k`` become finished
hypotheses; the best ``k`` non-EOS candidates stay alive. A sentence stops when no live
beam can still beat its best finished hypothesis (an exact bound, see ``beam_search``),
when no beam is alive, or at the length limit. With ``k = 1`` and no length penalty this
is exactly greedy decoding (tested).
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from aiayn.data.special_tokens import BOS_ID, EOS_ID, PAD_ID
from aiayn.models import EncoderDecoder

__all__ = ["DecodeConfig", "Hypothesis", "beam_search", "greedy_decode", "length_penalty"]


@dataclass(frozen=True)
class DecodeConfig:
    """Decoding hyperparameters. Defaults are the paper's (§6.1)."""

    beam_size: int = 4
    length_penalty: float = 0.6
    max_len_a: float = 1.0
    max_len_b: int = 50

    def __post_init__(self) -> None:
        if self.beam_size < 1:
            raise ValueError("beam_size must be >= 1")
        if self.length_penalty < 0 or self.max_len_a < 0 or self.max_len_b < 1:
            raise ValueError("length_penalty and max_len_a must be >= 0, max_len_b >= 1")


@dataclass(frozen=True)
class Hypothesis:
    """A decoded sequence. ``tokens`` excludes BOS and EOS."""

    tokens: list[int]
    score: float  # sum of log-probabilities, including the EOS step
    normalized_score: float  # score / lp(len(tokens) + 1)


def length_penalty(length: int, alpha: float) -> float:
    """GNMT length penalty ``((5 + length) / 6) ** alpha`` (Wu et al., 2016)."""
    return float(((5.0 + length) / 6.0) ** alpha)


def _max_lengths(src: torch.Tensor, pad_id: int, config: DecodeConfig, cap: int) -> list[int]:
    """Per-sentence limit on generated tokens (EOS included), capped by the PE table."""
    content = (src != pad_id).sum(dim=1) - 1  # source tokens without EOS
    limits = [int(config.max_len_a * int(n) + config.max_len_b) for n in content]
    return [max(1, min(limit, cap)) for limit in limits]


def _step_log_probs(
    model: EncoderDecoder,
    tokens: torch.Tensor,
    memory: Any,
    src_mask: torch.Tensor,
    autocast_dtype: torch.dtype | None,
) -> torch.Tensor:
    """Log-probabilities ``[N, V]`` (float32 or wider) for the next token of each row."""
    ctx: Any = (
        torch.autocast(device_type=tokens.device.type, dtype=autocast_dtype)
        if autocast_dtype is not None
        else contextlib.nullcontext()
    )
    with ctx:
        hidden, _, _ = model.decode(tokens, memory, src_mask)
        logits = model.generator(hidden[:, -1])
    compute = torch.promote_types(logits.dtype, torch.float32)
    lprobs = F.log_softmax(logits.to(compute), dim=-1)
    lprobs[:, PAD_ID] = -math.inf
    lprobs[:, BOS_ID] = -math.inf
    return lprobs


def _force_eos(lprobs: torch.Tensor, rows: torch.Tensor) -> None:
    """In place: rows that reached their length limit may only produce EOS."""
    if bool(rows.any()):
        eos = lprobs[rows, EOS_ID].clone()
        lprobs[rows] = -math.inf
        lprobs[rows, EOS_ID] = eos


def _encode(
    model: EncoderDecoder, src: torch.Tensor, autocast_dtype: torch.dtype | None
) -> tuple[Any, torch.Tensor]:
    ctx: Any = (
        torch.autocast(device_type=src.device.type, dtype=autocast_dtype)
        if autocast_dtype is not None
        else contextlib.nullcontext()
    )
    with ctx:
        memory, src_mask, _ = model.encode(src)
    return memory, src_mask


@torch.no_grad()
def greedy_decode(
    model: EncoderDecoder,
    src: torch.Tensor,
    config: DecodeConfig | None = None,
    *,
    autocast_dtype: torch.dtype | None = None,
) -> list[Hypothesis]:
    """Batched greedy decoding. ``src [B, S]`` -> one hypothesis per sentence.

    ``config.beam_size`` is ignored; the length limits and length penalty (used only for
    the reported ``normalized_score``) are honoured.
    """
    config = config or DecodeConfig(beam_size=1)
    model.eval()
    batch = src.shape[0]
    limits = _max_lengths(src, model.pad_id, config, model.max_positions - 1)
    limit_t = torch.tensor(limits, device=src.device)
    memory, src_mask = _encode(model, src, autocast_dtype)
    tokens = torch.full((batch, 1), BOS_ID, dtype=torch.long, device=src.device)
    scores = torch.zeros(batch, dtype=torch.float64, device=src.device)
    finished = torch.zeros(batch, dtype=torch.bool, device=src.device)
    for step in range(1, max(limits) + 1):
        lprobs = _step_log_probs(model, tokens, memory, src_mask, autocast_dtype)
        _force_eos(lprobs, limit_t <= step)
        best_lp, best = lprobs.max(dim=-1)
        best = torch.where(finished, torch.full_like(best, PAD_ID), best)
        scores += torch.where(finished, torch.zeros_like(best_lp), best_lp).double()
        tokens = torch.cat([tokens, best.unsqueeze(1)], dim=1)
        finished |= best == EOS_ID
        if bool(finished.all()):
            break
    out: list[Hypothesis] = []
    for row, score in zip(tokens[:, 1:].tolist(), scores.tolist(), strict=True):
        seq = row[: row.index(EOS_ID)] if EOS_ID in row else row
        out.append(
            Hypothesis(seq, score, score / length_penalty(len(seq) + 1, config.length_penalty))
        )
    return out


@torch.no_grad()
def beam_search(
    model: EncoderDecoder,
    src: torch.Tensor,
    config: DecodeConfig | None = None,
    *,
    autocast_dtype: torch.dtype | None = None,
) -> list[Hypothesis]:
    """Batched beam search with the GNMT length penalty. ``src [B, S]`` -> best per sentence."""
    config = config or DecodeConfig()
    model.eval()
    batch, k, vocab = src.shape[0], config.beam_size, model.vocab_size
    limits = _max_lengths(src, model.pad_id, config, model.max_positions - 1)
    memory, src_mask = _encode(model, src, autocast_dtype)
    memory = memory.repeat_interleave(k, dim=0)  # [B*k, S, d] (or RNNMemory)
    src_mask = src_mask.repeat_interleave(k, dim=0)  # [B*k, 1, 1, S]
    device = src.device
    tokens = torch.full((batch * k, 1), BOS_ID, dtype=torch.long, device=device)
    scores = torch.full((batch, k), -math.inf, dtype=torch.float64, device=device)
    scores[:, 0] = 0.0  # all beams start identical: keep only one alive
    finished: list[list[Hypothesis]] = [[] for _ in range(batch)]
    done = [False] * batch
    row_limits = torch.tensor(limits, device=device).repeat_interleave(k)

    for step in range(1, max(limits) + 1):
        lprobs = _step_log_probs(model, tokens, memory, src_mask, autocast_dtype).double()
        _force_eos(lprobs, row_limits <= step)
        candidates = (scores.view(-1, 1) + lprobs).view(batch, k * vocab)
        top_scores, top_index = candidates.topk(2 * k, dim=1)  # [B, 2k]
        top_scores_l, top_index_l = top_scores.tolist(), top_index.tolist()
        history = tokens[:, 1:].tolist()

        next_scores = torch.full((batch, k), -math.inf, dtype=torch.float64, device=device)
        next_rows = torch.zeros(batch * k, dtype=torch.long, device=device)
        next_tokens = torch.full((batch * k,), PAD_ID, dtype=torch.long, device=device)
        for b in range(batch):
            base = b * k
            if done[b]:
                next_rows[base : base + k] = base  # keep shapes; these rows are never read
                continue
            alive: list[tuple[float, int, int]] = []
            for rank, (cand_score, index) in enumerate(
                zip(top_scores_l[b], top_index_l[b], strict=True)
            ):
                if cand_score == -math.inf:
                    break
                beam, token = divmod(index, vocab)
                if token == EOS_ID:
                    if rank < k:  # only EOS candidates within the top k are finalized
                        seq = history[base + beam]
                        lp = length_penalty(len(seq) + 1, config.length_penalty)
                        finished[b].append(Hypothesis(seq, cand_score, cand_score / lp))
                elif len(alive) < k:
                    alive.append((cand_score, beam, token))
            if not alive or step >= limits[b]:
                done[b] = True
            elif finished[b]:
                # Exact stopping rule: a live beam's score can only decrease and its length
                # penalty can only grow up to the limit, so ``best_alive / lp(limit)`` bounds
                # every normalized score still reachable. Stop once a finished hypothesis
                # beats it. (Stopping after k finished hypotheses truncates outputs: the
                # copy-task evaluation caught exactly that.)
                best_finished = max(h.normalized_score for h in finished[b])
                bound = alive[0][0] / length_penalty(limits[b], config.length_penalty)
                done[b] = best_finished >= bound
            for slot, (cand_score, beam, token) in enumerate(alive):
                next_scores[b, slot] = cand_score
                next_rows[base + slot] = base + beam
                next_tokens[base + slot] = token
            for slot in range(len(alive), k):
                next_rows[base + slot] = base
        if all(done):
            break
        tokens = torch.cat([tokens[next_rows], next_tokens.unsqueeze(1)], dim=1)
        scores = next_scores

    best: list[Hypothesis] = []
    for b in range(batch):
        if not finished[b]:  # unreachable: EOS is forced at the length limit
            raise RuntimeError(f"sentence {b} finished without a hypothesis")
        best.append(max(finished[b], key=lambda h: h.normalized_score))
    return best
