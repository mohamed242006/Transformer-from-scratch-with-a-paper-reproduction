"""RNN encoder-decoder baselines (DESIGN.md §7; ADR-021).

Data flow (B batch, S source, T decoder length, E embedding, H hidden, He = H or 2H)::

    src ─ embed ─ dropout ─ (Bi)RNN encoder (packed) ─ dropout ─► values [B,S,He]
                                   └─ final states ─ [bridge: tanh(W[fwd; bwd])] ─► s_0
    none:      o_t = RNN(e(y_<t), s_0)                    readout([o_t])
    luong:     o_t = RNN(e(y_<t), s_0); c_t = attn(o_t)    readout([c_t; o_t])
    bahdanau:  c_t = attn(s_{t-1}); s_t = RNN([e(y_{t-1}); c_t], s_{t-1})
                                                         readout([s_t; c_t; e(y_{t-1})])
    readout: r_t = dropout(tanh(W_r · features + b)) ∈ R^E;  logits = r_t Eᵀ (tied, no bias)

Shared with the Transformer by design (fair comparison): joint vocabulary, three-way
tying, no output bias, and the ``EncoderDecoder`` interface, so training, beam search and
evaluation code is identical. Deliberate simplifications (documented in ADR-021): no
source reversal, no input feeding (Luong), no maxout readout (Bahdanau).

Only this package may use ``nn.LSTM``/``nn.GRU`` (ADR-001; enforced by the policy test).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from aiayn.models.base import EncoderDecoder
from aiayn.models.baselines.attention import BahdanauAttention, LuongAttention
from aiayn.models.baselines.config import RNNConfig

__all__ = ["RNNDecoderState", "RNNMemory", "RNNSeq2Seq", "build_rnn_seq2seq"]

State = tuple[torch.Tensor, ...]  # (h,) for GRU, (h, c) for LSTM; each [L, B, H]

INIT_RANGE = 0.1  # uniform(-0.1, 0.1): Sutskever et al. (2014), Luong et al. (2015)


@dataclass(frozen=True)
class RNNMemory:
    """Encoder output for the RNN decoder. Batch is dim 0 of ``values``/``keys`` and dim 1
    of the states (PyTorch RNN layout)."""

    values: torch.Tensor  # [B, S, He]
    keys: torch.Tensor | None  # projected once for attention, [B, S, A]
    state: State  # decoder initial state

    def repeat_interleave(self, repeats: int, dim: int = 0) -> RNNMemory:
        """Beam expansion (``Tensor.repeat_interleave`` semantics along the batch)."""
        if dim != 0:
            raise ValueError("RNNMemory repeats along the batch dimension only")
        return RNNMemory(
            self.values.repeat_interleave(repeats, dim=0),
            None if self.keys is None else self.keys.repeat_interleave(repeats, dim=0),
            tuple(s.repeat_interleave(repeats, dim=1) for s in self.state),
        )


@dataclass
class RNNDecoderState:
    """Incremental decoding: recurrent state after the tokens fed so far."""

    memory: RNNMemory
    src_mask: torch.Tensor  # [B, S]
    rnn: State


class RNNSeq2Seq(EncoderDecoder):
    """LSTM/GRU encoder-decoder with optional Bahdanau or Luong attention."""

    output_projection: nn.Parameter | None

    def __init__(self, config: RNNConfig, *, vocab_size: int, pad_id: int) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if not 0 <= pad_id < vocab_size:
            raise ValueError(f"pad_id {pad_id} outside [0, {vocab_size})")
        self.config = config
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        e, h, n = config.embedding_dim, config.hidden_size, config.num_layers
        he = config.encoder_output_dim
        rnn = nn.LSTM if config.cell == "lstm" else nn.GRU
        inter_layer_dropout = config.dropout if n > 1 else 0.0

        def embedding() -> nn.Embedding:
            return nn.Embedding(vocab_size, e, padding_idx=pad_id)

        if config.tie_weights:
            self.embeddings = nn.ModuleDict({"shared": embedding()})
            self.register_parameter("output_projection", None)
        else:
            self.embeddings = nn.ModuleDict({"source": embedding(), "target": embedding()})
            self.output_projection = nn.Parameter(torch.empty(vocab_size, e))
        self.dropout = nn.Dropout(config.dropout)
        self.encoder = rnn(
            e,
            h,
            num_layers=n,
            batch_first=True,
            dropout=inter_layer_dropout,
            bidirectional=config.bidirectional_encoder,
        )
        num_states = 2 if config.cell == "lstm" else 1
        self.bridges = nn.ModuleList(
            [nn.Linear(2 * h, h) for _ in range(num_states)] if config.bidirectional_encoder else []
        )
        decoder_input = e + (he if config.attention == "bahdanau" else 0)
        self.decoder = rnn(
            decoder_input, h, num_layers=n, batch_first=True, dropout=inter_layer_dropout
        )
        self.attention: BahdanauAttention | LuongAttention | None
        if config.attention == "bahdanau":
            self.attention = BahdanauAttention(h, he, h)
        elif config.attention == "luong":
            self.attention = LuongAttention(h, he)
        else:
            self.attention = None
        self.readout = nn.Linear(config.readout_inputs, e)
        self.reset_parameters()

    # ------------------------------------------------------------------ #
    @property
    def max_positions(self) -> int:
        return self.config.max_positions

    @property
    def src_embedding(self) -> nn.Embedding:
        key = "shared" if self.config.tie_weights else "source"
        return self.embeddings[key]  # type: ignore[return-value]

    @property
    def tgt_embedding(self) -> nn.Embedding:
        key = "shared" if self.config.tie_weights else "target"
        return self.embeddings[key]  # type: ignore[return-value]

    @property
    def output_weight(self) -> torch.Tensor:
        if self.output_projection is not None:
            return self.output_projection
        return self.tgt_embedding.weight

    def reset_parameters(self) -> None:
        """Uniform(-0.1, 0.1) for every parameter; embedding padding rows are zero."""
        with torch.no_grad():
            for p in self.parameters():
                p.uniform_(-INIT_RANGE, INIT_RANGE)
            for emb in self.embeddings.values():
                emb.weight[self.pad_id].zero_()  # type: ignore[index]

    # ------------------------------------------------------------------ #
    @staticmethod
    def _check_tokens(tokens: torch.Tensor, name: str) -> None:
        if tokens.dim() != 2 or tokens.shape[1] == 0:
            raise ValueError(f"{name} must have shape [B, L>0], got {tuple(tokens.shape)}")
        if tokens.dtype not in (torch.int64, torch.int32):
            raise TypeError(f"{name} must be int64/int32 token IDs, got {tokens.dtype}")

    def _run(
        self, rnn: nn.Module, inputs: torch.Tensor, state: State
    ) -> tuple[torch.Tensor, State]:
        if self.config.cell == "lstm":
            out, (h, c) = rnn(inputs, (state[0], state[1]))
            return out, (h, c)
        out, h = rnn(inputs, state[0])
        return out, (h,)

    def _initial_state(self, final: State) -> State:
        """Encoder final states ``[L·dirs, B, H]`` -> decoder initial states ``[L, B, H]``."""
        if not self.config.bidirectional_encoder:
            return tuple(s.contiguous() for s in final)
        layers = self.config.num_layers
        bridged = []
        for s, bridge in zip(final, self.bridges, strict=True):
            s = s.view(layers, 2, s.shape[1], s.shape[2])  # [L, dirs, B, H]
            both = torch.cat([s[:, 0], s[:, 1]], dim=-1)  # [L, B, 2H]: forward; backward
            bridged.append(torch.tanh(bridge(both)).contiguous())
        return tuple(bridged)

    def encode(
        self, src: torch.Tensor, *, need_weights: bool = False
    ) -> tuple[RNNMemory, torch.Tensor, tuple[torch.Tensor, ...]]:
        """``src [B,S]`` -> ``(RNNMemory, key mask [B,S], ())``."""
        self._check_tokens(src, "src")
        key_mask = src != self.pad_id
        lengths = key_mask.sum(dim=1).clamp(min=1).cpu()
        x = self.dropout(self.src_embedding(src))
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, final = self.encoder(packed)
        values, _ = pad_packed_sequence(out, batch_first=True, total_length=src.shape[1])
        values = self.dropout(values)
        final_state: State = final if isinstance(final, tuple) else (final,)
        keys = self.attention.project_keys(values) if self.attention is not None else None
        return RNNMemory(values, keys, self._initial_state(final_state)), key_mask, ()

    def decode(
        self,
        tgt_in: torch.Tensor,
        memory: RNNMemory,
        src_mask: torch.Tensor,
        *,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """``tgt_in [B,T]`` -> ``(readout [B,T,E], (), cross maps ([B,T,S],) or ())``."""
        self._check_tokens(tgt_in, "tgt_in")
        if tgt_in.shape[0] != memory.values.shape[0]:
            raise ValueError(
                f"batch mismatch: tgt_in {tgt_in.shape[0]} vs memory {memory.values.shape[0]}"
            )
        y = self.dropout(self.tgt_embedding(tgt_in))  # [B, T, E]
        weights: torch.Tensor | None = None
        if self.config.attention == "bahdanau":
            features, weights = self._decode_bahdanau(y, memory, src_mask)
        else:
            outputs, _ = self._run(self.decoder, y, memory.state)
            if self.attention is None:
                features = outputs
            else:
                assert memory.keys is not None
                context, weights = self.attention(outputs, memory.keys, memory.values, src_mask)
                features = torch.cat([context.to(outputs.dtype), outputs], dim=-1)
        hidden = self.dropout(torch.tanh(self.readout(features)))
        maps = (weights.detach(),) if need_weights and weights is not None else ()
        return hidden, (), maps

    def _decode_bahdanau(
        self, y: torch.Tensor, memory: RNNMemory, src_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sequential: the query at step t is the previous top-layer state s_{t-1}."""
        state = memory.state
        features, weights = [], []
        for t in range(y.shape[1]):
            f, w, state = self._bahdanau_step(y[:, t : t + 1], memory, src_mask, state)
            features.append(f)
            weights.append(w)
        return torch.cat(features, dim=1), torch.cat(weights, dim=1)

    def _bahdanau_step(
        self, y_t: torch.Tensor, memory: RNNMemory, src_mask: torch.Tensor, state: State
    ) -> tuple[torch.Tensor, torch.Tensor, State]:
        """One target position ``y_t [B, 1, E]`` -> ``(features [B, 1, F], weights, state)``."""
        assert self.attention is not None and memory.keys is not None
        query = state[0][-1].unsqueeze(1)  # s_{t-1}, top layer: [B, 1, H]
        context, w = self.attention(query, memory.keys, memory.values, src_mask)
        step_input = torch.cat([y_t, context.to(y_t.dtype)], dim=-1)
        out, state = self._run(self.decoder, step_input, state)
        features = torch.cat([out, context.to(out.dtype), y_t.to(out.dtype)], dim=-1)
        return features, w, state

    # ------------------------------------------------------------------ #
    # Incremental decoding (ADR-023 §6)
    # ------------------------------------------------------------------ #
    supports_incremental = True

    def init_state(self, memory: RNNMemory, src_mask: torch.Tensor) -> RNNDecoderState:
        return RNNDecoderState(memory, src_mask, memory.state)

    def decode_step(
        self, tokens: torch.Tensor, state: RNNDecoderState
    ) -> tuple[torch.Tensor, RNNDecoderState]:
        """Newest token per row ``tokens [N]`` -> readout ``[N, E]`` at that position."""
        y = self.dropout(self.tgt_embedding(tokens[:, None]))  # [N, 1, E]
        if self.config.attention == "bahdanau":
            features, _, state.rnn = self._bahdanau_step(y, state.memory, state.src_mask, state.rnn)
        else:
            out, state.rnn = self._run(self.decoder, y, state.rnn)
            if self.attention is None:
                features = out
            else:
                assert state.memory.keys is not None
                context, _ = self.attention(
                    out, state.memory.keys, state.memory.values, state.src_mask
                )
                features = torch.cat([context.to(out.dtype), out], dim=-1)
        hidden = self.dropout(torch.tanh(self.readout(features)))
        return hidden[:, 0], state

    def reorder_state(self, state: RNNDecoderState, rows: torch.Tensor) -> RNNDecoderState:
        m = state.memory
        state.memory = RNNMemory(
            m.values[rows],
            None if m.keys is None else m.keys[rows],
            tuple(s[:, rows] for s in m.state),
        )
        state.src_mask = state.src_mask[rows]
        state.rnn = tuple(s[:, rows].contiguous() for s in state.rnn)
        return state

    def generator(self, hidden: torch.Tensor) -> torch.Tensor:
        """``hidden [B,T,E] -> logits [B,T,V]`` = ``r Eᵀ`` (no bias)."""
        return F.linear(hidden, self.output_weight)

    def extra_repr(self) -> str:
        c = self.config
        return (
            f"cell={c.cell}, attention={c.attention}, bidirectional={c.bidirectional_encoder}, "
            f"vocab_size={self.vocab_size}, tied={c.tie_weights}"
        )


def build_rnn_seq2seq(
    config: RNNConfig, *, vocab_size: int, pad_id: int, seed: int | None = None
) -> RNNSeq2Seq:
    """Construct an ``RNNSeq2Seq``; with ``seed``, initialization leaves the global RNG alone."""
    if seed is None:
        return RNNSeq2Seq(config, vocab_size=vocab_size, pad_id=pad_id)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return RNNSeq2Seq(config, vocab_size=vocab_size, pad_id=pad_id)
