"""Kaggle / environment validation (docs/kaggle_validation_notebook.md, docs/KAGGLE.md).

``run_validation(options)`` executes the checks in dependency order and returns a
``Report``. Checks whose prerequisites failed are recorded as ``skip``. Every check is
also usable on a CPU-only machine: CUDA-specific checks skip (or fail with
``require_cuda``), and VRAM is then *estimated* from the tensors autograd saves.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F

from aiayn import __version__
from aiayn.data.collate import Batch, Collator
from aiayn.data.config import DataConfig, load_data_config
from aiayn.data.datasets import read_lines
from aiayn.data.example import Example, frame_source, frame_target
from aiayn.data.loading import build_dataloader, build_dataset, train_tokenizer
from aiayn.data.prepare import data_paths, prepare_translation_data
from aiayn.data.special_tokens import BOS_ID, EOS_ID, NUM_SPECIAL_TOKENS, PAD_ID, UNK_ID
from aiayn.data.tokenizer import SentencePieceTokenizer, train_sentencepiece
from aiayn.diagnostics.packaging import MANIFEST_NAME
from aiayn.diagnostics.report import Outcome, Report, Status
from aiayn.models import ModelConfig, Transformer, build_transformer, expected_parameter_count
from aiayn.models.config import load_model_config
from aiayn.modules import LayerNorm, MultiHeadAttention
from aiayn.utils.params import count_parameters
from aiayn.utils.paths import find_repo_root

__all__ = ["ValidationOptions", "run_validation"]

MiB = 1024**2

#: (minimum version, exclusive maximum or None, critical) from pyproject.toml.
DEPENDENCIES: dict[str, tuple[str, str | None, bool]] = {
    "torch": ("2.3", "3", True),
    "numpy": ("1.26", None, True),
    "PyYAML": ("6.0", None, True),
    "sentencepiece": ("0.2.0", None, True),
    "sacrebleu": ("2.4", None, False),  # first used in Week 6
}


@dataclass
class ValidationOptions:
    data_config: Path
    model_configs: list[Path]
    data_overrides: tuple[str, ...] = ()
    data_mirrors: list[Path] = field(default_factory=list)
    tokenizer_path: Path | None = None
    expected_fingerprint: str | None = None
    train_tokenizer_if_missing: bool = False
    retrain_tokenizer_check: bool = False
    device: Literal["auto", "cuda", "cpu"] = "auto"
    require_cuda: bool = False
    amp: Literal["auto", "fp16", "off"] = "auto"
    vram_max_tokens: int | None = None
    steps: int = 30
    warmup: int = 5
    expected_torch_version: str | None = None
    check_internet: bool = True
    seed: int = 0


@dataclass
class _State:
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    amp_dtype: torch.dtype | None = None
    data_config: DataConfig | None = None
    tokenizer: SentencePieceTokenizer | None = None
    vocab_size: int = 0
    datasets: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _version_tuple(text: str) -> tuple[int, ...]:
    core = text.split("+", 1)[0]
    parts: list[int] = []
    for piece in core.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _ram_gib() -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024**2, 1)
    except OSError:
        pass
    try:
        psutil: Any = importlib.import_module("psutil")
        return float(round(psutil.virtual_memory().total / 1024**3, 1))
    except ImportError:
        return None


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast(state: _State) -> Any:
    if state.amp_dtype is None:
        return torch.autocast(device_type=state.device.type, enabled=False)
    return torch.autocast(device_type=state.device.type, dtype=state.amp_dtype)


def _loss(model: Transformer, batch: Batch) -> torch.Tensor:
    """Plain token-level cross-entropy in float32 (diagnostic only; the training loss with
    label smoothing arrives in Week 5)."""
    logits = model(batch.src, batch.tgt_in)
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        batch.tgt_out.reshape(-1),
        ignore_index=model.pad_id,
    )


def _worst_case_batch(state: _State, max_tokens: int) -> tuple[Batch, int]:
    """Longest allowed sentences, as many as the token budget admits (peak memory)."""
    cfg = state.data_config
    assert cfg is not None
    if cfg.translation is not None:
        length = cfg.translation.filter.max_len  # tensor length limit for S and T
    else:
        assert cfg.copy is not None
        length = cfg.copy.max_len + 1
    batch_size = max(1, max_tokens // length)
    g = torch.Generator().manual_seed(0)
    content = torch.randint(
        NUM_SPECIAL_TOKENS, state.vocab_size, (batch_size, length - 1), generator=g
    )
    examples = [
        Example(index=i, src=frame_source(row.numpy()), tgt=frame_target(row.numpy()))
        for i, row in enumerate(content)
    ]
    return Collator(PAD_ID)(examples), batch_size * length


def _saved_activation_bytes(model: Transformer, batch: Batch, state: _State) -> int:
    """Bytes of unique tensors autograd keeps for backward during one forward pass."""
    seen: dict[tuple[str, int], int] = {}

    def pack(t: torch.Tensor) -> torch.Tensor:
        storage = t.untyped_storage()
        key = (str(storage.device), storage.data_ptr())
        seen[key] = max(seen.get(key, 0), storage.nbytes())
        return t

    params = {p.untyped_storage().data_ptr() for p in model.parameters()}
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t), _autocast(state):
        model(batch.src, batch.tgt_in).float().sum().backward()
    model.zero_grad(set_to_none=True)
    return sum(n for (_, ptr), n in seen.items() if ptr not in params)


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def _check_python() -> Outcome:
    v = sys.version_info
    ok = (3, 10) <= (v.major, v.minor) <= (3, 12)
    return Outcome(
        "pass" if ok else "fail",
        f"Python {platform.python_version()}",
        {"version": platform.python_version()},
    )


def _check_platform(output_dir: Path) -> Outcome:
    disk = shutil.disk_usage(output_dir if output_dir.exists() else Path.cwd())
    free_gib = round(disk.free / 1024**3, 1)
    details: dict[str, Any] = {
        "system": platform.platform(),
        "on_kaggle": bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle").is_dir(),
        "cpu_count": os.cpu_count(),
        "ram_gib": _ram_gib(),
        "free_disk_gib": free_gib,
    }
    status: Status = "pass" if free_gib >= 5 else "warn"
    return Outcome(
        status,
        f"{details['cpu_count']} CPUs, {details['ram_gib']} GiB RAM, "
        f"{details['free_disk_gib']} GiB free",
        details,
    )


def _check_internet() -> Outcome:
    try:
        request = urllib.request.Request("https://pypi.org/simple/", method="HEAD")
        with urllib.request.urlopen(request, timeout=4):
            pass
        return Outcome("pass", "internet reachable", {"internet": True})
    except Exception as exc:
        return Outcome(
            "warn", f"no internet ({type(exc).__name__}); offline mode", {"internet": False}
        )


def _check_dependencies() -> Outcome:
    versions: dict[str, str | None] = {}
    problems: list[str] = []
    warnings: list[str] = []
    for name, (low, high, critical) in DEPENDENCIES.items():
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        versions[name] = version
        bucket = problems if critical else warnings
        if version is None:
            bucket.append(f"{name} missing")
        elif _version_tuple(version) < _version_tuple(low) or (
            high is not None and _version_tuple(version) >= _version_tuple(high)
        ):
            bucket.append(f"{name} {version} outside [{low}, {high or 'inf'})")
    versions["aiayn"] = __version__
    if problems:
        return Outcome("fail", "; ".join(problems), versions)
    if warnings:
        return Outcome("warn", "; ".join(warnings), versions)
    return Outcome("pass", ", ".join(f"{k} {v}" for k, v in versions.items()), versions)


def _check_torch_unchanged(expected: str) -> Outcome:
    actual = torch.__version__
    ok = actual == expected
    return Outcome(
        "pass" if ok else "fail",
        f"torch {actual}" + ("" if ok else f" (was {expected} before install)"),
        {"before": expected, "after": actual},
    )


def _check_cuda(state: _State, options: ValidationOptions) -> Outcome:
    if not torch.cuda.is_available():
        status: Status = "fail" if options.require_cuda or options.device == "cuda" else "skip"
        return Outcome(status, "CUDA not available", {"available": False})
    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    major, minor = props.major, props.minor
    arch = f"sm_{major}{minor}"
    arch_list = torch.cuda.get_arch_list()
    details = {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "name": props.name,
        "capability": f"{major}.{minor}",
        "total_memory_gib": round(props.total_memory / 1024**3, 2),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),  # type: ignore[no-untyped-call]
        "arch_supported": arch in arch_list
        or any(a.startswith(f"compute_{major}") for a in arch_list),
        "arch_list": arch_list,
    }
    if not details["arch_supported"]:
        return Outcome(
            "fail",
            f"{props.name} ({arch}) not supported by this PyTorch build; "
            "switch the accelerator to T4",
            details,
        )
    state.device = torch.device("cuda", index)
    return Outcome("pass", f"{props.name}, {details['total_memory_gib']} GiB, {arch}", details)


def _check_cuda_numerics(state: _State) -> Outcome:
    torch.manual_seed(0)
    mha = MultiHeadAttention(64, 4).double().eval()
    norm = LayerNorm(64).double()
    x = torch.randn(4, 16, 64, dtype=torch.float64)
    ref_attn, _ = mha(x, x, x)
    ref_norm = norm(x * 30 + 5)
    gpu_x = x.float().to(state.device)
    mha_gpu, norm_gpu = mha.float().to(state.device), norm.float().to(state.device)
    with torch.no_grad():
        attn_err = (mha_gpu(gpu_x, gpu_x, gpu_x)[0].double().cpu() - ref_attn).abs().max().item()
        norm_err = (norm_gpu(gpu_x * 30 + 5).double().cpu() - ref_norm).abs().max().item()
        details: dict[str, Any] = {"fp32_attention_err": attn_err, "fp32_layernorm_err": norm_err}
        fp16_ok = True
        if state.amp_dtype is not None:
            with torch.autocast("cuda", dtype=state.amp_dtype):
                half = mha_gpu(gpu_x, gpu_x, gpu_x)[0]
            details["fp16_attention_err"] = (half.double().cpu() - ref_attn).abs().max().item()
            fp16_ok = bool(torch.isfinite(half).all()) and details["fp16_attention_err"] < 5e-2
    ok = attn_err < 1e-4 and norm_err < 1e-4 and fp16_ok
    return Outcome(
        "pass" if ok else "fail",
        f"GPU vs CPU-fp64: attention {attn_err:.1e}, LayerNorm {norm_err:.1e}",
        details,
    )


def _check_data(state: _State, options: ValidationOptions) -> Outcome:
    cfg = load_data_config(options.data_config, options.data_overrides)
    state.data_config = cfg
    if cfg.translation is None:
        return Outcome("pass", f"{cfg.name}: synthetic task (no files needed)", {"task": cfg.task})
    paths = prepare_translation_data(cfg, mirror_dirs=options.data_mirrors)
    sizes = {
        split: len(read_lines(paths.text_file(split, cfg.translation.src_lang)))
        for split in ("train", "valid", "test")
    }
    return Outcome(
        "pass",
        f"{cfg.name}: checksums verified, splits {sizes}",
        {"processed_dir": str(paths.processed_dir), "split_sizes": sizes},
    )


def _check_tokenizer(state: _State, options: ValidationOptions) -> Outcome:
    cfg = state.data_config
    assert cfg is not None
    if cfg.translation is None:
        state.vocab_size = cfg.copy.vocab_size if cfg.copy else 0
        return Outcome("skip", "synthetic task uses integer IDs", {"vocab_size": state.vocab_size})
    path = options.tokenizer_path or data_paths(cfg).tokenizer_model
    trained = False
    if not path.is_file():
        if not options.train_tokenizer_if_missing:
            return Outcome("fail", f"tokenizer not found at {path}")
        path, trained = train_tokenizer(cfg), True
    tok = SentencePieceTokenizer(path)
    state.tokenizer, state.vocab_size = tok, tok.vocab_size
    ids = (
        tok.id_to_piece(PAD_ID),
        tok.id_to_piece(UNK_ID),
        tok.id_to_piece(BOS_ID),
        tok.id_to_piece(EOS_ID),
    )
    details: dict[str, Any] = {
        "path": str(path),
        "trained_here": trained,
        "vocab_size": tok.vocab_size,
        "fingerprint": tok.fingerprint,
        "special_pieces": ids,
    }
    expected_vocab = cfg.translation.tokenizer.vocab_size
    if cfg.translation.tokenizer.hard_vocab_limit and tok.vocab_size != expected_vocab:
        return Outcome("fail", f"vocab {tok.vocab_size} != configured {expected_vocab}", details)
    sample = read_lines(data_paths(cfg).text_file("valid", cfg.translation.src_lang))[:200]
    exact = sum(tok.decode(tok.encode(s)) == s for s in sample)
    details["roundtrip_exact"] = f"{exact}/{len(sample)}"
    status: Status = "pass" if exact >= 0.99 * len(sample) else "warn"
    return Outcome(status, f"V={tok.vocab_size}, round-trip {exact}/{len(sample)}", details)


def _check_fingerprint(state: _State, expected: str) -> Outcome:
    """Separate from loading so a mismatch is reported without hiding the later checks."""
    tok = state.tokenizer
    assert tok is not None
    details = {"expected": expected, "actual": tok.fingerprint}
    if tok.fingerprint == expected:
        return Outcome("pass", f"fingerprint matches ({expected[:12]})", details)
    return Outcome(
        "fail",
        "fingerprint differs: tokenizer is not the one used locally "
        "(different SentencePiece version or training data); do not compare results",
        details,
    )


def _check_tokenizer_determinism(state: _State) -> Outcome:
    cfg, tok = state.data_config, state.tokenizer
    assert cfg is not None and cfg.translation is not None and tok is not None
    paths = data_paths(cfg)
    with tempfile.TemporaryDirectory() as tmp:
        inputs = [
            paths.text_file("train", cfg.translation.src_lang),
            paths.text_file("train", cfg.translation.tgt_lang),
        ]
        fresh = SentencePieceTokenizer(
            train_sentencepiece(inputs, Path(tmp) / "check", cfg.translation.tokenizer)
        )
        same = fresh.fingerprint == tok.fingerprint
    return Outcome(
        "pass" if same else "fail",
        "retrained tokenizer is identical" if same else "retrained tokenizer differs",
        {"retrained": fresh.fingerprint, "loaded": tok.fingerprint},
    )


def _check_loader(state: _State) -> Outcome:
    cfg = state.data_config
    assert cfg is not None
    details: dict[str, Any] = {}
    for split in ("train", "valid"):
        dataset = build_dataset(cfg, split, tokenizer=state.tokenizer)
        state.datasets[split] = dataset
        details[f"{split}_examples"] = len(dataset)
    loader = build_dataloader(cfg, state.datasets["train"], "train")
    start, seen = time.perf_counter(), 0
    for batch in loader:
        seen += batch.batch_size
    details["epoch_iteration_s"] = round(time.perf_counter() - start, 2)
    details["padding_efficiency"] = round(loader.batch_sampler.padding_efficiency(), 3)  # type: ignore[union-attr]
    details["batches"] = len(loader)
    ok = seen == len(state.datasets["train"])
    return Outcome(
        "pass" if ok else "fail",
        f"{details['train_examples']} train examples in {details['batches']} batches, "
        f"padding efficiency {details['padding_efficiency']}",
        details,
    )


def _train_batches(state: _State) -> Iterator[Batch]:
    cfg = state.data_config
    assert cfg is not None
    loader = build_dataloader(cfg, state.datasets["train"], "train")
    while True:
        yield from loader


@dataclass
class _ModelRun:
    config: ModelConfig
    model: Transformer | None = None


def _check_model_build(run: _ModelRun, state: _State, seed: int) -> Outcome:
    model = build_transformer(run.config, vocab_size=state.vocab_size, pad_id=PAD_ID, seed=seed)
    counted = count_parameters(model)
    expected = expected_parameter_count(run.config, state.vocab_size)
    tied_ok = (model.output_weight is model.src_embedding.weight) == run.config.tie_weights
    run.model = model.to(state.device)
    details = {"parameters": counted, "expected": expected, "tied": run.config.tie_weights}
    ok = counted == expected and tied_ok
    return Outcome(
        "pass" if ok else "fail", f"{counted:,} parameters (analytic {expected:,})", details
    )


def _check_forward(run: _ModelRun, state: _State) -> Outcome:
    model = run.model
    assert model is not None
    batch = Collator(PAD_ID)(
        [state.datasets["valid"][i] for i in range(min(32, len(state.datasets["valid"])))]
    )
    batch = batch.to(state.device)
    model.eval()
    model.debug_checks = True
    with torch.no_grad():
        logits = model(batch.src, batch.tgt_in)
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            batch.tgt_out.reshape(-1),
            ignore_index=PAD_ID,
        ).item()
    model.debug_checks = False
    low, high = math.log(state.vocab_size), math.log(state.vocab_size) + 1.5
    details = {
        "logits_shape": list(logits.shape),
        "initial_loss": round(loss, 4),
        "expected_band": [round(low, 3), round(high, 3)],
    }
    ok = low <= loss <= high
    return Outcome(
        "pass" if ok else "fail",
        f"logits {list(logits.shape)}, initial CE {loss:.3f} in [{low:.2f}, {high:.2f}]",
        details,
    )


def _check_backward(run: _ModelRun, state: _State) -> Outcome:
    model = run.model
    assert model is not None
    model.train()
    batch = next(_train_batches(state)).to(state.device)
    scaler = torch.amp.GradScaler(state.device.type, enabled=state.amp_dtype is torch.float16)
    with _autocast(state):
        loss = _loss(model, batch)
    scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
    scaler.unscale_(torch.optim.SGD(model.parameters(), lr=0.0))
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    finite = all(
        bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None
    )
    norm = math.sqrt(
        sum(float(p.grad.float().pow(2).sum()) for p in model.parameters() if p.grad is not None)
    )
    model.zero_grad(set_to_none=True)
    details = {
        "loss": round(loss.item(), 4),
        "grad_norm": round(norm, 4),
        "missing_grads": missing,
        "amp": str(state.amp_dtype),
    }
    ok = not missing and finite and math.isfinite(norm)
    return Outcome(
        "pass" if ok else "fail",
        f"loss {loss.item():.3f}, grad-norm {norm:.2f}, AMP {state.amp_dtype}",
        details,
    )


def _check_memory(run: _ModelRun, state: _State, max_tokens: int) -> Outcome:
    model = run.model
    assert model is not None
    model.train()
    batch, tokens = _worst_case_batch(state, max_tokens)
    details: dict[str, Any] = {
        "batch_shape": [batch.batch_size, batch.src.shape[1]],
        "max_tokens": max_tokens,
        "padded_tokens_per_side": tokens,
    }
    if state.device.type != "cuda":
        # Estimate: activations are linear in the number of sentences at fixed length, so
        # measure a 2-sentence slice and scale. Emulate T4 fp16 with bf16 (same 2 bytes).
        n_params = count_parameters(model)
        est_state = replace(state, amp_dtype=torch.bfloat16)
        sentences = min(2, batch.batch_size)
        probe, _ = _worst_case_batch(state, sentences * int(batch.src.shape[1]))
        activations = (
            _saved_activation_bytes(model, probe, est_state) * batch.batch_size / sentences
        )
        static_estimate = 18 * n_params  # fp32 weights + grads + 2 Adam moments + fp16 cache
        logits_transient = batch.tgt_out.numel() * state.vocab_size * 4 * 2  # fp32 logits + grad
        estimate = static_estimate + activations + logits_transient
        details.update(
            {
                "static_estimate_mib": round(static_estimate / MiB, 1),
                "saved_activations_mib": round(activations / MiB, 1),
                "logits_transient_mib": round(logits_transient / MiB, 1),
                "estimated_peak_mib": round(estimate / MiB, 1),
                "estimate_precision": "bfloat16 (as a stand-in for fp16)",
            }
        )
        return Outcome(
            "skip",
            f"no CUDA; estimated peak {estimate / MiB:.0f} MiB "
            f"at {max_tokens} tokens/side (measured on GPU)",
            details,
        )
    batch = batch.to(state.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-9)
    scaler = torch.amp.GradScaler("cuda", enabled=state.amp_dtype is torch.float16)

    def step(b: Batch) -> None:
        with _autocast(state):
            loss = _loss(model, b)
        scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    small, _ = _worst_case_batch(state, max(64, max_tokens // 16))
    step(small.to(state.device))  # materializes Adam state
    _sync(state.device)
    torch.cuda.empty_cache()
    static = torch.cuda.memory_allocated(state.device)
    torch.cuda.reset_peak_memory_stats(state.device)
    step(batch)
    _sync(state.device)
    peak = torch.cuda.max_memory_allocated(state.device)
    reserved = torch.cuda.max_memory_reserved(state.device)
    total = torch.cuda.get_device_properties(state.device).total_memory
    budget = 0.8 * total - static
    suggested = int(max_tokens * budget / max(peak - static, 1))
    details.update(
        {
            "static_mib": round(static / MiB, 1),
            "peak_allocated_mib": round(peak / MiB, 1),
            "peak_reserved_mib": round(reserved / MiB, 1),
            "device_total_mib": round(total / MiB, 1),
            "suggested_max_tokens_80pct": suggested,
        }
    )
    status: Status = "pass" if reserved < 0.9 * total else "warn"
    return Outcome(
        status,
        f"peak {peak / MiB:.0f} MiB allocated / {reserved / MiB:.0f} MiB reserved "
        f"of {total / MiB:.0f} MiB at {max_tokens} tokens/side; "
        f"80% budget allows ~{suggested} tokens",
        details,
    )


def _check_throughput(run: _ModelRun, state: _State, steps: int, warmup: int) -> Outcome:
    model = run.model
    assert model is not None
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-9)
    scaler = torch.amp.GradScaler(state.device.type, enabled=state.amp_dtype is torch.float16)
    batches = _train_batches(state)
    src_tokens = tgt_tokens = 0
    elapsed = 0.0
    for i in range(warmup + steps):
        batch = next(batches).to(state.device, non_blocking=True)
        _sync(state.device)
        start = time.perf_counter()
        with _autocast(state):
            loss = _loss(model, batch)
        scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        _sync(state.device)
        if i >= warmup:
            elapsed += time.perf_counter() - start
            src_tokens += batch.num_src_tokens
            tgt_tokens += batch.num_tgt_tokens
    train = state.datasets["train"]
    epoch_tgt = int(train.tgt_lengths.sum())
    tgt_per_s = tgt_tokens / elapsed
    details = {
        "steps": steps,
        "ms_per_step": round(1000 * elapsed / steps, 1),
        "target_tokens_per_s": round(tgt_per_s),
        "total_tokens_per_s": round((src_tokens + tgt_tokens) / elapsed),
        "estimated_epoch_s": round(epoch_tgt / tgt_per_s, 1),
        "amp": str(state.amp_dtype),
    }
    return Outcome(
        "pass",
        f"{details['target_tokens_per_s']:,} tgt tok/s, "
        f"{details['ms_per_step']} ms/step, ~{details['estimated_epoch_s']} s/epoch",
        details,
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "aiayn": __version__,
    }
    try:
        manifest = json.loads((find_repo_root() / MANIFEST_NAME).read_text(encoding="utf-8"))
        env["code_tree_sha256"] = manifest.get("tree_sha256")
    except (OSError, ValueError):
        env["code_tree_sha256"] = None  # running from a working copy, not a packaged bundle
    return env


def _run_model_checks(
    report: Report, run: _ModelRun, state: _State, options: ValidationOptions, max_tokens: int
) -> None:
    prefix = f"model[{run.config.name}]"
    report.run(f"{prefix}.build", lambda: _check_model_build(run, state, options.seed))
    if run.model is None or not report.passed(f"{prefix}.build"):
        return
    report.run(f"{prefix}.forward", lambda: _check_forward(run, state))
    report.run(f"{prefix}.backward", lambda: _check_backward(run, state))
    report.run(f"{prefix}.memory", lambda: _check_memory(run, state, max_tokens), critical=False)
    report.run(
        f"{prefix}.throughput",
        lambda: _check_throughput(run, state, options.steps, options.warmup),
        critical=False,
    )
    run.model = None
    if state.device.type == "cuda":
        torch.cuda.empty_cache()


def run_validation(options: ValidationOptions, output_dir: Path | None = None) -> Report:
    """Run every check in dependency order; never raises for a failing check."""
    torch.manual_seed(options.seed)
    report = Report()
    report.environment = _environment()
    state = _State()

    report.run("env.python", _check_python)
    report.run("env.platform", lambda: _check_platform(output_dir or Path.cwd()), critical=False)
    if options.check_internet:
        report.run("env.internet", _check_internet, critical=False)
    report.run("deps.versions", _check_dependencies)
    if options.expected_torch_version:
        report.run(
            "deps.torch_unchanged",
            lambda: _check_torch_unchanged(options.expected_torch_version or ""),
        )

    want_cuda = options.device in ("auto", "cuda")
    if want_cuda:
        report.run(
            "cuda.available",
            lambda: _check_cuda(state, options),
            critical=options.require_cuda or options.device == "cuda",
        )
    if state.device.type == "cuda":
        major = torch.cuda.get_device_capability(state.device)[0]
        if options.amp == "fp16" or (options.amp == "auto" and major >= 7):
            state.amp_dtype = torch.float16  # ADR-014: fp16 + GradScaler on T4 and newer
        report.run("cuda.numerics", lambda: _check_cuda_numerics(state))
    elif options.amp == "fp16":
        report.skip("cuda.numerics", "fp16 AMP requested but no CUDA device")
    report.environment["device"] = str(state.device)
    report.environment["amp"] = str(state.amp_dtype)

    if not report.passed("deps.versions"):
        report.skip("data.prepare", "dependencies failed")
        return report
    report.run("data.prepare", lambda: _check_data(state, options))
    if not report.passed("data.prepare"):
        return report
    report.run("tokenizer.load", lambda: _check_tokenizer(state, options))
    tok_result = report.get("tokenizer.load")
    tokenizer_ok = tok_result is not None and tok_result.status in ("pass", "warn", "skip")
    if options.expected_fingerprint and state.tokenizer is not None:
        expected = options.expected_fingerprint
        report.run("tokenizer.fingerprint", lambda: _check_fingerprint(state, expected))
    if options.retrain_tokenizer_check and state.tokenizer is not None:
        report.run(
            "tokenizer.deterministic", lambda: _check_tokenizer_determinism(state), critical=False
        )
    if not tokenizer_ok:
        return report
    report.run("data.loader", lambda: _check_loader(state))
    if not report.passed("data.loader"):
        return report

    assert state.data_config is not None
    max_tokens = options.vram_max_tokens or state.data_config.batching.max_tokens
    for path in options.model_configs:
        _run_model_checks(report, _ModelRun(load_model_config(path)), state, options, max_tokens)

    if report.total_seconds > 600:
        report.run(
            "time.budget",
            lambda: Outcome("warn", f"validation took {report.total_seconds}s > 600s"),
            critical=False,
        )
    return report
