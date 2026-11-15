"""Config -> data batch -> model, and save/load consistency (ADR-015 §9)."""

from __future__ import annotations

import io

import pytest
import torch

from aiayn.data.config import load_data_config
from aiayn.data.loading import build_dataloader, build_dataset
from aiayn.data.special_tokens import PAD_ID
from aiayn.models import ModelConfig, Transformer, build_transformer, load_model_config
from aiayn.utils.config import config_to_dict, from_dict
from tests.conftest import REPO_ROOT, small_model_config

pytestmark = pytest.mark.integration


def test_copy_task_batches_flow_through_the_tiny_model() -> None:
    data_cfg = load_data_config(
        REPO_ROOT / "configs" / "data" / "copy_task.yaml", ("copy.num_train=300",)
    )
    model_cfg = load_model_config(
        REPO_ROOT / "configs" / "model" / "transformer_tiny.yaml",
        ("d_model=64", "d_ff=128", "num_encoder_layers=1", "num_decoder_layers=1"),
    )
    dataset = build_dataset(data_cfg, "train")
    model = build_transformer(model_cfg, vocab_size=dataset.vocab_size, pad_id=PAD_ID, seed=0)
    model.debug_checks = True  # mask invariants, ID ranges and finite logits on every batch
    model.train()
    batches = 0
    for batch in build_dataloader(data_cfg, dataset, "train"):
        logits = model(batch.src, batch.tgt_in)
        assert logits.shape == (*batch.tgt_out.shape, dataset.vocab_size)
        logits.float().logsumexp(-1).mean().backward()  # backward runs end to end
        batches += 1
    assert batches > 0
    assert all(p.grad is not None for p in model.parameters())


def _round_trip(model: Transformer) -> Transformer:
    """Serialize state + config the way a checkpoint will (Week 5), then rebuild."""
    buffer = io.BytesIO()
    torch.save(
        {
            "config": config_to_dict(model.config),
            "vocab_size": model.vocab_size,
            "pad_id": model.pad_id,
            "state": model.state_dict(),
        },
        buffer,
    )
    buffer.seek(0)
    payload = torch.load(buffer, weights_only=True)
    rebuilt = build_transformer(
        from_dict(ModelConfig, payload["config"]),
        vocab_size=payload["vocab_size"],
        pad_id=payload["pad_id"],
        seed=999,  # different init: everything must come from the state dict
    )
    rebuilt.load_state_dict(payload["state"])
    return rebuilt


@pytest.mark.parametrize(
    "overrides",
    [{}, {"tie_weights": False}, {"norm_position": "pre", "positional_encoding": "learned"}],
)
def test_save_load_reproduces_outputs_exactly(overrides: dict[str, object]) -> None:
    model = build_transformer(small_model_config(**overrides), vocab_size=60, pad_id=0, seed=1)
    with torch.no_grad():  # simulate training having moved every parameter
        for p in model.parameters():
            p.add_(0.01 * torch.randn_like(p))
    model.eval()
    rebuilt = _round_trip(model).eval()
    src = torch.tensor([[5, 6, 7, 3], [8, 3, 0, 0]])
    tgt = torch.tensor([[2, 9, 10], [2, 11, 0]])
    assert torch.equal(model(src, tgt), rebuilt(src, tgt))
    if model.config.tie_weights:
        assert rebuilt.output_weight is rebuilt.src_embedding.weight  # tying survives loading


def test_loading_into_a_different_architecture_fails_loudly() -> None:
    small = build_transformer(small_model_config(), vocab_size=60, pad_id=0, seed=0)
    deeper = build_transformer(
        small_model_config(num_encoder_layers=3), vocab_size=60, pad_id=0, seed=0
    )
    with pytest.raises(RuntimeError, match="Missing key"):
        deeper.load_state_dict(small.state_dict())
    other_vocab = build_transformer(small_model_config(), vocab_size=61, pad_id=0, seed=0)
    with pytest.raises(RuntimeError, match="size mismatch"):
        other_vocab.load_state_dict(small.state_dict())


def test_state_dict_is_device_and_dtype_portable() -> None:
    model = build_transformer(small_model_config(), vocab_size=60, pad_id=0, seed=0).eval()
    as_double = build_transformer(small_model_config(), vocab_size=60, pad_id=0, seed=5).double()
    as_double.load_state_dict(model.state_dict())  # float32 state into a float64 model
    src = torch.tensor([[5, 6, 3]])
    tgt = torch.tensor([[2, 7]])
    torch.testing.assert_close(as_double.eval()(src, tgt).float(), model(src, tgt))
