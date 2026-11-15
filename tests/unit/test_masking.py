"""Mask construction and semantics (ADR-008).

The semantic tests use a tiny reference masked softmax defined *here* (not in src/)
to prove that the masks block exactly the intended information flow.
"""

from __future__ import annotations

import pytest
import torch

from aiayn.data.special_tokens import PAD_ID
from aiayn.modules.masking import (
    assert_valid_attention_mask,
    causal_mask,
    decoder_attention_mask,
    encoder_attention_mask,
    padding_mask,
)

# Two sequences: lengths 3 and 5, padded to 5.
TOKENS = torch.tensor([[5, 6, 7, PAD_ID, PAD_ID], [5, 6, 7, 8, 9]])


def reference_attention(
    scores: torch.Tensor, values: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """softmax(scores masked to -inf where mask is False) @ values."""
    weights = scores.masked_fill(~mask, float("-inf")).softmax(dim=-1)
    return weights @ values


def test_padding_mask_shape_dtype_values() -> None:
    m = padding_mask(TOKENS, PAD_ID)
    assert m.shape == (2, 1, 1, 5) and m.dtype == torch.bool
    assert m[0, 0, 0].tolist() == [True, True, True, False, False]
    assert m[1].all()


def test_padding_mask_requires_rank_2() -> None:
    with pytest.raises(ValueError):
        padding_mask(TOKENS[0], PAD_ID)


def test_causal_mask_is_lower_triangular_including_diagonal() -> None:
    m = causal_mask(4)
    assert m.shape == (1, 1, 4, 4) and m.dtype == torch.bool
    expected = torch.tensor(
        [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool
    )
    assert torch.equal(m[0, 0], expected)
    assert causal_mask(1).item() is True


def test_causal_mask_rejects_empty() -> None:
    with pytest.raises(ValueError):
        causal_mask(0)


def test_decoder_mask_combines_causal_and_padding() -> None:
    m = decoder_attention_mask(TOKENS, PAD_ID)
    assert m.shape == (2, 1, 5, 5)
    # Sequence 0: keys 3, 4 are padding -> never attendable, even by later queries.
    assert not m[0, 0, :, 3:].any()
    # Query 2 of sequence 0 sees keys 0..2.
    assert m[0, 0, 2].tolist() == [True, True, True, False, False]
    # Sequence 1 (no padding) equals the pure causal mask.
    assert torch.equal(m[1, 0], causal_mask(5)[0, 0])


def test_encoder_mask_broadcasts_to_scores_for_self_and_cross_attention() -> None:
    m = encoder_attention_mask(TOKENS, PAD_ID)
    batch, heads, num_queries, num_keys = 2, 4, 7, 5  # T != S for cross-attention
    scores = torch.zeros(batch, heads, num_queries, num_keys)
    assert scores.masked_fill(~m, -1.0).shape == (batch, heads, num_queries, num_keys)


def test_every_query_row_has_an_attendable_key() -> None:
    # Guaranteed by framing: sources end with EOS, decoder inputs start with BOS.
    assert_valid_attention_mask(encoder_attention_mask(TOKENS, PAD_ID), num_keys=5)
    assert_valid_attention_mask(decoder_attention_mask(TOKENS, PAD_ID), num_queries=5, num_keys=5)


@pytest.mark.parametrize(
    ("mask", "message"),
    [
        (torch.ones(1, 1, 1, 3, dtype=torch.int64), "torch.bool"),
        (torch.ones(1, 1, 3, dtype=torch.bool), "rank 4"),
        (torch.ones(1, 2, 1, 3, dtype=torch.bool), "head dimension"),
        (torch.tensor([[[[True, False], [False, False]]]]), "no attendable key"),
    ],
)
def test_validator_rejects_bad_masks(mask: torch.Tensor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        assert_valid_attention_mask(mask)


def test_validator_checks_sizes() -> None:
    m = encoder_attention_mask(TOKENS, PAD_ID)
    with pytest.raises(ValueError, match="key dimension"):
        assert_valid_attention_mask(m, num_keys=4)


def test_masks_follow_input_device() -> None:
    assert decoder_attention_mask(TOKENS, PAD_ID).device == TOKENS.device


# ---------------------------------------------------------------------- #
# Semantic tests: information flow
# ---------------------------------------------------------------------- #
def test_causal_mask_blocks_information_from_the_future() -> None:
    torch.manual_seed(0)
    t = 6
    scores, values = torch.randn(1, 1, t, t), torch.randn(1, 1, t, 3)
    mask = causal_mask(t)
    base = reference_attention(scores, values, mask)
    for future in range(1, t):
        perturbed = values.clone()
        perturbed[..., future, :] += 100.0
        out = reference_attention(scores, perturbed, mask)
        # Outputs at positions < future are unchanged; position `future` is affected.
        torch.testing.assert_close(out[..., :future, :], base[..., :future, :])
        assert not torch.allclose(out[..., future, :], base[..., future, :])


def test_padding_mask_makes_padded_keys_irrelevant() -> None:
    torch.manual_seed(0)
    mask = encoder_attention_mask(TOKENS, PAD_ID)
    scores, values = torch.randn(2, 1, 5, 5), torch.randn(2, 1, 5, 3)
    base = reference_attention(scores, values, mask)
    perturbed_scores, perturbed_values = scores.clone(), values.clone()
    perturbed_scores[0, ..., 3:] = 1e4  # huge scores on padded keys of sequence 0
    perturbed_values[0, :, 3:, :] = 1e4
    out = reference_attention(perturbed_scores, perturbed_values, mask)
    torch.testing.assert_close(out, base)
    assert torch.isfinite(out).all()
