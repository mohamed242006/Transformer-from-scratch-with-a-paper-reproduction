from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from aiayn.modules.feedforward import PositionwiseFeedForward


@pytest.fixture
def ffn() -> PositionwiseFeedForward:
    torch.manual_seed(0)
    return PositionwiseFeedForward(8, 32).double().eval()


def test_matches_equation_2(ffn: PositionwiseFeedForward) -> None:
    x = torch.randn(2, 5, 8, dtype=torch.float64)
    hidden = torch.clamp(x @ ffn.linear1.weight.T + ffn.linear1.bias, min=0)
    expected = hidden @ ffn.linear2.weight.T + ffn.linear2.bias
    torch.testing.assert_close(ffn(x), expected, atol=1e-12, rtol=0)


def test_is_position_wise(ffn: PositionwiseFeedForward) -> None:
    x = torch.randn(2, 6, 8, dtype=torch.float64)
    perm = torch.randperm(6)
    torch.testing.assert_close(ffn(x[:, perm]), ffn(x)[:, perm])
    single = ffn(x[:, 3:4])
    torch.testing.assert_close(single, ffn(x)[:, 3:4])


def test_parameter_count_matches_formula() -> None:
    d, d_ff = 512, 2048
    module = PositionwiseFeedForward(d, d_ff)
    assert sum(p.numel() for p in module.parameters()) == 2 * d * d_ff + d_ff + d


def test_inner_dropout_only_in_training() -> None:
    torch.manual_seed(0)
    module = PositionwiseFeedForward(8, 64, dropout=0.5)
    x = torch.randn(4, 8)
    module.eval()
    torch.testing.assert_close(module(x), module(x))
    module.train()
    assert not torch.allclose(module(x), module(x))
    no_dropout = PositionwiseFeedForward(8, 64).train()
    torch.testing.assert_close(no_dropout(x), no_dropout(x))


def test_uses_relu() -> None:
    module = PositionwiseFeedForward(2, 2).double()
    with torch.no_grad():
        module.linear1.weight.copy_(torch.eye(2))
        module.linear2.weight.copy_(torch.eye(2))
    x = torch.tensor([[-3.0, 4.0]], dtype=torch.float64)
    torch.testing.assert_close(module(x), F.relu(x))


def test_validation() -> None:
    with pytest.raises(ValueError):
        PositionwiseFeedForward(0, 4)
    with pytest.raises(ValueError):
        PositionwiseFeedForward(4, 4, dropout=1.0)
    with pytest.raises(ValueError):
        PositionwiseFeedForward(4, 8)(torch.randn(2, 5))
