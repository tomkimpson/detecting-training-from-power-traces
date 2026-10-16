"""E1 pilot (exploratory): variable-G gradient accumulation and the noise-scale
probe give the right gradients. CPU-only; skipped where torch is absent."""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "e1"))
import train  # noqa: E402


def _mse(model, x, y):
    return torch.nn.functional.mse_loss(model(x), y)


def _setup(G, m=4, seed=0):
    torch.manual_seed(seed)
    model = torch.nn.Linear(5, 1)
    batches = [(torch.randn(m, 5), torch.randn(m, 1)) for _ in range(G)]
    return model, batches


def _flat_grad(model):
    return torch.cat([p.grad.flatten() for p in model.parameters()]).clone()


@pytest.mark.parametrize("G", [1, 3, 8])
def test_accumulation_equals_mean_loss_gradient(G):
    model, batches = _setup(G)
    loss, _ = train.accumulate(model, iter(batches), G, _mse)
    acc = _flat_grad(model)

    model.zero_grad()
    x = torch.cat([b[0] for b in batches])
    y = torch.cat([b[1] for b in batches])
    ref_loss = _mse(model, x, y)
    ref_loss.backward()
    assert torch.allclose(acc, _flat_grad(model), atol=1e-6)
    assert abs(loss - ref_loss.item()) < 1e-6


def test_probe_returns_micro_and_mean_squared_norms():
    G = 5
    model, batches = _setup(G)
    _, probe = train.accumulate(model, iter(batches), G, _mse, probe=True)
    acc = _flat_grad(model)

    per = []
    for x, y in batches:
        model.zero_grad()
        _mse(model, x, y).backward()
        per.append(_flat_grad(model))
    per = torch.stack(per)
    small_sq = float((per**2).sum(1).mean())
    big_sq = float((per.mean(0) ** 2).sum())
    assert np.isclose(probe[0], small_sq, rtol=1e-5)
    assert np.isclose(probe[1], big_sq, rtol=1e-5)
    # probing must not change the accumulated gradient
    assert torch.allclose(acc, per.mean(0), atol=1e-6)


def test_lr_schedule_is_token_indexed():
    D, lr = 1000, 1e-3
    assert train.lr_at(0, D, lr) == 0.0
    assert np.isclose(train.lr_at(20, D, lr), lr)            # end of 2% warmup
    assert np.isclose(train.lr_at(D, D, lr), 0.1 * lr)       # cosine floor
    assert train.lr_at(500, D, lr) < lr
