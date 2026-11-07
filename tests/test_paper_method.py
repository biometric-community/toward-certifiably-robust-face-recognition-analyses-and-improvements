"""Certificate, Lipschitz estimate and attack, checked against Theorem 3 and Sec. 4."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from certrob.eval import best_acc_eer
from certrob.models.fr_net import FaceEmbedder
from certrob.robust.metrics import certificate_proxy, estimate_lipschitz, pgd_attack_pair


def certified(score: float, label: int, eps: float, threshold: float = 0.5, lip: float = 1.0) -> float:
    result = certificate_proxy(np.array([score]), np.array([label]), threshold, lip, eps)
    return result["cert_rate"]


def theorem_3_radius(score: float, tau: float, norm: float = 1.0) -> float:
    theta = math.acos(score)
    gap = math.sqrt(1 - tau**2) * math.cos(theta) - tau * math.sin(theta)
    return norm * abs(gap)


@pytest.mark.parametrize("eps,expected", [(0.3, 1.0), (0.6, 0.0)])
def test_dodging_criterion_follows_theorem_3(eps, expected):
    """A mated pair at cos 0.9 and threshold 0.5 has a certified radius of 0.561."""
    assert theorem_3_radius(0.9, 0.5) == pytest.approx(0.561, abs=1e-3)
    assert certified(0.9, 1, eps) == expected


@pytest.mark.parametrize("eps,expected", [(0.25, 1.0), (0.4, 0.0)])
def test_impersonation_criterion_follows_theorem_3(eps, expected):
    """A non-mated pair at cos 0.2 and threshold 0.5 has a certified radius of 0.317."""
    assert theorem_3_radius(0.2, 0.5) == pytest.approx(0.317, abs=1e-3)
    assert certified(0.2, 0, eps) == expected


def test_a_wrong_decision_is_never_certified():
    assert certified(0.3, 1, 0.01) == 0.0
    assert certified(0.7, 0, 0.01) == 0.0


class LinearEmbedder(nn.Module):
    """A map with a known Lipschitz constant: its largest singular value."""

    def __init__(self, gains: torch.Tensor):
        super().__init__()
        self.gains = gains

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(1) * self.gains


def test_lipschitz_estimate_does_not_underestimate_a_known_constant():
    """A certificate needs an upper bound; random directions only give a lower one."""
    gains = torch.ones(3 * 16 * 16) * 0.1
    gains[0] = 10.0
    images = torch.rand(8, 3, 16, 16)
    assert estimate_lipschitz(LinearEmbedder(gains), images) >= 10.0 * 0.99


class ConvEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 4, 3), nn.Flatten())

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.net(x), dim=1)


def test_pgd_respects_an_l2_noise_bound():
    """Sec. 4: PGD runs in the l2 norm, the norm the certificate is stated in."""
    torch.manual_seed(0)
    a, b = torch.rand(2, 3, 16, 16) * 2 - 1, torch.rand(2, 3, 16, 16) * 2 - 1
    adv = pgd_attack_pair(ConvEmbedder(), a, b, torch.tensor([1, 0]), eps=0.5, steps=5, step_size=0.2)
    assert (adv - a).flatten(1).norm(dim=1).max().item() <= 0.5 + 1e-4


def test_pgd_lowers_the_score_of_a_mated_pair():
    torch.manual_seed(1)
    model = ConvEmbedder()
    a = torch.rand(1, 3, 16, 16) * 2 - 1
    adv = pgd_attack_pair(model, a, a.clone(), torch.tensor([1]), eps=0.1, steps=5, step_size=0.05)
    with torch.no_grad():
        before = (model.embed(a) * model.embed(a)).sum()
        after = (model.embed(adv) * model.embed(a)).sum()
    assert after < before


def test_spectral_layers_are_normalised():
    model = FaceEmbedder(5, emb_dim=8, pretrained=False, spectral=True).train()
    for _ in range(30):
        model(torch.rand(2, 3, 64, 64))
    assert torch.linalg.matrix_norm(model.fc.weight, ord=2).item() == pytest.approx(1.0, abs=0.05)


def test_embeddings_have_unit_norm():
    model = FaceEmbedder(5, emb_dim=8, pretrained=False).eval()
    with torch.no_grad():
        assert torch.allclose(model.embed(torch.rand(2, 3, 64, 64)).norm(dim=1), torch.ones(2), atol=1e-5)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def test_eer_is_the_point_where_fmr_equals_fnmr():
    lo, hi = 0.0, 1.0
    for _ in range(80):
        t = 0.5 * (lo + hi)
        fmr, fnmr = 1.0 - _norm_cdf(t), _norm_cdf((t - 1.0) / 0.1)
        lo, hi = (t, hi) if fmr > fnmr else (lo, t)
    expected = 1.0 - _norm_cdf(t)

    rng = np.random.RandomState(3)
    scores = np.concatenate([rng.normal(1.0, 0.1, 200000), rng.normal(0.0, 1.0, 200000)])
    labels = np.concatenate([np.ones(200000, dtype=int), np.zeros(200000, dtype=int)])
    assert best_acc_eer(scores, labels)[1] == pytest.approx(expected, abs=0.01)
