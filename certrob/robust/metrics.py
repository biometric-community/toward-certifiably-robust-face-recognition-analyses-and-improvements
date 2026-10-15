"""Empirical Lip estimate, PGD on pairs, simple L2 certificate proxy."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def estimate_lipschitz(model, images: torch.Tensor, eps: float = 0.05, n_probe: int = 4) -> float:
    """Max ||f(x+δ)-f(x)||_2 / ||δ||_2 over random L2 probes (batch mean of max)."""
    model.eval()
    emb0 = model.embed(images)
    ratios = []
    for _ in range(n_probe):
        noise = torch.randn_like(images)
        noise = noise / (noise.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-8) * eps
        emb1 = model.embed(images + noise)
        num = (emb1 - emb0).norm(dim=1)
        den = noise.flatten(1).norm(dim=1).clamp_min(1e-8)
        ratios.append((num / den).cpu())
    return float(torch.stack(ratios).max(dim=0).values.mean().item())


def pgd_attack_pair(
    model,
    a: torch.Tensor,
    b: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 8 / 255,
    steps: int = 7,
    step_size: float = 2 / 255,
) -> torch.Tensor:
    """
    Untargeted PGD on image A to flip verification decision vs B (cosine).
    labels: 1=same, 0=diff. Returns adversarial A.
    """
    model.eval()
    a0 = a.detach()
    adv = a0.clone().detach().requires_grad_(True)
    with torch.no_grad():
        eb = model.embed(b)
    for _ in range(steps):
        ea = model.embed(adv)
        score = (ea * eb).sum(dim=1)
        # maximize loss for correct decision: same → decrease score; diff → increase
        loss = torch.where(labels > 0, -score, score).mean()
        model.zero_grad(set_to_none=True)
        if adv.grad is not None:
            adv.grad.zero_()
        loss.backward()
        with torch.no_grad():
            adv = adv + step_size * adv.grad.sign()
            delta = (adv - a0).clamp(-eps, eps)
            adv = (a0 + delta).clamp(-1.0, 1.0).detach().requires_grad_(True)
    return adv.detach()


@torch.no_grad()
def certificate_proxy(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    lip: float,
    eps: float,
) -> Dict[str, float]:
    """
    Naive certificate: genuine pair certified if score - thr > 2*lip*eps
    (embedding L2 change ≤ lip*eps each side upper-bounds cosine shift loosely).
    """
    margin = scores - threshold
    # for same pairs, need score stay above thr; for diff, below
    cert = np.zeros_like(scores, dtype=bool)
    bud = 2.0 * max(lip, 1e-6) * eps
    same = labels == 1
    diff = labels == 0
    cert[same] = margin[same] > bud
    cert[diff] = (-margin[diff]) > bud
    return {
        "lip_est": float(lip),
        "eps": float(eps),
        "budget": float(bud),
        "cert_rate": float(cert.mean()) if len(cert) else 0.0,
        "cert_rate_same": float(cert[same].mean()) if same.any() else 0.0,
        "cert_rate_diff": float(cert[diff].mean()) if diff.any() else 0.0,
    }
