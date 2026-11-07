"""Local Lipschitz estimate, l2 PGD on pairs, Theorem 3 certificate."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch


def _per_sample_norm(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, *([1] * (x.dim() - 1)))


def estimate_lipschitz(model, images: torch.Tensor, n_iter: int = 10, h: float = 1e-3) -> float:
    """Largest Jacobian singular value over ``images``, by power iteration.

    Exact for a linear map. For a deep network this is the local constant at these
    images, which can still be below the global one (see DEVIATIONS D4).
    """
    model.eval()
    v = torch.randn_like(images)
    v = v / _per_sample_norm(v)
    sigma = images.new_zeros(images.shape[0])
    with torch.enable_grad():
        for _ in range(n_iter):
            x = images.detach().requires_grad_(True)
            emb = model.embed(x)
            with torch.no_grad():
                jv = (model.embed(images + h * v) - emb) / h
            sigma = jv.flatten(1).norm(dim=1)
            (g,) = torch.autograd.grad((emb * jv).sum(), x)
            v = g / _per_sample_norm(g)
    return float(sigma.max().item())


def pgd_attack_pair(
    model,
    a: torch.Tensor,
    b: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1.5,
    steps: int = 20,
    step_size: Optional[float] = None,
) -> torch.Tensor:
    """
    Untargeted l2 PGD on image A to flip verification decision vs B (cosine).
    labels: 1=same, 0=diff. Returns adversarial A.
    """
    model.eval()
    step = 2.5 * eps / steps if step_size is None else step_size
    a0 = a.detach()
    adv = a0.clone().detach().requires_grad_(True)
    with torch.no_grad():
        eb = model.embed(b)
    for _ in range(steps):
        ea = model.embed(adv)
        score = (ea * eb).sum(dim=1)
        # maximize loss for correct decision: same → decrease score; diff → increase
        loss = torch.where(labels > 0, -score, score).sum()
        (grad,) = torch.autograd.grad(loss, adv)
        with torch.no_grad():
            delta = adv + step * grad / _per_sample_norm(grad) - a0
            delta = delta * (eps / _per_sample_norm(delta)).clamp(max=1.0)
            adv = (a0 + delta).clamp(-1.0, 1.0).detach().requires_grad_(True)
    return adv.detach()


def certificate_proxy(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    lip: float,
    eps: float,
    feature_norm: float = 1.0,
) -> Dict[str, float]:
    """
    Theorem 3 criteria for a K-Lipschitz backbone, with M = feature_norm / K.
    Mated pairs certify against dodging, non-mated pairs against impersonation.
    """
    scores = np.clip(np.asarray(scores, dtype=np.float64), -1.0, 1.0)
    labels = np.asarray(labels)
    tau = float(threshold)
    m = feature_norm / max(lip, 1e-12)
    # sqrt(1 - tau^2) cos(theta) - tau sin(theta), signed towards the mated side
    gap = np.sqrt(1.0 - tau**2) * scores - tau * np.sqrt(1.0 - scores**2)
    same = labels == 1
    diff = labels == 0
    cert = np.zeros(scores.shape, dtype=bool)
    cert[same] = (scores[same] > tau) & (m * gap[same] > eps)
    cert[diff] = (scores[diff] < tau) & (-m * gap[diff] > eps)
    return {
        "lip_est": float(lip),
        "eps": float(eps),
        "cert_rate": float(cert.mean()) if len(cert) else 0.0,
        "cert_rate_same": float(cert[same].mean()) if same.any() else 0.0,
        "cert_rate_diff": float(cert[diff].mean()) if diff.any() else 0.0,
    }
