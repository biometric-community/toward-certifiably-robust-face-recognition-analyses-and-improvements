"""LFW utility + Lip estimate + PGD robust acc + certificate proxy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from certrob.data.celeba import CelebAIdDataset, resolve_celeba_root
from certrob.data.pairs import build_calfw, build_cplfw, build_lfw
from certrob.models.fr_net import FaceEmbedder
from certrob.robust.metrics import certificate_proxy, estimate_lipschitz, pgd_attack_pair


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _device(cfg: dict, override: Optional[str] = None) -> torch.device:
    name = override or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def best_acc_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    thresholds = np.linspace(scores.min(), scores.max(), 200)
    best_acc, best_eer, best_t = 0.0, 1.0, 0.0
    for t in thresholds:
        pred = (scores >= t).astype(np.int32)
        acc = float((pred == labels).mean())
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
        pos, neg = labels == 1, labels == 0
        if pos.any() and neg.any():
            fpr = float((pred[neg] == 1).mean())
            fnr = float((pred[pos] == 0).mean())
            best_eer = min(best_eer, 0.5 * (fpr + fnr))
    return best_acc, best_eer, best_t


def load_model(ckpt: Path, device) -> FaceEmbedder:
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    model = FaceEmbedder(
        int(blob["num_classes"]),
        emb_dim=int(blob.get("emb_dim", 128)),
        pretrained=False,
        spectral=bool(blob.get("spectral", False)),
    ).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model


@torch.no_grad()
def score_pairs(model, loader, device):
    scores, labels = [], []
    for a, b, y in tqdm(loader, desc="verify", leave=False):
        ea = model.embed(a.to(device))
        eb = model.embed(b.to(device))
        scores.append((ea * eb).sum(dim=1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(scores), np.concatenate(labels)


def eval_model(model, cfg, device, tag: str) -> Dict:
    image_size = int(cfg["model"].get("image_size", 112))
    bs = int(cfg["eval"].get("batch_size", 128))
    nw = int(cfg["train"].get("num_workers", 4))
    max_p = cfg["eval"].get("max_lfw_pairs")
    max_p = int(max_p) if max_p is not None else None
    eps = float(cfg["eval"].get("pgd_eps", 8 / 255))
    steps = int(cfg["eval"].get("pgd_steps", 7))
    max_atk = int(cfg["eval"].get("max_attack_pairs", 256))

    out: Dict = {"tag": tag, "verification": {}, "robustness": {}}
    for name, builder in (("lfw", build_lfw), ("calfw", build_calfw), ("cplfw", build_cplfw)):
        try:
            ds = builder(cfg, image_size, max_p)
        except FileNotFoundError as e:
            print(f"[skip] {name}: {e}")
            continue
        loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw)
        s, y = score_pairs(model, loader, device)
        acc, eer, thr = best_acc_eer(s, y)
        out["verification"][name] = {"accuracy": acc, "eer": eer, "threshold": thr, "n_pairs": int(len(y))}
        print(f"[{tag}/{name}] acc={acc:.4f} eer={eer:.4f}")

    # Lip on CelebA val batch
    root = resolve_celeba_root(cfg)
    max_n = cfg["data"].get("max_images")
    max_n = int(max_n) if max_n is not None else None
    val_ds = CelebAIdDataset(
        root,
        split="val",
        image_size=image_size,
        train_ratio=float(cfg["data"].get("train_ratio", 0.7)),
        max_n=max_n,
        seed=int(cfg["train"].get("seed", 42)),
        train=False,
    )
    n_lip = min(len(val_ds), int(cfg["eval"].get("n_lip_images", 64)))
    imgs = torch.stack([val_ds[i][0] for i in range(n_lip)]).to(device)
    lip = estimate_lipschitz(model, imgs, eps=0.05, n_probe=4)
    print(f"[{tag}] lip_est={lip:.4f}")

    # PGD on LFW subset
    lfw = build_lfw(cfg, image_size, max_p)
    n_atk = min(len(lfw), max_atk)
    loader = DataLoader(lfw, batch_size=min(bs, 32), shuffle=False, num_workers=nw)
    clean_ok, rob_ok, n_seen = 0, 0, 0
    thr = out["verification"].get("lfw", {}).get("threshold", 0.0)
    all_scores, all_labels = [], []
    for a, b, y in loader:
        if n_seen >= n_atk:
            break
        a, b, y = a.to(device), b.to(device), y.to(device)
        take = min(a.size(0), n_atk - n_seen)
        a, b, y = a[:take], b[:take], y[:take]
        with torch.no_grad():
            sc = (model.embed(a) * model.embed(b)).sum(dim=1)
            pred = (sc >= thr).long()
            clean_ok += int((pred == y).sum().item())
        adv = pgd_attack_pair(model, a, b, y, eps=eps, steps=steps)
        with torch.no_grad():
            sc_a = (model.embed(adv) * model.embed(b)).sum(dim=1)
            pred_a = (sc_a >= thr).long()
            rob_ok += int((pred_a == y).sum().item())
            all_scores.append(sc.cpu().numpy())
            all_labels.append(y.cpu().numpy())
        n_seen += take
    s_arr = np.concatenate(all_scores) if all_scores else np.array([])
    y_arr = np.concatenate(all_labels) if all_labels else np.array([])
    cert = certificate_proxy(s_arr, y_arr, thr, lip, eps)
    out["robustness"] = {
        "lip_est": lip,
        "pgd_eps": eps,
        "pgd_steps": steps,
        "n_attack_pairs": int(n_seen),
        "clean_acc": clean_ok / max(n_seen, 1),
        "robust_acc": rob_ok / max(n_seen, 1),
        "certificate": cert,
    }
    print(
        f"[{tag}/robust] clean={out['robustness']['clean_acc']:.4f} "
        f"robust={out['robustness']['robust_acc']:.4f} cert={cert['cert_rate']:.4f}"
    )
    return out


def evaluate(cfg: dict, device_override: Optional[str] = None) -> Dict:
    device = _device(cfg, device_override)
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    results = {
        "baseline": eval_model(load_model(ckpt_dir / "baseline.pt", device), cfg, device, "baseline"),
        "lipschitz": eval_model(load_model(ckpt_dir / "lipschitz.pt", device), cfg, device, "lipschitz"),
    }
    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / "eval.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"Wrote {out}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    evaluate(load_config(args.config), device_override=args.device)


if __name__ == "__main__":
    main()
