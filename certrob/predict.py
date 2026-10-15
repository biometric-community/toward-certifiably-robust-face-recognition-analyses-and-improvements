"""Embed one image with lipschitz checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from certrob.models.fr_net import FaceEmbedder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", default="lipschitz", choices=["baseline", "lipschitz"])
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(Path(cfg["paths"]["checkpoint_dir"]) / f"{args.model}.pt", map_location=device, weights_only=False)
    model = FaceEmbedder(
        int(blob["num_classes"]), emb_dim=int(blob.get("emb_dim", 128)), pretrained=False, spectral=bool(blob.get("spectral", False))
    ).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    size = int(cfg["model"].get("image_size", 112))
    tf = transforms.Compose(
        [transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )
    x = tf(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.embed(x)[0].cpu().tolist()
    out_dir = Path(cfg["paths"]["pred_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "embedding.json").write_text(json.dumps({"model": args.model, "embedding": emb}) + "\n")
    print(f"Wrote {out_dir / 'embedding.json'}")


if __name__ == "__main__":
    main()
