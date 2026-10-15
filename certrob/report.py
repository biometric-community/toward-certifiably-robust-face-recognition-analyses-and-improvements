"""Figures + REPORT.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from certrob.plot_style import FIGSIZE, LINEWIDTH_MAIN, MULTI_SERIES_COLORS, apply_rcparams, apply_style


def plot_train(summary, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, key in enumerate(("baseline", "lipschitz")):
        hist = summary[key]["history"]
        ax.plot(
            [h["epoch"] for h in hist],
            [h["loss"] for h in hist],
            color=MULTI_SERIES_COLORS[i],
            linewidth=LINEWIDTH_MAIN,
            label=key,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CE loss")
    ax.set_title("Baseline vs spectral Lip FR")
    ax.legend()
    apply_style(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def plot_robust(eval_res, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    tags = ["baseline", "lipschitz"]
    clean = [eval_res[t]["robustness"]["clean_acc"] * 100 for t in tags]
    rob = [eval_res[t]["robustness"]["robust_acc"] * 100 for t in tags]
    cert = [eval_res[t]["robustness"]["certificate"]["cert_rate"] * 100 for t in tags]
    x = np.arange(len(tags))
    w = 0.25
    ax.bar(x - w, clean, w, label="clean", color=MULTI_SERIES_COLORS[0], edgecolor="#002080")
    ax.bar(x, rob, w, label="PGD robust", color=MULTI_SERIES_COLORS[1], edgecolor="#002080")
    ax.bar(x + w, cert, w, label="cert rate", color=MULTI_SERIES_COLORS[2], edgecolor="#002080")
    ax.set_xticks(x)
    ax.set_xticklabels(tags)
    ax.set_ylabel("%")
    ax.set_title("Robustness comparison (LFW subset)")
    ax.legend()
    apply_style(ax, grid_axis="y")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def write_report(cfg, train_s, eval_res, fig_dir: Path, path: Path) -> None:
    protocol = cfg["train"].get("protocol", "full")
    lines = [
        "# REPORT — Certifiably Robust Face Recognition",
        "",
        f"**Protocol:** `{protocol}`",
        "",
        "Lab twin of Paik et al.: baseline vs spectral-normalized LLN FR,",
        "empirical Lip, PGD pair attack, and L2 certificate proxy on CelebA/LFW.",
        "",
        "## Training",
        "",
    ]
    for key in ("baseline", "lipschitz"):
        t = train_s[key]
        lines.append(f"- **{key}**: epochs={t['epochs']}, n={t['n_train']}, loss={t['history'][-1]['loss']:.4f}")
    lines += [
        "",
        f"![Training]({fig_dir.as_posix()}/train_loss.svg)",
        "",
        "## Verification",
        "",
        "| Model | Dataset | Accuracy | EER |",
        "|-------|---------|----------|-----|",
    ]
    for tag in ("baseline", "lipschitz"):
        for ds, v in sorted(eval_res[tag].get("verification", {}).items()):
            lines.append(f"| {tag} | {ds} | {v['accuracy']*100:.2f}% | {v['eer']*100:.2f}% |")
    lines += [
        "",
        "## Robustness",
        "",
        "| Model | Lip est | Clean | PGD robust | Cert rate |",
        "|-------|---------|-------|------------|-----------|",
    ]
    for tag in ("baseline", "lipschitz"):
        r = eval_res[tag]["robustness"]
        lines.append(
            f"| {tag} | {r['lip_est']:.4f} | {r['clean_acc']*100:.2f}% | "
            f"{r['robust_acc']*100:.2f}% | {r['certificate']['cert_rate']*100:.2f}% |"
        )
    lines += [
        "",
        f"![Robust]({fig_dir.as_posix()}/robustness.svg)",
        "",
        "## Notes",
        "",
        "- Paper uses SLL / full Lip architectures + MS1M; we use spectral_norm + LLN on CelebA.",
        "- See `DEVIATIONS.md` and upstream `SOURCE_CODE.md`.",
        "",
    ]
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


def generate(cfg: dict) -> None:
    log_dir = Path(cfg["paths"]["log_dir"])
    fig_dir = Path(cfg["paths"]["figure_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    train_s = json.loads((log_dir / "train_summary.json").read_text())
    eval_res = json.loads((log_dir / "eval.json").read_text())
    plot_train(train_s, fig_dir / "train_loss.svg")
    plot_robust(eval_res, fig_dir / "robustness.svg")
    write_report(cfg, train_s, eval_res, fig_dir, Path("REPORT.md"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    generate(yaml.safe_load(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
