"""REPORT.md, the committed output of the full run, against the published results."""

from __future__ import annotations

import re

import pytest


def robustness(report: str, model: str) -> dict:
    match = re.search(rf"\| {model} \| ([\d.]+) \| ([\d.]+)% \| ([\d.]+)% \| ([\d.]+)% \|", report)
    assert match, f"{model} robustness row missing from REPORT.md"
    lip, clean, robust, cert = (float(g) for g in match.groups())
    return {"lip": lip, "clean": clean, "robust": robust, "cert": cert}


def lfw_accuracy(report: str, model: str) -> float:
    match = re.search(rf"\| {model} \| lfw \| ([\d.]+)% \|", report)
    assert match, f"{model} LFW row missing from REPORT.md"
    return float(match.group(1))


@pytest.mark.parametrize("model", ["baseline", "lipschitz"])
def test_certified_pairs_survive_the_attack(report, model):
    """A certified pair cannot be broken at that noise bound, so certified <= robust."""
    row = robustness(report, model)
    assert row["cert"] <= row["robust"]


def test_lipschitz_model_is_more_robust_than_the_baseline(report):
    """Sec. 4: the 1-Lipschitz model keeps accuracy under attacks that break the others."""
    assert robustness(report, "lipschitz")["robust"] > robustness(report, "baseline")["robust"]


def test_lipschitz_constant_is_lower_for_the_lipschitz_model(report):
    assert robustness(report, "lipschitz")["lip"] < robustness(report, "baseline")["lip"]


def test_lfw_accuracy_reaches_the_published_tar(report, targets):
    """Table 1: TAR@FAR=1e-3 is 83.97%, and accuracy is at least that high."""
    assert lfw_accuracy(report, "lipschitz") >= targets["lfw_tar_far_1e3"]
