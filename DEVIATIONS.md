# DEVIATIONS — Certifiably Robust FR

| ID | Item | Paper | Ours | Reason |
|----|------|-------|------|--------|
| D1 | Train set | MS1M / WebFace scale | CelebA | Local <5 GiB gate |
| D2 | Backbone | SLL / iResNet Lip nets | ResNet-18 + spectral_norm FC/cls | Tractable lab twin |
| D3 | Header | Full LLN analysis | L2-normalized embeddings | Core LLN idea |
| D4 | Certificates | Paper formal bounds | Margin / (2 Lip_est eps) proxy | Empirical proxy |
| D5 | Attacks | PGD / C&W suite | PGD on LFW pairs | Scope |
