# DEVIATIONS — Certifiably Robust FR

| ID | Item | Paper | Ours | Reason |
|----|------|-------|------|--------|
| D1 | Train set | MS1M / WebFace scale | CelebA | Local <5 GiB gate |
| D2 | Backbone | SLL / iResNet Lip nets | ResNet-18 + spectral_norm FC/cls | Tractable lab twin |
| D3 | Header | Full LLN analysis | L2-normalized embeddings | Core LLN idea |
| D4 | Certificates | Theorem 3 on a 1-Lipschitz backbone | Theorem 3 with K from Jacobian power iteration | Backbone is not 1-Lipschitz, so K is local and certificates are not guarantees |
| D5 | Attacks | l2 PGD / C&W suite | l2 PGD on LFW pairs | Scope |
