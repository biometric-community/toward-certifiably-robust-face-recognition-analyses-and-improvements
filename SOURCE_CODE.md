# SOURCE_CODE

Upstream: https://github.com/Cryptology-Algorithm-Lab/CertRobFR (ECCV 2024 / TBIOM extension).

This lab twin does **not** vendor InsightFace/SLL stacks. It reimplements a proof-of-concept under `certrob/`:
- spectral-normalized Linear layers + LLN embeddings
- PGD pair attacks and a simple L2 certificate proxy

Reference only; training uses CelebA instead of MS1M.
