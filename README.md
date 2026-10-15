# Toward Certifiably Robust Face Recognition

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Lab twin of Paik et al.: baseline vs spectral-normalized LLN face recognition with PGD robustness and a certificate proxy.

**Repository:** https://github.com/biometric-community/toward-certifiably-robust-face-recognition-analyses-and-improvements

**Paper:** [10.1109/TBIOM.2025.3644396](https://doi.org/10.1109/TBIOM.2025.3644396)  
**Upstream:** [CertRobFR](https://github.com/Cryptology-Algorithm-Lab/CertRobFR)

## Setup

```bash
bash scripts/setup_env.sh
bash scripts/smoke.sh
bash scripts/train_full.sh
```

See [REPORT.md](./REPORT.md), [DEVIATIONS.md](./DEVIATIONS.md), [SOURCE_CODE.md](./SOURCE_CODE.md).

## License

[CC BY 4.0](./LICENSE)
