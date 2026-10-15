# REPORT — Certifiably Robust Face Recognition

**Protocol:** `full`

Lab twin of Paik et al.: baseline vs spectral-normalized LLN FR,
empirical Lip, PGD pair attack, and L2 certificate proxy on CelebA/LFW.

## Training

- **baseline**: epochs=6, n=141610, loss=8.6508
- **lipschitz**: epochs=6, n=141610, loss=8.8125

![Training](outputs/figures/full/train_loss.svg)

## Verification

| Model | Dataset | Accuracy | EER |
|-------|---------|----------|-----|
| baseline | calfw | 54.05% | 45.95% |
| baseline | cplfw | 53.95% | 46.05% |
| baseline | lfw | 62.83% | 37.17% |
| lipschitz | calfw | 54.78% | 45.22% |
| lipschitz | cplfw | 56.72% | 43.28% |
| lipschitz | lfw | 76.22% | 23.78% |

## Robustness

| Model | Lip est | Clean | PGD robust | Cert rate |
|-------|---------|-------|------------|-----------|
| baseline | 0.0033 | 60.94% | 1.17% | 60.94% |
| lipschitz | 0.0140 | 76.56% | 0.00% | 76.56% |

![Robust](outputs/figures/full/robustness.svg)

## Notes

- Paper uses SLL / full Lip architectures + MS1M; we use spectral_norm + LLN on CelebA.
- See `DEVIATIONS.md` and upstream `SOURCE_CODE.md`.
