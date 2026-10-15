# IMPLEMENTATION_PLAN — CertRobFR Lab Twin

| Component | Choice |
|-----------|--------|
| Baseline | ResNet-18 CE + LLN emb |
| Lip model | spectral_norm on FC/classifier |
| Robust eval | PGD on LFW pairs |
| Certificate | margin vs 2·Lip·ε proxy |
