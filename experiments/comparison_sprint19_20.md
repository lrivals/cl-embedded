# Comparaison board vs simulation — Sprint 19 & 20

| Expérience | Modèle | λ | Platform | acc_final | avg_forgetting | BWT | RAM (Ko) | Latence (ms) | Gap2 ✓ |
|-----------|--------|---|----------|:---------:|:--------------:|:---:|:--------:|:------------:|:------:|
| E19-01 | mahalanobis | — | nucleo_f439zi | 0.6285 | 0.0000 | -0.0000 | — | 0.0040 | ✅ |
| E19-02 | ewc | — | nucleo_f439zi | 0.0800 | 0.0000 | -0.0000 | — | 0.0040 | ✅ |
| baseline | ewc | 0 | nucleo_f439zi | 0.6118 | 0.3084 | -0.3084 | 9.5 | 5.4418 | ✅ |
| baseline-board | ewc | 0 | nucleo_f439zi | 0.9036 | 0.0542 | -0.0542 | — | 0.2507 | ✅ |
| ewc | ewc | 400 | nucleo_f439zi | 0.7818 | 0.0534 | -0.0534 | 9.5 | 5.4418 | ✅ |
| ewc100 | ewc | 100 | nucleo_f439zi | 0.7818 | 0.0534 | -0.0534 | 9.5 | 5.4418 | ✅ |
| ewc100-board | ewc | 100 | nucleo_f439zi | 0.9016 | 0.0090 | -0.0090 | — | 0.2480 | ✅ |
| ewc400-board | ewc | 400 | nucleo_f439zi | 0.8976 | 0.0090 | -0.0090 | — | 0.2481 | ✅ |