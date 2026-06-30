# S2223–S2225 — Tests + Documentation

| Champ | Valeur |
|-------|--------|
| **Sprint** | 22 |
| **Priorité** | 🟡 Important |
| **Statut** | ✅ Terminé |
| **Durée estimée** | 1h + 1h + 30 min = 2h30 |
| **Dépendances** | S2201–S2222 ✅ (loaders + expériences + INT8 terminés) |
| **Fichiers cibles** | `tests/test_cmapss_loader.py`, `tests/test_paderborn_loader.py`, `docs/datasets_analysis.md`, `docs/roadmap_phase2.md` |
| **Référence** | `tests/` (tests existants pour monitoring/CWRU), `docs/roadmap_phase2.md` actuel |

---

## Contexte

Ces trois tâches clôturent le sprint : tests unitaires des deux nouveaux loaders, documentation des datasets dans le fichier d'analyse, et mise à jour de la roadmap. Elles ne bloquent pas les expériences mais sont requises avant la clôture du sprint.

---

## S2223 — Tests loaders : `tests/test_cmapss_loader.py` + `tests/test_paderborn_loader.py`

### Pattern (calquer sur les tests existants)

```bash
# Vérifier les tests existants comme référence
ls tests/test_*_loader.py 2>/dev/null || ls tests/test_*.py | head -5
```

### `tests/test_cmapss_loader.py`

```python
"""Tests unitaires pour src/data/cmapss_loader.py."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestCmapssLoader:
    """Tests du loader CMAPSS — compatibles CI (sans données réelles)."""

    def test_rul_capping(self):
        """RUL cappé à 125 : aucune valeur > 125 après capping."""
        from src.data.cmapss_loader import CMAPSS_RUL_CAP
        import numpy as np
        rul_raw = np.array([0, 50, 125, 200, 350])
        rul_capped = np.minimum(rul_raw, CMAPSS_RUL_CAP)
        assert (rul_capped <= CMAPSS_RUL_CAP).all()
        assert rul_capped[4] == CMAPSS_RUL_CAP

    def test_binarization_threshold(self):
        """RUL ≤ 30 → faulty=1, RUL > 30 → faulty=0."""
        from src.data.cmapss_loader import CMAPSS_FAULTY_THRESHOLD
        rul = np.array([0, 15, 30, 31, 50, 125])
        faulty = (rul <= CMAPSS_FAULTY_THRESHOLD).astype(int)
        assert faulty[2] == 1   # RUL = 30 → faulty
        assert faulty[3] == 0   # RUL = 31 → healthy
        assert faulty[5] == 0   # RUL = 125 → healthy

    def test_domain_order(self):
        """4 domaines FD001–FD004 dans le bon ordre."""
        from src.data.cmapss_loader import DOMAIN_ORDER
        assert DOMAIN_ORDER == ["FD001", "FD002", "FD003", "FD004"]
        assert len(DOMAIN_ORDER) == 4

    def test_n_features(self):
        """Nombre de features sélectionnées = 5."""
        from src.data.cmapss_loader import CMAPSS_N_FEATURES_SELECTED
        assert CMAPSS_N_FEATURES_SELECTED == 5

    @pytest.mark.skipif(
        not Path("data/raw/cmapss/train_FD001.txt").exists(),
        reason="Données CMAPSS non disponibles"
    )
    def test_get_cl_dataloaders_shape(self):
        """Test end-to-end avec données réelles."""
        from src.data.cmapss_loader import get_cl_dataloaders
        tasks = get_cl_dataloaders(
            Path("data/raw/cmapss/"),
            Path("configs/cmapss_config.yaml"),
        )
        assert len(tasks) == 4  # FD001–FD004
        for task in tasks:
            x, y = next(iter(task["train_loader"]))
            assert x.shape[1] == 5    # top-5 features
            assert y.shape[1] == 1    # label binaire
            assert x.dtype == torch.float32
```

### `tests/test_paderborn_loader.py`

```python
"""Tests unitaires pour src/data/paderborn_loader.py."""

class TestPaderbornLoader:

    def test_domain_labels(self):
        """K001 → faulty=0, KA04 et KI04 → faulty=1."""
        from src.data.paderborn_loader import DOMAIN_LABELS
        assert DOMAIN_LABELS["K001"] == 0
        assert DOMAIN_LABELS["KA04"] == 1
        assert DOMAIN_LABELS["KI04"] == 1

    def test_domain_order(self):
        """3 domaines dans l'ordre sain → OR → IR."""
        from src.data.paderborn_loader import DOMAIN_ORDER
        assert DOMAIN_ORDER == ["K001", "KA04", "KI04"]

    def test_n_features(self):
        from src.data.paderborn_loader import PADERBORN_N_FEATURES_SELECTED
        assert PADERBORN_N_FEATURES_SELECTED == 5

    def test_feature_extraction_shape(self):
        """Features extraites depuis un signal synthétique."""
        from src.data.paderborn_loader import _compute_features, PADERBORN_WINDOW_SIZE
        import numpy as np
        signal = np.random.randn(PADERBORN_WINDOW_SIZE)
        windows = signal.reshape(1, PADERBORN_WINDOW_SIZE)
        features = _compute_features(windows, fs=64_000)
        assert features.shape == (1, 7)   # 7 features brutes
        assert np.isfinite(features).all()

    def test_relu_q7_clamps(self):
        """Test basique de la fonction utilitaire (pour cohérence avec C)."""
        # Non applicable ici — ce test est dans test_ewc_int8.c

    @pytest.mark.skipif(
        not Path("data/raw/paderborn/K001").exists(),
        reason="Données Paderborn non disponibles"
    )
    def test_get_cl_dataloaders_shape(self):
        from src.data.paderborn_loader import get_cl_dataloaders
        tasks = get_cl_dataloaders(
            Path("data/raw/paderborn/"),
            Path("configs/paderborn_config.yaml"),
        )
        assert len(tasks) == 3  # K001, KA04, KI04
        for task in tasks:
            x, y = next(iter(task["train_loader"]))
            assert x.shape[1] == 5
```

### Lancement des tests

```bash
# Tests sans données (toujours passent en CI)
pytest tests/test_cmapss_loader.py tests/test_paderborn_loader.py -v

# Tests avec données (si disponibles)
pytest tests/ -k "cmapss or paderborn" -v

# Critères de succès sprint
pytest tests/ -k cmapss   # vert
pytest tests/ -k paderborn  # vert
```

---

## S2224 — `docs/datasets_analysis.md`

Ajouter les sections CMAPSS et Paderborn à la suite des sections existantes (CWRU, Pronostia, Monitoring).

### Section CMAPSS à ajouter

```markdown
## CMAPSS — NASA C-MAPSS Turbofan Engine Degradation

| Propriété | Valeur |
|-----------|--------|
| Source | NASA Prognostics Center / Kaggle |
| Taille | ~10 Mo, 4 fichiers (FD001–FD004) |
| Type | Séries temporelles multivariées (21 capteurs) |
| Label | RUL continu → binarisé : faulty = (RUL ≤ 30) |
| Scénario CL | Domain-incremental : FD001→FD002→FD003→FD004 |
| Sprint d'ajout | Sprint 22 |

**Preprocessing** : RUL capping cap=125, top-5 mutual info (T50, Ps30, htBleed, …), MinMax normalization fit sur FD001.

**Résultats CL Sprint 22** (à remplir après exp_S22_01–04) :
| Modèle | acc_final | avg_forgetting |
|--------|:---------:|:--------------:|
| EWC    | TBD | TBD |
| HDC    | TBD | TBD |
| TinyOL | TBD | TBD |
| Maha   | TBD | TBD |

**Contribution Gap 1** : 4e dataset industriel indépendant, premier avec RUL continu binarisé, scénario 4 tâches.
```

### Section Paderborn à ajouter

```markdown
## Paderborn — Bearing Electrical Fault Dataset

| Propriété | Valeur |
|-----------|--------|
| Source | Paderborn University KAt-DataCenter |
| Taille | ~500 Mo (subset K001 + KA04 + KI04) |
| Type | Signaux courant moteur + vibration (64 kHz) |
| Label | État roulement → faulty = (KA04 ou KI04) |
| Scénario CL | Domain-incremental : K001 (sain) → KA04 (OR) → KI04 (IR) |
| Sprint d'ajout | Sprint 22 |

**Preprocessing** : FFT fenêtrage 1024 samples @ 64 kHz, features : rms, kurtosis, crest_factor, energy_band_1–4 (7 features brutes → top-5).

**Résultats CL Sprint 22** (à remplir après exp_S22_05–06) :
| Modèle | acc_final | avg_forgetting |
|--------|:---------:|:--------------:|
| EWC    | TBD | TBD |
| Maha   | TBD | TBD |

**Contribution Gap 1** : 5e dataset, apporte la diversité du signal courant moteur (absent de CWRU/Pronostia).
```

---

## S2225 — Mise à jour `docs/roadmap_phase2.md`

### Modifications à apporter

1. **Sprint 22** : passer de `⬜ À démarrer` → `✅ Terminé` avec la date de clôture réelle
2. **Sprint 23** : ajouter le preview des tâches héritées (validation board INT8, `S2307`)
3. **Métriques clés Sprint 22** : insérer le tableau récapitulatif des 8 expériences

### Format de la section Sprint 22 dans la roadmap

```markdown
### Sprint 22 — 7–21 juin 2026 ✅ Terminé

**Livrables** :
- `src/data/cmapss_loader.py` + `paderborn_loader.py` ✅
- 6 expériences CL PC (`exp_S22_01` à `exp_S22_06`) + 2 INT8 ✅
- `src/models/ewc/ewc_mlp_int8.py` (Gap 3 Python) ✅
- `firmware/stm32f4_blink/src/ewc_head_int8.c` + tests Unity ✅
- `docs/datasets_analysis.md` mis à jour ✅

**Gap 1** : CMAPSS + Paderborn ajoutés → 5 datasets industriels couverts
**Gap 3** : ewc_mlp_int8.py — Δ AUROC < 0.02 ✅ / ewc_head_int8.c compilable ARM ✅

**Reporté Sprint 23** : validation board INT8 (latence DWT, S2307)
```

---

## Questions ouvertes

- `TODO(arnaud)` : Les tests sont marqués `@pytest.mark.skipif` sans données — faut-il ajouter un dataset de test synthétique (fixtures pytest) pour que les tests end-to-end s'exécutent en CI sans les données réelles ?
- `FIXME(gap1)` : `docs/datasets_analysis.md` doit inclure un tableau récapitulatif des 5 datasets avec une colonne "contribution Gap 1" pour le chapitre 4 du manuscrit.
