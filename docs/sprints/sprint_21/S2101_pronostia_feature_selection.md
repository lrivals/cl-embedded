# S2101 — Feature selection Pronostia 13→5

| Champ | Valeur |
|-------|--------|
| **Sprint** | 21 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Terminé |
| **Durée estimée** | 2h |
| **Dépendances** | `src/data/pronostia_dataset.py` ✅, données `data/raw/Pronostia dataset/binaries/` |
| **Fichiers cibles** | `scripts/pronostia_feature_selection.py`, `configs/pronostia_feature_subset.yaml` |
| **Référence** | `pronostia_dataset.py:FEATURE_NAMES` (13 features), `Benatti2019HDC`, `Hurtado2023CLPdM` |

---

## Contexte

Le firmware est compilé avec `N_FEATURES=5` (constante `MAHA_DIM`, `EWC_IN`, `TINYOL_IN`).  
Pronostia produit 13 features time-domain après fenêtrage :

```
idx  0  mean_acc_horiz
idx  1  std_acc_horiz
idx  2  rms_acc_horiz
idx  3  kurtosis_acc_horiz
idx  4  peak_acc_horiz
idx  5  crest_factor_acc_horiz
idx  6  mean_acc_vert
idx  7  std_acc_vert
idx  8  rms_acc_vert
idx  9  kurtosis_acc_vert
idx 10  peak_acc_vert
idx 11  crest_factor_acc_vert
idx 12  temporal_position
```

Il faut sélectionner 5 indices compatibles avec le pipeline board existant.

---

## Ce qu'il faut implémenter

### `scripts/pronostia_feature_selection.py`

Script autonome qui :

1. Tente de charger `data/raw/Pronostia dataset/binaries/` via `load_condition_features()`
2. Si données disponibles : calcule `sklearn.feature_selection.mutual_info_classif` (labels binaires faulty/normal) + ranking par variance
3. Si données absentes : applique la sélection expert (fallback)
4. Sauvegarde le résultat dans `configs/pronostia_feature_subset.yaml`
5. Affiche un tableau de ranking pour validation visuelle

```bash
# Usage
python scripts/pronostia_feature_selection.py
python scripts/pronostia_feature_selection.py --method variance   # alternatif
python scripts/pronostia_feature_selection.py --n-features 5 --output configs/pronostia_feature_subset.yaml
```

### Sélection expert (fallback si données absentes)

Justification domain-knowledge pour roulements :

| Rang | Indice | Feature | Justification |
|------|:------:|---------|---------------|
| 1 | 2 | `rms_acc_horiz` | Niveau d'énergie vibratoire — indicateur de dégradation |
| 2 | 3 | `kurtosis_acc_horiz` | Impulsivité — signature de choc de défaut |
| 3 | 8 | `rms_acc_vert` | Redondance canal vertical (orthogonal au canal horiz) |
| 4 | 9 | `kurtosis_acc_vert` | Impulsivité verticale |
| 5 | 12 | `temporal_position` | Trajectoire de dégradation normalisée [0, 1] |

### Format `configs/pronostia_feature_subset.yaml`

```yaml
# pronostia_feature_subset.yaml — Top-5 features Pronostia pour board (N_FEATURES=5)
# Généré par scripts/pronostia_feature_selection.py
method: mutual_info  # ou "variance" ou "expert_fallback"
n_features_total: 13
n_features_selected: 5
feature_indices: [2, 3, 8, 9, 12]
feature_names:
  - rms_acc_horiz
  - kurtosis_acc_horiz
  - rms_acc_vert
  - kurtosis_acc_vert
  - temporal_position
ranking:
  rms_acc_horiz: 0.412
  kurtosis_acc_horiz: 0.387
  rms_acc_vert: 0.351
  kurtosis_acc_vert: 0.298
  temporal_position: 0.245
```

---

## Vérification

```bash
# Générer le subset (avec ou sans données)
python scripts/pronostia_feature_selection.py

# Vérifier le YAML produit
python -c "import yaml; d=yaml.safe_load(open('configs/pronostia_feature_subset.yaml')); \
    assert len(d['feature_indices']) == 5; print('OK', d['feature_names'])"

# Charger le subset dans sensor_stream (test import)
python -c "
from pathlib import Path
import yaml, numpy as np
subset = yaml.safe_load(open('configs/pronostia_feature_subset.yaml'))
print('feature_indices:', subset['feature_indices'])
print('feature_names:', subset['feature_names'])
"
```

---

## Questions ouvertes

- `TODO(arnaud)` : Valider le choix mutual_info vs variance — les deux donnent les mêmes top-5 sur Pronostia ?
- `TODO(arnaud)` : Faut-il inclure `temporal_position` (idx 12) ? Elle encode la dégradation mais n'est pas disponible en temps réel strict. Alternative : crest_factor (idx 5 ou 11).
