# S10-03 — EDA Pronostia — Section 3 dans `01_data_exploration.ipynb`

| Champ | Valeur |
|-------|--------|
| **ID** | S10-03 |
| **Sprint** | Sprint 10 — Phase 1 Extension |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 2h |
| **Dépendances** | S10-01 (`PronostiaDataset` opérationnel) |
| **Fichiers cibles** | `notebooks/01_data_exploration.ipynb`, `notebooks/figures/eda/pronostia/` |

---

## Objectif

Ajouter une **section 3** dans le notebook d'exploration existant `01_data_exploration.ipynb` pour documenter le dataset FEMTO PRONOSTIA. Cette section est le support visuel de la description du Dataset 3 dans le manuscrit.

Les sections 1 et 2 couvrent déjà les datasets Monitoring (tabulaire) et Pump Maintenance (temporel). La section 3 documente la vibrométrie industrielle réelle de PRONOSTIA.

---

## Structure de la section 3

### Cellule 3.0 — En-tête Markdown
```markdown
## Section 3 — Dataset 3 : FEMTO PRONOSTIA (IEEE PHM 2012)

**Source** : FEMTO-ST Institute / INSA Lyon — Bearing accelerated degradation dataset  
**Type** : Séries temporelles d'accélérométrie (2 canaux : horizontal + vertical)  
**Label** : Binaire — pré-défaillance (1) = derniers 10% du signal  
**Scénario CL** : Domain-incremental par condition opératoire (3 tâches)  
**Gap** : `FIXME(gap1)` — données industrielles réelles pour la validation Phase 1
```

### Cellule 3.1 — Chargement et statistiques globales
```python
from src.data.pronostia_dataset import PronostiaDataset

BEARING_IDS_ALL = [
    "Bearing1_1", "Bearing1_2",  # Condition 1
    "Bearing2_1", "Bearing2_2",  # Condition 2
    "Bearing3_1", "Bearing3_2",  # Condition 3
]
dataset = PronostiaDataset(npy_dir=NPY_DIR, bearing_ids=BEARING_IDS_ALL)

print(f"Total fenêtres : {len(dataset)}")
print(f"Features par fenêtre : {dataset[0][0].shape[0]}")
print(f"Proportion pré-défaillance : {sum(y for _, y in dataset) / len(dataset):.2%}")
```

### Cellule 3.2 — Signal brut (roulement entier)
```python
# Visualiser la trajectoire de dégradation d'un roulement complet
# Bearing1_1 : ~2 046 epochs jusqu'à la défaillance
import matplotlib.pyplot as plt

raw = np.load(f"{NPY_DIR}/Bearing1_1.npy")  # (N_epochs, 2, 2560)
rms_horiz = np.sqrt(np.mean(raw[:, 0, :]**2, axis=1))
rms_vert  = np.sqrt(np.mean(raw[:, 1, :]**2, axis=1))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
ax1.plot(rms_horiz, label="Horizontal RMS")
ax1.axvline(int(len(rms_horiz) * 0.9), color='red', linestyle='--', label='Seuil pré-défaillance (90%)')
ax2.plot(rms_vert,  label="Vertical RMS",   color='orange')
ax2.axvline(int(len(rms_vert) * 0.9),   color='red', linestyle='--')
```
Sauvegarder : `notebooks/figures/eda/pronostia/raw_signal_bearing1_1.png`

### Cellule 3.3 — Distribution des labels par condition
```python
# Vérifier l'équilibre des classes dans chaque condition
CONDITION_MAP = {
    1: ["Bearing1_1", "Bearing1_2"],
    2: ["Bearing2_1", "Bearing2_2"],
    3: ["Bearing3_1", "Bearing3_2"],
}
# Barplot : proportion label=1 par condition et par roulement
```
Sauvegarder : `notebooks/figures/eda/pronostia/label_distribution_by_condition.png`

### Cellule 3.4 — Comparaison inter-conditions (feature space)
```python
# PCA 2D sur les 13 features : visualiser si les conditions sont séparables
# Chaque point = une fenêtre, couleur = condition (1/2/3), forme = label (0/1)
from sklearn.decomposition import PCA
```
Sauvegarder : `notebooks/figures/eda/pronostia/pca_conditions.png`

### Cellule 3.5 — Tableau récapitulatif
```python
# Résumé statistique : n_epochs par roulement, durée de vie, proportion label=1
summary = pd.DataFrame({
    "Roulement": BEARING_IDS_ALL,
    "Condition": [1, 1, 2, 2, 3, 3],
    "N_epochs": [...],
    "Durée (s)": [...],
    "Label=1 (%)": [...],
})
```

---

## Structure des figures cibles

```
notebooks/figures/eda/pronostia/
├── raw_signal_bearing1_1.png         ← trajectoire RMS horizontal+vertical
├── label_distribution_by_condition.png  ← barplot labels par condition
└── pca_conditions.png                ← espace features 2D PCA
```

---

## Critères d'acceptation

- [ ] Section 3 ajoutée dans `01_data_exploration.ipynb` (après les sections Monitoring et Pump)
- [ ] Notebook exécutable sans erreur après "Restart Kernel & Run All Cells"
- [ ] 3 figures sauvegardées dans `notebooks/figures/eda/pronostia/`
- [ ] Tableau récapitulatif présent avec n_epochs, durée de vie et proportion label=1 par roulement
- [ ] La cellule 3.0 référence `FIXME(gap1)` et pointe vers exp_050–055

---

## Questions ouvertes

- `TODO(arnaud)` : Faut-il inclure une comparaison visuelle avec les données Pump Maintenance (Dataset 1) pour montrer le contraste temporel/fréquentiel ? Cela enrichirait la section 3.4.

---

**Complété le** : _(à renseigner)_
