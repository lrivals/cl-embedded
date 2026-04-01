# S1-02 — Télécharger Dataset 2 (Monitoring) + exploration

| Champ | Valeur |
|-------|--------|
| **ID** | S1-02 |
| **Sprint** | Sprint 1 — Semaine 1 (15–22 avril 2026) |
| **Priorité** | ✅ Terminé |
| **Durée estimée** | 2h |
| **Dépendances** | S1-01 (repo opérationnel) |
| **Fichier cible** | `notebooks/01_data_exploration.ipynb` |
| **Complété le** | 2 avril 2026 |

---

## Objectif

Télécharger le Dataset 2 (Industrial Equipment Monitoring) dans `data/raw/equipment_monitoring/`, valider son intégrité, et produire un notebook d'exploration documentant la structure du dataset, les distributions par domaine, et les statistiques de normalisation nécessaires à `src/data/monitoring_dataset.py`.

**Critère de succès** : le notebook tourne de bout en bout sans erreur et produit les stats mean/std par feature (valeurs numériques à reporter dans `configs/monitoring_normalizer.yaml`).

---

## Sous-tâches

### 1. Configurer l'API Kaggle

```bash
# Vérifier que kaggle.json est en place
ls ~/.kaggle/kaggle.json

# Si absent : aller sur kaggle.com → Account → API → Create New Token
# Placer kaggle.json dans ~/.kaggle/ avec chmod 600
chmod 600 ~/.kaggle/kaggle.json
pip install kaggle
```

### 2. Télécharger le dataset

```bash
# Remplacer [slug] par le slug Kaggle exact une fois identifié
# TODO(arnaud) : confirmer le slug exact du dataset Equipment Monitoring
kaggle datasets download -d [slug_dataset_monitoring] -p data/raw/equipment_monitoring/ --unzip
```

Vérifier la présence du fichier CSV après décompression :

```bash
ls data/raw/equipment_monitoring/
# Attendu : au moins un fichier .csv
```

### 3. Valider l'intégrité du dataset

Dans le notebook ou en script rapide :

```python
import pandas as pd

df = pd.read_csv("data/raw/equipment_monitoring/<fichier>.csv")

# Colonnes attendues (voir docs/context/datasets.md)
EXPECTED_COLS = ["temperature", "pressure", "vibration", "humidity",
                 "equipment", "location", "faulty"]

assert all(c in df.columns for c in EXPECTED_COLS), f"Colonnes manquantes"
assert df["faulty"].isin([0, 1]).all(), "Label inattendu"
assert set(df["equipment"].unique()) >= {"pump", "turbine", "compressor"}

print(f"Shape : {df.shape}")
print(f"Équipements : {df['equipment'].value_counts().to_dict()}")
print(f"Taux de défaut global : {df['faulty'].mean():.3f}")
```

### 4. Créer `notebooks/01_data_exploration.ipynb`

Sections obligatoires du notebook :

#### Section 1 — Chargement et vue d'ensemble
- Shape, dtypes, valeurs manquantes
- Répartition des équipements (nb échantillons par domaine)

#### Section 2 — Statistiques descriptives par domaine
```python
df.groupby("equipment")[["temperature", "pressure", "vibration", "humidity"]].describe()
```
→ Vérifier que les distributions diffèrent entre domaines (drift justifié scientifiquement)

#### Section 3 — Distribution du label `faulty` par domaine
```python
df.groupby("equipment")["faulty"].mean()
```
→ Détecter un déséquilibre de classes par domaine (impact sur loss function EWC)

#### Section 4 — Corrélations entre features
- Heatmap `seaborn.heatmap(df[feature_cols].corr())`
- Identifier des features corrélées (redondance potentielle)

#### Section 5 — Visualisation drift inter-domaine
- Boxplots de chaque feature colorés par domaine
- Objectif : montrer visuellement que les domaines sont distribués différemment → justification du scénario Domain-Incremental

#### Section 6 — Statistiques de normalisation (à exporter)
```python
# Calculer mean/std sur les pumps UNIQUEMENT (Task 1 → fit du normalizer)
pumps = df[df["equipment"] == "pump"][["temperature", "pressure", "vibration", "humidity"]]
stats = {
    "mean": pumps.mean().to_dict(),
    "std": pumps.std().to_dict()
}
print(stats)
# → Reporter ces valeurs dans configs/monitoring_normalizer.yaml
```

> Règle : la normalisation est **fit sur Task 1 (pumps) uniquement** pour éviter la fuite d'information des tâches futures. Référence : `docs/context/datasets.md`.

### 5. Documenter la stratégie CL dans le notebook

Ajouter une cellule Markdown récapitulant :

```markdown
## Scénario CL retenu

- **Type** : Domain-Incremental
- **Ordre des tâches** : T1 = pump → T2 = turbine → T3 = compressor
- **Frontières** : explicites (task label disponible à l'entraînement, pas à l'inférence)
- **N_FEATURES_FINAL** : 6 (4 numériques + 2 one-hot pour `equipment`)
- **Normalisation** : Z-score, fit sur T1 uniquement
- **Référence CLAUDE.md** : DOMAIN_ORDER = ["pump", "turbine", "compressor"]
```

---

## Critères d'acceptation

- [x] `data/raw/equipment_monitoring/` contient au moins un fichier CSV (`equipment_anomaly_data.csv`, 7672 lignes)
- [x] Le notebook tourne sans erreur de `Kernel → Restart & Run All`
- [x] Les 3 équipements (Pump, Turbine, Compressor) sont présents dans le dataset
- [x] Les stats mean/std des features numériques (sur Pumps) sont calculées et affichées
- [x] `configs/monitoring_normalizer.yaml` généré automatiquement par la Section 6 du notebook
- [x] Le notebook est sauvegardé dans `notebooks/01_data_exploration.ipynb`
- [x] `DOMAIN_ORDER = ["Pump", "Turbine", "Compressor"]` validé et documenté (Section 7 du notebook)

> **Note** : les valeurs `equipment` sont capitalisées dans le CSV ("Pump", "Turbine", "Compressor").

---

## Sorties attendues à reporter ailleurs

| Élément | Où reporter | Statut |
| ------- | ----------- | ------ |
| mean/std features (sur Pumps) | `configs/monitoring_normalizer.yaml` | ✅ Fait |
| Nb échantillons par domaine | Commentaire dans `src/data/monitoring_dataset.py` | ⬜ S1-03 |
| Taux de défaut par domaine | `experiments/exp_001_ewc_dataset2/config_snapshot.yaml` | ⬜ S1-09 |

### Valeurs mesurées (N=2534 Pumps, Task 1)

| Feature | mean | std |
| ------- | ---- | --- |
| temperature | 70.634028 | 15.781869 |
| pressure | 35.629021 | 10.501268 |
| vibration | 1.613323 | 0.700299 |
| humidity | 50.197351 | 11.874382 |

### Comptages par domaine

| Domaine | N | Taux de défaut |
| ------- | - | -------------- |
| Pump (T1) | 2534 | 9.9% |
| Turbine (T2) | 2565 | 10.1% |
| Compressor (T3) | 2573 | 10.0% |

---

## Questions ouvertes

- `TODO(arnaud)` : slug Kaggle exact du dataset Equipment Monitoring à confirmer (voir `data/raw/equipment_monitoring/` — commande de téléchargement à compléter dans `docs/context/datasets.md`)
- `TODO(arnaud)` : ordre des domaines CL fixé à pump→turbine→compressor ? Ou à ajuster selon le nombre d'échantillons par domaine ?
- `TODO(arnaud)` : normalisation globale (sur tout le train) vs par domaine vs sur T1 uniquement — confirmer le choix retenu pour la comparaison équitable EWC/HDC
