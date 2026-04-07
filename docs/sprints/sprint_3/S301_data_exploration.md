# S3-01 — Télécharger Dataset 1 (Pump) + exploration

| Champ | Valeur |
|-------|--------|
| **ID** | S3-01 |
| **Sprint** | Sprint 3 — Semaine 3 (29 avril – 6 mai 2026) |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2h |
| **Dépendances** | — (indépendant du sprint 2) |
| **Fichier cible** | `notebooks/01_data_exploration.ipynb` (Section 2) |
| **Complété le** | 7 avril 2026 |
| **Statut** | ✅ Terminé |

---

## Objectif

Télécharger le Dataset 1 (Large Industrial Pump Maintenance) dans `data/raw/pump_maintenance/`, valider son intégrité, et ajouter une **Section 2** au notebook existant `notebooks/01_data_exploration.ipynb` documentant :
- La structure temporelle des données (drift sain → usure → pré-panne)
- Les statistiques descriptives par canal
- La distribution du label `maintenance_required` dans le temps
- Les statistiques de normalisation (mean/std) calculées sur la première fraction temporelle (Task 1) → à exporter dans `configs/pump_normalizer.yaml`

**Critère de succès** : le notebook tourne sans erreur de bout en bout, le slug Kaggle est documenté dans `docs/context/datasets.md`, shape + dtype validés, taux de `maintenance_required` par tiers chronologique calculé et affiché.

---

## Sous-tâches

### 1. Télécharger le dataset

```bash
# Vérifier que kaggle.json est en place
ls ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Téléchargement — remplacer [slug] par le slug exact (voir TODO(arnaud) ci-dessous)
kaggle datasets download -d [slug_dataset_pump] -p data/raw/pump_maintenance/ --unzip
```

Vérifier la présence du CSV après décompression :

```bash
ls data/raw/pump_maintenance/
# Attendu : au moins un fichier .csv contenant les colonnes temporelles
```

### 2. Valider l'intégrité du dataset

```python
import pandas as pd

df = pd.read_csv("data/raw/pump_maintenance/<fichier>.csv", parse_dates=["timestamp"])

# Colonnes attendues (voir docs/context/datasets.md)
EXPECTED_COLS = ["timestamp", "temperature", "vibration", "pressure", "rpm",
                 "maintenance_required"]

assert all(c in df.columns for c in EXPECTED_COLS), \
    f"Colonnes manquantes : {set(EXPECTED_COLS) - set(df.columns)}"
assert df["maintenance_required"].isin([0, 1]).all(), "Label inattendu (attendu : 0/1)"
assert df["timestamp"].is_monotonic_increasing, "Données non triées chronologiquement"

print(f"Shape : {df.shape}")
print(f"Période : {df['timestamp'].min()} → {df['timestamp'].max()}")
print(f"Taux de maintenance global : {df['maintenance_required'].mean():.3f}")
```

### 3. Ajouter la Section 2 dans `notebooks/01_data_exploration.ipynb`

#### Section 2.1 — Chargement et vue d'ensemble Dataset 1

```python
# Shape, dtypes, valeurs manquantes, étendue temporelle
print(df.dtypes)
print(df.isnull().sum())
print(f"Durée totale : {(df['timestamp'].max() - df['timestamp'].min()).days} jours")
```

#### Section 2.2 — Distribution temporelle du label

```python
import matplotlib.pyplot as plt

# Fenêtre glissante de 500 points pour visualiser le drift
df["maint_roll"] = df["maintenance_required"].rolling(500).mean()

plt.figure(figsize=(14, 4))
plt.plot(df["timestamp"], df["maint_roll"])
plt.title("Taux de maintenance glissant (fenêtre 500) — Dataset 1 Pump")
plt.xlabel("Temps")
plt.ylabel("Taux maintenance_required")
plt.savefig("notebooks/figures/pump_maintenance_drift.png", dpi=150, bbox_inches="tight")
plt.show()
```

> L'objectif est de montrer le gradient de dégradation temporel (justification du scénario Domain-Incremental avec drift temporel).

#### Section 2.3 — Statistiques descriptives par canal

```python
FEATURE_COLS = ["temperature", "vibration", "pressure", "rpm"]

# Stats globales
df[FEATURE_COLS].describe()

# Stats par tiers chronologique (approximation des 3 tâches CL)
n = len(df)
for i, label in enumerate(["T1 (sain)", "T2 (usure)", "T3 (pré-panne)"]):
    subset = df.iloc[i * n // 3 : (i + 1) * n // 3]
    print(f"\n{label} — N={len(subset)}, taux panne={subset['maintenance_required'].mean():.3f}")
    print(subset[FEATURE_COLS].describe())
```

#### Section 2.4 — Corrélations inter-variables

```python
import seaborn as sns

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(df[FEATURE_COLS + ["maintenance_required"]].corr(),
            annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Corrélations — Dataset 1 Pump")
plt.tight_layout()
plt.savefig("notebooks/figures/pump_correlations.png", dpi=150)
plt.show()
```

#### Section 2.5 — Statistiques de normalisation Task 1 (à exporter)

```python
import yaml
from pathlib import Path

# FIT sur Task 1 UNIQUEMENT (premier tiers chronologique = état sain)
# Ne jamais recalculer à l'inférence — voir tinyol_spec.md §3.3
task1 = df.iloc[: len(df) // 3][FEATURE_COLS]

normalizer_stats = {
    "mean": task1.mean().to_dict(),
    "std": task1.std().to_dict(),
    "fit_on": "task1_chronological",
    "n_samples": len(task1),
}

print("Stats normalisation Task 1 :")
print(normalizer_stats)

# Export (si confirmé par TODO(arnaud))
# Path("configs/pump_normalizer.yaml").write_text(yaml.dump(normalizer_stats))
```

> **Règle** : les statistiques sont fit sur Task 1 uniquement pour éviter toute fuite d'information des tâches futures. Conforme à `docs/context/datasets.md`.

#### Section 2.6 — Scénario CL retenu

```markdown
## Scénario CL Dataset 1 — Pump Maintenance

- **Type** : Domain-Incremental avec drift temporel naturel
- **Ordre des tâches** : T1 = état sain → T2 = usure naissante → T3 = pré-panne
- **Frontières** : implicites (découpages chronologiques, task-free scenario)
- **N_TASKS** : 3 (découpage 33%/33%/33% chronologique — à confirmer)
- **N_FEATURES** : 25 (6 features × 4 canaux + 1 label temporel normalisé)
- **WINDOW_SIZE** : 32, **STEP_SIZE** : 16
- **Normalisation** : Z-score, fit sur T1 uniquement → configs/pump_normalizer.yaml
- **Référence** : CLAUDE.md, docs/context/datasets.md, docs/models/tinyol_spec.md
```

---

## Critères d'acceptation

- [x] `data/raw/pump_maintenance/` contient au moins un fichier CSV avec les 6 colonnes attendues
- [x] Le notebook tourne sans erreur de `Kernel → Restart & Run All`
- [x] `df["operational_hours"].is_monotonic_increasing` vérifié (proxy temporel — pas de colonne `timestamp` dans le CSV)
- [x] `df["maintenance_required"].isin([0, 1]).all()` vérifié
- [x] Taux de `maintenance_required` calculé et affiché pour chacun des 3 tiers chronologiques
- [x] Graphique du drift temporel sauvegardé dans `notebooks/figures/pump_maintenance_drift.png`
- [x] Stats mean/std des 5 canaux (sur T1) calculées et affichées (`flow_rate` présent dans le CSV — canal supplémentaire non documenté initialement)
- [ ] Slug Kaggle documenté dans `docs/context/datasets.md` — `TODO(arnaud)` en attente

> **Note exécution** : colonnes CSV réelles différent des specs initiales (`Maintenance_Flag` au lieu de `maintenance_required`, `Operational_Hours` au lieu de `timestamp`, noms PascalCase). Renommage appliqué dans le notebook + `datasets.md` mis à jour.

---

## Sorties attendues à reporter ailleurs

| Élément | Où reporter | Statut |
|---------|-------------|--------|
| mean/std features (sur T1) | `configs/pump_normalizer.yaml` | ✅ Calculées dans notebook — export commenté, attente TODO(arnaud) → S3-04 |
| Nb échantillons total + par tiers | En-tête de `src/data/pump_dataset.py` | ⬜ S3-02 |
| Taux de panne par tiers | `experiments/exp_003_tinyol_dataset1/config_snapshot.yaml` | ⬜ S3-06 |
| Slug Kaggle exact | `docs/context/datasets.md` | ⬜ TODO(arnaud) — placeholder ajouté |
| Colonnes CSV réelles (PascalCase + Flow_Rate) | `docs/context/datasets.md` | ✅ Mis à jour |

---

## Questions ouvertes

- `TODO(arnaud)` : slug Kaggle exact du Dataset 1 (Large Industrial Pump Maintenance) — à renseigner dans `docs/context/datasets.md`
- `TODO(arnaud)` : stratégie de découpage en 3 tâches — découpage chronologique égal (33%/33%/33%) ou basé sur des seuils de taux de panne observés ? Impact sur le drift inter-tâche.
- `TODO(arnaud)` : le dataset contient-il des trous temporels (capteur arrêté, maintenance) ? Politique à adopter (suppression, interpolation, marquage).
- `FIXME(gap1)` : comparer la distribution de ce dataset simulé avec FEMTO PRONOSTIA (données réelles) pour justifier la représentativité industrielle dans le manuscrit — section à écrire après l'exploration.
