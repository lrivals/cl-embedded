# S2311–S2314 — Paderborn sur board : feature selection FFT + streaming + 2 expériences

| Champ | Valeur |
|-------|--------|
| **Sprint** | 23 |
| **Priorité** | 🔴 Critique (S2311–S2313) / 🟡 Important (S2314) |
| **Statut** | ✅ Terminé — 2026-06-02 |
| **Durée estimée** | 1h + 1h + 1h + 30 min = 3h30 |
| **Dépendances** | Sprint 22 ✅ — `src/data/paderborn_loader.py` livré, features FFT extraites, `data/raw/paderborn/` présent |
| **Fichiers cibles** | `scripts/paderborn_feature_selection.py`, `configs/board_paderborn.yaml`, `configs/paderborn_feature_subset.yaml`, `scripts/sensor_stream.py`, `experiments/exp_S23_05/`, `experiments/exp_S23_06/` |
| **Référence** | `scripts/pronostia_feature_selection.py` (pattern), `configs/board_pronostia.yaml`, `scripts/board_experiment_recorder.py` |

---

## Contexte

Le dataset Paderborn contient des signaux vibratoires de roulements (sain, OR = outer race fault, IR = inner race fault). `paderborn_loader.py` (Sprint 22) extrait des **features FFT** (amplitude à fréquences caractéristiques, RMS, kurtosis) — les signaux bruts ne sont pas streamés vers la board, seulement les features pré-calculées.

Le scénario CL Paderborn sur board est **domain-incremental par type de défaut** : Task 1 = sain → Task 2 = OR → Task 3 = IR. Ce scénario test la capacité des modèles MCU à apprendre de nouveaux modes de défaillance sans oublier les précédents.

La simplification clé par rapport à CMAPSS : les features FFT sont **déjà de dimension réduite** dans `paderborn_loader.py`. Il s'agit de sélectionner les top-5 parmi les features extraites (pas parmi des capteurs bruts).

---

## S2311 — `scripts/paderborn_feature_selection.py`

```python
"""
paderborn_feature_selection.py — Sélection top-5 features FFT Paderborn.

Fit sur les données saines (Task 1) uniquement pour éviter la fuite.
Sauvegarde dans configs/paderborn_feature_subset.yaml.

Usage :
    python scripts/paderborn_feature_selection.py
    python scripts/paderborn_feature_selection.py --n-features 5 --fit-condition healthy
"""

from pathlib import Path
import numpy as np
import yaml
from sklearn.feature_selection import mutual_info_classif

OUTPUT_PATH = Path("configs/paderborn_feature_subset.yaml")
DATA_DIR    = Path("data/raw/paderborn/")
```

### Structure de sortie `configs/paderborn_feature_subset.yaml`

```yaml
# paderborn_feature_subset.yaml — Top-5 features FFT Paderborn
# Généré par scripts/paderborn_feature_selection.py — NE PAS éditer manuellement.
# Fit sur données saines (Task 1) uniquement.
selected_features:
  - "rms_vibration"       # exemple (à confirmer par exécution du script)
  - "kurtosis_vibration"
  - "fft_amp_bpfo"        # Ball Pass Frequency Outer race
  - "fft_amp_bpfi"        # Ball Pass Frequency Inner race
  - "fft_amp_bsf"         # Ball Spin Frequency
n_features: 5
fit_condition: "healthy"
method: "mutual_info_classif"
label_column: "fault_class"
```

### Vérification

```bash
python scripts/paderborn_feature_selection.py
python -c "
import yaml
d = yaml.safe_load(open('configs/paderborn_feature_subset.yaml'))
assert len(d['selected_features']) == 5, f\"attendu 5, obtenu {len(d['selected_features'])}\"
print('feature selection OK:', d['selected_features'])
"
```

---

## `configs/board_paderborn.yaml`

```yaml
# board_paderborn.yaml — Config board NUCLEO pour Paderborn (domain-incremental)
# Ne pas modifier — créer une copie pour chaque variante expérimentale.

MODEL: ewc_head
PLATFORM: nucleo_f439zi

# Architecture
EWC_IN: 5            # top-5 features FFT
EWC_H1: 32
EWC_H2: 16
EWC_OUT: 2           # faulty / normal (binarisé : sain=0, défaut=1)

# Hyperparamètres EWC (inchangés vs Monitoring)
EWC_LR: 0.01
EWC_LAMBDA: 400.0
FISHER_EMA_DECAY: 0.99

# Dataset Paderborn
dataset: "paderborn"
data_dir: "data/raw/paderborn/"
feature_subset_path: "configs/paderborn_feature_subset.yaml"
condition_order: ["healthy", "OR", "IR"]   # 3 tâches CL
n_tasks_board: 3
n_samples_per_task: 150
label_column: "fault_class"   # 0=sain, 1=faulty (OR ou IR)

# Budget mémoire
ram_budget_bytes: 65536

# Export poids C
WEIGHTS_HEADER: "firmware/stm32f4_blink/inc/model_weights.h"
WEIGHTS_CHECKPOINT: "experiments/exp_S22_paderborn_base/ewc_weights.pkl"
```

---

## S2312 — `scripts/sensor_stream.py` : `--dataset paderborn`

Ajouter le cas `paderborn` dans la fonction `load_dataset()` de `sensor_stream.py` :

```python
elif args.dataset == "paderborn":
    from src.data.paderborn_loader import get_cl_dataloaders
    import yaml
    feature_subset = yaml.safe_load(
        Path("configs/paderborn_feature_subset.yaml").read_text()
    )["selected_features"]
    tasks = get_cl_dataloaders(
        data_dir=Path("data/raw/paderborn/"),
        config_path=Path("configs/board_paderborn.yaml"),
        feature_names=feature_subset,
    )
```

> **Note streaming** : les features FFT sont pré-calculées par `paderborn_loader.py` — on ne streame pas les signaux vibratoires bruts (4 096 points) mais uniquement les 5 scalaires résultants. Cela réduit la bande passante UART de ~100× et garantit une latence de streaming < 1 ms/échantillon.

### Vérification

```bash
python scripts/sensor_stream.py --dataset paderborn --model ewc \
    --dry-run --n-samples 50 --tasks 3
# Attendu : 3 tâches (healthy → OR → IR), 50 échantillons par tâche, aucune erreur
```

---

## S2313 — exp_S23_05 : EWC / Paderborn board (sain → OR → IR)

### Structure du dossier

```
experiments/exp_S23_05/
├── config_snapshot.yaml
├── stream_task1_healthy.json
├── stream_task2_OR.json
├── stream_task3_IR.json
└── results.json
```

### `config_snapshot.yaml`

```yaml
exp_id: "exp_S23_05"
model: "ewc"
dataset: "paderborn"
platform: "nucleo_f439zi"
tasks: ["healthy", "OR", "IR"]
n_samples_per_task: 150
board_config: "configs/board_paderborn.yaml"
feature_subset: "configs/paderborn_feature_subset.yaml"
ewc_lambda: 400.0
seed: 42
sprint: 23
date: "2026-06-29"
```

### Commandes de lancement

```bash
# Dry-run
python scripts/sensor_stream.py \
    --dataset paderborn --model ewc \
    --dry-run --n-samples 150 --tasks 3 \
    --output experiments/exp_S23_05/stream_dryrun.json

# Live avec board
python scripts/sensor_stream.py \
    --dataset paderborn --model ewc \
    --port /dev/ttyACM0 --baud 115200 \
    --n-samples 150 --tasks 3 --rate-hz 20 \
    --update --consolidate \
    --output experiments/exp_S23_05/stream_task{task_id}.json

# Enregistrement résultats
python scripts/board_experiment_recorder.py \
    --exp-dir experiments/exp_S23_05/ \
    --model ewc --dataset paderborn
```

### `results.json` attendu

```json
{
  "exp_id": "exp_S23_05",
  "model": "ewc",
  "dataset": "paderborn",
  "platform": "nucleo_f439zi",
  "tasks": ["healthy", "OR", "IR"],
  "acc_final": ">= 0.80",
  "avg_forgetting": "< 0.08",
  "latency_ms": "< 1.0",
  "ram_peak_bytes": "<= 65536",
  "gap2_latency_compliant": true,
  "n_samples_total": 450
}
```

---

## S2314 — exp_S23_06 : Mahalanobis / Paderborn board (baseline)

```bash
# Dry-run Mahalanobis Paderborn
python scripts/sensor_stream.py \
    --dataset paderborn --model mahalanobis \
    --dry-run --n-samples 150 --tasks 3 \
    --output experiments/exp_S23_06/stream_dryrun.json

# Live
python scripts/sensor_stream.py \
    --dataset paderborn --model mahalanobis \
    --port /dev/ttyACM0 --baud 115200 \
    --n-samples 150 --tasks 3 --rate-hz 50 \
    --update \
    --output experiments/exp_S23_06/stream_task{task_id}.json
```

### `results.json` attendu

```json
{
  "exp_id": "exp_S23_06",
  "model": "mahalanobis",
  "dataset": "paderborn",
  "acc_final": ">= 0.65",
  "latency_ms": "< 0.1",
  "ram_peak_bytes": "<= 1024",
  "gap2_latency_compliant": true,
  "note": "baseline non-neuronal pour comparaison Gap 1"
}
```

---

## Vérification end-to-end

```bash
# 1. Feature selection
python scripts/paderborn_feature_selection.py
cat configs/paderborn_feature_subset.yaml

# 2. Dry-run 2 modèles
for model in ewc mahalanobis; do
    python scripts/sensor_stream.py \
        --dataset paderborn --model $model \
        --dry-run --n-samples 50 --tasks 3 && \
    echo "$model Paderborn dry-run OK"
done

# 3. Vérifier les 2 dossiers d'expériences
ls experiments/exp_S23_0{5,6}/results.json
```

---

## Résultats dry-run board (2026-06-02) — valeurs simulées NUCLEO-F439ZI

| Expérience | Modèle | acc_final | avg_forgetting | latence P50 (ms) | latence P99 (ms) | RAM (B) | Gap 2 |
|---|---|---:|---:|---:|---:|---:|---|
| exp_S23_05 | EWC | **0.782** | 0.053 | 0.546 | 0.795 | 9 728 | ✅ |
| exp_S23_06 | Mahalanobis | **0.613** | 0.163 | 0.054 | 0.074 | 1 200 | ✅ |

**Critères sprint** : EWC acc≥0.80 — ⚠️ 0.782 (légèrement sous le seuil en dry-run, vérifier sur board réelle) | Maha acc≥0.65 — ⚠️ 0.613 (idem) | toutes latences < 100 ms ✅

> **Note** : acc_final sur 3 tâches légèrement sous les seuils-cibles dans la simulation. Les valeurs réelles board peuvent diverger selon la distribution des données Paderborn. À vérifier lors du run live.

### Résultats réels board NUCLEO-F439ZI (2026-06-02)

| Expérience | Modèle | acc_final | avg_forgetting | latence P50 (ms) | RAM statique (B) | Gap 2 | Notes |
|---|---|---:|---:|---:|---:|---|---|
| exp_S23_05 | EWC | **0.931** | 0.077 | 0.251 | 9 728 | ✅ | 3 tâches, λ=400 |
| exp_S23_06 | Mahalanobis | 0.380 | 0.690 | 0.004 | 1 200 | ✅ | ⚠️ AF élevé (pas de CL multi-tâche) |

**Critères sprint** : EWC acc≥0.80 ✅ (0.931) | Maha acc≥0.65 ❌ (0.380 — forgetting 0.69 sur 3 tâches) | latences < 100 ms ✅

> **Mahalanobis Paderborn faible** : Mahalanobis ne supporte pas le domain-incremental multi-tâche — il réapprend sur la dernière tâche (IR) et oublie les précédentes (healthy, OR). AF=0.69 confirme l'oubli catastrophique. Résultat cohérent avec la baseline non-neuronale sans régularisation.

### Vérification board réelle (2026-06-02)

```text
ewc/paderborn         → ✅  acc=0.931  lat=0.251ms  AF=0.077  9728B   3 tâches
mahalanobis/paderborn → ⚠️  acc=0.380  lat=0.004ms  AF=0.690  1200B   oubli catastrophique 3 tâches
```

---

## Questions ouvertes

- `TODO(arnaud)` : Le scénario sain → OR → IR sur 3 tâches est-il le bon ordre pour tester le CL ? Alternativement : OR → IR → sain (où la tâche finale est la plus différente des premières) ?
- `TODO(arnaud)` : Faut-il inclure une 4ème tâche Paderborn (combinaison OR+IR simultanés) pour enrichir le Gap 1 ? Ou les 3 tâches actuelles suffisent ?
- `FIXME(gap1)` : Documenter le lien entre les fréquences caractéristiques BPFO/BPFI/BSF et les features FFT sélectionnées — c'est la justification physique du choix de features pour le manuscrit.
- `FIXME(gap2)` : Les latences EWC et Mahalanobis sur Paderborn devraient être identiques à celles sur CMAPSS et Monitoring (même architecture, 5 features). Documenter cette cohérence dans le tableau Gap 2 comparatif.
