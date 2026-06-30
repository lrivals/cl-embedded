# S2405 — Re-runs expériences clés Sprints 5–21 avec améliorations Sprint 4

| Champ | Valeur |
|-------|--------|
| **Sprint** | 24 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Terminé — flags CLI ajoutés aux 4 scripts (--dataset, --scenario, --profile_memory, --output_dir) |
| **Durée estimée** | S2405a : 2h / S2405b : 1h / S2405c : 1h = 4h total |
| **Dépendances** | S2404a ✅ (`--all` flag dans `profile_memory.py`) |
| **Fichiers cibles** | `experiments/exp_S24_04/` à `experiments/exp_S24_12/` |
| **Référence** | exp_S22_01 à exp_S22_08 (pattern résultats Sprint 22) |

---

## Contexte

Les expériences clés des Sprints 5–21 doivent être re-lancées avec les améliorations Sprint 4 (profiling RAM unifié + ONNX export) pour assurer la comparabilité dans le notebook final. On ne re-lance pas toutes les 160+ expériences historiques — uniquement les **représentatives par modèle × dataset**, couvrant les 5 datasets.

**Critère de sélection des re-runs** : une expérience est re-lancée si elle est la plus représentative pour son (modèle, dataset, scénario CL) et si elle n'a pas de profil RAM unifié ni d'export ONNX.

---

## S2405a — CWRU (4 modèles) → exp_S24_04 à exp_S24_07

### Expériences originales de référence

| Exp S24 | Modèle | Dataset | Expérience historique | Sprint |
|---------|--------|---------|----------------------|--------|
| exp_S24_04 | EWC | CWRU (by fault type) | exp_100-115 zone | Sprint 12 |
| exp_S24_05 | HDC | CWRU (by fault type) | exp_100-115 zone | Sprint 12 |
| exp_S24_06 | TinyOL | CWRU (by fault type) | exp_100-115 zone | Sprint 12 |
| exp_S24_07 | Mahalanobis | CWRU (by fault type) | exp_100-115 zone | Sprint 12-15 |

### Commandes

```bash
# exp_S24_04 : EWC / CWRU
python scripts/train_ewc.py \
  --config configs/ewc_config.yaml \
  --dataset cwru \
  --scenario by_fault_type \
  --exp_id exp_S24_04 \
  --output_dir experiments/exp_S24_04/ \
  --profile_memory

# exp_S24_05 : HDC / CWRU
python scripts/train_hdc.py \
  --config configs/hdc_config.yaml \
  --dataset cwru \
  --scenario by_fault_type \
  --exp_id exp_S24_05 \
  --output_dir experiments/exp_S24_05/ \
  --profile_memory

# exp_S24_06 : TinyOL / CWRU
python scripts/train_tinyol.py \
  --config configs/tinyol_config.yaml \
  --dataset cwru \
  --scenario by_fault_type \
  --exp_id exp_S24_06 \
  --output_dir experiments/exp_S24_06/ \
  --profile_memory

# exp_S24_07 : Mahalanobis / CWRU
python scripts/train_mahalanobis.py \
  --config configs/ewc_config.yaml \
  --dataset cwru \
  --scenario by_fault_type \
  --exp_id exp_S24_07 \
  --output_dir experiments/exp_S24_07/ \
  --profile_memory
```

### Métriques attendues (cohérence avec Sprints 12–15)

| Exp | acc_final attendu | AF attendu | Source |
|-----|:-----------------:|:----------:|--------|
| exp_S24_04 (EWC) | ≥ 0.90 | ≤ 0.05 | Sprint 12 results |
| exp_S24_05 (HDC) | ≥ 0.75 | ≤ 0.05 | Sprint 12 results |
| exp_S24_06 (TinyOL) | ≥ 0.80 | ≤ 0.08 | Sprint 12 results |
| exp_S24_07 (Mahalanobis) | ≥ 0.70 | ≤ 0.10 | Sprint 15 results |

**Critère de cohérence** : `|acc_final_S24 - acc_final_historique| ≤ 0.005` (seed=42 fixé)

---

## S2405b — Pronostia (2 modèles) → exp_S24_08 à exp_S24_09

### Expériences originales de référence

| Exp S24 | Modèle | Dataset | Expérience historique | Sprint |
|---------|--------|---------|----------------------|--------|
| exp_S24_08 | EWC | Pronostia (by condition) | Sprint 10 results | Sprint 10 |
| exp_S24_09 | Mahalanobis | Pronostia (by condition) | Sprint 15 results | Sprint 15 |

### Commandes

```bash
# exp_S24_08 : EWC / Pronostia
python scripts/train_ewc.py \
  --config configs/ewc_config.yaml \
  --dataset pronostia \
  --scenario by_condition \
  --exp_id exp_S24_08 \
  --output_dir experiments/exp_S24_08/ \
  --profile_memory

# exp_S24_09 : Mahalanobis / Pronostia
python scripts/train_mahalanobis.py \
  --config configs/ewc_config.yaml \
  --dataset pronostia \
  --scenario by_condition \
  --exp_id exp_S24_09 \
  --output_dir experiments/exp_S24_09/ \
  --profile_memory
```

---

## S2405c — Pump temporal (3 modèles + UINT8 TinyOL) → exp_S24_10 à exp_S24_12

### Expériences originales de référence

| Exp S24 | Modèle | Dataset | Expérience historique | Sprint |
|---------|--------|---------|----------------------|--------|
| exp_S24_10 | EWC | Pump temporal | exp_025_ewc_pump_temporal | Sprint 6 |
| exp_S24_11 | HDC | Pump temporal | exp_026_hdc_pump_temporal | Sprint 6 |
| exp_S24_12 | TinyOL UINT8 | Pump temporal | exp_003 + exp_004 (UINT8) | Sprints 3+4 |

### Commandes

```bash
# exp_S24_10 : EWC / Pump temporal
python scripts/train_ewc.py \
  --config configs/tinyol_config.yaml \
  --dataset pump \
  --scenario temporal \
  --exp_id exp_S24_10 \
  --output_dir experiments/exp_S24_10/ \
  --profile_memory

# exp_S24_11 : HDC / Pump temporal
python scripts/train_hdc.py \
  --config configs/hdc_config.yaml \
  --dataset pump \
  --scenario temporal \
  --exp_id exp_S24_11 \
  --output_dir experiments/exp_S24_11/ \
  --profile_memory

# exp_S24_12 : TinyOL UINT8 / Pump temporal (extension exp_004 sur Pump)
python scripts/train_tinyol.py \
  --config configs/tinyol_config.yaml \
  --dataset pump \
  --scenario temporal \
  --uint8_activations \
  --exp_id exp_S24_12 \
  --output_dir experiments/exp_S24_12/ \
  --profile_memory
```

### Contenu `results.json` pour exp_S24_12 (TinyOL UINT8 Pump)

```json
{
  "exp_id": "exp_S24_12",
  "model": "tinyol",
  "dataset": "pump",
  "scenario": "temporal",
  "uint8_activations": true,
  "acc_final": "...",
  "avg_forgetting": "...",
  "fp32_activations_bytes": "...",
  "uint8_activations_bytes": "...",
  "compression_ratio": "...",
  "delta_aa_vs_fp32": "...",
  "reference_exp_fp32": "exp_024_tinyol_pump_temporal",
  "reference_exp_uint8_monitoring": "exp_004_tinyol_uint8"
}
```

**Critères** :
- `compression_ratio ≥ 3.5×` (cohérent avec exp_004) ✓
- `|delta_aa_vs_fp32| ≤ 0.005` ✓
- `gap2_compliant: true` ✓

---

## Structure commune des dossiers `exp_S24_XX`

```
experiments/exp_S24_{NN}/
├── config_snapshot.yaml      ← snapshot config au moment du run (seed, hyperparams)
├── results.json              ← métriques CL + profil RAM + référence exp historique
└── training_curves.npy       ← matrice accuracy-après-tâche [n_tasks × n_tasks]
```
