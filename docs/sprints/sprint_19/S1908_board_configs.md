# S1908 — Configs YAML modèles embarqués : dims, seuils, Fisher decay, LR

| Champ | Valeur |
|-------|--------|
| **Sprint** | 19 |
| **Priorité** | 🟡 Important |
| **Statut** | ✅ Créés (à vérifier/compléter) |
| **Durée estimée** | 2h |
| **Dépendances** | S1902 (ewc_consolidate), S1903 (tinyol poids) |
| **Fichiers cibles** | `configs/board_mahalanobis.yaml`, `configs/board_ewc.yaml`, `configs/board_tinyol.yaml` |

---

## Contexte

Conformément à la règle CLAUDE.md :

> **Règle de code** : tout paramètre de taille (couches, buffer, embeddings) doit avoir une constante nommée dans `configs/` avec une valeur par défaut respectant la contrainte 64 Ko.

Les configs `board_*.yaml` sont la source de vérité pour les hyperparamètres embarqués. Elles servent à :
1. Générer les constantes C dans `model_weights.h` via `export_weights_c.py`
2. Être copiées comme `config_snapshot.yaml` dans `experiments/exp_S19_XX/`
3. Documenter les choix d'architecture embarquée

---

## État actuel — 3 fichiers créés ✅

Les fichiers `configs/board_mahalanobis.yaml`, `configs/board_ewc.yaml`, `configs/board_tinyol.yaml` ont été créés le 21 mai 2026. Cette tâche vérifie leur complétude et cohérence avec le code C.

---

## Contenu attendu par fichier

### `configs/board_mahalanobis.yaml`

```yaml
# Mahalanobis embarqué — NUCLEO-F439ZI / STM32N6
model: mahalanobis
platform: nucleo_f439zi

# Architecture
n_features: 5          # = MAHA_DIM dans mahalanobis.h
n_params: 30           # mean(5) + precision(5×5)

# Hyperparamètres
threshold_init: 3.0    # seuil initial distance Mahalanobis
ema_alpha: 0.05        # vitesse d'adaptation EMA moyenne (MAHA_EMA_ALPHA)

# Contrainte mémoire
ram_budget_bytes: 65536  # 64 Ko — contrainte STM32N6
ram_model_bytes: 200     # mean + precision + threshold + alpha

# Évaluation
n_tasks: 3
n_samples: 500
dataset: cwru
```

### `configs/board_ewc.yaml`

```yaml
# EWC MLP embarqué — NUCLEO-F439ZI / STM32N6
model: ewc
platform: nucleo_f439zi

# Architecture — doit correspondre aux #define dans ewc_head.h
n_in: 5        # EWC_IN
n_h1: 32       # EWC_H1
n_h2: 16       # EWC_H2
n_out: 2       # EWC_OUT
n_params: 1538 # poids + Fisher + star_w (total struct EWCHead / 4)

# Hyperparamètres
lr: 0.01              # EWC_LR dans ewc_head.h
lambda_ewc: 100.0     # h->lambda — coefficient pénalité EWC
fisher_decay: 0.9     # alpha dans ewc_consolidate() — EMA Fisher

# Contrainte mémoire
ram_budget_bytes: 65536
ram_model_bytes: 9728   # 9.5 Ko EWCHead en .bss

# Évaluation
n_tasks: 3
n_samples: 500
dataset: monitoring
```

### `configs/board_tinyol.yaml`

```yaml
# TinyOL autoencoder embarqué — NUCLEO-F439ZI / STM32N6
model: tinyol
platform: nucleo_f439zi

# Architecture — doit correspondre aux #define dans tinyol.h
n_in: 5        # TINYOL_IN
n_h1: 32       # TINYOL_H1
n_emb: 16      # TINYOL_EMB (goulot d'étranglement)
n_out: 5       # TINYOL_OUT (= n_in, reconstruction)
n_params: 881  # encoder uniquement (decoder exclu Sprint 19)

# Hyperparamètres
threshold: 0.05    # TINYOL_THRESHOLD — seuil MSE reconstruction
# OtO head (hors Sprint 19)
# oto_lr: 0.01
# oto_n_proto: 5

# Contrainte mémoire
ram_budget_bytes: 65536
ram_model_bytes: 5800   # encoder + decoder structs en .bss (si copié en RAM)
ram_flash_bytes: 5600   # poids en Flash (const statiques)

# Évaluation
n_tasks: 3
n_samples: 500
dataset: cwru  # ou pronostia selon expérience
```

---

## Points de vérification

### Cohérence avec le code C

| Config | Champ YAML | Correspondance C |
|--------|-----------|------------------|
| board_ewc.yaml | `n_in: 5` | `#define EWC_IN 5` dans `ewc_head.h` |
| board_ewc.yaml | `n_h1: 32` | `#define EWC_H1 32` |
| board_ewc.yaml | `n_h2: 16` | `#define EWC_H2 16` |
| board_ewc.yaml | `n_out: 2` | `#define EWC_OUT 2` |
| board_ewc.yaml | `lr: 0.01` | `#define EWC_LR 0.01f` |
| board_tinyol.yaml | `n_in: 5` | `#define TINYOL_IN 5` dans `tinyol.h` |
| board_tinyol.yaml | `n_emb: 16` | `#define TINYOL_EMB 16` |
| board_mahalanobis.yaml | `n_features: 5` | `#define MAHA_DIM 5` |

Si une valeur diverge, c'est la config YAML qui fait foi — modifier le `#define` en conséquence.

### Cohérence ram_model_bytes vs budget

Vérifier que `ram_model_bytes ≤ ram_budget_bytes - 15360` (15 Ko réservés système/HAL).

---

## Fichiers cibles

| Fichier | Action |
|---------|--------|
| `configs/board_mahalanobis.yaml` | Vérifier champs ci-dessus, compléter si manquants |
| `configs/board_ewc.yaml` | Idem + vérifier `fisher_decay` et `lambda_ewc` |
| `configs/board_tinyol.yaml` | Idem + vérifier `threshold` |
| `firmware/stm32f4_blink/inc/ewc_head.h` | Vérifier cohérence #define vs YAML |
| `firmware/stm32f4_blink/inc/tinyol.h` | Idem |

---

## Vérification

- [ ] `python -c "import yaml; yaml.safe_load(open('configs/board_ewc.yaml'))"` → pas d'erreur
- [ ] Tous les champs `n_*` concordent avec les `#define` correspondants dans les headers C
- [ ] `ram_model_bytes` ≤ contrainte 64 Ko moins overhead système pour les 3 configs
- [ ] `board_experiment_recorder.py --dry-run` lit correctement les configs et les copie dans `config_snapshot.yaml`
