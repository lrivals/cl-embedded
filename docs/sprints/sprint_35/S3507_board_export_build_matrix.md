# S3507 — Export poids/projections par condition + matrice de build

| Champ | Valeur |
|-------|--------|
| **Sprint** | 35 |
| **Priorité** | 🔴 Critique — fournit les binaires board par condition |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 4h |
| **Dépendances** | S3506 (dims configurables), S3503 (modèles PC entraînés par condition), `scripts/export_weights_c.py` ✅ |
| **Fichiers cibles** | `scripts/export_weights_c.py`, `firmware/stm32f4_blink/Makefile` |
| **Références** | `inc/model_weights_ewc.h` (généré, Sprint 32), règle CLAUDE.md « ne jamais éditer model_weights.h à la main » |

---

## Contexte

Chaque condition (`5feat`/`all`/`best`) a des dims d'entrée et des poids différents → il faut
**régénérer les headers de poids** (EWC, Mahalanobis) et les **projections** (HDC) par condition,
puis builder un binaire par condition. **1 flash par condition** (le firmware n'embarque pas
plusieurs jeux de dims simultanément).

## Spec

- Étendre `export_weights_c.py` pour prendre la condition + le subset de features en entrée
  (`--condition {5feat,all,best} --model ... --dataset ...`), produisant les headers de poids
  aux dims correspondantes. **Jamais d'édition manuelle des headers** (règle CLAUDE.md).
- HDC : régénérer la projection embarquée à `HDC_N_FEATURES` de la condition.
- TinyOL : init en ligne (HW-only) — pas de poids exportés, seule la dim change au build.
- Matrice de build : `Makefile` accepte les dims par condition ; produire un `.bin` par
  `(condition, dataset)` flashable.

| Condition | Dims modèles | Exports requis |
|-----------|-------------|----------------|
| 5feat | 5 (existant) | EWC, Maha (déjà OK) |
| all | dim native (S3502) | EWC, Maha régénérés |
| best | k\* par modèle (S3501) | EWC, Maha régénérés |

**Règle** : poids générés par script uniquement ; parité board↔PC vérifiée en S3508 (EWC+Maha).

## Vérification

```bash
python scripts/export_weights_c.py --ewc-head --condition best --model ewc --dataset cwru
cd firmware/stm32f4_blink && make EWC_IN=<k*> size   # build avec poids régénérés
```

## Implémentation (✅)

- **`export_weights_c.py`** : `export_ewc_head_board_to_c()` lit désormais `EWC_IN=k` depuis
  le checkpoint (`fc1.weight.shape[1]`) au lieu de figer 5 ; émet `#define EWC_HEAD_NATIVE_DIM k`.
  Mahalanobis émet `#define MAHA_NATIVE_DIM d`. Nouveau résolveur CLI
  `--mahal/--ewc-head` (sans valeur) + `--condition --model --dataset` → localise
  `experiments/exp_S35_board_{condition}_{model}_{dataset}/checkpoints/`. **Headers générés
  uniquement** (règle CLAUDE.md).
- **Firmware** (`pipeline.c`) : gardes de copie passées de `WEIGHTS_NATIVE_DIM` (=5 figé) à
  **dim native par modèle** : `EWC_IN == EWC_HEAD_NATIVE_DIM` (EWC) et `MAHA_DIM == MAHA_NATIVE_DIM`
  (Maha), avec fallback `#ifndef … WEIGHTS_NATIVE_DIM` → **0 régression 5feat** (`.bss=104 956 B`
  inchangé). Les poids exportés se chargent donc à la dim de la condition (parité `all`/`best`).
- **Matrice de build** : `Makefile` (S3506) accepte `EWC_IN/MAHA_DIM/TINYOL_IN/HDC_N_FEATURES`
  (+`PROTO_MAX_N` si k>16) via `-D`. Builds vérifiés : 5feat (104 956 B), `EWC_IN=9 MAHA_DIM=9`
  (107 116 B), all-cmapss=21. Parité board↔PC **exacte vérifiée** (EWC+Maha) sur cellule réelle
  `all×monitoring` (k=4). `make test` Unity 103/105 (2 TinyOL préexistants).
