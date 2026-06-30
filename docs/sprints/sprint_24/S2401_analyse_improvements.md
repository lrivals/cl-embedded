# S2401 — Matrice des améliorations Sprint 4 : analyse et trous identifiés

| Champ | Valeur |
|-------|--------|
| **Sprint** | 24 |
| **Priorité** | 🟡 Important |
| **Statut** | ✅ Complété |
| **Durée estimée** | 1h |
| **Dépendances** | Sprint 4 ✅ (quantization.py, export_onnx.py, profile_memory.py disponibles) |
| **Fichier cible** | ce fichier (mis à jour à la complétion) |

---

## Contexte

Sprint 4 a introduit trois améliorations transversales. Ce document est la source de vérité sur lesquelles ont été appliquées à quelles expériences, et lesquelles sont encore manquantes.

---

## Matrice des améliorations Sprint 4

### Amélioration 1 : UINT8 Quantization (`src/utils/quantization.py`)

| Modèle | Monitoring | Pump | CWRU | Pronostia | CMAPSS | Paderborn |
|--------|:----------:|:----:|:----:|:---------:|:------:|:---------:|
| EWC | ✅ exp_S24_01 | ❌ manquant | ❌ manquant | ❌ manquant | ❌ manquant | ❌ manquant |
| HDC | ✅ exp_S24_02 (profil INT natif) | ❌ | ❌ | ❌ | ❌ | ❌ |
| TinyOL | ✅ exp_004 | ❌ manquant | ❌ manquant | — | — | — |
| Mahalanobis | — (pas de réseau) | — | — | — | — | — |

**Trous prioritaires** :
- EWC UINT8 sur Monitoring (référence baseline → `exp_S24_01`)
- TinyOL UINT8 sur Pump temporal (comparaison exp_004 qui était sur Dataset2 → `exp_S24_12`)
- HDC : architecture déjà INT32/binarisée mais jamais profilée explicitement en mode INT8

### Amélioration 2 : Export ONNX (`scripts/export_onnx.py`)

| Modèle | Monitoring | Pump | CWRU | Pronostia | CMAPSS | Paderborn |
|--------|:----------:|:----:|:----:|:---------:|:------:|:---------:|
| EWC | ✅ exp_001 base | ❌ | ❌ | ❌ | ❌ | ❌ |
| HDC | ✅ exp_002 base | ❌ | ❌ | ❌ | ❌ | ❌ |
| TinyOL | ✅ exp_003 base | ❌ | ❌ | — | — | — |
| Mahalanobis | ❌ (non exporté Sprint 4) | ❌ | ❌ | ❌ | ❌ | ❌ |

**Trous prioritaires** : toutes les variantes multi-datasets des Sprints 5–23

### Amélioration 3 : Profiling RAM systématique (`scripts/profile_memory.py`)

| Modèle | Monitoring | Pump | CWRU | Pronostia | CMAPSS | Paderborn |
|--------|:----------:|:----:|:----:|:---------:|:------:|:---------:|
| EWC | ✅ Sprint 4 | ✅ Sprint 4 | ❌ | ❌ partiel | ❌ | ❌ |
| HDC | ✅ Sprint 4 | ✅ Sprint 4 | ❌ | ❌ partiel | ❌ | ❌ |
| TinyOL | ✅ Sprint 4 | ✅ Sprint 4 | ❌ | — | — | — |
| Mahalanobis | ✅ Sprint 14 | ✅ Sprint 15 | ❌ partiel | ❌ partiel | ❌ | ❌ |

**Trous prioritaires** : CWRU + Pronostia + CMAPSS + Paderborn pour tous modèles

---

## Plan de résolution

| Trou identifié | Tâche Sprint 24 | Exp cible |
|---------------|----------------|-----------|
| EWC UINT8 / Monitoring | S2402a + S2402b | ✅ exp_S24_01 (AA=0.911, RAM_uint8=705 B) |
| HDC INT8 profile / Monitoring | S2402c + S2402d | ✅ exp_S24_02 (compression 2.67×, AA=0.870) |
| ONNX tous datasets | S2403a + S2403b | onnx_sprint24/ |
| Profiling CWRU tous modèles | S2404b + S2405a | exp_S24_03, exp_S24_04–07 |
| Profiling Pronostia | S2405b | exp_S24_08–09 |
| Profiling Pump temporal avec ONNX | S2405c | exp_S24_10–12 |
| TinyOL UINT8 Pump temporal | S2402 + S2405c | exp_S24_12 |

---

## Expériences historiques non retestées (hors scope Sprint 24)

Les expériences suivantes des Sprints 5–21 ne sont **pas** relancées car elles ne bénéficient pas significativement des améliorations S4 (baselines non-neuronales, scénarios board déjà profilés, etc.) :

- exp_005 à exp_029 (KMeans, DBSCAN, PCA) — pas de réseau → UINT8/ONNX non applicables
- exp_030 à exp_039 (single task) — déjà couverts par les re-runs multi-task
- Expériences board Sprint 20–21 (exp_S20_*, exp_S21_*) — profiling board via DWT déjà en place
- Expériences Paderborn Sprints 22–23 — INT8 déjà inclus (S2217–S2222)

---

## Résultat attendu

À la complétion de S2401, ce document est mis à jour avec les statuts réels (✅/❌) et sert de référence pour valider que Sprint 24 a bien comblé tous les trous prioritaires.
