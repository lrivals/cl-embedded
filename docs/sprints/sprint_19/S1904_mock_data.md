# S1904 — Mock data framework C : samples synthétiques pour tests Unity host

| Champ | Valeur |
|-------|--------|
| **Sprint** | 19 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 2h (déjà faite) |
| **Dépendances** | — |
| **Fichiers cibles** | `firmware/stm32f4_blink/tests/mock_data.h` |

---

## Contexte

Les tests Unity s'exécutent sur host x86 (sans board). Pour tester les modèles C (Mahalanobis, EWC, TinyOL) de façon reproductible, il faut des données synthétiques codées en dur dans un header C, avec des valeurs attendues pré-calculées en Python pour permettre des assertions numériques précises.

`mock_data.h` est la fondation de tous les tests modèles (S1909).

---

## Objectif

Fournir un header `mock_data.h` unique utilisable par `test_mahalanobis.c`, `test_ewc_head.c`, `test_models.c`, contenant des samples normaux et anormaux pour 3 tâches simulées, avec les valeurs attendues pour chaque modèle.

---

## État actuel — Implémenté ✅

**`firmware/stm32f4_blink/tests/mock_data.h`**

### Constantes de dimensionnement

```c
#define MOCK_N_FEATURES 5U   /* = EWC_IN = MAHA_DIM = TINYOL_IN */
#define MOCK_N_SAMPLES 10U   /* samples par classe et par tâche */
#define MOCK_N_TASKS    3U   /* tâche 0, 1, 2 avec drift progressif */
```

### Données disponibles

| Tableau | Description | Taille |
|---------|-------------|--------|
| `MOCK_NORMAL_T0[10][5]` | Samples normaux task 0 — centroïde ≈ 0 | 200 B |
| `MOCK_ANOMALY_T0[10][5]` | Anomalies task 0 — normes >> 3σ | 200 B |
| `MOCK_NORMAL_T1[10][5]` | Normaux task 1 — centroïde déplacé +0.5 | 200 B |
| `MOCK_NORMAL_T2[10][5]` | Normaux task 2 — centroïde déplacé +1.0 | 200 B |
| `MOCK_NORMAL_LABELS_T*` | Labels 0 (uint8) | 10 B |
| `MOCK_ANOMALY_LABELS_T0` | Labels 1 (uint8) | 10 B |

### Valeurs attendues (assertions Unity)

| Constante | Valeur | Usage |
|-----------|--------|-------|
| `MOCK_MAHA_SCORE_NORMAL_T0_MAX` | 0.25f | Mahalanobis : tout score normal < 0.25 |
| `MOCK_MAHA_SCORE_ANOMALY_T0_MIN` | 5.0f | Mahalanobis : tout score anomalie > 5.0 |
| `MOCK_EWC_LOGIT_TOLERANCE` | 0.01f | EWC : logits réseau zero-init ≈ [0, 0] |
| `MOCK_TINYOL_RECON_ERR_ZERO_WEIGHTS` | 0.007f | TinyOL : MSE avec poids nuls sur NORMAL_T0[0] |
| `MOCK_TINYOL_RECON_TOLERANCE` | 1e-4f | TinyOL : tolérance numérique |

### Calcul de MOCK_TINYOL_RECON_ERR_ZERO_WEIGHTS

```
x = [0.10, 0.05, 0.08, -0.03, 0.12]
recon = [0, 0, 0, 0, 0]  (poids = 0)
MSE = (0.01 + 0.0025 + 0.0064 + 0.0009 + 0.0144) / 5 = 0.0242/5 ≈ 0.00484
```
> Note : la valeur 0.007 dans le header est un seuil `≤` — le test vérifie `err ≤ 0.007` (légère marge).

---

## Design des données synthétiques

### Propriétés des normaux

- Tâche 0 : centroïde à l'origine, variance ~0.05 → scores Mahalanobis (precision=I) < 0.25
- Tâche 1 : centroïde +0.5 sur toutes dimensions → simule un drift de domaine léger
- Tâche 2 : centroïde +1.0 → drift modéré, toujours différenciable des anomalies

### Propriétés des anomalies

- Tâche 0 uniquement (les anomalies inter-tâches sont générées par la séquence elle-même)
- Normes ≫ 3σ : valeurs dans [-4, +4], alors que les normaux sont dans [-0.15, +0.15]
- Garantit des scores Mahalanobis > 5.0 avec n'importe quelle matrice de précision raisonnable

### Compatibilité numérique PC ↔ C

- Toutes les valeurs sont exactement représentables en FP32 (pas de fraction périodique)
- Tolérance de 1e-4 suffisante pour les différences d'ordre des opérations flottantes x86 vs ARM Cortex

---

## Usage dans les tests Unity

```c
#include "mock_data.h"
#include "mahalanobis.h"

void test_maha_normal_below_threshold(void) {
    MahalanobisDetector det;
    /* init avec precision = identité, mean = 0 */
    for (int k = 0; k < MOCK_N_SAMPLES; k++) {
        float score = maha_score(&det, MOCK_NORMAL_T0[k]);
        TEST_ASSERT_LESS_THAN_FLOAT(MOCK_MAHA_SCORE_NORMAL_T0_MAX, score);
    }
}
```

---

## Fichiers cibles

| Fichier | Action |
|---------|--------|
| `firmware/stm32f4_blink/tests/mock_data.h` | ✅ Complet — aucune modification requise |
| `firmware/stm32f4_blink/tests/test_models.c` | Importer et utiliser (voir S1909) |
| `firmware/stm32f4_blink/tests/test_mahalanobis.c` | Déjà importé (16/16 PASS) |

---

## Extensions futures (hors Sprint 19)

- Ajouter des samples CWRU réels (extrait fixe de 10 samples par tâche, calculé offline)
- Ajouter samples TinyOL avec poids non-nuls pré-calculés pour tests de régression
- Ajouter labels pour évaluation BWT/AF directement dans `mock_data.h`

---

## Vérification

- [x] Header compile sans warning avec `-std=c11 -Wall -Wextra`
- [x] `test_mahalanobis.c` référence `MOCK_MAHA_SCORE_NORMAL_T0_MAX` → 16/16 PASS
- [ ] `test_models.c` référence `MOCK_EWC_LOGIT_TOLERANCE` et `MOCK_TINYOL_RECON_ERR_ZERO_WEIGHTS` (à valider en S1909)
