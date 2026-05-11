# S1003 — Portage C MVP : Mahalanobis + Tête EWC

| Champ | Valeur |
|-------|--------|
| **ID** | S1003 |
| **Sprint** | Sprint 16 — Semaine 3 (4–10 juin 2026) |
| **Priorité** | Critique |
| **Durée estimée** | 14h |
| **Dépendances** | S1001 (toolchain), S1002 (ONNX export) |
| **Fichiers cibles** | `firmware/stm32f4_blink/src/` |

---

## Objectif

Porter les deux modèles les plus légers en C natif sur NUCLEO-F439ZI :

1. **Mahalanobis** (80 B RAM) — le plus simple, zéro rétropropagation
2. **Tête EWC MLP** (3 couches, SGD 1 step) — le plus représentatif du workflow CL

**Pipeline cible** :

```
capteur simulé → normalisation Z-score → forward → décision (OK / anomalie)
                                        ↑ update SGD (EWC head uniquement)
```

---

## Implémentation Mahalanobis en C

### `firmware/stm32f4_blink/inc/mahalanobis.h`

```c
#pragma once
#include <stdint.h>

#define MAHA_DIM    5   /* Nombre de features (Monitoring dataset) */

typedef struct {
    float mean[MAHA_DIM];
    float precision[MAHA_DIM][MAHA_DIM]; /* Matrice précision (inv. covariance) */
    float threshold;
    float ema_alpha;                     /* EMA pour update incrémental */
} MahalanobisDetector;

void  maha_init(MahalanobisDetector *det, float threshold, float ema_alpha);
float maha_score(const MahalanobisDetector *det, const float *x);
void  maha_update(MahalanobisDetector *det, const float *x);
int   maha_predict(const MahalanobisDetector *det, const float *x);
```

**Budget RAM** :
- `mean[5]` = 20 B @ FP32
- `precision[5][5]` = 100 B @ FP32
- `threshold` + `ema_alpha` = 8 B
- **Total : ~128 B** (conforme contrainte 64 Ko)

### `firmware/stm32f4_blink/src/mahalanobis.c`

Implémenter :
- `maha_score` : distance de Mahalanobis `d = sqrt((x-μ)ᵀ Σ⁻¹ (x-μ))`
- `maha_update` : EMA sur `mean` (sans ré-inverser la matrice online)
- `maha_predict` : retourne 1 si score > threshold, 0 sinon

---

## Implémentation tête EWC MLP en C

### Architecture cible

```
Input (5) → ReLU(32) → ReLU(16) → Output (2)
```

Poids chargés depuis Flash (export depuis Python après entraînement).

### `firmware/stm32f4_blink/inc/ewc_head.h`

```c
#pragma once
#include <stdint.h>

#define EWC_IN      5
#define EWC_H1     32
#define EWC_H2     16
#define EWC_OUT     2
#define EWC_LR      0.01f

typedef struct {
    float w1[EWC_H1][EWC_IN];   float b1[EWC_H1];
    float w2[EWC_H2][EWC_H1];   float b2[EWC_H2];
    float w3[EWC_OUT][EWC_H2];  float b3[EWC_OUT];
    /* Fisher diagonale (régularisation EWC) */
    float fisher1[EWC_H1][EWC_IN];
    float fisher2[EWC_H2][EWC_H1];
    float fisher3[EWC_OUT][EWC_H2];
    float lambda;   /* Coefficient EWC */
    float star_w1[EWC_H1][EWC_IN];  /* Poids de référence tâche précédente */
    float star_w2[EWC_H2][EWC_H1];
    float star_w3[EWC_OUT][EWC_H2];
} EWCHead;

void  ewc_forward(const EWCHead *h, const float *x, float *out);
int   ewc_predict(const EWCHead *h, const float *x);
void  ewc_sgd_step(EWCHead *h, const float *x, int label);
```

**Budget RAM** :
- Poids : (5×32 + 32 + 32×16 + 16 + 16×2 + 2) × 4 = ~3 Ko
- Fisher + star (×2 poids) : ~6 Ko
- **Total : ~10 Ko** (conforme contrainte 64 Ko)

---

## Pipeline intégration

### `firmware/stm32f4_blink/src/pipeline.c`

```c
/* Pipeline minimaliste : UART → normalisation → décision → LED */
void pipeline_run(void) {
    float raw[5];
    uart_receive_sample(raw);         /* Données capteur via UART */
    normalize_zscore(raw, MAHA_DIM);  /* Stats figées en Flash */

    int anomaly = maha_predict(&g_detector, raw);
    led_set(anomaly ? LED_RED : LED_GREEN);

    maha_update(&g_detector, raw);    /* Update incrémental */
}
```

---

## Script de génération des poids C

Créer `scripts/export_weights_c.py` pour convertir les poids PyTorch en tableaux C :

```python
def export_to_c_header(model, output_path: str) -> None:
    """Génère inc/model_weights.h depuis un checkpoint PyTorch."""
    # Pour chaque couche : np.array2string(w.numpy(), separator=', ')
    # Format : static const float W1[H1][IN] = { {...}, ... };
```

---

## Critères d'acceptation

- [x] `mahalanobis.c` compile sans warning avec `-Wall -Wextra`
- [x] Score Mahalanobis identique entre Python et C (tolérance 1e-4) — validé via tests Unity (9/9 PASS)
- [x] `ewc_head.c` forward pass identique à PyTorch (tolérance 1e-4) — validé via tests Unity (7/7 PASS)
- [x] `ewc_sgd_step` réduit la loss sur 10 samples synthétiques
- [ ] Pipeline complet tourne sur NUCLEO-F439ZI sans hardfault — en attente accès board

---

## Notes

- Pas de malloc, pas de stdlib — allocation statique uniquement
- Annotations MEM obligatoires sur chaque struct (conforme CLAUDE.md)
- Stats Z-score (mean/std par feature) stockées en Flash comme constantes C

**Complété le** : 2026-05-11 (implémentation C + tests host) — test on-board en attente
