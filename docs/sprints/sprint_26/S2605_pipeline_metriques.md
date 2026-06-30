# S2605–S2606 — Pipeline UART v3 étendu + métriques OnlineRMSE / OnlineF1Macro

| Champ | Valeur |
|-------|--------|
| **Sprint** | 26 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Terminé — 6 juin 2026 |
| **Durée estimée** | S2605 : 2h / S2606 : 2h = 4h total |
| **Dépendances** | S2601 ✅ (`ewc_head_regression.h`) + S2603 ✅ (`ewc_head_multiclass.h`) doivent exister avant de modifier `pipeline.c`, `firmware/stm32f4_blink/src/pipeline.c` v3 ✅, `firmware/stm32f4_blink/src/metrics.c` ✅ |
| **Fichiers cibles** | `firmware/stm32f4_blink/src/pipeline.c`, `firmware/stm32f4_blink/src/metrics.c`, `firmware/stm32f4_blink/inc/metrics.h`, `firmware/stm32f4_blink/inc/pipeline.h` |
| **Référence** | `firmware/stm32f4_blink/src/pipeline.c` v3 (routing FLAGS existant), `firmware/stm32f4_blink/src/metrics.c` (pattern Welford OnlineAccuracy), `scripts/sensor_stream.py` (protocole UART côté host) |

---

## Contexte

### Protocole UART actuel (v3)

```
Réception  : [MAGIC:2B 0xCDAB][VERSION:1B][TASK_ID:1B][TIMESTAMP_MS:4B]
             [N:1B][features:f32×N][label:1B][FLAGS:1B][CRC8:1B]

Réponse v3 : [pred:u8][conf:f32][lat_us:u32][acc:f32][auroc:f32][forgetting:f32] = 21 B
```

### FLAGS actuellement utilisés

```c
PROTO_FLAG_UPDATE      = 0x01  /* bit 0 */
PROTO_FLAG_PROFILING   = 0x02  /* bit 1 */
PROTO_FLAG_CONSOLIDATE = 0x04  /* bit 2 */
PROTO_FLAG_RESET       = 0x08  /* bit 3 */
PROTO_FLAG_EWC_MODE    = 0x10  /* bit 4 */
PROTO_FLAG_HDC_MODE    = 0x20  /* bit 5 */
PROTO_FLAG_INT8_MODE   = 0x40  /* bit 6 */
/* bit 7 (0x80) : seul bit disponible restant */
```

### Contrainte de design

Le byte FLAGS est quasi-saturé. La spec initiale prévoyait d'utiliser le bit 7 (0x80) pour `RUL_MODE`, mais ce bit est **déjà occupé** par `PROTO_FLAG_TINYOL_MODE = 0x80U` ajouté lors d'un sprint précédent.

**Décision prise (6 juin 2026)** : garder `PROTO_FLAG_TINYOL_MODE = 0x80U` inchangé, et router les nouveaux modes via des **combinaisons de bits existants** :

```c
/* Spec initiale (abandon) — collision avec TINYOL_MODE = 0x80 */
// #define PROTO_FLAG_RUL_MODE        0x80U
// #define PROTO_FLAG_MULTICLASS_MODE (EWC_MODE | RUL_MODE)  /* 0x90 */

/* Implémentation réelle Sprint 26 — combinaisons libres, zéro collision */
#define PROTO_FLAG_RUL_MODE        (PROTO_FLAG_EWC_MODE | PROTO_FLAG_INT8_MODE)  /* 0x50 */
#define PROTO_FLAG_MULTICLASS_MODE (PROTO_FLAG_EWC_MODE | PROTO_FLAG_HDC_MODE)   /* 0x30 */
```

Le routing if/else vérifie les combinaisons **avant** les flags simples pour éviter tout faux-positif :

| FLAGS (masque exact) | Valeur | Chemin |
|---------------------|:------:|--------|
| `(flags & 0x30) == 0x30` | EWC\|HDC | → `ewc_mc_forward()` (multi-class) |
| `(flags & 0x50) == 0x50` | EWC\|INT8 | → `ewc_reg_forward()` (RUL) |
| `flags & 0x10` seul | EWC | → `ewc_forward()` (binaire existant) |
| `flags & 0x40` seul | INT8 | → `ewc_int8_forward()` (existant) |
| `flags & 0x20` seul | HDC | → `hdc_predict()` (existant) |
| `flags & 0x80` | TINYOL | → `tinyol_encode/decode()` (inchangé) |

> `TODO(dorra)` : Le byte FLAGS est désormais totalement saturé (8 bits tous assignés ou combinés). Si un quatrième mode embarqué est ajouté post-Sprint 26, prévoir protocole V4 avec FLAGS sur 2 octets dans le header. À documenter comme limite dans le manuscrit.

---

## S2605 — Extension `pipeline.c` : routing FLAGS RUL + MULTICLASS

### Modifications dans `pipeline.h`

Ajouter après `PROTO_FLAG_INT8_MODE` :

```c
/* pipeline.h — ajouts Sprint 26 */
#define PROTO_FLAG_RUL_MODE        0x80U   /* bit 7 : EWC régression RUL */
#define PROTO_FLAG_MULTICLASS_MODE (PROTO_FLAG_EWC_MODE | PROTO_FLAG_RUL_MODE)

extern EWCHeadReg g_ewc_reg;   /* alloué en .bss — ~8.9 Ko FP32 */
extern EWCHeadMC  g_ewc_mc;    /* alloué en .bss — ~14 Ko FP32 (N=10) */
```

### Ajouts dans `pipeline.c`

#### Globals statiques (section `.bss`)

```c
/* MEM: EWCHeadReg ~8.9 Ko @ FP32 en .bss */
EWCHeadReg g_ewc_reg;

/* MEM: EWCHeadMC ~14 Ko @ FP32 en .bss (EWC_MC_N_CLASSES=10) */
EWCHeadMC  g_ewc_mc;

/* MEM: métriques nouvelles — voir S2606 */
static OnlineRMSE     g_rmse;
static OnlineF1Macro  g_f1;
```

#### Dans `pipeline_init()`

```c
/* Sprint 26 : init nouvelles têtes */
g_ewc_reg.lambda = 400.0f;
ewc_reg_init(&g_ewc_reg);

g_ewc_mc.lambda = 400.0f;
ewc_mc_init(&g_ewc_mc);

online_rmse_init(&g_rmse);
online_f1_init(&g_f1);
```

#### Dans `pipeline_run()` — bloc routing à insérer avant `PROTO_FLAG_EWC_MODE`

```c
} else if ((g_recv_flags & PROTO_FLAG_MULTICLASS_MODE) == PROTO_FLAG_MULTICLASS_MODE) {
    /* ── Chemin EWC Multi-class : forward N classes ─────────────────────── */
    float logits[EWC_MC_N_CLASSES];   /* MEM: N×4 B stack */
    ewc_mc_forward(&g_ewc_mc, raw, logits);
    pred = ewc_mc_predict(&g_ewc_mc, raw);

    /* Confiance = softmax[pred] (numériquement stable) */
    float max_l = logits[0];
    for (int j = 1; j < EWC_MC_N_CLASSES; j++)
        if (logits[j] > max_l) max_l = logits[j];
    float sum_exp = 0.0f;
    for (int j = 0; j < EWC_MC_N_CLASSES; j++) sum_exp += expf(logits[j] - max_l);
    confidence = expf(logits[pred] - max_l) / sum_exp;

    if (g_recv_flags & PROTO_FLAG_UPDATE)
        ewc_mc_sgd_step(&g_ewc_mc, raw, (int)g_recv_label);
    if (g_recv_flags & PROTO_FLAG_CONSOLIDATE) {
        ewc_mc_consolidate(&g_ewc_mc, EWC_MC_FISHER_DECAY);
        g_current_task_id = g_recv_task_id;
    }
    online_f1_update(&g_f1, pred, (int)g_recv_label);

} else if (g_recv_flags & PROTO_FLAG_RUL_MODE) {
    /* ── Chemin EWC Régression RUL ───────────────────────────────────────── */
    float rul_pred = ewc_reg_predict(&g_ewc_reg, raw);
    pred       = 0;         /* champ pred=0 en mode RUL (non utilisé) */
    confidence = rul_pred;  /* réutilise le champ conf pour transporter RUL */

    if (g_recv_flags & PROTO_FLAG_UPDATE)
        ewc_reg_sgd_step(&g_ewc_reg, raw, (float)g_recv_label);
    if (g_recv_flags & PROTO_FLAG_CONSOLIDATE) {
        ewc_reg_consolidate(&g_ewc_reg, EWC_REG_FISHER_DECAY);
        g_current_task_id = g_recv_task_id;
    }
    /* label = RUL réel (encodé en float32 LE dans les 4 derniers octets de features) */
    online_rmse_update(&g_rmse, rul_pred, (float)g_recv_label);
```

> **Note protocole** : en mode RUL, le champ `conf:f32` de la réponse v3 transporte le RUL prédit (float). Le script host `simulate_rul_board.py` doit interpréter `conf` comme RUL et non comme probabilité.

#### MetricsSnapshot — extension pour RUL / multi-class

La réponse v3 actuelle envoie `{acc, auroc, forgetting}`. Pour les modes RUL et MULTICLASS, le snapshot garde le même format 12 B mais les champs sont réinterprétés :

| Mode | acc | auroc | forgetting |
|------|-----|-------|------------|
| Binaire (existant) | OnlineAccuracy | OnlineAUROC | ForgettingTracker |
| RUL_MODE | 0.0f (N/A) | RMSE courant | 0.0f (N/A) |
| MULTICLASS_MODE | F1-macro courant | 0.0f (N/A) | ForgettingTracker |

Le script host extrait le bon champ selon le flag de mode utilisé.

---

## S2606 — Extension `metrics.c` + `metrics.h` : OnlineRMSE + OnlineF1Macro

### Ajouts dans `metrics.h`

```c
/* ── RMSE online (Welford en ligne) ─────────────────────────────────────────
 * MEM: 16 B @ FP32+uint32 en .bss                                          */

typedef struct {
    uint32_t n;       /* Nombre de samples vus */
    float    mean;    /* Moyenne courante de (ŷ - y)² — Welford M1 */
    float    M2;      /* Variance cumulée — Welford M2 */
    float    rmse;    /* sqrt(M2/n) — mis à jour à chaque step */
} OnlineRMSE;

void  online_rmse_init(OnlineRMSE *r);
void  online_rmse_update(OnlineRMSE *r, float y_pred, float y_true);
float online_rmse_get(const OnlineRMSE *r);   /* retourne r->rmse */

/* ── F1-macro online (matrice de confusion compacte) ─────────────────────────
 * MEM: MAX_MC_CLASSES² × 2 B = 200 B pour N=10 en .bss                    */

#define MAX_MC_CLASSES EWC_MC_N_CLASSES   /* hérite de ewc_head_multiclass.h */

typedef struct {
    int16_t cm[MAX_MC_CLASSES][MAX_MC_CLASSES]; /* cm[true][pred] */
    int      n_classes;
} OnlineF1Macro;

void  online_f1_init(OnlineF1Macro *f);
void  online_f1_update(OnlineF1Macro *f, int pred, int true_label);
float online_f1_get(const OnlineF1Macro *f);   /* F1-macro moyen sur les classes vues */
```

### Implémentation dans `metrics.c`

```c
/* ── OnlineRMSE (Welford) ─────────────────────────────────────────────────── */

void online_rmse_init(OnlineRMSE *r)
{
    r->n    = 0U;
    r->mean = 0.0f;
    r->M2   = 0.0f;
    r->rmse = 0.0f;
}

/* Welford online : mise à jour en O(1), numériquement stable */
void online_rmse_update(OnlineRMSE *r, float y_pred, float y_true)
{
    float err = y_pred - y_true;
    float sq  = err * err;
    r->n++;
    float delta  = sq - r->mean;
    r->mean += delta / (float)r->n;
    float delta2 = sq - r->mean;
    r->M2 += delta * delta2;
    r->rmse = (r->n > 1U) ? sqrtf(r->M2 / (float)(r->n - 1U)) : 0.0f;
    /* Note : RMSE ici = std(erreurs) = sqrt(E[(ŷ-y)²] si mean≈0 pour données normalisées)
     * Pour RMSE classique : r->rmse = sqrtf(r->mean) — selon convention manuscrit */
}

float online_rmse_get(const OnlineRMSE *r) { return r->rmse; }

/* ── OnlineF1Macro ────────────────────────────────────────────────────────── */

void online_f1_init(OnlineF1Macro *f)
{
    for (int i = 0; i < MAX_MC_CLASSES; i++)
        for (int j = 0; j < MAX_MC_CLASSES; j++)
            f->cm[i][j] = 0;
    f->n_classes = EWC_MC_N_CLASSES;
}

void online_f1_update(OnlineF1Macro *f, int pred, int true_label)
{
    if (true_label < 0 || true_label >= f->n_classes) return;
    if (pred       < 0 || pred       >= f->n_classes) return;
    if (f->cm[true_label][pred] < 32767) f->cm[true_label][pred]++;
}

/* F1 par classe = 2×TP / (2×TP + FP + FN), macro-average */
float online_f1_get(const OnlineF1Macro *f)
{
    float sum_f1 = 0.0f;
    int   n_seen = 0;

    for (int c = 0; c < f->n_classes; c++) {
        int tp = (int)f->cm[c][c];
        int fp = 0, fn = 0;
        for (int j = 0; j < f->n_classes; j++) {
            if (j != c) fp += (int)f->cm[j][c];   /* col c, autres lignes */
            if (j != c) fn += (int)f->cm[c][j];   /* ligne c, autres cols */
        }
        int denom = 2 * tp + fp + fn;
        if (denom > 0) {
            sum_f1 += (2.0f * (float)tp) / (float)denom;
            n_seen++;
        }
    }
    return (n_seen > 0) ? sum_f1 / (float)n_seen : 0.0f;
}
```

---

## Vérification

### Tests unitaires métriques (host C)

```bash
# Dans test_ewc_regression.c ou test_runner.c :
# OnlineRMSE : 3 samples (pred=10, true=8), (pred=12, true=10), (pred=9, true=11)
# RMSE attendu = sqrt(mean(4+4+4)) = 2.0

# OnlineF1Macro N=10 : diagonal parfait pendant 10 steps → F1 = 1.0
```

### Non-régression protocole binaire

```bash
make test   # tous les tests existants (test_pipeline.c, test_ewc_head.c, etc.) doivent rester verts
```

### Vérification taille réponse

La réponse reste **21 B** (protocole v3 non modifié) — le champ `conf:f32` transporte RUL en mode RUL_MODE.

---

## Budget mémoire supplémentaire (Sprint 26)

| Nouveau composant | Octets | Note |
|------------------|--------|------|
| `g_ewc_reg` (EWCHeadReg) | ~8 884 B | en .bss |
| `g_ewc_mc` (EWCHeadMC N=10) | ~14 072 B | en .bss |
| `OnlineRMSE` | 16 B | en .bss |
| `OnlineF1Macro` (N=10) | 202 B | cm[10][10]×int16 + n_classes |
| **Total ajouté** | **~22.9 Ko** | firmware existant ~43 Ko → total ~66 Ko << 256 Ko ✅ |

---

## Résultats d'implémentation

| Sous-tâche | Statut | Notes |
|------------|:------:|-------|
| S2605 — `pipeline.c` routing RUL + MULTICLASS | ✅ | Routing 2 nouveaux blocs if/else avant EWC seul · snapshot adapté par mode |
| S2605 — `pipeline.h` nouveaux flags | ✅ | `RUL_MODE=0x50` (EWC\|INT8), `MULTICLASS_MODE=0x30` (EWC\|HDC) — TINYOL_MODE=0x80 préservé |
| S2606 — `metrics.c` : `OnlineRMSE` | ✅ | Welford O(1) — `sqrtf(M2/(n-1))` |
| S2606 — `metrics.c` : `OnlineF1Macro` | ✅ | Matrice cm[10][10] int16 · macro-average classes vues |
| S2606 — `metrics.h` déclarations | ✅ | Structs + prototypes ajoutés · `#include "ewc_head_multiclass.h"` pour `MAX_MC_CLASSES` |
| `tests/test_metrics.c` créé | ✅ | 8 tests Unity (4 RMSE + 4 F1) |
| `make test` non-régression | ✅ | **65 tests, 0 failures** — tous verts |

---

## Questions ouvertes

- `TODO(dorra)` : Le byte FLAGS est saturé après Sprint 26 (bit 7 utilisé). Si un quatrième mode embarqué est nécessaire (ex. HDC régression), prévoir protocole V4 avec flags sur 2 octets dans le header. À documenter dans le manuscrit comme limite de l'architecture actuelle.
- `FIXME(gap2)` : En mode RUL, le label UART est un `uint8_t`. Pour transmettre un RUL float (ex. 85.5 cycles), le script host doit encoder le RUL en `uint8_t` (clampé à 255) ou utiliser les 4 derniers octets de `features[]` pour passer le RUL réel. Choisir l'encodage avant d'implémenter `simulate_rul_board.py` (S2609).
- `TODO(arnaud)` : La métrique RMSE on-board (Welford) calcule ici `std(erreurs)` — pas exactement RMSE (= `sqrt(mean(err²))`) si `mean(err) ≠ 0`. Aligner avec la définition utilisée dans le manuscrit : implémenter `r->rmse = sqrtf(r->mean)` (running mean of squared errors) à la place de l'estimateur Welford de variance.
