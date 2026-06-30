# S2701–S2704 — Bloc DUAL_MODE dans `pipeline.c` / `pipeline.h`

| Champ | Valeur |
|-------|--------|
| **Sprint** | 27 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Implémenté — firmware compilé (0 erreur), tests T76–T79 PASS, validé board |
| **Durée estimée** | ~3h30 |
| **Dépendances** | Sprint 26 ✅ — `ewc_head_regression.c`, `ewc_head_multiclass.c`, `pipeline.c` v3 avec `g_ewc_reg` + `g_ewc_mc` alloués statiquement |
| **Fichiers cibles** | `firmware/stm32f4_blink/inc/pipeline.h`, `firmware/stm32f4_blink/src/pipeline.c` |
| **Référence** | Bloc `MULTICLASS_MODE` (lignes ~368–388 de `pipeline.c`) — pattern à adapter pour le dual dispatch |

---

## Contexte

`pipeline_run()` dispatche chaque trame UART vers un modèle selon les bits du byte `FLAGS`. Le Sprint 26 a ajouté :
- `PROTO_FLAG_RUL_MODE = 0x50` → `g_ewc_reg` (régression RUL)
- `PROTO_FLAG_MULTICLASS_MODE = 0x30` → `g_ewc_mc` (classification faute)

Sprint 27 ajoute `PROTO_FLAG_DUAL_MODE = 0x70` qui déclenche les **deux modèles en séquence** sur la même trame et retourne une réponse étendue de 25 octets.

---

## S2701 — `pipeline.h` : defines DUAL_MODE

Ajouter après la ligne `PROTO_FLAG_MULTICLASS_MODE` :

```c
/* Sprint 27 — DUAL_MODE : EWC_REG (RUL) + EWC_MC (faute) pipeline simultané
 * Valeur : EWC_MODE(0x10) | HDC_MODE(0x20) | INT8_MODE(0x40) = 0x70
 * ATTENTION : 0x70 & 0x30 == 0x30 — le bloc DUAL_MODE doit passer AVANT
 *             le check MULTICLASS_MODE dans pipeline_run() */
#define PROTO_FLAG_DUAL_MODE    (PROTO_FLAG_EWC_MODE | PROTO_FLAG_HDC_MODE | PROTO_FLAG_INT8_MODE)  /* 0x70 */
#define RESPONSE_DUAL_SIZE       25U  /* [pred_fault:u8][conf_fault:f32][rul_pred:f32]
                                       * [lat_us:u32][f1_macro:f32][rmse_rul:f32][forgetting:f32] */
```

---

## S2702 — `pipeline.c` : helper `uart_send_response_dual()`

Insérer juste après `uart_send_response_v3()`. Même pattern union float/bytes que les helpers existants.

```c
/* Sprint 27 — Réponse 25 B pour DUAL_MODE
 * Layout : [pred_fault:u8][conf_fault:f32][rul_pred:f32][lat_us:u32]
 *          [f1_macro:f32][rmse_rul:f32][forgetting:f32]             */
static void uart_send_response_dual(uint8_t pred_fault, float conf_fault,
                                     float rul_pred, float f1_macro,
                                     float rmse_rul, float forgetting)
{
    union { float f; uint8_t b[4]; } uc;
    uint8_t prof_buf[PROFILING_ENCODED_SIZE];
    profiling_encode(prof_buf);   /* [lat_us:u32][ram_b:u16][thr:u16] */

    uart_send_byte(pred_fault);   /* offset 0 — classe faute prédite */

    uc.f = conf_fault;            /* offset 1–4 — confiance softmax */
    uart_send_byte(uc.b[0]); uart_send_byte(uc.b[1]);
    uart_send_byte(uc.b[2]); uart_send_byte(uc.b[3]);

    uc.f = rul_pred;              /* offset 5–8 — RUL prédit (float) */
    uart_send_byte(uc.b[0]); uart_send_byte(uc.b[1]);
    uart_send_byte(uc.b[2]); uart_send_byte(uc.b[3]);

    /* offset 9–12 — latence combinée DWT (u32, µs) */
    uart_send_byte(prof_buf[0]); uart_send_byte(prof_buf[1]);
    uart_send_byte(prof_buf[2]); uart_send_byte(prof_buf[3]);

    uc.f = f1_macro;              /* offset 13–16 — F1-macro faute */
    uart_send_byte(uc.b[0]); uart_send_byte(uc.b[1]);
    uart_send_byte(uc.b[2]); uart_send_byte(uc.b[3]);

    uc.f = rmse_rul;              /* offset 17–20 — RMSE RUL */
    uart_send_byte(uc.b[0]); uart_send_byte(uc.b[1]);
    uart_send_byte(uc.b[2]); uart_send_byte(uc.b[3]);

    uc.f = forgetting;            /* offset 21–24 — forgetting moyen */
    uart_send_byte(uc.b[0]); uart_send_byte(uc.b[1]);
    uart_send_byte(uc.b[2]); uart_send_byte(uc.b[3]);
    /* Total : 1 + 4 + 4 + 4 + 4 + 4 + 4 = 25 B ✅ */
}
```

---

## S2703 — `pipeline.c` : bloc DUAL_MODE dans `pipeline_run()`

Insérer comme **premier bloc `if`** dans la chaîne de dispatch — après `profiling_start()`, **avant** le check `MULTICLASS_MODE` existant.

```c
    /* ── DUAL_MODE (0x70) : EWC_REG (RUL) + EWC_MC (faute) ──────────────────
     * Encodage labels :
     *   g_recv_task_id = fault_label ∈ [0, EWC_MC_N_CLASSES-1]  (byte TASK_ID réutilisé)
     *   g_recv_label   = rul_u8 = round(RUL / 300 × 255) → re-normalisé en [0,1]
     * Features :
     *   raw[0..4] → g_ewc_reg  (EWC_REG_IN = 5)
     *   raw[0..8] → g_ewc_mc   (EWC_MC_IN  = 9)
     * DOIT ÊTRE EN PREMIER — 0x70 & 0x30 == 0x30 matcherait MULTICLASS sinon */
    if ((g_recv_flags & PROTO_FLAG_DUAL_MODE) == PROTO_FLAG_DUAL_MODE) {
        uint8_t fault_label    = g_recv_task_id;
        float   rul_label_norm = (float)g_recv_label / 255.0f;

        /* ── EWC_REG : prédiction RUL sur raw[0..4] ─────────────────────── */
        float rul_pred = ewc_reg_predict(&g_ewc_reg, raw);

        /* ── EWC_MC : classification faute sur raw[0..8] ────────────────── */
        float logits[EWC_MC_N_CLASSES];   /* MEM: N×4 B stack */
        ewc_mc_forward(&g_ewc_mc, raw, logits);
        int fault_pred = ewc_mc_predict(&g_ewc_mc, raw);

        /* Confiance softmax (numériquement stable — identique bloc MULTICLASS) */
        float max_l = logits[0];
        for (int j = 1; j < EWC_MC_N_CLASSES; j++)
            if (logits[j] > max_l) max_l = logits[j];
        float sum_exp = 0.0f;
        for (int j = 0; j < EWC_MC_N_CLASSES; j++)
            sum_exp += expf(logits[j] - max_l);
        float conf_fault = expf(logits[fault_pred] - max_l) / sum_exp;

        /* ── Online learning (UPDATE flag) ──────────────────────────────── */
        if (g_recv_flags & PROTO_FLAG_UPDATE) {
            ewc_reg_sgd_step(&g_ewc_reg, raw, rul_label_norm);
            ewc_mc_sgd_step(&g_ewc_mc, raw, (int)fault_label);
        }

        /* ── Consolidation (CONSOLIDATE flag) ───────────────────────────── */
        if (g_recv_flags & PROTO_FLAG_CONSOLIDATE) {
            ewc_reg_consolidate(&g_ewc_reg, EWC_REG_FISHER_DECAY);
            ewc_mc_consolidate(&g_ewc_mc, EWC_MC_FISHER_DECAY);
            g_current_task_id++;
        }

        /* ── Métriques dual ──────────────────────────────────────────────── */
        online_rmse_update(&g_rmse, rul_pred, rul_label_norm);
        online_f1_update(&g_f1, fault_pred, (int)fault_label);

        profiling_stop();

        uart_send_response_dual(
            (uint8_t)fault_pred, conf_fault, rul_pred,
            online_f1_get(&g_f1),
            online_rmse_get(&g_rmse),
            fgt_avg_forgetting(&g_fgt)
        );
        return;   /* court-circuit — pas de uart_send_response_v3 en bas */
    }

    /* ── MULTICLASS_MODE (0x30) — inchangé ─────────────────────────────── */
    if ((g_recv_flags & PROTO_FLAG_MULTICLASS_MODE) == PROTO_FLAG_MULTICLASS_MODE) {
        /* ... code Sprint 26 intact ... */
    }
```

---

## S2704 — TEST_MODE exposure

Ajouter dans le bloc `#ifdef TEST_MODE` de `pipeline.c` (après les expositions existantes) :

```c
/* Sprint 27 — Expose uart_send_response_dual pour test taille trame */
void test_pipeline_send_response_dual(uint8_t pred_fault, float conf_fault,
                                       float rul_pred, float f1_macro,
                                       float rmse_rul, float forgetting)
{
    uart_send_response_dual(pred_fault, conf_fault, rul_pred,
                             f1_macro, rmse_rul, forgetting);
}
```

---

## Vérification

```bash
cd firmware/stm32f4_blink

# 1. Compilation ARM (cross-compile)
make -j4
# → 0 warnings, 0 erreurs

# 2. Tests host
make test
# → 79 Tests 0 Failures (75 existants + 4 nouveaux)

# 3. Vérification taille binaire
arm-none-eabi-size build/stm32f4_blink.elf
# → .bss ≈ 65 266 B (< 262144 B = 256 Ko) ✅
```

---

## Résultats attendus

| Sous-tâche | Assertion de test |
|-----------|-------------------|
| S2701 define | `PROTO_FLAG_DUAL_MODE == 0x70` à la compilation |
| S2702 helper | `uart_tx_count == 25` après appel |
| S2703 dispatch | Trame `0x70, N=9` → 25 B ; trame `0x30, N=9` → 21 B (non-régression) |
| S2704 TEST_MODE | Fonction appelable depuis `test_pipeline.c` |

---

## Questions ouvertes

- `FIXME(gap2)` : Vérifier que `logits[EWC_MC_N_CLASSES]` sur la pile du bloc DUAL_MODE (jusqu'à 40 B pour N=10) ne sature pas la pile Cortex-M4 cumulée avec les buffers `dh1`/`dh2` du SGD (estimé < 600 B total, bien dans `_Min_Stack_Size = 0x400`).
- `TODO(dorra)` : Si l'overhead de `uart_send_response_dual` (4 bytes de plus que v3) pose un problème de timing UART à 115200 bps, augmenter le baudrate à 230400 bps dans `hw_uart_init()`. À valider sur board.
