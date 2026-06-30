# S2705–S2710 — Tests Unity DUAL_MODE dans `test_pipeline.c`

| Champ | Valeur |
|-------|--------|
| **Sprint** | 27 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Implémenté (2026-06-12) — `make test` : 79 tests, T76–T79 PASS (4/4), 2 échecs TinyOL préexistants hors périmètre |
| **Durée estimée** | ~2h45 |
| **Dépendances** | S2703 ✅ (bloc DUAL_MODE dans `pipeline.c`), S2704 ✅ (TEST_MODE exposure), `firmware/stm32f4_blink/tests/test_pipeline.c` (75 tests existants) |
| **Fichiers cibles** | `firmware/stm32f4_blink/tests/test_pipeline.c` |
| **Référence** | Tests existants `test_pipeline_response_v3_21bytes()`, `test_pipeline_multiclass_dispatch()` — même pattern |

---

## Contexte

`test_pipeline.c` contient les tests Unity pour le protocole UART et le dispatch des modèles. Il compile en mode hôte (`TEST_MODE=1`, gcc non cross-compilé) et simule les buffers UART via un stub.

Sprint 26 est à 75/75. Sprint 27 ajoute **4 tests** (T76–T79) pour le mode DUAL_MODE, atteignant **79/79 PASS** au total.

---

## S2705 — Helper `build_dual_frame()`

Ajouter dans la section "helpers" de `test_pipeline.c`, après `build_frame_with_flags()` :

```c
/* Sprint 27 — Construit une trame DUAL_MODE (N=9, TASK_ID=fault_label, label=rul_u8) */
static void build_dual_frame(uint8_t fault_label, uint8_t rul_u8, uint8_t flags)
{
    uint8_t pay[64];
    int     pi = 0;

    /* MAGIC little-endian : 0xABCD → [0xCD, 0xAB] */
    pay[pi++] = 0xCDU;
    pay[pi++] = 0xABU;

    pay[pi++] = 0x03U;          /* VERSION v3 */
    pay[pi++] = fault_label;    /* TASK_ID réutilisé en DUAL_MODE : fault_label */

    /* TIMESTAMP_MS = 0 */
    pay[pi++] = 0x00U; pay[pi++] = 0x00U;
    pay[pi++] = 0x00U; pay[pi++] = 0x00U;

    pay[pi++] = 9U;             /* N = 9 features (DUAL_MODE) */

    /* Features — 9 floats à 0.0f */
    for (int i = 0; i < 9; i++) {
        pay[pi++] = 0x00U; pay[pi++] = 0x00U;
        pay[pi++] = 0x00U; pay[pi++] = 0x00U;
    }

    pay[pi++] = rul_u8;         /* label = RUL encodé uint8 */
    pay[pi++] = flags;          /* FLAGS */

    uint8_t crc = mock_crc8(pay, pi);
    memcpy(uart_rx_buf, pay, pi);
    uart_rx_buf[pi] = crc;
    uart_rx_len     = pi + 1;
    uart_rx_pos     = 0;
}
```

---

## T76 — `test_pipeline_response_dual_25bytes`

**Objectif** : vérifier que `uart_send_response_dual()` produit exactement 25 octets.

```c
void test_pipeline_response_dual_25bytes(void)
{
    uart_tx_reset();   /* reset compteur d'octets émis */
    test_pipeline_send_response_dual(
        /*pred_fault=*/2,
        /*conf_fault=*/0.75f,
        /*rul_pred=*/0.60f,
        /*f1_macro=*/0.65f,
        /*rmse_rul=*/0.08f,
        /*forgetting=*/0.01f
    );
    TEST_ASSERT_EQUAL_INT(RESPONSE_DUAL_SIZE, uart_tx_count);
    /* RESPONSE_DUAL_SIZE doit valoir 25 */
}
```

---

## T77 — `test_pipeline_dual_response_fields`

**Objectif** : vérifier que les 7 champs sont encodés aux bons offsets avec la précision float attendue.

```c
void test_pipeline_dual_response_fields(void)
{
    uart_tx_reset();
    uint8_t  expected_pred  = 3U;
    float    expected_conf  = 0.82f;
    float    expected_rul   = 0.45f;
    float    expected_f1    = 0.70f;
    float    expected_rmse  = 0.07f;
    float    expected_fgt   = 0.02f;

    test_pipeline_send_response_dual(expected_pred, expected_conf, expected_rul,
                                      expected_f1, expected_rmse, expected_fgt);

    TEST_ASSERT_EQUAL_INT(25, uart_tx_count);

    /* Offset 0 : pred_fault (u8) */
    TEST_ASSERT_EQUAL_UINT8(expected_pred, uart_tx_buf[0]);

    /* Offsets 1–4 : conf_fault (f32 little-endian) */
    float decoded_conf;
    memcpy(&decoded_conf, uart_tx_buf + 1, 4);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, expected_conf, decoded_conf);

    /* Offsets 5–8 : rul_pred (f32) */
    float decoded_rul;
    memcpy(&decoded_rul, uart_tx_buf + 5, 4);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, expected_rul, decoded_rul);

    /* Offsets 9–12 : lat_us (u32) — valeur profiling stub non testée ici */

    /* Offsets 13–16 : f1_macro (f32) */
    float decoded_f1;
    memcpy(&decoded_f1, uart_tx_buf + 13, 4);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, expected_f1, decoded_f1);

    /* Offsets 17–20 : rmse_rul (f32) */
    float decoded_rmse;
    memcpy(&decoded_rmse, uart_tx_buf + 17, 4);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, expected_rmse, decoded_rmse);

    /* Offsets 21–24 : forgetting (f32) */
    float decoded_fgt;
    memcpy(&decoded_fgt, uart_tx_buf + 21, 4);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, expected_fgt, decoded_fgt);
}
```

---

## T78 — `test_pipeline_dual_mode_dispatch`

**Objectif** : vérifier que `FLAGS=0x70` déclenche la réponse 25 B **et non** 21 B (non-régression MULTICLASS_MODE).

```c
void test_pipeline_dual_mode_dispatch(void)
{
    /* ── Test A : FLAGS=0x70 → 25 B (DUAL_MODE) ─────────────────────────── */
    pipeline_init();
    uart_tx_reset();
    build_dual_frame(/*fault_label=*/1, /*rul_u8=*/128, /*flags=*/PROTO_FLAG_DUAL_MODE);
    pipeline_run();
    TEST_ASSERT_EQUAL_INT_MESSAGE(25, uart_tx_count,
        "DUAL_MODE (0x70) doit produire 25 B, pas 21");

    /* ── Test B : FLAGS=0x30 → 21 B (MULTICLASS_MODE inchangé) ──────────── */
    pipeline_init();
    uart_tx_reset();
    /* build_frame_with_flags : trame N=9, FLAGS=0x30 */
    build_frame_with_flags(/*n=*/9, /*label=*/2, /*flags=*/PROTO_FLAG_MULTICLASS_MODE);
    pipeline_run();
    TEST_ASSERT_EQUAL_INT_MESSAGE(21, uart_tx_count,
        "MULTICLASS_MODE (0x30) ne doit pas être intercepté par DUAL_MODE");
}
```

---

## T79 — `test_pipeline_dual_mode_update`

**Objectif** : vérifier que `FLAGS=0x71` (DUAL+UPDATE) met à jour les poids des **deux modèles**.

```c
void test_pipeline_dual_mode_update(void)
{
    pipeline_init();

    /* Capturer valeurs de poids initiales */
    float w_reg_before = g_ewc_reg.w1[0][0];
    float w_mc_before  = g_ewc_mc.w1[0][0];

    /* Trame DUAL_MODE + UPDATE (FLAGS = 0x70 | 0x01 = 0x71) */
    build_dual_frame(
        /*fault_label=*/0,
        /*rul_u8=*/100,
        /*flags=*/(uint8_t)(PROTO_FLAG_DUAL_MODE | PROTO_FLAG_UPDATE)
    );
    uart_tx_reset();
    pipeline_run();

    /* Vérifier que les deux modèles ont été mis à jour */
    TEST_ASSERT_NOT_EQUAL_MESSAGE(w_reg_before, g_ewc_reg.w1[0][0],
        "ewc_reg.w1[0][0] doit changer après UPDATE en DUAL_MODE");
    TEST_ASSERT_NOT_EQUAL_MESSAGE(w_mc_before, g_ewc_mc.w1[0][0],
        "ewc_mc.w1[0][0] doit changer après UPDATE en DUAL_MODE");

    /* Réponse toujours 25 B même avec UPDATE */
    TEST_ASSERT_EQUAL_INT(25, uart_tx_count);
}
```

---

## Enregistrement dans `test_runner.c`

Ajouter les 4 nouveaux tests dans `RUN_TEST()` de `test_runner.c`, dans le groupe `test_pipeline.c` :

```c
/* Sprint 27 — DUAL_MODE tests */
RUN_TEST(test_pipeline_response_dual_25bytes);   /* T76 */
RUN_TEST(test_pipeline_dual_response_fields);    /* T77 */
RUN_TEST(test_pipeline_dual_mode_dispatch);      /* T78 */
RUN_TEST(test_pipeline_dual_mode_update);        /* T79 */
```

---

## Vérification

```bash
cd firmware/stm32f4_blink

make test
# Attendu :
# test_pipeline_response_dual_25bytes         PASS
# test_pipeline_dual_response_fields          PASS
# test_pipeline_dual_mode_dispatch            PASS
# test_pipeline_dual_mode_update              PASS
# ...
# 79 Tests 0 Failures 0 Ignored
```

---

## Checklist de non-régression

Vérifier que les tests Sprint 26 sont **tous toujours verts** après ajout du bloc DUAL_MODE :

| Test Sprint 26 critique | Attendu après Sprint 27 |
|------------------------|------------------------|
| `test_pipeline_response_v3_21bytes` | PASS (v3 inchangé) |
| `test_pipeline_multiclass_dispatch` | PASS (0x30 → 21 B) |
| `test_pipeline_rul_dispatch` | PASS (0x50 → 21 B) |
| `test_pipeline_consolidate_ewc_mc` | PASS |
| `test_ewc_regression_*` (5 tests) | PASS (ewc_head_regression inchangé) |
| `test_ewc_multiclass_*` (5 tests) | PASS (ewc_head_multiclass inchangé) |
