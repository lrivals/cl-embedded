# S1801 — Protocole UART étendu : timestamps, séquences continues, champ task_id

| Champ | Valeur |
|-------|--------|
| **ID** | S1801 |
| **Sprint** | Sprint 18 — 25 mai – 1er juin 2026 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 5h |
| **Dépendances** | Sprint 17 (NUCLEO UART opérationnel ✅) |
| **Fichiers cibles** | `firmware/stm32f4_blink/src/pipeline.c`, `inc/pipeline.h` |
| **Statut** | ⬜ À faire |

---

## Objectif

Migrer le firmware de pipeline.c du **protocole UART v1** (trame minimale, réponse 9 B) vers le **protocole v2** (trame étendue avec VERSION, TASK_ID, TIMESTAMP_MS, FLAGS ; réponse 14 B avec RAM et throughput).

Le protocole v2 est déjà implémenté côté PC dans `scripts/sensor_stream.py`. Cette tâche ferme la boucle côté firmware.

---

## État actuel — Protocole v1 (pipeline.c)

Le `pipeline.c` actuel implémente la réception v1 :

```c
// Trame reçue v1 (pipeline.c actuel)
[MAGIC 0xABCD : 2B]
[N_FEATURES : 1B]
[features : f32×N]
[label : 1B]
[CRC8 : 1B]

// Réponse v1 = 9 B
[pred : u8][conf : f32][latency_us : u32]
```

La fonction `uart_receive_sample()` parse directement `N` après le MAGIC, sans lire VERSION ni TASK_ID.
La fonction `uart_send_response()` envoie 9 octets fixes.

---

## Cible — Protocole v2

### Trame PC → MCU (v2)

```
Offset  Champ           Type    Taille  Description
------  -----           ----    ------  -----------
0       MAGIC           u16     2 B     0xABCD little-endian
2       VERSION         u8      1 B     0x02 pour v2
3       TASK_ID         u8      1 B     Tâche CL courante (0–7)
4       TIMESTAMP_MS    u32     4 B     ms depuis début de session
8       N_FEATURES      u8      1 B     Nombre de features (≤ 16)
9       features        f32×N   N×4 B   Features normalisées
9+N×4   label           u8      1 B     Ground truth
10+N×4  FLAGS           u8      1 B     bit0=update_req, bit1=profiling_req
11+N×4  CRC8            u8      1 B     Polynomial 0x07 sur tout le payload
```

Taille totale pour N=5 : 9 + 5×4 + 3 = **32 B**.

### Réponse MCU → PC (v2) — 14 B

```c
// Offsets de la réponse v2 (little-endian)
[0]     pred_label  : u8     // prédiction (0 ou 1)
[1–4]   confidence  : f32    // score de confiance ∈ [0, 1]
[5–8]   latency_us  : u32    // latence DWT en µs
[9–10]  ram_used_b  : u16    // .bss + stack peak estimé (octets)
[11–12] throughput  : u16    // inférences/s (glissant)
[13]    status      : u8     // 0=OK, bit0=CRC_ERR, bit1=OOB, bit2=UPDATE_DONE
```

Défini dans `sensor_stream.py` :
```python
RESPONSE_V2_FMT  = "<BfIHHB"   # 14 B total
```

---

## Machine d'états de réception v2

```
WAIT_MAGIC0 ──→ WAIT_MAGIC1 ──→ READ_VERSION ──→ READ_TASK_ID
                                                        │
                                                  READ_TIMESTAMP (4 B)
                                                        │
                                                  READ_N_FEATURES
                                                        │
                                                  READ_FEATURES (N×4 B)
                                                        │
                                                  READ_LABEL
                                                        │
                                                  READ_FLAGS
                                                        │
                                                  CHECK_CRC ──(err)──→ WAIT_MAGIC0
                                                        │
                                                      OK
                                                        │
                                                  INFER + UPDATE
                                                        │
                                                  SEND_RESPONSE_V2 (14 B)
```

---

## Modifications à apporter à pipeline.c

### 1. Nouveaux champs de contexte de trame

```c
// À ajouter dans pipeline.c (variables statiques)
static uint8_t  g_recv_version;
static uint8_t  g_recv_task_id;
static uint32_t g_recv_timestamp_ms;
static uint8_t  g_recv_flags;
```

### 2. Constantes protocole v2

```c
// À ajouter dans pipeline.h
#define PROTO_VERSION_V2     0x02U
#define PROTO_FLAG_UPDATE    0x01U
#define PROTO_FLAG_PROFILING 0x02U
#define PROTO_STATUS_OK          0x00U
#define PROTO_STATUS_CRC_ERR     0x01U
#define PROTO_STATUS_OOB         0x02U
#define PROTO_STATUS_UPDATE_DONE 0x04U
```

### 3. Mise à jour de uart_receive_sample()

```c
void uart_receive_sample(float *buf)
{
    uint8_t payload[3U + 4U + 1U + PROTO_MAX_N * 4U + 2U];  /* header étendu */
    int pay_idx = 0;

resync:
    while (uart_getbyte() != PROTO_MAGIC0) {}
    if (uart_getbyte() != PROTO_MAGIC1) goto resync;

    payload[pay_idx++] = PROTO_MAGIC0;
    payload[pay_idx++] = PROTO_MAGIC1;

    g_recv_version = uart_getbyte();
    payload[pay_idx++] = g_recv_version;
    if (g_recv_version != PROTO_VERSION_V2) goto resync;   /* filtre v2 uniquement */

    g_recv_task_id = uart_getbyte();
    payload[pay_idx++] = g_recv_task_id;

    /* TIMESTAMP_MS : 4 octets little-endian */
    g_recv_timestamp_ms = 0U;
    for (int k = 0; k < 4; k++) {
        uint8_t b = uart_getbyte();
        payload[pay_idx++] = b;
        g_recv_timestamp_ms |= ((uint32_t)b << (k * 8U));
    }

    uint8_t n = uart_getbyte();
    payload[pay_idx++] = n;
    if (n > PROTO_MAX_N) goto resync;

    /* features */
    for (uint8_t i = 0; i < n; i++) {
        union { float f; uint8_t b[4]; } u;
        for (int k = 0; k < 4; k++) {
            u.b[k] = uart_getbyte();
            payload[pay_idx++] = u.b[k];
        }
        if (i < MAHA_DIM) buf[i] = u.f;
    }
    for (uint8_t i = n; i < MAHA_DIM; i++) buf[i] = 0.0f;

    uint8_t label = uart_getbyte();
    payload[pay_idx++] = label;
    g_recv_label = label;

    g_recv_flags = uart_getbyte();
    payload[pay_idx++] = g_recv_flags;

    uint8_t recv_crc = uart_getbyte();
    if (proto_crc8(payload, pay_idx) != recv_crc) goto resync;
}
```

### 4. Nouvelle fonction uart_send_response_v2()

```c
/* Réponse v2 : 14 B = [pred:u8][conf:f32][lat_us:u32][ram_b:u16][thr:u16][status:u8] */
static void uart_send_response_v2(uint8_t pred, float conf,
                                   uint32_t lat_us, uint8_t status)
{
    union { float f; uint8_t b[4]; } uc;
    uint8_t buf[PROFILING_ENCODED_SIZE];   /* 8 B : [lat_us][ram_b][thr] */

    profiling_encode(buf);

    /* pred */
    uart_send_byte(pred);

    /* conf (f32 little-endian) */
    uc.f = conf;
    uart_send_byte(uc.b[0]); uart_send_byte(uc.b[1]);
    uart_send_byte(uc.b[2]); uart_send_byte(uc.b[3]);

    /* lat_us (u32 little-endian) — depuis DWT, copié depuis profiling */
    uart_send_byte(buf[0]); uart_send_byte(buf[1]);
    uart_send_byte(buf[2]); uart_send_byte(buf[3]);

    /* ram_used_b (u16 little-endian) */
    uart_send_byte(buf[4]); uart_send_byte(buf[5]);

    /* throughput (u16 little-endian) */
    uart_send_byte(buf[6]); uart_send_byte(buf[7]);

    /* status */
    uart_send_byte(status);
}
```

### 5. Mise à jour de pipeline_run()

```c
void pipeline_run(void)
{
    float raw[MAHA_DIM];

    uart_receive_sample(raw);

    profiling_start();   /* Démarre le chrono DWT */

    normalize_zscore(raw, MAHA_DIM);
    float score   = maha_score(&g_detector, raw);
    int   anomaly = (score > g_detector.threshold) ? 1 : 0;
    led_set(anomaly ? LED_ON : LED_OFF);

    uint8_t status = PROTO_STATUS_OK;

    if ((g_recv_flags & PROTO_FLAG_UPDATE) && !anomaly) {
        maha_update(&g_detector, raw);
        status |= PROTO_STATUS_UPDATE_DONE;
    }

    profiling_stop();    /* Arrête le chrono, calcule throughput */

    float confidence = 1.0f / (1.0f + score);
    uart_send_response_v2((uint8_t)anomaly, confidence,
                           profiling_get_latency_us(), status);
}
```

---

## Critères d'acceptation

- [ ] `uart_receive_sample()` ignore les trames v1 (VERSION ≠ 0x02) et resynchronise
- [ ] `uart_send_response_v2()` envoie exactement 14 B dans le bon ordre little-endian
- [ ] `status` bit `UPDATE_DONE` levé uniquement si FLAGS demande un update ET que l'update a eu lieu
- [ ] `profiling_start()` / `profiling_stop()` encadrent uniquement l'inférence (pas la réception UART)
- [ ] `pipeline_run()` fonctionne en dry-run avec `sensor_stream.py --dry-run`

---

## Compatibilité descendante

Le protocole v2 est **incompatible avec v1**. `sensor_sim.py` (Sprint 16) envoie du v1 — il ne fonctionnera plus avec le firmware v2. Utiliser `sensor_stream.py` à la place.

---

## Questions ouvertes

- `TODO(dorra)` : NUCLEO-F439ZI ou attendre STM32N6 pour le test physique ?
- `TODO(dorra)` : La fréquence SYSCLK_HZ est-elle bien 180 MHz sur la board cible ? Confirmer via `SystemCoreClock`.
