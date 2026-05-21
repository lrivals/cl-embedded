# S1906 — Response protocol v3 : ajoute metrics_snapshot (acc, auroc, forgetting)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 19 |
| **Priorité** | 🔴 Critique |
| **Statut** | ⬜ À faire |
| **Durée estimée** | 3h |
| **Dépendances** | S1905 (metrics.c ✅) |
| **Fichiers cibles** | `firmware/stm32f4_blink/src/pipeline.c`, `firmware/stm32f4_blink/inc/pipeline.h` |

---

## Contexte

Le protocole UART v1 (Sprint 16, `S1603`) retourne 9 octets par sample : `[pred:u8][conf:f32][lat_us:u32]`. Pour permettre à `board_experiment_recorder.py` de collecter les métriques CL directement depuis le board (acc_final, AUROC, avg_forgetting), le firmware doit les inclure dans sa réponse.

Le protocole v3 étend la réponse avec un `MetricsSnapshot` de 12 octets supplémentaires.

---

## Objectif

Modifier `pipeline.c` pour que `uart_send_response()` émette 21 octets incluant le snapshot métriques, et mettre à jour `sensor_stream.py` pour parser les 21 octets.

---

## Protocole actuel — v1 (9 octets)

```
Réponse v1 :
  [pred:u8] [conf:f32 LE] [lat_us:u32 LE]
   1 B        4 B           4 B
              ─────────────── 9 B total ───────────────
```

Défini dans `pipeline.c:uart_send_response()` :
```c
static void uart_send_response(uint8_t pred, float conf, uint32_t lat_us)
```

---

## Protocole cible — v3 (21 octets)

```
Réponse v3 :
  [pred:u8] [conf:f32 LE] [lat_us:u32 LE] [acc:f32 LE] [auroc:f32 LE] [forgetting:f32 LE]
   1 B        4 B           4 B             4 B           4 B            4 B
              ───────────────────────────── 21 B total ──────────────────────────────────
```

> **Rétrocompatibilité** : `sensor_stream.py` doit détecter la version via la longueur de réponse ou un champ version. Option retenue : **champ longueur implicite** — le PC lit 9 B (v1) ou 21 B (v3) selon le flag `--protocol-version` passé à `sensor_stream.py`. Pas de changement côté firmware magic byte (le PC sait quelle version il a flashée).

---

## Ce qu'il faut modifier

### 1. `firmware/stm32f4_blink/src/pipeline.c`

**Ajouter les structs globaux statiques** pour les métriques :

```c
/* MEM: métriques on-board statiques — 314 B @ SRAM */
static OnlineAccuracy  g_acc;
static OnlineAUROC     g_auroc;
static ForgettingTracker g_fgt;
static uint8_t         g_current_task_id;
```

**Modifier `pipeline_init()`** — ajouter :
```c
acc_init(&g_acc);
auroc_init(&g_auroc);
fgt_init(&g_fgt);
g_current_task_id = 0U;
```

**Modifier `uart_send_response()`** — ajouter `MetricsSnapshot` :
```c
static void uart_send_response(uint8_t pred, float conf, uint32_t lat_us,
                                const MetricsSnapshot *snap)
{
    union { float f; uint8_t b[4]; } uc;

    uart_send_byte(pred);

    uc.f = conf;
    uart_send_byte(uc.b[0]); uart_send_byte(uc.b[1]);
    uart_send_byte(uc.b[2]); uart_send_byte(uc.b[3]);

    uart_send_byte((uint8_t)(lat_us));
    uart_send_byte((uint8_t)(lat_us >> 8));
    uart_send_byte((uint8_t)(lat_us >> 16));
    uart_send_byte((uint8_t)(lat_us >> 24));

    /* MetricsSnapshot — 12 B little-endian */
    uint8_t snap_buf[12];
    metrics_encode_snapshot(snap, snap_buf);
    for (int i = 0; i < 12; i++) uart_send_byte(snap_buf[i]);
}
```

**Modifier `pipeline_run()`** — mettre à jour métriques et passer snapshot :
```c
void pipeline_run(void)
{
    float raw[MAHA_DIM];
    uart_receive_sample(raw);

    uint32_t t0 = DWT_CYCCNT;
    normalize_zscore(raw, MAHA_DIM);
    float score   = maha_score(&g_detector, raw);
    int   anomaly = (score > g_detector.threshold) ? 1 : 0;
    led_set(anomaly ? LED_ON : LED_OFF);
    if (!anomaly) maha_update(&g_detector, raw);
    uint32_t lat_us = (DWT_CYCCNT - t0) / (SYSCLK_HZ / 1000000U);

    /* Mise à jour métriques */
    acc_update(&g_acc, anomaly, (int)g_recv_label);
    auroc_update(&g_auroc, score, (int)g_recv_label);
    fgt_update(&g_fgt, g_current_task_id, acc_compute(&g_acc));

    MetricsSnapshot snap = {
        .accuracy   = acc_compute(&g_acc),
        .auroc      = auroc_compute(&g_auroc),
        .forgetting = fgt_avg_forgetting(&g_fgt),
    };

    float confidence = 1.0f / (1.0f + score);
    uart_send_response((uint8_t)anomaly, confidence, lat_us, &snap);
}
```

### 2. `firmware/stm32f4_blink/inc/pipeline.h`

Ajouter l'include de `metrics.h` et exporter `pipeline_set_task()` :
```c
#include "metrics.h"

void pipeline_set_task(uint8_t task_id);  /* change g_current_task_id */
```

### 3. `scripts/sensor_stream.py`

Modifier le parser de réponse pour lire 21 B en v3 :
```python
# v3 : pred(1) + conf(4) + lat_us(4) + acc(4) + auroc(4) + forgetting(4) = 21 B
resp = ser.read(21)
pred, conf, lat_us, acc, auroc, forgetting = struct.unpack('<BfIff f', resp)
```

---

## Fichiers cibles

| Fichier | Action |
|---------|--------|
| `firmware/stm32f4_blink/src/pipeline.c` | Modifier — ajouter metrics globals + uart_send_response v3 |
| `firmware/stm32f4_blink/inc/pipeline.h` | Modifier — ajouter pipeline_set_task, include metrics.h |
| `scripts/sensor_stream.py` | Modifier — parser réponse 21 B |
| `scripts/board_experiment_recorder.py` | Vérifier — utilise sensor_stream.py en passant --protocol-version 3 |

---

## Budget RAM additionnel (v3)

| Struct | Taille |
|--------|--------|
| `OnlineAccuracy` | 8 B |
| `OnlineAUROC` | 258 B |
| `ForgettingTracker` | 36 B |
| `snap_buf[12]` (stack temp) | 12 B |
| **Delta RAM vs v1** | **+314 B** |

RAM totale pipeline après extension : ~220 B (Mahalanobis) + 314 B (métriques) = **~534 B**.

---

## Vérification

- [ ] Compilation sans warning après modification
- [ ] Test Unity : `uart_send_response` encode bien 21 B (via loopback UART host)
- [ ] `sensor_stream.py` reçoit et parse correctement les 21 B en dry-run
- [ ] `board_experiment_recorder.py --dry-run` produit un JSON avec acc, auroc, forgetting non-null

---

## Questions ouvertes

- `TODO(arnaud)` : Faut-il envoyer le `task_id` courant dans la réponse v3 pour permettre au PC de segmenter les métriques par tâche, ou le PC se base-t-il sur le compteur de samples reçus ?
