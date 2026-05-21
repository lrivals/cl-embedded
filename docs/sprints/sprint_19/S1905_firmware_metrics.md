# S1905 — Firmware metrics : accuracy, AUROC sliding window, forgetting tracker

| Champ | Valeur |
|-------|--------|
| **Sprint** | 19 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 3h (déjà faite) |
| **Dépendances** | S1904 (mock_data) |
| **Fichiers cibles** | `firmware/stm32f4_blink/src/metrics.c`, `firmware/stm32f4_blink/inc/metrics.h` |

---

## Contexte

Pour valider les modèles CL sur board et produire les métriques obligatoires du projet (`acc_final`, `avg_forgetting`, `backward_transfer`), le firmware doit calculer ces métriques en ligne, sans buffer complet en RAM. Les métriques sont encodées dans la réponse UART (protocole v3, voir S1906) et capturées par `board_experiment_recorder.py`.

---

## Objectif

Fournir des structures C sans malloc pour calculer accuracy online, AUROC fenêtre glissante, et forgetting inter-tâches — compatibles Cortex-M4/M55.

---

## État actuel — Implémenté ✅

**`firmware/stm32f4_blink/src/metrics.c`**
**`firmware/stm32f4_blink/inc/metrics.h`**

### OnlineAccuracy

```c
typedef struct { uint32_t n_correct; uint32_t n_total; } OnlineAccuracy;
```

| Fonction | Complexité | Description |
|----------|-----------|-------------|
| `acc_init(a)` | O(1) | Remet à zéro |
| `acc_update(a, pred, true_label)` | O(1) | Incrémente n_correct si pred == label |
| `acc_compute(a)` | O(1) | Retourne n_correct / n_total |

**MEM : 8 B @ SRAM (2 × uint32)**

### OnlineAUROC (fenêtre glissante Wilcoxon-Mann-Whitney)

```c
#define AUROC_WINDOW 50U
typedef struct {
    float   scores[AUROC_WINDOW];
    uint8_t labels[AUROC_WINDOW];
    uint32_t head; uint32_t count;
} OnlineAUROC;
```

| Fonction | Complexité | Description |
|----------|-----------|-------------|
| `auroc_init(a)` | O(W) | memset |
| `auroc_update(a, score, label)` | O(1) | Buffer circulaire |
| `auroc_compute(a)` | O(W²) | WMW : paires concordantes / (n_pos × n_neg) |

**MEM : 50×4 + 50×1 + 8 = 258 B @ SRAM**

**Complexité** : O(W²) = O(2500) — acceptable pour W=50 sur Cortex-M55 à 400 MHz.

> **Choix de design** : W=50 est un compromis entre précision AUROC (besoin d'au moins 10 positifs et 10 négatifs dans la fenêtre) et mémoire. Pour un dataset CWRU avec ~20% d'anomalies, W=50 → ~10 positifs en régime stationnaire.

### ForgettingTracker

```c
#define MAX_TASKS 4U
typedef struct {
    float   peak_acc[MAX_TASKS];
    float   current_acc[MAX_TASKS];
    uint8_t seen[MAX_TASKS];
} ForgettingTracker;
```

| Fonction | Description |
|----------|-------------|
| `fgt_init(f)` | memset |
| `fgt_update(f, task_id, acc)` | Met à jour current_acc et peak_acc (max) |
| `fgt_avg_forgetting(f)` | AF = mean(peak_acc[t] - current_acc[t]) pour t vu |
| `fgt_backward_transfer(f)` | BWT ≈ -AF (proxy pour détection anomalie) |

**MEM : 4×4 + 4×4 + 4×1 = 36 B @ SRAM**

> **Note BWT** : l'implémentation BWT ≈ -AF est un proxy valide pour la détection d'anomalie (où les tâches sont non-supervisées). Elle diverge du BWT standard uniquement en classification supervisée multi-classe.

### MetricsSnapshot (encodage UART)

```c
typedef struct { float accuracy; float auroc; float forgetting; } MetricsSnapshot;
```

`metrics_encode_snapshot(s, buf)` : encode 12 octets little-endian `[acc:f32][auroc:f32][forgetting:f32]` pour envoi UART.

---

## Budget RAM total modules metrics

| Struct | Taille |
|--------|--------|
| `OnlineAccuracy` | 8 B |
| `OnlineAUROC` | 258 B |
| `ForgettingTracker` | 36 B |
| `MetricsSnapshot` | 12 B (stack temp) |
| **Total** | **~314 B** |

---

## Intégration avec les métriques Phase 1

| Métrique Phase 1 | Source firmware | Notes |
|------------------|-----------------|-------|
| `acc_final` | `acc_compute()` après dernier sample | Accuracy cumulée toutes tâches |
| `avg_forgetting` (AF) | `fgt_avg_forgetting()` | Peak minus final par tâche |
| `backward_transfer` (BWT) | `fgt_backward_transfer()` | Proxy -AF |
| `ram_peak_bytes` | DWT + `profiling.c` | Mesuré à l'exécution |
| `inference_latency_ms` | DWT cycle counter dans `pipeline.c` | lat_us / 1000 |
| `n_params` | Constant calculé offline | Encodé dans `board_experiment_recorder.py` |

---

## Fichiers cibles

| Fichier | Action |
|---------|--------|
| `firmware/stm32f4_blink/src/metrics.c` | ✅ Complet — aucune modification requise |
| `firmware/stm32f4_blink/inc/metrics.h` | ✅ Complet |
| `firmware/stm32f4_blink/src/pipeline.c` | Intégrer `OnlineAUROC`, `ForgettingTracker` (S1906) |

---

## Vérification

- [ ] Tests Unity `test_models.c` : `acc_compute` après 10 prédictions correctes → 1.0f
- [ ] `auroc_compute` sur fenêtre moitié-positive moitié-négative séparée → ≈ 1.0f
- [ ] `fgt_avg_forgetting` après drop artificiel d'accuracy → valeur positive attendue
- [ ] `metrics_encode_snapshot` → décoder 12 octets en Python : `struct.unpack('<fff', buf)` → valeurs identiques
