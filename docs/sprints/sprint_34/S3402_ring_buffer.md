# S3402 — Abstraction buffer circulaire `ring_buffer.c/.h`

| Champ | Valeur |
|-------|--------|
| **Sprint** | 34 |
| **Priorité** | 🟡 Important |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 3h |
| **Dépendances** | `firmware/stm32f4_blink/src/hdc.c` ✅ (ring buffer spécifique HDC) |
| **Fichiers cibles** | `firmware/stm32f4_blink/src/ring_buffer.c`, `firmware/stm32f4_blink/inc/ring_buffer.h` |
| **Références** | `hdc.h:23-31` (struct `buf_x`/`buf_y`/`buf_head`/`buf_count`), `hdc.c:65-71` (push `buf_head = (buf_head+1) % HDC_RETRAIN_BUF`), `hdc.c:112-126` (lecture fenêtre FIFO), `HDC_RETRAIN_BUF=50` (hdc.h:13) |

---

## Contexte

Seul un ring buffer **spécifique à HDC** existe aujourd'hui (confirmé :
`buf_x[HDC_RETRAIN_BUF][HDC_N_FEATURES]`, `buf_y[HDC_RETRAIN_BUF]`, `buf_head`, `buf_count`,
push par `(buf_head+1) % HDC_RETRAIN_BUF`). Ce sprint généralise ce pattern en une
abstraction réutilisable `(W, S)` — fenêtre `W`, stride `S` — exploitable par d'autres
modèles et par l'étude de streaming S3401/S3403, **sans changer le comportement HDC**.

---

## Spec header `ring_buffer.h`

```c
#pragma once
#include <stdint.h>

/* Buffer circulaire générique (W, S) : capacité W, lecture par fenêtre avec stride S.
 * 0 malloc — allocation statique, tailles via #define à l'instanciation.
 */

#define RING_BUFFER_CAPACITY(W) (W)   /* macro de dimensionnement, pas un define global figé */

typedef struct {
    uint8_t *storage;      /* pointeur vers un tableau statique externe, pas alloué ici */
    int      elem_size;    /* sizeof(un élément), ex. sizeof(float) ou N x sizeof(float) */
    int      capacity;     /* W */
    int      head;         /* indice circulaire d'écriture */
    int      count;        /* nb d'éléments valides (<= capacity) */
} RingBuffer;

void ring_buffer_init(RingBuffer *rb, uint8_t *storage, int elem_size, int capacity);
void ring_buffer_push(RingBuffer *rb, const void *elem);
int  ring_buffer_window(const RingBuffer *rb, void *out_window, int window_size, int stride);
int  ring_buffer_is_full(const RingBuffer *rb);
```

**Règles** :
- **0 malloc** : `storage` est toujours un tableau statique externe fourni par l'appelant
  (ex. `buf_x` de HDC redéclaré comme `uint8_t storage[HDC_RETRAIN_BUF][HDC_N_FEATURES]`
  passé à `ring_buffer_init`). Toute taille via `#define` dans le header appelant.
- **Non-régression HDC** : `hdc.c` doit pouvoir s'appuyer sur `ring_buffer_push()` /
  `ring_buffer_window()` à la place de son indexation manuelle actuelle
  (`buf_head = (buf_head+1) % HDC_RETRAIN_BUF`) **sans changer le résultat numérique** du
  ré-entraînement HDC (mêmes échantillons lus dans le même ordre FIFO).
- API push/window/stride générique — `window_size` et `stride` passés à l'appel, pas figés
  dans la struct, pour servir à la fois HDC (`stride=1` implicite aujourd'hui) et au futur
  balayage stride de S3403.

---

## Vérification

```bash
cd firmware/stm32f4_blink && make all && make test
# test_ring_buffer.c (S3409) : push/window/stride, wrap-around
# vérifier HDC non régressé : mêmes résultats test_hdc.c avant/après migration
```
