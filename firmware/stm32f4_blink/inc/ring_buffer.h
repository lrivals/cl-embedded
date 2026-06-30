/* ring_buffer.h — Buffer circulaire générique (W, S) — Sprint 34 S3402
 *
 * Abstraction réutilisable extraite du buffer FIFO spécifique HDC (hdc.c) :
 *   - capacité W, lecture par fenêtre avec stride S
 *   - 0 malloc : `storage` est un tableau statique externe fourni par l'appelant
 *   - tailles via #define à l'instanciation (jamais en dur, règle CLAUDE.md)
 *
 * Sert à la fois HDC (stride=1, ré-entraînement FIFO) et l'étude de streaming
 * board S3403 (balayage de stride, remplissage de buffer paramétré par W).
 */

#ifndef RING_BUFFER_H
#define RING_BUFFER_H

#include <stdint.h>

#define RING_BUFFER_CAPACITY(W) (W)   /* macro de dimensionnement, pas un define global figé */

typedef struct {
    uint8_t *storage;      /* pointeur vers un tableau statique externe, pas alloué ici */
    int      elem_size;    /* sizeof(un élément), ex. sizeof(float) ou N x sizeof(float) */
    int      capacity;     /* W */
    int      head;         /* indice circulaire d'écriture (prochain slot) */
    int      count;        /* nb d'éléments valides (<= capacity) */
} RingBuffer;

/* Initialise le buffer sur un `storage` externe de `capacity * elem_size` octets. */
void ring_buffer_init(RingBuffer *rb, uint8_t *storage, int elem_size, int capacity);

/* Pousse un élément (copie `elem_size` octets) ; écrase le plus ancien si plein. */
void ring_buffer_push(RingBuffer *rb, const void *elem);

/* Copie jusqu'à `window_size` éléments dans `out_window`, du plus ancien au plus récent,
 * en sautant de `stride` éléments. Renvoie le nombre d'éléments effectivement copiés. */
int  ring_buffer_window(const RingBuffer *rb, void *out_window, int window_size, int stride);

/* Renvoie 1 si le buffer contient `capacity` éléments, 0 sinon. */
int  ring_buffer_is_full(const RingBuffer *rb);

#endif /* RING_BUFFER_H */
