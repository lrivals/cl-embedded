/* ring_buffer.c — Buffer circulaire générique (W, S) — Sprint 34 S3402
 *
 * Généralise l'indexation FIFO du buffer HDC (hdc.c) :
 *   push  : head = (head + 1) % capacity, count plafonné à capacity
 *   window: lecture du plus ancien au plus récent avec stride
 * 0 malloc — `storage` est fourni par l'appelant.
 */

#include "ring_buffer.h"
#include <string.h>

void ring_buffer_init(RingBuffer *rb, uint8_t *storage, int elem_size, int capacity)
{
    rb->storage   = storage;
    rb->elem_size = elem_size;
    rb->capacity  = capacity;
    rb->head      = 0;
    rb->count     = 0;
}

void ring_buffer_push(RingBuffer *rb, const void *elem)
{
    int slot = rb->head % rb->capacity;
    memcpy(rb->storage + (size_t)slot * rb->elem_size, elem, rb->elem_size);
    rb->head = (rb->head + 1) % rb->capacity;
    if (rb->count < rb->capacity) rb->count++;
}

int ring_buffer_window(const RingBuffer *rb, void *out_window,
                       int window_size, int stride)
{
    if (stride < 1) stride = 1;

    /* Plus ancien élément valide : start = (head + capacity - count) % capacity
     * (identique à hdc.c, garantit l'ordre FIFO). */
    int count = (rb->count < rb->capacity) ? rb->count : rb->capacity;
    int start = (rb->head + rb->capacity - count) % rb->capacity;

    uint8_t *out = (uint8_t *)out_window;
    int copied = 0;
    for (int k = 0; k * stride < count && copied < window_size; k++) {
        int idx = (start + k * stride) % rb->capacity;
        memcpy(out + (size_t)copied * rb->elem_size,
               rb->storage + (size_t)idx * rb->elem_size,
               rb->elem_size);
        copied++;
    }
    return copied;
}

int ring_buffer_is_full(const RingBuffer *rb)
{
    return rb->count >= rb->capacity;
}
