/**
 * test_ring_buffer.c — Tests unitaires Unity pour ring_buffer.c (Sprint 34 S3402)
 *
 * Vérifie l'abstraction buffer circulaire générique (W, S) :
 *   - test_ring_buffer_push_increments        : head/count après push
 *   - test_ring_buffer_count_caps_at_capacity : count plafonné à capacity
 *   - test_ring_buffer_wraps_at_capacity      : head wrap-around modulo capacity
 *   - test_ring_buffer_is_full                : is_full après capacity pushes
 *   - test_ring_buffer_window_fifo_order      : lecture du plus ancien au plus récent
 *   - test_ring_buffer_window_after_wrap      : ordre FIFO préservé après écrasement
 *   - test_ring_buffer_window_stride          : stride > 1 saute des éléments
 *   - test_ring_buffer_window_size_limit      : window_size borne le nb d'éléments copiés
 *   - test_ring_buffer_multibyte_elem         : éléments multi-octets (features+label)
 */

#include "unity.h"
#include "ring_buffer.h"
#include <string.h>

#define CAP   4
#define ELEM  1   /* éléments d'1 octet pour les tests simples */

void test_ring_buffer_push_increments(void)
{
    uint8_t storage[CAP * ELEM];
    RingBuffer rb;
    ring_buffer_init(&rb, storage, ELEM, CAP);
    TEST_ASSERT_EQUAL_INT(0, rb.head);
    TEST_ASSERT_EQUAL_INT(0, rb.count);

    uint8_t v = 42;
    ring_buffer_push(&rb, &v);
    TEST_ASSERT_EQUAL_INT(1, rb.head);
    TEST_ASSERT_EQUAL_INT(1, rb.count);
}

void test_ring_buffer_count_caps_at_capacity(void)
{
    uint8_t storage[CAP * ELEM];
    RingBuffer rb;
    ring_buffer_init(&rb, storage, ELEM, CAP);
    for (int i = 0; i < CAP + 5; i++) {
        uint8_t v = (uint8_t)i;
        ring_buffer_push(&rb, &v);
    }
    TEST_ASSERT_EQUAL_INT(CAP, rb.count);
}

void test_ring_buffer_wraps_at_capacity(void)
{
    uint8_t storage[CAP * ELEM];
    RingBuffer rb;
    ring_buffer_init(&rb, storage, ELEM, CAP);
    for (int i = 0; i < CAP; i++) {
        uint8_t v = (uint8_t)i;
        ring_buffer_push(&rb, &v);
    }
    TEST_ASSERT_EQUAL_INT(0, rb.head);   /* (CAP) % CAP == 0 */
    uint8_t v = 99;
    ring_buffer_push(&rb, &v);
    TEST_ASSERT_EQUAL_INT(1, rb.head);
}

void test_ring_buffer_is_full(void)
{
    uint8_t storage[CAP * ELEM];
    RingBuffer rb;
    ring_buffer_init(&rb, storage, ELEM, CAP);
    TEST_ASSERT_EQUAL_INT(0, ring_buffer_is_full(&rb));
    for (int i = 0; i < CAP; i++) {
        uint8_t v = (uint8_t)i;
        ring_buffer_push(&rb, &v);
    }
    TEST_ASSERT_EQUAL_INT(1, ring_buffer_is_full(&rb));
}

void test_ring_buffer_window_fifo_order(void)
{
    /* Pousse 3 éléments < capacity → window lit 0,1,2 dans l'ordre. */
    uint8_t storage[CAP * ELEM];
    RingBuffer rb;
    ring_buffer_init(&rb, storage, ELEM, CAP);
    for (int i = 0; i < 3; i++) {
        uint8_t v = (uint8_t)(10 + i);
        ring_buffer_push(&rb, &v);
    }
    uint8_t out[CAP];
    int n = ring_buffer_window(&rb, out, CAP, 1);
    TEST_ASSERT_EQUAL_INT(3, n);
    TEST_ASSERT_EQUAL_UINT8(10, out[0]);
    TEST_ASSERT_EQUAL_UINT8(11, out[1]);
    TEST_ASSERT_EQUAL_UINT8(12, out[2]);
}

void test_ring_buffer_window_after_wrap(void)
{
    /* Pousse CAP+2 éléments → les 2 plus anciens écrasés ; window lit du plus ancien. */
    uint8_t storage[CAP * ELEM];
    RingBuffer rb;
    ring_buffer_init(&rb, storage, ELEM, CAP);
    for (int i = 0; i < CAP + 2; i++) {
        uint8_t v = (uint8_t)i;   /* 0..5 ; CAP=4 → restent 2,3,4,5 */
        ring_buffer_push(&rb, &v);
    }
    uint8_t out[CAP];
    int n = ring_buffer_window(&rb, out, CAP, 1);
    TEST_ASSERT_EQUAL_INT(CAP, n);
    TEST_ASSERT_EQUAL_UINT8(2, out[0]);
    TEST_ASSERT_EQUAL_UINT8(3, out[1]);
    TEST_ASSERT_EQUAL_UINT8(4, out[2]);
    TEST_ASSERT_EQUAL_UINT8(5, out[3]);
}

void test_ring_buffer_window_stride(void)
{
    /* stride=2 sur [0,1,2,3] → lit 0 puis 2. */
    uint8_t storage[CAP * ELEM];
    RingBuffer rb;
    ring_buffer_init(&rb, storage, ELEM, CAP);
    for (int i = 0; i < CAP; i++) {
        uint8_t v = (uint8_t)i;
        ring_buffer_push(&rb, &v);
    }
    uint8_t out[CAP];
    int n = ring_buffer_window(&rb, out, CAP, 2);
    TEST_ASSERT_EQUAL_INT(2, n);
    TEST_ASSERT_EQUAL_UINT8(0, out[0]);
    TEST_ASSERT_EQUAL_UINT8(2, out[1]);
}

void test_ring_buffer_window_size_limit(void)
{
    /* window_size=2 borne la sortie à 2 même si 4 éléments présents. */
    uint8_t storage[CAP * ELEM];
    RingBuffer rb;
    ring_buffer_init(&rb, storage, ELEM, CAP);
    for (int i = 0; i < CAP; i++) {
        uint8_t v = (uint8_t)i;
        ring_buffer_push(&rb, &v);
    }
    uint8_t out[CAP];
    int n = ring_buffer_window(&rb, out, 2, 1);
    TEST_ASSERT_EQUAL_INT(2, n);
    TEST_ASSERT_EQUAL_UINT8(0, out[0]);
    TEST_ASSERT_EQUAL_UINT8(1, out[1]);
}

void test_ring_buffer_multibyte_elem(void)
{
    /* Élément 3 octets (façon features+label HDC) — copie correcte. */
    #define MB 3
    uint8_t storage[CAP * MB];
    RingBuffer rb;
    ring_buffer_init(&rb, storage, MB, CAP);
    uint8_t a[MB] = {1, 2, 3};
    uint8_t b[MB] = {4, 5, 6};
    ring_buffer_push(&rb, a);
    ring_buffer_push(&rb, b);
    uint8_t out[CAP * MB];
    int n = ring_buffer_window(&rb, out, CAP, 1);
    TEST_ASSERT_EQUAL_INT(2, n);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(a, out, MB);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(b, out + MB, MB);
    #undef MB
}
