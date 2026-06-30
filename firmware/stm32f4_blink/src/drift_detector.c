/* drift_detector.c — Détecteur FAULT/DRIFT à fenêtre glissante — Sprint 38 S3803
 *
 * Port C de src/evaluation/drift_detector.py::SlidingWindowDriftDetector.
 * Parité bit-à-bit avec le Python : même ordre d'opérations (push → FAULT
 * instantané → DRIFT sur fraction de fenêtre) et même dénominateur (count courant).
 */

#include "drift_detector.h"

void drift_init(DriftDetector *d, int window_size,
                float fault_threshold, float drift_threshold, float drift_ratio)
{
    if (window_size > DRIFT_WINDOW_MAX) window_size = DRIFT_WINDOW_MAX;
    if (window_size < 1) window_size = 1;
    d->window_size      = window_size;
    d->fault_threshold  = fault_threshold;
    d->drift_threshold  = drift_threshold;
    d->drift_ratio      = drift_ratio;
    ring_buffer_init(&d->window, (uint8_t *)d->storage, (int)sizeof(float), window_size);
}

DriftVerdict drift_update(DriftDetector *d, float score)
{
    /* (1) Append — la fenêtre écrase le plus ancien si pleine (deque maxlen). */
    ring_buffer_push(&d->window, &score);

    /* (2) FAULT : dépassement instantané (testé AVANT le DRIFT, comme le Python). */
    if (score > d->fault_threshold) return DRIFT_FAULT;

    /* (3) DRIFT : fraction des scores valides au-dessus du seuil de dérive. */
    int n = d->window.count;
    if (n > 0) {
        float buf[DRIFT_WINDOW_MAX];   /* MEM: 256 B @ FP32 (stack) */
        int got = ring_buffer_window(&d->window, buf, d->window_size, 1);
        int above = 0;
        for (int i = 0; i < got; i++)
            if (buf[i] > d->drift_threshold) above++;
        if (got > 0 && (float)above / (float)got > d->drift_ratio)
            return DRIFT_DRIFT;
    }

    return DRIFT_NORMAL;
}

void drift_reset(DriftDetector *d)
{
    d->window.head  = 0;
    d->window.count = 0;
}
