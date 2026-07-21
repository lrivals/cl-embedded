/* page_hinkley.c — Test séquentiel de Page-Hinkley (CUSUM) — Sprint 45 S4502
 *
 * Port C bit-à-bit de src/models/drift/page_hinkley.py::PageHinkley.update.
 * Même ordre d'opérations flottantes que le Python (FPU FP32, pas d'accumulation int).
 */

#include "drift/page_hinkley.h"
#include <math.h>

void ph_reset(PageHinkley *d)
{
    d->n       = 0;
    d->mean    = 0.0f;
    d->cum_sum = 0.0f;
    d->min_sum = 0.0f;
}

void ph_init(PageHinkley *d, float delta, float lambda, uint32_t min_instances)
{
    d->delta         = delta;
    d->lambda        = lambda;
    d->min_instances = min_instances;
    ph_reset(d);
}

DriftMethodVerdict ph_update(PageHinkley *d, float value)
{
    d->n += 1;
    /* Moyenne incrémentale (Python : mean += (value - mean) / n). */
    d->mean += (value - d->mean) / (float)d->n;
    d->cum_sum += value - d->mean - d->delta;
    d->min_sum = fminf(d->min_sum, d->cum_sum);

    if (d->n < d->min_instances) return DM_NORMAL;

    if (d->cum_sum - d->min_sum > d->lambda) {
        ph_reset(d);
        return DM_DRIFT;
    }
    return DM_NORMAL;
}
