/* ddm.c — Drift Detection Method (Gama 2004) — Sprint 45 S4502
 *
 * Port C bit-à-bit de src/models/drift/ddm.py::DDM.update.
 * Même ordre d'opérations flottantes que le Python (FPU FP32).
 */

#include "drift/ddm.h"
#include <math.h>

void ddm_reset(DDM *d)
{
    d->n     = 1;
    d->p     = 1.0f;
    d->s     = 0.0f;
    d->p_min = INFINITY;
    d->s_min = INFINITY;
}

void ddm_init(DDM *d, float warning_level, float drift_level, uint32_t min_instances)
{
    d->warning_level = warning_level;
    d->drift_level   = drift_level;
    d->min_instances = min_instances;
    ddm_reset(d);
}

DriftMethodVerdict ddm_update(DDM *d, float value)
{
    float error = (value >= 0.5f) ? 1.0f : 0.0f;
    /* Moyenne incrémentale + écart-type binomial (Python : ordre identique). */
    d->p = d->p + (error - d->p) / (float)d->n;
    d->s = sqrtf(d->p * (1.0f - d->p) / (float)d->n);
    d->n += 1;

    if (d->n < d->min_instances) return DM_NORMAL;

    /* Mise à jour du minimum de (p+s). */
    if (d->p + d->s <= d->p_min + d->s_min) {
        d->p_min = d->p;
        d->s_min = d->s;
    }

    if (d->p + d->s > d->p_min + d->drift_level * d->s_min) {
        ddm_reset(d);
        return DM_DRIFT;
    }
    if (d->p + d->s > d->p_min + d->warning_level * d->s_min) {
        return DM_WARNING;
    }
    return DM_NORMAL;
}
