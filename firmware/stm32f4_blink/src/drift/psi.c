/* psi.c — Population Stability Index sur histogramme à bacs fixes — Sprint 45 S4502
 *
 * Port C bit-à-bit de src/models/drift/psi.py::PSI (metric="psi").
 * Même définition de bac (searchsorted side="right" − 1, queues clampées), même _EPS,
 * même ordre d'accumulation flottante que le Python (FPU FP32).
 */

#include "drift/psi.h"
#include <math.h>

void psi_reset(PSI *d)
{
    for (int i = 0; i < d->bins; i++) d->cur_counts[i] = 0;
    d->block_seen = 0;
}

void psi_init(PSI *d, int bins, int block_size, float threshold,
              const float *edges, const float *ref_probs)
{
    if (bins > PSI_BINS) bins = PSI_BINS;
    if (bins < 1)        bins = 1;
    d->bins       = bins;
    d->block_size = block_size;
    d->threshold  = threshold;
    d->last_stat  = 0.0f;
    for (int i = 0; i < bins + 1; i++) d->edges[i]     = edges[i];
    for (int i = 0; i < bins;     i++) d->ref_probs[i] = ref_probs[i];
    psi_reset(d);
}

/* searchsorted(edges, x, side="right") − 1, borné à [0, bins-1] (miroir _bin_index). */
static int psi_bin_index(const PSI *d, float x)
{
    int count = 0;   /* nombre de bornes <= x (side="right") */
    for (int i = 0; i < d->bins + 1; i++)
        if (d->edges[i] <= x) count++;
    int idx = count - 1;
    if (idx < 0)             idx = 0;
    if (idx > d->bins - 1)   idx = d->bins - 1;
    return idx;
}

DriftMethodVerdict psi_update(PSI *d, float value)
{
    d->cur_counts[psi_bin_index(d, value)] += 1;
    d->block_seen += 1;
    if (d->block_seen < d->block_size) return DM_NORMAL;

    /* Fin de bloc : cur_probs = counts/block_seen + eps ; PSI vs ref. */
    float stat = 0.0f;
    for (int i = 0; i < d->bins; i++) {
        float cur = (float)d->cur_counts[i] / (float)d->block_seen + PSI_EPS;
        stat += (cur - d->ref_probs[i]) * logf(cur / d->ref_probs[i]);
    }
    d->last_stat = stat;
    psi_reset(d);
    return (stat > d->threshold) ? DM_DRIFT : DM_NORMAL;
}
