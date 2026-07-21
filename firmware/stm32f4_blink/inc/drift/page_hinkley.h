/* page_hinkley.h — Test séquentiel de Page-Hinkley (CUSUM) — Sprint 45 S4502
 *
 * Port C de src/models/drift/page_hinkley.py::PageHinkley. Détecteur supervisé à état
 * O(1) sur le flux d'erreur 1[pred != label] (S4501). Cumule l'écart à la moyenne
 * courante moins une tolérance delta ; DRIFT quand le cumul décroche de son minimum de
 * plus de lambda (puis reset). Ne produit pas de WARNING (test binaire).
 *
 * Parité bit-à-bit avec le Python : mêmes opérations, même ordre (moyenne incrémentale
 * mean += (x-mean)/n ; cum += x-mean-delta ; min = fminf(min,cum)).
 *
 * État = {mean, cum_sum, min_sum, n} → 4 scalaires.  # MEM: 16 B @ FP32
 */

#ifndef DRIFT_PAGE_HINKLEY_H
#define DRIFT_PAGE_HINKLEY_H

#include <stdint.h>
#include "drift_method.h"

typedef struct {
    float    delta;          /* tolérance (magnitude de changement ignorée) */
    float    lambda;         /* seuil de détection sur le cumul */
    uint32_t min_instances;  /* échantillons min. avant tout verdict */
    float    mean;           /* moyenne courante x̄_t          # MEM: 4 B @ FP32 */
    float    cum_sum;        /* cumul m_T = Σ(x_t − x̄_t − δ)   # MEM: 4 B @ FP32 */
    float    min_sum;        /* minimum de m_T                 # MEM: 4 B @ FP32 */
    uint32_t n;              /* échantillons depuis le dernier reset */
} PageHinkley;

/* Initialise le détecteur (état à zéro, seuils posés). */
void ph_init(PageHinkley *d, float delta, float lambda, uint32_t min_instances);

/* Traite un échantillon (erreur 0/1 ou feature scalaire) et renvoie le verdict.
 * Parité bit-à-bit avec PageHinkley.update (page_hinkley.py). */
DriftMethodVerdict ph_update(PageHinkley *d, float value);

/* Réinitialise l'état interne (mean/cum/min/n) sans toucher aux seuils. */
void ph_reset(PageHinkley *d);

#endif /* DRIFT_PAGE_HINKLEY_H */
