/* ddm.h — Drift Detection Method (Gama 2004) — Sprint 45 S4502
 *
 * Port C de src/models/drift/ddm.py::DDM. Détecteur supervisé à état O(1) sur le flux
 * d'erreur 1[pred != label] (S4501). Surveille le taux d'erreur p_t et son écart-type
 * binomial s_t = sqrt(p·(1−p)/n) ; WARNING à 2σ, DRIFT à 3σ du minimum de (p+s).
 *
 * Parité bit-à-bit avec le Python : mêmes opérations, même ordre, mêmes initialisations
 * (p=1, s=0, n=1, p_min=s_min=+inf).
 *
 * État = {p, s, p_min, s_min, n} → 5 scalaires.  # MEM: 20 B @ FP32
 */

#ifndef DRIFT_DDM_H
#define DRIFT_DDM_H

#include <stdint.h>
#include "drift_method.h"

typedef struct {
    float    warning_level;  /* p+s >= p_min + warning_level·s_min → WARNING (défaut 2.0) */
    float    drift_level;    /* p+s >= p_min + drift_level·s_min   → DRIFT   (défaut 3.0) */
    uint32_t min_instances;  /* échantillons min. avant tout verdict */
    float    p;              /* taux d'erreur courant  # MEM: 4 B @ FP32 */
    float    s;              /* écart-type binomial     # MEM: 4 B @ FP32 */
    float    p_min;          /* minimum mémorisé de p */
    float    s_min;          /* écart-type au minimum de (p+s) */
    uint32_t n;              /* échantillons depuis le dernier reset */
} DDM;

/* Initialise le détecteur (état à p=1,s=0,n=1,min=+inf ; seuils posés). */
void ddm_init(DDM *d, float warning_level, float drift_level, uint32_t min_instances);

/* Traite un échantillon (erreur 0/1 : value >= 0.5 → 1) et renvoie le verdict.
 * Parité bit-à-bit avec DDM.update (ddm.py). */
DriftMethodVerdict ddm_update(DDM *d, float value);

/* Réinitialise l'état interne (nouveau régime) sans toucher aux seuils. */
void ddm_reset(DDM *d);

#endif /* DRIFT_DDM_H */
