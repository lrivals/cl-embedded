/* drift_detector.h — Détecteur FAULT/DRIFT à fenêtre glissante — Sprint 38 S3803
 *
 * Port C de src/evaluation/drift_detector.py::SlidingWindowDriftDetector.
 * Discrimine, à partir du score brut d'un détecteur non supervisé (Mahalanobis) :
 *   FAULT  : dépassement instantané      score_t > fault_threshold
 *   DRIFT  : fraction(window > drift_threshold) > drift_ratio
 *   NORMAL : sinon
 * Priorité FAULT > DRIFT > NORMAL (identique au Python : FAULT testé en premier).
 *
 * État = fenêtre glissante de `window_size` scores (RingBuffer, S3402), 0 malloc.
 * Pour W=50 → 50 × 4 B = 200 B @ FP32.  # MEM: 200 B @ FP32
 *
 * Gate de mise à jour autonome (Sprint 38) : le verdict pilote ewc_sgd_step /
 * maha_update à la place du flag UART PROTO_FLAG_UPDATE (cf. pipeline.c sous
 * -DEWC_AUTO_UPDATE). Seuils calibrés sur l'enrôlement healthy (P95 × mult.) et
 * fournis par inc/drift_thresholds.h (généré par export_weights_c.py).
 */

#ifndef DRIFT_DETECTOR_H
#define DRIFT_DETECTOR_H

#include "ring_buffer.h"

/* Capacité max de la fenêtre (statique). Surchargeable au build : make DRIFT_WINDOW_MAX=… */
#ifndef DRIFT_WINDOW_MAX
#define DRIFT_WINDOW_MAX 64   /* couvre window_size=50 par défaut (config S3801) */
#endif

typedef enum {
    DRIFT_NORMAL = 0,
    DRIFT_FAULT  = 1,
    DRIFT_DRIFT  = 2,
} DriftVerdict;

typedef struct {
    RingBuffer window;                          /* fenêtre glissante des scores */
    float      storage[DRIFT_WINDOW_MAX];       /* MEM: 256 B @ FP32 — backing statique */
    float      fault_threshold;                 /* P95_healthy × fault_multiplier */
    float      drift_threshold;                 /* P95_healthy × drift_multiplier */
    float      drift_ratio;                     /* seuil de fraction pour DRIFT */
    int        window_size;                     /* <= DRIFT_WINDOW_MAX */
} DriftDetector;

/* Initialise le détecteur (vide la fenêtre, pose les seuils). window_size est
 * borné à DRIFT_WINDOW_MAX. */
void drift_init(DriftDetector *d, int window_size,
                float fault_threshold, float drift_threshold, float drift_ratio);

/* Traite un nouveau score et renvoie le verdict courant (parité bit-à-bit avec
 * le Python : push → test FAULT instantané → test DRIFT sur la fenêtre). */
DriftVerdict drift_update(DriftDetector *d, float score);

/* Vide la fenêtre glissante (nouvelle machine / nouveau contexte). */
void drift_reset(DriftDetector *d);

#endif /* DRIFT_DETECTOR_H */
