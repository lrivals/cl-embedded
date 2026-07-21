/* psi.h — Population Stability Index sur histogramme à bacs fixes — Sprint 45 S4502
 *
 * Port C de src/models/drift/psi.py::PSI. Détecteur non-supervisé le plus MCU-friendly :
 * histogramme à bacs fixes calibrés à l'enrôlement (bornes + distribution de référence
 * fournies par le header généré inc/drift_methods_params.h, S4503). État O(bins),
 * indépendant de la taille de fenêtre W — argument MCU (S4501). Sur le board, branché sur
 * le scalaire maha_score (0 coût d'acquisition).
 *
 * Compte chaque échantillon dans son bac ; tous les block_size échantillons, calcule
 * PSI = Σ (cur − ref)·log(cur/ref) vs la référence figée et compare au seuil, puis reset.
 * Parité bit-à-bit avec le Python (mêmes bornes, même _EPS, même ordre).
 *
 * État = edges (bins+1) + ref_probs (bins) + cur_counts (bins).  # MEM: (3·PSI_BINS+1)·4 B
 */

#ifndef DRIFT_PSI_H
#define DRIFT_PSI_H

#include <stdint.h>
#include "drift_method.h"

/* Nombre de bacs (statique, surchargeable : make EXTRA_CFLAGS="-DPSI_BINS=…"). */
#ifndef PSI_BINS
#define PSI_BINS 10
#endif

/* Échantillons par bloc de comparaison (surchargeable). */
#ifndef PSI_BLOCK_SIZE
#define PSI_BLOCK_SIZE 200
#endif

#define PSI_EPS 1e-8f   /* miroir de _EPS (psi.py) */

typedef struct {
    int      bins;                    /* <= PSI_BINS (dim active) */
    int      block_size;              /* échantillons par bloc */
    float    threshold;               /* DRIFT si stat > threshold (psi_threshold) */
    float    edges[PSI_BINS + 1];     /* bornes fixes calibrées   # MEM: (PSI_BINS+1)·4 B */
    float    ref_probs[PSI_BINS];     /* distribution de référence + eps  # MEM: PSI_BINS·4 B */
    uint16_t cur_counts[PSI_BINS];    /* comptage du bloc courant  # MEM: PSI_BINS·2 B */
    int      block_seen;              /* échantillons vus dans le bloc courant */
    float    last_stat;              /* dernière statistique calculée (debug/parité) */
} PSI;

/* Initialise le détecteur : copie les bornes (bins+1) et ref_probs (bins) calibrées, pose
 * le seuil et vide le bloc courant. bins doit être <= PSI_BINS. */
void psi_init(PSI *d, int bins, int block_size, float threshold,
              const float *edges, const float *ref_probs);

/* Traite un échantillon scalaire (feature/score), renvoie DRIFT à la fin d'un bloc si
 * PSI > threshold, NORMAL sinon. Parité bit-à-bit avec PSI.update (psi.py). */
DriftMethodVerdict psi_update(PSI *d, float value);

/* Vide le bloc courant (cur_counts=0, block_seen=0) sans toucher aux bornes/référence. */
void psi_reset(PSI *d);

#endif /* DRIFT_PSI_H */
