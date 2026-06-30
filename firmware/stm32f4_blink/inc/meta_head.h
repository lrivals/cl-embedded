#pragma once
#include <stdint.h>
#include "meta_weights.h"   /* META_N_FEATURES, META_HIDDEN, META_W[...] (générés) */

/* Méta-modèle de stacking (Sprint 31 / S3105) : META_N_FEATURES entrées → verdict binaire.
 * Arbitre les 2 sorties d'une paire Mahalanobis + supervisé (cf. pipeline.c TRIPLE_MODE).
 *
 * Deux variantes selon META_HIDDEN (défini dans meta_weights.h, généré) :
 *   - META_HIDDEN == 0 : régression logistique  → sigmoid(w·x + b)
 *   - META_HIDDEN  > 0 : MLP 1 couche cachée     → sigmoid(w2·relu(W1·x + b1) + b2)
 *
 * MEM : logreg ≈ (META_N_FEATURES+1)·4 B ; MLP ≈ META_HIDDEN·(META_N_FEATURES+2)·4 B → quelques
 * centaines d'octets. Allocation statique, pas de malloc. Toute taille via #define (règle CLAUDE.md).
 * Poids générés par scripts/export_weights_c.py — NE PAS éditer meta_weights.h à la main. */

typedef struct {
    float w[META_N_FEATURES];   /* logreg : vecteur de poids */
    float b;                    /* logreg : biais            */
#if META_HIDDEN > 0
    float w1[META_HIDDEN][META_N_FEATURES];  /* MLP : couche cachée */
    float b1[META_HIDDEN];
    float w2[META_HIDDEN];                   /* MLP : couche de sortie (1 neurone) */
    float b2;
#endif
} MetaHead;

void  meta_init(MetaHead *m);                               /* charge les poids depuis meta_weights.h */
float meta_forward(const MetaHead *m, const float *feats);  /* proba sigmoïde ∈ [0, 1] */
int   meta_predict(const MetaHead *m, const float *feats);  /* 0/1 (seuil 0.5) */
