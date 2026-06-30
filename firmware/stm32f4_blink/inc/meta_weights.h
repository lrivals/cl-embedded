/**
 * meta_weights.h — Poids méta-modèle (stacking) générés automatiquement
 * Généré par scripts/export_weights_c.py — ne pas modifier à la main.
 * kind = logreg, features = ['p_maha', 'p_sup', 'disagreement', 'conf_sup']
 */

#pragma once

#define META_N_FEATURES 4
#define META_HIDDEN     0   /* 0 = logreg ; >0 = MLP 1 couche cachée */

static const float META_W[4] = {1.30799758f, 4.57682610f, 2.17184734f, 0.00356082f};
static const float META_B = -3.10590217f;
