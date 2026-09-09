/**
 * tinyol_reference.h — Référence Python du forward TinyOL (parité C↔Python).
 * Généré par scripts/export_weights_tinyol.py — ne pas modifier à la main.
 *
 * Régénéré à chaque export de poids : les valeurs correspondent TOUJOURS aux
 * poids présents dans inc/model_weights.h.
 */

#pragma once

#define TINYOL_REF_DIM 5

static const float TINYOL_REF_INPUT[5] = {0.10000000f, 0.05000000f, 0.08000000f, -0.03000000f, 0.12000000f};
static const float TINYOL_REF_EMB[16] = {0.06414336f, 0.00000000f, 0.17732355f, 0.22379175f, 0.00000000f, 0.00000000f, 0.35520974f, 0.04974330f, 0.00000000f, 0.03074270f, 0.44409114f, 0.36780122f, 0.23670070f, 0.00000000f, 0.00000000f, 0.00000000f};
static const float TINYOL_REF_RECON[5] = {-0.04256399f, -0.00482249f, 0.13878909f, 0.04252982f, 0.16751267f};
static const float TINYOL_REF_MSE = 0.00686084f;
