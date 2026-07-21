/* drift_methods_params.h — Paramètres calibrés des détecteurs de drift — Sprint 45 S4503 */
/* GÉNÉRÉ par scripts/export_weights_c.py --drift-methods. NE PAS ÉDITER À LA MAIN. */
/* Source : /home/leonard/Documents/ENAC/cl-embedded/experiments/exp_S45_board_psi_gas_sensor_drift/drift_methods_params.json */

#ifndef DRIFT_METHODS_PARAMS_H
#define DRIFT_METHODS_PARAMS_H

#define DRIFT_METHODS_PARAMS_PROVIDED 1

/* ── Page-Hinkley (supervisé) ─────────────────────────────────────────────── */
#define PAGE_HINKLEY_DELTA   0.00500000f
#define PAGE_HINKLEY_LAMBDA  1000000000000000019884624838656.00000000f

/* ── DDM (supervisé) ──────────────────────────────────────────────────────── */
#define DDM_WARN_SIGMA       1000000000000000019884624838656.00000000f
#define DDM_DRIFT_SIGMA      1000000000000000019884624838656.00000000f

/* ── Commun supervisés ────────────────────────────────────────────────────── */
#define DRIFT_MIN_INSTANCES  30

/* ── PSI (non-supervisé, histogramme à bacs fixes) ────────────────────────── */
#define PSI_REF_BINS         10
#define PSI_BLOCK_SIZE_PARAM 200
#define PSI_THRESHOLD_PARAM  0.20000000f
#define PSI_BIN_EDGES { 3.67168164f, 9.19942505f, 14.72716846f, 20.25491188f, 25.78265529f, 31.31039870f, 36.83814211f, 42.36588552f, 47.89362893f, 53.42137234f, 58.94911575f }
#define PSI_REF_PROBS { 0.68564856f, 0.22807019f, 0.03968940f, 0.01092897f, 0.01466783f, 0.00690251f, 0.00517689f, 0.00287605f, 0.00115043f, 0.00488928f }

#endif /* DRIFT_METHODS_PARAMS_H */
