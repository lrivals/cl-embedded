/* drift_method.h — Interface commune des détecteurs de drift portés — Sprint 45 S4502
 *
 * Miroir C de src/models/drift/base.py::BaseDriftDetector (S4401). Chaque détecteur
 * (page_hinkley.c, ddm.c, psi.c) expose *_init / *_update(value) -> DriftMethodVerdict /
 * *_reset, avec un état en struct à backing statique (0 malloc), à parité bit-à-bit avec
 * le Python (mêmes opérations flottantes, même ordre — pas d'accumulation int cachée).
 *
 * Sélection à la COMPILATION (le nibble de flags UART est saturé — CLAUDE.md) :
 *   make EXTRA_CFLAGS="-DDRIFT_DETECT -DDRIFT_METHOD=DRIFT_PSI"
 * Un binaire = une méthode (miroir run_s29_board_extend). Build par défaut (sans
 * -DDRIFT_DETECT) strictement inchangé : les détecteurs sont liés mais inactifs.
 */

#ifndef DRIFT_METHOD_H
#define DRIFT_METHOD_H

/* Verdict à 3 niveaux — miroir de DriftVerdict (base.py). WARNING requis par DDM. */
typedef enum {
    DM_NORMAL  = 0,
    DM_WARNING = 1,
    DM_DRIFT   = 2,
} DriftMethodVerdict;

/* Identifiants de méthode (sélection build-time). -DDRIFT_METHOD=<un de ceux-ci>. */
#define DRIFT_PAGE_HINKLEY 0
#define DRIFT_DDM          1
#define DRIFT_PSI          2

/* Défaut neutre : Page-Hinkley (état O(1)). Surchargé au build. */
#ifndef DRIFT_METHOD
#define DRIFT_METHOD DRIFT_PAGE_HINKLEY
#endif

#endif /* DRIFT_METHOD_H */
