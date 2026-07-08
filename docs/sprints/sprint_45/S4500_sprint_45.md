# Sprint 45 — Portage board des détecteurs de drift (NUCLEO-F439ZI)

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 45 |
| **Semaine** | 3 – 9 août 2026 |
| **Statut** | 📝 Doc — spec complète ; implémentation à venir |
| **Priorité globale** | 🔴 Critique — démontre quels détecteurs de drift **tiennent sur MCU** et à quel coût RAM/latence réel ; clôt le triple gap sur l'axe drift. |
| **Durée estimée totale** | ~30h (sélection/cadrage ~3h · firmware ~10h · export/parité ~7h · mesure board ~6h · notebook+tests+docs ~4h) |
| **Dépendances** | Sprint 44 ✅ (détecteurs PC + reco MCU) · `firmware/.../inc/ring_buffer.h` ✅ · `firmware/.../src/drift_detector.c` ✅ (précédent de port, S3803) · `firmware/.../src/profiling.c` ✅ (DWT + `.bss`) · `scripts/export_weights_c.py` ✅ · pattern `run_sprint38_board.py`/`board_pc_parity38.py`/`aggregate_sprint38.py` ✅ |

## Contexte et motivation

Le Sprint 44 a évalué et classé les détecteurs de drift sur PC, avec une **recommandation MCU** (S4406).
Ce sprint **porte en C** les détecteurs retenus sur la **NUCLEO-F439ZI réelle**, mesure leur **RAM `.bss`
et latence DWT** effectives, vérifie la **parité board↔PC**, et les intègre au firmware sans casser le
protocole UART. C'est l'étape qui transforme « détecteur portable en théorie » en « détecteur mesuré sur
carte » — cohérent avec la méthodologie appariée PC→board→parité du projet (Sprints 36/38).

Le seul détecteur de drift déjà porté est `drift_detector.c` (baseline fenêtre, S3803). Ce sprint ajoute
la/les famille(s) statistique(s) et test(s) retenues (Page-Hinkley, DDM/EDDM, PSI, éventuellement ADWIN).

## Décisions de cadrage (utilisateur, 7 juillet 2026)

- **Porter les détecteurs retenus MCU-viables** issus de la reco S44 (pas tous — sélection justifiée par
  état borné / latence / autonomie).
- **Mesures RAM et latence réelles** (DWT + `make size`), pas des proxies — c'est l'objet du sprint.
- **Parité board↔PC** exigée (bit-à-bit sur séquence connue, comme `drift_detector.c` S3803).
- **Ne pas toucher au protocole UART** : le nibble de flags est **saturé** (0x10..0xF0) → sélection à la
  **compilation** (`-DDRIFT_DETECT`), précédent `-DEWC_AUTO_UPDATE`/`-DMAHA_INT8`.
- **Langue** : français.

## Contraintes hardware (rappel)

- **RAM ≤ 256 Ko** (192 Ko SRAM + 64 Ko CCM) ; `.bss` mesuré par `make size` ; état des détecteurs en
  **backing statique** (0 malloc), fenêtres via `ring_buffer.h`.
- **Latence ≤ 100 ms** par update (Gap 2, DWT `profiling_start/stop`) — les détecteurs O(1)/O(W) doivent
  y tenir très largement.
- Toute taille de fenêtre/bins en `#define` surchargeable (jamais hardcodée).

## Nœud honnête : ce que le portage vérifie et ce qu'il ne change pas

Porter un détecteur ne change pas ses **métriques de détection** (délai/FAR) — elles sont établies au
S44. Le portage vérifie : (1) **parité** (même verdict que le Python sur la même séquence), (2) **coût
réel** (`.bss` + latence DWT), (3) **faisabilité** (compile, tient en RAM, pas de régression du build par
défaut). C'est un résultat *système*, distinct du résultat *algorithmique* (S44).

## Tâches

### Bloc A — Sélection & cadrage

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4501 | **Sélection des détecteurs à porter** (reco S44 : Page-Hinkley, DDM/EDDM, PSI, baseline ; ADWIN/KS/MMD si tenable) + cadrage protocole (pas de flag UART neuf) | 🔴 | `docs/sprints/sprint_45/S4501_selection_cadrage.md` | 📝 Doc |

### Bloc B — Firmware

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4502 | **Port C des détecteurs** (état statique, 0 malloc, `ring_buffer.h`), interface commune, intégration `pipeline.c` sous `-DDRIFT_DETECT`, Makefile, tests Unity de parité | 🔴 | `firmware/.../inc/drift/*.h`, `firmware/.../src/drift/*.c`, `pipeline.c`, `Makefile`, `tests/test_drift_methods.c` | 📝 Doc |

### Bloc C — Export, parité & mesure

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4503 | **Export params/seuils** (`export_weights_c.py --drift-methods` → header généré) + driver board + parité board↔PC | 🔴 | `scripts/export_weights_c.py`, `inc/drift_methods_params.h`, `scripts/run_sprint45_board.py`, `scripts/board_pc_parity45.py` | 📝 Doc |
| S4504 | **Mesure board réelle** (latence DWT P50/P99, `.bss`, grille) + agrégat | 🔴 | `scripts/aggregate_sprint45.py` → `experiments/exp_S45_summary.json`, `exp_S45_board_*` | 📝 Doc |

### Bloc D — Assemblage & clôture

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4505 | Notebook board (heatmaps PC↔board, parité) + tests + roadmap + `CLAUDE.md` + graphify | 🟡 | `notebooks/cl_eval/drift_detection_board/comparison.ipynb`, `tests/test_sprint45_board.py` | 📝 Doc |

## Ordre d'exécution recommandé

```
S4501 (sélection) ─► S4502 (firmware) ─► S4503 (export+parité) ─► S4504 (mesure board+agrégat) ─► S4505 (notebook+tests+clôture)
```

Chaîne strictement séquentielle (chaque étape dépend de la précédente), miroir du pattern Sprint 38.

## Sources de données (Sprint 44, lecture seule)

| Artefact S44 | Contenu utilisé en S45 |
|--------------|------------------------|
| `docs/context/drift_detectors.md` (reco MCU) | liste des détecteurs à porter (S4501) |
| `experiments/exp_S44_PC_*/results.json` | verdicts de référence (parité S4503) + coût proxy (comparaison à la mesure réelle S4504) |
| séquences de test S44 | vecteurs de parité C↔Python (S4502) |

## Livrables

1. `docs/sprints/sprint_45/` (ce dossier) — specs S4500–S4505.
2. `firmware/.../src/drift/*.c` + `inc/drift/*.h` + intégration `pipeline.c` + tests Unity.
3. `inc/drift_methods_params.h` (généré) + `scripts/run_sprint45_board.py` + `board_pc_parity45.py`.
4. `experiments/exp_S45_board_*` + `exp_S45_summary.json` (latence DWT, `.bss`, parité — `« à mesurer »`
   tant que non flashé).
5. `notebooks/cl_eval/drift_detection_board/comparison.ipynb` + `tests/test_sprint45_board.py`.

## Questions ouvertes

- `TODO(dorra)` : ADWIN (histogrammes exponentiels) tient-il dans le budget `.bss` avec une borne de
  buckets, ou faut-il le laisser PC-only ?
- `TODO(arnaud)` : le verdict à 3 niveaux PC (NORMAL/WARNING/DRIFT) est mappé au binaire board
  (NORMAL/DRIFT) — le `WARNING` doit-il être remonté dans la réponse UART (réinterprétation de champ,
  comme S3805) ou ignoré board ?
- `TODO(dorra)` : où brancher le détecteur dans `pipeline.c` — sur le score Maha existant (non-supervisé,
  0 coût d'acquisition supplémentaire) ou sur une feature dédiée ?

## Bilan (à compléter)

| Tâche | Statut | Temps réel | Notes |
|-------|:------:|:----------:|-------|
| S4500 | 📝 Doc | — | Overview + cadrage |
| S4501–S4505 | 📝 Doc | — | Documentés ; implémentation à venir |
