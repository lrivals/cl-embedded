# S4505 — Notebook board, tests & clôture

| Champ | Valeur |
|-------|--------|
| **Sprint** | 45 |
| **Priorité** | 🟡 Moyenne — assemblage, non-régression, clôture du triple sprint drift. |
| **Statut** | ✅ Implémenté — notebook PC↔board + tests summary + clôture (roadmap/triple_gap/CLAUDE.md). |
| **Durée estimée** | 4h |
| **Dépendances** | S4503 ✅ (parité) · S4504 ✅ (agrégat) · `pytest`, `nbconvert` · Unity firmware `make test` |
| **Fichiers cibles** | `notebooks/cl_eval/drift_detection_board/comparison.ipynb`, `tests/test_sprint45_board.py`, `docs/roadmap_phase2.md`, `docs/triple_gap.md`, `CLAUDE.md` |
| **Références** | Pattern de clôture Sprint 38 S3809 / Sprint 36 S3607 (notebook PC↔board) |

---

## Contexte

Clôture : notebook comparant PC ↔ board (parité, coût réel vs proxy), tests garantissant la parité et
Gap 2/3, puis mise à jour roadmap + `triple_gap.md` + statut + graphe. Message final : **quels détecteurs
de drift tiennent sur MCU, à quel coût mesuré, et lesquels rester PC-only**.

## Spec

### 1. Notebook — `notebooks/cl_eval/drift_detection_board/comparison.ipynb`

Charge `exp_S45_summary.json` (S4504) + `exp_S45_parity_*` (S4503) :
- **Heatmaps détecteur × dataset** : latence board, `.bss`, parité — symétriques au PC (S4405).
- **Proxy-PC ↔ mesuré-board** : figure d'écart (latence/état) — honnête sur ce que le proxy prédit mal.
- **Parité board↔PC** : par cellule (attendu 1.000 déterministe).
- **Synthèse de portabilité** : verdict final par détecteur (portable/coûteux/PC-only) avec le chiffre
  mesuré à l'appui. Exécutable nbconvert (cellules « à mesurer » en gris si non flashé).

### 2. Tests — `tests/test_sprint45_board.py`

- Structure `exp_S45_summary.json` (`[detector][dataset][platform]`, clés attendues).
- `verdict_parity == 1.0` pour les cellules mesurées déterministes.
- **Gap 2** : toutes latences board < 100 ms.
- **Gap 3** : `.bss` dans le budget ; delta build défaut = 0 (0 régression).
- 0 chiffre en dur (tout depuis JSON). Skip honnête si `exp_S45_*` absent (non flashé).
- Unity firmware : `make test` → `test_drift_methods` PASS + 0 régression (build défaut invariant).

### 3. Clôture

- `docs/roadmap_phase2.md` : bloc Sprint 45 + ligne de statut.
- `docs/triple_gap.md` : enrichir § Gap 2 (latence détecteurs de drift board) + § Gap 3 (RAM détecteurs).
- `CLAUDE.md` : Sprint 45 dans la ligne de statut sprint.
- `graphify_sprint_update` (skill).
- Si dernière tâche OK → proposer message de commit.
- **Pointeur vers la suite** : renvoyer à `docs/context/drift_fault_tandem.md` (sprint futur : drift +
  faute en tandem, autonome sur carte).

## Contraintes

- Notebook dans `notebooks/` ; aucune donnée brute committée.
- Tout chiffre tracé à `exp_S45_*` ; distinction mesuré-board / proxy-PC maintenue.

## Vérification

```bash
pytest tests/test_sprint45_board.py -v
cd firmware/stm32f4_blink && make test          # test_drift_methods PASS, 0 régression
jupyter nbconvert --to notebook --execute notebooks/cl_eval/drift_detection_board/comparison.ipynb
```
- Tests PASS (ou skip honnête si non flashé) ; `make test` sans régression.
- roadmap + `triple_gap.md` + `CLAUDE.md` reflètent Sprint 45 ; renvoi au doc tandem présent.

---

## Résolution (implémentée)

**Fichiers** : `notebooks/cl_eval/drift_detection_board/comparison.ipynb` (13 cellules, nbconvert
OK) ; `tests/test_sprint45_board.py` étendu (+5 tests summary) ; clôture
`roadmap_phase2.md` / `triple_gap.md` / `CLAUDE.md`.

**Notebook** (symétrique au PC S44, tout chargé depuis `exp_S45_summary.json` + `exp_S45_parity_*`,
0 chiffre en dur) : (1) tableau de synthèse board↔proxy-PC ; (2) heatmaps détecteur × dataset
(latence board, `.bss`, parité — **gris** pour non mesuré/N/A) ; (3) barres latence mesuré-board
vs proxy-PC en échelle log (paradoxe FPU S29) ; (4) parité par cellule mesurée ; (5) synthèse de
portabilité (portable/coûteux/PC-only) + rappel ADWIN/KS/KSWIN/MMD PC-only (S4501) ; renvoi au
doc tandem.

**Tests ajoutés** (`test_summary_*`, skip honnête si summary absent) : structure
`[dataset][detector][platform]` + `pc_proxy.is_proxy` (proxy jamais confondu avec board) ; Gap 2
(p99 < 100 ms sur cellules mesurées) ; Gap 3 (`bss_bytes` < 256 Ko + `bss_default = 105 036 B`
invariant + deltas méthode documentés) ; parité déterministe = 1.000 ; **0 chiffre en dur**
(rechargement croisé des JSON board). `pytest tests/test_sprint45_board.py` → **15 PASS + 1 skip**
(cellules non mesurées).

**Firmware** : `make test` → **134 tests, 2 échecs = TinyOL préexistants hors périmètre**
(`test_tinyol_predict_normal_zero_weights`, `test_tinyol_forward_delta`), **tous les tests drift
PASS** (`test_drift_methods` ph/ddm/psi + `test_drift_detector` 6/6), **`.bss` défaut invariant
105 036 B** → **0 régression**.

**Résultats mesurés (colonne `gas_sensor_drift`, board réelle)** : Page-Hinkley + DDM parité
board↔PC **1.000**, latence DWT **270 µs ≪ 100 ms (Gap 2 ✅)**, `.bss` ≈ 166 Ko (Gap 3 ✅) ; **PSI
N/A honnête** (overflow SRAM au link — signal Mahalanobis O(k²) à k=128 features, cf. S4504).

**Message de clôture du triple sprint drift (S43→S45)** : **Page-Hinkley/DDM (O(1)) sont portables
sur MCU** avec parité exacte et coût mesuré négligeable ; **PSI est portable en basse dimension
seulement** (sa source de signal Mahalanobis déborde en haute dim) ; **ADWIN/KS/KSWIN/MMD restent
PC-only**. **Suite** : détection drift + faute *en tandem* autonome sur carte →
[`docs/context/drift_fault_tandem.md`](../../context/drift_fault_tandem.md).
