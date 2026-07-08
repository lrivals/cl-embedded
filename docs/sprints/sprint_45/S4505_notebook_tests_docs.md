# S4505 — Notebook board, tests & clôture

| Champ | Valeur |
|-------|--------|
| **Sprint** | 45 |
| **Priorité** | 🟡 Moyenne — assemblage, non-régression, clôture du triple sprint drift. |
| **Statut** | 📝 Doc — spec ; implémentation à venir. |
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
