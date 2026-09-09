# S4905 — Fixes plots RAM (étiquettes réseau de neurones + instants de mesure du pic)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 49 |
| **Priorité** | 🟡 Moyen — lisibilité et cohérence des figures (CR §2 « corrections à apporter aux plots »). |
| **Statut** | ✅ Implémenté — catalogue `ram_full` (4 PNG, phases réelles, 0 chiffre en dur) + MAJ `presentation_ram_measurement.md`. |
| **Durée estimée** | 3h |
| **Dépendances** | S4903/S4904 (`exp_S49_ram/`) · registre figures `src/figures/` ✅ (S4201) · `docs/presentation_ram_measurement.md` ✅ |
| **Fichiers cibles** | `src/figures/catalogs/ram_full.py` (nouveau) · `docs/figures/ram_full/` · MAJ `docs/presentation_ram_measurement.md` |
| **Références** | CR §2 |

## Contexte

Le CR relève deux défauts dans les plots RAM actuels : (1) des **étiquettes d'étapes attribuées à des
architectures de réseaux de neurones**, incohérentes avec les modèles réellement utilisés (HDC, Mahalanobis ne
sont pas des NN) ; (2) les **instants de mesure du pic de pile** ne collent pas aux phases réelles d'exécution.
Cette tâche produit un catalogue de figures corrigé (registre S4201) et met à jour la doc RAM.

## Spec

### 1. Corrections demandées

| Défaut CR | Correction |
|-----------|------------|
| Étiquettes d'étapes « réseau de neurones » (forward/backward de couches) | Remplacer par les **phases réelles des modèles** : `idle` / `inférence` / `mise à jour CL`. Neutres, valables pour EWC/HDC/TinyOL/Maha. |
| Instants de mesure du pic mal placés | Aligner sur les phases (S4901) : pic mesuré **après** chaque phase, tracé chronologiquement. |
| « RAM modulable » dans légendes | → « mémoire non constante » (coordonné S4906). |

### 2. Figures du catalogue `ram_full` (registre `@register_catalog`)

| Fig | Contenu | Source |
|-----|---------|--------|
| R1 | RAM totale empilée `.data`/`.bss`/pic par modèle × encodage | `summary.json` |
| R2 | **Historique du pic de pile** (phase par phase, une courbe par modèle) | `stack_history` |
| R3 | Ratio int8 vs fp32 de la RAM totale | `summary.json` |
| R4 | Board vs PC (colonnes séparées, non fusionnées, badge plateforme) | `summary.json` |

Toutes les valeurs **rechargées via `load_experiment`** (0 chiffre en dur, garde AST comme S42/S44/S47).
N/A en gris. Étiquettes = phases réelles, pas d'étapes NN.

### 3. MAJ `presentation_ram_measurement.md`

Corriger les figures/légendes du doc existant : phases réelles, « mémoire non constante », instants du pic.

## Format de sortie

- `src/figures/catalogs/ram_full.py` → PNG dans `docs/figures/ram_full/` (R1–R4).
- MAJ des sections figures de `docs/presentation_ram_measurement.md`.

## Contraintes

- 0 chiffre en dur (garde AST) ; valeurs via `load_experiment`.
- Étiquettes de phase **neutres** (pas d'architecture NN).
- Board/PC jamais fusionnés (badge plateforme).

## Vérification

```bash
python scripts/generate_figures.py --catalog ram_full
ls docs/figures/ram_full/*.png
python -m pytest tests/test_figures_library.py -q            # garde AST 0-chiffre
grep -i "réseau de neurones\|forward\|backward" src/figures/catalogs/ram_full.py   # doit être vide (pas d'étiquette NN)
grep -ci "mémoire non constante" docs/presentation_ram_measurement.md              # > 0
```
