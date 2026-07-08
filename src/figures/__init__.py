"""Bibliothèque de figures de présentation/manuscrit (Sprint 42, S4201).

Infra pérenne : style commun (`style.py`), chargement traçable des expériences
(`loaders.py`), registre de catalogues (`registry.py`). Les figures elles-mêmes
vivent dans `catalogs/` et se régénèrent via ``scripts/generate_figures.py``.

Règle d'honnêteté du sprint : **aucune valeur numérique de résultat en dur** —
tout chiffre affiché provient d'un JSON de `experiments/` ou d'un checkpoint,
chargé via `loaders` (provenance tracée).
"""

from __future__ import annotations
