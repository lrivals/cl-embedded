"""Catalogues de figures — l'import déclenche l'auto-enregistrement (S4201).

Ajouter un catalogue = créer un module ici avec ``@register_catalog("...")``
puis l'importer ci-dessous ; l'infra (registry, CLI) ne change pas.
"""

from __future__ import annotations

from src.figures.catalogs import quant_pedagogy  # noqa: F401  (S4203)
from src.figures.catalogs import quant_pipeline  # noqa: F401  (S4204)
from src.figures.catalogs import quant_impact  # noqa: F401  (S4205)
from src.figures.catalogs import quant_moment  # noqa: F401  (S4606)
from src.figures.catalogs import quant_ewc  # noqa: F401  (variante présentation EWC-only, sans Q15)
from src.figures.catalogs import drift_datasets  # noqa: F401  (S4304)
from src.figures.catalogs import drift_detection_pc  # noqa: F401  (S4405)
