"""Catalogues de figures — l'import déclenche l'auto-enregistrement (S4201).

Ajouter un catalogue = créer un module ici avec ``@register_catalog("...")``
puis l'importer ci-dessous ; l'infra (registry, CLI) ne change pas.
"""

from __future__ import annotations

from src.figures.catalogs import quant_pedagogy  # noqa: F401  (S4203)
from src.figures.catalogs import quant_pipeline  # noqa: F401  (S4204)
from src.figures.catalogs import quant_impact  # noqa: F401  (S4205)
from src.figures.catalogs import quant_moment  # noqa: F401  (S4606)
from src.figures.catalogs import quant_depth  # noqa: F401  (S4706)
from src.figures.catalogs import quant_depth_board  # noqa: F401  (S4806)
from src.figures.catalogs import quant_ewc  # noqa: F401  (variante présentation EWC-only, sans Q15)
from src.figures.catalogs import drift_datasets  # noqa: F401  (S4304)
from src.figures.catalogs import drift_detection_pc  # noqa: F401  (S4405)
from src.figures.catalogs import ram_full  # noqa: F401  (S4905)
from src.figures.catalogs import energy_real  # noqa: F401  (S5006)
from src.figures.catalogs import manuscrit_final  # noqa: F401  (S4109)
from src.figures.catalogs import soutenance  # noqa: F401  (figures projetées le jour de la soutenance)
from src.figures.catalogs import article_ewc  # noqa: F401  (S4009 — article EWC INT8 sur MCU)
from src.figures.catalogs import seminaire_s44_s53  # noqa: F401  (présentation encadrants S44→S53)
