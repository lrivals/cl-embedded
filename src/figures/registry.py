"""Registre des catalogues de figures (S4201).

Un catalogue = une fonction ``build(out_root: Path) -> list[Path]`` qui génère
ses figures sous ``out_root/<nom-du-catalogue>/`` (via ``style.savefig_png``)
et retourne les chemins produits. Zéro classe imposée : le décorateur
:func:`register_catalog` suffit, l'import du module (``catalogs/__init__.py``)
déclenche l'auto-enregistrement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

CatalogFn = Callable[[Path], "list[Path]"]

_CATALOGS: dict[str, CatalogFn] = {}


def register_catalog(name: str) -> Callable[[CatalogFn], CatalogFn]:
    """Décorateur : enregistre ``fn`` comme catalogue ``name`` (ex. ``quantization/pedagogy``)."""

    def deco(fn: CatalogFn) -> CatalogFn:
        if name in _CATALOGS:
            raise ValueError(f"Catalogue déjà enregistré : {name!r}")
        _CATALOGS[name] = fn
        return fn

    return deco


def get_catalog(name: str) -> CatalogFn:
    """Retourne la fonction de build du catalogue ``name`` (KeyError explicite)."""
    if name not in _CATALOGS:
        raise KeyError(
            f"Catalogue inconnu : {name!r} — disponibles : {list_catalogs()}"
        )
    return _CATALOGS[name]


def list_catalogs() -> list[str]:
    """Noms des catalogues enregistrés, triés."""
    return sorted(_CATALOGS)


def match_catalogs(prefix: str) -> list[str]:
    """Catalogues dont le nom égale ``prefix`` ou commence par ``prefix + "/"``."""
    return [
        n for n in list_catalogs() if n == prefix or n.startswith(prefix.rstrip("/") + "/")
    ]
