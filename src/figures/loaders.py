"""Chargement traçable des données de figures (S4201).

Toute donnée tracée dans une figure passe par ce module : JSON de
`experiments/` via :func:`load_experiment`, checkpoints (poids réels) déclarés
via :func:`record_source`. Chaque source est enregistrée avec chemin + mtime
(provenance), ce qui permet à la CLI d'afficher de quels fichiers chaque
catalogue provient.

Conventions d'honnêteté du dépôt (Sprints 29/33) respectées par
:func:`metric_or_na` : ``null`` + ``na_reason`` → ``None`` ; sentinel littéral
``"à mesurer"`` conservé tel quel ; **jamais** 0 par défaut.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

ROOT: Path = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR: Path = ROOT / "experiments"

#: Sentinel littéral des mesures non faites (convention Sprint 33 — LPM01A).
A_MESURER: str = "à mesurer"


@dataclass(frozen=True)
class Provenance:
    """Trace d'une source de données : chemin + date de dernière modification."""

    path: Path
    mtime: str  # ISO 8601

    def __str__(self) -> str:
        try:
            rel = self.path.relative_to(ROOT)
        except ValueError:
            rel = self.path
        return f"{rel} (modifié {self.mtime})"


# Sources chargées depuis le dernier appel à consume_session_sources() —
# permet à la CLI d'afficher la provenance par catalogue.
_SESSION_SOURCES: list[Provenance] = []


def record_source(path: str | Path) -> Provenance:
    """Enregistre une source non-JSON (checkpoint .pt/.pkl…) dans la provenance."""
    p = Path(path)
    prov = Provenance(path=p.resolve(), mtime=datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"))
    _SESSION_SOURCES.append(prov)
    return prov


def consume_session_sources() -> list[Provenance]:
    """Retourne puis vide la liste des sources chargées (usage CLI)."""
    out = list(_SESSION_SOURCES)
    _SESSION_SOURCES.clear()
    return out


def load_experiment(path: str | Path) -> tuple[dict, Provenance]:
    """Charge un JSON d'`experiments/` et retourne ``(data, provenance)``.

    Lève ``FileNotFoundError`` explicite si absent — aucune valeur par défaut
    silencieuse (règle « aucun chiffre inventé »).
    """
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(
            f"Expérience introuvable : {path} — aucune valeur par défaut n'est "
            "substituée ; générer d'abord l'expérience ou corriger le chemin."
        )
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return data, record_source(p)


def metric_or_na(data: dict, key: str) -> Any:
    """Valeur d'une métrique, ou son état « non mesuré » — jamais 0 par défaut.

    ``key`` accepte un chemin pointé (ex. ``"fp32.metric_value"``).

    Returns
    -------
    Any
        - la valeur si mesurée ;
        - la chaîne :data:`A_MESURER` si le champ porte le sentinel littéral ;
        - ``None`` si le champ est absent ou ``null`` (``na_reason`` honoré).
    """
    cur: Any = data
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    if cur is None:
        return None
    if isinstance(cur, str) and cur.strip().lower() == A_MESURER:
        return A_MESURER
    return cur


def iter_experiments(
    pattern: str, root: Path = EXPERIMENTS_DIR
) -> Iterator[tuple[dict, Provenance]]:
    """Itère ``(data, provenance)`` sur les JSON matchant ``pattern`` (glob)."""
    for p in sorted(root.glob(pattern)):
        if p.is_file():
            yield load_experiment(p)


def load_model_dataset_grid(
    pattern: str, root: Path = EXPERIMENTS_DIR
) -> dict[tuple[str, str], dict]:
    """Grille (modèle, dataset) → data pour les formes type ``exp_S28_PC_*``.

    Chaque JSON doit porter les champs ``model`` et ``dataset`` (convention
    ``results_{model}_{dataset}.json`` des Sprints 28/29/39).
    """
    grid: dict[tuple[str, str], dict] = {}
    for data, _prov in iter_experiments(pattern, root):
        model, dataset = data.get("model"), data.get("dataset")
        if model is None or dataset is None:
            continue
        grid[(str(model), str(dataset))] = data
    return grid
