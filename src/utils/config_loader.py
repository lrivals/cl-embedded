"""
config_loader.py — Chargement et validation des configs YAML du projet.

Usage :
    from src.utils.config_loader import load_config, save_config_snapshot
    cfg = load_config("configs/ewc_config.yaml")
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    """
    Charge un fichier de configuration YAML.

    Parameters
    ----------
    path : str
        Chemin vers le fichier .yaml.

    Returns
    -------
    dict : configuration parsée.

    Raises
    ------
    FileNotFoundError : si le fichier n'existe pas.
    ValueError : si le fichier n'est pas un YAML valide.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config introuvable : {path}\n" f"Vérifier que le fichier existe dans configs/"
        )

    with open(config_path, encoding="utf-8") as f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Erreur de parsing YAML ({path}) : {e}")

    if cfg is None:
        raise ValueError(f"Fichier YAML vide : {path}")

    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Fusionne récursivement ``override`` dans ``base`` (l'enfant écrase le parent).

    Les dicts imbriqués sont fusionnés clé à clé ; toute autre valeur est remplacée.

    Parameters
    ----------
    base : dict
        Configuration parent.
    override : dict
        Configuration enfant (prioritaire).

    Returns
    -------
    dict : nouvelle config fusionnée (les entrées ne sont pas mutées en place).
    """
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config_extends(path: str) -> dict:
    """
    Charge une config YAML avec support de l'héritage via la clé ``extends:``.

    Si la config contient ``extends: <chemin>``, la base est chargée récursivement
    (elle-même peut hériter) puis fusionnée par deep-merge — les clés de l'enfant
    écrasent celles du parent. La clé ``extends`` est retirée du résultat.

    Le chemin ``extends`` est résolu relativement au répertoire du fichier courant,
    puis (à défaut) relativement au répertoire de travail.

    Parameters
    ----------
    path : str
        Chemin vers le fichier .yaml (potentiellement avec ``extends:``).

    Returns
    -------
    dict : configuration fusionnée.
    """
    cfg = load_config(path)
    parent_ref = cfg.pop("extends", None)
    if parent_ref is None:
        return cfg

    parent_path = Path(path).parent / parent_ref
    if not parent_path.exists():
        parent_path = Path(parent_ref)
    base = load_config_extends(str(parent_path))
    return _deep_merge(base, cfg)


def save_config_snapshot(cfg: dict, exp_dir: str) -> str:
    """
    Sauvegarde une copie exacte de la config dans le répertoire d'expérience.

    Permet la reproductibilité : chaque expérience contient la config exacte utilisée.

    Parameters
    ----------
    cfg : dict
        Configuration à sauvegarder.
    exp_dir : str
        Répertoire de l'expérience (ex. "experiments/exp_001_ewc_dataset2/").

    Returns
    -------
    str : chemin du fichier config_snapshot.yaml créé.
    """
    out_dir = Path(exp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = out_dir / "config_snapshot.yaml"

    # Ajouter un timestamp pour traçabilité
    cfg_with_ts = {
        "_snapshot_timestamp": datetime.now().isoformat(),
        **cfg,
    }

    with open(snapshot_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg_with_ts, f, allow_unicode=True, default_flow_style=False)

    print(f"📸 Config snapshot → {snapshot_path}")
    return str(snapshot_path)


def get_exp_dir(cfg: dict) -> Path:
    """
    Retourne le répertoire d'expérience depuis la config.

    Parameters
    ----------
    cfg : dict
        Config contenant soit cfg["evaluation"]["output_dir"]
        soit cfg["exp_id"].

    Returns
    -------
    Path : répertoire d'expérience.
    """
    if "evaluation" in cfg and "output_dir" in cfg["evaluation"]:
        return Path(cfg["evaluation"]["output_dir"]).parent

    exp_id = cfg.get("exp_id", "exp_unknown")
    return Path(f"experiments/{exp_id}/")
