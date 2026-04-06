# src/models/hdc/base_vectors.py
# ruff: noqa: N803, N806  — H_level, H_pos, H, D sont des conventions mathématiques pour matrices HDC
from pathlib import Path

import numpy as np

# Constantes globales — valeurs par défaut conformes à hdc_config.yaml
# Ne jamais modifier ici : passer par configs/hdc_config.yaml
D: int = 1024  # Dimension des hypervecteurs (puissance de 2 pour SIMD)
N_LEVELS: int = 10  # Niveaux de quantification par feature
N_FEATURES: int = 4  # Features numériques : temperature, pressure, vibration, humidity
# (équipement exclut de X — utilisé uniquement pour le split domaine)
# Référence : src/data/monitoring_dataset.py::NUMERIC_FEATURES
# TODO(arnaud) : N_FEATURES = 4 (sans one-hot équipement) ou 6 (avec) ?
# La spec §2.1 dit 6 mais le loader retourne 4. Décision architecturale à fixer avant S2-02.


def _mean_dot(H: np.ndarray) -> float:
    """Calcule la moyenne des produits scalaires normalisés entre toutes les paires de lignes."""
    n = H.shape[0]
    d = H.shape[1]
    dots = []
    for i in range(n):
        for j in range(i + 1, n):
            dot = np.dot(H[i].astype(np.float32), H[j].astype(np.float32)) / d
            dots.append(abs(dot))
    return float(np.mean(dots)) if dots else 0.0


def generate_base_hvectors(
    D: int = D,
    n_levels: int = N_LEVELS,
    n_features: int = N_FEATURES,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Génère les hypervecteurs de base pseudo-aléatoires pour HDC.

    Ces vecteurs sont FIXES (non appris). Ils doivent être sauvegardés
    et rechargés identiquement sur PC et sur MCU.

    Parameters
    ----------
    D : int
        Dimension des hypervecteurs. Doit être une puissance de 2 (SIMD).
    n_levels : int
        Nombre de niveaux de quantification par feature.
    n_features : int
        Nombre de features numériques en entrée.
    seed : int
        Graine aléatoire. CRITIQUE : tout changement invalide le modèle.

    Returns
    -------
    H_level : np.ndarray [n_levels, D], dtype=int8, valeurs ∈ {-1, +1}
        MEM: 10 × 1024 × 1 B = 10 Ko @ INT8 (Flash MCU)
    H_pos : np.ndarray [n_features, D], dtype=int8, valeurs ∈ {-1, +1}
        MEM: 4 × 1024 × 1 B = 4 Ko @ INT8 (Flash MCU)

    Notes
    -----
    Référence : Benatti2019HDC, docs/models/hdc_spec.md §3.1
    """
    rng = np.random.default_rng(seed)
    H_level = rng.choice([-1, 1], size=(n_levels, D)).astype(np.int8)
    H_pos = rng.choice([-1, 1], size=(n_features, D)).astype(np.int8)
    return H_level, H_pos


def save_base_vectors(
    H_level: np.ndarray,
    H_pos: np.ndarray,
    path: str | Path,
) -> None:
    """
    Sauvegarde les hypervecteurs de base au format .npz.

    Le fichier .npz contient les clés 'H_level' et 'H_pos'.
    La sauvegarde est idempotente : même seed → même fichier.

    Parameters
    ----------
    H_level : np.ndarray [n_levels, D], dtype=int8
    H_pos : np.ndarray [n_features, D], dtype=int8
    path : str | Path
        Chemin de destination (ex. configs/hdc_base_vectors.npz).
        Le répertoire parent est créé si nécessaire.

    Notes
    -----
    MEM (fichier disque) :
        H_level : 10 × 1024 × 1 B = 10 240 B  # MEM: 10 Ko @ INT8
        H_pos   :  4 × 1024 × 1 B =  4 096 B  # MEM:  4 Ko @ INT8
        Total   : ~14 Ko (avec overhead .npz)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, H_level=H_level, H_pos=H_pos)
    print(
        f"[HDC] Base vectors saved → {path} "
        f"(H_level={H_level.shape}, H_pos={H_pos.shape}, dtype={H_level.dtype})"
    )


def load_base_vectors(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Charge les hypervecteurs de base depuis un fichier .npz.

    Parameters
    ----------
    path : str | Path
        Chemin vers le fichier .npz (ex. configs/hdc_base_vectors.npz).

    Returns
    -------
    H_level : np.ndarray [n_levels, D], dtype=int8
    H_pos : np.ndarray [n_features, D], dtype=int8

    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas. Lancer `generate_base_hvectors` + `save_base_vectors`.
    KeyError
        Si les clés 'H_level' ou 'H_pos' sont absentes du fichier.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Base vectors not found at {path}. "
            "Run generate_base_hvectors() + save_base_vectors() first."
        )
    data = np.load(path)
    return data["H_level"], data["H_pos"]


if __name__ == "__main__":
    import yaml

    config_path = Path("configs/hdc_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    hdc_cfg = config["hdc"]
    out_path = Path(hdc_cfg["base_vectors_path"])

    # NOTE: config["data"]["n_features"] = 6 (one-hot équipement inclus) — mais N_FEATURES=4
    # est la valeur correcte pour les features réellement retournées par le DataLoader.
    # TODO(arnaud) : confirmer si l'encodage one-hot doit être réintégré en feature pour HDC.
    H_level, H_pos = generate_base_hvectors(
        D=hdc_cfg["D"],
        n_levels=hdc_cfg["n_levels"],
        n_features=N_FEATURES,  # 4, pas config["data"]["n_features"] (=6)
        seed=hdc_cfg["seed"],
    )
    save_base_vectors(H_level, H_pos, out_path)
    print(f"H_level orthogonality (mean |dot|/D): {_mean_dot(H_level):.4f} (expected ~0)")
