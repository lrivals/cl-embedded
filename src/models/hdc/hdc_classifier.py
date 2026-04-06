# src/models/hdc/hdc_classifier.py
# ruff: noqa: N803, N806  — H_level, H_pos, H_sum, H_obs sont des conventions mathématiques HDC
"""
hdc_classifier.py — Classifieur Hyperdimensional Computing pour la maintenance prédictive.

Apprentissage incrémental sans gradient : accumulation additive de prototypes.
Pas d'oubli catastrophique par construction (mémoire additive INT32).

Référence : Benatti2019HDC, docs/models/hdc_spec.md
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.models.base_cl_model import BaseCLModel
from src.models.hdc.base_vectors import (
    generate_base_hvectors,
    load_base_vectors,
    save_base_vectors,
)

# Constantes — valeurs par défaut conformes à hdc_config.yaml
# Ne jamais modifier ici : passer par configs/hdc_config.yaml
D: int = 1024  # Dimension des hypervecteurs (puissance de 2 pour SIMD)
N_LEVELS: int = 10  # Niveaux de quantification par feature
N_FEATURES: int = 4  # Features : temperature, pressure, vibration, humidity
N_CLASSES: int = 2  # Classes : 0=normal, 1=faulty


# ---------------------------------------------------------------------------
# Fonctions d'encodage stateless — réutilisables telles quelles sur MCU
# ---------------------------------------------------------------------------


def quantize_feature(
    value: float,
    feature_min: float,
    feature_max: float,
    n_levels: int,
) -> int:
    """
    Mappe une feature continue dans [feature_min, feature_max] vers un indice ∈ [0, n_levels-1].

    Quantification linéaire uniforme. Clip aux bornes pour les valeurs hors plage.

    Parameters
    ----------
    value : float
        Valeur de la feature (normalisée Z-score).
    feature_min, feature_max : float
        Bornes observées sur Task 1 (chargées depuis hdc_config.yaml → feature_bounds).
    n_levels : int
        Nombre de niveaux de quantification.

    Returns
    -------
    int
        Indice de niveau ∈ [0, n_levels - 1].

    Notes
    -----
    Sur MCU : calcul en FP32 une fois par feature. Résultat stocké en uint8.
    Référence : docs/models/hdc_spec.md §3.2
    """
    normalized = (value - feature_min) / (feature_max - feature_min + 1e-8)
    level_idx = int(round(normalized * (n_levels - 1)))
    return int(np.clip(level_idx, 0, n_levels - 1))


def encode_observation(
    x: np.ndarray,
    H_level: np.ndarray,
    H_pos: np.ndarray,
    feature_bounds: list[tuple[float, float]],
    n_levels: int = N_LEVELS,
    D: int = D,
) -> np.ndarray:
    """
    Encode un vecteur de features en hypervecteur d'observation binarisé.

    Pipeline :
    1. Pour chaque feature i : quantifier → indice de niveau l_i
    2. H_feature_i = H_level[l_i] ⊗ H_pos[i]  (produit Hadamard, XOR sur MCU)
    3. H_sum = Σ H_feature_i  (sommation entière)
    4. H_obs_bin = sign(H_sum)  (binarisation)

    Parameters
    ----------
    x : np.ndarray [n_features]
        Vecteur de features normalisé (Z-score, float32).
    H_level : np.ndarray [n_levels, D], dtype=int8
    H_pos : np.ndarray [n_features, D], dtype=int8
    feature_bounds : list[tuple[float, float]]
        [(min_0, max_0), ..., (min_{n-1}, max_{n-1})] — calculées sur Task 1.
    n_levels : int
    D : int

    Returns
    -------
    np.ndarray [D], dtype=int8, valeurs ∈ {-1, +1}

    Notes
    -----
    Référence : docs/models/hdc_spec.md §3.3
    """
    H_sum = np.zeros(D, dtype=np.int32)  # MEM: 1024 × 4 B = 4 Ko @ INT32 (temporaire)

    for i, (feat_val, (f_min, f_max)) in enumerate(zip(x, feature_bounds)):
        level_idx = quantize_feature(float(feat_val), f_min, f_max, n_levels)
        H_feature = H_level[level_idx] * H_pos[i]  # Hadamard (XOR sur MCU)
        H_sum += H_feature.astype(np.int32)

    H_obs_bin = np.sign(H_sum).astype(np.int8)
    H_obs_bin[H_obs_bin == 0] = 1  # cas dégénéré (parité exacte)
    return H_obs_bin


# ---------------------------------------------------------------------------
# Classe principale
# ---------------------------------------------------------------------------


class HDCClassifier(BaseCLModel):
    """
    Classifieur HDC (Hyperdimensional Computing) pour la maintenance prédictive.

    Apprentissage incrémental sans gradient : accumulation additive de prototypes.
    Pas d'oubli catastrophique par construction (mémoire additive).

    Architecture :
        - Encodage : quantize → Hadamard → sommation → binarisation
        - Prototypes : C_c = Σ H_obs pour chaque obs de classe c (INT32 accumulateurs)
        - Inférence : ŷ = argmax_c cosine_similarity(H_obs, C_c)

    Budget mémoire :
        - prototypes_acc [2, 1024] INT32 : 2 × 1024 × 4 B = 8 Ko   # MEM: 8 Ko @ INT32
        - prototypes_bin [2, 1024] INT8  : 2 × 1024 × 1 B = 2 Ko   # MEM: 2 Ko @ INT8
        - buffer encodage [1024]   INT32 : 1 × 1024 × 4 B = 4 Ko   # MEM: 4 Ko @ INT32 (temporaire)
        - TOTAL RAM : < 14 Ko (cible STM32N6 : ≤ 64 Ko)

    Parameters
    ----------
    config : dict
        Configuration chargée depuis configs/hdc_config.yaml.
        La clé ``feature_bounds`` doit être au niveau racine du dict.

    References
    ----------
    Benatti2019HDC, docs/models/hdc_spec.md
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)

        hdc_cfg = config["hdc"]
        self.D: int = hdc_cfg["D"]  # 1024
        self.n_levels: int = hdc_cfg["n_levels"]  # 10
        self.n_classes: int = config["data"]["n_classes"]  # 2
        self.n_features: int = config["data"]["n_features"]  # 4

        # Feature bounds pour quantification — calculées sur Task 1
        # Format : [(min_0, max_0), ..., (min_3, max_3)]
        self.feature_bounds: list[tuple[float, float]] = self._load_feature_bounds(config)

        # Hypervecteurs de base (depuis .npz ou génération on-the-fly)
        bv_path = Path(hdc_cfg["base_vectors_path"])
        if bv_path.exists():
            self.H_level, self.H_pos = load_base_vectors(bv_path)
        else:
            self.H_level, self.H_pos = generate_base_hvectors(
                D=self.D,
                n_levels=self.n_levels,
                n_features=self.n_features,
                seed=hdc_cfg["seed"],
            )
            save_base_vectors(self.H_level, self.H_pos, bv_path)

        # Prototypes de classe (accumulateurs INT32 + version binarisée pour l'inférence)
        self.prototypes_acc = np.zeros((self.n_classes, self.D), dtype=np.int32)
        # MEM: 2 × 1024 × 4 B = 8 Ko @ INT32
        self.prototypes_bin = np.zeros((self.n_classes, self.D), dtype=np.int8)
        # MEM: 2 × 1024 × 1 B = 2 Ko @ INT8
        self.class_counts = np.zeros(self.n_classes, dtype=np.int32)

        self._fitted: bool = False  # True après au moins 1 appel à update()

    # ------------------------------------------------------------------
    # Interface BaseCLModel — méthodes abstraites
    # ------------------------------------------------------------------

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Inférence par similarité cosinus entre H_obs et les prototypes binarisés.

        Sur MCU : calcul via POPCOUNT(XOR(H_obs, C_c)) pour chaque classe c.

        Parameters
        ----------
        x : np.ndarray [batch_size, n_features], dtype=float32
            Observations normalisées.

        Returns
        -------
        np.ndarray [batch_size], dtype=int64
            Classe prédite ∈ {0, 1} pour chaque observation.

        Raises
        ------
        RuntimeError
            Si appelé avant tout appel à update().
        """
        if not self._fitted:
            raise RuntimeError("HDCClassifier not fitted. Call update() first.")

        preds = []
        for sample in x:
            H_obs = encode_observation(
                sample,
                self.H_level,
                self.H_pos,
                self.feature_bounds,
                self.n_levels,
                self.D,
            )
            # Similarité cosinus avec chaque prototype binarisé
            # Sur MCU : dot product INT8 (équivalent à count_agreements - count_disagreements)
            similarities = self.prototypes_bin.astype(np.float32) @ H_obs.astype(np.float32)
            preds.append(int(np.argmax(similarities)))
        return np.array(preds, dtype=np.int64)

    def update(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Mise à jour incrémentale : accumulation des hypervecteurs dans les prototypes.

        Complexité : O(batch_size × N_FEATURES × D) en temps, O(1) en mémoire additionnelle.
        Pas de gradient, pas de catastrophic forgetting par construction.

        Parameters
        ----------
        x : np.ndarray [batch_size, n_features], dtype=float32
        y : np.ndarray [batch_size], dtype=int

        Returns
        -------
        float
            Taux d'erreur sur ce batch (proxy de loss pour compatibilité BaseCLModel).
        """
        for sample, label in zip(x, y):
            H_obs = encode_observation(
                sample,
                self.H_level,
                self.H_pos,
                self.feature_bounds,
                self.n_levels,
                self.D,
            )
            self.prototypes_acc[int(label)] += H_obs.astype(np.int32)
            self.class_counts[int(label)] += 1

        self._rebinarize_prototypes()
        self._fitted = True

        preds = self.predict(x)
        errors = int(np.sum(preds != y.astype(np.int64)))
        return errors / len(y)

    def on_task_end(self, task_id: int, dataloader: Any) -> None:
        """
        Callback fin de tâche. Re-binarise les prototypes pour cohérence.

        HDC n'a pas de post-processing obligatoire. Optionnel : renormalisation
        des accumulateurs INT32 pour éviter le débordement sur très longues séquences
        (hors scope Sprint 2 — TODO(dorra)).

        Parameters
        ----------
        task_id : int
        dataloader : iterable (non utilisé par HDC)
        """
        self._rebinarize_prototypes()

    def count_parameters(self) -> int:
        """
        Retourne le nombre d'éléments dans les prototypes accumulateurs.

        HDC n'a pas de paramètres au sens neuronal. On compte les éléments
        des prototypes INT32 (état entraînable) pour comparaison inter-modèles.

        Returns
        -------
        int
            n_classes × D = 2 × 1024 = 2048 éléments.
        """
        return self.n_classes * self.D  # MEM: 2 × 1024 = 2048 éléments

    def estimate_ram_bytes(self, dtype: str = "fp32") -> int:
        """
        Estime l'empreinte RAM du modèle HDC.

        Parameters
        ----------
        dtype : str
            "fp32" → utilise les prototypes INT32 (4 B/élément, worst case).
            "int8" → utilise les prototypes binarisés INT8 (1 B/élément).

        Returns
        -------
        int
            Estimation en octets.

        Notes
        -----
        Budget détaillé (docs/models/hdc_spec.md §2.3) :
            prototypes_acc [2, 1024] INT32 : 8 192 B  # MEM: 8 Ko @ INT32
            prototypes_bin [2, 1024] INT8  : 2 048 B  # MEM: 2 Ko @ INT8
            buffer encodage [1024]   INT32 : 4 096 B  # MEM: 4 Ko @ INT32 (temporaire)
            class_counts    [2]      INT32 :     8 B
            TOTAL (fp32 worst case)        : 14 344 B (~14 Ko)
            TOTAL (int8)                   :  6 152 B (~ 6 Ko)
        """
        if dtype == "int8":
            return (
                self.n_classes * self.D * 1  # prototypes_bin INT8
                + 4 * self.D  # buffer encodage INT32 (temporaire)
                + self.n_classes * 4  # class_counts INT32
            )
        else:  # fp32 / worst case
            return (
                self.n_classes * self.D * 4  # prototypes_acc INT32
                + self.n_classes * self.D * 1  # prototypes_bin INT8
                + 4 * self.D  # buffer encodage INT32 (temporaire)
                + self.n_classes * 4  # class_counts INT32
            )

    def save(self, path: str) -> None:
        """Sauvegarde l'état complet (prototypes + counts) en .npz."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out,
            prototypes_acc=self.prototypes_acc,
            prototypes_bin=self.prototypes_bin,
            class_counts=self.class_counts,
        )

    def load(self, path: str) -> None:
        """Charge un état sauvegardé depuis un .npz."""
        data = np.load(path)
        self.prototypes_acc = data["prototypes_acc"]
        self.prototypes_bin = data["prototypes_bin"]
        self.class_counts = data["class_counts"]
        self._fitted = True

    # ------------------------------------------------------------------
    # Méthodes internes
    # ------------------------------------------------------------------

    def _rebinarize_prototypes(self) -> None:
        """Re-binarise les prototypes INT32 → INT8 après accumulation."""
        proto_bin = np.sign(self.prototypes_acc).astype(np.int8)
        proto_bin[proto_bin == 0] = 1  # cas dégénéré (parité exacte)
        self.prototypes_bin = proto_bin

    @staticmethod
    def _load_feature_bounds(config: dict) -> list[tuple[float, float]]:
        """
        Charge les bornes de features depuis la clé ``feature_bounds`` à la racine du config.

        Returns
        -------
        list[tuple[float, float]]
            [(min_0, max_0), ..., (min_3, max_3)]

        Raises
        ------
        ValueError
            Si les bornes contiennent des None (non encore calculées sur Task 1)
            ou si la clé est absente / vide.
        """
        bounds_cfg = config.get("feature_bounds", {})
        if not bounds_cfg:
            raise ValueError("feature_bounds is empty or missing in config.")
        bounds = []
        for feat_name, bounds_val in bounds_cfg.items():
            f_min, f_max = bounds_val
            if f_min is None or f_max is None:
                raise ValueError(
                    f"Feature bound '{feat_name}' contains None. "
                    "Run S2-03 (train_hdc.py Task 1 fit) to compute bounds from data."
                )
            bounds.append((float(f_min), float(f_max)))
        return bounds
