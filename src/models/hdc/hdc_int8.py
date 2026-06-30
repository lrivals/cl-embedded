# src/models/hdc/hdc_int8.py
# ruff: noqa: N803, N806  — H_level, H_pos, H_obs, D sont des conventions mathématiques HDC
"""
hdc_int8.py — Classifieur HDC à stockage INT8/INT16 (Sprint 28, S2803).

HDC est architecturalement entier : après binarisation, les hypervecteurs ont des
valeurs ±1 exactement représentables en ``int8``. La mémoire associative (associative
memory, AM) accumule des bundles de N hypervecteurs ±1 → plage [-N, +N] : stockée en
``int16`` pour éviter l'overflow sans recourir au INT32 du modèle FP32.

Différence avec les modèles neuronaux INT8 (ewc_mlp_int8.py, tinyol_int8.py) : ici la
quantification n'est PAS une approximation. Les valeurs binarisées sont exactes en INT8,
donc la métrique INT8 est identique à la métrique FP32 — seule la RAM diffère
(compression ≈ ×2–3 selon les structures, cf. get_memory_footprint_int8()).

Réutilise les fonctions stateless de hdc_classifier.py (encode_observation,
quantize_feature) et base_vectors.py (generate_base_hvectors / load_base_vectors).

Référence : Benatti2019HDC, docs/sprints/sprint_28/S2803_hdc_int8_python.md,
docs/sprints/sprint_24/S2402_uint8_ewc_hdc.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.models.hdc.base_vectors import (
    generate_base_hvectors,
    load_base_vectors,
    save_base_vectors,
)
from src.models.hdc.hdc_classifier import encode_observation

# Borne de saturation de l'AM INT16 — clip pour garantir l'absence d'overflow.
_INT16_MAX: int = 32767
_INT16_MIN: int = -32768


class HDCClassifierInt8:
    """Classifieur HDC à stockage INT8 (hypervecteurs de base) + INT16 (AM).

    Hypervecteurs de base ``H_level`` / ``H_pos`` stockés en ``int8`` (±1), mémoire
    associative ``am`` en ``int16`` (accumulation de bundles sans overflow).

    L'interface reproduit celle de :class:`HDCClassifier` (``update`` / ``predict``)
    afin d'être interchangeable dans l'adaptateur benchmark (S2801), et ajoute les
    méthodes ``*_int8`` explicites demandées par la spec S2803.

    Parameters
    ----------
    config : dict
        Configuration. Accepte le schéma projet (``config["hdc"]["D"]``,
        ``config["data"]["n_classes"]`` …) ou le schéma plat de la spec
        (``config["hdc_dim"]``, ``config["n_classes"]`` …).

    References
    ----------
    Benatti2019HDC, docs/models/hdc_spec.md
    """

    def __init__(self, config: dict) -> None:
        self.config = config

        hdc_cfg = config.get("hdc", {})
        data_cfg = config.get("data", {})

        # Schéma projet en priorité, repli sur le schéma plat de la spec S2803.
        self.D: int = int(hdc_cfg.get("D", config.get("hdc_dim", 2048)))
        self.n_levels: int = int(hdc_cfg.get("n_levels", config.get("n_levels", 10)))
        self.n_classes: int = int(data_cfg.get("n_classes", config.get("n_classes", 4)))
        self.n_features: int = int(data_cfg.get("n_features", config.get("n_features", 9)))

        self.feature_bounds: list[tuple[float, float]] = self._load_feature_bounds(config)

        # Hypervecteurs de base : int8 (±1). Chargés depuis .npz ou générés on-the-fly.
        bv_path = hdc_cfg.get("base_vectors_path")
        if bv_path is not None and Path(bv_path).exists():
            self.H_level, self.H_pos = load_base_vectors(bv_path)
        else:
            self.H_level, self.H_pos = generate_base_hvectors(
                D=self.D,
                n_levels=self.n_levels,
                n_features=self.n_features,
                seed=int(hdc_cfg.get("seed", config.get("seed", 42))),
            )
            if bv_path is not None:
                save_base_vectors(self.H_level, self.H_pos, bv_path)

        # Mémoire associative — INT16 (vs INT32 en FP32) : bundles ±1 ∈ [-N, +N].
        # MEM: n_classes × D × 2 B = 4 × 2048 × 2 = 16 384 B @ INT16
        self.am: np.ndarray = np.zeros((self.n_classes, self.D), dtype=np.int16)
        self.class_counts: np.ndarray = np.zeros(self.n_classes, dtype=np.int32)
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Encodage
    # ------------------------------------------------------------------

    def encode_int8(self, x: np.ndarray) -> np.ndarray:
        """Encode un échantillon en hypervecteur d'observation binarisé INT8.

        Parameters
        ----------
        x : np.ndarray [n_features], float32
            Vecteur de features normalisé.

        Returns
        -------
        np.ndarray [D], dtype=int8, valeurs ∈ {-1, +1}
        """
        return encode_observation(
            np.asarray(x, dtype=np.float32),
            self.H_level,
            self.H_pos,
            self.feature_bounds,
            self.n_levels,
            self.D,
        )

    # ------------------------------------------------------------------
    # Entraînement / mise à jour
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HDCClassifierInt8":
        """Encode et accumule tous les échantillons dans l'AM INT16.

        Parameters
        ----------
        X : np.ndarray [N, n_features], float32
        y : np.ndarray [N], int

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).ravel().astype(int)
        for sample, label in zip(X, y):
            self.update_int8(sample, int(label))
        return self

    def update_int8(self, x: np.ndarray, y: int) -> None:
        """Mise à jour en ligne : ajoute l'hypervecteur encodé à ``am[y]`` (INT16).

        L'accumulation est saturée à la plage INT16 pour garantir l'absence d'overflow
        (équivalent MCU : ``__SSAT`` sur 16 bits).
        """
        hv = self.encode_int8(x)  # MEM: D × 1 B @ INT8
        acc = self.am[y].astype(np.int32) + hv.astype(np.int32)  # MEM: D × 4 B (temporaire)
        np.clip(acc, _INT16_MIN, _INT16_MAX, out=acc)
        self.am[y] = acc.astype(np.int16)
        self.class_counts[y] += 1
        self._fitted = True

    # Alias d'interface compatible avec HDCClassifier (adaptateur benchmark / boucle CL).
    def update(self, x: np.ndarray, y: np.ndarray) -> float:
        """Met à jour l'AM sur un batch ; retourne le taux d'erreur (proxy de loss)."""
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y).ravel().astype(int)
        for sample, label in zip(x, y):
            self.update_int8(sample, int(label))
        preds = self.predict(x)
        return float(np.mean(preds != y)) if len(y) else 0.0

    def on_task_end(self, task_id: int, dataloader: Any) -> None:
        """Callback fin de tâche. HDC additif : aucun post-traitement requis."""
        # Pas d'oubli catastrophique par construction (mémoire additive).
        return None

    # ------------------------------------------------------------------
    # Inférence
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Inférence par similarité (dot-product entier) entre H_obs et l'AM.

        Sur MCU : produit scalaire INT8×INT16 accumulé en INT32. Ici on calcule le
        dot-product en ``int32`` puis ``argmax`` — équivalent au cosinus à norme près
        (l'AM binarisée n'est pas renormalisée, cohérent avec HDCClassifier).

        Parameters
        ----------
        X : np.ndarray [batch_size, n_features], float32

        Returns
        -------
        np.ndarray [batch_size], dtype=int64
        """
        if not self._fitted:
            raise RuntimeError("HDCClassifierInt8 not fitted. Call fit()/update() first.")

        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        am_i32 = self.am.astype(np.int32)  # MEM: n_classes × D × 4 B (temporaire)
        preds = []
        for sample in X:
            hv = self.encode_int8(sample).astype(np.int32)  # MEM: D × 4 B (temporaire)
            similarities = am_i32 @ hv  # [n_classes] — dot-product INT32
            preds.append(int(np.argmax(similarities)))
        return np.array(preds, dtype=np.int64)

    # ------------------------------------------------------------------
    # Empreinte mémoire
    # ------------------------------------------------------------------

    def estimate_ram_bytes(self, dtype: str = "int8") -> int:
        """Estime la RAM des structures persistantes (base vectors + AM).

        Parameters
        ----------
        dtype : "int8" → AM stockée en INT16 (modèle de ce fichier).
                "fp32" → AM hypothétique en INT32 (modèle HDCClassifier de référence).
        """
        base_bytes = self.H_level.size + self.H_pos.size  # int8 → 1 B/élément
        if dtype == "fp32":
            am_bytes = self.n_classes * self.D * 4  # INT32 (référence FP32)
        else:
            am_bytes = self.n_classes * self.D * 2  # INT16 (ce modèle)
        return int(base_bytes + am_bytes)

    def get_memory_footprint_int8(self) -> dict[str, int]:
        """Empreinte mémoire INT8/INT16 réelle des structures allouées.

        Note de spec : S2803 nomme la structure d'encodage ``base_vecs`` indexée par
        ``n_features``. Le code réel (hérité de hdc_classifier.py) sépare
        ``H_level`` [n_levels, D] et ``H_pos`` [n_features, D] — les deux sont en INT8.
        ``base_vecs_bytes`` agrège donc H_level + H_pos.

        Returns
        -------
        dict avec base_vecs_bytes, am_bytes, total_bytes (+ détails H_level/H_pos).
        """
        h_level_bytes = int(self.H_level.size)  # int8 → 1 B/élément
        h_pos_bytes = int(self.H_pos.size)
        base_bytes = h_level_bytes + h_pos_bytes
        am_bytes = self.n_classes * self.D * 2  # int16
        return {
            "h_level_bytes": h_level_bytes,
            "h_pos_bytes": h_pos_bytes,
            "base_vecs_bytes": base_bytes,
            "am_bytes": am_bytes,
            "total_bytes": base_bytes + am_bytes,
        }

    def count_parameters(self) -> int:
        """Nombre d'éléments de la mémoire associative (état entraînable)."""
        return self.n_classes * self.D

    # ------------------------------------------------------------------
    # Persistance
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Sauvegarde l'AM et les compteurs en .npz."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, am=self.am, class_counts=self.class_counts)

    def load(self, path: str) -> None:
        """Charge un état sauvegardé depuis un .npz."""
        data = np.load(path)
        self.am = data["am"].astype(np.int16)
        self.class_counts = data["class_counts"]
        self._fitted = True

    # ------------------------------------------------------------------
    # Interne
    # ------------------------------------------------------------------

    @staticmethod
    def _load_feature_bounds(config: dict) -> list[tuple[float, float]]:
        """Charge ``feature_bounds`` (même logique que HDCClassifier)."""
        bounds_cfg = config.get("feature_bounds", {})
        if not bounds_cfg:
            raise ValueError("feature_bounds is empty or missing in config.")
        bounds = []
        for feat_name, bounds_val in bounds_cfg.items():
            f_min, f_max = bounds_val
            if f_min is None or f_max is None:
                raise ValueError(
                    f"Feature bound '{feat_name}' contains None. "
                    "Run train_hdc.py Task 1 fit to compute bounds from data."
                )
            bounds.append((float(f_min), float(f_max)))
        return bounds
