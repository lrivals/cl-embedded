# ruff: noqa: N803, N806  — X, Sigma sont des conventions mathématiques ML (sklearn API)
"""
mahalanobis_int8.py — Détecteur de Mahalanobis à paramètres INT8 (Sprint 28, S2805).

Quantifie ``mu_`` (vecteur moyen, INT8 affine par-vecteur) et ``sigma_inv_`` (matrice
de covariance inverse, INT8 affine avec scale global par-matrice). La distance de
Mahalanobis est recalculée en FP32 après dequantification — pas de mise à jour de poids
en ligne (le fit est offline).

Mahalanobis est le détecteur le plus compact du projet (~200 B FP32 pour d=5, mesuré
Sprint 20). L'impact absolu de l'INT8 est faible mais valide la généralité de l'approche
de quantification à un modèle non-neuronal.

Mode ``quant="q15"`` (Sprint 34, S3405) : fallback ``TODO(arnaud)`` pour les matrices
``sigma_inv_`` à grande dynamique, où l'INT8 affine global casse la distance (Sprint 28 :
AUROC −0.236 CWRU / −0.238 Pronostia). ``sigma_inv_`` passe en int16 Q15 (scale par-tenseur,
256× plus de résolution), ``mu_`` reste INT8. RAM ÷2 vs FP32 au lieu de ÷4 en INT8.

Réutilise MahalanobisDetector (fit_task, _compute_distances, predict, seuil) et
src/utils/quantization.py (compute_scale_zero_point).

Référence : docs/sprints/sprint_28/S2805_mahalanobis_int8_python.md
"""

from __future__ import annotations

import numpy as np

from src.models.unsupervised.mahalanobis_detector import MahalanobisDetector
from src.utils.quantization import compute_scale_zero_point


def _quantize_affine_int8(x: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Quantification affine UINT8 par-tenseur d'un array numpy.

    Réutilise compute_scale_zero_point (compatible CMSIS-NN). Stockage en ``uint8``
    (1 B/élément, équivalent INT8 en empreinte). Renvoie (q, scale, zero_point).
    """
    scale, zp = compute_scale_zero_point(x)
    q = np.clip(np.round(x / scale) + zp, 0, 255).astype(np.uint8)
    return q, float(scale), int(zp)


def _dequantize_affine_int8(q: np.ndarray, scale: float, zp: int) -> np.ndarray:
    """Reconstruction FP32 d'un array quantifié affine."""
    return (q.astype(np.float32) - zp) * scale


def _quantize_sigma_inv_q15(sigma_inv: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantification Q15 symétrique (int16) par-tenseur de ``sigma_inv_`` (S3405).

    Contrairement à l'INT8 affine global (8 bits de mantisse utile, qui casse sur les
    matrices à grande dynamique — cf. Sprint 28 : AUROC −0.236 CWRU / −0.238 Pronostia),
    Q15 offre 16 bits → résolution 256× plus fine. Scale unique par-matrice (pas par-ligne),
    pas de zero-point (symétrique, adapté à une matrice SPD à valeurs signées).

    Returns
    -------
    (q : np.ndarray[int16], scale : float) tels que ``sigma_inv ≈ q * scale``.
    """
    sigma_inv = np.asarray(sigma_inv, dtype=np.float32)
    max_abs = float(np.max(np.abs(sigma_inv))) if sigma_inv.size else 0.0
    scale = max_abs / 32767.0 if max_abs > 0.0 else 1.0
    q = np.clip(np.round(sigma_inv / scale), -32768, 32767).astype(np.int16)
    return q, float(scale)


def _dequantize_q15(q: np.ndarray, scale: float) -> np.ndarray:
    """Reconstruction FP32 d'un array quantifié Q15 symétrique."""
    return q.astype(np.float32) * scale


class MahalanobisDetectorInt8(MahalanobisDetector):
    """Détecteur de Mahalanobis à stockage INT8 de ``mu_`` et ``sigma_inv_``.

    Hérite de :class:`MahalanobisDetector` pour le fit offline, le seuil et ``predict``.
    Ajoute ``calibrate_int8`` (quantification des paramètres) et ``score_int8`` /
    ``anomaly_score_int8`` (distance recalculée après dequantification).

    Attributes
    ----------
    mu_q_, sigma_inv_q_ : np.ndarray | None
        Paramètres quantifiés (uint8). None tant que calibrate_int8 n'a pas tourné.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        # Mode de quantification : "int8" (défaut, comportement historique) ou "q15"
        # (S3405 : sigma_inv_ en int16 fixed-point, fallback grande dynamique TODO arnaud).
        self.quant: str = str(config.get("quantization", "int8")).lower()
        if self.quant not in ("int8", "q15"):
            raise ValueError(f"quantization inconnue : {self.quant!r} (attendu int8|q15)")
        self.mu_q_: np.ndarray | None = None
        self.sigma_inv_q_: np.ndarray | None = None
        self._mu_scale: float = 1.0
        self._mu_zp: int = 0
        self._sigma_scale: float = 1.0
        self._sigma_zp: int = 0
        self._int8_ready: bool = False
        # Q15 (S3405) : sigma_inv_ int16 + scale par-tenseur ; mu_ reste INT8 affine.
        self.sigma_inv_q15_: np.ndarray | None = None
        self._sigma_q15_scale: float = 1.0
        self._q15_ready: bool = False

    # ------------------------------------------------------------------
    # Fit + calibration
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, task_id: int = 0) -> "MahalanobisDetectorInt8":
        """Fit offline (via fit_task) puis calibration selon ``quant``. Retourne self."""
        self.fit_task(np.asarray(X, dtype=np.float32), task_id)
        self.calibrate()
        return self

    def calibrate(self) -> None:
        """Calibre les paramètres quantifiés selon ``self.quant`` (int8 ou q15)."""
        if self.quant == "q15":
            self.calibrate_q15()
        else:
            self.calibrate_int8()

    def calibrate_int8(self) -> None:
        """Quantifie ``mu_`` (INT8 affine) et ``sigma_inv_`` (INT8 affine par-matrice).

        ``sigma_inv_`` peut avoir une grande dynamique → scale global unique par-matrice
        (pas par-ligne), conformément à la spec S2805.
        """
        if self.mu_ is None or self.sigma_inv_ is None:
            raise RuntimeError("calibrate_int8() requiert un fit_task() préalable.")

        self.mu_q_, self._mu_scale, self._mu_zp = _quantize_affine_int8(
            np.asarray(self.mu_, dtype=np.float32)
        )
        # MEM: mu_q_ = d × 1 B @ INT8 (vs d × 4 B @ FP32)
        self.sigma_inv_q_, self._sigma_scale, self._sigma_zp = _quantize_affine_int8(
            np.asarray(self.sigma_inv_, dtype=np.float32)
        )
        # MEM: sigma_inv_q_ = d² × 1 B @ INT8 (vs d² × 4 B @ FP32)
        self._int8_ready = True

        # Réponse au TODO(arnaud) → calibrate_q15() (S3405). Validé S3406 :
        # ΔAUROC < 0.02 en Q15 sur CWRU/Pronostia (vs −0.236/−0.238 en INT8).

    def calibrate_q15(self) -> None:
        """Quantifie ``mu_`` (INT8 affine, inchangé) et ``sigma_inv_`` (int16 Q15, S3405).

        Réponse au ``TODO(arnaud)`` : ``sigma_inv_`` a une grande dynamique → l'INT8 (8 bits)
        casse la distance. Q15 (16 bits, scale par-tenseur) garde 256× plus de résolution.
        ``mu_`` reste INT8 affine : sa dynamique est faible, il n'est pas concerné par le bug.
        """
        if self.mu_ is None or self.sigma_inv_ is None:
            raise RuntimeError("calibrate_q15() requiert un fit_task() préalable.")

        self.mu_q_, self._mu_scale, self._mu_zp = _quantize_affine_int8(
            np.asarray(self.mu_, dtype=np.float32)
        )
        # MEM: mu_q_ = d × 1 B @ INT8 (vs d × 4 B @ FP32)
        self.sigma_inv_q15_, self._sigma_q15_scale = _quantize_sigma_inv_q15(
            np.asarray(self.sigma_inv_, dtype=np.float32)
        )
        # MEM: sigma_inv_q15_ = d² × 2 B @ Q15 (vs d² × 4 B @ FP32, vs d² × 1 B @ INT8)
        self._q15_ready = True

    # ------------------------------------------------------------------
    # Inférence INT8
    # ------------------------------------------------------------------

    def _dequant_params(self) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruit (mu, sigma_inv) FP32 depuis les paramètres quantifiés."""
        if not self._int8_ready:
            raise RuntimeError("Paramètres INT8 non calibrés. Appeler calibrate_int8().")
        mu = _dequantize_affine_int8(self.mu_q_, self._mu_scale, self._mu_zp)
        sigma_inv = _dequantize_affine_int8(self.sigma_inv_q_, self._sigma_scale, self._sigma_zp)
        return mu, sigma_inv

    def anomaly_score_int8(self, X: np.ndarray) -> np.ndarray:
        """Distance de Mahalanobis avec paramètres INT8 dequantifiés (FP32).

        Parameters
        ----------
        X : np.ndarray [N, d] ou [d]

        Returns
        -------
        np.ndarray [N], float32 — score d'anomalie (élevé = anormal).
        """
        mu, sigma_inv = self._dequant_params()
        X2 = np.atleast_2d(np.asarray(X, dtype=np.float32))
        diff = X2 - mu  # [N, d]
        dist_sq = (diff @ sigma_inv * diff).sum(axis=1)  # (x-μ)ᵀ Σ⁻¹ (x-μ)
        return np.sqrt(np.maximum(dist_sq, 0.0)).astype(np.float32)

    def score_int8(self, x: np.ndarray) -> float:
        """Score d'anomalie INT8 pour un échantillon unique."""
        return float(self.anomaly_score_int8(np.atleast_2d(x))[0])

    def predict_int8(self, X: np.ndarray) -> np.ndarray:
        """Prédit le label binaire (0=normal, 1=anormal) via le score INT8 + seuil."""
        if self.threshold_ is None:
            raise RuntimeError("Seuil non calculé. Appeler fit_task(X, task_id=0) d'abord.")
        return (self.anomaly_score_int8(X) > self.threshold_).astype(np.int64)

    # ------------------------------------------------------------------
    # Inférence Q15 (S3405)
    # ------------------------------------------------------------------

    def _dequant_params_q15(self) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruit (mu, sigma_inv) FP32 depuis les paramètres Q15 (mu INT8 + Σ⁻¹ int16)."""
        if not self._q15_ready:
            raise RuntimeError("Paramètres Q15 non calibrés. Appeler calibrate_q15().")
        mu = _dequantize_affine_int8(self.mu_q_, self._mu_scale, self._mu_zp)
        sigma_inv = _dequantize_q15(self.sigma_inv_q15_, self._sigma_q15_scale)
        return mu, sigma_inv

    def anomaly_score_q15(self, X: np.ndarray) -> np.ndarray:
        """Distance de Mahalanobis avec ``sigma_inv_`` Q15 dequantifié (FP32).

        Parameters
        ----------
        X : np.ndarray [N, d] ou [d]

        Returns
        -------
        np.ndarray [N], float32 — score d'anomalie (élevé = anormal).
        """
        mu, sigma_inv = self._dequant_params_q15()
        X2 = np.atleast_2d(np.asarray(X, dtype=np.float32))
        diff = X2 - mu  # [N, d]
        dist_sq = (diff @ sigma_inv * diff).sum(axis=1)  # (x-μ)ᵀ Σ⁻¹ (x-μ)
        return np.sqrt(np.maximum(dist_sq, 0.0)).astype(np.float32)

    def score_q15(self, x: np.ndarray) -> float:
        """Score d'anomalie Q15 pour un échantillon unique."""
        return float(self.anomaly_score_q15(np.atleast_2d(x))[0])

    def predict_q15(self, X: np.ndarray) -> np.ndarray:
        """Prédit le label binaire (0=normal, 1=anormal) via le score Q15 + seuil."""
        if self.threshold_ is None:
            raise RuntimeError("Seuil non calculé. Appeler fit_task(X, task_id=0) d'abord.")
        return (self.anomaly_score_q15(X) > self.threshold_).astype(np.int64)

    # ------------------------------------------------------------------
    # Empreinte mémoire
    # ------------------------------------------------------------------

    def get_memory_footprint_int8(self) -> dict[str, int]:
        """Empreinte INT8 : ``mu_`` (d B) + ``sigma_inv_`` (d² B) + scales FP32 (overhead).

        Returns
        -------
        dict avec mu_bytes, sigma_inv_bytes, scales_bytes, total_bytes (poids purs),
        total_with_scales_bytes.
        """
        d = int(self.mu_.shape[0]) if self.mu_ is not None else self.n_features_
        mu_bytes = d * 1  # int8
        sigma_bytes = d * d * 1  # int8
        # 2 tenseurs quantifiés × (scale fp32 + zero_point int32) → overhead.
        scales_bytes = 2 * (4 + 4)
        return {
            "mu_bytes": mu_bytes,
            "sigma_inv_bytes": sigma_bytes,
            "scales_bytes": scales_bytes,
            "total_bytes": mu_bytes + sigma_bytes,
            "total_with_scales_bytes": mu_bytes + sigma_bytes + scales_bytes,
        }

    def get_memory_footprint_q15(self) -> dict[str, int]:
        """Empreinte Q15 : ``mu_`` (d B, INT8) + ``sigma_inv_`` (d² × 2 B, int16) + scales.

        Économie ×2 vs FP32 (au lieu de ×4 en INT8) sur ``sigma_inv_``, en échange de la
        résolution 256× plus fine qui restaure l'AUROC sur matrices à grande dynamique.

        Returns
        -------
        dict avec mu_bytes, sigma_inv_bytes, scales_bytes, total_bytes (poids purs),
        total_with_scales_bytes.
        """
        d = int(self.mu_.shape[0]) if self.mu_ is not None else self.n_features_
        mu_bytes = d * 1  # int8 affine
        sigma_bytes = d * d * 2  # int16 Q15
        # mu : scale fp32 + zero_point int32 ; sigma_inv : scale fp32 (symétrique, pas de zp).
        scales_bytes = (4 + 4) + 4
        return {
            "mu_bytes": mu_bytes,
            "sigma_inv_bytes": sigma_bytes,
            "scales_bytes": scales_bytes,
            "total_bytes": mu_bytes + sigma_bytes,
            "total_with_scales_bytes": mu_bytes + sigma_bytes + scales_bytes,
        }
