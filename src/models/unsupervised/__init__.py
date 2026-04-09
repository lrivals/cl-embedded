"""
Module baselines non supervisées pour la détection d'anomalies en scénario CL.

Ces modèles sont PC-only — ils ne sont pas soumis à la contrainte 64 Ko STM32N6.
Les labels ne sont utilisés qu'en évaluation (pas pendant l'entraînement).

Modèles disponibles
-------------------
KMeansDetector      : K-Means avec sélection K dynamique via silhouette/elbow.
KNNDetector         : KNN distance-based anomaly detection.
PCABaseline         : PCA reconstruction error baseline.
MahalanobisDetector : Distance de Mahalanobis (μ + Σ⁻¹ offline). 80 B @ FP32 pour d=4.
DBSCANDetector      : DBSCAN density-based clustering. Points bruit = anomalies. Score = dist core.

Références
----------
docs/models/unsupervised_spec.md (à créer en S5-08)
"""

from .dbscan_detector import DBSCANDetector
from .kmeans_detector import KMeansDetector
from .knn_detector import KNNDetector
from .mahalanobis_detector import MahalanobisDetector
from .pca_baseline import PCABaseline

__all__ = ["KMeansDetector", "KNNDetector", "PCABaseline", "MahalanobisDetector", "DBSCANDetector"]
