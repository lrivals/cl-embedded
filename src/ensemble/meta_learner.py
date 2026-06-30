# ruff: noqa: N803  — X est une convention mathématique ML (sklearn API)
"""
meta_learner.py — Méta-modèle d'arbitrage (stacking) — Sprint 31 (S3101).

Le Sprint 30 fusionnait les 2 sorties d'une :class:`~src.ensemble.model_pair.ModelPair`
(Mahalanobis non-supervisé + supervisé EWC/HDC/TinyOL) par une **règle statique**
(`or`/`and`/`soft_vote`/`weighted`). :class:`MetaLearner` remplace cette fusion fixe par un
**arbitrage appris** : un modèle léger (régression logistique ou petit MLP 1 couche) entraîné sur
un vecteur d'entrée compact dérivé des 2 modèles de base, qui apprend *quand* faire confiance à
quel modèle.

Contraintes de design (portage board S3105) :

- **Vecteur d'entrée compact** (≤ 8 features) et **borné [0, 1]** : pas de scaler à embarquer,
  cohérent avec :meth:`ModelPair._maha_proba` (sigmoïde score-seuil). Voir
  :func:`build_meta_features`.
- **Poids exportables** en FP32 via :meth:`MetaLearner.export_weights` (consommés par
  `scripts/export_weights_c.py` → `meta_head.c`, S3105).
- Aucun hyperparamètre en dur : `kind`, `input_features`, `hidden_size` pilotés par
  `configs/meta_stacking.yaml` (règle CLAUDE.md).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

if TYPE_CHECKING:  # évite l'import circulaire au runtime (model_pair n'importe pas meta_learner)
    from src.ensemble.model_pair import ModelPair

VALID_KINDS = ("logreg", "mlp")

# Features par défaut (toutes ∈ [0, 1]) — surchargeable via `input_features`.
DEFAULT_FEATURES: tuple[str, ...] = ("p_maha", "p_sup", "disagreement", "conf_sup")

# Features disponibles → tout est borné [0, 1] pour rester portable MCU (pas de scaler).
AVAILABLE_FEATURES: tuple[str, ...] = (
    "p_maha",        # proba d'anomalie Mahalanobis (sigmoïde score-seuil)
    "p_sup",         # proba de panne du modèle supervisé
    "pred_maha",     # décision binaire Mahalanobis (0/1)
    "pred_sup",      # décision binaire supervisée (0/1)
    "disagreement",  # 1 si les 2 décisions diffèrent, 0 sinon
    "conf_sup",      # confiance supervisée |p_sup - 0.5| * 2 ∈ [0, 1]
    "conf_maha",     # confiance Mahalanobis |p_maha - 0.5| * 2 ∈ [0, 1]
)


def build_meta_features(
    pair: ModelPair,
    X: np.ndarray,
    feature_names: list[str] | tuple[str, ...] = DEFAULT_FEATURES,
) -> tuple[np.ndarray, list[str]]:
    """Construit le vecteur d'entrée du méta-modèle depuis les sorties des 2 bases.

    Réutilise les internes de :class:`ModelPair` (déjà calibrés Sprint 30). Toutes les features
    sont bornées [0, 1] → directement portables MCU sans normalisation.

    Parameters
    ----------
    pair : ModelPair
        Paire entraînée (détecteur + supervisé).
    X : np.ndarray [N, d]
        Échantillons d'évaluation.
    feature_names : list[str]
        Sous-ensemble ordonné de :data:`AVAILABLE_FEATURES`.

    Returns
    -------
    meta_X : np.ndarray [N, F], float32
        Matrice de features méta.
    feature_names : list[str]
        Noms des colonnes (ordre des colonnes de `meta_X`).
    """
    unknown = [f for f in feature_names if f not in AVAILABLE_FEATURES]
    if unknown:
        raise ValueError(f"features inconnues : {unknown} (disponibles : {AVAILABLE_FEATURES}).")

    p_maha = pair._maha_proba(X)                       # MEM: N×8 B @ FP64 → cast FP32 plus bas
    p_sup = pair._supervised_proba(X)                  # MEM: N×8 B
    pred_maha, pred_sup = pair.predict_individual(X)    # MEM: N×8 B chacun

    columns: dict[str, np.ndarray] = {
        "p_maha": p_maha,
        "p_sup": p_sup,
        "pred_maha": pred_maha.astype(np.float64),
        "pred_sup": pred_sup.astype(np.float64),
        "disagreement": (pred_maha != pred_sup).astype(np.float64),
        "conf_sup": np.abs(p_sup - 0.5) * 2.0,
        "conf_maha": np.abs(p_maha - 0.5) * 2.0,
    }
    meta_X = np.column_stack([columns[f] for f in feature_names]).astype(np.float32)  # MEM: N×F×4 B
    return meta_X, list(feature_names)


class MetaLearner:
    """Méta-modèle de stacking : arbitre les sorties de 2 modèles de base.

    Parameters
    ----------
    kind : {"logreg", "mlp"}
        "logreg" = régression logistique (1 vecteur de poids, le plus léger pour le MCU) ;
        "mlp" = perceptron 1 couche cachée (`hidden_size` neurones).
    input_features : list[str] | None
        Noms des features d'entrée (cf. :func:`build_meta_features`). Stockés pour l'export C.
        `None` → :data:`DEFAULT_FEATURES`.
    hidden_size : int
        Nombre de neurones cachés (mode "mlp" uniquement).
    class_weight : str | dict | None
        Pondération des classes de la régression logistique (`"balanced"` recommandé en détection
        de panne déséquilibrée pour éviter l'effondrement vers la classe majoritaire). Ignoré en
        mode "mlp" (non supporté par `MLPClassifier`).
    seed : int
        Graine de reproductibilité du solveur.
    config : dict | None
        Sous-section optionnelle ; clés lues : `kind`, `input_features`, `hidden_size`,
        `class_weight`, `seed`.
    """

    def __init__(
        self,
        kind: str = "logreg",
        input_features: list[str] | None = None,
        hidden_size: int = 8,
        class_weight: str | dict | None = "balanced",
        seed: int = 42,
        config: dict | None = None,
    ) -> None:
        config = config or {}
        kind = config.get("kind", kind)
        if kind not in VALID_KINDS:
            raise ValueError(f"kind invalide : {kind!r} (attendu {VALID_KINDS}).")

        self.kind = kind
        self.input_features = list(config.get("input_features", input_features or DEFAULT_FEATURES))
        self.hidden_size = int(config.get("hidden_size", hidden_size))
        self.class_weight = config.get("class_weight", class_weight)
        self.seed = int(config.get("seed", seed))
        self._model: Any = None

    def fit(self, meta_X: np.ndarray, y: np.ndarray) -> MetaLearner:
        """Entraîne le méta sur `meta_X` (features out-of-fold) et les labels binaires `y`.

        Parameters
        ----------
        meta_X : np.ndarray [N, F]
            Features dérivées des 2 modèles de base (collectées hors fold d'entraînement).
        y : np.ndarray [N]
            Labels binaires (0 = normal, 1 = panne).
        """
        meta_X = np.asarray(meta_X, dtype=np.float32)
        y = np.asarray(y).ravel().astype(int)
        if self.kind == "logreg":
            self._model = LogisticRegression(
                max_iter=1000, random_state=self.seed, class_weight=self.class_weight
            )
        else:  # mlp 1 couche cachée
            self._model = MLPClassifier(
                hidden_layer_sizes=(self.hidden_size,),
                max_iter=1000,
                random_state=self.seed,
            )
        self._model.fit(meta_X, y)
        return self

    def predict(self, meta_X: np.ndarray) -> np.ndarray:
        """Décision binaire arbitrée [N] (0 = normal, 1 = panne)."""
        self._check_fitted()
        return self._model.predict(np.asarray(meta_X, dtype=np.float32)).astype(int)

    def predict_proba(self, meta_X: np.ndarray) -> np.ndarray:
        """Proba de panne arbitrée [N] ∈ [0, 1] (colonne de la classe 1)."""
        self._check_fitted()
        proba = self._model.predict_proba(np.asarray(meta_X, dtype=np.float32))
        # Robuste au cas dégénéré (une seule classe vue à l'entraînement).
        if proba.shape[1] == 1:
            return np.full(proba.shape[0], float(self._model.classes_[0]))
        return proba[:, 1]

    def export_weights(self) -> dict:
        """Poids FP32 prêts pour `scripts/export_weights_c.py` → `meta_head.c` (S3105).

        Returns
        -------
        dict
            logreg → ``{"kind", "w" [F], "b", "feature_names"}`` ;
            mlp → ``{"kind", "w1" [H, F], "b1" [H], "w2" [1, H], "b2", "feature_names"}``.
        """
        self._check_fitted()
        if self.kind == "logreg":
            return {
                "kind": "logreg",
                "w": self._model.coef_.ravel().astype(np.float32),
                "b": float(self._model.intercept_[0]),
                "feature_names": list(self.input_features),
            }
        # mlp : sklearn stocke coefs_ = [W_input→hidden, W_hidden→output].
        w1 = self._model.coefs_[0].T.astype(np.float32)   # [H, F]
        b1 = self._model.intercepts_[0].astype(np.float32)  # [H]
        w2 = self._model.coefs_[1].T.astype(np.float32)    # [1, H]
        b2 = float(self._model.intercepts_[1][0])
        return {
            "kind": "mlp",
            "w1": w1,
            "b1": b1,
            "w2": w2,
            "b2": b2,
            "feature_names": list(self.input_features),
        }

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("MetaLearner non entraîné : appeler fit() d'abord.")

    def __repr__(self) -> str:
        return (
            f"MetaLearner(kind={self.kind!r}, hidden_size={self.hidden_size}, "
            f"input_features={self.input_features})"
        )
