# ruff: noqa: N803  — X est une convention mathématique ML (sklearn API)
"""
model_pair.py — Co-exécution d'un détecteur non-supervisé et d'un modèle supervisé.

`ModelPair` est le socle du benchmark « paire » du Sprint 30 (et du méta-modèle
d'arbitrage du Sprint 31). Il associe un `MahalanobisDetector` (baseline anomaly,
sortie 0/1, ~128 B RAM, déployable « à froid ») à un modèle supervisé du projet
(`HDC`, `EWC`, `TinyOL`) et expose :

- `predict_individual(x)` → `(pred_maha, pred_sup)`, sorties brutes alignées par
  échantillon ;
- `predict_ensemble(x, rule)` → fusion des 2 sorties (`or` / `and` / `soft_vote` /
  `weighted`) ;
- `predict_proba(x)` → scores continus d'ensemble pour l'AUROC.

Mode `"binary"` (Partie A) : binarise la sortie supervisée en *normal-vs-fault* via
la convention exacte du Sprint 28 (cf. :func:`_binarize_labels`) → les 2 sorties sont
directement comparables (ensemble / désaccord propres).
Mode `"native"` (Partie B) : conserve la sortie native du modèle supervisé.

Les modèles supervisés du projet n'ont pas d'interface homogène (HDC.predict → labels
`int64`, EWC = `nn.Module` dont `forward` renvoie une proba sigmoïde, TinyOL.predict →
`tuple`). Une fine couche d'adaptation (`_supervised_labels` / `_supervised_proba`)
normalise chaque sortie en `np.ndarray` plutôt que de forcer un héritage commun
(cf. S3000 §Notes, S3001 ligne 49).
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Valeurs par défaut — surchargeables via `config` (règle CLAUDE.md : aucun
# hyperparamètre en dur dans le code).
DEFAULT_MODE: str = "binary"
DEFAULT_FUSION_RULE: str = "or"
DEFAULT_WEIGHTS: tuple[float, float] = (0.5, 0.5)  # (poids_maha, poids_supervisé)
ENSEMBLE_THRESHOLD: float = 0.5  # seuil de décision sur proba fusionnée

VALID_MODES = ("binary", "native")
VALID_RULES = ("or", "and", "soft_vote", "weighted")


def _binarize_labels(y: np.ndarray) -> np.ndarray:
    """Réduit des labels {0,1}, multiclasse ou continus à du binaire *normal-vs-fault*.

    Convention identique au Sprint 28 (`scripts/benchmark_int8_fp32.py`) : la classe
    « normale / saine » est la plus petite valeur (typiquement 0) ; tout le reste est
    une panne.

    Parameters
    ----------
    y : np.ndarray
        Labels ou prédictions brutes.

    Returns
    -------
    np.ndarray [N], int
        0 = normal, 1 = panne.
    """
    y = np.asarray(y).ravel()
    uniq = np.unique(y)
    if uniq.size <= 2:
        # Déjà binaire (ou constant) : recaler sur {0, 1} si besoin.
        return (y != uniq.min()).astype(int) if uniq.size == 2 else y.astype(int)
    return (y != uniq.min()).astype(int)


def _to_numpy(value: Any) -> np.ndarray:
    """Convertit une sortie modèle (torch.Tensor, scalaire, liste, ndarray) en ndarray."""
    if hasattr(value, "detach"):  # torch.Tensor
        value = value.detach().cpu().numpy()
    return np.asarray(value)


# Règles native_to_fault valides (S3007) — pilotées par config, aucun seuil en dur.
VALID_NATIVE_RULES = ("identity", "rul_threshold", "nonzero_class")


def native_to_fault(
    pred_native: np.ndarray,
    rule: str = "identity",
    threshold: float | None = None,
) -> np.ndarray:
    """Dérive une décision binaire *faute oui/non* depuis une sortie native (S3007).

    Mode `"native"` (Partie B) : la sortie supervisée n'est pas binarisée par
    convention « plus petite classe = normal » mais réduite à une décision de panne
    via une règle explicite par dataset (configurée dans le bloc `native:`), afin de
    comparer le désaccord avec le détecteur d'anomalie binaire.

    Parameters
    ----------
    pred_native : np.ndarray
        Sortie native du modèle supervisé (RUL en cycles, label multi-classe, ou
        proba binaire selon la tâche).
    rule : {"identity", "rul_threshold", "nonzero_class"}
        - "identity"     : la sortie est déjà 0/1 (tâche binaire).
        - "rul_threshold": faute si RUL ≤ `threshold` (criticité imminente).
        - "nonzero_class": faute si la classe ≠ 0 (0 = normal/sain).
    threshold : float | None
        Seuil de criticité (requis pour `rul_threshold`).

    Returns
    -------
    np.ndarray [N], int
        0 = normal, 1 = faute.
    """
    if rule not in VALID_NATIVE_RULES:
        raise ValueError(f"rule native invalide : {rule!r} (attendu {VALID_NATIVE_RULES}).")
    y = _to_numpy(pred_native).ravel()
    if rule == "rul_threshold":
        if threshold is None:
            raise ValueError("rule='rul_threshold' requiert un threshold.")
        return (y <= float(threshold)).astype(int)
    if rule == "nonzero_class":
        return (np.rint(y) != 0).astype(int)
    # identity : sortie déjà binaire (0/1) éventuellement en proba → seuil 0.5.
    return (y >= 0.5).astype(int) if np.issubdtype(y.dtype, np.floating) else (y != 0).astype(int)


class ModelPair:
    """Co-exécution Mahalanobis (non-supervisé) + modèle supervisé.

    Parameters
    ----------
    detector : MahalanobisDetector
        Baseline anomaly ; `predict(x)` → 0/1, `anomaly_score(x)` → distance continue.
    classifier : Any
        Modèle supervisé du projet (HDC / EWC / TinyOL). Duck-typé : on cherche
        `predict_proba`, puis `predict`, puis l'appel direct (`__call__`/`forward`).
    mode : {"binary", "native"}
        "binary" (Partie A) binarise la sortie supervisée en normal-vs-fault.
        "native" (Partie B) conserve la sortie native du modèle supervisé.
    fusion_rule : {"or", "and", "soft_vote", "weighted"}
        Règle de fusion d'ensemble par défaut.
    config : dict | None
        Sous-section optionnelle ; clés lues : `fusion_rule`, `weights`
        (`[poids_maha, poids_supervisé]`), `ensemble_threshold`.
    """

    def __init__(
        self,
        detector: Any,
        classifier: Any,
        mode: str = DEFAULT_MODE,
        fusion_rule: str = DEFAULT_FUSION_RULE,
        config: dict | None = None,
    ) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"mode invalide : {mode!r} (attendu {VALID_MODES}).")

        config = config or {}
        fusion_rule = config.get("fusion_rule", fusion_rule)
        if fusion_rule not in VALID_RULES:
            raise ValueError(f"fusion_rule invalide : {fusion_rule!r} (attendu {VALID_RULES}).")

        self.detector = detector
        self.classifier = classifier
        self.mode = mode
        self.fusion_rule = fusion_rule

        weights = config.get("weights", DEFAULT_WEIGHTS)
        w = np.asarray(weights, dtype=np.float64)
        if w.size != 2 or w.sum() <= 0:
            raise ValueError(f"weights doit être [w_maha, w_sup] de somme > 0, reçu {weights!r}.")
        self.weights = w / w.sum()  # normalisé
        self.ensemble_threshold = float(config.get("ensemble_threshold", ENSEMBLE_THRESHOLD))

    # ------------------------------------------------------------------
    # Couche d'adaptation supervisée (interfaces hétérogènes HDC/EWC/TinyOL)
    # ------------------------------------------------------------------

    def _supervised_proba(self, x: np.ndarray) -> np.ndarray:
        """Proba de panne ∈ [0, 1] du modèle supervisé, normalisée en ndarray [N]."""
        clf = self.classifier
        if hasattr(clf, "predict_proba"):
            proba = _to_numpy(clf.predict_proba(x))
        elif hasattr(clf, "predict"):
            out = _to_numpy(clf.predict(x))
            # HDC.predict → labels int ; on les traite comme proba 0/1 faute de mieux.
            proba = out.astype(np.float64)
        else:  # nn.Module (EWC) : forward renvoie une proba sigmoïde [N, 1]
            proba = _to_numpy(clf(x))

        proba = np.asarray(proba, dtype=np.float64).reshape(proba.shape[0], -1)
        if proba.shape[1] == 1:
            return proba.ravel()
        # Sortie multi-classe : proba de panne = 1 - proba(classe normale=index 0).
        return 1.0 - proba[:, 0]

    def _supervised_labels(self, x: np.ndarray) -> np.ndarray:
        """Labels du modèle supervisé, normalisés (binarisés si mode == 'binary')."""
        clf = self.classifier
        if hasattr(clf, "predict"):
            labels = _to_numpy(clf.predict(x)).ravel()
        else:  # pas de predict : seuiller la proba
            labels = (self._supervised_proba(x) >= ENSEMBLE_THRESHOLD).astype(int)
        if self.mode == "binary":
            return _binarize_labels(labels)
        return labels.astype(int)

    def _maha_proba(self, x: np.ndarray) -> np.ndarray:
        """Score Mahalanobis → proba ∈ [0, 1], logistique centrée sur le seuil.

        `sigmoid(score - threshold)` : monotone, borné, et > 0.5 ⟺ score > seuil ⟺
        `detector.predict == 1` (cohérence décision/proba).
        """
        scores = _to_numpy(self.detector.anomaly_score(x)).astype(np.float64).ravel()
        thr = self.detector.threshold_
        if thr is None:
            raise RuntimeError("detector.threshold_ non calculé (fit_task sur Task 0 requis).")
        z = np.clip(scores - thr, -60.0, 60.0)  # garde-fou overflow exp
        return 1.0 / (1.0 + np.exp(-z))

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def predict_individual(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Retourne `(pred_maha, pred_sup)` — sorties brutes alignées par échantillon."""
        pred_maha = _to_numpy(self.detector.predict(x)).astype(int).ravel()  # MEM: N×8 B
        pred_sup = self._supervised_labels(x)  # MEM: N×8 B
        return pred_maha, pred_sup

    def predict_ensemble(self, x: np.ndarray, rule: str | None = None) -> np.ndarray:
        """Combine les 2 sorties selon `fusion_rule` (override possible via `rule`)."""
        rule = rule or self.fusion_rule
        if rule not in VALID_RULES:
            raise ValueError(f"rule invalide : {rule!r} (attendu {VALID_RULES}).")

        if rule in ("or", "and"):
            pred_maha, pred_sup = self.predict_individual(x)
            a = pred_maha.astype(bool)
            b = pred_sup.astype(bool)
            combined = (a | b) if rule == "or" else (a & b)  # MEM: N×1 B (bool)
            return combined.astype(int)

        # soft_vote / weighted : seuil sur la proba fusionnée.
        proba = self._fused_proba(x, rule)
        return (proba >= self.ensemble_threshold).astype(int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Score continu d'ensemble ∈ [0, 1] pour l'AUROC (proba fusionnée)."""
        return self._fused_proba(x, self.fusion_rule)

    def _fused_proba(self, x: np.ndarray, rule: str) -> np.ndarray:
        """Proba fusionnée : moyenne (pondérée si `weighted`) des 2 probas. # MEM: N×8 B"""
        p_maha = self._maha_proba(x)
        p_sup = self._supervised_proba(x)
        if rule == "weighted":
            return self.weights[0] * p_maha + self.weights[1] * p_sup
        # soft_vote (et fallback or/and via proba) : moyenne simple.
        return 0.5 * (p_maha + p_sup)

    def __repr__(self) -> str:
        return (
            f"ModelPair(detector={type(self.detector).__name__}, "
            f"classifier={type(self.classifier).__name__}, "
            f"mode={self.mode!r}, fusion_rule={self.fusion_rule!r})"
        )
