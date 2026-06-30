# ruff: noqa: N803  — X est une convention mathématique ML (sklearn API)
"""
disagreement_metrics.py — Métriques de désaccord inter-modèles (Sprint 30).

Quantifie, pour une paire de modèles (cf. `src/ensemble/model_pair.py`) :

- **où** ils divergent — `disagreement_rate`, `per_sample_disagreement_mask` ;
- **à quel point** leur accord dépasse le hasard — `cohen_kappa` ;
- **qui a raison** quand ils divergent — `disagreement_confusion` ;
- **pourquoi** ils divergent (origine dans l'espace des features / score Mahalanobis /
  proximité de frontière) — `analyze_disagreement_origin`.

C'est ce qui distingue le benchmark « paire » d'un simple empilement de deux résultats
individuels. Usage typique : récupérer `(y_true, y_pred_a)` et `(y_true, y_pred_b)`
via `run_cl_scenario_full()` pour chaque modèle, aligner par index d'échantillon, puis
appliquer ces fonctions (cf. S3000 §Notes).

Complète `metrics.py` (AA/AF/BWT) et `anomaly_metrics.py` (AUROC/F1).
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import cohen_kappa_score


def _align(*arrays: np.ndarray) -> list[np.ndarray]:
    """Aplatit et vérifie l'alignement par index d'échantillon."""
    flat = [np.asarray(a).ravel() for a in arrays]
    n = flat[0].size
    if any(a.size != n for a in flat):
        raise ValueError(f"Tableaux non alignés : tailles {[a.size for a in flat]}.")
    return flat


def disagreement_rate(y_a: np.ndarray, y_b: np.ndarray) -> float:
    """Fraction d'échantillons où `pred_a != pred_b`.

    Parameters
    ----------
    y_a, y_b : np.ndarray [N]
        Prédictions des deux modèles, alignées par index.

    Returns
    -------
    float
        Taux de désaccord ∈ [0, 1] (0.0 si tableaux vides).
    """
    a, b = _align(y_a, y_b)
    if a.size == 0:
        return 0.0
    return float(np.mean(a != b))


def cohen_kappa(y_a: np.ndarray, y_b: np.ndarray) -> float:
    """Accord inter-modèles corrigé du hasard (kappa de Cohen).

    1.0 = accord parfait, 0.0 ≈ accord attendu au hasard, < 0 = pire que le hasard.

    Parameters
    ----------
    y_a, y_b : np.ndarray [N]

    Returns
    -------
    float
        Kappa de Cohen. Retourne 1.0 si les deux modèles sont identiques et constants
        (sklearn renvoie `nan` dans ce cas dégénéré).
    """
    a, b = _align(y_a, y_b)
    if a.size == 0:
        return 0.0
    kappa = cohen_kappa_score(a, b)
    if np.isnan(kappa):
        # Cas dégénéré (une seule classe observée) : accord parfait ⇒ 1.0, sinon 0.0.
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(kappa)


def per_sample_disagreement_mask(y_a: np.ndarray, y_b: np.ndarray) -> np.ndarray:
    """Masque booléen des échantillons en désaccord (`y_a != y_b`), pour analyse d'origine.

    Parameters
    ----------
    y_a, y_b : np.ndarray [N]

    Returns
    -------
    np.ndarray [N], bool
    """
    a, b = _align(y_a, y_b)
    return a != b


def disagreement_confusion(y_a: np.ndarray, y_b: np.ndarray, y_true: np.ndarray) -> dict:
    """Sur le sous-ensemble en désaccord : qui a raison ?

    Parameters
    ----------
    y_a, y_b : np.ndarray [N]
        Prédictions des deux modèles.
    y_true : np.ndarray [N]
        Labels vrais.

    Returns
    -------
    dict
        `a_correct`   : nb d'échantillons où seul A a raison ;
        `b_correct`   : nb d'échantillons où seul B a raison ;
        `both_wrong`  : nb d'échantillons où ni A ni B n'a raison ;
        `n_disagree`  : taille du sous-ensemble en désaccord (= somme des 3 ci-dessus).

    Notes
    -----
    Sur le masque `y_a != y_b`, les 2 modèles ne peuvent pas avoir raison simultanément,
    donc les 3 catégories partitionnent le sous-ensemble : `n_disagree == a_correct +
    b_correct + both_wrong`.
    """
    a, b, t = _align(y_a, y_b, y_true)
    mask = a != b
    a_d, b_d, t_d = a[mask], b[mask], t[mask]

    a_correct = int(np.sum((a_d == t_d) & (b_d != t_d)))
    b_correct = int(np.sum((b_d == t_d) & (a_d != t_d)))
    both_wrong = int(np.sum((a_d != t_d) & (b_d != t_d)))

    return {
        "a_correct": a_correct,
        "b_correct": b_correct,
        "both_wrong": both_wrong,
        "n_disagree": int(mask.sum()),
    }


def analyze_disagreement_origin(
    X: np.ndarray,
    mask: np.ndarray,
    y_true: np.ndarray,
    maha_scores: np.ndarray | None = None,
    boundary_scores: np.ndarray | None = None,
    top_k: int = 5,
) -> dict:
    """Corrèle le désaccord aux features, au score Mahalanobis et à la frontière de décision.

    Quantifie l'**origine** du désaccord : quelles features distinguent le plus les
    échantillons en désaccord, et le désaccord est-il concentré sur des scores
    d'anomalie élevés / près de la frontière de décision (explication attendue).

    Parameters
    ----------
    X : np.ndarray [N, d]
        Features (alignées par index avec `mask`).
    mask : np.ndarray [N], bool
        Masque des échantillons en désaccord (cf. `per_sample_disagreement_mask`).
    y_true : np.ndarray [N]
        Labels vrais (taux de panne dans/hors masque).
    maha_scores : np.ndarray [N] | None
        Score d'anomalie Mahalanobis continu. Si fourni, compare sa moyenne dans/hors
        masque.
    boundary_scores : np.ndarray [N] | None
        Proba/score continu ∈ [0, 1] (ex. `ModelPair.predict_proba`). Si fourni, mesure
        la proximité de frontière `|score - 0.5|` dans/hors masque (plus petit = plus
        proche de la frontière = explication attendue du désaccord).
    top_k : int
        Nombre de features les plus discriminantes à retourner.

    Returns
    -------
    dict
        `n_disagree`, `disagreement_rate`,
        `top_features` (indices des `top_k` features de plus grand |Δ moyenne|),
        `feature_deltas` (|Δ moyenne in/out| par feature, longueur d),
        `fault_rate_in` / `fault_rate_out`,
        `maha_score_in` / `maha_score_out` (si `maha_scores`),
        `boundary_dist_in` / `boundary_dist_out` (si `boundary_scores`).
    """
    X = np.asarray(X, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool).ravel()
    t = np.asarray(y_true).ravel()
    if X.ndim != 2:
        raise ValueError(f"X doit être 2D [N, d], reçu shape {X.shape}.")
    if X.shape[0] != mask.size:
        raise ValueError(f"X ({X.shape[0]}) et mask ({mask.size}) non alignés.")

    n_disagree = int(mask.sum())
    d = X.shape[1]
    result: dict = {
        "n_disagree": n_disagree,
        "disagreement_rate": float(mask.mean()) if mask.size else 0.0,
    }

    if n_disagree == 0 or n_disagree == mask.size:
        warnings.warn(
            f"analyze_disagreement_origin : masque dégénéré (n_disagree={n_disagree}, "
            f"N={mask.size}). Analyse d'origine non significative.",
            stacklevel=2,
        )
        result["top_features"] = []
        result["feature_deltas"] = np.zeros(d).tolist()
        return result

    mean_in = X[mask].mean(axis=0)
    mean_out = X[~mask].mean(axis=0)
    deltas = np.abs(mean_in - mean_out)  # [d]
    order = np.argsort(deltas)[::-1]  # features les plus discriminantes d'abord

    result["top_features"] = order[:top_k].tolist()
    result["feature_deltas"] = deltas.tolist()
    result["fault_rate_in"] = float((t[mask] != t.min()).mean()) if t.size else 0.0
    result["fault_rate_out"] = float((t[~mask] != t.min()).mean()) if t.size else 0.0

    if maha_scores is not None:
        s = np.asarray(maha_scores, dtype=np.float64).ravel()
        result["maha_score_in"] = float(s[mask].mean())
        result["maha_score_out"] = float(s[~mask].mean())

    if boundary_scores is not None:
        bs = np.asarray(boundary_scores, dtype=np.float64).ravel()
        dist = np.abs(bs - 0.5)  # proximité de frontière
        result["boundary_dist_in"] = float(dist[mask].mean())
        result["boundary_dist_out"] = float(dist[~mask].mean())

    return result
