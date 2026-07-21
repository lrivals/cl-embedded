#!/usr/bin/env python3
"""
scripts/run_sprint44_pc.py — Sprint 44 (S4405) : grille PC des détecteurs de drift.

Rejoue chaque flux de drift (Sprint 43) échantillon par échantillon à travers chaque détecteur
(Sprint 44), applique le harnais S4404 (métriques de détection + coût proxy PC), et écrit un
``results.json`` reproductible par cellule ``(détecteur, dataset)`` :

    experiments/exp_S44_PC_{detector}_{dataset}/results.json

Familles (axe scientifique supervisé ∥ non-supervisé) :

- **non-supervisés** (``requires_label = False``) : features z-scorées figées sur l'enrôlement.
  Univariés (PSI/KSWIN/KSTest/ADWIN) enveloppés par ``MultiFeatureDriftDetector`` (agrégation ``max``
  par défaut, config) ; MMD nativement multivarié.
- **supervisés** (``requires_label = True`` : DDM/EDDM/Page-Hinkley) : consomment le **flux d'erreur**
  ``e_t = 1[ŷ_t ≠ y_t]`` d'un **modèle de faute** (décision S4400 : ``LogisticRegression`` sklearn
  entraînée sur le segment d'enrôlement — lève ``TODO(arnaud)`` pour la grille PC ; le vrai modèle
  EWC embarqué est du ressort du Sprint 45). Le label alimente le flux d'erreur, **jamais** le
  détecteur directement.
- **baseline** ``sliding_window_baseline`` (``SlidingWindowDriftDetector``, déjà porté C) via un
  adaptateur : score d'anomalie = ``max_j |z_j|`` (agrégation ``max`` cohérente), verdicts chaînes.

Règles (héritées) : seed 42 ; ``null`` tant que non exécuté / non calculable (aucun chiffre inventé) ;
même loader/segment d'enrôlement que le Sprint 45. Les chiffres RAM/latence sont des **proxies PC**.

Usage
-----
    python scripts/run_sprint44_pc.py --detector page_hinkley --dataset synthetic
    python scripts/run_sprint44_pc.py --all
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import DRIFT_CONFIGS, DRIFT_LOADERS  # noqa: E402
from src.data.drift_dataset import DriftDataset, freeze_zscore  # noqa: E402
from src.evaluation.drift_detector import SlidingWindowDriftDetector  # noqa: E402
from src.evaluation.drift_metrics import (  # noqa: E402
    alarms_from_verdicts,
    compute_drift_metrics,
    profile_drift_detector,
    save_drift_metrics,
)
from src.models.drift import DRIFT_DETECTORS, MMD, MultiFeatureDriftDetector  # noqa: E402
from src.models.drift import error_stream  # noqa: E402
from src.utils.reproducibility import set_seed  # noqa: E402

DEFAULT_CONFIG = "configs/sprint44_drift_detection.yaml"
OUT_ROOT = ROOT / "experiments"
BASELINE_NAME = "sliding_window_baseline"
BASELINE_STATE_BYTES = 200  # # MEM: 200 B @ FP32 (W=50, d=4) — src/evaluation/drift_detector.py

# Tolérance de détection (échantillons après un point de drift) — largeur de la fenêtre de comptage.
# Constante nommée (traçable dans results.json), surchargée par --tolerance.
TOLERANCE_DEFAULT = 200

# Seuils de viabilité MCU dérivés de l'état mesuré (state_bytes) — décision traçable S4406.
VIABILITY_HAUTE_MAX = 1_024      # O(1) / O(bins) compact → embarquable sans réserve
VIABILITY_MOYENNE_MAX = 16_384   # fenêtre bornée modérée → embarquable sous budget
# au-delà → "pc_only" (ex. MMD stocke toute la référence n_ref·d).

ALL_DETECTORS = [*DRIFT_DETECTORS.keys(), BASELINE_NAME]


def _viabilite_mcu(state_bytes: int | None) -> str | None:
    """Classe la viabilité board à partir de l'empreinte d'état mesurée (seuils nommés ci-dessus)."""
    if state_bytes is None:
        return None
    if state_bytes <= VIABILITY_HAUTE_MAX:
        return "haute"
    if state_bytes <= VIABILITY_MOYENNE_MAX:
        return "moyenne"
    return "pc_only"


def build_fault_model(X_enroll: np.ndarray, y_enroll: np.ndarray, seed: int):
    """Modèle de faute pour le flux d'erreur supervisé : LogisticRegression (décision S4400).

    Repli ``DummyClassifier(most_frequent)`` si le segment d'enrôlement est mono-classe (logreg non
    entraînable) — honnête, sans planter la cellule.
    """
    classes = np.unique(y_enroll)
    if classes.size < 2:
        model = DummyClassifier(strategy="most_frequent")
    else:
        model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(X_enroll, y_enroll)
    return model


def _binarize_labels(y: np.ndarray, y_enroll: np.ndarray) -> np.ndarray:
    """Binarise en « normal (classe dominante à l'enrôlement) vs reste » (datasets multiclasses)."""
    y = np.asarray(y)
    uniq = np.unique(y)
    if uniq.size <= 2:  # déjà binaire (synthetic/hydraulic/electricity)
        return (y != uniq[0]).astype(np.int64) if uniq.size == 2 and set(uniq) != {0, 1} else y
    vals, counts = np.unique(y_enroll, return_counts=True)
    normal_cls = vals[int(np.argmax(counts))]
    return (y != normal_cls).astype(np.int64)


def _make_detector(name: str, cfg: dict, n_features: int):
    """Instancie le détecteur (non-supervisé) : univarié enveloppé, MMD natif, baseline adapté."""
    calib = cfg.get("calibration", {})
    if name == "mmd":
        return MMD(cfg.get("mmd", {}))
    section = dict(cfg.get(name, {}))
    section.update({k: calib[k] for k in ("aggregation", "fraction_threshold") if k in calib})
    cls = DRIFT_DETECTORS[name]
    return MultiFeatureDriftDetector(lambda: cls(dict(cfg.get(name, {}))), n_features, section)


def run_cell(detector: str, dataset: str, cfg: dict, tolerance: int,
             max_samples: int | None = None) -> dict:
    """Exécute une cellule (détecteur × dataset) et retourne le dict ``results.json``."""
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    d: DriftDataset = DRIFT_LOADERS[dataset](DRIFT_CONFIGS[dataset])
    X, y, drift_points = d.X, d.y, d.drift_points
    if max_samples is not None and X.shape[0] > max_samples:
        X = X[:max_samples]
        if y is not None:
            y = y[:max_samples]
        if drift_points is not None:
            drift_points = [p for p in drift_points if p < max_samples]

    n_samples = X.shape[0]
    n_enroll = max(2, int(cfg.get("calibration", {}).get("enrollment_fraction", 0.25) * n_samples))

    # Normalisation z-score figée sur l'enrôlement (S43 : le drift reste visible).
    X_norm, _, _ = freeze_zscore(X, (0, n_enroll))
    X_enroll = X_norm[:n_enroll]

    is_baseline = detector == BASELINE_NAME
    requires_label = (not is_baseline) and DRIFT_DETECTORS[detector]._REQUIRES_LABEL

    na_reason: str | None = None
    verdict_names: list[str] = []
    profile_stream: list = []

    # ── Construction du flux + streaming ─────────────────────────────────────────────────────────
    if is_baseline:
        # Score d'anomalie = max_j |z_j| (agrégation "max"), seuils P95 sur l'enrôlement.
        scores = np.max(np.abs(X_norm), axis=1)
        det = SlidingWindowDriftDetector(**{
            k: cfg[BASELINE_NAME][k]
            for k in ("window_size", "fault_multiplier", "drift_multiplier", "drift_ratio")
            if k in cfg.get(BASELINE_NAME, {})
        })
        det.set_thresholds_from_normal(scores[:n_enroll])
        verdict_names = [det.update(float(s)) for s in scores]
        profile_stream = scores.tolist()
        state_bytes = BASELINE_STATE_BYTES

    elif requires_label:
        if y is None:
            na_reason = "supervisé : labels absents pour ce dataset"
        else:
            y_bin = _binarize_labels(y, y[:n_enroll])
            model = build_fault_model(X_enroll, y_bin[:n_enroll], seed)
            err = error_stream(model, X_norm, y_bin)  # 0/1
            det = DRIFT_DETECTORS[detector](dict(cfg.get(detector, {})))
            verdict_names = [det.update(float(e)).name for e in err]
            profile_stream = err.tolist()
            state_bytes = det.get_state_bytes()

    else:  # non-supervisé
        det = _make_detector(detector, cfg, X.shape[1])
        det.set_params_from_reference(X_enroll)
        if detector == "mmd":
            verdict_names = [det.update(X_norm[i]).name for i in range(n_samples)]
            profile_stream = [X_norm[i] for i in range(n_samples)]
        else:
            verdict_names = [det.update(X_norm[i]).name for i in range(n_samples)]
            profile_stream = [X_norm[i] for i in range(n_samples)]
        state_bytes = det.get_state_bytes()

    # ── Métriques + coût (proxy PC) ──────────────────────────────────────────────────────────────
    if na_reason is not None:
        drift_metrics = None
        cost = None
        viabilite = None
    else:
        alarms = alarms_from_verdicts(verdict_names)
        drift_metrics = compute_drift_metrics(alarms, drift_points, n_samples, tolerance)
        # Profilage sur une instance dédiée (le streaming modifie l'état).
        prof_det = _build_profiling_detector(detector, cfg, X, X_enroll, is_baseline, requires_label)
        cost = profile_drift_detector(prof_det, profile_stream, state_bytes=state_bytes)
        viabilite = _viabilite_mcu(cost.get("state_bytes"))

    return {
        "exp_id": f"exp_S44_PC_{detector}_{dataset}",
        "platform": "pc",
        "detector": detector,
        "dataset": dataset,
        "date": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "n_samples": n_samples,
        "n_enroll": n_enroll,
        "n_features": int(X.shape[1]),
        "drift_points": list(drift_points) if drift_points else None,
        "tolerance": tolerance,
        "requires_label": bool(requires_label),
        "family": "supervised" if requires_label else ("baseline" if is_baseline else "unsupervised"),
        "verdicts": verdict_names,
        "drift_metrics": drift_metrics,
        "cost": cost,
        "viabilite_mcu": viabilite,
        "na_reason": na_reason,
        "config_snapshot": cfg,
    }


def _build_profiling_detector(detector, cfg, X, X_enroll, is_baseline, requires_label):
    """Instance neuve pour le profilage de coût (isole l'état du streaming de verdicts)."""
    if is_baseline:
        det = SlidingWindowDriftDetector(**{
            k: cfg[BASELINE_NAME][k]
            for k in ("window_size", "fault_multiplier", "drift_multiplier", "drift_ratio")
            if k in cfg.get(BASELINE_NAME, {})
        })
        det.set_thresholds_from_normal(np.max(np.abs(X_enroll), axis=1))
        return det
    if requires_label:
        return DRIFT_DETECTORS[detector](dict(cfg.get(detector, {})))
    det = _make_detector(detector, cfg, X.shape[1])
    det.set_params_from_reference(X_enroll)
    return det


def _write(result: dict) -> Path:
    exp_dir = OUT_ROOT / result["exp_id"]
    out = exp_dir / "results.json"
    save_drift_metrics({}, out)  # crée le dossier
    save_drift_metrics(result, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Grille PC des détecteurs de drift (Sprint 44, S4405).")
    ap.add_argument("--detector", choices=ALL_DETECTORS)
    ap.add_argument("--dataset", choices=list(DRIFT_LOADERS.keys()))
    ap.add_argument("--all", action="store_true", help="Exécute toute la grille détecteur × dataset.")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--tolerance", type=int, default=TOLERANCE_DEFAULT)
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Cap tractabilité (tronque le flux, drift_points > cap retirés).")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    datasets = list(cfg.get("datasets", DRIFT_LOADERS.keys()))

    if args.all:
        cells = [(det, ds) for det in ALL_DETECTORS for ds in datasets]
    else:
        if not args.detector or not args.dataset:
            ap.error("préciser --detector ET --dataset, ou --all")
        cells = [(args.detector, args.dataset)]

    for det, ds in cells:
        print(f"[S44] {det} × {ds} …", flush=True)
        try:
            result = run_cell(det, ds, cfg, args.tolerance, args.max_samples)
        except (FileNotFoundError, OSError) as e:
            print(f"    ⚠ données absentes ({type(e).__name__}) — cellule sautée : {e}", flush=True)
            continue
        out = _write(result)
        dm = result["drift_metrics"]
        if dm is None:
            print(f"    → {out.relative_to(ROOT)}  (N/A : {result['na_reason']})", flush=True)
        else:
            print(f"    → {out.relative_to(ROOT)}  "
                  f"délai={dm['mean_detection_delay']} FAR={dm['false_alarm_rate']} "
                  f"F1={dm['f1']} state={result['cost']['state_bytes']}B "
                  f"({result['viabilite_mcu']})", flush=True)


if __name__ == "__main__":
    main()
