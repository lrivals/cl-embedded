"""Catalogue `drift_datasets` — rend le drift visuel (Sprint 43, S4304).

Cinq familles de figures FR, régénérables et **sans chiffre de résultat en dur** :
timelines de statistique glissante, shift de distribution avant/après un point de
drift, trajectoire PCA temporelle, heatmap distance-distribution × temps, et un
comparatif inter-datasets qui justifie la diversité du corpus.

Sources (lecture seule) :
- **JSON de caractérisation** S4303 (`experiments/exp_S43_drift_char/<ds>/characterization.json`)
  via :func:`load_experiment` — suffit pour la timeline (fig 1) et le comparatif (fig 5).
- **Données brutes** via ``DRIFT_LOADERS`` (``src.data``) — nécessaires pour les figures
  qui rejouent les fenêtres (shift fig 2, PCA fig 3, heatmap fig 4). **Skip gracieux** si
  ``data/raw/<ds>/`` est absent : aucune valeur inventée, la figure n'est simplement pas produite.

Honnêteté (règles héritées S33/S40/S42) :
- Le dataset **synthétique** est étiqueté « outil de validation » — jamais présenté comme donnée
  réelle.
- ``drift_points`` sont légendés « vérité-terrain » ; ``alignment_score`` n'est annoté que s'il est
  mesuré (``null`` → mention honnête « ground-truth non ponctuelle »).
- Aucun littéral numérique de résultat : toute valeur tracée sort d'un JSON ou d'un loader (garde AST
  ``test_no_hardcoded_results_drift`` en S4305).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from scripts.characterize_drift import _js_divergence  # réutilisé, pas ré-implémenté
from src.data import DRIFT_CONFIGS, DRIFT_LOADERS
from src.evaluation.plots import plot_anomaly_score_distributions
from src.figures.loaders import load_experiment
from src.figures.registry import register_catalog
from src.figures.style import savefig_png

CATALOG = "drift_datasets"

# Ordre canonique d'affichage : le synthétique en tête (validation de la chaîne de mesure),
# puis les datasets réels (dérive de capteur / de régime documentée).
DATASETS: list[str] = ["synthetic", "gas_sensor_drift", "hydraulic", "electricity"]

# Libellés FR lisibles en slide/manuscrit.
DATASET_LABELS_FR: dict[str, str] = {
    "synthetic": "Synthétique (validation)",
    "gas_sensor_drift": "Gas Sensor Array Drift",
    "hydraulic": "Condition Monitoring Hydraulique",
    "electricity": "Electricity (ELEC2)",
}

# Couleurs par statistique glissante (palette Material du projet).
SERIES_COLORS: dict[str, str] = {
    "ks": "#2196F3",           # bleu
    "mmd": "#FF9800",          # orange
    "mahalanobis": "#9C27B0",  # violet
}
SERIES_LABELS_FR: dict[str, str] = {
    "ks": "KS (moy. par feature)",
    "mmd": "MMD (RBF)",
    "mahalanobis": "Mahalanobis / segment 0",
}

# Palette du comparatif (une couleur par dataset).
DATASET_COLORS: dict[str, str] = {
    "synthetic": "#9E9E9E",
    "gas_sensor_drift": "#FF9800",
    "hydraulic": "#4CAF50",
    "electricity": "#2196F3",
}

# Nombre max de features affichées dans le shift (fig 2) et pris en compte dans la heatmap (fig 4),
# pour rester lisible/tractable — constante de layout, pas un résultat.
MAX_SHIFT_FEATURES: int = 4
MAX_HEATMAP_FEATURES: int = 16
MAX_HEATMAP_WINDOWS: int = 80


# ── Chargement traçable ───────────────────────────────────────────────────────

def _char(ds: str) -> dict | None:
    """JSON de caractérisation S4303, ou ``None`` si non produit (skip honnête)."""
    try:
        data, _ = load_experiment(f"experiments/exp_S43_drift_char/{ds}/characterization.json")
    except FileNotFoundError:
        return None
    return data


def _raw(ds: str):
    """``DriftDataset`` via ``DRIFT_LOADERS``, ou ``None`` si ``data/raw`` absent (skip)."""
    try:
        return DRIFT_LOADERS[ds](DRIFT_CONFIGS[ds])
    except (FileNotFoundError, OSError):
        return None


def _normalize01(a: np.ndarray) -> np.ndarray:
    """Ramène une série dans [0, 1] (min-max) ; série plate → zéros (aucune valeur inventée)."""
    a = np.asarray(a, dtype=float)
    lo, hi = np.min(a), np.max(a)
    span = hi - lo
    return (a - lo) / span if span > 0 else np.zeros_like(a)


def _windows(X: np.ndarray, window: int, stride: int) -> tuple[np.ndarray, list[int]]:
    """Empile les moyennes de fenêtres glissantes + leurs centres (miroir characterize_drift)."""
    n = X.shape[0]
    means, centers = [], []
    for start in range(0, n - window + 1, stride):
        means.append(X[start : start + window].mean(axis=0))
        centers.append(start + window // 2)
    return np.asarray(means), centers


# ── Fig 1 — timeline de statistique glissante ────────────────────────────────

def _fig_timeline(ds: str) -> plt.Figure | None:
    data = _char(ds)
    if data is None:
        return None
    centers = np.asarray(data["window_centers"], dtype=float)
    series = data["series"]

    fig, ax = plt.subplots()
    for key in ("ks", "mmd", "mahalanobis"):
        ax.plot(centers, _normalize01(series[key]), color=SERIES_COLORS[key],
                label=SERIES_LABELS_FR[key])

    drift_points = data.get("drift_points")
    if drift_points:
        for k, dp in enumerate(drift_points):
            ax.axvline(dp, color="#212121", linestyle="-", linewidth=1.6,
                       label="Point de drift (vérité-terrain)" if k == 0 else None)
    for k, pc in enumerate(data.get("peak_centers", []) or []):
        ax.axvline(pc, color="#F44336", linestyle="--", linewidth=1.2,
                   label="Pic mesuré (change-point)" if k == 0 else None)

    align = data.get("alignment_score")
    if drift_points and isinstance(align, (int, float)):
        note = f"score d'alignement pic ↔ point = {align:g} échantillons (médiane)"
    elif not drift_points:
        note = "ground-truth non ponctuelle → alignement non calculable (honnête)"
    else:
        note = ""

    label = DATASET_LABELS_FR[ds]
    ax.set_xlabel("Indice échantillon (ordre temporel)")
    ax.set_ylabel("Statistique glissante (normalisée [0, 1])")
    ax.set_title(f"Timeline de drift — {label} ({data['drift_type_confirmed']})")
    ax.legend(loc="upper left", fontsize=9)
    if note:
        fig.text(0.01, 0.005, note, fontsize=8, color="#666666")
    return fig


# ── Fig 2 — shift de distribution avant/après un point de drift ──────────────

def _fig_shift(ds: str) -> plt.Figure | None:
    data = _char(ds)
    d = _raw(ds)
    if data is None or d is None:
        return None

    # Point de coupure : 1er drift_point si ponctuel, sinon frontière 1er/dernier tiers (Electricity).
    drift_points = data.get("drift_points")
    n = d.X.shape[0]
    if drift_points:
        cut = int(drift_points[0])
        cut_label = f"1er point de drift (idx {cut}, vérité-terrain)"
    else:
        cut = n // 3
        cut_label = f"frontière 1er tiers (idx {cut}, pas de point ponctuel)"

    feats = [f["feature"] for f in data.get("features_most_drifted", [])][:MAX_SHIFT_FEATURES]
    name_to_col = {name: j for j, name in enumerate(d.feature_names)}

    # Réutilise plot_anomaly_score_distributions : model = feature, classe 0 = avant, 1 = après.
    scores_by_model: dict[str, list[np.ndarray]] = {}
    labels_by_task: list[np.ndarray] = []
    for name in feats:
        col = name_to_col[name]
        before = d.X[:cut, col]
        after = d.X[cut:, col]
        scores_by_model[name] = [np.concatenate([before, after])]
        labels_by_task = [np.concatenate([np.zeros(len(before)), np.ones(len(after))]).astype(int)]

    if not scores_by_model:
        return None

    fig = plot_anomaly_score_distributions(
        scores_by_model, labels_by_task, task_names=["avant vs après"]
    )
    fig.suptitle(
        f"Shift de distribution — {DATASET_LABELS_FR[ds]}\n"
        f"Vert = avant / Rouge = après · coupure : {cut_label}",
        fontsize=12,
    )
    fig.subplots_adjust(top=0.88)
    return fig


# ── Fig 3 — trajectoire PCA temporelle ───────────────────────────────────────

def _fig_pca(ds: str) -> plt.Figure | None:
    data = _char(ds)
    d = _raw(ds)
    if data is None or d is None:
        return None

    X = d.X
    window = int(data["window_size"])
    stride = int(data["stride"])
    means, centers = _windows(X, window, stride)
    if len(means) < 2:
        return None

    # PCA fit sur le segment 0 (miroir characterize_drift : composantes de référence).
    ref = X[: min(window, X.shape[0])]
    n_comp = 2 if X.shape[1] >= 2 else 1
    pca = PCA(n_components=n_comp, random_state=42).fit(ref)
    proj = pca.transform(means)
    if proj.shape[1] == 1:  # dataset 1-D : étale sur l'axe temps
        proj = np.column_stack([proj[:, 0], np.zeros(len(proj))])

    fig, ax = plt.subplots()
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=centers, cmap="viridis", s=60, zorder=3)
    ax.plot(proj[:, 0], proj[:, 1], color="#BDBDBD", linewidth=1.0, zorder=1)

    # Marque les fenêtres proches des points de drift (vérité-terrain).
    drift_points = data.get("drift_points")
    if drift_points:
        centers_arr = np.asarray(centers)
        for k, dp in enumerate(drift_points):
            i = int(np.argmin(np.abs(centers_arr - dp)))
            ax.scatter(proj[i, 0], proj[i, 1], facecolors="none", edgecolors="#F44336",
                       s=220, linewidths=2.0, zorder=4,
                       label="Fenêtre au point de drift (vérité-terrain)" if k == 0 else None)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Indice échantillon (temps)")
    ax.set_xlabel("Composante principale 1 (segment 0)")
    ax.set_ylabel("Composante principale 2 (segment 0)")
    ax.set_title(f"Trajectoire temporelle en espace PCA — {DATASET_LABELS_FR[ds]}")
    if drift_points:
        ax.legend(loc="best", fontsize=9)
    fig.text(0.01, 0.005,
             "Chaque point = moyenne d'une fenêtre glissante, projetée sur les axes du segment 0. "
             "Glissement continu = incrémental ; saut = soudain ; retour = récurrent.",
             fontsize=8, color="#666666")
    return fig


# ── Fig 4 — heatmap distance-distribution × temps ────────────────────────────

def _fig_heatmap(ds: str) -> plt.Figure | None:
    data = _char(ds)
    d = _raw(ds)
    if data is None or d is None:
        return None

    X = d.X
    window = int(data["window_size"])
    stride = int(data["stride"])
    bins = int(data.get("metadata", {}).get("config_snapshot", {})
               .get("characterization", {}).get("hist_bins", 20))

    # Bornes de fenêtres.
    n = X.shape[0]
    starts = list(range(0, n - window + 1, stride))
    if len(starts) > MAX_HEATMAP_WINDOWS:  # sous-échantillonne uniformément pour rester lisible
        idx = np.linspace(0, len(starts) - 1, MAX_HEATMAP_WINDOWS).astype(int)
        starts = [starts[i] for i in idx]
    wins = [X[s : s + window] for s in starts]
    centers = [s + window // 2 for s in starts]

    # Features prises en compte : top-k les plus dérivées (tractabilité + pertinence).
    feats = [f["feature"] for f in data.get("features_most_drifted", [])][:MAX_HEATMAP_FEATURES]
    name_to_col = {name: j for j, name in enumerate(d.feature_names)}
    cols = [name_to_col[f] for f in feats] or list(range(min(X.shape[1], MAX_HEATMAP_FEATURES)))

    m = len(wins)
    mat = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            dist = float(np.mean([_js_divergence(wins[i][:, c], wins[j][:, c], bins) for c in cols]))
            mat[i, j] = mat[j, i] = dist

    fig, ax = plt.subplots()
    im = ax.imshow(mat, cmap="magma", origin="lower")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Distance de Jensen-Shannon moyenne (par feature)")

    # Repère les points de drift sur les deux axes.
    drift_points = data.get("drift_points")
    if drift_points:
        centers_arr = np.asarray(centers)
        for k, dp in enumerate(drift_points):
            pos = int(np.argmin(np.abs(centers_arr - dp)))
            ax.axvline(pos, color="#00E5FF", linestyle="--", linewidth=1.0,
                       label="Point de drift (vérité-terrain)" if k == 0 else None)
            ax.axhline(pos, color="#00E5FF", linestyle="--", linewidth=1.0)
        ax.legend(loc="upper left", fontsize=8)

    ax.set_xlabel("Fenêtre j")
    ax.set_ylabel("Fenêtre i")
    ax.set_title(f"Distance distribution × temps — {DATASET_LABELS_FR[ds]}")
    fig.text(0.01, 0.005,
             "Bloc-diagonale = régime stable ; ruptures nettes = drift. "
             f"Moyenne sur {len(cols)} feature(s) les plus dérivées.",
             fontsize=8, color="#666666")
    return fig


# ── Fig 5 — comparatif inter-datasets ────────────────────────────────────────

def _fig_comparatif() -> plt.Figure | None:
    rows = []
    for ds in DATASETS:
        data = _char(ds)
        if data is None:
            continue
        centers = np.asarray(data["window_centers"], dtype=float)
        t_rel = _normalize01(centers) if len(centers) > 1 else np.zeros(len(centers))
        rows.append((ds, t_rel, _normalize01(data["series"]["mmd"]), data))

    if not rows:
        return None

    fig, ax = plt.subplots()
    for k, (ds, t_rel, mmd, data) in enumerate(rows):
        offset = float(k)  # décalage vertical par dataset (layout)
        ax.plot(t_rel, mmd + offset, color=DATASET_COLORS[ds], label=DATASET_LABELS_FR[ds])
        ax.axhline(offset, color="#E0E0E0", linewidth=0.8, zorder=0)
        # Points de drift en temps relatif.
        drift_points = data.get("drift_points")
        if drift_points and data["n_samples"] > 0:
            for dp in drift_points:
                ax.plot(dp / data["n_samples"], offset, marker="|", color="#212121",
                        markersize=14, markeredgewidth=1.6)

    ax.set_yticks([float(k) for k in range(len(rows))])
    ax.set_yticklabels([DATASET_LABELS_FR[ds] for ds, *_ in rows])
    ax.set_xlabel("Temps relatif (0 → 1)")
    ax.set_title("Intensité de drift comparée (MMD normalisée) — diversité du corpus")
    ax.legend(loc="upper right", fontsize=9)
    fig.text(0.01, 0.005,
             "Une ligne par dataset (MMD glissante normalisée, décalée verticalement). "
             "Traits noirs = points de drift (vérité-terrain) en temps relatif.",
             fontsize=8, color="#666666")
    return fig


# ── Build du catalogue ────────────────────────────────────────────────────────

@register_catalog(CATALOG)
def build(out_root: Path) -> list[Path]:
    """Génère les figures de drift sous ``out_root/drift_datasets/`` ; retourne les chemins.

    Les figures dépendant des données brutes (shift/PCA/heatmap) sont **omises silencieusement**
    si ``data/raw/<ds>/`` est absent (log clair), sans jamais inventer de valeur.
    """
    paths: list[Path] = []
    per_dataset = [
        ("timeline", _fig_timeline),
        ("shift", _fig_shift),
        ("pca", _fig_pca),
        ("heatmap", _fig_heatmap),
    ]
    for ds in DATASETS:
        for kind, fn in per_dataset:
            fig = fn(ds)
            if fig is None:
                print(f"[figures] drift_datasets/{kind}_{ds} — sauté (source absente)")
                continue
            paths.append(savefig_png(fig, CATALOG, f"{kind}_{ds}", out_root))

    comp = _fig_comparatif()
    if comp is not None:
        paths.append(savefig_png(comp, CATALOG, "comparatif_datasets", out_root))
    return paths
