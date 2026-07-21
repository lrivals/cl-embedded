"""Catalogue `drift_detection_pc` — figures d'impact de la grille PC S4405 (Sprint 44).

Répond visuellement à « **quel détecteur, à quel coût, pour quel délai** » à partir des
``experiments/exp_S44_PC_{detector}_{dataset}/results.json`` produits par
``scripts/run_sprint44_pc.py`` :

1. **Délai vs FAR** — compromis central, familles supervisée/non-supervisée/baseline en couleur.
2. **Alarmes vs points de drift** — raster des alarmes de chaque détecteur sur un dataset à
   vérité-terrain (synthétique), lignes verticales = ``drift_points`` : lisibilité du déclenchement.
3. **État mémoire / latence** — barres par détecteur, annotées viabilité MCU (prépare S45).
4. **Heatmap détecteur × dataset** — F1 de détection (cellules ``null`` en gris = honnête).
5. **Supervisé ∥ non-supervisé** — synthèse de l'axe d'étude (F1 vs coût/autonomie).

Honnêteté (règles héritées S33/S40/S42/S43) :
- **0 chiffre de résultat en dur** : toute valeur tracée sort d'un ``results.json`` (garde AST
  ``test_no_hardcoded_results_drift_pc`` en S4406).
- Cellules non calculables (délai/F1 sur Electricity sans GT ponctuelle) → **gris / omises**, jamais 0.
- RAM/latence = **proxy PC** (distingué dans les légendes de la mesure board S45).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.figures.loaders import load_experiment
from src.figures.registry import register_catalog
from src.figures.style import savefig_png
from src.models.drift import DRIFT_DETECTORS

CATALOG = "drift_detection_pc"

DETECTORS: list[str] = [*DRIFT_DETECTORS.keys(), "sliding_window_baseline"]
DATASETS: list[str] = ["synthetic", "gas_sensor_drift", "hydraulic", "electricity"]

DATASET_LABELS_FR: dict[str, str] = {
    "synthetic": "Synthétique (validation)",
    "gas_sensor_drift": "Gas Sensor Array Drift",
    "hydraulic": "Monitoring Hydraulique",
    "electricity": "Electricity (ELEC2)",
}
DETECTOR_LABELS_FR: dict[str, str] = {
    "ddm": "DDM",
    "eddm": "EDDM",
    "page_hinkley": "Page-Hinkley",
    "adwin": "ADWIN",
    "kswin": "KSWIN",
    "ks_test": "KS-Test",
    "mmd": "MMD",
    "psi": "PSI",
    "sliding_window_baseline": "Sliding-Window (baseline)",
}

# Couleurs par famille (axe scientifique supervisé ∥ non-supervisé).
FAMILY_COLORS: dict[str, str] = {
    "supervised": "#F44336",     # rouge — exige un label (coût d'autonomie)
    "unsupervised": "#2196F3",   # bleu — autonome
    "baseline": "#9E9E9E",       # gris — référence projet
}
FAMILY_LABELS_FR: dict[str, str] = {
    "supervised": "Supervisé (flux d'erreur, label requis)",
    "unsupervised": "Non-supervisé (features, autonome)",
    "baseline": "Baseline (fenêtre glissante)",
}

# Dataset de référence pour les vues « par détecteur » (GT ponctuelle, coût multivarié modéré).
REF_DATASET = "hydraulic"


def _load(detector: str, dataset: str) -> dict | None:
    """``results.json`` d'une cellule, ou ``None`` si non produit (skip honnête)."""
    try:
        data, _ = load_experiment(
            f"experiments/exp_S44_PC_{detector}_{dataset}/results.json"
        )
    except FileNotFoundError:
        return None
    return data


def _all_cells() -> list[dict]:
    """Toutes les cellules disponibles (résultats non ``None``)."""
    cells = []
    for det in DETECTORS:
        for ds in DATASETS:
            r = _load(det, ds)
            if r is not None:
                cells.append(r)
    return cells


# ── Fig 1 — Délai de détection vs taux de fausses alarmes ─────────────────────

def _fig_delay_vs_far() -> plt.Figure | None:
    cells = _all_cells()
    fig, ax = plt.subplots()
    seen_families: set[str] = set()
    plotted = False
    for r in cells:
        dm = r.get("drift_metrics")
        if not dm:
            continue
        delay = dm.get("mean_detection_delay")
        far = dm.get("false_alarm_rate")
        if delay is None or far is None:
            continue
        fam = r["family"]
        ax.scatter(far, delay, color=FAMILY_COLORS[fam], s=70, edgecolors="#212121",
                   linewidths=0.5, zorder=3,
                   label=FAMILY_LABELS_FR[fam] if fam not in seen_families else None)
        seen_families.add(fam)
        ax.annotate(DETECTOR_LABELS_FR.get(r["detector"], r["detector"]),
                    (far, delay), fontsize=7, xytext=(4, 3), textcoords="offset points")
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel("Taux de fausses alarmes (FAR) — proxy PC")
    ax.set_ylabel("Délai de détection moyen (échantillons)")
    ax.set_title("Compromis réactivité ↔ robustesse — détecteurs de drift (PC)")
    ax.legend(loc="best", fontsize=8)
    fig.text(0.01, 0.005,
             "Chaque point = une cellule (détecteur × dataset) à vérité-terrain ponctuelle. "
             "En bas à gauche = idéal (rapide + peu de fausses alarmes).",
             fontsize=8, color="#666666")
    return fig


# ── Fig 2 — Alarmes vs points de drift (raster, dataset à GT exacte) ──────────

def _fig_alarms_timeline(dataset: str = "synthetic") -> plt.Figure | None:
    rows = []
    for det in DETECTORS:
        r = _load(det, dataset)
        if r is None or not r.get("verdicts"):
            continue
        alarms = [i for i, v in enumerate(r["verdicts"]) if str(v).upper() == "DRIFT"]
        rows.append((det, alarms, r))
    if not rows:
        return None
    ref = rows[0][2]
    drift_points = ref.get("drift_points")
    n = ref["n_samples"]

    fig, ax = plt.subplots(figsize=(9, 4))
    for k, (det, alarms, r) in enumerate(rows):
        fam = r["family"]
        ax.scatter(alarms, [k] * len(alarms), color=FAMILY_COLORS[fam], s=18, marker="|",
                   linewidths=1.4)
    if drift_points:
        for j, dp in enumerate(drift_points):
            ax.axvline(dp, color="#212121", linestyle="--", linewidth=1.2,
                       label="Point de drift (vérité-terrain)" if j == 0 else None)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([DETECTOR_LABELS_FR.get(det, det) for det, *_ in rows], fontsize=8)
    ax.set_xlim(0, n)
    ax.set_xlabel("Indice échantillon (ordre temporel)")
    ax.set_title(f"Déclenchements des alarmes — {DATASET_LABELS_FR[dataset]}")
    ax.legend(loc="upper right", fontsize=8)
    fig.text(0.01, 0.005,
             "Chaque trait = une alarme DRIFT. Alignement sur les lignes = détection ; "
             "traits épars ailleurs = fausses alarmes.",
             fontsize=8, color="#666666")
    return fig


# ── Fig 3 — État mémoire / latence par détecteur (proxy PC) ───────────────────

def _fig_cost_bars(dataset: str = REF_DATASET) -> plt.Figure | None:
    names, state_b, lat, colors, viab = [], [], [], [], []
    for det in DETECTORS:
        r = _load(det, dataset)
        if r is None or not r.get("cost"):
            continue
        c = r["cost"]
        if c.get("state_bytes") is None:
            continue
        names.append(DETECTOR_LABELS_FR.get(det, det))
        state_b.append(c["state_bytes"])
        lat.append(c.get("latency_us_per_update"))
        colors.append(FAMILY_COLORS[r["family"]])
        viab.append(r.get("viabilite_mcu"))
    if not names:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    y = np.arange(len(names))
    ax1.barh(y, state_b, color=colors)
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xscale("log")
    ax1.set_xlabel("État mémoire (octets, échelle log) — proxy MCU")
    ax1.set_title("Empreinte d'état")
    for yi, (sb, v) in enumerate(zip(state_b, viab)):
        ax1.annotate(f"{v}", (sb, yi), fontsize=7, va="center", xytext=(3, 0),
                     textcoords="offset points", color="#444444")

    lat_plot = [x_ if x_ is not None else 0.0 for x_ in lat]
    ax2.barh(y, lat_plot, color=colors)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlabel("Latence par update (µs) — proxy PC")
    ax2.set_title("Latence")

    fig.suptitle(f"Coût des détecteurs — {DATASET_LABELS_FR[dataset]} (proxy PC, mesure board = S45)",
                 fontsize=12)
    fig.text(0.01, 0.005,
             "Annotation = viabilité MCU dérivée de l'état mesuré. Rouge = supervisé, bleu = "
             "non-supervisé, gris = baseline.",
             fontsize=8, color="#666666")
    fig.subplots_adjust(top=0.86, bottom=0.14)
    return fig


# ── Fig 4 — Heatmap détecteur × dataset (F1 de détection) ─────────────────────

def _fig_f1_heatmap() -> plt.Figure | None:
    mat = np.full((len(DETECTORS), len(DATASETS)), np.nan)
    for i, det in enumerate(DETECTORS):
        for j, ds in enumerate(DATASETS):
            r = _load(det, ds)
            if r is None:
                continue
            dm = r.get("drift_metrics")
            if dm and dm.get("f1") is not None:
                mat[i, j] = dm["f1"]
    if np.all(np.isnan(mat)):
        return None

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#BDBDBD")  # cellules None → gris (honnête)
    im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([DATASET_LABELS_FR[d] for d in DATASETS], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(DETECTORS)))
    ax.set_yticklabels([DETECTOR_LABELS_FR.get(d, d) for d in DETECTORS], fontsize=8)
    for i in range(len(DETECTORS)):
        for j in range(len(DATASETS)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if mat[i, j] < 0.6 else "black")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("F1 de détection (appariement dans la tolérance)")
    ax.set_title("F1 de détection — détecteur × dataset")
    fig.text(0.01, 0.005, "Gris = non calculable (pas de vérité-terrain ponctuelle, ex. Electricity).",
             fontsize=8, color="#666666")
    fig.subplots_adjust(bottom=0.2)
    return fig


# ── Fig 5 — Synthèse supervisé ∥ non-supervisé ────────────────────────────────

def _fig_family_synthesis() -> plt.Figure | None:
    # F1 vs état mémoire (log), marqueur par famille : précision vs coût/autonomie.
    cells = _all_cells()
    fig, ax = plt.subplots()
    seen: set[str] = set()
    plotted = False
    for r in cells:
        dm = r.get("drift_metrics")
        cost = r.get("cost")
        if not dm or not cost or dm.get("f1") is None or cost.get("state_bytes") is None:
            continue
        fam = r["family"]
        ax.scatter(cost["state_bytes"], dm["f1"], color=FAMILY_COLORS[fam], s=70,
                   edgecolors="#212121", linewidths=0.5, zorder=3,
                   label=FAMILY_LABELS_FR[fam] if fam not in seen else None)
        seen.add(fam)
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xscale("log")
    ax.set_xlabel("État mémoire (octets, log) — proxy MCU")
    ax.set_ylabel("F1 de détection")
    ax.set_title("Précision de détection vs coût — axe supervisé ∥ non-supervisé")
    ax.legend(loc="best", fontsize=8)
    fig.text(0.01, 0.005,
             "Supervisé (rouge) : F1 élevé à état O(1), mais exige un label (coût d'autonomie). "
             "Non-supervisé (bleu) : autonome, coût mémoire variable.",
             fontsize=8, color="#666666")
    return fig


# ── Build du catalogue ────────────────────────────────────────────────────────

@register_catalog(CATALOG)
def build(out_root: Path) -> list[Path]:
    """Génère les figures d'impact sous ``out_root/drift_detection_pc/`` ; retourne les chemins.

    Chaque figure est **omise silencieusement** si ses ``results.json`` sources sont absents (log
    clair), sans jamais inventer de valeur.
    """
    paths: list[Path] = []
    figures = [
        ("delay_vs_far", _fig_delay_vs_far),
        ("alarms_timeline_synthetic", lambda: _fig_alarms_timeline("synthetic")),
        ("cost_bars", lambda: _fig_cost_bars(REF_DATASET)),
        ("f1_heatmap", _fig_f1_heatmap),
        ("family_synthesis", _fig_family_synthesis),
    ]
    for name, fn in figures:
        fig = fn()
        if fig is None:
            print(f"[figures] {CATALOG}/{name} — sauté (source absente)")
            continue
        paths.append(savefig_png(fig, CATALOG, name, out_root))
    return paths
